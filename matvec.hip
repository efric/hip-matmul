// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hip"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

// Device function that implements a matrix vector multiplication over data in
// the 'global' layout. This is not intended to work with data tiling.
using matvec_func_t = void (*)(const void * /*A*/, const void * /*B*/,
                               void * /*C*/, int /*M*/, int /*N*/, int /*K*/);

struct ProblemProperties {
  Type A_type;
  Type B_type;
  Type C_type;
  MNKShape total;
  MNKShape outer;
  MNKShape tile;
};

using num_threads_func_t = int (*)(const ProblemProperties & /*problem*/);
using shared_mem_bytes_func_t = size_t (*)(const ProblemProperties & /*problem*/);

size_t no_shared_mem(const ProblemProperties &) { return 0; }

static constexpr int kSubgroupSize32 = 32;
static constexpr int kSubgroupSize64 = 64;

struct DeviceCapabilities {
  const char *name = "unknown";
  int subgroup_size = 0;
  bool has_f16_f32_dot = false;
  bool has_f16_output_dot = false;
  bool has_fp8_dot = false;
};

bool architectureNameStartsWith(const char *name, const char *target) {
  size_t target_len = std::strlen(target);
  if (std::strncmp(name, target, target_len) != 0) {
    return false;
  }
  return name[target_len] == '\0' || name[target_len] == ':' ||
         name[target_len] == '-';
}

DeviceCapabilities detectDeviceCapabilities() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties = {};
  HIP_CHECK(hipGetDeviceProperties(&properties, device));

  DeviceCapabilities result;
  if (architectureNameStartsWith(properties.gcnArchName, "gfx1100")) {
    result = {"gfx1100", kSubgroupSize32, true, true, false};
  } else if (architectureNameStartsWith(properties.gcnArchName, "gfx1201")) {
    result = {"gfx1201", kSubgroupSize32, true, true, true};
  } else if (architectureNameStartsWith(properties.gcnArchName, "gfx942")) {
    result = {"gfx942", kSubgroupSize64, true, false, false};
  } else if (architectureNameStartsWith(properties.gcnArchName, "gfx950")) {
    result = {"gfx950", kSubgroupSize64, true, false, false};
  } else {
    fprintf(stderr,
            "Warning: active device reports unrecognized target '%s'; "
            "subgroup and dot kernels will be skipped.\n",
            properties.gcnArchName);
  }
  return result;
}

void print(const ProblemProperties &problem, FILE *file = stderr) {
  fprintf(file, "A:%s, B:%s, C:%s, Total MxNxK=%dx%dx%d\n", str(problem.A_type),
          str(problem.B_type), str(problem.C_type), problem.total.M,
          problem.total.N, problem.total.K);
  fprintf(file, "\tOuter MxNxK=%dx%dx%d, Tile MxNxK=%dx%dx%d\n",
          problem.outer.M, problem.outer.N, problem.outer.K, problem.tile.M,
          problem.tile.N, problem.tile.K);
}

int flat_a_size(const ProblemProperties &problem) {
  return problem.total.M * problem.total.K;
}

int flat_b_size(const ProblemProperties &problem) {
  return problem.total.N * problem.total.K;
}

int flat_c_size(const ProblemProperties &problem) {
  return problem.total.M * problem.total.N;
}

struct MatvecKernel {
  const char *name; // Kernel name.
  Type A_type;      // Element type of the LHS matrix.
  Type B_type;      // Element type of the RHS matrix.
  Type C_type;      // Element type of the result matrix.
  int M_tile;       // M-dimension tile size (rows of accumulator).
  int N_tile;       // N-dimension tile size (columns of accumulator).
  int K_tile;       // K-dimension tile size (reduction dimension).
  num_threads_func_t num_threads_func; // Number of threads that the kernel
                                       // requires running on.
  shared_mem_bytes_func_t
      shared_mem_bytes_func;      // Amount of shared memory per workgroup.
  matvec_func_t matvec_func; // Device kernel pointer.
  int subgroup_size = 0;     // Zero means the kernel is not subgroup-shaped.
};

ProblemProperties getBenchmarkProblemSize(const MatvecKernel &kernel,
                                    MNKShape total) {
  ProblemProperties problem_size = {};
  problem_size.A_type = kernel.A_type;
  problem_size.B_type = kernel.B_type;
  problem_size.C_type = kernel.C_type;
  problem_size.tile = {kernel.M_tile, kernel.N_tile, kernel.K_tile};

  problem_size.total = total;
  problem_size.outer.M = ceil_div(total.M, kernel.M_tile);
  problem_size.outer.N = ceil_div(total.N, kernel.N_tile);
  problem_size.outer.K = ceil_div(total.K, kernel.K_tile);
  return problem_size;
}

ProblemProperties getCheckProblemSize(const MatvecKernel &kernel) {
  int M = getIntEnvVar("M", 4096);
  int N = getIntEnvVar("N", 1);
  int K = getIntEnvVar("K", 4096);
  MNKShape total = {M, N, K};
  return getBenchmarkProblemSize(kernel, total);
}

template <Type T> float referenceValue(CType<T> value) {
  return static_cast<float>(value);
}

template <> float referenceValue<Type::FP8>(CType<Type::FP8> value) {
  return ocp_e4m3_fp8_to_float(value);
}

template <Type A_type, Type B_type, Type C_type>
void checkMatvecResults(const void *A_data_void, const void *B_data_void,
                        const void *C_data_void, const ProblemProperties &problem) {
  if (getIntEnvVar("DEBUG", 0)) {
    fprintf(stderr, "Checking matvec result\n");
    print(problem);
  }
  using TA = CType<A_type>;
  using TB = CType<B_type>;
  using TC = CType<C_type>;
  const TA *A_data = static_cast<const TA *>(A_data_void);
  const TB *B_data = static_cast<const TB *>(B_data_void);
  const TC *C_data = static_cast<const TC *>(C_data_void);
  // This reference code is slow. To make the checks not too slow on
  // large matmuls, we only check the 9 corner/middle tiles.
  for (int m_outer : {0, problem.outer.M / 2, problem.outer.M - 1}) {
    for (int n_outer : {0, problem.outer.N / 2, problem.outer.N - 1}) {
      for (int m_tile = 0; m_tile < problem.tile.M; ++m_tile) {
        for (int n_tile = 0; n_tile < problem.tile.N; ++n_tile) {
          int global_m = m_outer * problem.tile.M + m_tile;
          int global_n = n_outer * problem.tile.N + n_tile;
          if (global_m >= problem.total.M || global_n >= problem.total.N) {
            continue;
          }
          TC c = {0};
          for (int k_outer = 0; k_outer < problem.outer.K; ++k_outer) {
            for (int k_tile = 0; k_tile < problem.tile.K; ++k_tile) {
              int global_k = k_outer * problem.tile.K + k_tile;
              if (global_k >= problem.total.K) {
                continue;
              }
              TA a = A_data[global_m * problem.total.K + global_k];
              TB b = B_data[global_n * problem.total.K + global_k];
              c += static_cast<TC>(referenceValue<A_type>(a)) *
                   static_cast<TC>(referenceValue<B_type>(b));
            }
          }
          TC expected = c;
          TC actual = C_data[global_m * problem.total.N + global_n];
          if (actual != expected) {
            fprintf(stderr,
                    "matmul numerical error: actual(%g) != "
                    "expected(%g), at m_outer=%d n_outer=%d m_tile=%d "
                    "n_tile=%d, at %s:%d. Note: outer MxNxK = %dx%dx%d\n",
                    static_cast<float>(actual), static_cast<float>(expected),
                    m_outer, n_outer, m_tile, n_tile, __FILE__, __LINE__,
                    problem.outer.M, problem.outer.N, problem.outer.K);
            abort();
          }
        }
      }
    }
  }
}

void checkMatvecResults(Type A_type, Type B_type, Type C_type,
                        const void *A_data_void, const void *B_data_void,
                        const void *C_data_void, const ProblemProperties &problem) {
#define HANDLE_CASE(A, B, C)                                                   \
  if (A_type == Type::A && B_type == Type::B && C_type == Type::C) {           \
    checkMatvecResults<Type::A, Type::B, Type::C>(A_data_void, B_data_void,    \
                                                  C_data_void, problem);       \
    return;                                                                    \
  }
  HANDLE_CASE(FP32, FP32, FP32)
  HANDLE_CASE(FP8, FP8, FP32)
  HANDLE_CASE(FP16, FP16, FP32)
  HANDLE_CASE(FP16, FP16, FP16)
  HANDLE_CASE(SI8, SI8, SI32)
#undef HANDLE_CASE

  fprintf(stderr, "%s:%d: unhandled types\n", __FILE__, __LINE__);
  abort();
}

void check(const MatvecKernel &kernel, const ProblemProperties &problem) {
  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type, flat_a_size(problem), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type, flat_b_size(problem), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type, flat_c_size(problem), random_engine);

  void *A_device_buffer = nullptr;
  void *B_device_buffer = nullptr;
  void *C_device_buffer = nullptr;
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipGetLastError());

  const dim3 grid_dim(problem.outer.M, problem.outer.N);
  const dim3 block_dim(kernel.num_threads_func(problem));
  const size_t shared_mem_bytes = kernel.shared_mem_bytes_func(problem);
  fprintf(stderr, "Num threads: %d\n", block_dim.x);
  fprintf(stderr, "Shared mem bytes: %zu\n", shared_mem_bytes);

  HIP_CHECK(hipGetLastError());
  kernel
      .matvec_func<<<grid_dim, block_dim, shared_mem_bytes, hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer, problem.total.M,
          problem.total.N, problem.total.K);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_host_data.size(),
                      hipMemcpyDeviceToHost));
  checkMatvecResults(kernel.A_type, kernel.B_type, kernel.C_type,
                     A_host_data.data(), B_host_data.data(), C_host_data.data(),
                     problem);

  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
}

void check(const MatvecKernel &kernel) {
  check(kernel, getCheckProblemSize(kernel));
}

void benchmark(const MatvecKernel &kernel, MNKShape total) {
  ProblemProperties problem = getBenchmarkProblemSize(kernel, total);
  std::printf("  Benchmarking: ");
  print(problem, stdout);

  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type, flat_a_size(problem), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type, flat_b_size(problem), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type, flat_c_size(problem), random_engine);

  void *A_device_buffer{};
  void *B_device_buffer{};
  void *C_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));

  const dim3 grid_dim(problem.outer.M, problem.outer.N);
  const dim3 block_dim(kernel.num_threads_func(problem));
  const size_t shared_mem_bytes = kernel.shared_mem_bytes_func(problem);

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  float elapsed_ms{};
  float min_elapsed_ms = getIntEnvVar("BENCHMARK_MIN_MS", 100);
  int fixed_iterations = getIntEnvVar("FIXED_ITERATIONS", 0);
  int iterations = fixed_iterations ? fixed_iterations : 1;
  while (true) {
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));
    for (int b = 0; b < iterations; ++b) {
      kernel.matvec_func<<<grid_dim, block_dim, shared_mem_bytes,
                           hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer, problem.total.M,
          problem.total.N, problem.total.K);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    if (elapsed_ms >= min_elapsed_ms || fixed_iterations) {
      break;
    }
    if (iterations > (1 << 20)) {
      fprintf(stderr, "Vacuous kernel? Only taking %g ms at iterations=%d.\n",
              elapsed_ms, iterations);
      abort();
    }
    iterations *= 2;
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));

  // Calculate the actual amount of memory read during the calculation, taking
  // into account the tile sizes.
  float A_element_bytes = type_size(kernel.A_type);
  float B_element_bytes = type_size(kernel.B_type);
  float A_bytes = A_element_bytes * problem.total.M * problem.total.K;
  float B_bytes = B_element_bytes * problem.total.N * problem.total.K;
  float kernel_bytes_read = A_bytes + B_bytes;

  float kernel_ms = elapsed_ms / iterations;
  float kernel_ops = 2.f * problem.total.M * problem.total.N * problem.total.K;
  float kernel_ops_per_s = 1000.f * kernel_ops / kernel_ms;
  float kernel_bytes_read_per_s = 1000.f * kernel_bytes_read / kernel_ms;
  std::printf(
      "\tRead %.4g TB/s, %.4g Tflop/s, latency %.2g ms, iterations=%d\n",
      1.e-12f * kernel_ops_per_s, 1e-12f * kernel_bytes_read_per_s, kernel_ms,
      iterations);
}

struct NaiveMatmulKernel : MatvecKernel {
  using TA = float;
  using TB = float;
  using TC = float;
  static constexpr int T_M_tile = 64; // One subgroup.
  static constexpr int T_N_tile = 1;
  static constexpr int T_K_tile = 4; // dword_x4.

  NaiveMatmulKernel() {
    name = __FUNCTION__;
    A_type = Type::FP32;
    B_type = Type::FP32;
    C_type = Type::FP32;
    M_tile = T_M_tile;
    N_tile = T_N_tile;
    K_tile = T_K_tile;
    num_threads_func = [](const ProblemProperties &) {
      return T_M_tile * T_N_tile;
    };
    shared_mem_bytes_func = no_shared_mem;
    matvec_func = run;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int m_tile = threadIdx.x / T_N_tile;
    int n_tile = threadIdx.x % T_N_tile;

    int K_outer = ceil_div(K, T_K_tile);

    int global_m = m_outer * T_M_tile + m_tile;
    int global_n = n_outer * T_N_tile + n_tile;
    if (global_m >= M || global_n >= N)
      return;

    TC c = {0};
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int k_tile = 0; k_tile < T_K_tile; ++k_tile) {
        int global_k = k_outer * T_K_tile + k_tile;
        if (global_k >= K) {
          break;
        }

        TA a = static_cast<const TA *>(A_data)[global_m * K + global_k];
        TB b = static_cast<const TB *>(B_data)[global_n * K + global_k];
        c += static_cast<TC>(a) * static_cast<TC>(b);
      }
    }

    static_cast<TC *>(C_data)[global_m * N + global_n] = c;
  }
};

template <typename T> __device__ static T fma(T a, T b, T c) {
  if constexpr (std::is_integral_v<T>) {
    return a * b + c;
  } else {
    return std::fma(a, b, c);
  }
}

template <int T_SubgroupSize, typename T>
__device__ static T subgroup_reduce(T value) {
  for (int offset = T_SubgroupSize >> 1; offset > 0; offset = offset >> 1) {
    value += __shfl_down(value, offset, T_SubgroupSize);
  }
  return value;
};

template <int T_SubgroupSize>
__device__ static _Float16 subgroup_reduce(_Float16 value) {
  float reduced = static_cast<float>(value);
  for (int offset = T_SubgroupSize >> 1; offset > 0; offset = offset >> 1) {
    reduced += __shfl_down(reduced, offset, T_SubgroupSize);
  }
  return static_cast<_Float16>(reduced);
}

template <Type A_type, Type B_type, Type C_type> struct AmdDotTraits;

using f16x2 = _Float16 __attribute__((ext_vector_type(2)));

__device__ static _Float16 f16FromBits(uint16_t bits) {
  union {
    uint16_t bits;
    _Float16 value;
  } converter = {bits};
  return converter.value;
}

__device__ static uint16_t f16ToBits(_Float16 value) {
  union {
    _Float16 value;
    uint16_t bits;
  } converter = {value};
  return converter.bits;
}

__device__ static float f32FromBits(uint32_t bits) {
  union {
    uint32_t bits;
    float value;
  } converter = {bits};
  return converter.value;
}

__device__ static uint32_t f32ToBits(float value) {
  union {
    float value;
    uint32_t bits;
  } converter = {value};
  return converter.bits;
}

enum DotSignedZeroCheckIndex {
  kF16F16ToF32DotSignedZero = 0,
  kF16F16ToF16DotSignedZero = 1,
  kFp8Fp8ToF32DotSignedZero = 2,
  kNumDotSignedZeroChecks = 3,
};

enum DotSignedZeroInputIndex {
  kInputF16A0 = 0,
  kInputF16A1 = 1,
  kInputF16B0 = 2,
  kInputF16B1 = 3,
  kInputF16C = 4,
  kInputF32C = 5,
  kInputFp8A0 = 6,
  kInputFp8A1 = 7,
  kInputFp8A2 = 8,
  kInputFp8A3 = 9,
  kInputFp8B0 = 10,
  kInputFp8B1 = 11,
  kInputFp8B2 = 12,
  kInputFp8B3 = 13,
  kNumDotSignedZeroInputs = 14,
};

__global__ static void checkDotSignedZeroKernel(const uint32_t *input_bits,
                                                uint32_t *output_bits) {
  for (int i = 0; i < kNumDotSignedZeroChecks; ++i) {
    output_bits[i] = 0xffffffffu;
  }

  const f16x2 f16_a = {
      f16FromBits(static_cast<uint16_t>(input_bits[kInputF16A0])),
      f16FromBits(static_cast<uint16_t>(input_bits[kInputF16A1]))};
  const f16x2 f16_b = {
      f16FromBits(static_cast<uint16_t>(input_bits[kInputF16B0])),
      f16FromBits(static_cast<uint16_t>(input_bits[kInputF16B1]))};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fdot2_f16_f16)) {
    const _Float16 c =
        f16FromBits(static_cast<uint16_t>(input_bits[kInputF16C]));
    output_bits[kF16F16ToF16DotSignedZero] =
        f16ToBits(__builtin_amdgcn_fdot2_f16_f16(f16_a, f16_b, c));
  }

  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fdot2)) {
    const float c = f32FromBits(input_bits[kInputF32C]);
    output_bits[kF16F16ToF32DotSignedZero] =
        f32ToBits(__builtin_amdgcn_fdot2(f16_a, f16_b, c, false));
  }

#if defined(__gfx1201__)
  auto pack4_u8 = [](const uint32_t *values) {
    return (values[0] & 0xffu) | ((values[1] & 0xffu) << 8) |
           ((values[2] & 0xffu) << 16) | ((values[3] & 0xffu) << 24);
  };
  const float c = f32FromBits(input_bits[kInputF32C]);
  output_bits[kFp8Fp8ToF32DotSignedZero] = f32ToBits(
      __builtin_amdgcn_dot4_f32_fp8_fp8(pack4_u8(&input_bits[kInputFp8A0]),
                                        pack4_u8(&input_bits[kInputFp8B0]),
                                        c));
#endif
}

struct DotSignedZeroCheck {
  const char *name;
  DotSignedZeroCheckIndex index;
  uint32_t expected_bits;
  bool supported;
};

bool shouldRunDotSignedZeroCheck(const DotSignedZeroCheck &check) {
  const char *filter = getenv("FILTER");
  if (!check.supported) {
    return false;
  }
  if (filter && !strstr(check.name, filter)) {
    return false;
  }
  return true;
}

void checkDotSignedZero(const DeviceCapabilities &device) {
  const DotSignedZeroCheck checks[] = {
      {"AmdDotReduceKernel.fdot2_f32_f16_signed_zero",
       kF16F16ToF32DotSignedZero, 0x80000000u, device.has_f16_f32_dot},
      {"AmdDotReduceKernel.fdot2_f16_f16_signed_zero",
       kF16F16ToF16DotSignedZero, 0x8000u, device.has_f16_output_dot},
      {"AmdDotReduceKernel.dot4_f32_fp8_fp8_signed_zero",
       kFp8Fp8ToF32DotSignedZero, 0x80000000u, device.has_fp8_dot},
  };

  bool should_run = false;
  for (const DotSignedZeroCheck &check : checks) {
    should_run |= shouldRunDotSignedZeroCheck(check);
  }
  if (!should_run) {
    return;
  }

  // f16 LHS = {-0, -0}; f16 RHS = {1, 1}; f16 C = -0; f32 C = -0;
  // fp8 LHS = {-0, -0, -0, -0}; fp8 RHS = {1, 1, 1, 1}.
  const uint32_t input_host[] = {0x8000, 0x8000, 0x3c00, 0x3c00, 0x8000,
                                 0x80000000u, 0x80, 0x80, 0x80, 0x80,
                                 0x38, 0x38, 0x38, 0x38};
  static_assert(sizeof(input_host) / sizeof(input_host[0]) ==
                kNumDotSignedZeroInputs);
  uint32_t output_host[kNumDotSignedZeroChecks] = {};
  uint32_t *input_device = nullptr;
  uint32_t *output_device = nullptr;
  HIP_CHECK(hipMalloc(&input_device, sizeof(input_host)));
  HIP_CHECK(hipMalloc(&output_device, sizeof(output_host)));
  HIP_CHECK(hipMemcpy(input_device, input_host, sizeof(input_host),
                      hipMemcpyHostToDevice));
  checkDotSignedZeroKernel<<<1, 1>>>(input_device, output_device);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(output_host, output_device, sizeof(output_host),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(input_device));
  HIP_CHECK(hipFree(output_device));

  bool failed = false;
  for (const DotSignedZeroCheck &check : checks) {
    if (!shouldRunDotSignedZeroCheck(check)) {
      continue;
    }
    uint32_t actual_bits = output_host[check.index];
    if (actual_bits != check.expected_bits) {
      fprintf(stderr,
              "%s: expected signed-zero result bits 0x%08x, got 0x%08x\n",
              check.name, check.expected_bits, actual_bits);
      failed = true;
    }
  }
  if (failed) {
    fprintf(stderr,
            "Dot signed-zero check failed; continuing to run benchmarks.\n");
  }
}

template <> struct AmdDotTraits<Type::FP16, Type::FP16, Type::FP32> {
  using TA = CType<Type::FP16>;
  using TB = CType<Type::FP16>;
  using TC = CType<Type::FP32>;
  static constexpr int kElementsPerDot = 2;

  __device__ static TC dot(const TA *a, const TB *b, TC c) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fdot2)) {
      const f16x2 a_vec = {a[0], a[1]};
      const f16x2 b_vec = {b[0], b[1]};
      return __builtin_amdgcn_fdot2(a_vec, b_vec, c, false);
    }
    __builtin_trap();
    return c;
  }
};

template <> struct AmdDotTraits<Type::FP16, Type::FP16, Type::FP16> {
  using TA = CType<Type::FP16>;
  using TB = CType<Type::FP16>;
  using TC = CType<Type::FP16>;
  static constexpr int kElementsPerDot = 2;

  __device__ static TC dot(const TA *a, const TB *b, TC c) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fdot2_f16_f16)) {
      const f16x2 a_vec = {a[0], a[1]};
      const f16x2 b_vec = {b[0], b[1]};
      return __builtin_amdgcn_fdot2_f16_f16(a_vec, b_vec, c);
    }
    __builtin_trap();
    return c;
  }
};

template <> struct AmdDotTraits<Type::FP8, Type::FP8, Type::FP32> {
  using TA = CType<Type::FP8>;
  using TB = CType<Type::FP8>;
  using TC = CType<Type::FP32>;
  static constexpr int kElementsPerDot = 4;

  __device__ static uint32_t pack4(const uint8_t *values) {
    return static_cast<uint32_t>(values[0]) |
           (static_cast<uint32_t>(values[1]) << 8) |
           (static_cast<uint32_t>(values[2]) << 16) |
           (static_cast<uint32_t>(values[3]) << 24);
  }

  __device__ static TC dot(const TA *a, const TB *b, TC c) {
#if defined(__gfx1201__)
    return __builtin_amdgcn_dot4_f32_fp8_fp8(pack4(a), pack4(b), c);
#else
    // Host-side target gating skips this kernel outside gfx1201. Keep this
    // fallback so other offload-arch builds can still instantiate the template.
    (void)a;
    (void)b;
    __builtin_trap();
    return c;
#endif
  }
};

template <int T_M_tile, int T_K_tile, int T_SubgroupSize, Type T_A_type,
          Type T_B_type, Type T_C_type>
struct AmdDotReduceKernel : MatvecKernel {
  using Dot = AmdDotTraits<T_A_type, T_B_type, T_C_type>;
  using TA = CType<T_A_type>;
  using TB = CType<T_B_type>;
  using TC = CType<T_C_type>;
  static constexpr int T_N_tile = 1;
  static constexpr int K_VEC = T_K_tile / T_SubgroupSize;
  static_assert(T_K_tile == K_VEC * T_SubgroupSize,
                "Invalid configuration");
  static_assert(K_VEC % Dot::kElementsPerDot == 0,
                "K vector width must contain whole dot operands");

  AmdDotReduceKernel() {
    name = __FUNCTION__;
    this->A_type = T_A_type;
    this->B_type = T_B_type;
    this->C_type = T_C_type;
    M_tile = T_M_tile;
    N_tile = T_N_tile;
    K_tile = T_K_tile;
    num_threads_func = [](const ProblemProperties &) { return T_SubgroupSize; };
    shared_mem_bytes_func = no_shared_mem;
    matvec_func = run;
    this->subgroup_size = T_SubgroupSize;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int lane = threadIdx.x % T_SubgroupSize;

    int K_outer = ceil_div(K, T_K_tile);
    int global_m = m_outer * T_M_tile;
    int global_n = n_outer * T_N_tile;
    if (global_m >= M || global_n >= N)
      return;

    int tile_k = lane * K_VEC;
    TC c[T_M_tile] = {0};
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      TB b[K_VEC] = {0};
      for (int k = 0; k < K_VEC; ++k) {
        int global_k = k_outer * T_K_tile + tile_k + k;
        if (global_k < K) {
          b[k] = static_cast<const TB *>(B_data)[global_n * K + global_k];
        }
      }

      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        TA a[K_VEC] = {0};
        if (global_m + m_tile < M) {
          for (int k = 0; k < K_VEC; ++k) {
            int global_k = k_outer * T_K_tile + tile_k + k;
            if (global_k < K) {
              a[k] = static_cast<const TA *>(
                  A_data)[(global_m + m_tile) * K + global_k];
            }
          }
        }
        for (int k = 0; k < K_VEC; k += Dot::kElementsPerDot) {
          c[m_tile] = Dot::dot(&a[k], &b[k], c[m_tile]);
        }
      }
    }

    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      c[m_tile] = subgroup_reduce<T_SubgroupSize>(c[m_tile]);
    }

    if (lane == 0) {
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        int row = global_m + m_tile;
        if (row < M) {
          static_cast<TC *>(C_data)[row * N + global_n] = c[m_tile];
        }
      }
    }
  }
};

template <int T_M_tile, int T_K_tile, int T_SubgroupSize>
struct SubgroupReduceKernel : MatvecKernel {
  using TA = float;
  using TB = float;
  using TC = float;
  static constexpr int T_N_tile = 1;

  // In this matvec kernel, the whole subgroup processes a row and cooperatively
  // reduces the result using subgroup operations. At the end, only one thread
  // within the subgroups stores the row value to the main memory.
  SubgroupReduceKernel() {
    name = __FUNCTION__;
    A_type = Type::FP32;
    B_type = Type::FP32;
    C_type = Type::FP32;
    M_tile = T_M_tile;
    N_tile = T_N_tile;
    K_tile = T_K_tile;
    num_threads_func = [](const ProblemProperties &) { return T_SubgroupSize; };
    shared_mem_bytes_func = no_shared_mem;
    matvec_func = run;
    this->subgroup_size = T_SubgroupSize;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int lane = threadIdx.x % T_SubgroupSize;

    // The number of adjacent elements to be accessed by a thread. This is used
    // to guarantee vectorized memory accesses, e.g., `global_load_dwordx4`.
    static constexpr int K_VEC = T_K_tile / T_SubgroupSize;
    static_assert(T_K_tile == K_VEC * T_SubgroupSize,
                  "Invalid configuration");
    int K_outer = ceil_div(K, T_K_tile);

    int global_m = m_outer * T_M_tile;
    int global_n = n_outer * T_N_tile;
    if (global_m >= M || global_n >= N)
      return;

    int tile_k = lane * K_VEC;
    // Partial accumulator for the thread,
    // storing `K_VEC` values per row.
    TC c[T_M_tile][K_VEC] = {{0}};

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      // Load the vector operand first to hide the latency. We will need these
      // values for every row slice of the matrix operand.
      TA b[T_N_tile][K_VEC] = {{0}};
      for (int k = 0; k < K_VEC; ++k) {
        int global_k = k_outer * T_K_tile + tile_k + k;
        if (global_k < K) {
          b[0][k] = static_cast<const TB *>(B_data)[global_n * K + global_k];
        }
      }

      TA a[T_M_tile][K_VEC] = {{0}};
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        int row = global_m + m_tile;
        if (row < M) {
          for (int k = 0; k < K_VEC; ++k) {
            int global_k = k_outer * T_K_tile + tile_k + k;
            if (global_k < K) {
              a[m_tile][k] =
                  static_cast<const TA *>(A_data)[row * K + global_k];
            }
          }
        }
      }

      // Partial reduciton. Ideally, we'd like this to generate FMAs.
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        for (int k = 0; k < K_VEC; ++k) {
          c[m_tile][k] = fma(static_cast<TC>(a[m_tile][k]),
                             static_cast<TC>(b[0][k]), c[m_tile][k]);
        }
      }
    }

    // Perform thread-level reduction first, producing a single partial result
    // per thread.
    TC res[T_M_tile] = {{0}};
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      for (int k = 0; k < K_VEC; ++k) {
        res[m_tile] += c[m_tile][k];
      }
    }

    // Perform subgroup-level reduction, such that thread 0 contains the final
    // result for the whole subgroup.
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      res[m_tile] = subgroup_reduce<T_SubgroupSize>(res[m_tile]);
    }
    if (lane == 0) {
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        int row = global_m + m_tile;
        if (row < M) {
          static_cast<TC *>(C_data)[row * N + global_n] = res[m_tile];
        }
      }
    }
  }
};

template <int T_M_tile, int T_K_tile, int T_SubgroupSize>
struct WorkgroupReduceKernel : MatvecKernel {
  using TA = float;
  using TB = float;
  using TC = float;
  static constexpr int T_N_tile = 1;
  static constexpr int K_VEC =
      std::min(T_K_tile, 16 / static_cast<int>(sizeof(TA)));
  static constexpr int K_outer = T_K_tile / K_VEC;
  static_assert(T_K_tile == K_VEC * K_outer, "Invalid configuration");

  // In this matvec kernel, the whole workgroups processes a row and cooperatively
  // reduces the result using subgroup operations. At the end, only one thread
  // within the subgroups stores the row value to the main memory.
  WorkgroupReduceKernel() {
    name = __FUNCTION__;
    A_type = Type::FP32;
    B_type = Type::FP32;
    C_type = Type::FP32;
    M_tile = T_M_tile;
    N_tile = T_N_tile;
    K_tile = T_K_tile;
    num_threads_func = [](const ProblemProperties &problem) {
      assert(problem.total.K % T_K_tile == 0 && "Unsupported input");
      return problem.total.K / T_K_tile;
    };
    shared_mem_bytes_func = [](const ProblemProperties &problem) {
      int num_subgroups =
          ceil_div(problem.total.K / T_K_tile, T_SubgroupSize);
      assert(num_subgroups <= 32 && "Too many subgroups per workgroup");
      return sizeof(TC) * T_M_tile * num_subgroups;
    };
    matvec_func = run;
    this->subgroup_size = T_SubgroupSize;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    extern __shared__ TC shared_mem[];

    const int num_subgroups = ceil_div(K / T_K_tile, T_SubgroupSize);
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int lane = threadIdx.x % T_SubgroupSize;
    int subgroup = threadIdx.x / T_SubgroupSize;

    int global_m = m_outer * T_M_tile;
    int global_n = n_outer * T_N_tile;
    if (global_m >= M || global_n >= N)
      return;

    static constexpr int K_per_subgroup = T_K_tile * T_SubgroupSize;
    static constexpr int K_per_outer = K_VEC * T_SubgroupSize;
    int tile_k = subgroup * K_per_subgroup + lane * K_VEC;

    // Load the vector operand first to hide the latency. We will need these
    // values for every row slice of the matrix operand.
    TA b[T_N_tile][K_outer][K_VEC] = {{{0}}};
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int k = 0; k < K_VEC; ++k) {
        int global_k = tile_k + k_outer * K_per_outer + k;
        if (global_k < K) {
          b[0][k_outer][k] =
              static_cast<const TB *>(B_data)[global_n * K + global_k];
        }
      }
    }

    TA a[T_M_tile][K_outer][K_VEC] = {{{0}}};
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      int row = global_m + m_tile;
      if (row < M) {
        for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
          for (int k = 0; k < K_VEC; ++k) {
            int global_k = tile_k + k_outer * K_per_outer + k;
            if (global_k < K) {
              a[m_tile][k_outer][k] =
                  static_cast<const TA *>(A_data)[row * K + global_k];
            }
          }
        }
      }
    }

    // Partial accumulator for the thread, storing `T_K_tile` values per row.
    TC c[T_M_tile][K_outer][K_VEC] = {{{0}}};
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
        for (int k = 0; k < K_VEC; ++k) {
          c[m_tile][k_outer][k] =
              fma(static_cast<TC>(a[m_tile][k_outer][k]),
                  static_cast<TC>(b[0][k_outer][k]), c[m_tile][k_outer][k]);
        }
      }
    }

    // Perform thread-level reduction first, producing a single partial
    // result per thread.
    TC res[T_M_tile] = {0};
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
        for (int k = 0; k < K_VEC; ++k) {
          res[m_tile] += c[m_tile][k_outer][k];
        }
      }
    }

    // Perform subgroup-level reduction, such that thread 0 contains the
    // final result for the whole subgroup.
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      TC thread_res = subgroup_reduce<T_SubgroupSize>(res[m_tile]);
      res[m_tile] = thread_res;
    }

    // Store the subgroup results in shared memory.
    if (lane == 0) {
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        shared_mem[subgroup * T_M_tile + m_tile] = res[m_tile];
      }
    }

    __syncthreads();

    // Each subgroup will reduce one or more rows.
    for (int m_tile = subgroup; m_tile < T_M_tile; m_tile += num_subgroups) {
      TC thread_res = -TC{0};
      if (lane < num_subgroups) {
        thread_res = shared_mem[lane * T_M_tile + m_tile];
      }

      TC final = subgroup_reduce<T_SubgroupSize>(thread_res);
      if (lane == 0) {
        int row = global_m + m_tile;
        if (row < M) {
          static_cast<TC *>(C_data)[row * N + global_n] = final;
        }
      }
    }
  }
};

void test(const MatvecKernel &kernel) {
  const char *filter = getenv("FILTER");
  if (filter && !strstr(kernel.name, filter)) {
    return;
  }
  std::printf("%s: A:%s, B:%s, C:%s, tile MxNxK=%dx%dx%d", kernel.name,
              str(kernel.A_type), str(kernel.B_type), str(kernel.C_type),
              kernel.M_tile, kernel.N_tile, kernel.K_tile);
  if (kernel.subgroup_size != 0) {
    std::printf(", subgroup=%d", kernel.subgroup_size);
  }
  std::printf("\n");

  if (!getenv("SKIP_CHECK")) {
    check(kernel);
  }

  MNKShape test_shapes[] = {
      {14336, 1, 4096}, {4096, 1, 14336}, {4096, 1, 4096}, {1024, 1, 4096}};
  for (MNKShape shape : test_shapes) {
    benchmark(kernel, shape);
  }
}

template <int T_SubgroupSize>
void testSubgroupReduceKernels() {
  test(SubgroupReduceKernel<1, T_SubgroupSize, T_SubgroupSize>{});
  test(SubgroupReduceKernel<1, T_SubgroupSize * 4, T_SubgroupSize>{});
  test(SubgroupReduceKernel<4, T_SubgroupSize, T_SubgroupSize>{});
  test(SubgroupReduceKernel<4, T_SubgroupSize * 4, T_SubgroupSize>{});
}

template <int T_SubgroupSize>
void testF16F32DotKernels() {
  test(AmdDotReduceKernel<1, T_SubgroupSize * 4, T_SubgroupSize, Type::FP16,
                          Type::FP16, Type::FP32>{});
  test(AmdDotReduceKernel<4, T_SubgroupSize * 4, T_SubgroupSize, Type::FP16,
                          Type::FP16, Type::FP32>{});
}

template <int T_SubgroupSize>
void testF16F16DotKernels() {
  test(AmdDotReduceKernel<1, T_SubgroupSize * 4, T_SubgroupSize, Type::FP16,
                          Type::FP16, Type::FP16>{});
  test(AmdDotReduceKernel<4, T_SubgroupSize * 4, T_SubgroupSize, Type::FP16,
                          Type::FP16, Type::FP16>{});
}

template <int T_SubgroupSize>
void testFp8DotKernels() {
  test(AmdDotReduceKernel<1, T_SubgroupSize * 4, T_SubgroupSize, Type::FP8,
                          Type::FP8, Type::FP32>{});
  test(AmdDotReduceKernel<4, T_SubgroupSize * 4, T_SubgroupSize, Type::FP8,
                          Type::FP8, Type::FP32>{});
}

template <int T_SubgroupSize>
void testWorkgroupReduceKernels() {
  test(WorkgroupReduceKernel<1, 4 * 4, T_SubgroupSize>{});
  test(WorkgroupReduceKernel<2, 4 * 4, T_SubgroupSize>{});
  test(WorkgroupReduceKernel<3, 4 * 4, T_SubgroupSize>{});
  test(WorkgroupReduceKernel<4, 4 * 4, T_SubgroupSize>{});
}

int main(int argc, char **argv) {
  if (argc != 1) {
    fprintf(stderr, "usage: %s\n", argv[0]);
    return EXIT_FAILURE;
  }
  DeviceCapabilities device = detectDeviceCapabilities();
  std::printf("Detected target: %s\n", device.name);

  if (!getenv("SKIP_CHECK")) {
    checkDotSignedZero(device);
  }

  test(NaiveMatmulKernel{});

  if (device.subgroup_size == kSubgroupSize32) {
    testSubgroupReduceKernels<kSubgroupSize32>();
    testF16F32DotKernels<kSubgroupSize32>();
    if (device.has_f16_output_dot) {
      testF16F16DotKernels<kSubgroupSize32>();
    }
    if (device.has_fp8_dot) {
      testFp8DotKernels<kSubgroupSize32>();
    }
    testWorkgroupReduceKernels<kSubgroupSize32>();
  } else if (device.subgroup_size == kSubgroupSize64) {
    testSubgroupReduceKernels<kSubgroupSize64>();
    testF16F32DotKernels<kSubgroupSize64>();
    if (device.has_f16_output_dot) {
      testF16F16DotKernels<kSubgroupSize64>();
    }
    testWorkgroupReduceKernels<kSubgroupSize64>();
  }
  return 0;
}
