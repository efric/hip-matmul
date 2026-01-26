// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hip"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

// Keep the compute path disabled until the sparse-trick logic is wired up.
#ifndef SKINNY_GEMM_ENABLE_COMPUTE
#define SKINNY_GEMM_ENABLE_COMPUTE 0
#endif

// Device function signature for a skinny GEMM (A[MxK] * B[NxK]) -> C[MxN].
using skinny_gemm_func_t = void (*)(const void * /*A*/, const void * /*B*/,
                                    void * /*C*/, int /*M*/, int /*N*/,
                                    int /*K*/);

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

void print(const ProblemProperties &problem, FILE *file = stderr) {
  fprintf(file, "A:%s, B:%s, C:%s, Total MxNxK=%dx%dx%d\n", str(problem.A_type),
          str(problem.B_type), str(problem.C_type), problem.total.M,
          problem.total.N, problem.total.K);
  fprintf(file, "\tOuter MxNxK=%dx%dx%d, Tile MxNxK=%dx%dx%d\n",
          problem.outer.M, problem.outer.N, problem.outer.K, problem.tile.M,
          problem.tile.N, problem.tile.K);
}

int flat_a_size(const ProblemProperties &problem) {
  return type_size(problem.A_type) * problem.total.M * problem.total.K;
}

int flat_b_size(const ProblemProperties &problem) {
  return type_size(problem.B_type) * problem.total.N * problem.total.K;
}

int flat_c_size(const ProblemProperties &problem) {
  return type_size(problem.C_type) * problem.total.M * problem.total.N;
}

struct SkinnyGemmKernel {
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
      shared_mem_bytes_func; // Amount of shared memory per workgroup.
  skinny_gemm_func_t gemm_func; // Device kernel pointer.
  bool supports_check;          // Whether to run correctness checks.
};

void printSparseTrickScaffold(const SkinnyGemmKernel &kernel,
                              const ProblemProperties &problem,
                              const dim3 &grid_dim, const dim3 &block_dim,
                              size_t shared_mem_bytes, size_t alloc_a_bytes,
                              size_t alloc_b_bytes, size_t alloc_c_bytes);

ProblemProperties getBenchmarkProblemSize(const SkinnyGemmKernel &kernel,
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

ProblemProperties getCheckProblemSize(const SkinnyGemmKernel &kernel) {
  int M = getIntEnvVar("M", 8);
  int N = getIntEnvVar("N", 13312);
  int K = getIntEnvVar("K", 16384);
  MNKShape total = {M, N, K};
  return getBenchmarkProblemSize(kernel, total);
}

template <Type A_type, Type B_type, Type C_type>
void checkSkinnyGemmResults(const void *A_data_void, const void *B_data_void,
                            const void *C_data_void,
                            const ProblemProperties &problem) {
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
          TC c = {0};
          for (int k_outer = 0; k_outer < problem.outer.K; ++k_outer) {
            for (int k_tile = 0; k_tile < problem.tile.K; ++k_tile) {
              int global_k = k_outer * problem.tile.K + k_tile;
              TA a = A_data[global_m * problem.total.K + global_k];
              TB b = B_data[global_n * problem.total.K + global_k];
              c += static_cast<TC>(a) * static_cast<TC>(b);
            }
          }
          TC expected = c;
          TC actual = C_data[global_m * problem.total.N + global_n];
          if (actual != expected) {
            fprintf(stderr,
                    "skinny_gemm numerical error: actual(%g) != "
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

void checkSkinnyGemmResults(Type A_type, Type B_type, Type C_type,
                            const void *A_data_void, const void *B_data_void,
                            const void *C_data_void,
                            const ProblemProperties &problem) {
  if (A_type == Type::FP8 || B_type == Type::FP8) {
    fprintf(stderr,
            "Skipping FP8 correctness check (host conversion not wired up yet)\n");
    return;
  }
#define HANDLE_CASE(A, B, C)                                                   \
  if (A_type == Type::A && B_type == Type::B && C_type == Type::C) {           \
    checkSkinnyGemmResults<Type::A, Type::B, Type::C>(A_data_void, B_data_void,\
                                                      C_data_void, problem);  \
    return;                                                                    \
  }
  HANDLE_CASE(FP32, FP32, FP32)
  HANDLE_CASE(FP16, FP16, FP32)
  HANDLE_CASE(SI8, SI8, SI32)
#undef HANDLE_CASE

  fprintf(stderr, "%s:%d: unhandled types\n", __FILE__, __LINE__);
  abort();
}

void check(const SkinnyGemmKernel &kernel, const ProblemProperties &problem) {
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
  printSparseTrickScaffold(kernel, problem, grid_dim, block_dim,
                           shared_mem_bytes, A_host_data.size(),
                           B_host_data.size(), C_host_data.size());

  HIP_CHECK(hipGetLastError());
  kernel.gemm_func<<<grid_dim, block_dim, shared_mem_bytes, hipStreamDefault>>>(
      A_device_buffer, B_device_buffer, C_device_buffer, problem.total.M,
      problem.total.N, problem.total.K);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_host_data.size(),
                      hipMemcpyDeviceToHost));
  checkSkinnyGemmResults(kernel.A_type, kernel.B_type, kernel.C_type,
                         A_host_data.data(), B_host_data.data(),
                         C_host_data.data(), problem);

  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
}

void check(const SkinnyGemmKernel &kernel) {
  check(kernel, getCheckProblemSize(kernel));
}

void benchmark(const SkinnyGemmKernel &kernel, MNKShape total) {
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
  printSparseTrickScaffold(kernel, problem, grid_dim, block_dim,
                           shared_mem_bytes, A_host_data.size(),
                           B_host_data.size(), C_host_data.size());

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
      kernel.gemm_func<<<grid_dim, block_dim, shared_mem_bytes,
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

// FP8 types and packed vector aliases, mirroring hf-rocm-kernels.
using fp8 = __hip_fp8_storage_t;
using fp8x8 = __attribute__((__vector_size__(8 * sizeof(fp8)))) fp8;
using fp8x16 = __attribute__((__vector_size__(16 * sizeof(fp8)))) fp8;
using fp8_4x2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using fp8_4x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

struct SparseTrickConfig {
  static constexpr int WARP_SIZE = 64;
  static constexpr int NB_BANKS = 32;
  static constexpr int E_PER_BANK = 4;
  static constexpr int A_LANES = 1;
  static constexpr int B_LANES = 1;
  static constexpr int OP_M = 8;
  static constexpr int OP_N = 16;
  static constexpr int OP_K = 64;
  static constexpr int OPS = 4;
  static constexpr int WARPTILE_M = OP_M * A_LANES;
  static constexpr int WARPTILE_N = OP_N * B_LANES;
  static constexpr int WARPTILE_K = OP_K * OPS;
  static constexpr int A_OP_BYTES = OP_M * OP_K; // 8x64 bytes (one SMFMAC A).
  static constexpr int B_OP_BYTES = OP_N * OP_K; // 16x64 bytes (one SMFMAC B).
  static constexpr int A_BANK_BYTES = A_OP_BYTES; // 8x64 bank.
  static constexpr int A_HALF_TILE_BYTES = A_BANK_BYTES * 2; // 8x128 bytes.
  static constexpr int B_PANEL_BYTES = B_OP_BYTES / 2; // 16x32 bytes.
  static constexpr int A_TILE_BYTES = WARPTILE_M * WARPTILE_K; // 8x256 bytes.
  static constexpr int B_TILE_BYTES = WARPTILE_N * WARPTILE_K; // 16x256 bytes.
  static constexpr int SPARSITY_EVEN = 0x00004444;
  static constexpr int SPARSITY_ODD = 0x0000EEEE;
};

static_assert(sizeof(fp8) == 1, "FP8 storage size should be 1 byte.");
static_assert(sizeof(fp8x8) == 8, "fp8x8 should pack 8 bytes.");
static_assert(sizeof(fp8x16) == 16, "fp8x16 should pack 16 bytes.");
static_assert(sizeof(fp8_4x2) == 8, "fp8_4x2 should pack 8 bytes.");
static_assert(sizeof(fp8_4x4) == 16, "fp8_4x4 should pack 16 bytes.");
static_assert(sizeof(f32x4) == 16, "f32x4 should pack 16 bytes.");
static_assert(SparseTrickConfig::OP_M == 8, "Scaffold assumes OP_M=8.");
static_assert(SparseTrickConfig::OP_N == 16, "Scaffold assumes OP_N=16.");
static_assert(SparseTrickConfig::OP_K == 64, "Scaffold assumes OP_K=64.");
static_assert(SparseTrickConfig::OPS == 4, "Scaffold assumes OPS=4.");
static_assert(SparseTrickConfig::A_LANES == 1,
              "Scaffold assumes A_LANES=1.");
static_assert(SparseTrickConfig::B_LANES == 1,
              "Scaffold assumes B_LANES=1.");
static_assert(SparseTrickConfig::OPS % 2 == 0,
              "OPS must be even to form 8x128 half tiles.");
static_assert(SparseTrickConfig::OP_K % 2 == 0,
              "OP_K must be even to form 16x32 panels.");
static_assert(SparseTrickConfig::A_TILE_BYTES ==
                  SparseTrickConfig::A_OP_BYTES * SparseTrickConfig::OPS,
              "A tile bytes mismatch.");
static_assert(SparseTrickConfig::B_TILE_BYTES ==
                  SparseTrickConfig::B_OP_BYTES * SparseTrickConfig::OPS,
              "B tile bytes mismatch.");
static_assert(SparseTrickConfig::A_HALF_TILE_BYTES ==
                  SparseTrickConfig::A_OP_BYTES * 2,
              "A half-tile bytes mismatch.");
static_assert(SparseTrickConfig::B_PANEL_BYTES ==
                  SparseTrickConfig::B_OP_BYTES / 2,
              "B panel bytes mismatch.");

struct SharedMemLayout {
  static constexpr int kAlign = 16;
  static constexpr int A_offset = 0;
  static constexpr int A_bytes = SparseTrickConfig::A_TILE_BYTES;
  static constexpr int B_offset = round_up_to_po2<kAlign>(A_bytes);
  static constexpr int B_bytes = SparseTrickConfig::B_TILE_BYTES;
  static constexpr int total_bytes = B_offset + B_bytes;
};

void printSparseTrickScaffold(const SkinnyGemmKernel &kernel,
                              const ProblemProperties &problem,
                              const dim3 &grid_dim, const dim3 &block_dim,
                              size_t shared_mem_bytes, size_t alloc_a_bytes,
                              size_t alloc_b_bytes, size_t alloc_c_bytes) {
  const size_t A_elements =
      static_cast<size_t>(problem.total.M) * problem.total.K;
  const size_t B_elements =
      static_cast<size_t>(problem.total.N) * problem.total.K;
  const size_t C_elements =
      static_cast<size_t>(problem.total.M) * problem.total.N;
  const size_t A_bytes =
      A_elements * static_cast<size_t>(type_size(kernel.A_type));
  const size_t B_bytes =
      B_elements * static_cast<size_t>(type_size(kernel.B_type));
  const size_t C_bytes =
      C_elements * static_cast<size_t>(type_size(kernel.C_type));
  const int k_blocks =
      ceil_div(problem.total.K, SparseTrickConfig::WARPTILE_K);

  fprintf(stderr, "Skinny GEMM scaffold sizes:\n");
  print(problem, stderr);
  fprintf(stderr,
          "\tSparse trick: OP_M=%d OP_N=%d OP_K=%d OPS=%d (WARPTILE_K=%d)\n",
          SparseTrickConfig::OP_M, SparseTrickConfig::OP_N,
          SparseTrickConfig::OP_K, SparseTrickConfig::OPS,
          SparseTrickConfig::WARPTILE_K);
  fprintf(stderr,
          "\tWarp tile: M=%d N=%d K=%d (A_LANES=%d, B_LANES=%d, warp=%d)\n",
          SparseTrickConfig::WARPTILE_M, SparseTrickConfig::WARPTILE_N,
          SparseTrickConfig::WARPTILE_K, SparseTrickConfig::A_LANES,
          SparseTrickConfig::B_LANES, SparseTrickConfig::WARP_SIZE);
  fprintf(stderr,
          "\tA tile: %dx%d bytes=%d (per-op=%d, half-tile=%d, bank=%d)\n",
          SparseTrickConfig::OP_M, SparseTrickConfig::WARPTILE_K,
          SparseTrickConfig::A_TILE_BYTES, SparseTrickConfig::A_OP_BYTES,
          SparseTrickConfig::A_HALF_TILE_BYTES,
          SparseTrickConfig::A_BANK_BYTES);
  fprintf(stderr,
          "\tB tile: %dx%d bytes=%d (per-op=%d, panel=%d)\n",
          SparseTrickConfig::OP_N, SparseTrickConfig::WARPTILE_K,
          SparseTrickConfig::B_TILE_BYTES, SparseTrickConfig::B_OP_BYTES,
          SparseTrickConfig::B_PANEL_BYTES);
  fprintf(stderr,
          "\tShared mem: align=%d A_off=%d A_bytes=%d B_off=%d B_bytes=%d "
          "total=%d\n",
          SharedMemLayout::kAlign, SharedMemLayout::A_offset,
          SharedMemLayout::A_bytes, SharedMemLayout::B_offset,
          SharedMemLayout::B_bytes, SharedMemLayout::total_bytes);
  fprintf(stderr,
          "\tBuffers (logical bytes): A=%zu B=%zu C=%zu (elements A=%zu "
          "B=%zu C=%zu)\n",
          A_bytes, B_bytes, C_bytes, A_elements, B_elements, C_elements);
  fprintf(stderr, "\tBuffers (allocated bytes): A=%zu B=%zu C=%zu\n",
          alloc_a_bytes, alloc_b_bytes, alloc_c_bytes);
  fprintf(stderr,
          "\tLaunch: grid=(%u,%u,%u) block=(%u,%u,%u) threads=%u shared=%zu\n",
          static_cast<unsigned>(grid_dim.x),
          static_cast<unsigned>(grid_dim.y),
          static_cast<unsigned>(grid_dim.z),
          static_cast<unsigned>(block_dim.x),
          static_cast<unsigned>(block_dim.y),
          static_cast<unsigned>(block_dim.z),
          static_cast<unsigned>(block_dim.x), shared_mem_bytes);
  fprintf(stderr, "\tK blocks: %d (K=%d / %d)\n", k_blocks, problem.total.K,
          SparseTrickConfig::WARPTILE_K);
  fprintf(stderr, "\tAlignment: M%%OP_M=%d N%%OP_N=%d K%%WARPTILE_K=%d\n",
          problem.total.M % SparseTrickConfig::OP_M,
          problem.total.N % SparseTrickConfig::OP_N,
          problem.total.K % SparseTrickConfig::WARPTILE_K);
#if SKINNY_GEMM_ENABLE_COMPUTE
  fprintf(stderr, "\tCompute path: enabled\n");
#else
  fprintf(stderr, "\tCompute path: disabled (stub)\n");
#endif
}

struct SparseTrickOpM8Kernel : SkinnyGemmKernel {
  SparseTrickOpM8Kernel() {
    name = __FUNCTION__;
    A_type = Type::FP8;
    B_type = Type::FP8;
    C_type = Type::FP16;
    M_tile = SparseTrickConfig::OP_M;
    N_tile = SparseTrickConfig::OP_N;
    K_tile = SparseTrickConfig::WARPTILE_K;
    num_threads_func = [](const ProblemProperties &) {
      return SparseTrickConfig::WARP_SIZE;
    };
    shared_mem_bytes_func = [](const ProblemProperties &) {
      return static_cast<size_t>(SharedMemLayout::total_bytes);
    };
    gemm_func = run;
    supports_check = false;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    constexpr int OP_M = SparseTrickConfig::OP_M;
    constexpr int OP_N = SparseTrickConfig::OP_N;
    constexpr int OP_K = SparseTrickConfig::OP_K;
    constexpr int OPS = SparseTrickConfig::OPS;
    constexpr int WARPTILE_K = SparseTrickConfig::WARPTILE_K;

    const int lane = threadIdx.x & (warpSize - 1);
    const int m_outer = blockIdx.x;
    const int n_outer = blockIdx.y;
    const int global_m_base = m_outer * OP_M;
    const int global_n_base = n_outer * OP_N;
    if (global_m_base >= M || global_n_base >= N) {
      return;
    }
    if (global_m_base + OP_M > M || global_n_base + OP_N > N) {
      // TODO: handle partial tiles when M/N are not multiples of OP_M/OP_N.
      return;
    }

    const fp8 *A = static_cast<const fp8 *>(A_data) + global_m_base * K;
    const fp8 *B = static_cast<const fp8 *>(B_data) + global_n_base * K;
    _Float16 *C =
        static_cast<_Float16 *>(C_data) + global_m_base * N + global_n_base;

    extern __shared__ std::byte smem[];
    fp8 *A_smem = reinterpret_cast<fp8 *>(smem + SharedMemLayout::A_offset);
    fp8 *B_smem = reinterpret_cast<fp8 *>(smem + SharedMemLayout::B_offset);

    const int K_outer = ceil_div(K, WARPTILE_K);

#if SKINNY_GEMM_ENABLE_COMPUTE
    // Sparse trick overview (see fullpicture-sparse.md):
    // 1) Load one 8x256 A tile per K block into LDS using the swizzle
    //    K0 K1 K4 K5 | K8 K9 K12 K13 | K2 K3 K6 K7 | K10 K11 K14 K15,
    //    split into 8x64 banks (two banks per 8x128 half-tile).
    // 2) Load one 16x256 B tile per K block into LDS using two 4x32 panels
    //    per 16x64 sub-tile, stacked for OPS=4.
    // 3) For op=0..3, load A/B registers from LDS and call
    //    __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8 with sparse indices
    //    (even lanes: 0x00004444, odd lanes: 0x0000EEEE).
    // 4) Fuse D regs (D0+D1, D2+D3), shuffle lane pairs to map rows, convert
    //    to fp16, and write to C.
    //
    // TODO: implement the LDS producers/consumers without warp specialization.
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    const int sparsity_indices =
        (lane & 1) ? SparseTrickConfig::SPARSITY_ODD
                   : SparseTrickConfig::SPARSITY_EVEN;

    for (int k_block = 0; k_block < K_outer; ++k_block) {
      const int k_base = k_block * WARPTILE_K;
      (void)k_base;

      // TODO: load A into A_smem (sparse swizzle) and B into B_smem (dense).
      // TODO: issue OPS smfmacs, loading fp8x8/fp8x16 from LDS each op.
      (void)A_smem;
      (void)B_smem;
      (void)sparsity_indices;
      (void)OP_K;
      (void)OPS;
    }

    // TODO: post-process acc and store to C.
    (void)acc;
#else
    // Stub: keep the kernel non-empty and compilable.
    if (lane == 0) {
      C[0] = static_cast<_Float16>(0);
    }
    (void)A;
    (void)B;
    (void)A_smem;
    (void)B_smem;
    (void)K_outer;
    (void)OP_K;
    (void)OPS;
#endif
  }
};

void test(const SkinnyGemmKernel &kernel) {
  const char *filter = getenv("FILTER");
  if (filter && !strstr(kernel.name, filter)) {
    return;
  }
  std::printf("%s: A:%s, B:%s, C:%s, tile MxNxK=%dx%dx%d\n", kernel.name,
              str(kernel.A_type), str(kernel.B_type), str(kernel.C_type),
              kernel.M_tile, kernel.N_tile, kernel.K_tile);

  if (!getenv("SKIP_CHECK") && kernel.supports_check) {
    check(kernel);
  } else if (!kernel.supports_check) {
    std::fprintf(stderr,
                 "Skipping check for %s (supports_check=false)\n",
                 kernel.name);
  }

  MNKShape test_shapes[] = {{8, 13312, 16384}};
  for (MNKShape shape : test_shapes) {
    benchmark(kernel, shape);
  }
}

int main() {
  test(SparseTrickOpM8Kernel{});
  return 0;
}
