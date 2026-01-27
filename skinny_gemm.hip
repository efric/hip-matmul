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
using shared_mem_bytes_func_t =
    size_t (*)(const ProblemProperties & /*problem*/);

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
      shared_mem_bytes_func;    // Amount of shared memory per workgroup.
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
    fprintf(
        stderr,
        "Skipping FP8 correctness check (host conversion not wired up yet)\n");
    return;
  }
#define HANDLE_CASE(A, B, C)                                                   \
  if (A_type == Type::A && B_type == Type::B && C_type == Type::C) {           \
    checkSkinnyGemmResults<Type::A, Type::B, Type::C>(                         \
        A_data_void, B_data_void, C_data_void, problem);                       \
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
                         hipStreamDefault>>>(A_device_buffer, B_device_buffer,
                                             C_device_buffer, problem.total.M,
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
using fp8_4 = int;
using fp8x8 = __attribute__((__vector_size__(8 * sizeof(fp8)))) fp8;
using fp8x16 = __attribute__((__vector_size__(16 * sizeof(fp8)))) fp8;
using fp8_4x2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using fp8_4x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

struct SparseTrickConfig {
  static constexpr int WARP_SIZE = 64;
  static constexpr int NB_BANKS = 32;
  static constexpr int E_PER_BANK = 4;
  static constexpr int OP_M = 8;
  static constexpr int OP_N = 16;
  static constexpr int OP_K = 64;
  static constexpr int OPS = 4;
  static constexpr int WARPTILE_K = OP_K * OPS;          // 256
  static constexpr int A_TILE_BYTES = OP_M * WARPTILE_K; // 2048
  static constexpr int B_TILE_BYTES = OP_N * WARPTILE_K; // 4096
  static constexpr int SPARSITY_EVEN = 0x00004444;
  static constexpr int SPARSITY_ODD = 0x0000EEEE;
};

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
  // Local constants for printing only (not required for computation).
  constexpr int A_LANES = 1;
  constexpr int B_LANES = 1;
  constexpr int WARPTILE_M = SparseTrickConfig::OP_M * A_LANES;
  constexpr int WARPTILE_N = SparseTrickConfig::OP_N * B_LANES;
  constexpr int A_OP_BYTES = SparseTrickConfig::OP_M * SparseTrickConfig::OP_K;
  constexpr int B_OP_BYTES = SparseTrickConfig::OP_N * SparseTrickConfig::OP_K;
  constexpr int A_BANK_BYTES = A_OP_BYTES;
  constexpr int A_HALF_TILE_BYTES = A_BANK_BYTES * 2;
  constexpr int B_PANEL_BYTES = B_OP_BYTES / 2;

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
  const int k_blocks = ceil_div(problem.total.K, SparseTrickConfig::WARPTILE_K);

  fprintf(stderr, "Skinny GEMM scaffold sizes:\n");
  print(problem, stderr);
  fprintf(stderr,
          "\tSparse trick: OP_M=%d OP_N=%d OP_K=%d OPS=%d (WARPTILE_K=%d)\n",
          SparseTrickConfig::OP_M, SparseTrickConfig::OP_N,
          SparseTrickConfig::OP_K, SparseTrickConfig::OPS,
          SparseTrickConfig::WARPTILE_K);
  fprintf(stderr,
          "\tWarp tile: M=%d N=%d K=%d (A_LANES=%d, B_LANES=%d, warp=%d)\n",
          WARPTILE_M, WARPTILE_N, SparseTrickConfig::WARPTILE_K, A_LANES,
          B_LANES, SparseTrickConfig::WARP_SIZE);
  fprintf(
      stderr, "\tA tile: %dx%d bytes=%d (per-op=%d, half-tile=%d, bank=%d)\n",
      SparseTrickConfig::OP_M, SparseTrickConfig::WARPTILE_K,
      SparseTrickConfig::A_TILE_BYTES, A_OP_BYTES, A_HALF_TILE_BYTES,
      A_BANK_BYTES);
  fprintf(stderr, "\tB tile: %dx%d bytes=%d (per-op=%d, panel=%d)\n",
          SparseTrickConfig::OP_N, SparseTrickConfig::WARPTILE_K,
          SparseTrickConfig::B_TILE_BYTES, B_OP_BYTES, B_PANEL_BYTES);
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
          static_cast<unsigned>(grid_dim.x), static_cast<unsigned>(grid_dim.y),
          static_cast<unsigned>(grid_dim.z), static_cast<unsigned>(block_dim.x),
          static_cast<unsigned>(block_dim.y),
          static_cast<unsigned>(block_dim.z),
          static_cast<unsigned>(block_dim.x), shared_mem_bytes);
  fprintf(stderr, "\tK blocks: %d (K=%d / %d)\n", k_blocks, problem.total.K,
          SparseTrickConfig::WARPTILE_K);
  fprintf(stderr, "\tAlignment: M%%OP_M=%d N%%OP_N=%d K%%WARPTILE_K=%d\n",
          problem.total.M % SparseTrickConfig::OP_M,
          problem.total.N % SparseTrickConfig::OP_N,
          problem.total.K % SparseTrickConfig::WARPTILE_K);
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

    // Sparse trick overview (see fullpicture-sparse-trick.md):
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
    const int sparsity_indices = (lane & 1) ? SparseTrickConfig::SPARSITY_ODD
                                            : SparseTrickConfig::SPARSITY_EVEN;

    // =========================================================================
    // A tile producer (no queue/split-K): mirrors produce_4_half_tiles.
    // =========================================================================
    // Thread layout for loading one 8x128 half-tile (see fullpicture doc):
    //   K        0   16   32   48   64   80   96  112
    //   row 0:  t0   t1   t2   t3   t4   t5   t6   t7
    //   row 1:  t8   t9   t10  t11  t12  t13  t14  t15
    //   ...
    //   row 7:  t56  t57  t58  t59  t60  t61  t62  t63
    //
    // Each thread loads 16 contiguous FP8 elements along K.
    // Two loads (reg0, reg1) cover the full 8x256 A tile:
    //   reg0: K 0-127 (first half-tile)
    //   reg1: K 128-255 (second half-tile, offset by 128 bytes)
    // =========================================================================

    constexpr int E_PER_THREAD = 16; // Elements per thread per load.
    constexpr int E_PER_BANK =
        SparseTrickConfig::E_PER_BANK; // 4 bytes per LDS column.
    constexpr int THREADS_PER_LD =
        8; // Threads per row (8 threads cover 128 K elements).

    const int lane_id = lane;
    // curr_ld: K offset within row (0, 16, 32, ..., 112).
    const int curr_ld = (lane_id % THREADS_PER_LD) * E_PER_THREAD;
    // curr_ad: Row index (0..7).
    const int curr_ad = lane_id / THREADS_PER_LD;

    // =========================================================================
    // B tile producer constants (mirrors n_full_tiles.cu for OP_K=64, OP_N=16).
    // Unlike A (8 threads per row, 8 rows), B uses 4 threads per K, 16 for N:
    //   Thread layout (K chunk -> 0, 16, 32, 48; N -> 0..15):
    //     K chunk:    0     16    32    48
    //     N=0:       L0    L1    L2    L3
    //     N=1:       L4    L5    L6    L7
    //     ...
    //     N=15:      L60   L61   L62   L63
    // =========================================================================
    constexpr int B_THREADS_PER_LD = 4;  // 4 threads cover 64 K elements.
    constexpr int B_THREADS_PER_AD = 16; // 16 threads for 16 N columns.
    constexpr int B_LOADS = OPS;         // 4 loads for 4 sub-tiles.
    constexpr int NB_BANKS = SparseTrickConfig::NB_BANKS;

    // Thread position for B loading.
    const int b_curr_ld =
        (lane_id % B_THREADS_PER_LD) * E_PER_THREAD;  // K: 0,16,32,48
    const int b_curr_ad = lane_id / B_THREADS_PER_LD; // N: 0..15

    // Compute source pointer for this thread.
    // Each thread reads from row curr_ad, starting at K position curr_ld.
    const fp8 *a_src = A + curr_ad * K + curr_ld;

    // =========================================================================
    // Compute LDS buffer pointer for this thread.
    // The LDS layout for one 8x128 half-tile has two 8x64 banks:
    //   Bank 0 (K 0-63):   4 lines x 32 columns x 4 bytes = 512 bytes
    //   Bank 1 (K 64-127): 4 lines x 32 columns x 4 bytes = 512 bytes
    //
    // Each line has 32 columns (8 rows x 4 K chunks per bank).
    // Column assignment:
    //   col = curr_ad + (curr_ld % 64) / 16 * 8
    //       = row + K_chunk_within_bank * 8
    //
    // The formula below encodes this:
    //   E_PER_BANK * curr_ad     -> column offset for row (0, 4, 8, ..., 28)
    //   (curr_ld / 64) * 512     -> bank offset (0 for K<64, 512 for K>=64)
    //   (curr_ld % 64) * 2       -> K chunk offset within bank (0, 32, 64, 96)
    // =========================================================================
    fp8 *a_buf_base = A_smem;
    a_buf_base += E_PER_BANK * curr_ad;
    a_buf_base += (curr_ld / 64) * SparseTrickConfig::NB_BANKS * E_PER_BANK * 4;
    a_buf_base += (curr_ld % 64) * 2;

    // =========================================================================
    // B source pointer (B is NxK layout, stride = K).
    // Each thread reads from column b_curr_ad, starting at K position
    // b_curr_ld.
    // =========================================================================
    const fp8 *b_src = B + b_curr_ad * K + b_curr_ld;

    // =========================================================================
    // B LDS buffer pointer.
    // LDS layout for one 16x64 sub-tile uses two 4-line panels:
    //   Panel 0 (K 0..31): cols 0-15 for K 0-15, cols 16-31 for K 16-31
    //   Panel 1 (K 32..63): cols 0-15 for K 32-47, cols 16-31 for K 48-63
    // Each panel: 4 lines x 32 columns x 4 bytes = 512 bytes.
    //
    // Buffer pointer formula (from n_full_tiles.cu for OP_K=64):
    //   b_curr_ad * E_PER_BANK              -> N column offset (0, 4, 8, ...,
    //   60) (b_curr_ld / 32) * 512              -> panel offset (0 or 512)
    //   (b_curr_ld % 32 >= 16) ? 64 : 0     -> K chunk within panel
    // =========================================================================
    fp8 *b_buf_base = B_smem;
    b_buf_base += b_curr_ad * E_PER_BANK;
    b_buf_base += (b_curr_ld / 32) * NB_BANKS * E_PER_BANK * 4;
    b_buf_base += (b_curr_ld % 32 >= 16) ? 16 * E_PER_BANK : 0;

    // Register arrays for loading and swizzling.
    // Each reg array holds 4 dwords (16 bytes = E_PER_THREAD).
    fp8_4 reg0[E_PER_THREAD / 4];
    fp8_4 reg1[E_PER_THREAD / 4];
    fp8_4 swap_reg[E_PER_THREAD / 4];

    // B tile producer registers (4 loads x 16 bytes each, stored as dwords).
    fp8_4 b_reg[B_LOADS][E_PER_THREAD / 4];

    // =========================================================================
    // Consumer-side buffer pointers (for SMFMAC consumption).
    // A: Lane pairing - even lanes read lines 0-1, odd lanes read lines 2-3.
    // B: Panel-based - lanes 0-31 read panel 0, lanes 32-63 read panel 1.
    // =========================================================================
    fp8 *a_cons_base = A_smem;
    a_cons_base += (lane_id / 2) * E_PER_BANK;
    a_cons_base += (lane_id % 2) * E_PER_BANK * 2 * NB_BANKS;

    fp8 *b_cons_base = B_smem;
    b_cons_base += (lane_id % 32) * E_PER_BANK;
    b_cons_base += (lane_id / 32) * E_PER_BANK * 4 * NB_BANKS;

    // Consumer registers for SMFMAC.
    fp8x8 a_cons[OPS];  // 4 x 8 bytes = 32 bytes for A.
    fp8x16 b_cons[OPS]; // 4 x 16 bytes = 64 bytes for B.

    const int k_blocks = K_outer;

    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      // Source pointer for this K block.
      const fp8 *src_k = a_src + k_block * WARPTILE_K;
      // LDS buffer pointer (fixed for single-buffering).
      fp8_4 *buf = reinterpret_cast<fp8_4 *>(a_buf_base);

      // -----------------------------------------------------------------------
      // Load from global memory to registers.
      // Two 16-byte loads at offset 0 and 128 cover the 8x256 tile:
      //   reg0 <- K[curr_ld .. curr_ld+15] (first half-tile)
      //   reg1 <- K[curr_ld+128 .. curr_ld+143] (second half-tile)
      // -----------------------------------------------------------------------
      asm volatile("global_load_dwordx4 %0, %2, off offset:0  \n\t"
                   "global_load_dwordx4 %1, %2, off offset:128\n\t"
                   : "=v"(reg0), "=v"(reg1)
                   : "v"(src_k));

      // -----------------------------------------------------------------------
      // Swizzle and store first half-tile (reg0) to LDS.
      // The swizzle permutes K0..K15 into:
      //   reg0[0] = K0 K1 K4 K5    (line 0)
      //   reg0[1] = K8 K9 K12 K13  (line 1)
      //   reg0[2] = K2 K3 K6 K7    (line 2)
      //   reg0[3] = K10 K11 K14 K15 (line 3)
      //
      // The v_pack_b32_f16 + v_swap_b32 sequence achieves this permutation.
      // -----------------------------------------------------------------------
      asm volatile("s_waitcnt vmcnt(1)\n\t" // Wait for reg0 load to complete.
                   "v_pack_b32_f16 %0, %4, %5, op_sel:[0,0]\n\t"
                   "v_pack_b32_f16 %1, %4, %5, op_sel:[1,1]\n\t"
                   "v_pack_b32_f16 %2, %6, %7, op_sel:[0,0]\n\t"
                   "v_pack_b32_f16 %3, %6, %7, op_sel:[1,1]\n\t"
                   "v_swap_b32 %0, %4\n\t"
                   "v_swap_b32 %1, %6\n\t"
                   "v_swap_b32 %2, %5\n\t"
                   "v_swap_b32 %3, %7"
                   : "=v"(swap_reg[0]), "=v"(swap_reg[1]), "=v"(swap_reg[2]),
                     "=v"(swap_reg[3])
                   : "v"(reg0[0]), "v"(reg0[1]), "v"(reg0[2]), "v"(reg0[3]));

      // Store swizzled reg0 to LDS. Each line is NB_BANKS (32) dwords apart.
#pragma unroll
      for (int line = 0; line < 4; ++line) {
        buf[line * SparseTrickConfig::NB_BANKS] = reg0[line];
      }

      // -----------------------------------------------------------------------
      // Swizzle and store second half-tile (reg1) to LDS.
      // Same swizzle pattern, but stored at offset OP_K * OP_M / 2 = 256 dwords
      // (1024 bytes) from the first half-tile.
      // -----------------------------------------------------------------------
      asm volatile("s_waitcnt vmcnt(0)\n\t" // Wait for reg1 load to complete.
                   "v_pack_b32_f16 %0, %4, %5, op_sel:[0,0]\n\t"
                   "v_pack_b32_f16 %1, %4, %5, op_sel:[1,1]\n\t"
                   "v_pack_b32_f16 %2, %6, %7, op_sel:[0,0]\n\t"
                   "v_pack_b32_f16 %3, %6, %7, op_sel:[1,1]\n\t"
                   "v_swap_b32 %0, %4\n\t"
                   "v_swap_b32 %1, %6\n\t"
                   "v_swap_b32 %2, %5\n\t"
                   "v_swap_b32 %3, %7"
                   : "=v"(swap_reg[0]), "=v"(swap_reg[1]), "=v"(swap_reg[2]),
                     "=v"(swap_reg[3])
                   : "v"(reg1[0]), "v"(reg1[1]), "v"(reg1[2]), "v"(reg1[3]));

      // Store swizzled reg1 at second half-tile offset (1024 bytes = 256
      // dwords).
      constexpr int HALF_TILE_DWORDS =
          SparseTrickConfig::OP_K * SparseTrickConfig::OP_M / 2;
#pragma unroll
      for (int line = 0; line < 4; ++line) {
        buf[line * SparseTrickConfig::NB_BANKS + HALF_TILE_DWORDS] = reg1[line];
      }

      // -----------------------------------------------------------------------
      // Load B tile from global memory (4 loads for OPS=4 sub-tiles).
      // Each sub-tile is 16x64 bytes (1024 bytes). Offsets 0, 64, 128, 192
      // advance through K positions 0-63, 64-127, 128-191, 192-255.
      // -----------------------------------------------------------------------
      const fp8 *b_src_k = b_src + k_block * WARPTILE_K;
      asm volatile("global_load_dwordx4 %0, %4, off offset:0   \n\t"
                   "global_load_dwordx4 %1, %4, off offset:64  \n\t"
                   "global_load_dwordx4 %2, %4, off offset:128 \n\t"
                   "global_load_dwordx4 %3, %4, off offset:192 \n\t"
                   : "=v"(b_reg[0]), "=v"(b_reg[1]), "=v"(b_reg[2]),
                     "=v"(b_reg[3])
                   : "v"(b_src_k));

      // -----------------------------------------------------------------------
      // Store B tile to LDS (dense, no swizzle).
      // LDS layout for each 16x64 sub-tile: 4 lines x 32 columns x 4 bytes.
      // Each thread stores 16 bytes (4 lines x 4 bytes per line).
      // -----------------------------------------------------------------------
      fp8 *b_buf = b_buf_base;
#pragma unroll
      for (int load = 0; load < B_LOADS; ++load) {
        // Wait for this load to complete. vmcnt(3) for load 0, down to
        // vmcnt(0).
        switch (B_LOADS - load - 1) {
        case 3:
          asm volatile("s_waitcnt vmcnt(3)");
          break;
        case 2:
          asm volatile("s_waitcnt vmcnt(2)");
          break;
        case 1:
          asm volatile("s_waitcnt vmcnt(1)");
          break;
        case 0:
          asm volatile("s_waitcnt vmcnt(0)");
          break;
        }
        // Store 16 bytes: 4 lines x 4 elements, each line NB_BANKS*E_PER_BANK
        // apart.
        const fp8 *b_bytes = reinterpret_cast<const fp8 *>(b_reg[load]);
#pragma unroll
        for (int line = 0; line < E_PER_THREAD / E_PER_BANK; ++line) {
#pragma unroll
          for (int elem = 0; elem < E_PER_BANK; ++elem) {
            b_buf[line * (NB_BANKS * E_PER_BANK) + elem] =
                b_bytes[E_PER_BANK * line + elem];
          }
        }
        // Advance to next sub-tile: 64 threads x 16 bytes = 1024 bytes.
        b_buf += SparseTrickConfig::WARP_SIZE * E_PER_THREAD;
      }

      __syncthreads();

      // -----------------------------------------------------------------------
      // SMFMAC consumption: 4 ops per K block.
      // Each op consumes 8x64 from A and 16x64 from B.
      // -----------------------------------------------------------------------

      // Load A and B for all ops first (latency hiding).
#pragma unroll
      for (int op = 0; op < OPS; ++op) {
        // Load A for this op (8 bytes from swizzled A tile).
        // Per-op offset: op * OP_M * OP_K = op * 512 bytes.
        fp8 *a_op = a_cons_base + op * OP_M * OP_K;
#pragma unroll
        for (int i = 0; i < E_PER_BANK; ++i) {
          a_cons[op][i + 0 * E_PER_BANK] = a_op[i + 0 * NB_BANKS * E_PER_BANK];
        }
#pragma unroll
        for (int i = 0; i < E_PER_BANK; ++i) {
          a_cons[op][i + 1 * E_PER_BANK] = a_op[i + 1 * NB_BANKS * E_PER_BANK];
        }

        // Load B for this op (16 bytes from dense B tile).
        // Per-op offset: op * OP_N * OP_K = op * 1024 bytes.
        fp8 *b_op = b_cons_base + op * OP_N * OP_K;
#pragma unroll
        for (int line = 0; line < 4; ++line) {
#pragma unroll
          for (int i = 0; i < E_PER_BANK; ++i) {
            b_cons[op][i + line * E_PER_BANK] =
                b_op[i + line * NB_BANKS * E_PER_BANK];
          }
        }
      }

      // Wait for LDS reads to complete.
      asm volatile("s_waitcnt lgkmcnt(0)");

      // Issue 4 SMFMACs.
#pragma unroll
      for (int op = 0; op < OPS; ++op) {
        acc = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
            *reinterpret_cast<fp8_4x2 *>(&a_cons[op]),
            *reinterpret_cast<fp8_4x4 *>(&b_cons[op]), acc, sparsity_indices,
            7, // cbsz
            1  // abid
        );
      }
    }

    // =========================================================================
    // Post-processing: Fuse partial results, shuffle, convert to fp16, output.
    // =========================================================================

    // Fusing: Combine 4 partial results into 2 (from fullpicture lines
    // 306-311). D[0] = D[0] + D[1], D[1] = D[2] + D[3].
    acc[0] = acc[0] + acc[1];
    acc[1] = acc[2] + acc[3];

    // Compute output N column: 2 * ((lane_id % 16) / 2) = 0, 0, 2, 2, 4, 4, ...
    const int out_n = 2 * ((lane_id % 16) / 2);

    // Shuffle: Exchange between even/odd lane pairs (from fullpicture lines
    // 317-325). Even lanes keep d0, odd lanes keep d1. Then swap to get 2x2
    // tile per pair.
    const int id_to_swap = 1 - (lane_id % 2);
    const int src_lane = lane_id + 1 - 2 * (lane_id % 2);
    acc[id_to_swap] = __shfl(acc[id_to_swap], src_lane);

    // Compute output M row: (lane_id / 16) * 2 + (lane_id % 2).
    const int out_m = (lane_id / 16) * 2 + (lane_id % 2);

    // Convert to fp16 and write to C.
    // Each lane writes 2 elements at C[out_m, out_n] and C[out_m, out_n+1].
    _Float16 *c_out = C + out_m * N + out_n;
    c_out[0] = static_cast<_Float16>(acc[0]);
    c_out[1] = static_cast<_Float16>(acc[1]);
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
    std::fprintf(stderr, "Skipping check for %s (supports_check=false)\n",
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
