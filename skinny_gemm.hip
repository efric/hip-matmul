// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hip"

#include <cassert>
#include <cstdio>
#include <random>
#include <set>
#include <utility>
#include <vector>

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
  skinny_gemm_func_t gemm_func;        // Device kernel pointer.
};

void printSparseTrickScaffold(const SkinnyGemmKernel &kernel,
                              const ProblemProperties &problem,
                              const dim3 &grid_dim, const dim3 &block_dim,
                              size_t alloc_a_bytes, size_t alloc_b_bytes,
                              size_t alloc_c_bytes);

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
  int N = getIntEnvVar("N", 32);
  int K = getIntEnvVar("K", 512);
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
              c += static_cast<float>(a) * static_cast<float>(b);
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
#define HANDLE_CASE(A, B, C)                                                   \
  if (A_type == Type::A && B_type == Type::B && C_type == Type::C) {           \
    checkSkinnyGemmResults<Type::A, Type::B, Type::C>(                         \
        A_data_void, B_data_void, C_data_void, problem);                       \
    return;                                                                    \
  }
  HANDLE_CASE(FP8, FP8, FP16)
  // TODO: Add more supported cases.
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
  printSparseTrickScaffold(kernel, problem, grid_dim, block_dim,
                           A_host_data.size(), B_host_data.size(),
                           C_host_data.size());

  HIP_CHECK(hipGetLastError());
  kernel.gemm_func<<<grid_dim, block_dim>>>(A_device_buffer, B_device_buffer,
                                            C_device_buffer, problem.total.M,
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
  printSparseTrickScaffold(kernel, problem, grid_dim, block_dim,
                           A_host_data.size(), B_host_data.size(),
                           C_host_data.size());

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
      kernel.gemm_func<<<grid_dim, block_dim>>>(
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

using fp8 = __hip_fp8_storage_t;
using fp8_4 = int;
using fp8x8 = __attribute__((__vector_size__(8 * sizeof(fp8)))) fp8;
using fp8x16 = __attribute__((__vector_size__(16 * sizeof(fp8)))) fp8;
using fp8_4x2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using fp8_4x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

constexpr int WARP_SIZE = 64;
constexpr int NB_BANKS = 32;
constexpr int SPARSITY_EVEN = 0x00004444;
constexpr int SPARSITY_ODD = 0x0000EEEE;

void printSparseTrickScaffold(const SkinnyGemmKernel &kernel,
                              const ProblemProperties &problem,
                              const dim3 &grid_dim, const dim3 &block_dim,
                              size_t alloc_a_bytes, size_t alloc_b_bytes,
                              size_t alloc_c_bytes) {
  if (getIntEnvVar("PRINT_INFO", 0) == 0) {
    return;
  }

  constexpr int A_LANES = 1;
  constexpr int B_LANES = 1;
  const int op_m = kernel.M_tile;
  const int op_n = kernel.N_tile;
  const int op_k = 512 / op_m;
  const int ops = kernel.K_tile / op_k;
  const int warptile_k = kernel.K_tile;
  const int warptile_m = op_m * A_LANES;
  const int warptile_n = op_n * B_LANES;
  const int a_op_bytes = op_m * op_k;
  const int b_op_bytes = op_n * op_k;
  const int a_bank_bytes = a_op_bytes;
  const int a_half_tile_bytes = a_bank_bytes * 2;
  const int b_panel_bytes = b_op_bytes / 2;
  const int a_tile_bytes = op_m * warptile_k;
  const int b_tile_bytes = op_n * warptile_k;

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
  const int k_blocks = ceil_div(problem.total.K, warptile_k);

  fprintf(stderr, "Skinny GEMM FP8 configuration:\n");
  print(problem, stderr);
  fprintf(stderr,
          "\tSparse trick: OP_M=%d OP_N=%d OP_K=%d OPS=%d (WARPTILE_K=%d)\n",
          op_m, op_n, op_k, ops, warptile_k);
  fprintf(stderr,
          "\tWarp tile: M=%d N=%d K=%d (A_LANES=%d, B_LANES=%d, warp=%d)\n",
          warptile_m, warptile_n, warptile_k, A_LANES, B_LANES, WARP_SIZE);
  fprintf(stderr,
          "\tA tile: %dx%d bytes=%d (per-op=%d, half-tile=%d, bank=%d)\n", op_m,
          warptile_k, a_tile_bytes, a_op_bytes, a_half_tile_bytes,
          a_bank_bytes);
  fprintf(stderr, "\tB tile: %dx%d bytes=%d (per-op=%d, panel=%d)\n", op_n,
          warptile_k, b_tile_bytes, b_op_bytes, b_panel_bytes);
  const int shared_total_bytes = a_tile_bytes + b_tile_bytes;
  fprintf(stderr, "\tShared mem (static): A_bytes=%d B_bytes=%d total=%d\n",
          a_tile_bytes, b_tile_bytes, shared_total_bytes);
  fprintf(stderr,
          "\tBuffers (logical bytes): A=%zu B=%zu C=%zu (elements A=%zu "
          "B=%zu C=%zu)\n",
          A_bytes, B_bytes, C_bytes, A_elements, B_elements, C_elements);
  fprintf(stderr, "\tBuffers (allocated bytes): A=%zu B=%zu C=%zu\n",
          alloc_a_bytes, alloc_b_bytes, alloc_c_bytes);
  fprintf(stderr, "\tLaunch: grid=(%u,%u,%u) block=(%u,%u,%u) threads=%u\n",
          static_cast<unsigned>(grid_dim.x), static_cast<unsigned>(grid_dim.y),
          static_cast<unsigned>(grid_dim.z), static_cast<unsigned>(block_dim.x),
          static_cast<unsigned>(block_dim.y),
          static_cast<unsigned>(block_dim.z),
          static_cast<unsigned>(block_dim.x));
  fprintf(stderr, "\tK blocks: %d (K=%d / %d)\n", k_blocks, problem.total.K,
          warptile_k);
}

template <int OP_M, int OP_N, int OPS>
struct SparseTrickFP8Kernel : SkinnyGemmKernel {
  SparseTrickFP8Kernel() {
    name = __FUNCTION__;
    A_type = Type::FP8;
    B_type = Type::FP8;
    C_type = Type::FP16;
    constexpr int OP_K = 512 / OP_M;
    constexpr int WARPTILE_K = OP_K * OPS;
    M_tile = OP_M;
    N_tile = OP_N;
    K_tile = WARPTILE_K;
    num_threads_func = [](const ProblemProperties &) { return WARP_SIZE; };
    gemm_func = run;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    constexpr int OP_K = 512 / OP_M;
    constexpr int WARPTILE_K = OP_K * OPS;
    constexpr int A_TILE_BYTES = OP_M * WARPTILE_K;
    constexpr int B_TILE_BYTES = OP_N * WARPTILE_K;

    // OPS-dependent load counts:
    // A tile: each 16-byte load covers K=16, need OPS/2 loads to cover WARPTILE_K
    //   (with 8 threads per row, 8 loads cover K=128; for WARPTILE_K=64*OPS,
    //    each thread needs OPS/2 loads)
    // B tile: each 16-byte load covers one OP_K=64 chunk, need OPS loads
    static constexpr int A_LOADS = OPS / 2;       // 1, 2, or 4
    static constexpr int B_LOADS = OPS;           // 2, 4, or 8

    const int lane_id = threadIdx.x & (warpSize - 1);
    const int m_outer = blockIdx.x;
    const int n_outer = blockIdx.y;
    const int global_m_base = m_outer * OP_M;
    const int global_n_base = n_outer * OP_N;
    if (global_m_base >= M || global_n_base >= N) {
      return;
    }

    const fp8 *A = static_cast<const fp8 *>(A_data) + global_m_base * K;
    const fp8 *B = static_cast<const fp8 *>(B_data) + global_n_base * K;
    __half *C =
        static_cast<__half *>(C_data) + global_m_base * N + global_n_base;

    __shared__ fp8 A_shared[A_TILE_BYTES];
    __shared__ fp8 B_shared[B_TILE_BYTES];

    const int K_outer = ceil_div(K, WARPTILE_K);

    // Sparse trick overview:
    // 1) Load one 8x256 A tile per K block into LDS using the swizzle
    //    K0 K1 K4 K5 | K8 K9 K12 K13 | K2 K3 K6 K7 | K10 K11 K14 K15,
    //    split into 8x64 banks (two banks per 8x128 half-tile).
    // 2) Load one 16x256 B tile per K block into LDS using two 4x32 sub-slices
    //    per 16x64 slice, stacked for OPS=4.
    // 3) For op=0..3, load A/B registers from LDS and call
    //    __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8 with sparse indices
    //    (even lanes: 0x00004444, odd lanes: 0x0000EEEE).
    // 4) Fuse D regs (D0+D1, D2+D3), shuffle lane pairs to map rows, convert
    //    to fp16, and write to C.
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    const int sparsity_indices = (lane_id & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    // =========================================================================
    // Loading A tile:
    // Thread layout for loading one 8x128 half-tile:
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

    constexpr int E_PER_THREAD = 16;  // Elements per thread.
    constexpr int E_PER_BANK = 4;     // 4 bytes per LDS column.
    constexpr int THREADS_PER_LD = 8; // Threads per row.

    // curr_ld: K offset within row (0, 16, 32, ..., 112).
    const int curr_ld = (lane_id % THREADS_PER_LD) * E_PER_THREAD;
    // curr_ad: Row index (0..7).
    const int curr_ad = lane_id / THREADS_PER_LD;

    // =========================================================================
    // B tile producer constants (OP_K=64, OP_N=16).
    //   Thread layout (K chunk -> 0, 16, 32, 48; N -> 0..15):
    //     K chunk:    0     16    32    48
    //     N=0:       L0    L1    L2    L3
    //     N=1:       L4    L5    L6    L7
    //     ...
    //     N=15:      L60   L61   L62   L63
    // =========================================================================
    constexpr int B_THREADS_PER_LD = OP_K / E_PER_THREAD; // 4 threads per row.

    // Thread position for B loading.
    const int b_curr_ld =
        (lane_id % B_THREADS_PER_LD) * E_PER_THREAD;  // K: 0,16,32,48
    const int b_curr_ad = lane_id / B_THREADS_PER_LD; // N: 0..15

    // Each thread reads from row curr_ad, starting at K position curr_ld.
    const fp8 *a_src = A + curr_ad * K + curr_ld;

    // =========================================================================
    // A LDS buffer pointer.
    // The LDS layout for one 8x128 "half-tile" has two 8x64 logical slices:
    //   slice 0 (K 0-63):   4 lines x 32 columns x 4 bytes = 512 bytes
    //   slice 1 (K 64-127): 4 lines x 32 columns x 4 bytes = 512 bytes
    //
    // Each line has 32 columns.
    // Column assignment:
    //   col = curr_ad + (curr_ld % 64) / 16 * 8
    //       = row + K_chunk_within_bank * 8
    //
    // The formula below encodes this:
    //   E_PER_BANK * curr_ad     -> column offset for row (0, 4, 8, ..., 28)
    //   (curr_ld / 64) * 512     -> bank offset (0 for K<64, 512 for K>=64)
    //   (curr_ld % 64) * 2       -> K chunk offset within bank (0, 32, 64, 96)
    // =========================================================================
    fp8 *a_buf_base = A_shared;
    a_buf_base += E_PER_BANK * curr_ad;
    a_buf_base += (curr_ld / 64) * NB_BANKS * E_PER_BANK * 4;
    a_buf_base += (curr_ld % 64) * 2;

    const fp8 *b_src = B + b_curr_ad * K + b_curr_ld;

    // =========================================================================
    // B LDS buffer pointer.
    // LDS layout for one 16x64 tile is divided into two slices:
    //   Slice 0 (K 0..31): cols 0-15 for K 0-15, cols 16-31 for K 16-31
    //   Slice 1 (K 32..63): cols 0-15 for K 32-47, cols 16-31 for K 48-63
    // Each slice: 4 lines x 32 columns x 4 bytes = 512 bytes.
    //
    // Buffer pointer formula (OP_K=64):
    //   b_curr_ad * E_PER_BANK              -> N column offset (0, 4, 8, ...,
    //   (b_curr_ld / 32) * 512              -> sub-bank offset (0 or 512)
    //   (b_curr_ld % 32 >= 16) ? 64 : 0     -> K chunk within sub-bank
    // =========================================================================
    fp8 *b_buf_base = B_shared;
    b_buf_base += b_curr_ad * E_PER_BANK;
    b_buf_base += (b_curr_ld / 32) * NB_BANKS * E_PER_BANK * 4;
    b_buf_base += (b_curr_ld % 32 >= 16) ? 16 * E_PER_BANK : 0;

    // A tile producer registers: A_LOADS 16-byte loads.
    // Each load covers K=16 for one thread; 8 threads per row cover K=128.
    // For WARPTILE_K = 64*OPS, each thread needs A_LOADS = OPS/2 loads.
    fp8_4 a_reg[A_LOADS][E_PER_THREAD / 4];
    fp8_4 swap_reg[E_PER_THREAD / 4];

    // B tile producer registers (B_LOADS loads x 16 bytes each).
    fp8_4 b_reg[B_LOADS][E_PER_THREAD / 4];

    // Consumer-side buffer pointers.
    // Logically, A has four 512 byte LDS sub-banks and B has four 1024
    // sub-banks ones. For one SMFMAC call:
    // A: even lanes read lines 0-1, odd lanes read lines 2-3 of one 512 byte
    // sub-bank.
    // B: lanes 0-31 read slice 0, lanes 32-63 slice 1 of one 1024 byte
    // sub-bank.
    fp8 *a_cons_base = A_shared;
    a_cons_base += (lane_id / 2) * E_PER_BANK;
    a_cons_base += (lane_id % 2) * E_PER_BANK * 2 * NB_BANKS;

    fp8 *b_cons_base = B_shared;
    b_cons_base += (lane_id % 32) * E_PER_BANK;
    b_cons_base += (lane_id / 32) * E_PER_BANK * 4 * NB_BANKS;

    // Consumer registers for SMFMAC.
    fp8x8 a_cons[OPS];  // 4 x 8 = 32 bytes for A.
    fp8x16 b_cons[OPS]; // 4 x 16 = 64 bytes for B.

    const int k_blocks = K_outer;

    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      const fp8 *a_src_k = a_src + k_block * WARPTILE_K;
      const fp8 *b_src_k = b_src + k_block * WARPTILE_K;
      fp8_4 *A_buffer = reinterpret_cast<fp8_4 *>(a_buf_base);

      // Load A tile.
      // Each thread loads A_LOADS 16-byte chunks at 128-byte intervals.
      // With 8 threads per row, each 128-byte offset covers the next K=128 slice.
      // OPS=2 (WARPTILE_K=128): 1 load at offset 0
      // OPS=4 (WARPTILE_K=256): 2 loads at offsets 0, 128
      // OPS=8 (WARPTILE_K=512): 4 loads at offsets 0, 128, 256, 384
      if constexpr (OPS == 2) {
        asm volatile("global_load_dwordx4 %0, %1, off offset:0"
                     : "=v"(a_reg[0])
                     : "v"(a_src_k));
      } else if constexpr (OPS == 4) {
        asm volatile("global_load_dwordx4 %0, %2, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %2, off offset:128\n\t"
                     : "=v"(a_reg[0]), "=v"(a_reg[1])
                     : "v"(a_src_k));
      } else if constexpr (OPS == 8) {
        asm volatile("global_load_dwordx4 %0, %4, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %4, off offset:128\n\t"
                     "global_load_dwordx4 %2, %4, off offset:256\n\t"
                     "global_load_dwordx4 %3, %4, off offset:384\n\t"
                     : "=v"(a_reg[0]), "=v"(a_reg[1]),
                       "=v"(a_reg[2]), "=v"(a_reg[3])
                     : "v"(a_src_k));
      }

      // Swizzle and store A tile. Each load covers one 8x128 sub-tile (1024 bytes).
      // HALF_TILE_DWORDS = 64*8/2 = 256 (in fp8_4 units = 1024 bytes).
      // This is the stride between consecutive loads in the LDS layout.
      constexpr int HALF_TILE_DWORDS = OP_K * OP_M / 2;

      // Helper macro for swizzle+store of one 16-byte register.
      // The swizzle rearranges the data for efficient SMFMAC consumption.
#define SWIZZLE_AND_STORE_A(load_idx, vmcnt_val, bank_offset)                 \
      asm volatile("s_waitcnt vmcnt(" #vmcnt_val ")\n\t"                       \
                   "v_pack_b32_f16 %0, %4, %5, op_sel:[0,0]\n\t"               \
                   "v_pack_b32_f16 %1, %4, %5, op_sel:[1,1]\n\t"               \
                   "v_pack_b32_f16 %2, %6, %7, op_sel:[0,0]\n\t"               \
                   "v_pack_b32_f16 %3, %6, %7, op_sel:[1,1]\n\t"               \
                   "v_swap_b32 %0, %4\n\t"                                     \
                   "v_swap_b32 %1, %6\n\t"                                     \
                   "v_swap_b32 %2, %5\n\t"                                     \
                   "v_swap_b32 %3, %7"                                         \
                   : "=v"(swap_reg[0]), "=v"(swap_reg[1]), "=v"(swap_reg[2]),  \
                     "=v"(swap_reg[3])                                         \
                   : "v"(a_reg[load_idx][0]), "v"(a_reg[load_idx][1]),         \
                     "v"(a_reg[load_idx][2]), "v"(a_reg[load_idx][3]));        \
      _Pragma("unroll")                                                        \
      for (int line = 0; line < 4; ++line) {                                   \
        A_buffer[line * NB_BANKS + (bank_offset)] = a_reg[load_idx][line];     \
      }

      // Process each A load. vmcnt values count down from total A loads.
      // OPS=2: 1 load. vmcnt(0) for load 0.
      // OPS=4: 2 loads. vmcnt(1,0) for loads 0,1.
      // OPS=8: 4 loads. vmcnt(3,2,1,0) for loads 0-3.
      if constexpr (OPS == 2) {
        SWIZZLE_AND_STORE_A(0, 0, 0);
      } else if constexpr (OPS == 4) {
        SWIZZLE_AND_STORE_A(0, 1, 0);
        SWIZZLE_AND_STORE_A(1, 0, HALF_TILE_DWORDS);
      } else if constexpr (OPS == 8) {
        SWIZZLE_AND_STORE_A(0, 3, 0);
        SWIZZLE_AND_STORE_A(1, 2, HALF_TILE_DWORDS);
        SWIZZLE_AND_STORE_A(2, 1, 2 * HALF_TILE_DWORDS);
        SWIZZLE_AND_STORE_A(3, 0, 3 * HALF_TILE_DWORDS);
      }
#undef SWIZZLE_AND_STORE_A

      // Load B tile (B_LOADS = OPS loads for OPS sub-tiles).
      // Each sub-tile is 16x64 bytes (1024 bytes). Offsets advance by 64 bytes
      // through K positions: 0-63, 64-127, etc.
      if constexpr (OPS == 2) {
        asm volatile("global_load_dwordx4 %0, %2, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %2, off offset:64 \n\t"
                     : "=v"(b_reg[0]), "=v"(b_reg[1])
                     : "v"(b_src_k));
      } else if constexpr (OPS == 4) {
        asm volatile("global_load_dwordx4 %0, %4, off offset:0   \n\t"
                     "global_load_dwordx4 %1, %4, off offset:64  \n\t"
                     "global_load_dwordx4 %2, %4, off offset:128 \n\t"
                     "global_load_dwordx4 %3, %4, off offset:192 \n\t"
                     : "=v"(b_reg[0]), "=v"(b_reg[1]), "=v"(b_reg[2]),
                       "=v"(b_reg[3])
                     : "v"(b_src_k));
      } else if constexpr (OPS == 8) {
        asm volatile("global_load_dwordx4 %0, %8, off offset:0   \n\t"
                     "global_load_dwordx4 %1, %8, off offset:64  \n\t"
                     "global_load_dwordx4 %2, %8, off offset:128 \n\t"
                     "global_load_dwordx4 %3, %8, off offset:192 \n\t"
                     "global_load_dwordx4 %4, %8, off offset:256 \n\t"
                     "global_load_dwordx4 %5, %8, off offset:320 \n\t"
                     "global_load_dwordx4 %6, %8, off offset:384 \n\t"
                     "global_load_dwordx4 %7, %8, off offset:448 \n\t"
                     : "=v"(b_reg[0]), "=v"(b_reg[1]), "=v"(b_reg[2]),
                       "=v"(b_reg[3]), "=v"(b_reg[4]), "=v"(b_reg[5]),
                       "=v"(b_reg[6]), "=v"(b_reg[7])
                     : "v"(b_src_k));
      }

      // Store B tile to LDS (dense, no swizzle).
      fp8 *b_buf = b_buf_base;
      asm volatile("s_waitcnt vmcnt(0)");
#pragma unroll
      for (int load = 0; load < OPS; ++load) {
        const fp8 *b_bytes = reinterpret_cast<const fp8 *>(b_reg[load]);
#pragma unroll
        for (int slice = 0; slice < E_PER_THREAD / E_PER_BANK; ++slice) {
#pragma unroll
          for (int elem = 0; elem < E_PER_BANK; ++elem) {
            b_buf[slice * (NB_BANKS * E_PER_BANK) + elem] =
                b_bytes[E_PER_BANK * slice + elem];
          }
        }
        b_buf += WARP_SIZE * E_PER_THREAD;
      }
      __syncthreads();

      // SMFMAC consumption: OPS ops per K block.
      // Each op consumes 8x64 from A and 16x64 from B.
#pragma unroll
      for (int op = 0; op < OPS; ++op) {
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

        // Per-op offset: op * OP_N * OP_K = op * 1024 bytes.
        fp8 *b_op = b_cons_base + op * OP_N * OP_K;
#pragma unroll
        for (int slice = 0; slice < 4; ++slice) {
#pragma unroll
          for (int i = 0; i < E_PER_BANK; ++i) {
            b_cons[op][i + slice * E_PER_BANK] =
                b_op[i + slice * NB_BANKS * E_PER_BANK];
          }
        }
      }

#pragma unroll
      for (int op = 0; op < OPS; ++op) {
        acc = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(
            reinterpret_cast<fp8_4x2>(a_cons[op]),
            reinterpret_cast<fp8_4x4>(b_cons[op]), acc, sparsity_indices,
            7, // cbsz
            1  // abid
        );
      }
    }

    // Post-processing: Fuse partial results, shuffle, convert to fp16, output.
    acc[0] = acc[0] + acc[1];
    acc[1] = acc[2] + acc[3];

    // Compute output N column and M row.
    const int out_n = 2 * ((lane_id % 16) / 2);
    const int out_m = (lane_id / 16) * 2 + (lane_id % 2);

    // Shuffle: Exchange between even/odd lane pairs. Even lanes keep d0, odd
    // lanes keep d1. Then swap to get 2x2 tile per pair.
    const int id_to_swap = 1 - (lane_id % 2);
    const int src_lane = lane_id + 1 - 2 * (lane_id % 2);
    acc[id_to_swap] = __shfl(acc[id_to_swap], src_lane);

    // Each lane writes 2 elements at C[out_m, out_n] and C[out_m, out_n+1].
    __half *c_out = C + out_m * N + out_n;
    *reinterpret_cast<__half2 *>(c_out) = __floats2half2_rn(acc[0], acc[1]);
  }
};

// ============================================================================
// PaddingFP8Kernel: Dense MFMA with zero-padded rows
// ============================================================================
// This kernel pads A from OP_M rows (e.g., 8) to MFMA_M rows (16) with zeros,
// then uses V_MFMA_F32_16x16x32_FP8_FP8. This wastes 50% compute on padded rows
// but avoids the complex swizzle and SMFMAC sparsity indexing.
//
// Key differences from SparseTrickFP8Kernel:
// - Instruction: __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8 (dense)
// - OP_K = 32 (fixed for 16x16x32 MFMA), not 64
// - OPS = WARPTILE_K / 32 (8 for WARPTILE_K=256, vs 4 for sparse trick)
// - No A swizzle (simple row-major LDS layout)
// - No post-MFMA fusing (dense MFMA produces complete results)
// - Only lanes 0-31 write valid output (lanes 32-63 compute garbage rows 8-15)
// ============================================================================

template <int OP_M, int OP_N, int OPS>
struct PaddingFP8Kernel : SkinnyGemmKernel {
  static_assert(OP_M == 8, "PaddingFP8Kernel only supports OP_M=8");
  static_assert(OP_N == 16, "PaddingFP8Kernel only supports OP_N=16");
  static_assert(OPS >= 4, "OPS=2 not supported: thread layout (8 threads/row × 16 bytes) "
                          "exceeds WARPTILE_K=64, requiring complex partial-thread handling");

  // Dense MFMA parameters
  static constexpr int OP_K = 32;       // Fixed for MFMA 16x16x32
  static constexpr int MFMA_M = 16;     // Padded row count
  static constexpr int WARPTILE_K = OP_K * OPS;

  // Load counts depend on OPS (OPS=8: 2 A loads, 4 B loads; OPS=4: 1 A load, 2 B loads)
  static constexpr int A_LOADS = OPS / 4;  // 2 for OPS=8, 1 for OPS=4
  static constexpr int B_LOADS = OPS / 2;  // 4 for OPS=8, 2 for OPS=4

  PaddingFP8Kernel() {
    name = __FUNCTION__;
    A_type = Type::FP8;
    B_type = Type::FP8;
    C_type = Type::FP16;
    M_tile = OP_M;
    N_tile = OP_N;
    K_tile = WARPTILE_K;
    num_threads_func = [](const ProblemProperties &) { return WARP_SIZE; };
    gemm_func = run;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    // LDS layout: Row-major for simplicity (no swizzle)
    // A: MFMA_M rows × WARPTILE_K cols (padded, rows 8-15 are zeros)
    // B: OP_N rows × WARPTILE_K cols
    constexpr int A_TILE_BYTES = MFMA_M * WARPTILE_K;
    constexpr int B_TILE_BYTES = OP_N * WARPTILE_K;

    const int lane_id = threadIdx.x & (warpSize - 1);
    const int m_outer = blockIdx.x;
    const int n_outer = blockIdx.y;
    const int global_m_base = m_outer * OP_M;
    const int global_n_base = n_outer * OP_N;
    if (global_m_base >= M || global_n_base >= N) {
      return;
    }

    const fp8 *A = static_cast<const fp8 *>(A_data) + global_m_base * K;
    const fp8 *B = static_cast<const fp8 *>(B_data) + global_n_base * K;
    __half *C =
        static_cast<__half *>(C_data) + global_m_base * N + global_n_base;

    __shared__ fp8 A_shared[A_TILE_BYTES];
    __shared__ fp8 B_shared[B_TILE_BYTES];

    // Zero-initialize A's padded region (rows 8-15) once at start.
    // Each thread zeros 4 bytes at a time for efficiency.
    // Padded region: MFMA_M * WARPTILE_K / 2 bytes = rows 8-15
    constexpr int PADDED_START = OP_M * WARPTILE_K;
    constexpr int PADDED_BYTES = (MFMA_M - OP_M) * WARPTILE_K;
    for (int i = lane_id * 4; i < PADDED_BYTES; i += WARP_SIZE * 4) {
      *reinterpret_cast<int *>(A_shared + PADDED_START + i) = 0;
    }
    __syncthreads();

    const int K_outer = ceil_div(K, WARPTILE_K);

    // Accumulator: 4 f32 values per lane (MFMA output layout)
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    // =========================================================================
    // Producer layout for loading A (OP_M=8 rows, WARPTILE_K columns)
    // Thread layout: 64 threads cover 8 rows × WARPTILE_K columns
    // Each thread loads 16 bytes (128 bits) at a time.
    //
    // For WARPTILE_K=256: 8 rows × 256 cols = 2048 bytes
    //   64 threads × 16 bytes = 1024 bytes per load
    //   Need 2 loads to cover full tile
    //
    // Thread assignment:
    //   - a_row = lane_id / 8 (8 threads per row, 0..7)
    //   - a_k_chunk = lane_id % 8 (K offset within row: 0,16,32,...,112)
    // =========================================================================
    constexpr int A_THREADS_PER_ROW = 8;
    constexpr int A_BYTES_PER_THREAD = 16;
    const int a_row = lane_id / A_THREADS_PER_ROW;
    const int a_k_chunk = lane_id % A_THREADS_PER_ROW;
    const int a_k_offset = a_k_chunk * A_BYTES_PER_THREAD;

    // Producer A pointers: load from row a_row, K offset a_k_offset
    // Two loads at offset 0 and 128 cover WARPTILE_K=256
    const fp8 *a_src = A + a_row * K + a_k_offset;

    // LDS store pointer for A (row-major, no swizzle)
    fp8 *a_dst = A_shared + a_row * WARPTILE_K + a_k_offset;

    // =========================================================================
    // Producer layout for loading B (OP_N=16 rows, WARPTILE_K columns)
    // 64 threads × 16 bytes = 1024 bytes per load
    // For WARPTILE_K=256: 16 rows × 256 cols = 4096 bytes → 4 loads
    //
    // Thread assignment:
    //   - b_row = lane_id / 4 (4 threads per row, 0..15)
    //   - b_k_chunk = lane_id % 4 (K offset: 0,16,32,48)
    // =========================================================================
    constexpr int B_THREADS_PER_ROW = 4;
    constexpr int B_BYTES_PER_THREAD = 16;
    const int b_row = lane_id / B_THREADS_PER_ROW;
    const int b_k_chunk = lane_id % B_THREADS_PER_ROW;
    const int b_k_offset = b_k_chunk * B_BYTES_PER_THREAD;

    const fp8 *b_src = B + b_row * K + b_k_offset;
    fp8 *b_dst = B_shared + b_row * WARPTILE_K + b_k_offset;

    // =========================================================================
    // Consumer layout for MFMA 16x16x32
    // From ISA: A[i,k] in item (k % 8) of lane (i + 16 * (k / 8))
    //   - Lanes 0-15: rows 0-15, k=0..7
    //   - Lanes 16-31: rows 0-15, k=8..15
    //   - Lanes 32-47: rows 0-15, k=16..23
    //   - Lanes 48-63: rows 0-15, k=24..31
    //
    // Each lane needs: A[lane_id % 16, (lane_id / 16) * 8 .. +7]
    // =========================================================================
    const int cons_row = lane_id % MFMA_M;      // 0-15
    const int cons_k_group = lane_id / MFMA_M;  // 0-3 (which 8-element K group)

    // Consumer A base: row cons_row, K group offset cons_k_group * 8
    // Per-op: add op * OP_K to get the K offset for that op

    // Consumer B layout is similar: B[j,k] where j=0..15, k=0..31

    // Temporary registers for loading
    fp8_4 a_reg[A_LOADS][A_BYTES_PER_THREAD / 4];
    fp8_4 b_reg[B_LOADS][B_BYTES_PER_THREAD / 4];

    for (int k_block = 0; k_block < K_outer; ++k_block) {
      const fp8 *a_src_k = a_src + k_block * WARPTILE_K;
      const fp8 *b_src_k = b_src + k_block * WARPTILE_K;

      // Load A tile (A_LOADS loads at 128-byte intervals)
      if constexpr (OPS == 4) {
        asm volatile("global_load_dwordx4 %0, %1, off offset:0"
                     : "=v"(a_reg[0])
                     : "v"(a_src_k));
      } else if constexpr (OPS == 8) {
        asm volatile("global_load_dwordx4 %0, %2, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %2, off offset:128\n\t"
                     : "=v"(a_reg[0]), "=v"(a_reg[1])
                     : "v"(a_src_k));
      }

      // Load B tile (B_LOADS loads at 64-byte intervals)
      if constexpr (OPS == 4) {
        asm volatile("global_load_dwordx4 %0, %2, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %2, off offset:64 \n\t"
                     : "=v"(b_reg[0]), "=v"(b_reg[1])
                     : "v"(b_src_k));
      } else if constexpr (OPS == 8) {
        asm volatile("global_load_dwordx4 %0, %4, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %4, off offset:64 \n\t"
                     "global_load_dwordx4 %2, %4, off offset:128\n\t"
                     "global_load_dwordx4 %3, %4, off offset:192\n\t"
                     : "=v"(b_reg[0]), "=v"(b_reg[1]), "=v"(b_reg[2]),
                       "=v"(b_reg[3])
                     : "v"(b_src_k));
      }

      asm volatile("s_waitcnt vmcnt(0)");

      // Store A to LDS (row-major, no swizzle)
      // Only store valid rows (0-7), padded rows (8-15) stay zero
#pragma unroll
      for (int load = 0; load < A_LOADS; ++load) {
        fp8 *a_store = a_dst + load * 128;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          reinterpret_cast<fp8_4 *>(a_store)[d] = a_reg[load][d];
        }
      }

      // Store B to LDS (row-major)
#pragma unroll
      for (int load = 0; load < B_LOADS; ++load) {
        fp8 *b_store = b_dst + load * 64;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          reinterpret_cast<fp8_4 *>(b_store)[d] = b_reg[load][d];
        }
      }

      __syncthreads();

      // MFMA consumption: OPS ops per K block
      // Each op: 16x16x32 matrix multiply
#pragma unroll
      for (int op = 0; op < OPS; ++op) {
        // Load A from LDS for this op
        // Lane needs: A_shared[cons_row * WARPTILE_K + op * OP_K + cons_k_group * 8 + i]
        // for i in 0..7
        fp8 *a_op = A_shared + cons_row * WARPTILE_K + op * OP_K +
                    cons_k_group * 8;
        fp8x8 a_cons;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          a_cons[i] = a_op[i];
        }

        // Load B from LDS for this op
        // B layout for MFMA 16x16x32: similar pattern
        // B[j,k] in item (k % 8) of lane (j + 16 * (k / 8))
        // Each lane needs: B[lane_id % 16, (lane_id / 16) * 8 .. +7]
        fp8 *b_op = B_shared + cons_row * WARPTILE_K + op * OP_K +
                    cons_k_group * 8;
        fp8x8 b_cons;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          b_cons[i] = b_op[i];
        }

        // Call dense MFMA
        acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            reinterpret_cast<long>(a_cons), reinterpret_cast<long>(b_cons), acc,
            0, 0, 0);
      }

      __syncthreads();
    }

    // Output: Only lanes 0-31 write valid results (rows 0-7)
    // MFMA output layout: ci = 4 * (lane_id / 16) + gpr, cj = lane_id % 16
    // For lanes 0-31:
    //   - Lanes 0-15: rows 0-3 (gpr 0-3)
    //   - Lanes 16-31: rows 4-7 (gpr 0-3)
    // Lanes 32-63 would write rows 8-15 (garbage, skip them)
    if (lane_id < 32) {
      const int out_row_base = (lane_id / 16) * 4;  // 0 or 4
      const int out_col = lane_id % 16;
      __half *c_ptr = C + out_row_base * N + out_col;

#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (out_row_base + r < OP_M) {
          c_ptr[r * N] = __float2half(acc[r]);
        }
      }
    }
  }
};

// ============================================================================
// PaddingFP8KernelWithSwizzle: Dense MFMA with scatter-gather LDS layout
// ============================================================================
// This kernel combines PaddingFP8Kernel's zero-padding approach with
// SparseTrickFP8Kernel's scatter-gather LDS layout to reduce bank conflicts.
//
// Key insight: PaddingFP8Kernel has 16-way bank conflicts because all 16 rows
// hit the same banks when reading contiguous data. By using scatter-gather:
// - Instead of reading 8 contiguous bytes, read 2 × 4-byte chunks 128 bytes apart
// - Interleave row + k_group to spread 64 lanes across 32 bank slots
// - Achieve 2-way conflicts (64 lanes / 32 banks) instead of 16-way
//
// LDS Layout (per op, 512 bytes for A, 512 bytes for B):
// - Slots 0-15 (K-groups 0,2): bytes [0-63] first 4B, [64-127] second 4B
// - Slots 16-31 (K-groups 1,3): bytes [128-191] first 4B, [192-255] second 4B
// - K-group pairs 2-3 at offset 256-511
// ============================================================================

template <int OP_M, int OP_N, int OPS>
struct PaddingFP8KernelWithSwizzle : SkinnyGemmKernel {
  static_assert(OP_M == 8, "PaddingFP8KernelWithSwizzle only supports OP_M=8");
  static_assert(OP_N == 16, "PaddingFP8KernelWithSwizzle only supports OP_N=16");
  static_assert(OPS >= 4, "OPS=2 not supported: thread layout (8 threads/row × 16 bytes) "
                          "exceeds WARPTILE_K=64, requiring complex partial-thread handling");

  // Dense MFMA parameters (same as PaddingFP8Kernel)
  static constexpr int OP_K = 32;       // Fixed for MFMA 16x16x32
  static constexpr int MFMA_M = 16;     // Padded row count
  static constexpr int WARPTILE_K = OP_K * OPS;

  // Load counts depend on OPS (OPS=8: 2 A loads, 4 B loads; OPS=4: 1 A load, 2 B loads)
  static constexpr int A_LOADS = OPS / 4;  // 2 for OPS=8, 1 for OPS=4
  static constexpr int B_LOADS = OPS / 2;  // 4 for OPS=8, 2 for OPS=4

  // Scatter-gather constants
  static constexpr int E_PER_BANK = 4;  // 4 bytes per LDS bank slot

  PaddingFP8KernelWithSwizzle() {
    name = __FUNCTION__;
    A_type = Type::FP8;
    B_type = Type::FP8;
    C_type = Type::FP16;
    M_tile = OP_M;
    N_tile = OP_N;
    K_tile = WARPTILE_K;
    num_threads_func = [](const ProblemProperties &) { return WARP_SIZE; };
    gemm_func = run;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    // LDS layout: Scatter-gather for both A and B
    // A: 512 bytes per op × OPS ops = 512 * OPS bytes (MFMA_M=16 padded rows)
    // B: 512 bytes per op × OPS ops = 512 * OPS bytes (OP_N=16 rows)
    //
    // Per-op layout (512 bytes):
    //   Offset 0-63:    Slots 0-15,  K-group 0/2 first 4 bytes
    //   Offset 64-127:  Slots 0-15,  K-group 0/2 second 4 bytes
    //   Offset 128-191: Slots 16-31, K-group 1/3 first 4 bytes
    //   Offset 192-255: Slots 16-31, K-group 1/3 second 4 bytes
    //   Offset 256-511: Same pattern for K-group pairs 2-3
    constexpr int BYTES_PER_OP = 512;  // 16 rows × 32 K = 512 bytes
    constexpr int A_TILE_BYTES = BYTES_PER_OP * OPS;
    constexpr int B_TILE_BYTES = BYTES_PER_OP * OPS;

    const int lane_id = threadIdx.x & (warpSize - 1);
    const int m_outer = blockIdx.x;
    const int n_outer = blockIdx.y;
    const int global_m_base = m_outer * OP_M;
    const int global_n_base = n_outer * OP_N;
    if (global_m_base >= M || global_n_base >= N) {
      return;
    }

    const fp8 *A = static_cast<const fp8 *>(A_data) + global_m_base * K;
    const fp8 *B = static_cast<const fp8 *>(B_data) + global_n_base * K;
    __half *C =
        static_cast<__half *>(C_data) + global_m_base * N + global_n_base;

    __shared__ fp8 A_shared[A_TILE_BYTES];
    __shared__ fp8 B_shared[B_TILE_BYTES];

    // Zero-initialize A's padded slots (rows 8-15) once at start.
    // In scatter-gather layout, padded rows occupy specific slot positions.
    // Rows 8-15 use slots 8-15 (for K-groups 0,2) and slots 24-31 (for K-groups 1,3).
    //
    // For simplicity, zero the entire A_shared buffer since only valid rows
    // will be overwritten by producers. This is safe and simple.
    for (int i = lane_id * 4; i < A_TILE_BYTES; i += WARP_SIZE * 4) {
      *reinterpret_cast<int *>(A_shared + i) = 0;
    }
    __syncthreads();

    const int K_outer = ceil_div(K, WARPTILE_K);

    // Accumulator: 4 f32 values per lane (MFMA output layout)
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};

    // =========================================================================
    // Producer layout for A (OP_M=8 valid rows, WARPTILE_K columns)
    //
    // Thread assignment: 64 threads, each loads 16 bytes per iteration
    //   a_row = lane_id / 8 (0-7)
    //   a_k_chunk = lane_id % 8 (0-7, each covering 16 bytes of K)
    //
    // Each thread's 16 bytes spans 2 K-groups (16 bytes = 2 × 8 bytes per K-group)
    // =========================================================================
    constexpr int A_THREADS_PER_ROW = 8;
    constexpr int A_BYTES_PER_THREAD = 16;
    const int a_row = lane_id / A_THREADS_PER_ROW;
    const int a_k_chunk = lane_id % A_THREADS_PER_ROW;

    // Global memory source pointer for A
    const fp8 *a_src = A + a_row * K + a_k_chunk * A_BYTES_PER_THREAD;

    // =========================================================================
    // Producer layout for B (OP_N=16 rows, WARPTILE_K columns)
    //
    // Thread assignment:
    //   b_row = lane_id / 4 (0-15)
    //   b_k_chunk = lane_id % 4 (0-3, each covering 16 bytes of K)
    // =========================================================================
    constexpr int B_THREADS_PER_ROW = 4;
    constexpr int B_BYTES_PER_THREAD = 16;
    const int b_row = lane_id / B_THREADS_PER_ROW;
    const int b_k_chunk = lane_id % B_THREADS_PER_ROW;

    const fp8 *b_src = B + b_row * K + b_k_chunk * B_BYTES_PER_THREAD;

    // =========================================================================
    // Consumer layout for MFMA 16x16x32 with scatter-gather
    //
    // MFMA lane mapping: A[i,k] in item (k % 8) of lane (i + 16 * (k / 8))
    //   cons_row = lane_id % 16 (0-15)
    //   cons_k_group = lane_id / 16 (0-3, which 8-element K group)
    //
    // Scatter-gather slot assignment (for 2-way bank conflicts):
    //   slot = cons_row + (cons_k_group % 2) * 16  (0-31)
    //
    // This spreads 64 lanes across 32 bank slots:
    //   K-group 0 (lanes 0-15):  slots 0-15  → banks 0-15
    //   K-group 1 (lanes 16-31): slots 16-31 → banks 16-31
    //   K-group 2 (lanes 32-47): slots 0-15  → banks 0-15 (different op region)
    //   K-group 3 (lanes 48-63): slots 16-31 → banks 16-31 (different op region)
    // =========================================================================
    const int cons_row = lane_id % MFMA_M;       // 0-15
    const int cons_k_group = lane_id / MFMA_M;   // 0-3

    // Region-based addressing for correct data placement
    // Layout per op (512 bytes):
    //   K-group 0: k_pair_offset=0,   k_odd=0 -> offset = 0 + 0 + row*4
    //   K-group 1: k_pair_offset=0,   k_odd=1 -> offset = 0 + 128 + row*4
    //   K-group 2: k_pair_offset=256, k_odd=0 -> offset = 256 + 0 + row*4
    //   K-group 3: k_pair_offset=256, k_odd=1 -> offset = 256 + 128 + row*4
    const int k_odd = cons_k_group % 2;          // 0 or 1
    const int k_pair_offset = (cons_k_group / 2) * 256;  // 0 for K-groups 0-1, 256 for K-groups 2-3

    // Temporary registers for loading
    fp8_4 a_reg[A_LOADS][4];
    fp8_4 b_reg[B_LOADS][4];

    for (int k_block = 0; k_block < K_outer; ++k_block) {
      const fp8 *a_src_k = a_src + k_block * WARPTILE_K;
      const fp8 *b_src_k = b_src + k_block * WARPTILE_K;

      // Load A tile (A_LOADS loads at 128-byte intervals)
      if constexpr (OPS == 4) {
        asm volatile("global_load_dwordx4 %0, %1, off offset:0"
                     : "=v"(a_reg[0])
                     : "v"(a_src_k));
      } else if constexpr (OPS == 8) {
        asm volatile("global_load_dwordx4 %0, %2, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %2, off offset:128\n\t"
                     : "=v"(a_reg[0]), "=v"(a_reg[1])
                     : "v"(a_src_k));
      }

      // Load B tile (B_LOADS loads at 64-byte intervals)
      if constexpr (OPS == 4) {
        asm volatile("global_load_dwordx4 %0, %2, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %2, off offset:64 \n\t"
                     : "=v"(b_reg[0]), "=v"(b_reg[1])
                     : "v"(b_src_k));
      } else if constexpr (OPS == 8) {
        asm volatile("global_load_dwordx4 %0, %4, off offset:0  \n\t"
                     "global_load_dwordx4 %1, %4, off offset:64 \n\t"
                     "global_load_dwordx4 %2, %4, off offset:128\n\t"
                     "global_load_dwordx4 %3, %4, off offset:192\n\t"
                     : "=v"(b_reg[0]), "=v"(b_reg[1]), "=v"(b_reg[2]),
                       "=v"(b_reg[3])
                     : "v"(b_src_k));
      }

      asm volatile("s_waitcnt vmcnt(0)");

      // =====================================================================
      // Store A to LDS with scatter-gather layout
      //
      // Each thread's 16 bytes (from a_k_chunk position) spans 2 K-groups.
      // K position range: [a_k_chunk * 16 .. a_k_chunk * 16 + 15]
      //
      // For OPS=8, WARPTILE_K=256, OP_K=32:
      //   a_k_chunk=0: K[0..15]   -> K-groups 0-1 within op 0
      //   a_k_chunk=1: K[16..31]  -> K-groups 2-3 within op 0
      //   a_k_chunk=2: K[32..47]  -> K-groups 0-1 within op 1
      //   ...
      //
      // Within each 16-byte load:
      //   Bytes 0-7:  First K-group (8 elements)
      //   Bytes 8-15: Second K-group (8 elements)
      //
      // Scatter pattern for each 8-byte group:
      //   4 bytes to base + row * E_PER_BANK
      //   4 bytes to base + row * E_PER_BANK + 64
      // =====================================================================

      // Process A_LOADS loads (at 128-byte intervals)
#pragma unroll
      for (int load = 0; load < A_LOADS; ++load) {
        // K offset for this load within WARPTILE_K
        int k_offset = a_k_chunk * A_BYTES_PER_THREAD + load * 128;

        // Which op does this K range belong to?
        // op = k_offset / OP_K
        // k_within_op = k_offset % OP_K
        int op = k_offset / OP_K;
        int k_within_op = k_offset % OP_K;

        // Which K-group within the op?
        int k_group_base = k_within_op / 8;  // 0-3

        // Region-based addressing for the two K-groups in this 16-byte load
        // K-group 0,2 -> k_odd=0 (offset 0 within 128-byte sub-region)
        // K-group 1,3 -> k_odd=1 (offset 128 within 256-byte k_pair region)
        int k_odd0 = k_group_base % 2;
        int k_odd1 = (k_group_base + 1) % 2;

        // K-pair offset within the op's 512 bytes
        int k_pair0 = (k_group_base / 2) * 256;
        int k_pair1 = ((k_group_base + 1) / 2) * 256;

        // Base address in LDS for this op
        fp8 *op_base = A_shared + op * BYTES_PER_OP;

        // Destination pointers for the two K-groups (region-based addressing)
        fp8 *a_dst0 = op_base + k_pair0 + k_odd0 * 128 + a_row * E_PER_BANK;
        fp8 *a_dst1 = op_base + k_pair1 + k_odd1 * 128 + a_row * E_PER_BANK;

        // Get bytes from the loaded register
        const fp8 *a_bytes = reinterpret_cast<const fp8 *>(&a_reg[load][0]);

        // Scatter first 8 bytes (K-group k_group_base)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          a_dst0[i] = a_bytes[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          a_dst0[64 + i] = a_bytes[4 + i];
        }

        // Scatter second 8 bytes (K-group k_group_base + 1)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          a_dst1[i] = a_bytes[8 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          a_dst1[64 + i] = a_bytes[12 + i];
        }
      }

      // =====================================================================
      // Store B to LDS with scatter-gather layout (same pattern as A)
      // =====================================================================
#pragma unroll
      for (int load = 0; load < B_LOADS; ++load) {
        int k_offset = b_k_chunk * B_BYTES_PER_THREAD + load * 64;
        int op = k_offset / OP_K;
        int k_within_op = k_offset % OP_K;
        int k_group_base = k_within_op / 8;

        int k_odd0 = k_group_base % 2;
        int k_odd1 = (k_group_base + 1) % 2;

        int k_pair0 = (k_group_base / 2) * 256;
        int k_pair1 = ((k_group_base + 1) / 2) * 256;

        fp8 *op_base = B_shared + op * BYTES_PER_OP;
        fp8 *b_dst0 = op_base + k_pair0 + k_odd0 * 128 + b_row * E_PER_BANK;
        fp8 *b_dst1 = op_base + k_pair1 + k_odd1 * 128 + b_row * E_PER_BANK;

        const fp8 *b_bytes = reinterpret_cast<const fp8 *>(&b_reg[load][0]);

#pragma unroll
        for (int i = 0; i < 4; ++i) {
          b_dst0[i] = b_bytes[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          b_dst0[64 + i] = b_bytes[4 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          b_dst1[i] = b_bytes[8 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          b_dst1[64 + i] = b_bytes[12 + i];
        }
      }

      __syncthreads();

      // =====================================================================
      // MFMA consumption: OPS ops per K block
      // Each op: 16x16x32 matrix multiply
      //
      // Consumer scatter-gather read: 8 bytes as 2 × 4-byte gathers
      // =====================================================================
#pragma unroll
      for (int op = 0; op < OPS; ++op) {
        // A consumer base for this op (region-based addressing)
        fp8 *a_op_base = A_shared + op * BYTES_PER_OP + k_pair_offset + k_odd * 128 + cons_row * E_PER_BANK;

        // Scatter-gather read: 8 bytes from 2 locations 64 bytes apart
        fp8x8 a_cons;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          a_cons[i] = a_op_base[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          a_cons[4 + i] = a_op_base[64 + i];
        }

        // B consumer base for this op (same layout as A, region-based addressing)
        fp8 *b_op_base = B_shared + op * BYTES_PER_OP + k_pair_offset + k_odd * 128 + cons_row * E_PER_BANK;

        fp8x8 b_cons;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          b_cons[i] = b_op_base[i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          b_cons[4 + i] = b_op_base[64 + i];
        }

        // Call dense MFMA
        acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            reinterpret_cast<long>(a_cons), reinterpret_cast<long>(b_cons), acc,
            0, 0, 0);
      }

      __syncthreads();
    }

    // Output: Only lanes 0-31 write valid results (rows 0-7)
    // MFMA output layout: ci = 4 * (lane_id / 16) + gpr, cj = lane_id % 16
    if (lane_id < 32) {
      const int out_row_base = (lane_id / 16) * 4;
      const int out_col = lane_id % 16;
      __half *c_ptr = C + out_row_base * N + out_col;

#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (out_row_base + r < OP_M) {
          c_ptr[r * N] = __float2half(acc[r]);
        }
      }
    }
  }
};

// ============================================================================
// Layout Tests (GPU Micro-Kernels)
// ============================================================================
namespace layout_gpu_tests {

// Architecture constants (must match production kernel)
constexpr int E_PER_THREAD = 16; // Bytes per thread per load
constexpr int E_PER_BANK = 4;    // 4 bytes per LDS column

// Static asserts for architectural invariants
static_assert(WARP_SIZE == 64, "Tests assume 64-lane wavefront");
static_assert(NB_BANKS == 32, "Tests assume 32 LDS banks");

// Swizzle output pattern: input K0..K15 -> output positions
// After v_pack + v_swap, byte at input position i ends up at SWIZZLE_PATTERN[i]
// Documented in fullpicture-sparse-trick.md
constexpr uint8_t SWIZZLE_PATTERN[16] = {0,  1,  4,  5, 8, 9, 12, 13,
                                         2,  3,  6,  7, 10, 11, 14, 15};

// Consumer lane patterns after LDS round-trip
// Even lanes receive K positions: 0,1,4,5,8,9,12,13
// Odd lanes receive K positions: 2,3,6,7,10,11,14,15
constexpr uint8_t EVEN_LANE_PATTERN[8] = {0, 1, 4, 5, 8, 9, 12, 13};
constexpr uint8_t ODD_LANE_PATTERN[8] = {2, 3, 6, 7, 10, 11, 14, 15};

// Compile-time validation that patterns are complementary
constexpr bool validate_lane_patterns() {
  bool seen[16] = {};
  for (int i = 0; i < 8; ++i) {
    if (EVEN_LANE_PATTERN[i] >= 16 || ODD_LANE_PATTERN[i] >= 16)
      return false;
    seen[EVEN_LANE_PATTERN[i]] = true;
    seen[ODD_LANE_PATTERN[i]] = true;
  }
  for (int i = 0; i < 16; ++i)
    if (!seen[i])
      return false;
  return true;
}
static_assert(validate_lane_patterns(),
              "Lane patterns must cover all 16 K positions");

// Validate swizzle is a valid permutation
constexpr bool validate_swizzle_pattern() {
  bool seen[16] = {};
  for (int i = 0; i < 16; ++i) {
    if (SWIZZLE_PATTERN[i] >= 16 || seen[SWIZZLE_PATTERN[i]])
      return false;
    seen[SWIZZLE_PATTERN[i]] = true;
  }
  return true;
}
static_assert(validate_swizzle_pattern(),
              "Swizzle must be valid permutation of 0-15");

// ============================================================================
// Test: Swizzle (v_pack + v_swap sequence)
// ============================================================================

// Swizzle test is independent of OPS - tests the v_pack + v_swap sequence.
// Each thread processes 16 bytes using the production swizzle pattern.
//
// The v_swap_b32 instruction swaps both operands. Production code relies on
// reg0 being modified as a side effect (input-only constraint but actually
// read-modify-write). For the test, we use explicit output variables to make
// the data flow clear and avoid undefined behavior per GCC inline asm semantics.
__global__ void test_swizzle_kernel(const fp8 *input, fp8 *output) {
  int lane_id = threadIdx.x;
  if (lane_id >= WARP_SIZE)
    return;

  const fp8 *src = input + lane_id * E_PER_THREAD;
  fp8_4 reg0[4];
  fp8_4 swizzled[4] = {}; // Zero-init; receives swapped values

  asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(reg0) : "v"(src));

  // Swizzle: v_pack interleaves bytes, v_swap places result in swizzled[]
  // v_pack creates: swap_reg[0]={b0,b1,b4,b5}, swap_reg[1]={b2,b3,b6,b7}, etc.
  // v_swap then exchanges swap_reg[i] <-> reg0[j], putting packed data into
  // what were the input registers.
  //
  // We use "+v" (read-write) constraints for the swap destinations so the
  // compiler knows both operands of v_swap are modified.
  fp8_4 swap_reg[4];
  asm volatile(
      "s_waitcnt vmcnt(0)\n\t"
      "v_pack_b32_f16 %0, %8, %9, op_sel:[0,0]\n\t"  // swap_reg[0] = {b0,b1,b4,b5}
      "v_pack_b32_f16 %1, %8, %9, op_sel:[1,1]\n\t"  // swap_reg[1] = {b2,b3,b6,b7}
      "v_pack_b32_f16 %2, %10, %11, op_sel:[0,0]\n\t" // swap_reg[2] = {b8,b9,b12,b13}
      "v_pack_b32_f16 %3, %10, %11, op_sel:[1,1]\n\t" // swap_reg[3] = {b10,b11,b14,b15}
      "v_swap_b32 %4, %0\n\t"  // swizzled[0] <-> swap_reg[0]
      "v_swap_b32 %5, %2\n\t"  // swizzled[1] <-> swap_reg[2]
      "v_swap_b32 %6, %1\n\t"  // swizzled[2] <-> swap_reg[1]
      "v_swap_b32 %7, %3\n\t"  // swizzled[3] <-> swap_reg[3]
      : "=&v"(swap_reg[0]), "=&v"(swap_reg[1]), "=&v"(swap_reg[2]),
        "=&v"(swap_reg[3]), "+v"(swizzled[0]), "+v"(swizzled[1]),
        "+v"(swizzled[2]), "+v"(swizzled[3])
      : "v"(reg0[0]), "v"(reg0[1]), "v"(reg0[2]), "v"(reg0[3]));

  // Store swizzled output
  fp8 *dst = output + lane_id * E_PER_THREAD;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reinterpret_cast<fp8_4 *>(dst)[i] = swizzled[i];
  }
}

void test_swizzle_gpu() {
  constexpr int NUM_THREADS = WARP_SIZE;
  constexpr int TOTAL_BYTES = NUM_THREADS * E_PER_THREAD;

  // Allocate and initialize input: each thread gets bytes 0..15
  std::vector<uint8_t> h_input(TOTAL_BYTES);
  for (int t = 0; t < NUM_THREADS; ++t) {
    for (int b = 0; b < E_PER_THREAD; ++b) {
      h_input[t * E_PER_THREAD + b] = static_cast<uint8_t>(b);
    }
  }

  fp8 *d_input = nullptr;
  fp8 *d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_input, TOTAL_BYTES));
  HIP_CHECK(hipMalloc(&d_output, TOTAL_BYTES));
  HIP_CHECK(
      hipMemcpy(d_input, h_input.data(), TOTAL_BYTES, hipMemcpyHostToDevice));

  test_swizzle_kernel<<<1, NUM_THREADS>>>(d_input, d_output);
  HIP_CHECK(hipGetLastError());

  std::vector<uint8_t> h_output(TOTAL_BYTES);
  HIP_CHECK(
      hipMemcpy(h_output.data(), d_output, TOTAL_BYTES, hipMemcpyDeviceToHost));

  // Verify against SWIZZLE_PATTERN constant
  bool passed = true;
  for (int t = 0; t < NUM_THREADS && passed; ++t) {
    for (int b = 0; b < E_PER_THREAD; ++b) {
      uint8_t expected = SWIZZLE_PATTERN[b];
      uint8_t actual = h_output[t * E_PER_THREAD + b];
      if (actual != expected) {
        fprintf(stderr,
                "test_swizzle_gpu FAILED: thread %d byte %d = %d, expected %d "
                "(input byte at position %d)\n",
                t, b, actual, expected, SWIZZLE_PATTERN[b]);
        passed = false;
        break;
      }
    }
  }

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  if (passed) {
    printf("test_swizzle_gpu: PASSED\n");
  } else {
    abort();
  }
}

// ============================================================================
// Test: LDS Round-Trip (Templated on <OP_M, OP_N, OPS>)
// ============================================================================

// GPU kernel to test the full LDS round-trip for ALL half-tiles and ALL ops.
// Producer: load from global, swizzle, store to LDS.
// Consumer: load from LDS in SMFMAC-ready layout.
//
// Template parameters match production SparseTrickFP8Kernel<OP_M, OP_N, OPS>.
template <int OP_M, int OP_N, int OPS>
__global__ void test_lds_roundtrip_kernel(const fp8 *input, fp8 *output) {
  // Derive constants from template parameters (same as SparseTrickFP8Kernel)
  constexpr int OP_K = 512 / OP_M; // 64 for OP_M=8
  constexpr int WARPTILE_K = OP_K * OPS;
  constexpr int A_TILE_BYTES = OP_M * WARPTILE_K;
  constexpr int HALF_TILE_K = 128; // Each load covers 128 K elements
  constexpr int HALF_TILE_DWORDS = OP_K * OP_M / 2;
  constexpr int NUM_HALF_TILES = WARPTILE_K / HALF_TILE_K;
  constexpr int THREADS_PER_LD = 8;

  // Static asserts for this instantiation
  static_assert(OP_K == 64, "Sparse trick requires OP_K=64 for OP_M=8");
  static_assert(WARPTILE_K == OP_K * OPS, "WARPTILE_K derivation check");
  static_assert(NUM_HALF_TILES == OPS / 2, "Each half-tile covers 2 ops");

  __shared__ fp8 A_shared[A_TILE_BYTES];

  int lane_id = threadIdx.x;
  if (lane_id >= WARP_SIZE)
    return;

  // Producer position (same as production)
  int curr_ld = (lane_id % THREADS_PER_LD) * E_PER_THREAD;
  int curr_ad = lane_id / THREADS_PER_LD;

  // Producer buffer pointer (same as production lines 491-494)
  fp8 *a_buf_base = A_shared;
  a_buf_base += E_PER_BANK * curr_ad;
  a_buf_base += (curr_ld / 64) * NB_BANKS * E_PER_BANK * 4;
  a_buf_base += (curr_ld % 64) * 2;
  fp8_4 *A_buffer = reinterpret_cast<fp8_4 *>(a_buf_base);

  const fp8 *src = input + lane_id * E_PER_THREAD;

  // Load ALL half-tiles based on OPS
  // For OPS=4: 2 half-tiles (K 0-127, 128-255)
  // For OPS=8: 4 half-tiles (K 0-127, 128-255, 256-383, 384-511)
  fp8_4 reg[NUM_HALF_TILES][4];

#pragma unroll
  for (int ht = 0; ht < NUM_HALF_TILES; ++ht) {
    asm volatile("global_load_dwordx4 %0, %1, off"
                 : "=v"(reg[ht])
                 : "v"(src + ht * HALF_TILE_K));
  }

  // Swizzle and store each half-tile
#pragma unroll
  for (int ht = 0; ht < NUM_HALF_TILES; ++ht) {
    fp8_4 swap_reg[4];
    asm volatile("s_waitcnt vmcnt(0)\n\t"
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
                 : "v"(reg[ht][0]), "v"(reg[ht][1]), "v"(reg[ht][2]),
                   "v"(reg[ht][3]));

#pragma unroll
    for (int line = 0; line < 4; ++line) {
      A_buffer[line * NB_BANKS + ht * HALF_TILE_DWORDS] = reg[ht][line];
    }
  }

  __syncthreads();

  // Consumer: read ALL ops
  fp8 *a_cons_base = A_shared;
  a_cons_base += (lane_id / 2) * E_PER_BANK;
  a_cons_base += (lane_id % 2) * E_PER_BANK * 2 * NB_BANKS;

  // Output: 8 bytes per op, OPS ops total
  constexpr int OUTPUT_BYTES_PER_LANE = 8 * OPS;
  fp8 *out = output + lane_id * OUTPUT_BYTES_PER_LANE;

#pragma unroll
  for (int op = 0; op < OPS; ++op) {
    fp8 *a_op = a_cons_base + op * OP_M * OP_K;
    fp8 a_cons[8];

#pragma unroll
    for (int i = 0; i < E_PER_BANK; ++i) {
      a_cons[i + 0 * E_PER_BANK] = a_op[i + 0 * NB_BANKS * E_PER_BANK];
    }
#pragma unroll
    for (int i = 0; i < E_PER_BANK; ++i) {
      a_cons[i + 1 * E_PER_BANK] = a_op[i + 1 * NB_BANKS * E_PER_BANK];
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out[op * 8 + i] = a_cons[i];
    }
  }
}

template <int OP_M, int OP_N, int OPS> void test_lds_roundtrip_gpu() {
  constexpr int OP_K = 512 / OP_M;
  constexpr int WARPTILE_K = OP_K * OPS;
  constexpr int A_TILE_BYTES = OP_M * WARPTILE_K;
  constexpr int NUM_THREADS = WARP_SIZE;
  constexpr int INPUT_BYTES = A_TILE_BYTES;
  constexpr int OUTPUT_BYTES = NUM_THREADS * 8 * OPS;
  constexpr int HALF_TILE_K = 128; // Each load covers 128 K elements
  constexpr int NUM_HALF_TILES = WARPTILE_K / HALF_TILE_K;

  // Generate input with lane-strided layout matching kernel load pattern.
  //
  // The kernel loads: src = input + lane_id * E_PER_THREAD
  // Each lane loads 16 bytes per half-tile, at offsets [0..15] and [128..143]
  // relative to its base address.
  //
  // We encode byte position within each 16-byte chunk (0..15) so verification
  // can check the swizzle pattern is correct, regardless of which producer lane
  // the data came from. After swizzle, consumers should see SWIZZLE_PATTERN.
  constexpr int E_PER_THREAD = 16;
  std::vector<uint8_t> h_input(INPUT_BYTES);
  for (int lane = 0; lane < WARP_SIZE; ++lane) {
    for (int ht = 0; ht < NUM_HALF_TILES; ++ht) {
      for (int b = 0; b < E_PER_THREAD; ++b) {
        int offset = lane * E_PER_THREAD + ht * HALF_TILE_K + b;
        // Encode byte position within the 16-byte chunk (0..15)
        h_input[offset] = static_cast<uint8_t>(b);
      }
    }
  }

  fp8 *d_input = nullptr;
  fp8 *d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_input, INPUT_BYTES));
  HIP_CHECK(hipMalloc(&d_output, OUTPUT_BYTES));
  HIP_CHECK(
      hipMemcpy(d_input, h_input.data(), INPUT_BYTES, hipMemcpyHostToDevice));

  test_lds_roundtrip_kernel<OP_M, OP_N, OPS>
      <<<1, NUM_THREADS>>>(d_input, d_output);
  HIP_CHECK(hipGetLastError());

  std::vector<uint8_t> h_output(OUTPUT_BYTES);
  HIP_CHECK(hipMemcpy(h_output.data(), d_output, OUTPUT_BYTES,
                      hipMemcpyDeviceToHost));

  // Verify consumer output for ALL ops.
  //
  // With the lane-strided input where each 16-byte chunk contains [0,1,2,...,15],
  // we verify that the swizzle+LDS mechanics produce the correct byte selection:
  // - Even lanes receive bytes at positions: 0,1,4,5,8,9,12,13 from a 16-byte chunk
  // - Odd lanes receive bytes at positions: 2,3,6,7,10,11,14,15 from a 16-byte chunk
  //
  // Since input[position] = position, expected output = pattern[i] directly.
  // This validates the swizzle pattern regardless of producer→consumer lane mapping.
  bool passed = true;
  for (int lane = 0; lane < NUM_THREADS && passed; ++lane) {
    bool is_even = (lane % 2 == 0);
    const uint8_t *pattern = is_even ? EVEN_LANE_PATTERN : ODD_LANE_PATTERN;

    for (int op = 0; op < OPS; ++op) {
      for (int i = 0; i < 8; ++i) {
        uint8_t expected = pattern[i]; // Direct: input[pos] = pos
        uint8_t actual = h_output[lane * 8 * OPS + op * 8 + i];
        if (actual != expected) {
          fprintf(stderr,
                  "test_lds_roundtrip_gpu<%d,%d,%d> FAILED: lane %d (%s), "
                  "op %d, byte %d = %d, expected %d (pattern[%d])\n",
                  OP_M, OP_N, OPS, lane, is_even ? "even" : "odd", op, i,
                  actual, expected, i);
          passed = false;
          break;
        }
      }
      if (!passed)
        break;
    }
  }

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));

  if (passed) {
    printf("test_lds_roundtrip_gpu<%d,%d,%d>: PASSED\n", OP_M, OP_N, OPS);
  } else {
    abort();
  }
}

// ============================================================================
// Test: Shuffle Exchange (post-SMFMAC result fusing)
// ============================================================================

// GPU kernel to test the shuffle exchange that fuses results between lane pairs.
// In production, after SMFMAC ops, adjacent lanes exchange partial sums.
//
// CRITICAL FIX: Previous version had identical values in all lanes, making the
// shuffle meaningless. Now we initialize with LANE-UNIQUE values to actually
// verify the shuffle exchanges data correctly.
__global__ void test_shuffle_kernel(float *output) {
  int lane_id = threadIdx.x;
  if (lane_id >= WARP_SIZE)
    return;

  // Initialize with LANE-UNIQUE values simulating post-SMFMAC state.
  // We encode lane identity: acc[i] = lane_id * 100 + i
  // This lets us verify which lane's data ends up where after shuffle.
  f32x4 acc;
  acc[0] = static_cast<float>(lane_id * 100 + 0);
  acc[1] = static_cast<float>(lane_id * 100 + 1);
  acc[2] = 0.0f; // Unused after fuse
  acc[3] = 0.0f; // Unused after fuse

  // Shuffle exchange (production lines 673-675)
  // Even lanes swap acc[1], odd lanes swap acc[0]
  int id_to_swap = 1 - (lane_id % 2);
  int src_lane = lane_id + 1 - 2 * (lane_id % 2);
  acc[id_to_swap] = __shfl(acc[id_to_swap], src_lane);

  // Each lane writes to ITS OWN output position (fixed: was writing to same addr)
  output[lane_id * 2 + 0] = acc[0];
  output[lane_id * 2 + 1] = acc[1];
}

void test_shuffle_exchange_gpu() {
  constexpr int NUM_THREADS = WARP_SIZE;
  constexpr int OUTPUT_FLOATS = NUM_THREADS * 2;

  float *d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, OUTPUT_FLOATS * sizeof(float)));

  test_shuffle_kernel<<<1, NUM_THREADS>>>(d_output);
  HIP_CHECK(hipGetLastError());

  std::vector<float> h_output(OUTPUT_FLOATS);
  HIP_CHECK(hipMemcpy(h_output.data(), d_output, OUTPUT_FLOATS * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Verify the shuffle exchange:
  // Before: even lane has (even*100+0, even*100+1), odd has (odd*100+0, odd*100+1)
  //
  // CRITICAL: __shfl is a warp collective where each lane provides a value and
  // reads from a source lane. When different lanes shuffle DIFFERENT array
  // indices (even shuffles acc[1], odd shuffles acc[0]), the returned value is
  // whatever the source lane PROVIDED, not the same-index value from source.
  //
  // Execution trace for lanes 0 and 1:
  //   Lane 0: provides acc[1]=1, reads from lane 1 → gets 100 (lane 1's acc[0])
  //   Lane 1: provides acc[0]=100, reads from lane 0 → gets 1 (lane 0's acc[1])
  //
  // After shuffle (cross-index exchange):
  //   Even lane: keeps acc[0], acc[1] receives neighbor's acc[0]
  //     -> (even*100+0, odd*100+0)
  //   Odd lane: acc[0] receives neighbor's acc[1], keeps acc[1]
  //     -> (even*100+1, odd*100+1)
  bool passed = true;
  for (int lane = 0; lane < NUM_THREADS && passed; ++lane) {
    float actual_0 = h_output[lane * 2 + 0];
    float actual_1 = h_output[lane * 2 + 1];

    float expected_0, expected_1;
    if (lane % 2 == 0) {
      // Even lane: keeps acc[0], acc[1] receives neighbor's acc[0] (cross-index)
      expected_0 = static_cast<float>(lane * 100 + 0);       // own acc[0]
      expected_1 = static_cast<float>((lane + 1) * 100 + 0); // neighbor's acc[0]!
    } else {
      // Odd lane: acc[0] receives neighbor's acc[1] (cross-index), keeps acc[1]
      expected_0 = static_cast<float>((lane - 1) * 100 + 1); // neighbor's acc[1]!
      expected_1 = static_cast<float>(lane * 100 + 1);       // own acc[1]
    }

    if (actual_0 != expected_0 || actual_1 != expected_1) {
      fprintf(stderr,
              "test_shuffle_exchange_gpu FAILED: lane %d (%s): "
              "got (%.0f, %.0f), expected (%.0f, %.0f)\n",
              lane, (lane % 2 == 0) ? "even" : "odd", actual_0, actual_1,
              expected_0, expected_1);
      passed = false;
    }
  }

  HIP_CHECK(hipFree(d_output));

  if (passed) {
    printf("test_shuffle_exchange_gpu: PASSED\n");
  } else {
    abort();
  }
}

void run_gpu_tests() {
  printf("Running GPU layout tests...\n");

  // Swizzle test (independent of OPS)
  test_swizzle_gpu();

  // LDS round-trip test - templated to match production kernel
  test_lds_roundtrip_gpu<8, 16, 2>();
  test_lds_roundtrip_gpu<8, 16, 4>();
  test_lds_roundtrip_gpu<8, 16, 8>();

  // Shuffle exchange test (independent of OPS)
  test_shuffle_exchange_gpu();

  printf("All GPU layout tests passed.\n");
}

} // namespace layout_gpu_tests

void test(const SkinnyGemmKernel &kernel) {
  const char *filter = getenv("FILTER");
  if (filter && !strstr(kernel.name, filter)) {
    return;
  }
  std::printf("%s: A:%s, B:%s, C:%s, tile MxNxK=%dx%dx%d\n", kernel.name,
              str(kernel.A_type), str(kernel.B_type), str(kernel.C_type),
              kernel.M_tile, kernel.N_tile, kernel.K_tile);

  if (!getenv("SKIP_CHECK")) {
    check(kernel);
  }

  MNKShape test_shapes[] = {
      {8, 2304, 16384}, {8, 6656, 16384}, {8, 13312, 16384}};
  for (MNKShape shape : test_shapes) {
    benchmark(kernel, shape);
  }
}

int main() {
  bool run_gpu_tests = getenv("TEST_LAYOUT_GPU") != nullptr;

  if (run_gpu_tests) {
    layout_gpu_tests::run_gpu_tests();
  }

  test(SparseTrickFP8Kernel<8, 16, 2>{});
  test(SparseTrickFP8Kernel<8, 16, 4>{});
  test(SparseTrickFP8Kernel<8, 16, 8>{});
  test(PaddingFP8Kernel<8, 16, 4>{});
  test(PaddingFP8Kernel<8, 16, 8>{});
  test(PaddingFP8KernelWithSwizzle<8, 16, 4>{});
  test(PaddingFP8KernelWithSwizzle<8, 16, 8>{});
  return 0;
}
