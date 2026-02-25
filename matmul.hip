// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hip"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cxxabi.h>
#include <optional>
#include <random>
#include <typeinfo>
#include <vector>

typedef void (*mmt_func_t)(const void *, const void *, void *, void *, int, int,
                           int);

typedef int (*tile_layout_func_t)(int, int);

struct TiledMatrixShape {
  int rows_outer, cols_outer;
  int rows_tile, cols_tile;
  tile_layout_func_t tile_layout;
};

struct TiledMmtShape {
  MNKShape outer, tile;
  tile_layout_func_t A_tile_layout, B_tile_layout, C_tile_layout;
};

__device__ __host__ TiledMatrixShape A_shape(const TiledMmtShape &s) {
  return {s.outer.M, s.outer.K, s.tile.M, s.tile.K, s.A_tile_layout};
}

__device__ __host__ TiledMatrixShape B_shape(const TiledMmtShape &s) {
  return {s.outer.N, s.outer.K, s.tile.N, s.tile.K, s.B_tile_layout};
}

__device__ __host__ TiledMatrixShape C_shape(const TiledMmtShape &s) {
  return {s.outer.M, s.outer.N, s.tile.M, s.tile.N, s.C_tile_layout};
}

__device__ __host__ int flatsize(const TiledMatrixShape &s) {
  return s.rows_outer * s.cols_outer * s.rows_tile * s.cols_tile;
}

__device__ __host__ int offset(const TiledMatrixShape &s, int r_outer,
                               int c_outer, int r_tile, int c_tile) {
  return s.tile_layout(r_tile, c_tile) +
         s.rows_tile * s.cols_tile * (c_outer + s.cols_outer * r_outer);
}

// Base class for matrix-times-matrix-transposed ("mmt") kernels.
// As the RHS is transposed, the dimensions are:
// LHS = "A-matrix" : MxK
// RHS = "B-matrix" : NxK
// Accumulator = "C-matrix": MxN
//
// The data layout is tiled with tile sizes given by the {M,N,K}_tile methods
// and tile layouts given by the {A,B,C}_layout methods.
class MmtKernel {
public:
  virtual ~MmtKernel() {};
  // Returns the element type of the A-matrix (LHS)
  virtual Type A_type() const = 0;
  // Returns the element type of the B-matrix (RHS)
  virtual Type B_type() const = 0;
  // Returns the element type of the C-matrix (accumulator/result)
  virtual Type C_type() const = 0;
  // Returns the M-dimension tile size (rows of accumulator)
  virtual int M_tile() const = 0;
  // Returns the N-dimension tile size (columns of accumulator)
  virtual int N_tile() const = 0;
  // Returns the K-dimension tile size (reduction dimension)
  virtual int K_tile() const = 0;
  // Returns the offset-computation function describing the A-matrix layout.
  virtual tile_layout_func_t A_tile_layout() const = 0;
  // Returns the offset-computation function describing the B-matrix layout.
  virtual tile_layout_func_t B_tile_layout() const = 0;
  // Returns the offset-computation function describing the C-matrix layout.
  virtual tile_layout_func_t C_tile_layout() const = 0;
  // Returns the number of threads that the kernel requires running on.
  virtual int num_threads() const = 0;
  // Returns a pointer to the device kernel.
  virtual mmt_func_t mmt_func() const = 0;
  // Optional: kernels may override this method to override the default grid.
  virtual std::optional<dim3>
  get_work_centric_grid(const MNKShape & /*outer*/) const {
    return {};
  }
  // Optional: kernels may override this to get an auxiliary device buffer of
  // the given size in bytes.
  virtual int aux_buffer_size(const MNKShape & /*outer*/) const { return 0; }
};

MNKShape getBenchmarkMNKShape(const MmtKernel &kernel) {
  int M = getIntEnvVar("M", 4096);
  int N = getIntEnvVar("N", 4096);
  int K = getIntEnvVar("K", 4096);
  MNKShape o;
  o.M = std::max(1, M / kernel.M_tile());
  o.N = std::max(1, N / kernel.N_tile());
  o.K = std::max(1, K / kernel.K_tile());
  return o;
}

TiledMmtShape getTestShape(const MmtKernel &kernel, const MNKShape &o) {
  TiledMmtShape s;
  s.outer = o;
  s.tile.M = kernel.M_tile();
  s.tile.N = kernel.N_tile();
  s.tile.K = kernel.K_tile();
  s.A_tile_layout = kernel.A_tile_layout();
  s.B_tile_layout = kernel.B_tile_layout();
  s.C_tile_layout = kernel.C_tile_layout();
  return s;
}

dim3 getLaunchGrid(const MmtKernel &kernel, const TiledMmtShape &s) {
  return kernel.get_work_centric_grid(s.outer).value_or(
      dim3(s.outer.M, s.outer.N));
}

template <Type A_type, Type B_type, Type C_type>
void checkMmtResults(const void *A_data_void, const void *B_data_void,
                     const void *C_data_void, const TiledMmtShape &s) {
  using TA = CType<A_type>;
  using TB = CType<B_type>;
  using TC = CType<C_type>;
  const TA *A_data = static_cast<const TA *>(A_data_void);
  const TB *B_data = static_cast<const TB *>(B_data_void);
  const TC *C_data = static_cast<const TC *>(C_data_void);
  // This reference code is slow. To make the checks not too slow on
  // large matmuls, we only check the 4 corner tiles.
  for (int m_outer : {0, s.outer.M - 1}) {
    for (int n_outer : {0, s.outer.N - 1}) {
      for (int m_tile = 0; m_tile < s.tile.M; ++m_tile) {
        for (int n_tile = 0; n_tile < s.tile.N; ++n_tile) {
          float c = 0.f;
          for (int k_outer = 0; k_outer < s.outer.K; ++k_outer) {
            for (int k_tile = 0; k_tile < s.tile.K; ++k_tile) {
              TA a =
                  A_data[offset(A_shape(s), m_outer, k_outer, m_tile, k_tile)];
              TB b =
                  B_data[offset(B_shape(s), n_outer, k_outer, n_tile, k_tile)];
              c += static_cast<TC>(a) * static_cast<TC>(b);
            }
          }
          TC expected = c;
          TC actual =
              C_data[offset(C_shape(s), m_outer, n_outer, m_tile, n_tile)];
          if (actual != expected) {
            fprintf(stderr,
                    "matmul numerical error: actual(%g) != "
                    "expected(%g), at m_outer=%d n_outer=%d m_tile=%d "
                    "n_tile=%d, at %s:%d. Note: outer MxNxK = %dx%dx%d\n",
                    static_cast<float>(actual), static_cast<float>(expected),
                    m_outer, n_outer, m_tile, n_tile, __FILE__, __LINE__,
                    s.outer.M, s.outer.N, s.outer.K);
            abort();
          }
        }
      }
    }
  }
}

void checkMmtResults(Type A_type, Type B_type, Type C_type,
                     const void *A_data_void, const void *B_data_void,
                     const void *C_data_void, const TiledMmtShape &s) {
#define HANDLE_CASE(A, B, C)                                                   \
  if (A_type == Type::A && B_type == Type::B && C_type == Type::C) {           \
    checkMmtResults<Type::A, Type::B, Type::C>(A_data_void, B_data_void,       \
                                               C_data_void, s);                \
    return;                                                                    \
  }
  HANDLE_CASE(FP32, FP32, FP32)
  HANDLE_CASE(FP16, FP16, FP32)
  HANDLE_CASE(SI8, SI8, SI32)
#undef HANDLE_CASE

  fprintf(stderr, "%s:%d: unhandled types\n", __FILE__, __LINE__);
  abort();
}

void check(const MmtKernel &kernel, const MNKShape &o) {
  TiledMmtShape s = getTestShape(kernel, o);
  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type(), flatsize(A_shape(s)), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type(), flatsize(B_shape(s)), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type(), flatsize(C_shape(s)), random_engine);

  void *A_device_buffer{};
  void *B_device_buffer{};
  void *C_device_buffer{};
  void *aux_device_buffer{};
  TiledMmtShape *shape_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));
  HIP_CHECK(hipGetLastError());
  int aux_buffer_size = kernel.aux_buffer_size(o);
  HIP_CHECK(hipMalloc(&aux_device_buffer, aux_buffer_size));
  HIP_CHECK(hipMemset(aux_device_buffer, 0, aux_buffer_size));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&shape_device_buffer, sizeof s));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));

  HIP_CHECK(
      hipMemcpy(shape_device_buffer, &s, sizeof s, hipMemcpyHostToDevice));
  HIP_CHECK(hipGetLastError());
  const dim3 grid_dim = getLaunchGrid(kernel, s);
  const dim3 block_dim(kernel.num_threads());
  HIP_CHECK(hipGetLastError());
  kernel.mmt_func()<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      A_device_buffer, B_device_buffer, C_device_buffer, aux_device_buffer,
      s.outer.M, s.outer.N, s.outer.K);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_host_data.size(),
                      hipMemcpyDeviceToHost));
  checkMmtResults(kernel.A_type(), kernel.B_type(), kernel.C_type(),
                  A_host_data.data(), B_host_data.data(), C_host_data.data(),
                  s);

  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
  HIP_CHECK(hipFree(aux_device_buffer));
  HIP_CHECK(hipFree(shape_device_buffer));
}

void check(const MmtKernel &kernel) {
  std::printf("  Checking correctness... ");
  // Test with more generic shapes than just M==N==K==2^x.
  for (MNKShape o : {MNKShape{1, 1, 1}, MNKShape{2, 1, 1}, MNKShape{1, 2, 1},
                     MNKShape{1, 1, 2}, MNKShape{1, 1, 3}, MNKShape{1, 1, 4},
                     MNKShape{1, 1, 5}, MNKShape{2, 2, 2}, MNKShape{2, 3, 4},
                     MNKShape{5, 2, 3}, MNKShape{1, 1, 10}, MNKShape{4, 4, 8},
                     MNKShape{305, 1, 1}, MNKShape{20, 20, 20}}) {
    check(kernel, o);
  }
  std::printf("OK\n");
}

void benchmark(const MmtKernel &kernel, const MNKShape &o) {
  TiledMmtShape s = getTestShape(kernel, o);
  std::printf("  Benchmarking: total MxNxK=%dx%dx%d, outer MxNxK=%dx%dx%d ... ",
              s.outer.M * s.tile.M, s.outer.N * s.tile.N, s.outer.K * s.tile.K,
              s.outer.M, s.outer.N, s.outer.K);

  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type(), flatsize(A_shape(s)), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type(), flatsize(B_shape(s)), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type(), flatsize(C_shape(s)), random_engine);

  void *A_device_buffer{};
  void *B_device_buffer{};
  void *C_device_buffer{};
  void *aux_device_buffer{};
  TiledMmtShape *shape_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));
  int aux_buffer_size = kernel.aux_buffer_size(o);
  HIP_CHECK(hipMalloc(&aux_device_buffer, aux_buffer_size));
  HIP_CHECK(hipMemset(aux_device_buffer, 0, aux_buffer_size));
  HIP_CHECK(hipMalloc(&shape_device_buffer, sizeof s));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(shape_device_buffer, &s, sizeof s, hipMemcpyHostToDevice));

  const dim3 grid_dim = getLaunchGrid(kernel, s);
  const dim3 block_dim(kernel.num_threads());

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
      kernel.mmt_func()<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer, aux_device_buffer,
          s.outer.M, s.outer.N, s.outer.K);
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

  // Calculate the actual amount of memory read during the calculation, taking
  // into account the tile sizes.
  float A_element_bytes = type_size(kernel.A_type());
  float B_element_bytes = type_size(kernel.B_type());
  float MNK = static_cast<float>(s.outer.M) * s.outer.N * s.outer.K;
  float kernel_bytes_read = MNK * ((A_element_bytes * s.tile.M * s.tile.K) +
                                   (B_element_bytes * s.tile.N * s.tile.K));

  float kernel_ms = elapsed_ms / iterations;
  float kernel_ops =
      2.f * s.outer.M * s.outer.N * s.outer.K * s.tile.M * s.tile.N * s.tile.K;
  float kernel_ops_per_s = 1000.f * kernel_ops / kernel_ms;
  float kernel_bytes_read_per_s = 1000.f * kernel_bytes_read / kernel_ms;
  std::printf("%.4g Tflop/s, read %.4g TB/s, latency %.2g ms, iterations=%d\n",
              1.e-12f * kernel_ops_per_s, 1e-12f * kernel_bytes_read_per_s,
              kernel_ms, iterations);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
  HIP_CHECK(hipFree(aux_device_buffer));
  HIP_CHECK(hipFree(shape_device_buffer));
}

void test(const MmtKernel &kernel) {
  char *name =
      abi::__cxa_demangle(typeid(kernel).name(), nullptr, nullptr, nullptr);
  const char *filter = getenv("FILTER");
  if (filter && !strstr(name, filter)) {
    return;
  }
  std::printf("%s: A:%s, B:%s, C:%s, tile MxNxK=%dx%dx%d, num_threads=%d\n",
              name, str(kernel.A_type()), str(kernel.B_type()),
              str(kernel.C_type()), kernel.M_tile(), kernel.N_tile(),
              kernel.K_tile(), kernel.num_threads());
  free(name);

  if (!getenv("SKIP_CHECK")) {
    check(kernel);
  }

  MNKShape o = getBenchmarkMNKShape(kernel);
  benchmark(kernel, o);
}

template <Type T_A_type, Type T_B_type, Type T_C_type, int T_M_tile,
          int T_N_tile, int T_K_tile>
class MmtKernel_generic : public MmtKernel {
  virtual Type A_type() const override { return T_A_type; }
  virtual Type B_type() const override { return T_B_type; }
  virtual Type C_type() const override { return T_C_type; }
  virtual int M_tile() const override { return T_M_tile; }
  virtual int N_tile() const override { return T_N_tile; }
  virtual int K_tile() const override { return T_K_tile; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return k + T_K_tile * m; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return k + T_K_tile * n; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return n + T_N_tile * m; };
  }
  virtual int num_threads() const override { return T_M_tile * T_N_tile; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, void * /*aux_data*/, int /*M_outer*/,
                             int N_outer, int K_outer) {
    using TA = CType<T_A_type>;
    using TB = CType<T_B_type>;
    using TC = CType<T_C_type>;
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int m_tile = threadIdx.x / T_N_tile;
    int n_tile = threadIdx.x % T_N_tile;
    TC c = {0};
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int k_tile = 0; k_tile < T_K_tile; ++k_tile) {
        TA a = static_cast<const TA *>(
            A_data)[k_tile +
                    T_K_tile *
                        (m_tile + T_M_tile * (k_outer + K_outer * m_outer))];
        TB b = static_cast<const TB *>(
            B_data)[k_tile +
                    T_K_tile *
                        (n_tile + T_N_tile * (k_outer + K_outer * n_outer))];
        c += static_cast<TC>(a) * static_cast<TC>(b);
      }
    }
    static_cast<TC *>(
        C_data)[n_tile + T_N_tile * (m_tile + T_M_tile * (n_outer +
                                                          N_outer * m_outer))] =
        c;
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_rowmajor : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return 4 * m + k; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return 16 * k + n; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 16 * m + n; };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;
    int ai = tid % 16;
    int ak = tid / 16;
    int bj = tid % 16;
    int bk = tid / 16;

    const float *A_ptr = static_cast<const float *>(A_data) +
                         m_outer * K_outer * 64 + ai * 4 + ak;
    const float *B_ptr = static_cast<const float *>(B_data) +
                         n_outer * K_outer * 64 + bk * 16 + bj;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    for (int gpr = 0; gpr < 4; ++gpr) {
      int ci = 4 * (tid / 16) + gpr;
      int cj = tid % 16;
      static_cast<float *>(
          C_data)[m_outer * N_outer * 256 + n_outer * 256 + ci * 16 + cj] =
          acc[gpr];
    }
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_directAB_rowmajorC
    : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m + 16 * k; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return n + 16 * k; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 16 * m + n; };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr =
        static_cast<const float *>(A_data) + m_outer * K_outer * 64 + tid;
    const float *B_ptr =
        static_cast<const float *>(B_data) + n_outer * K_outer * 64 + tid;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    for (int gpr = 0; gpr < 4; ++gpr) {
      int ci = 4 * (tid / 16) + gpr;
      int cj = tid % 16;
      static_cast<float *>(
          C_data)[m_outer * N_outer * 256 + n_outer * 256 + ci * 16 + cj] =
          acc[gpr];
    }
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m + 16 * k; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return n + 16 * k; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 64 * (m / 4) + 4 * n + (m % 4); };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr =
        static_cast<const float *>(A_data) + m_outer * K_outer * 64 + tid;
    const float *B_ptr =
        static_cast<const float *>(B_data) + n_outer * K_outer * 64 + tid;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    static_cast<floatx4_t *>(C_data)[64 * (N_outer * m_outer + n_outer) + tid] =
        acc;
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4 : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return (k / 4) + 4 * (m + 16 * (k % 4)); };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return (k / 4) + 4 * (n + 16 * (k % 4)); };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 64 * (m / 4) + 4 * n + (m % 4); };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr =
        static_cast<const float *>(A_data) + m_outer * K_outer * 256 + 4 * tid;
    const float *B_ptr =
        static_cast<const float *>(B_data) + n_outer * K_outer * 256 + 4 * tid;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_ptr[0], B_ptr[0], acc, 0, 0,
                                                 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_ptr[1], B_ptr[1], acc, 0, 0,
                                                 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_ptr[2], B_ptr[2], acc, 0, 0,
                                                 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_ptr[3], B_ptr[3], acc, 0, 0,
                                                 0);
      A_ptr += 256;
      B_ptr += 256;
    }

    static_cast<floatx4_t *>(C_data)[64 * (N_outer * m_outer + n_outer) + tid] =
        acc;
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4_unrollx4
    : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return (k / 4) + 4 * (m + 16 * (k % 4)); };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return (k / 4) + 4 * (n + 16 * (k % 4)); };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 64 * (m / 4) + 4 * n + (m % 4); };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const floatx4_t *A_ptr =
        static_cast<const floatx4_t *>(A_data) + m_outer * K_outer * 64 + tid;
    const floatx4_t *B_ptr =
        static_cast<const floatx4_t *>(B_data) + n_outer * K_outer * 64 + tid;

    int k_outer = 0;
    for (; k_outer <= K_outer - 4; k_outer += 4) {
      floatx4_t a0 = A_ptr[0];
      floatx4_t b0 = B_ptr[0];
      floatx4_t a1 = A_ptr[64];
      floatx4_t b1 = B_ptr[64];
      floatx4_t a2 = A_ptr[128];
      floatx4_t b2 = B_ptr[128];
      floatx4_t a3 = A_ptr[192];
      floatx4_t b3 = B_ptr[192];
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[0], b0[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[1], b0[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[2], b0[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[3], b0[3], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1[0], b1[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1[1], b1[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1[2], b1[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1[3], b1[3], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2[0], b2[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2[1], b2[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2[2], b2[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2[3], b2[3], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3[0], b3[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3[1], b3[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3[2], b3[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3[3], b3[3], acc, 0, 0, 0);
      A_ptr += 256;
      B_ptr += 256;
    }
    for (; k_outer < K_outer; ++k_outer) {
      floatx4_t a0 = A_ptr[0];
      floatx4_t b0 = B_ptr[0];
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[0], b0[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[1], b0[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[2], b0[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[3], b0[3], acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    static_cast<floatx4_t *>(C_data)[64 * (N_outer * m_outer + n_outer) + tid] =
        acc;
  }
};

class MmtKernel_128t_1x2_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 32; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return 16 * k + m; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int ni = n % 16;
      int no = n / 16;
      return 256 * no + 64 * (m / 4) + 4 * ni + (m % 4);
    };
  }
  virtual int num_threads() const override { return 128; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(128) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr = static_cast<const float *>(A_data) +
                         m_outer * K_outer * 64 + (tid % 64);
    const float *B_ptr =
        static_cast<const float *>(B_data) + n_outer * K_outer * 128 + tid;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 128;
    }

    static_cast<floatx4_t *>(
        C_data)[128 * (N_outer * m_outer + n_outer) + tid] = acc;
  }
};

class MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 32; }
  virtual int N_tile() const override { return 32; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 64 * mo + 16 * k + mi;
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return 512 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr = static_cast<const float *>(A_data) +
                         m_outer * K_outer * 128 + (tid % 64) +
                         64 * (tid / 128);
    const float *B_ptr = static_cast<const float *>(B_data) +
                         n_outer * K_outer * 128 + (tid % 128);

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 128;
      B_ptr += 128;
    }

    static_cast<floatx4_t *>(
        C_data)[256 * (N_outer * m_outer + n_outer) + tid] = acc;
  }
};

class MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_shared : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 32; }
  virtual int N_tile() const override { return 32; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 64 * mo + 16 * k + mi;
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return 512 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    int A_thread_offset = (tid % 64) + 64 * (tid / 128);
    int B_thread_offset = tid % 128;

    constexpr int A_tile_size = 32 * 4;
    constexpr int B_tile_size = 32 * 4;

    const float *A_global =
        static_cast<const float *>(A_data) + m_outer * K_outer * A_tile_size;
    const float *B_global =
        static_cast<const float *>(B_data) + n_outer * K_outer * B_tile_size;

    constexpr int K_outer_shared_size = 4; // Tuned.

    __shared__ float A_shared[K_outer_shared_size * A_tile_size];
    __shared__ float B_shared[K_outer_shared_size * B_tile_size];

    const float *A_global_ptr = A_global + A_thread_offset;
    const float *B_global_ptr = B_global + B_thread_offset;
    float *A_shared_base_ptr = A_shared + A_thread_offset;
    float *B_shared_base_ptr = B_shared + B_thread_offset;

    // Main loop: handle full-size shared tiles.
    int k_outer_global = 0;
    for (; k_outer_global <= K_outer - K_outer_shared_size;
         k_outer_global += K_outer_shared_size) {
      {
        float *A_shared_ptr = A_shared_base_ptr;
        float *B_shared_ptr = B_shared_base_ptr;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          *A_shared_ptr = *A_global_ptr;
          *B_shared_ptr = *B_global_ptr;
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
          A_global_ptr += A_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *A_shared_ptr = A_shared_base_ptr;
        const float *B_shared_ptr = B_shared_base_ptr;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          acc = __builtin_amdgcn_mfma_f32_16x16x4f32(
              *A_shared_ptr, *B_shared_ptr, acc, 0, 0, 0);
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    // Handle remainder: the last shared tile has a smaller K-size.
    if (k_outer_global < K_outer) {
      int K_remaining_outer_size = K_outer - k_outer_global;
      {
        float *A_shared_ptr = A_shared_base_ptr;
        float *B_shared_ptr = B_shared_base_ptr;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          *A_shared_ptr = *A_global_ptr;
          *B_shared_ptr = *B_global_ptr;
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
          A_global_ptr += A_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *A_shared_ptr = A_shared_base_ptr;
        const float *B_shared_ptr = B_shared_base_ptr;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          acc = __builtin_amdgcn_mfma_f32_16x16x4f32(
              *A_shared_ptr, *B_shared_ptr, acc, 0, 0, 0);
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    static_cast<floatx4_t *>(
        C_data)[256 * (N_outer * m_outer + n_outer) + tid] = acc;
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 64 * mo + 16 * k + mi;
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size = MS * 16 * 4;
    constexpr int B_tile_size = NS * 16 * 4;

    const float *A_global =
        static_cast<const float *>(A_data) + m_outer * K_outer * A_tile_size;
    const float *B_global =
        static_cast<const float *>(B_data) + n_outer * K_outer * B_tile_size;

    constexpr int K_outer_shared_size = 4; // Tuned.

    __shared__ float B_shared[K_outer_shared_size * B_tile_size];

    const float *A_global_ptr = A_global + (tid % 64);
    const float *B_global_ptr = B_global + tid;

    // Main loop: handle full-size shared tiles.
    int k_outer_global = 0;
    for (; k_outer_global <= K_outer - K_outer_shared_size;
         k_outer_global += K_outer_shared_size) {
      {
        float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          for (int j = 0; j < B_tile_size; j += 256) {
            B_shared_ptr[j] = B_global_ptr[j];
          }
          B_shared_ptr += B_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          for (int i = 0; i < MS; ++i) {
            for (int j = 0; j < NS / 4; ++j) {
              acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                  A_global_ptr[64 * i], B_shared_ptr[256 * j], acc[i][j], 0, 0,
                  0);
            }
          }
          A_global_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    // Handle remainder: the last shared tile has a smaller K-size.
    if (k_outer_global < K_outer) {
      int K_remaining_outer_size = K_outer - k_outer_global;
      {
        float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          for (int j = 0; j < B_tile_size; j += 256) {
            B_shared_ptr[j] = B_global_ptr[j];
          }
          B_shared_ptr += B_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          for (int i = 0; i < MS; ++i) {
            for (int j = 0; j < NS / 4; ++j) {
              acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                  A_global_ptr[64 * i], B_shared_ptr[256 * j], acc[i][j], 0, 0,
                  0);
            }
          }
          A_global_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[16 * 16 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 64 * mo + 16 * k + mi;
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size = MS * 16 * 4;
    constexpr int B_tile_size = NS * 16 * 4;

    const float *A_global =
        static_cast<const float *>(A_data) + m_outer * K_outer * A_tile_size;
    const float *B_global =
        static_cast<const float *>(B_data) + n_outer * K_outer * B_tile_size;

    constexpr int K_outer_shared_size = 4; // Tuned.

    __shared__ float A_shared[K_outer_shared_size * A_tile_size];
    __shared__ float B_shared[K_outer_shared_size * B_tile_size];

    const float *A_global_ptr = A_global + tid;
    const float *B_global_ptr = B_global + tid;

    // Main loop: handle full-size shared tiles.
    int k_outer_global = 0;
    for (; k_outer_global <= K_outer - K_outer_shared_size;
         k_outer_global += K_outer_shared_size) {
      {
        float *A_shared_ptr = A_shared + tid;
        float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          for (int i = 0; i < A_tile_size; i += 256) {
            A_shared_ptr[i] = A_global_ptr[i];
          }
          for (int j = 0; j < B_tile_size; j += 256) {
            B_shared_ptr[j] = B_global_ptr[j];
          }
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
          A_global_ptr += A_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *A_shared_ptr = A_shared + (tid % 64);
        const float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          for (int i = 0; i < MS; ++i) {
            for (int j = 0; j < NS / 4; ++j) {
              acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                  A_shared_ptr[64 * i], B_shared_ptr[256 * j], acc[i][j], 0, 0,
                  0);
            }
          }
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    // Handle remainder: the last shared tile has a smaller K-size.
    if (k_outer_global < K_outer) {
      int K_remaining_outer_size = K_outer - k_outer_global;
      {
        float *A_shared_ptr = A_shared + tid;
        float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          for (int i = 0; i < A_tile_size; i += 256) {
            A_shared_ptr[i] = A_global_ptr[i];
          }
          for (int j = 0; j < B_tile_size; j += 256) {
            B_shared_ptr[j] = B_global_ptr[j];
          }
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
          A_global_ptr += A_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *A_shared_ptr = A_shared + (tid % 64);
        const float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          for (int i = 0; i < MS; ++i) {
            for (int j = 0; j < NS / 4; ++j) {
              acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                  A_shared_ptr[64 * i], B_shared_ptr[256 * j], acc[i][j], 0, 0,
                  0);
            }
          }
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[16 * 16 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[B_tile_size_in_vec4];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[64 * i + (tid % 64)][k], B_shared[256 * j + tid][k],
                acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_subgroup2x2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return (NS / 2) * 1024 * (mo / 2) + 1024 * (no / 2) + 512 * (mo % 2) +
             256 * (no % 2) + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS / 2][NS / 2] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[B_tile_size_in_vec4];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS / 2; ++i) {
          for (int j = 0; j < NS / 2; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[128 * i + 64 * (tid / 128) + (tid % 64)][k],
                B_shared[128 * j + (tid % 128)][k], acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS / 2; ++i) {
      for (int j = 0; j < NS / 2; ++j) {
        C_ptr[256 * (NS / 2 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class
    MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_naive
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[2][A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[2][B_tile_size_in_vec4];

    for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
      A_shared[0][i + tid] = A_global_ptr[i];
    }
    for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
      B_shared[0][j + tid] = B_global_ptr[j];
    }
    A_global_ptr += A_tile_size_in_vec4;
    B_global_ptr += B_tile_size_in_vec4;
    // No __syncthreads here, the loop starts by loading separate parts of
    // A_shared and B_shared and then does a __syncthreads.

    for (int k_outer = 1; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[k_outer & 1][i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[k_outer & 1][j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[(k_outer ^ 1) & 1][64 * i + (tid % 64)][k],
                B_shared[(k_outer ^ 1) & 1][256 * j + tid][k], acc[i][j], 0, 0,
                0);
          }
        }
      }
      // We thought that there should be no __syncthreads here, as the next
      // iteration will load into separate parts of A_shared and B_shared.
      // But we observed a data race giving intermittent failures in practice.
      __syncthreads();
    }

    for (int k = 0; k < 4; ++k) {
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
              A_shared[(K_outer ^ 1) & 1][64 * i + (tid % 64)][k],
              B_shared[(K_outer ^ 1) & 1][256 * j + tid][k], acc[i][j], 0, 0,
              0);
        }
      }
    }
    __syncthreads();

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class
    MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_take2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[2][A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[2][B_tile_size_in_vec4];

    auto load_shared_from_global = [&](int buffer_id) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[buffer_id][i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[buffer_id][j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
    };

    auto mfma = [&](int buffer_id) {
      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[buffer_id][64 * i + (tid % 64)][k],
                B_shared[buffer_id][256 * j + tid][k], acc[i][j], 0, 0, 0);
          }
        }
      }
    };

    load_shared_from_global(0);

    while (K_outer >= 3) {
      load_shared_from_global(1);
      __syncthreads();
      mfma(0);
      load_shared_from_global(0);
      mfma(1);
      K_outer -= 2;
    }

    if (K_outer == 2) {
      load_shared_from_global(1);
      __syncthreads();
      mfma(0);
      mfma(1);
    } else {
      __syncthreads();
      mfma(0);
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = (m % 64) / 16;
      int mx = m / 64;
      int ni = n % 16;
      int no = (n % 64) / 16;
      int nx = n / 64;
      return (NS / 4) * 4096 * mx + 4096 * nx + 1024 * mo + 256 * no +
             64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 1024; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(1024) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS / 4][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[B_tile_size_in_vec4];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 1024) {
        if (tid < MS * 64)
          A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 1024) {
        if (tid < NS * 64)
          B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS / 4; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[256 * i + 64 * (tid / 256) + (tid % 64)][k],
                B_shared[256 * j + (tid % 256)][k], acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS / 4; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[1024 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class
    MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_misguidednobankconflicts
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx2_t = __attribute__((__vector_size__(2 * sizeof(float)))) float;
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec2 = MS * 16 * 4 * 2;
    constexpr int B_tile_size_in_vec2 = NS * 16 * 4 * 2;

    const floatx2_t *A_global = static_cast<const floatx2_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec2;
    const floatx2_t *B_global = static_cast<const floatx2_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec2;
    const floatx2_t *A_global_ptr = A_global + 2 * tid;
    const floatx2_t *B_global_ptr = B_global + 2 * tid;

    __shared__ floatx2_t A_shared[A_tile_size_in_vec2];
    __shared__ floatx2_t B_shared[B_tile_size_in_vec2];

    int k0 = (tid / 8) % 2;
    int k1 = 1 ^ k0;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec2; i += 2 * 256) {
        A_shared[i + 2 * tid + k0] = A_global_ptr[i];
        A_shared[i + 2 * tid + k1] = A_global_ptr[i + 1];
      }
      for (int j = 0; j < B_tile_size_in_vec2; j += 2 * 256) {
        B_shared[j + 2 * tid + k0] = B_global_ptr[j];
        B_shared[j + 2 * tid + k1] = B_global_ptr[j + 1];
      }
      A_global_ptr += A_tile_size_in_vec2;
      B_global_ptr += B_tile_size_in_vec2;
      __syncthreads();

      for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < MS; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[2 * 64 * i + 2 * (tid % 64) + k0][p],
                B_shared[2 * 256 * j + 2 * tid + k0][p], acc[i][j], 0, 0, 0);
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[2 * 64 * i + 2 * (tid % 64) + k1][p],
                B_shared[2 * 256 * j + 2 * tid + k1][p], acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[16 * 16 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 256 * mo + 64 * (k / 4) + 4 * mi + (k % 4);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 256 * no + 64 * (k / 4) + 4 * ni + (k % 4);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using float16x4_t =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const float16x4_t *A_global = static_cast<const float16x4_t *>(A_data) +
                                  m_outer * K_outer * A_tile_size_in_vec4;
    const float16x4_t *B_global = static_cast<const float16x4_t *>(B_data) +
                                  n_outer * K_outer * B_tile_size_in_vec4;
    const float16x4_t *A_global_ptr = A_global + tid;
    const float16x4_t *B_global_ptr = B_global + tid;

    __shared__ float16x4_t A_shared[A_tile_size_in_vec4];
    __shared__ float16x4_t B_shared[B_tile_size_in_vec4];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16f16(
              A_shared[64 * i + (tid % 64)], B_shared[256 * j + tid], acc[i][j],
              0, 0, 0);
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 32; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 512 * mo + 128 * ((k % 16) / 4) + 8 * mi + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 512 * no + 128 * ((k % 16) / 4) + 8 * ni + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using float16x4_t =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using float16x8_t =
        __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4;

    const float16x8_t *A_global = static_cast<const float16x8_t *>(A_data) +
                                  m_outer * K_outer * A_tile_size_in_vec8;
    const float16x8_t *B_global = static_cast<const float16x8_t *>(B_data) +
                                  n_outer * K_outer * B_tile_size_in_vec8;
    const float16x8_t *A_global_ptr = A_global + tid;
    const float16x8_t *B_global_ptr = B_global + tid;

    __shared__ float16x8_t A_shared[A_tile_size_in_vec8];
    __shared__ float16x8_t B_shared[B_tile_size_in_vec8];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          float16x8_t a = A_shared[64 * i + (tid % 64)];
          float16x8_t b = B_shared[256 * j + tid];
          float16x4_t a0 = (float16x4_t){a[0], a[1], a[2], a[3]};
          float16x4_t a1 = (float16x4_t){a[4], a[5], a[6], a[7]};
          float16x4_t b0 = (float16x4_t){b[0], b[1], b[2], b[3]};
          float16x4_t b1 = (float16x4_t){b[4], b[5], b[6], b[7]};
          acc[i][j] =
              __builtin_amdgcn_mfma_f32_16x16x16f16(a0, b0, acc[i][j], 0, 0, 0);
          acc[i][j] =
              __builtin_amdgcn_mfma_f32_16x16x16f16(a1, b1, acc[i][j], 0, 0, 0);
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

// A simple Stream-K kernel variant of
// MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2.
//
// The synchronization between workgroups cooperating on a tile is wait-free
// and produces deterministic results. This is achieved using a per-tile atomic
// counter to determine which workgroup is the last to complete its share of the
// tile's computation. That workgroup will then go on to perform the final
// reduction of the multiple workgroups' local accumulators into the final
// accumulator. The results are still deterministic thanks to performing that
// final reduction in order of increasing K-dimension indices, regardless of
// which workgroup happens to be doing it.
//
// The inner arithmetic loops isn't particularly optimized, being the same as in
// MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2.
// Like in other kernels, the template parameters MS and NS control the number
// of MFMA intrinsics in the kernel along the M and N dimensions respectively:
// MS x (NS / 4), while 4 subgroups are stacked along the NS dimension,
// achieving a total width of MS x NS MFMA intrinsics across all 4 subgroups.
//
template <int MS, int NS>
class MmtKernel_StreamK_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  // Defining some values as constants as they will be needed in static methods
  // below.
  static constexpr Type static_A_type = Type::FP16;
  static constexpr Type static_B_type = Type::FP16;
  static constexpr Type static_C_type = Type::FP32;
  static constexpr int static_M_tile = MS * 16;
  static constexpr int static_N_tile = NS * 16;
  static constexpr int static_K_tile = 32;
  // Standard MmtKernel interface methods override describing element types.
  virtual Type A_type() const override { return static_A_type; }
  virtual Type B_type() const override { return static_B_type; }
  virtual Type C_type() const override { return static_C_type; }
  // Standard MmtKernel interface methods override describing tile shapes.
  virtual int M_tile() const override { return static_M_tile; }
  virtual int N_tile() const override { return static_N_tile; }
  virtual int K_tile() const override { return static_K_tile; }
  // Standard MmtKernel interface methods override describing tile layouts.
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 512 * mo + 128 * ((k % 16) / 4) + 8 * mi + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 512 * no + 128 * ((k % 16) / 4) + 8 * ni + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  // Standard MmtKernel interface method override describing number of threads.
  virtual int num_threads() const override { return 256; }
  // Additional constants and local helpers (static methods) for work-centric
  // grid and auxiliary buffer allocation.
  //
  // Maximum number of Compute Units that we aim to use on sufficiently large
  // problems. This value assumes a MI300X in SPX mode.
  static constexpr int MaxWorkgroups = 304;
  // Alignment to use when coalescing device buffers to avoid alignment-related
  // performance caveats.
  static constexpr int device_alignment = 128;
  // Alignment to use for atomics to avoid false-sharing effects.
  static constexpr int atomics_alignment = 128;
  // Returns the number of Compute Units that we aim to use for the problem
  // shape with given `outer` shape (expressed in units of this kernel's tiles).
  static __device__ __host__ int get_workgroup_count(const MNKShape &outer) {
    int64_t total_iters = static_cast<int64_t>(outer.M) * outer.N * outer.K;
    return std::min(static_cast<int64_t>(MaxWorkgroups), total_iters);
  }
  static __device__ __host__ int get_tile_count(const MNKShape &outer) {
    return outer.M * outer.N;
  }
  // Returns the buffer bytes to allocate in the auxiliary buffer for the atomic
  // counters (one per tile).
  using atomic_t = int;
  static __device__ __host__ int atomics_buffer_size(const MNKShape &outer) {
    return round_up_to_po2<device_alignment>(
        get_tile_count(outer) *
        round_up_to_po2<atomics_alignment>(sizeof(atomic_t)));
  }
  // Returns the number of local accumulator buffers to allocate in the
  // auxiliary buffer, per tile. This value is a coarse upper bound, could be
  // refined to trim memory usage.
  static __device__ __host__ int
  local_accumulators_count_per_tile(const MNKShape &outer) {
    return 1 + ceil_div(get_workgroup_count(outer), get_tile_count(outer));
  }
  // Return the overall number of local accumulator buffers to allocate in the
  // auxiliary buffer. This value is a coarse upper bound, could be
  // refined to trim memory usage.
  static __device__ __host__ int
  local_accumulators_count(const MNKShape &outer) {
    return local_accumulators_count_per_tile(outer) * get_tile_count(outer);
  }
  // Return the overall auxiliary buffer bytes to allocate.
  static __device__ __host__ int
  local_accumulators_buffer_size(const MNKShape &outer) {
    return round_up_to_po2<device_alignment>(local_accumulators_count(outer) *
                                             static_M_tile * static_N_tile *
                                             type_size(static_C_type));
  }
  // Standard MmtKernel interface method override describing the work-centric
  // grid.
  virtual std::optional<dim3>
  get_work_centric_grid(const MNKShape &outer) const override {
    return {get_workgroup_count(outer)};
  }
  // Standard MmtKernel interface method override describing the auxiliary
  // buffer size to allocate.
  virtual int aux_buffer_size(const MNKShape &outer) const override {
    return atomics_buffer_size(outer) + local_accumulators_buffer_size(outer);
  }
  // Standard MmtKernel interface method override describing the actual device
  // kernel.
  virtual mmt_func_t mmt_func() const override { return run; };
  // Some typedefs used in the kernel implementation.
  using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  using float16x4_t =
      __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
  using float16x8_t =
      __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
  static constexpr int num_acc_vec4s = MS * NS / 4;
  using acc_array_t = std::array<floatx4_t, num_acc_vec4s>;
  // Device kernel implementation.
  __global__ __launch_bounds__(256) static void run(const void *A_data,
                                                    const void *B_data,
                                                    void *C_data,
                                                    void *aux_data, int M_outer,
                                                    int N_outer, int K_outer) {
    MNKShape outer{M_outer, N_outer, K_outer};
    // Get local accumulators buffer.
    char *aux_bytes = static_cast<char *>(aux_data);
    float *local_accumulators_buffer =
        reinterpret_cast<float *>(aux_bytes + atomics_buffer_size(outer));
    // General arithmetic just like in Algorithm 5 in
    // https://arxiv.org/pdf/2301.03598, except we use the term "workgroup"
    // instead of "CTA".
    int iters_per_tile = K_outer;
    int total_tiles = M_outer * N_outer;
    int total_iters = total_tiles * iters_per_tile;
    int workgroup_count = get_workgroup_count(outer);
    int workgroup = blockIdx.x;
    const int NUM_XCDS = 8;
    if (workgroup_count == MaxWorkgroups) {
      workgroup = (workgroup % NUM_XCDS) * (workgroup_count / NUM_XCDS) +
                  (workgroup / NUM_XCDS);
    }
    int iter_start = workgroup * total_iters / workgroup_count;
    int iter_end = (workgroup + 1) * total_iters / workgroup_count;
    int iter = iter_start;
    // Work-centric loop on iterations to be handled by this compute unit.
    while (iter < iter_end) {
      // Still same variable names as in Algorithm 5 in
      // https://arxiv.org/pdf/2301.03598.
      int tile_idx = iter / iters_per_tile;
      int tile_iter_start = tile_idx * iters_per_tile;
      int tile_iter_end = tile_iter_start + iters_per_tile;
      int local_iter = iter - tile_iter_start;
      int local_iter_end = std::min(tile_iter_end, iter_end) - tile_iter_start;
      if (local_iter == local_iter_end) {
        break;
      }
      // Map the 1D tile_idx to 2D (m_outer, n_outer) position in the C-matrix
      // tile space, using MxN-lexicographic order.
      int m_outer = tile_idx / N_outer;
      int n_outer = tile_idx - m_outer * N_outer;

      acc_array_t acc = mac_loop(A_data, B_data, K_outer, m_outer, n_outer,
                                 local_iter, local_iter_end);

      // The next step is specific to our wait-free atomic approach: we need to
      // know the interval of other compute units cooperating with the current
      // compute unit on the current tile.
      int cooperating_workgroup_start =
          ceil_div((tile_iter_start + 1) * workgroup_count, total_iters) - 1;
      int cooperating_workgroup_end =
          ceil_div(tile_iter_end * workgroup_count, total_iters);
      assert(workgroup >= cooperating_workgroup_start);
      assert(workgroup < cooperating_workgroup_end);

      int cooperating_workgroup_count =
          cooperating_workgroup_end - cooperating_workgroup_start;

      // Track whether we will ultimately perform the store of the final
      // accumulator. This is always true, except when cooperating with other
      // workgroups on the tile AND determining that another workgroup will be
      // the one doing the final reduction.
      bool should_store_final_accumulator = true;

      // Hard case: cooperating with other workgroups on this tile.
      if (cooperating_workgroup_count != 1) {
        int workgroup_idx_in_cooperating_group =
            workgroup - cooperating_workgroup_start;
        int local_accumulator_idx =
            workgroup_idx_in_cooperating_group +
            tile_idx * local_accumulators_count_per_tile(
                           MNKShape{M_outer, N_outer, K_outer});
        floatx4_t *local_accumulator = reinterpret_cast<floatx4_t *>(
            local_accumulators_buffer +
            local_accumulator_idx * static_M_tile * static_N_tile);

        // Store our own accumulators to memory. We don't know yet which
        // workgroup will perform the final reduction. If it is not us, then it
        // will need our local accumulators to have been stored to global
        // memory. If it is us, then this store is potentially redundant,
        // however:
        // 1. Something needs to provide memory ordering on the local
        // accumulator
        //    accesses made by different workgroups. At the moment, that thing
        //    is the same barrier that also determines which workgroup does the
        //    final reduction. If we try to potentially save the global store
        //    here, we will need to add a second fence.
        // 2. Trying to keep this in registers would mean more register pressure
        //    and potentially a new constraint on kernel tile size choice.
        // 3. Likewise if trying to keep that in shared memory.
        acc_store(acc, local_accumulator);

        atomic_t *atomic_counter = reinterpret_cast<atomic_t *>(
            aux_bytes + tile_idx * atomics_alignment);

        // Increment the atomic counter and agent-scope acquire-release fence.
        // This has the effect of ordering the above accumulator store before
        // the below accumulator loads across workgroups ("agent scope").
        int counter_value = workgroup_atomic_increment(atomic_counter);

        // Determine if we are the workgroup that should perform the final
        // reduction.
        if (counter_value == cooperating_workgroup_count) {
          // Perform the final reduction.
          acc = final_reduction(
              M_outer, N_outer, K_outer, local_accumulators_buffer, tile_idx,
              cooperating_workgroup_start, cooperating_workgroup_end);

          // Reset the atomic counter to 0 so that the auxiliary buffer is ready
          // to reuse in another kernel launch.
          if (threadIdx.x == 0) {
            *atomic_counter = 0;
          }
        } else {
          // Not performing the final reduction, so also not doing the store.
          should_store_final_accumulator = false;
        }
      }

      floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                         MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);

      if (should_store_final_accumulator) {
        // Store the final accumulator to the destination.
        // Note: consumer fusions go here.
        acc_store(acc, C_ptr);
      }

      iter = tile_iter_end;
    }
  }
  // Helper to load accumulators from memory.
  __device__ static acc_array_t acc_load(const floatx4_t *ptr) {
    acc_array_t acc;
    for (int i = 0; i < num_acc_vec4s; ++i) {
      acc[i] = ptr[256 * i + threadIdx.x];
    }
    return acc;
  }
  // Helper to store accumulators to memory.
  __device__ static void acc_store(acc_array_t acc, floatx4_t *ptr) {
    for (int i = 0; i < num_acc_vec4s; ++i) {
      ptr[256 * i + threadIdx.x] = acc[i];
    }
  }

  // Helper: inner MFMA loop. Similar to the MacLoop function given in
  // Algorithm 3 in https://arxiv.org/pdf/2301.03598.
  __device__ static acc_array_t mac_loop(const void *A_data, const void *B_data,
                                         int K_outer, int m_outer, int n_outer,
                                         int k_iter_start, int k_iter_end) {
    // Accumulator VGPRs.
    acc_array_t acc = {{0}};
    int tid = threadIdx.x;
    constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4;
    const float16x8_t *A_global =
        static_cast<const float16x8_t *>(A_data) +
        (m_outer * K_outer + k_iter_start) * A_tile_size_in_vec8;
    const float16x8_t *B_global =
        static_cast<const float16x8_t *>(B_data) +
        (n_outer * K_outer + k_iter_start) * B_tile_size_in_vec8;
    const float16x8_t *A_global_ptr = A_global + tid;
    const float16x8_t *B_global_ptr = B_global + tid;

    // Shared memory buffers for tiles of A and B matrices.
    __shared__ float16x8_t A_shared[A_tile_size_in_vec8];
    __shared__ float16x8_t B_shared[B_tile_size_in_vec8];

    // Inner loop on K dimenion.
    for (int k_iter = k_iter_start; k_iter < k_iter_end; ++k_iter) {
      // Load data from global to shared memory.
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();
      // Perform MFMA arithmetic on data in shared memory.
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          float16x8_t a = A_shared[64 * i + (tid % 64)];
          float16x8_t b = B_shared[256 * j + tid];
          float16x4_t a0 = (float16x4_t){a[0], a[1], a[2], a[3]};
          float16x4_t a1 = (float16x4_t){a[4], a[5], a[6], a[7]};
          float16x4_t b0 = (float16x4_t){b[0], b[1], b[2], b[3]};
          float16x4_t b1 = (float16x4_t){b[4], b[5], b[6], b[7]};
          floatx4_t acc0 = acc[i * (NS / 4) + j];
          acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a0, b0, acc0, 0, 0, 0);
          acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a1, b1, acc0, 0, 0, 0);
          acc[i * (NS / 4) + j] = acc0;
        }
      }
      __syncthreads();
    }
    return acc;
  }
  __device__ static int workgroup_atomic_increment(atomic_t *atomic) {
    __shared__ int workgroup_value;
    int tid = threadIdx.x;
    if (tid == 0) {
      // Thread0 fetches and increments the device-wide atomic.
      // This is a relaxed atomic, which is fine, as we provide the memory
      // ordering separately below with __builtin_amdgcn_fence, which usefully
      // allows us to specify the memory scope.
      int thread0_value = 1 + atomicAdd(atomic, 1);
      // Thread0 stores the value to the workgroup-wide shared location.
      workgroup_value = thread0_value;
      // Memory scope is "agent" because we need other workgroups to see this.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
    // Now all threads in the workgroup load the workgroup-wide shared value.
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return workgroup_value;
  }
  // Helper: maybe perform the final reduction of partial accumulators to the
  // destination C-matrix. The condition depends on atomic counter increments
  // telling us if we are the workgroup that is responsible for this among other
  // workgroups cooperating on a tile. It is thus non-deterministic which
  // workgroup will perform this final reduction, but the final C-matrix
  // elements must be deterministic. This is achieved by deterministically
  // looping over the partial accumulators in order of increasing K-dimension
  // indices, regardless of which partial accumulators were computed earlier
  // than others.
  __device__ static acc_array_t
  final_reduction(int M_outer, int N_outer, int K_outer,
                  float *local_accumulators_buffer, int tile_idx,
                  int cooperating_workgroup_start,
                  int cooperating_workgroup_end) {
    acc_array_t acc = {0};
    // Sum the partial accumulators in deterministic order of increasing
    // compute unit index, which makes to determinisic order of increasing K
    // dimension index.
    for (int other_workgroup = cooperating_workgroup_start;
         other_workgroup < cooperating_workgroup_end; ++other_workgroup) {
      int other_workgroup_idx_in_cooperating_group =
          other_workgroup - cooperating_workgroup_start;
      int other_local_accumulator_idx =
          other_workgroup_idx_in_cooperating_group +
          tile_idx * local_accumulators_count_per_tile(
                         MNKShape{M_outer, N_outer, K_outer});
      floatx4_t *other_accum_ptr = reinterpret_cast<floatx4_t *>(
          local_accumulators_buffer +
          other_local_accumulator_idx * static_M_tile * static_N_tile);
      for (int i = 0; i < num_acc_vec4s; ++i) {
        acc[i] += other_accum_ptr[256 * i + threadIdx.x];
      }
    }
    return acc;
  }
};

// A variant of
// MmtKernel_StreamK_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2 using a
// tree-reduction to better load-balance the accumulator reduction over multiple
// workgroups.
template <int MS, int NS>
class MmtKernel_StreamKTree_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  // Defining some values as constants as they will be needed in static methods
  // below.
  static constexpr Type static_A_type = Type::FP16;
  static constexpr Type static_B_type = Type::FP16;
  static constexpr Type static_C_type = Type::FP32;
  static constexpr int static_M_tile = MS * 16;
  static constexpr int static_N_tile = NS * 16;
  static constexpr int static_K_tile = 32;
  // Standard MmtKernel interface methods override describing element types.
  virtual Type A_type() const override { return static_A_type; }
  virtual Type B_type() const override { return static_B_type; }
  virtual Type C_type() const override { return static_C_type; }
  // Standard MmtKernel interface methods override describing tile shapes.
  virtual int M_tile() const override { return static_M_tile; }
  virtual int N_tile() const override { return static_N_tile; }
  virtual int K_tile() const override { return static_K_tile; }
  // Standard MmtKernel interface methods override describing tile layouts.
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 512 * mo + 128 * ((k % 16) / 4) + 8 * mi + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 512 * no + 128 * ((k % 16) / 4) + 8 * ni + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  // Standard MmtKernel interface method override describing number of threads.
  virtual int num_threads() const override { return 256; }
  // Additional constants and local helpers (static methods) for work-centric
  // grid and auxiliary buffer allocation.
  //
  // Maximum number of Compute Units that we aim to use on sufficiently large
  // problems. This value assumes a MI300X in SPX mode.
  static constexpr int MaxWorkgroups = 304;
  // Alignment to use when coalescing device buffers to avoid alignment-related
  // performance caveats.
  static constexpr int device_alignment = 128;
  // Alignment to use for atomics to avoid false-sharing effects.
  // Returns the number of Compute Units that we aim to use for the problem
  // shape with given `outer` shape (expressed in units of this kernel's tiles).
  static __device__ __host__ int get_workgroup_count(const MNKShape &outer) {
    int64_t total_iters = static_cast<int64_t>(outer.M) * outer.N * outer.K;
    return std::min(static_cast<int64_t>(MaxWorkgroups), total_iters);
  }
  static __device__ __host__ int get_tile_count(const MNKShape &outer) {
    return outer.M * outer.N;
  }
  // Returns the buffer bytes to allocate in the auxiliary buffer for the atomic
  // counters.
  using atomic_t = uint32_t;
  static __device__ __host__ int atomics_buffer_size(const MNKShape &outer) {
    return round_up_to_po2<device_alignment>(get_workgroup_count(outer) * 2 *
                                             sizeof(atomic_t));
  }
  // Returns the number of local accumulator buffers to allocate in the
  // auxiliary buffer, per tile. This value is a coarse upper bound, could be
  // refined to trim memory usage.
  static __device__ __host__ int
  local_accumulators_count_per_tile(const MNKShape &outer) {
    return 1 + ceil_div(get_workgroup_count(outer), get_tile_count(outer));
  }
  // Return the overall number of local accumulator buffers to allocate in the
  // auxiliary buffer. This value is a coarse upper bound, could be
  // refined to trim memory usage.
  static __device__ __host__ int
  local_accumulators_count(const MNKShape &outer) {
    return local_accumulators_count_per_tile(outer) * get_tile_count(outer);
  }
  // Return the overall auxiliary buffer bytes to allocate.
  static __device__ __host__ int
  local_accumulators_buffer_size(const MNKShape &outer) {
    return round_up_to_po2<device_alignment>(local_accumulators_count(outer) *
                                             static_M_tile * static_N_tile *
                                             type_size(static_C_type));
  }
  // Standard MmtKernel interface method override describing the work-centric
  // grid.
  virtual std::optional<dim3>
  get_work_centric_grid(const MNKShape &outer) const override {
    return {get_workgroup_count(outer)};
  }
  // Standard MmtKernel interface method override describing the auxiliary
  // buffer size to allocate.
  virtual int aux_buffer_size(const MNKShape &outer) const override {
    return atomics_buffer_size(outer) + local_accumulators_buffer_size(outer);
  }
  // Standard MmtKernel interface method override describing the actual device
  // kernel.
  virtual mmt_func_t mmt_func() const override { return run; };
  // Some typedefs used in the kernel implementation.
  using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  using float16x4_t =
      __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
  using float16x8_t =
      __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
  static constexpr int num_acc_vec4s = MS * NS / 4;
  using acc_array_t = std::array<floatx4_t, num_acc_vec4s>;
  // Device kernel implementation.
  __global__ __launch_bounds__(256) static void run(const void *A_data,
                                                    const void *B_data,
                                                    void *C_data,
                                                    void *aux_data, int M_outer,
                                                    int N_outer, int K_outer) {
    MNKShape outer{M_outer, N_outer, K_outer};
    // Get local accumulators buffer.
    char *aux_bytes = static_cast<char *>(aux_data);
    float *local_accumulators_buffer =
        reinterpret_cast<float *>(aux_bytes + atomics_buffer_size(outer));
    // General arithmetic just like in Algorithm 5 in
    // https://arxiv.org/pdf/2301.03598, except we use the term "workgroup"
    // instead of "CTA".
    int iters_per_tile = K_outer;
    int total_tiles = M_outer * N_outer;
    int total_iters = total_tiles * iters_per_tile;
    int workgroup_count = get_workgroup_count(outer);
    int workgroup = blockIdx.x;
    const int NUM_XCDS = 8;
    if (workgroup_count == MaxWorkgroups) {
      workgroup = (workgroup % NUM_XCDS) * (workgroup_count / NUM_XCDS) +
                  (workgroup / NUM_XCDS);
    }
    int iter_start = workgroup * total_iters / workgroup_count;
    int iter_end = (workgroup + 1) * total_iters / workgroup_count;
    int iter = iter_start;
    // Work-centric loop on iterations to be handled by this compute unit.
    while (iter < iter_end) {
      // Still same variable names as in Algorithm 5 in
      // https://arxiv.org/pdf/2301.03598.
      int tile_idx = iter / iters_per_tile;
      int tile_iter_start = tile_idx * iters_per_tile;
      int tile_iter_end = tile_iter_start + iters_per_tile;
      int local_iter = iter - tile_iter_start;
      int local_iter_end = std::min(tile_iter_end, iter_end) - tile_iter_start;
      if (local_iter == local_iter_end) {
        break;
      }
      // Map the 1D tile_idx to 2D (m_outer, n_outer) position in the C-matrix
      // tile space, using MxN-lexicographic order.
      int m_outer = tile_idx / N_outer;
      int n_outer = tile_idx - m_outer * N_outer;

      acc_array_t acc = mac_loop(A_data, B_data, K_outer, m_outer, n_outer,
                                 local_iter, local_iter_end);

      // The next step is specific to our wait-free atomic approach: we need to
      // know the interval of other compute units cooperating with the current
      // compute unit on the current tile.
      int cooperating_workgroup_start =
          ceil_div((tile_iter_start + 1) * workgroup_count, total_iters) - 1;
      int cooperating_workgroup_end =
          ceil_div(tile_iter_end * workgroup_count, total_iters);
      assert(workgroup >= cooperating_workgroup_start);
      assert(workgroup < cooperating_workgroup_end);
      int cooperating_workgroup_count =
          cooperating_workgroup_end - cooperating_workgroup_start;
      bool should_store_final_accumulator = true;

      // Hard case: multiple workgroups cooperating on this tile.
      if (cooperating_workgroup_count != 1) {
        int workgroup_idx_in_cooperating_group =
            workgroup - cooperating_workgroup_start;
        atomic_t *atomic_counter_base = reinterpret_cast<atomic_t *>(
            aux_bytes + cooperating_workgroup_start * 2 * sizeof(atomic_t));

        // Binary tree reduction.
        int tree_depth = 32 - std::countl_zero(static_cast<uint32_t>(
                                  cooperating_workgroup_count - 1));
        for (int tree_level = 0; tree_level < tree_depth; ++tree_level) {
          int bit = 1 << tree_level;
          // Determine the local accumulators to use. Clearing the low bits
          // ensures that the next level of the tree reduction will find the
          // accumulator that we wrote at this level, regardless of which
          // sibling wrote it.
          int local_accumulator_idx =
              (workgroup_idx_in_cooperating_group & ~(bit - 1)) +
              tile_idx * local_accumulators_count_per_tile(
                             MNKShape{M_outer, N_outer, K_outer});
          floatx4_t *local_accumulator = reinterpret_cast<floatx4_t *>(
              local_accumulators_buffer +
              local_accumulator_idx * static_M_tile * static_N_tile);
          int sibling_local_accumulator_idx =
              ((workgroup_idx_in_cooperating_group & ~(bit - 1)) ^ bit) +
              tile_idx * local_accumulators_count_per_tile(
                             MNKShape{M_outer, N_outer, K_outer});
          floatx4_t *sibling_local_accumulator = reinterpret_cast<floatx4_t *>(
              local_accumulators_buffer +
              sibling_local_accumulator_idx * static_M_tile * static_N_tile);
          int parent_idx =
              (workgroup_idx_in_cooperating_group & ~((bit << 1) - 1));
          // Check if we have a sibling.
          if ((parent_idx | bit) >= cooperating_workgroup_count) {
            // No sibling at this tree level. Continue to the next level.
            continue;
          }
          // Unconditionally store the local accumulator. This is redundant in
          // half of the cases (when this workgroup is the one performing the
          // reduction, using the accumulators already present in its VGPRs).
          // The point of doing that write unconditionally before the atomic
          // operation is that after the atomic operation we will have proven
          // that the write happened. Trying to save the redundant writes while
          // preserving the wait-free aspect would require more atomics.
          acc_store(acc, local_accumulator);
          // Determine the address of atomic counter to use. The `nudge` bit
          // is added to ensure that different tiles always use different
          // atomics. The problem is that otherwise, if two neighboring tiles
          // have the same cooperating_workgroup_start, then they have the same
          // atomic_counter_base. I believe 98% that this can only happen for a
          // pair of 2 neighboring tiles and therefore we only need that 2x
          // expansion of the space of atomic counters.
          int nudge = tile_idx & 1;
          atomic_t *atomic_counter =
              atomic_counter_base + 2 * parent_idx + nudge;
          // Perform the atomic operation to indicate that we have written our
          // partial accumulator.
          uint32_t counter_value = workgroup_atomic_or(atomic_counter, bit);
          // Test if the sibling workgroup had written their partial accumulator
          if (!(counter_value & bit)) {
            // If we are here, then our sibling has not written its partial
            // accumulator yet. We will break out of the tree reduction and let
            // our sibling continue down the tree.
            should_store_final_accumulator = false;
            break;
          }
          // Perform the current level of binary tree reduction.
          for (int i = 0; i < num_acc_vec4s; ++i) {
            acc[i] += sibling_local_accumulator[256 * i + threadIdx.x];
          }
        }
      }

      if (should_store_final_accumulator) {
        // Store the final accumulator to the destination.
        // Note: consumer fusions go here.
        acc_store(acc, static_cast<floatx4_t *>(C_data) +
                           MS * NS * 16 * 4 * (N_outer * m_outer + n_outer));

        // Reset atomic counters for the next kernel launch.
        atomic_t *atomic_counter_base = reinterpret_cast<atomic_t *>(
            aux_bytes + cooperating_workgroup_start * 2 * sizeof(atomic_t));
        for (int i = 0; i < cooperating_workgroup_count; ++i)
          atomic_counter_base[2 * i + (tile_idx & 1)] = 0;
      }

      iter = tile_iter_end;
    }
  }
  // Helper to store accumulators to memory.
  __device__ static void acc_store(acc_array_t acc, floatx4_t *ptr) {
    for (int i = 0; i < num_acc_vec4s; ++i) {
      ptr[256 * i + threadIdx.x] = acc[i];
    }
  }

  // Helper: inner MFMA loop. Similar to the MacLoop function given in
  // Algorithm 3 in https://arxiv.org/pdf/2301.03598.
  __device__ static acc_array_t mac_loop(const void *A_data, const void *B_data,
                                         int K_outer, int m_outer, int n_outer,
                                         int k_iter_start, int k_iter_end) {
    // Accumulator VGPRs.
    acc_array_t acc = {{0}};
    int tid = threadIdx.x;
    constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4;
    const float16x8_t *A_global =
        static_cast<const float16x8_t *>(A_data) +
        (m_outer * K_outer + k_iter_start) * A_tile_size_in_vec8;
    const float16x8_t *B_global =
        static_cast<const float16x8_t *>(B_data) +
        (n_outer * K_outer + k_iter_start) * B_tile_size_in_vec8;
    const float16x8_t *A_global_ptr = A_global + tid;
    const float16x8_t *B_global_ptr = B_global + tid;

    // Shared memory buffers for tiles of A and B matrices.
    __shared__ float16x8_t A_shared[A_tile_size_in_vec8];
    __shared__ float16x8_t B_shared[B_tile_size_in_vec8];

    // Inner loop on K dimenion.
    for (int k_iter = k_iter_start; k_iter < k_iter_end; ++k_iter) {
      // Load data from global to shared memory.
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();
      // Perform MFMA arithmetic on data in shared memory.
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          float16x8_t a = A_shared[64 * i + (tid % 64)];
          float16x8_t b = B_shared[256 * j + tid];
          float16x4_t a0 = (float16x4_t){a[0], a[1], a[2], a[3]};
          float16x4_t a1 = (float16x4_t){a[4], a[5], a[6], a[7]};
          float16x4_t b0 = (float16x4_t){b[0], b[1], b[2], b[3]};
          float16x4_t b1 = (float16x4_t){b[4], b[5], b[6], b[7]};
          floatx4_t acc0 = acc[i * (NS / 4) + j];
          acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a0, b0, acc0, 0, 0, 0);
          acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a1, b1, acc0, 0, 0, 0);
          acc[i * (NS / 4) + j] = acc0;
        }
      }
      __syncthreads();
    }
    return acc;
  }
  __device__ static uint32_t workgroup_atomic_or(atomic_t *atomic,
                                                 uint32_t value) {
    __shared__ int workgroup_value;
    int tid = threadIdx.x;
    if (tid == 0) {
      // Thread0 fetches and increments the device-wide atomic.
      // This is a relaxed atomic, which is fine, as we provide the memory
      // ordering separately below with __builtin_amdgcn_fence, which usefully
      // allows us to specify the memory scope.
      int thread0_value = atomicOr(atomic, value);
      // Thread0 stores the value to the workgroup-wide shared location.
      workgroup_value = thread0_value;
      // Memory scope is "agent" because we need other workgroups to see this.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
    // Now all threads in the workgroup load the workgroup-wide shared value.
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return workgroup_value;
  }
};

// Given an arbitrary 2D grid shape MxN (not necessarily square or power-of-2)
// and a linear index i, return a pair (x, y) where x is in [0 .. M-1] and
// y is in [0 .. N-1] and, as i ranges in [0 .. M*N-1], the pair (x, y) visits
// each position once, following a z-order curve to maximize locality.
static __device__ std::tuple<int, int> zorder(int M, int N, int i) {
  if (M == 1) {
    return {0, i};
  }
  if (N == 1) {
    return {i, 0};
  }
  int Mhalf = M / 2;
  int Nhalf = N / 2;
  int quadrant0 = Mhalf * Nhalf;
  int quadrant1 = Mhalf * N;
  int quadrant2 = quadrant1 + (M - Mhalf) * Nhalf;
  if (i < quadrant0) {
    auto [m, n] = zorder(Mhalf, Nhalf, i);
    return {m, n};
  }
  if (i < quadrant1) {
    auto [m, n] = zorder(Mhalf, N - Nhalf, i - quadrant0);
    return {m, n + Nhalf};
  }
  if (i < quadrant2) {
    auto [m, n] = zorder(M - Mhalf, Nhalf, i - quadrant1);
    return {m + Mhalf, n};
  }
  auto [m, n] = zorder(M - Mhalf, N - Nhalf, i - quadrant2);
  return {m + Mhalf, n + Nhalf};
}

template <int MS, int NS>
class
    MmtKernel_HybridStreamKTree_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  // Defining some values as constants as they will be needed in static methods
  // below.
  static constexpr Type static_A_type = Type::FP16;
  static constexpr Type static_B_type = Type::FP16;
  static constexpr Type static_C_type = Type::FP32;
  static constexpr int static_M_tile = MS * 16;
  static constexpr int static_N_tile = NS * 16;
  static constexpr int static_K_tile = 32;
  // Standard MmtKernel interface methods override describing element types.
  virtual Type A_type() const override { return static_A_type; }
  virtual Type B_type() const override { return static_B_type; }
  virtual Type C_type() const override { return static_C_type; }
  // Standard MmtKernel interface methods override describing tile shapes.
  virtual int M_tile() const override { return static_M_tile; }
  virtual int N_tile() const override { return static_N_tile; }
  virtual int K_tile() const override { return static_K_tile; }
  // Standard MmtKernel interface methods override describing tile layouts.
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 512 * mo + 128 * ((k % 16) / 4) + 8 * mi + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 512 * no + 128 * ((k % 16) / 4) + 8 * ni + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  // Standard MmtKernel interface method override describing number of threads.
  virtual int num_threads() const override { return 256; }
  // Additional constants and local helpers (static methods) for work-centric
  // grid and auxiliary buffer allocation.
  //
  // Maximum number of Compute Units that we aim to use on sufficiently large
  // problems. This value assumes a MI300X in SPX mode.
  static constexpr int MaxWorkgroups = 304;
  // Alignment to use when coalescing device buffers to avoid alignment-related
  // performance caveats.
  static constexpr int device_alignment = 128;
  // Alignment to use for atomics to avoid false-sharing effects.
  // Returns the number of Compute Units that we aim to use for the problem
  // shape with given `outer` shape (expressed in units of this kernel's tiles).
  static __device__ __host__ int get_workgroup_count(const MNKShape &outer) {
    (void)outer;
    return MaxWorkgroups;
    // int64_t total_iters = static_cast<int64_t>(outer.M) * outer.N * outer.K;
    // return std::min(static_cast<int64_t>(MaxWorkgroups), total_iters);
  }
  static __device__ __host__ int get_tile_count(const MNKShape &outer) {
    return outer.M * outer.N;
  }
  // Returns the buffer bytes to allocate in the auxiliary buffer for the atomic
  // counters.
  using atomic_t = uint32_t;
  static __device__ __host__ int atomics_buffer_size(const MNKShape &outer) {
    return round_up_to_po2<device_alignment>(get_workgroup_count(outer) *
                                             sizeof(atomic_t));
  }
  // Return the overall number of local accumulator buffers to allocate in the
  // auxiliary buffer. This value is a coarse upper bound, could be
  // refined to trim memory usage.
  static __device__ __host__ int
  local_accumulators_count(const MNKShape &outer) {
    return get_workgroup_count(outer);
  }
  // Return the overall auxiliary buffer bytes to allocate.
  static __device__ __host__ int
  local_accumulators_buffer_size(const MNKShape &outer) {
    return round_up_to_po2<device_alignment>(local_accumulators_count(outer) *
                                             static_M_tile * static_N_tile *
                                             type_size(static_C_type));
  }
  // Standard MmtKernel interface method override describing the work-centric
  // grid.
  virtual std::optional<dim3>
  get_work_centric_grid(const MNKShape &outer) const override {
    return {get_workgroup_count(outer)};
  }
  // Standard MmtKernel interface method override describing the auxiliary
  // buffer size to allocate.
  virtual int aux_buffer_size(const MNKShape &outer) const override {
    return atomics_buffer_size(outer) + local_accumulators_buffer_size(outer);
  }
  // Standard MmtKernel interface method override describing the actual device
  // kernel.
  virtual mmt_func_t mmt_func() const override { return run; };
  // Some typedefs used in the kernel implementation.
  using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  using float16x4_t =
      __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
  using float16x8_t =
      __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
  static constexpr int num_acc_vec4s = MS * NS / 4;
  using acc_array_t = std::array<floatx4_t, num_acc_vec4s>;
  // Device kernel implementation.
  __global__ __launch_bounds__(256) static void run(const void *A_data,
                                                    const void *B_data,
                                                    void *C_data,
                                                    void *aux_data, int M_outer,
                                                    int N_outer, int K_outer) {
    int workgroup = blockIdx.x;
    // Workgroup remapping for XCD locality
    constexpr int NUM_XCDS = 8;
    workgroup = (workgroup % NUM_XCDS) * (MaxWorkgroups / NUM_XCDS) +
                (workgroup / NUM_XCDS);
    MNKShape outer{M_outer, N_outer, K_outer};
    // Data-parallel part: try to process as many whole tiles as possible in
    // normal data-parallel mode.
    int total_tiles = M_outer * N_outer;
    int whole_tiles_per_workgroup = total_tiles / MaxWorkgroups;
    int whole_tiles = whole_tiles_per_workgroup * MaxWorkgroups;
    int this_workgroup_whole_tile_start = workgroup;
    int this_workgroup_whole_tile_end =
        this_workgroup_whole_tile_start +
        MaxWorkgroups * whole_tiles_per_workgroup;

    for (int tile_idx = this_workgroup_whole_tile_start;
         tile_idx < this_workgroup_whole_tile_end; tile_idx += MaxWorkgroups) {
      auto [m_outer, n_outer] = zorder(M_outer, N_outer, tile_idx);
      acc_array_t acc =
          mac_loop(A_data, B_data, K_outer, m_outer, n_outer, 0, K_outer);
      acc_store(acc, static_cast<floatx4_t *>(C_data) +
                         MS * NS * 16 * 4 * (N_outer * m_outer + n_outer));
    }

    // Early exit in the case of perfect sizes with no leftover tiles.
    if (whole_tiles == total_tiles) {
      return;
    }

    // There are leftover tiles, but by construction the number of leftover
    // tiles is less than the number of workgroups. Set up a simple form of
    // Stream-K to handle that cse.
    int streamk_tile_start = whole_tiles;
    int streamk_tile_end = total_tiles;
    int streamk_tile_count = streamk_tile_end - streamk_tile_start;
    int streamk_iter_count = streamk_tile_count * K_outer;
    // Clamp the number of workgroups to return early if a workgroup would have
    // no work to do. Also, round down the number of workgroups to the nearest
    // multiple of the number of tiles. This ensures that a given workgroup only
    // processes part of one tile. This considerably simplifies logic and
    // scheduling, as that ensures that all workgroups have roughly the same
    // amount of work, as opposed to having some workgroups dealing with
    // two tiles.
    int max_streamk_workgroups = std::min(streamk_iter_count, MaxWorkgroups);
    int streamk_workgroups_per_tile =
        max_streamk_workgroups / streamk_tile_count;
    int streamk_workgroups = streamk_workgroups_per_tile * streamk_tile_count;
    if (workgroup >= streamk_workgroups) {
      return;
    }
    int streamk_tile_idx = workgroup / streamk_workgroups_per_tile;
    int cooperating_workgroup_start =
        streamk_tile_idx * streamk_workgroups_per_tile;
    int cooperating_workgroup_end =
        cooperating_workgroup_start + streamk_workgroups_per_tile;
    int cooperating_workgroup_count =
        cooperating_workgroup_end - cooperating_workgroup_start;
    int workgroup_idx_in_cooperating_group =
        workgroup - cooperating_workgroup_start;
    int iter_start = workgroup_idx_in_cooperating_group * K_outer /
                     cooperating_workgroup_count;
    int iter_end = (workgroup_idx_in_cooperating_group + 1) * K_outer /
                   cooperating_workgroup_count;
    auto [m_outer, n_outer] =
        zorder(M_outer, N_outer, streamk_tile_start + streamk_tile_idx);
    // The arithmetic loop, computing the partial accumulators.
    acc_array_t acc = mac_loop(A_data, B_data, K_outer, m_outer, n_outer,
                               iter_start, iter_end);

    assert(workgroup >= cooperating_workgroup_start);
    assert(workgroup < cooperating_workgroup_end);

    char *aux_bytes = static_cast<char *>(aux_data);
    float *local_accumulators_buffer =
        reinterpret_cast<float *>(aux_bytes + atomics_buffer_size(outer));

    bool should_store_final_accumulator = true;

    // Hard case: multiple workgroups cooperating on this tile.
    if (cooperating_workgroup_count != 1) {
      atomic_t *atomic_counter_base = reinterpret_cast<atomic_t *>(
          aux_bytes + cooperating_workgroup_start * sizeof(atomic_t));

      // Binary tree reduction.
      int tree_depth = 32 - std::countl_zero(static_cast<uint32_t>(
                                cooperating_workgroup_count - 1));
      for (int tree_level = 0; tree_level < tree_depth; ++tree_level) {
        int bit = 1 << tree_level;
        // Determine the local accumulators to use. Clearing the low bits
        // is a convention for where we store the higher nodes: we recycle the
        // storage of the left leaf nodes.
        int local_idx = workgroup_idx_in_cooperating_group & ~(bit - 1);
        floatx4_t *local_accumulator = reinterpret_cast<floatx4_t *>(
            local_accumulators_buffer +
            (cooperating_workgroup_start + local_idx) * static_M_tile *
                static_N_tile);
        int sibling_idx = local_idx ^ bit;
        floatx4_t *sibling_accumulator = reinterpret_cast<floatx4_t *>(
            local_accumulators_buffer +
            (cooperating_workgroup_start + sibling_idx) * static_M_tile *
                static_N_tile);
        int parent_idx =
            (workgroup_idx_in_cooperating_group & ~((bit << 1) - 1));
        // Check if we have a sibling.
        if ((parent_idx | bit) >= cooperating_workgroup_count) {
          // No sibling at this tree level. Continue to the next level.
          continue;
        }
        // Unconditionally store the local accumulator. This is redundant in
        // half of the cases (when this workgroup is the one performing the
        // reduction, using the accumulators already present in its VGPRs).
        // The point of doing that write unconditionally before the atomic
        // operation is that after the atomic operation we will have proven
        // that the write happened. Trying to save the redundant writes while
        // preserving the wait-free aspect would require more atomics.
        acc_store(acc, local_accumulator);
        atomic_t *atomic_counter = atomic_counter_base + parent_idx;
        // Perform the atomic operation to indicate that we have written our
        // partial accumulator.
        uint32_t counter_value = workgroup_atomic_or(atomic_counter, bit);
        // Test if the sibling workgroup had written their partial accumulator
        if (!(counter_value & bit)) {
          // If we are here, then our sibling has not written its partial
          // accumulator yet. We will break out of the tree reduction and let
          // our sibling continue down the tree.
          should_store_final_accumulator = false;
          break;
        }
        // Perform the current level of binary tree reduction.
        for (int i = 0; i < num_acc_vec4s; ++i) {
          acc[i] += sibling_accumulator[256 * i + threadIdx.x];
        }
      }
    }

    if (should_store_final_accumulator) {
      // Store the final accumulator to the destination.
      // Note: consumer fusions go here.
      acc_store(acc, static_cast<floatx4_t *>(C_data) +
                         MS * NS * 16 * 4 * (N_outer * m_outer + n_outer));

      // Reset atomic counters for the next kernel launch.
      atomic_t *atomic_counter_base = reinterpret_cast<atomic_t *>(
          aux_bytes + cooperating_workgroup_start * sizeof(atomic_t));
      for (int i = 0; i < cooperating_workgroup_count; ++i)
        atomic_counter_base[i] = 0;
    }
  }
  // Helper to store accumulators to memory.
  __device__ static void acc_store(acc_array_t acc, floatx4_t *ptr) {
    for (int i = 0; i < num_acc_vec4s; ++i) {
      ptr[256 * i + threadIdx.x] = acc[i];
    }
  }

  // Helper: inner MFMA loop. Similar to the MacLoop function given in
  // Algorithm 3 in https://arxiv.org/pdf/2301.03598.
  __device__ static acc_array_t mac_loop(const void *A_data, const void *B_data,
                                         int K_outer, int m_outer, int n_outer,
                                         int k_iter_start, int k_iter_end) {
    // Accumulator VGPRs.
    acc_array_t acc = {{0}};
    int tid = threadIdx.x;
    constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4;
    const float16x8_t *A_global =
        static_cast<const float16x8_t *>(A_data) +
        (m_outer * K_outer + k_iter_start) * A_tile_size_in_vec8;
    const float16x8_t *B_global =
        static_cast<const float16x8_t *>(B_data) +
        (n_outer * K_outer + k_iter_start) * B_tile_size_in_vec8;
    const float16x8_t *A_global_ptr = A_global + tid;
    const float16x8_t *B_global_ptr = B_global + tid;

    // Shared memory buffers for tiles of A and B matrices.
    __shared__ float16x8_t A_shared[A_tile_size_in_vec8];
    __shared__ float16x8_t B_shared[B_tile_size_in_vec8];

    // Inner loop on K dimenion.
    for (int k_iter = k_iter_start; k_iter < k_iter_end; ++k_iter) {
      // Load data from global to shared memory.
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();
      // Perform MFMA arithmetic on data in shared memory.
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          float16x8_t a = A_shared[64 * i + (tid % 64)];
          float16x8_t b = B_shared[256 * j + tid];
          float16x4_t a0 = (float16x4_t){a[0], a[1], a[2], a[3]};
          float16x4_t a1 = (float16x4_t){a[4], a[5], a[6], a[7]};
          float16x4_t b0 = (float16x4_t){b[0], b[1], b[2], b[3]};
          float16x4_t b1 = (float16x4_t){b[4], b[5], b[6], b[7]};
          floatx4_t acc0 = acc[i * (NS / 4) + j];
          acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a0, b0, acc0, 0, 0, 0);
          acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a1, b1, acc0, 0, 0, 0);
          acc[i * (NS / 4) + j] = acc0;
        }
      }
      __syncthreads();
    }
    return acc;
  }
  __device__ static uint32_t workgroup_atomic_or(atomic_t *atomic,
                                                 uint32_t value) {
    __shared__ int workgroup_value;
    int tid = threadIdx.x;
    if (tid == 0) {
      // Thread0 fetches and increments the device-wide atomic.
      // This is a relaxed atomic, which is fine, as we provide the memory
      // ordering separately below with __builtin_amdgcn_fence, which usefully
      // allows us to specify the memory scope.
      int thread0_value = atomicOr(atomic, value);
      // Thread0 stores the value to the workgroup-wide shared location.
      workgroup_value = thread0_value;
      // Memory scope is "agent" because we need other workgroups to see this.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
    // Now all threads in the workgroup load the workgroup-wide shared value.
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return workgroup_value;
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_32x32x16i8_shared
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 32; }
  virtual int N_tile() const override { return NS * 32; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 32;
      int mo = m / 32;
      return 512 * mo + 256 * (k / 8) + 8 * mi + (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 32;
      int no = n / 32;
      return 512 * no + 256 * (k / 8) + 8 * ni + (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 32;
      int mo = m / 32;
      int ni = n % 32;
      int no = n / 32;
      return NS * 1024 * mo + 1024 * no + 16 * (((32 * (mi / 4)) % 64) + ni) +
             4 * (mi / 8) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int8x8_t = int64_t;
    using int32x16_t = __attribute__((__vector_size__(4 * 16))) int32_t;
    int32x16_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec8 = MS * 32 * 2;
    constexpr int B_tile_size_in_vec8 = NS * 32 * 2;

    const int8x8_t *A_global = static_cast<const int8x8_t *>(A_data) +
                               m_outer * K_outer * A_tile_size_in_vec8;
    const int8x8_t *B_global = static_cast<const int8x8_t *>(B_data) +
                               n_outer * K_outer * B_tile_size_in_vec8;
    const int8x8_t *A_global_ptr = A_global + tid;
    const int8x8_t *B_global_ptr = B_global + tid;

    __shared__ int8x8_t A_shared[A_tile_size_in_vec8];
    __shared__ int8x8_t B_shared[B_tile_size_in_vec8];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_i32_32x32x16_i8(
              A_shared[64 * i + (tid % 64)], B_shared[256 * j + tid], acc[i][j],
              0, 0, 0);
        }
      }
      __syncthreads();
    }

    int32x16_t *C_ptr = static_cast<int32x16_t *>(C_data) +
                        MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 32; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 512 * mo + 128 * (k / 8) + 8 * mi + (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 512 * no + 128 * (k / 8) + 8 * ni + (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 4 * (16 * (mi / 4) + ni) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int8x8_t = int64_t;
    using int32x4_t = __attribute__((__vector_size__(4 * 4))) int32_t;
    int32x4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4;

    const int8x8_t *A_global = static_cast<const int8x8_t *>(A_data) +
                               m_outer * K_outer * A_tile_size_in_vec8;
    const int8x8_t *B_global = static_cast<const int8x8_t *>(B_data) +
                               n_outer * K_outer * B_tile_size_in_vec8;
    const int8x8_t *A_global_ptr = A_global + tid;
    const int8x8_t *B_global_ptr = B_global + tid;

    __shared__ int8x8_t A_shared[A_tile_size_in_vec8];
    __shared__ int8x8_t B_shared[B_tile_size_in_vec8];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
              A_shared[64 * i + (tid % 64)], B_shared[256 * j + tid], acc[i][j],
              0, 0, 0);
        }
      }
      __syncthreads();
    }

    int32x4_t *C_ptr = static_cast<int32x4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

// The `smfmac` kernels are designed for skinny GEMMs (M=8)
// by using sparse MFMA intrinsics to perform a fully dense
// matmul. SMFMAC takes a 4:2-sparse operand A (where we
// specify which 2 of 4 elements are non-zero via a sparsity
// index) and a dense operand B. The "sparse trick" exploits
// this by splitting each dense A row across an even/odd lane
// pair, each holding half the K elements in a complementary
// 4:2 pattern:
//
//   even: K0 K1 _  _  K4 K5 _  _  K8  K9  _   _  ...
//   odd:  _  _  K2 K3 _  _  K6 K7 _   _  K10 K11 ...
//
// Together, the pair covers all K elements of one dense row.
// The SMFMAC instruction sees 16 "sparse" rows (from a
// 16x16x64 output tile) but physically only 8 dense rows
// exist. After the intrinsic, each even/odd pair holds
// partial sums over complementary K indices, which are
// reduced (D[0]+D[1], D[2]+D[3]) to recover the 8 complete
// dense output rows.
// Inspired by the implementation described in:
// https://huggingface.co/blog/mi300kernels
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  // Sparsity indices for the sparse trick
  static constexpr int SPARSITY_EVEN =
      0x00004444; // Selects positions 0,1 from each group of 4 8-bit elements
  static constexpr int SPARSITY_ODD =
      0x0000EEEE; // Selects positions 2,3 from each group of 4 8-bit elements
  static constexpr int NB_BANKS = 32;

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; }

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 64) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using i8x8 = __attribute__((__vector_size__(8))) int8_t;
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;
    using i32x4 = __attribute__((__vector_size__(16))) int32_t;
    using i16x4 = __attribute__((__vector_size__(8))) int16_t;

    union {
      i8x8 vec;
      int32_t i32[2];
    } a_reg;

    i32x4 acc[NS / 4] = {{0}};

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;

    // A tile: 8 rows × KS banks × 4 groups of K = KS*512 bytes
    // B tile: NS n-blocks × KS k-banks × 64 vec16 elements
    constexpr int A_TILE_VEC16 = 8 * KS * 4;
    constexpr int B_TILE_VEC16 = NS * KS * 64;

    const auto *A_base = static_cast<const i8x16 *>(A_data) +
                         blockIdx.x * K_outer * A_TILE_VEC16;
    const auto *B_base = static_cast<const i8x16 *>(B_data) +
                         blockIdx.y * K_outer * B_TILE_VEC16;
    const auto *B_global_ptr = B_base + tid;

    __shared__ i8x16 A_smem[A_TILE_VEC16];
    __shared__ i8x16 B_smem[B_TILE_VEC16];
    auto *A_smem_bytes = reinterpret_cast<int8_t *>(A_smem);

    const int a_m_pair = (lane % 16) / 2;
    const int odd_lane = lane % 2;
    const int a_k_half = lane / 16;
    const int a_lds_base = a_m_pair * (KS * 64) + a_k_half * 16;

    const int b_cons_n_col = lane % 16;
    const int b_cons_k_slice = lane / 16;

    const int sparsity_idx = (lane & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      if (tid < A_TILE_VEC16) {
        const i8x16 *A_ptr = A_base + k_outer * A_TILE_VEC16;
        A_smem[tid] = A_ptr[tid];
      }

      for (int i = 0; i < B_TILE_VEC16; i += 256) {
        B_smem[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_TILE_VEC16;
      __syncthreads();

      for (int kb = 0; kb < KS; ++kb) {
        // Both lanes in a pair load the same 8 bytes from LDS:
        //   all:  K0 K1 K2 K3 | K4 K5 K6 K7 | K8 K9 K10 K11 | K12 K13 K14 K15
        //
        // Viewed as two i16x4 halves (a_lo and a_hi):
        //   a_lo: {K0K1, K2K3, K4K5, K6K7}
        //   a_hi: {K8K9, K10K11, K12K13, K14K15}
        //
        // Shufflevector extracts even/odd i16 elements from each half:
        //   lo0 = {K0K1, K4K5}    hi0 = {K2K3, K6K7}
        //   lo1 = {K8K9, K12K13}  hi1 = {K10K11, K14K15}
        //
        // Even lanes select lo, odd lanes select hi:
        //   even: {K0K1, K4K5,  K8K9, K12K13}
        //   odd:  {K2K3, K6K7,  K10K11, K14K15}
        auto *a_ptr =
            reinterpret_cast<i16x4 *>(A_smem_bytes + a_lds_base + kb * 64);
        i16x4 a_lo = a_ptr[0];
        i16x4 a_hi = a_ptr[1];

        int lo0 =
            __builtin_bit_cast(int, __builtin_shufflevector(a_lo, a_lo, 0, 2));
        int hi0 =
            __builtin_bit_cast(int, __builtin_shufflevector(a_lo, a_lo, 1, 3));
        int lo1 =
            __builtin_bit_cast(int, __builtin_shufflevector(a_hi, a_hi, 0, 2));
        int hi1 =
            __builtin_bit_cast(int, __builtin_shufflevector(a_hi, a_hi, 1, 3));

        a_reg.i32[0] = !odd_lane ? lo0 : hi0;
        a_reg.i32[1] = !odd_lane ? lo1 : hi1;

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp;

          // Load B from LDS: 16 bytes for this lane's (n_col, k_slice)
          auto *b_ptr = reinterpret_cast<i8x16 *>(
              reinterpret_cast<int8_t *>(B_smem) + n_block * (KS * 1024) +
              b_cons_n_col * (KS * 64) + kb * 64 + b_cons_k_slice * 16);
          i8x16 b_reg = *b_ptr;

          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: fuse even + odd K partial sums to get the complete dot
    // product per row
    for (int j = 0; j < NS / 4; ++j) {
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      d[0] = d[0] + d[1]; // row0 complete
      d[1] = d[2] + d[3]; // row1 complete
    }

    constexpr int N_STRIDE = NS * 16;
    constexpr int TILE_ELEMS = 8 * N_STRIDE;

    const int m_pair = lane / 16;
    const int m0 = m_pair * 2;
    const int m1 = m0 + 1;
    const int n_col_out = lane % 16;

    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      TILE_ELEMS * (N_outer * blockIdx.x + blockIdx.y);

    for (int j = 0; j < NS / 4; ++j) {
      const int n = (j * 4 + warp) * 16 + n_col_out;
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      C_tile[m0 * N_STRIDE + n] = d[0];
      C_tile[m1 * N_STRIDE + n] = d[1];
    }
  }
};

// B-swizzle variant of the I8 SMFMAC sparse trick kernel.
// Instead of shuffling A to split even/odd K positions across lane pairs
// (which wastes 50% of A LDS bandwidth since both lanes load the same 16
// bytes), this variant keeps A ~natural and shuffles B instead.
//
// Each lane in a pair loads UNIQUE 8 bytes from A:
//   even lane: K0..K7    (8 bytes, sequential)
//   odd lane:  K8..K15   (8 bytes, sequential)
//
// B is BFI-shuffled to cross-mix groups at 16-bit granularity:
//   b_reg[0] = lo16(B_VGPR0) | hi16(B_VGPR2) = {B0,B1,B10,B11}
//   b_reg[1] = lo16(B_VGPR2) | hi16(B_VGPR0) = {B8,B9,B2,B3}
//   b_reg[2] = lo16(B_VGPR1) | hi16(B_VGPR3) = {B4,B5,B14,B15}
//   b_reg[3] = lo16(B_VGPR3) | hi16(B_VGPR1) = {B12,B13,B6,B7}
//
// Alternating sparsity indices select the correct positions:
//   even lane: 0xE4E4 -> group0:pos{0,1}, group1:pos{2,3}, ...
//   odd lane:  0x4E4E -> group0:pos{2,3}, group1:pos{0,1}, ...
//
// Odd lanes swap lo16/hi16 within each A VGPR (2x v_perm_b32) so that
// their K8..K15 data aligns with the B positions selected by 0x4E4E.
//
// Benefits over baseline:
//   - 50% less A LDS bandwidth (each lane loads unique 8 bytes vs shared 16)
//   - No shufflevector or conditional select on A
// Cost:
//   - 4 BFI instructions to construct B (was 0 in baseline)
//   - 2 v_perm_b32 for odd-lane A swap (was 4 shufflevector + 2 select)
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  // Alternating sparsity indices for the B-swizzle trick
  static constexpr int SPARSITY_EVEN =
      0x0000E4E4; // pos{0,1}, pos{2,3}, pos{0,1}, pos{2,3}
  static constexpr int SPARSITY_ODD =
      0x00004E4E; // pos{2,3}, pos{0,1}, pos{2,3}, pos{0,1}

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; }

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 64) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using i8x8 = __attribute__((__vector_size__(8))) int8_t;
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;
    using i32x4 = __attribute__((__vector_size__(16))) int32_t;

    union {
      i8x8 vec;
      int32_t i32[2];
    } a_reg;

    i32x4 acc[NS / 4] = {{0}};

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;

    // A tile: 8 rows × KS banks × 4 groups of K = KS*512 bytes
    // B tile: NS n-blocks × KS k-banks × 64 vec16 elements
    constexpr int A_TILE_VEC16 = 8 * KS * 4;
    constexpr int B_TILE_VEC16 = NS * KS * 64;

    const auto *A_base = static_cast<const i8x16 *>(A_data) +
                         blockIdx.x * K_outer * A_TILE_VEC16;
    const auto *B_base = static_cast<const i8x16 *>(B_data) +
                         blockIdx.y * K_outer * B_TILE_VEC16;
    const auto *B_global_ptr = B_base + tid;

    __shared__ i8x16 A_smem[A_TILE_VEC16];
    __shared__ i8x16 B_smem[B_TILE_VEC16];
    auto *A_smem_bytes = reinterpret_cast<int8_t *>(A_smem);

    const int a_m_pair = (lane % 16) / 2;
    const int odd_lane = lane % 2;
    const int a_k_half = lane / 16;
    // B-swizzle: each lane reads unique 8 bytes (even: K0-K7, odd: K8-K15)
    const int a_lds_base =
        a_m_pair * (KS * 64) + a_k_half * 16 + odd_lane * 8;

    const int b_cons_n_col = lane % 16;
    const int b_cons_k_slice = lane / 16;

    const int sparsity_idx = (lane & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      if (tid < A_TILE_VEC16) {
        const i8x16 *A_ptr = A_base + k_outer * A_TILE_VEC16;
        A_smem[tid] = A_ptr[tid];
      }

      for (int i = 0; i < B_TILE_VEC16; i += 256) {
        B_smem[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_TILE_VEC16;
      __syncthreads();

      for (int kb = 0; kb < KS; ++kb) {
        // A: load 8 bytes directly (no extraction shuffle needed)
        // Even lane gets K0..K7, odd lane gets K8..K15
        auto *a_ptr = reinterpret_cast<int32_t *>(
            A_smem_bytes + a_lds_base + kb * 64);
        a_reg.i32[0] = a_ptr[0];
        a_reg.i32[1] = a_ptr[1];

        // Odd lane: swap lo16/hi16 within each VGPR to align with
        // sparsity 0x4E4E which reads pos{2,3} first then pos{0,1}
        if (odd_lane) {
          a_reg.i32[0] = (((uint32_t)a_reg.i32[0]) >> 16) |
                         (((uint32_t)a_reg.i32[0]) << 16);
          a_reg.i32[1] = (((uint32_t)a_reg.i32[1]) >> 16) |
                         (((uint32_t)a_reg.i32[1]) << 16);
        }

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp;

          // B: load 16 bytes from LDS (same as baseline)
          auto *b_i32 = reinterpret_cast<int32_t *>(
              reinterpret_cast<int8_t *>(B_smem) + n_block * (KS * 1024) +
              b_cons_n_col * (KS * 64) + kb * 64 + b_cons_k_slice * 16);

          // B shuffle: cross-mix VGPRs 0↔2 and 1↔3 at 16-bit granularity
          // This places B values so that alternating sparsity indices
          // select the correct K positions for each lane.
          constexpr uint32_t LO16 = 0x0000FFFF;

          union {
            i8x16 vec;
            int32_t i32[4];
          } b_reg;
          b_reg.i32[0] = (b_i32[0] & LO16) | (b_i32[2] & ~LO16);
          b_reg.i32[1] = (b_i32[2] & LO16) | (b_i32[0] & ~LO16);
          b_reg.i32[2] = (b_i32[1] & LO16) | (b_i32[3] & ~LO16);
          b_reg.i32[3] = (b_i32[3] & LO16) | (b_i32[1] & ~LO16);

          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg.vec, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: fuse even + odd K partial sums to get the complete dot
    // product per row
    for (int j = 0; j < NS / 4; ++j) {
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      d[0] = d[0] + d[1]; // row0 complete
      d[1] = d[2] + d[3]; // row1 complete
    }

    constexpr int N_STRIDE = NS * 16;
    constexpr int TILE_ELEMS = 8 * N_STRIDE;

    const int m_pair = lane / 16;
    const int m0 = m_pair * 2;
    const int m1 = m0 + 1;
    const int n_col_out = lane % 16;

    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      TILE_ELEMS * (N_outer * blockIdx.x + blockIdx.y);

    for (int j = 0; j < NS / 4; ++j) {
      const int n = (j * 4 + warp) * 16 + n_col_out;
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      C_tile[m0 * N_STRIDE + n] = d[0];
      C_tile[m1 * N_STRIDE + n] = d[1];
    }
  }
};

// B-swizzle v2 variant of the I8 SMFMAC sparse trick kernel.
// This uses a different B register layout that eliminates the A swizzle
// entirely, saving 2 v_perm_b32 instructions per k_bank compared to v1.
//
// Like v1, each lane in a pair loads UNIQUE 8 bytes from A:
//   even lane: K0..K7    (8 bytes, sequential)
//   odd lane:  K8..K15   (8 bytes, sequential)
//
// Unlike v1, A data is used DIRECTLY -- no lo16/hi16 swap on odd lanes.
//
// B is constructed by mixing lo16 from K_lo VGPRs and hi16 from K_hi VGPRs
// within the same 16-byte load:
//   b_reg[0] = lo16(B_VGPR0) | (lo16(B_VGPR2) << 16) = {B0,B1,B8,B9}
//   b_reg[1] = (B_VGPR0 >> 16) | hi16(B_VGPR2)       = {B2,B3,B10,B11}
//   b_reg[2] = lo16(B_VGPR1) | (lo16(B_VGPR3) << 16) = {B4,B5,B12,B13}
//   b_reg[3] = (B_VGPR1 >> 16) | hi16(B_VGPR3)       = {B6,B7,B14,B15}
//
// This layout places K_lo data (B0-B7) in lo16 of each B VGPR and K_hi
// data (B8-B15) in hi16. Standard sparsity indices select correctly:
//   even lane: 0x4444 -> pos{0,1} = lo16 -> matches K0..K7 A data
//   odd lane:  0xEEEE -> pos{2,3} = hi16 -> matches K8..K15 A data
//
// Benefits over bswizzle v1:
//   - No A swizzle at all (saves 2 v_perm_b32 per k_bank)
//   - Same 4 BFI instructions for B construction
//   - Simpler sparsity constants (uniform 0x4444/0xEEEE, no alternation)
// Cost:
//   - 4 BFI instructions to construct B (same as v1)
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle_v2
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  // Standard sparsity indices (same as baseline)
  static constexpr int SPARSITY_EVEN =
      0x00004444; // pos{0,1} = lo16 of each B VGPR -> selects K_lo B values
  static constexpr int SPARSITY_ODD =
      0x0000EEEE; // pos{2,3} = hi16 of each B VGPR -> selects K_hi B values

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; }

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 64) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using i8x8 = __attribute__((__vector_size__(8))) int8_t;
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;
    using i32x4 = __attribute__((__vector_size__(16))) int32_t;

    union {
      i8x8 vec;
      int32_t i32[2];
    } a_reg;

    i32x4 acc[NS / 4] = {{0}};

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;

    // A tile: 8 rows x KS banks x 4 groups of K = KS*512 bytes
    // B tile: NS n-blocks x KS k-banks x 64 vec16 elements
    constexpr int A_TILE_VEC16 = 8 * KS * 4;
    constexpr int B_TILE_VEC16 = NS * KS * 64;

    const auto *A_base = static_cast<const i8x16 *>(A_data) +
                         blockIdx.x * K_outer * A_TILE_VEC16;
    const auto *B_base = static_cast<const i8x16 *>(B_data) +
                         blockIdx.y * K_outer * B_TILE_VEC16;
    const auto *B_global_ptr = B_base + tid;

    __shared__ i8x16 A_smem[A_TILE_VEC16];
    __shared__ i8x16 B_smem[B_TILE_VEC16];
    auto *A_smem_bytes = reinterpret_cast<int8_t *>(A_smem);

    const int a_m_pair = (lane % 16) / 2;
    const int odd_lane = lane % 2;
    const int a_k_half = lane / 16;
    // Same A LDS layout as bswizzle v1: each lane reads unique 8 bytes
    const int a_lds_base =
        a_m_pair * (KS * 64) + a_k_half * 16 + odd_lane * 8;

    const int b_cons_n_col = lane % 16;
    const int b_cons_k_slice = lane / 16;

    const int sparsity_idx = (lane & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      if (tid < A_TILE_VEC16) {
        const i8x16 *A_ptr = A_base + k_outer * A_TILE_VEC16;
        A_smem[tid] = A_ptr[tid];
      }

      for (int i = 0; i < B_TILE_VEC16; i += 256) {
        B_smem[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_TILE_VEC16;
      __syncthreads();

      for (int kb = 0; kb < KS; ++kb) {
        // A: load 8 bytes directly -- NO swizzle needed!
        // Even lane gets K0..K7, odd lane gets K8..K15.
        // Data is used as-is because the B layout and sparsity indices
        // are arranged so that pos{0,1} (lo16) maps to K_lo B values
        // and pos{2,3} (hi16) maps to K_hi B values.
        auto *a_ptr = reinterpret_cast<int32_t *>(
            A_smem_bytes + a_lds_base + kb * 64);
        a_reg.i32[0] = a_ptr[0];
        a_reg.i32[1] = a_ptr[1];

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp;

          // B: load 16 bytes from LDS (same as baseline/bswizzle v1)
          auto *b_i32 = reinterpret_cast<int32_t *>(
              reinterpret_cast<int8_t *>(B_smem) + n_block * (KS * 1024) +
              b_cons_n_col * (KS * 64) + kb * 64 + b_cons_k_slice * 16);

          // B construction via V_PERM_B32: byte-permute from two source VGPRs
          // perm(S0, S1, sel) forms tmp={S0,S1} (S0 high, S1 low)
          // and picks 4 bytes per selector.
          //
          // Standard layout from LDS:
          //   b_i32[0] = {B0,B1,B2,B3}      (K=0..3)
          //   b_i32[1] = {B4,B5,B6,B7}      (K=4..7)
          //   b_i32[2] = {B8,B9,B10,B11}    (K=8..11)
          //   b_i32[3] = {B12,B13,B14,B15}  (K=12..15)
          //
          // After perm:
          //   v0 = perm(b[2], b[0], SEL_LO) = {B0,B1,B8,B9}
          //   v1 = perm(b[2], b[0], SEL_HI) = {B2,B3,B10,B11}
          //   v2 = perm(b[3], b[1], SEL_LO) = {B4,B5,B12,B13}
          //   v3 = perm(b[3], b[1], SEL_HI) = {B6,B7,B14,B15}
          constexpr uint32_t SEL_LO = 0x05040100; // lo16(S0) | lo16(S1)
          constexpr uint32_t SEL_HI = 0x07060302; // hi16(S0) | hi16(S1)

          union {
            i8x16 vec;
            int32_t i32[4];
          } b_reg;
          b_reg.i32[0] = __builtin_amdgcn_perm(b_i32[2], b_i32[0], SEL_LO);
          b_reg.i32[1] = __builtin_amdgcn_perm(b_i32[2], b_i32[0], SEL_HI);
          b_reg.i32[2] = __builtin_amdgcn_perm(b_i32[3], b_i32[1], SEL_LO);
          b_reg.i32[3] = __builtin_amdgcn_perm(b_i32[3], b_i32[1], SEL_HI);

          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg.vec, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: fuse even + odd K partial sums to get the complete dot
    // product per row
    for (int j = 0; j < NS / 4; ++j) {
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      d[0] = d[0] + d[1]; // row0 complete
      d[1] = d[2] + d[3]; // row1 complete
    }

    constexpr int N_STRIDE = NS * 16;
    constexpr int TILE_ELEMS = 8 * N_STRIDE;

    const int m_pair = lane / 16;
    const int m0 = m_pair * 2;
    const int m1 = m0 + 1;
    const int n_col_out = lane % 16;

    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      TILE_ELEMS * (N_outer * blockIdx.x + blockIdx.y);

    for (int j = 0; j < NS / 4; ++j) {
      const int n = (j * 4 + warp) * 16 + n_col_out;
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      C_tile[m0 * N_STRIDE + n] = d[0];
      C_tile[m1 * N_STRIDE + n] = d[1];
    }
  }
};

// B-swizzle v3 variant of the I8 SMFMAC sparse trick kernel.
// Like v2, this eliminates A swizzle entirely. The difference is in how
// odd-indexed B VGPRs (v1, v3) are constructed:
//
//   v2: v1 = hi16(b[0]) | lo16_hi(b[2])  = {B2,B3,  B10,B11}
//   v3: v1 = hi16(b[2]) | lo16_hi(b[0])  = {B10,B11, B2,B3}   (halves swapped)
//
// This swaps which half of the odd-indexed VGPRs contains the LO vs HI data
// from each source, which changes the required sparsity from uniform (0x4444/
// 0xEEEE) to alternating (0xE4E4/0x4E4E) — but the A data still loads
// sequentially because the alternation in B content and sparsity cancel out.
//
// B register layout:
//   v0 = {B0,B1,  B8,B9}     [LO | LO]   sparsity even=0x4, odd=0xE
//   v1 = {B10,B11, B2,B3}    [HI | HI]   sparsity even=0xE, odd=0x4
//   v2 = {B4,B5,  B12,B13}   [LO | LO]   sparsity even=0x4, odd=0xE
//   v3 = {B14,B15, B6,B7}    [HI | HI]   sparsity even=0xE, odd=0x4
//
// B construction:
//   v0 = (b[0] & LO16) | (b[2] << 16)       -- same as v2
//   v1 = (b[2] >> 16)  | (b[0] & HI16)       -- v2 uses b[0]>>16 | b[2]&HI16
//   v2 = (b[1] & LO16) | (b[3] << 16)       -- same as v2
//   v3 = (b[3] >> 16)  | (b[1] & HI16)       -- v2 uses b[1]>>16 | b[3]&HI16
//
// Same instruction count as v2 (both use AND+SHIFT+OR, neither can use BFI).
// The purpose of this variant is to compare generated assembly — in particular,
// whether alternating vs uniform sparsity or the different source register
// pairing affects the compiler's instruction selection (e.g., v_perm_b32).
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle_v3
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  // Alternating sparsity indices (same as v1, but A data is still sequential)
  static constexpr int SPARSITY_EVEN =
      0x0000E4E4; // pos{0,1}, pos{2,3}, pos{0,1}, pos{2,3}
  static constexpr int SPARSITY_ODD =
      0x00004E4E; // pos{2,3}, pos{0,1}, pos{2,3}, pos{0,1}

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; }

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 64) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using i8x8 = __attribute__((__vector_size__(8))) int8_t;
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;
    using i32x4 = __attribute__((__vector_size__(16))) int32_t;

    union {
      i8x8 vec;
      int32_t i32[2];
    } a_reg;

    i32x4 acc[NS / 4] = {{0}};

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;

    constexpr int A_TILE_VEC16 = 8 * KS * 4;
    constexpr int B_TILE_VEC16 = NS * KS * 64;

    const auto *A_base = static_cast<const i8x16 *>(A_data) +
                         blockIdx.x * K_outer * A_TILE_VEC16;
    const auto *B_base = static_cast<const i8x16 *>(B_data) +
                         blockIdx.y * K_outer * B_TILE_VEC16;
    const auto *B_global_ptr = B_base + tid;

    __shared__ i8x16 A_smem[A_TILE_VEC16];
    __shared__ i8x16 B_smem[B_TILE_VEC16];
    auto *A_smem_bytes = reinterpret_cast<int8_t *>(A_smem);

    const int a_m_pair = (lane % 16) / 2;
    const int odd_lane = lane % 2;
    const int a_k_half = lane / 16;
    const int a_lds_base =
        a_m_pair * (KS * 64) + a_k_half * 16 + odd_lane * 8;

    const int b_cons_n_col = lane % 16;
    const int b_cons_k_slice = lane / 16;

    const int sparsity_idx = (lane & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      if (tid < A_TILE_VEC16) {
        const i8x16 *A_ptr = A_base + k_outer * A_TILE_VEC16;
        A_smem[tid] = A_ptr[tid];
      }

      for (int i = 0; i < B_TILE_VEC16; i += 256) {
        B_smem[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_TILE_VEC16;
      __syncthreads();

      for (int kb = 0; kb < KS; ++kb) {
        // A: load 8 bytes directly -- NO swizzle needed!
        // Sequential K data maps correctly through alternating sparsity.
        auto *a_ptr = reinterpret_cast<int32_t *>(
            A_smem_bytes + a_lds_base + kb * 64);
        a_reg.i32[0] = a_ptr[0];
        a_reg.i32[1] = a_ptr[1];

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp;

          auto *b_i32 = reinterpret_cast<int32_t *>(
              reinterpret_cast<int8_t *>(B_smem) + n_block * (KS * 1024) +
              b_cons_n_col * (KS * 64) + kb * 64 + b_cons_k_slice * 16);

          // B construction via V_PERM_B32: v3 layout
          // perm(S0, S1, sel) forms tmp={S0,S1} (S0 high, S1 low)
          //
          // Standard B from LDS:
          //   b_i32[0] = {B0,B1,B2,B3}      (K=0..3)
          //   b_i32[1] = {B4,B5,B6,B7}      (K=4..7)
          //   b_i32[2] = {B8,B9,B10,B11}    (K=8..11)
          //   b_i32[3] = {B12,B13,B14,B15}  (K=12..15)
          //
          // After perm:
          //   v0 = perm(b[2], b[0], SEL_LO)      = {B0,B1,B8,B9}
          //   v1 = perm(b[2], b[0], SEL_HI_SWAP)  = {B10,B11,B2,B3}
          //   v2 = perm(b[3], b[1], SEL_LO)      = {B4,B5,B12,B13}
          //   v3 = perm(b[3], b[1], SEL_HI_SWAP)  = {B14,B15,B6,B7}
          constexpr uint32_t SEL_LO = 0x05040100;      // lo16(S0) | lo16(S1)
          constexpr uint32_t SEL_HI_SWAP = 0x03020706;  // hi16(S1) | hi16(S0)

          union {
            i8x16 vec;
            int32_t i32[4];
          } b_reg;
          b_reg.i32[0] = __builtin_amdgcn_perm(b_i32[2], b_i32[0], SEL_LO);
          b_reg.i32[1] = __builtin_amdgcn_perm(b_i32[2], b_i32[0], SEL_HI_SWAP);
          b_reg.i32[2] = __builtin_amdgcn_perm(b_i32[3], b_i32[1], SEL_LO);
          b_reg.i32[3] = __builtin_amdgcn_perm(b_i32[3], b_i32[1], SEL_HI_SWAP);

          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg.vec, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: fuse even + odd K partial sums
    for (int j = 0; j < NS / 4; ++j) {
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      d[0] = d[0] + d[1];
      d[1] = d[2] + d[3];
    }

    constexpr int N_STRIDE = NS * 16;
    constexpr int TILE_ELEMS = 8 * N_STRIDE;

    const int m_pair = lane / 16;
    const int m0 = m_pair * 2;
    const int m1 = m0 + 1;
    const int n_col_out = lane % 16;

    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      TILE_ELEMS * (N_outer * blockIdx.x + blockIdx.y);

    for (int j = 0; j < NS / 4; ++j) {
      const int n = (j * 4 + warp) * 16 + n_col_out;
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      C_tile[m0 * N_STRIDE + n] = d[0];
      C_tile[m1 * N_STRIDE + n] = d[1];
    }
  }
};

template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  static constexpr int SPARSITY_EVEN = 0x00000044;
  static constexpr int SPARSITY_ODD = 0x000000EE;
  static constexpr int NB_BANKS = 32;

  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 32; }

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 32) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 512) + n_col * (KS * 32) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using f16x4 =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using f16x8 =
        __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    union {
      f16x4 vec;
      int32_t i32[2];
    } a_reg;

    f32x4 acc[NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    int warp_id = tid / 64;

    constexpr int A_tile_size_in_vec16 = 8 * KS * 4;
    constexpr int B_tile_size_in_vec16 = NS * KS * 64;

    const f16x8 *A_global = static_cast<const f16x8 *>(A_data) +
                            m_outer * K_outer * A_tile_size_in_vec16;
    const f16x8 *B_global = static_cast<const f16x8 *>(B_data) +
                            n_outer * K_outer * B_tile_size_in_vec16;
    const f16x8 *B_global_ptr = B_global + tid;

    __shared__ f16x8 A_shared[A_tile_size_in_vec16];
    __shared__ f16x8 B_shared[B_tile_size_in_vec16];
    auto *A_shared_bytes = reinterpret_cast<int8_t *>(A_shared);

    int a_m_row = (lane_id % 16) / 2;
    int odd_lane = lane_id % 2;
    int a_k_range = lane_id / 16;
    int a_lds_base = a_m_row * (KS * 64) + a_k_range * 16;

    int n_col = lane_id % 16;
    int k_slice = lane_id / 16;

    int sparsity_idx = (lane_id & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      const f16x8 *A_ptr = A_global + k_outer * A_tile_size_in_vec16;
      if (tid < A_tile_size_in_vec16) {
        A_shared[tid] = A_ptr[tid];
      }

      for (int i = 0; i < B_tile_size_in_vec16; i += 256) {
        B_shared[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      for (int k_bank = 0; k_bank < KS; ++k_bank) {
        auto *a_ptr = reinterpret_cast<f16x8 *>(A_shared_bytes + a_lds_base +
                                                k_bank * 64);
        auto *a_i32 = reinterpret_cast<int32_t *>(a_ptr);
        // even: {K0, K1,  K4, K5}
        // odd: {K2, K3,  K6, K7}
        a_reg.i32[0] = odd_lane ? a_i32[1] : a_i32[0];
        a_reg.i32[1] = odd_lane ? a_i32[3] : a_i32[2];

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp_id;

          int8_t *b_cons = reinterpret_cast<int8_t *>(B_shared) +
                           n_block * (KS * 1024) + n_col * (KS * 64) +
                           k_bank * 64 + k_slice * 16;
          f16x8 b_reg = *reinterpret_cast<f16x8 *>(b_cons);

          acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
              a_reg.vec, b_reg, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    for (int j = 0; j < NS / 4; ++j) {
      float *regD = reinterpret_cast<float *>(&acc[j]);
      regD[0] = regD[0] + regD[1];
      regD[1] = regD[2] + regD[3];
    }

    constexpr int N_stride = NS * 16;
    int m_pair = lane_id / 16;
    int m0 = m_pair * 2;
    int m1 = m_pair * 2 + 1;

    constexpr int tile_size = 8 * NS * 16;
    float *C_tile = static_cast<float *>(C_data) +
                    tile_size * (N_outer * m_outer + n_outer);

    for (int j = 0; j < NS / 4; ++j) {
      int n_block = j * 4 + warp_id;
      int n = n_block * 16 + n_col;

      float *regD = reinterpret_cast<float *>(&acc[j]);
      C_tile[m0 * N_stride + n] = regD[0];
      C_tile[m1 * N_stride + n] = regD[1];
    }
  }
};

// Unique-lane-data variant of the I8 SMFMAC sparse trick kernel.
// Instead of even/odd lanes reading the SAME 128-bit A data from LDS,
// each lane reads UNIQUE data: even lane gets K_lo (K0..K15),
// odd lane gets K_hi (K16..K31). B loads two 128-bit chunks per lane
// and mixes them at 16-bit granularity so that sparsity positions 0,1
// (even lanes) select B values from K_lo and positions 2,3 (odd lanes)
// select from K_hi. This doubles K_tile (KS * 128 instead of KS * 64)
// and issues 2 SMFMAC instructions per k_bank per N-block-group.
//
// A extraction: standard shufflevector (no BFI needed).
// B mixing: 16-bit level -- lo16 from blo, hi16 from bhi per VGPR.
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  // Sparsity indices: same as baseline
  static constexpr int SPARSITY_EVEN =
      0x00004444; // Selects positions 0,1 from each group of 4 8-bit elements
  static constexpr int SPARSITY_ODD =
      0x0000EEEE; // Selects positions 2,3 from each group of 4 8-bit elements

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 128; } // 2x baseline

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 128) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 2048) + n_col * (KS * 128) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using i8x8 = __attribute__((__vector_size__(8))) int8_t;
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;
    using i32x4 = __attribute__((__vector_size__(16))) int32_t;
    using i16x2 = __attribute__((__vector_size__(4))) int16_t;
    using i16x4 = __attribute__((__vector_size__(8))) int16_t;

    union {
      i8x8 vec;
      int32_t i32[2];
    } a_reg;

    i32x4 acc[NS / 4] = {{0}};

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;

    // A tile: 8 rows x KS*128 bytes = 8*KS*8 vec16 elements
    // B tile: NS n-blocks x KS k-banks x 128 vec16 elements
    constexpr int A_TILE_VEC16 = 8 * KS * 8;
    constexpr int B_TILE_VEC16 = NS * KS * 128;

    const auto *A_base = static_cast<const i8x16 *>(A_data) +
                         blockIdx.x * K_outer * A_TILE_VEC16;
    const auto *B_base = static_cast<const i8x16 *>(B_data) +
                         blockIdx.y * K_outer * B_TILE_VEC16;
    const auto *B_global_ptr = B_base + tid;

    __shared__ i8x16 A_smem[A_TILE_VEC16];
    __shared__ i8x16 B_smem[B_TILE_VEC16];
    auto *A_smem_bytes = reinterpret_cast<int8_t *>(A_smem);

    const int a_m_pair = (lane % 16) / 2;
    const int odd_lane = lane % 2;
    const int a_k_half = lane / 16;
    // Unique data: even/odd lanes read from different 16-byte regions
    const int a_lds_base =
        a_m_pair * (KS * 128) + a_k_half * 32 + odd_lane * 16;

    const int b_cons_n_col = lane % 16;
    const int b_cons_k_slice = lane / 16;

    const int sparsity_idx = (lane & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      // Load A tile: global -> shared
      const i8x16 *A_ptr = A_base + k_outer * A_TILE_VEC16;
      for (int i = tid; i < A_TILE_VEC16; i += 256) {
        A_smem[i] = A_ptr[i];
      }

      // Load B tile: global -> shared
      for (int i = 0; i < B_TILE_VEC16; i += 256) {
        B_smem[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_TILE_VEC16;
      __syncthreads();

      for (int kb = 0; kb < KS; ++kb) {
        // --- A extraction: standard shufflevector (same as baseline) ---
        // Even lane reads K0..K15 (unique), odd lane reads K16..K31.
        // Each lane applies the baseline shuffle to extract 8 of 16 bytes.
        //
        // Raw 16 bytes viewed as two i16x4:
        //   a_lo = {K0K1, K2K3, K4K5, K6K7}
        //   a_hi = {K8K9, K10K11, K12K13, K14K15}
        //
        // Shufflevector extracts even/odd i16 elements:
        //   lo0 = {K0K1, K4K5}  hi0 = {K2K3, K6K7}
        //   lo1 = {K8K9, K12K13}  hi1 = {K10K11, K14K15}
        //
        // For SMFMAC set 1 (sparsity pos 0,1): use lo0, lo1
        //   a_reg = {K0,K1,K4,K5, K8,K9,K12,K13}
        // For SMFMAC set 2 (sparsity pos 0,1): use hi0, hi1
        //   a_reg = {K2,K3,K6,K7, K10,K11,K14,K15}
        auto *a_ptr =
            reinterpret_cast<i16x4 *>(A_smem_bytes + a_lds_base + kb * 128);
        i16x4 a_lo = a_ptr[0];
        i16x4 a_hi = a_ptr[1];

        int lo0 =
            __builtin_bit_cast(int, __builtin_shufflevector(a_lo, a_lo, 0, 2));
        int hi0 =
            __builtin_bit_cast(int, __builtin_shufflevector(a_lo, a_lo, 1, 3));
        int lo1 =
            __builtin_bit_cast(int, __builtin_shufflevector(a_hi, a_hi, 0, 2));
        int hi1 =
            __builtin_bit_cast(int, __builtin_shufflevector(a_hi, a_hi, 1, 3));

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp;

          // --- B: two 128-bit loads (K_lo and K_hi) ---
          // blo = 16 bytes = B[K_lo_0..K_lo_15] for this lane's N column
          // bhi = 16 bytes = B[K_hi_0..K_hi_15]
          int b_byte_offset = n_block * (KS * 2048) +
                              b_cons_n_col * (KS * 128) + kb * 128 +
                              b_cons_k_slice * 32;
          auto *blo = reinterpret_cast<const i16x2 *>(
              reinterpret_cast<int8_t *>(B_smem) + b_byte_offset);
          auto *bhi = reinterpret_cast<const i16x2 *>(
              reinterpret_cast<int8_t *>(B_smem) + b_byte_offset + 16);

          // --- B mixing at 16-bit granularity via shufflevector ---
          // Each B VGPR (32 bits = 4 i8) holds one group of 4 K positions.
          // Sparsity pos 0,1 = bytes 0,1 (lo16); pos 2,3 = bytes 2,3 (hi16).
          // Even lane A needs B[K_lo] at pos 0,1 -> lo16 from blo.
          // Odd lane A needs B[K_hi] at pos 2,3 -> hi16 from bhi.
          //
          // b_reg_1.v[n] = shuffle(blo[n], bhi[n], {0,2})
          //   = {B_lo[4n], B_lo[4n+1], B_hi[4n], B_hi[4n+1]}
          //
          // b_reg_2.v[n] = shuffle(blo[n], bhi[n], {1,3})
          //   = {B_lo[4n+2], B_lo[4n+3], B_hi[4n+2], B_hi[4n+3]}
          union {
            i8x16 vec;
            int32_t i32[4];
          } b_reg_1, b_reg_2;

          // b_reg_1: {blo[n].lo16, bhi[n].lo16} for n=0..3
          b_reg_1.i32[0] = __builtin_bit_cast(int, __builtin_shufflevector(blo[0], bhi[0], 0, 2));
          b_reg_1.i32[1] = __builtin_bit_cast(int, __builtin_shufflevector(blo[1], bhi[1], 0, 2));
          b_reg_1.i32[2] = __builtin_bit_cast(int, __builtin_shufflevector(blo[2], bhi[2], 0, 2));
          b_reg_1.i32[3] = __builtin_bit_cast(int, __builtin_shufflevector(blo[3], bhi[3], 0, 2));

          // b_reg_2: {blo[n].hi16, bhi[n].hi16} for n=0..3
          b_reg_2.i32[0] = __builtin_bit_cast(int, __builtin_shufflevector(blo[0], bhi[0], 1, 3));
          b_reg_2.i32[1] = __builtin_bit_cast(int, __builtin_shufflevector(blo[1], bhi[1], 1, 3));
          b_reg_2.i32[2] = __builtin_bit_cast(int, __builtin_shufflevector(blo[2], bhi[2], 1, 3));
          b_reg_2.i32[3] = __builtin_bit_cast(int, __builtin_shufflevector(blo[3], bhi[3], 1, 3));

          // SMFMAC 1: A set 1 (pos 0,1 values) x b_reg_1
          // Even lane: K0*B0 + K1*B1 + K4*B4 + ... + K13*B13
          // Odd lane: K16*B16 + K17*B17 + ... + K29*B29
          a_reg.i32[0] = lo0;
          a_reg.i32[1] = lo1;
          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg_1.vec, acc[j], sparsity_idx, 0, 0);

          // SMFMAC 2: A set 2 (pos 2,3 values) x b_reg_2
          // Even lane: K2*B2 + K3*B3 + K6*B6 + ... + K15*B15
          // Odd lane: K18*B18 + K19*B19 + ... + K31*B31
          a_reg.i32[0] = hi0;
          a_reg.i32[1] = hi1;
          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg_2.vec, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: same 2-way reduction as baseline
    for (int j = 0; j < NS / 4; ++j) {
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      d[0] = d[0] + d[1]; // row0 complete
      d[1] = d[2] + d[3]; // row1 complete
    }

    constexpr int N_STRIDE = NS * 16;
    constexpr int TILE_ELEMS = 8 * N_STRIDE;

    const int m_pair = lane / 16;
    const int m0 = m_pair * 2;
    const int m1 = m0 + 1;
    const int n_col_out = lane % 16;

    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      TILE_ELEMS * (N_outer * blockIdx.x + blockIdx.y);

    for (int j = 0; j < NS / 4; ++j) {
      const int n = (j * 4 + warp) * 16 + n_col_out;
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      C_tile[m0 * N_STRIDE + n] = d[0];
      C_tile[m1 * N_STRIDE + n] = d[1];
    }
  }
};

// Unique-lane-data v2: eliminates __builtin_shufflevector from A extraction.
//
// In the existing uniquedata kernel, each lane loads 16 unique bytes from
// A LDS but then uses 4 shufflevector ops to interleave i16 elements:
//   lo0 = {K0K1, K4K5}, hi0 = {K2K3, K6K7}, etc.
// This interleaving was necessary because the standard B register layout
// maps each VGPR to a group of 4 K positions (blo[0]->v0, blo[1]->v1).
//
// In v2, A data is loaded DIRECTLY as two sequential 8-byte chunks:
//   a_reg_1 = {K0,K1,K2,K3,K4,K5,K6,K7}
//   a_reg_2 = {K8,K9,K10,K11,K12,K13,K14,K15}
// No shufflevector needed. To compensate, B is constructed differently:
// each input blo VGPR (4 bytes = one K group) is SPLIT into lo16 and hi16,
// and each half goes into a SEPARATE output B VGPR, mixed with bhi:
//   b_reg_1.v0 = blo[0].lo16 | (bhi[0].lo16 << 16)  = {B0,B1, B16,B17}
//   b_reg_1.v1 = (blo[0] >> 16) | (bhi[0] & HI16)    = {B2,B3, B18,B19}
//   b_reg_1.v2 = blo[1].lo16 | (bhi[1].lo16 << 16)  = {B4,B5, B20,B21}
//   b_reg_1.v3 = (blo[1] >> 16) | (bhi[1] & HI16)    = {B6,B7, B22,B23}
//
// This trades 4 shufflevector ops for 8 additional B ALU ops (16 total vs 8).
// Same A LDS layout, same tile dimensions, same sparsity indices.
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata_v2
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  // Sparsity indices: same as baseline and uniquedata v1
  static constexpr int SPARSITY_EVEN =
      0x00004444; // Selects positions 0,1 from each group of 4 8-bit elements
  static constexpr int SPARSITY_ODD =
      0x0000EEEE; // Selects positions 2,3 from each group of 4 8-bit elements

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 128; } // 2x baseline

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 128) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 2048) + n_col * (KS * 128) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using i8x8 = __attribute__((__vector_size__(8))) int8_t;
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;
    using i16x2 = __attribute__((__vector_size__(4))) int16_t;
    using i32x4 = __attribute__((__vector_size__(16))) int32_t;

    union {
      i8x8 vec;
      int32_t i32[2];
    } a_reg;

    i32x4 acc[NS / 4] = {{0}};

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;

    // A tile: 8 rows x KS*128 bytes = 8*KS*8 vec16 elements
    // B tile: NS n-blocks x KS k-banks x 128 vec16 elements
    constexpr int A_TILE_VEC16 = 8 * KS * 8;
    constexpr int B_TILE_VEC16 = NS * KS * 128;

    const auto *A_base = static_cast<const i8x16 *>(A_data) +
                         blockIdx.x * K_outer * A_TILE_VEC16;
    const auto *B_base = static_cast<const i8x16 *>(B_data) +
                         blockIdx.y * K_outer * B_TILE_VEC16;
    const auto *B_global_ptr = B_base + tid;

    __shared__ i8x16 A_smem[A_TILE_VEC16];
    __shared__ i8x16 B_smem[B_TILE_VEC16];
    auto *A_smem_bytes = reinterpret_cast<int8_t *>(A_smem);

    const int a_m_pair = (lane % 16) / 2;
    const int odd_lane = lane % 2;
    const int a_k_half = lane / 16;
    // Unique data: even/odd lanes read from different 16-byte regions
    // Same LDS layout as uniquedata v1
    const int a_lds_base =
        a_m_pair * (KS * 128) + a_k_half * 32 + odd_lane * 16;

    const int b_cons_n_col = lane % 16;
    const int b_cons_k_slice = lane / 16;

    const int sparsity_idx = (lane & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      // Load A tile: global -> shared
      const i8x16 *A_ptr = A_base + k_outer * A_TILE_VEC16;
      for (int i = tid; i < A_TILE_VEC16; i += 256) {
        A_smem[i] = A_ptr[i];
      }

      // Load B tile: global -> shared
      for (int i = 0; i < B_TILE_VEC16; i += 256) {
        B_smem[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_TILE_VEC16;
      __syncthreads();

      for (int kb = 0; kb < KS; ++kb) {
        // --- A extraction: DIRECT load, no shufflevector ---
        // Even lane reads K0..K15, odd lane reads K16..K31.
        // Load as 4 consecutive int32 words (16 bytes total):
        //   a_i32[0] = {K0,K1,K2,K3}   a_i32[1] = {K4,K5,K6,K7}
        //   a_i32[2] = {K8,K9,K10,K11}  a_i32[3] = {K12,K13,K14,K15}
        //
        // For SMFMAC 1: a_reg = {a_i32[0], a_i32[1]}
        //   = {K0,K1,K2,K3, K4,K5,K6,K7}
        // For SMFMAC 2: a_reg = {a_i32[2], a_i32[3]}
        //   = {K8,K9,K10,K11, K12,K13,K14,K15}
        auto *a_i32 = reinterpret_cast<int32_t *>(A_smem_bytes + a_lds_base +
                                                   kb * 128);

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp;

          // --- B: two 128-bit loads (K_lo and K_hi) ---
          // Same B addressing as uniquedata v1
          int b_byte_offset = n_block * (KS * 2048) +
                              b_cons_n_col * (KS * 128) + kb * 128 +
                              b_cons_k_slice * 32;
          auto *blo = reinterpret_cast<const i16x2 *>(
              reinterpret_cast<int8_t *>(B_smem) + b_byte_offset);
          auto *bhi = reinterpret_cast<const i16x2 *>(
              reinterpret_cast<int8_t *>(B_smem) + b_byte_offset + 16);

          // --- B mixing: split-VGPR layout for direct A loading ---
          // Each input blo VGPR is SPLIT: lo16 and hi16 go to consecutive
          // output VGPRs, mixed with corresponding bhi halves.
          //
          // b_reg_1 (pairs with a_reg_1 = first 8 A bytes):
          //   v0 = shuffle(blo[0], bhi[0], {0,2}) = {B0,B1, B16,B17}
          //   v1 = shuffle(blo[0], bhi[0], {1,3}) = {B2,B3, B18,B19}
          //   v2 = shuffle(blo[1], bhi[1], {0,2}) = {B4,B5, B20,B21}
          //   v3 = shuffle(blo[1], bhi[1], {1,3}) = {B6,B7, B22,B23}
          //
          // b_reg_2 (pairs with a_reg_2 = next 8 A bytes):
          //   v0 = shuffle(blo[2], bhi[2], {0,2}) = {B8,B9, B24,B25}
          //   v1 = shuffle(blo[2], bhi[2], {1,3}) = {B10,B11, B26,B27}
          //   v2 = shuffle(blo[3], bhi[3], {0,2}) = {B12,B13, B28,B29}
          //   v3 = shuffle(blo[3], bhi[3], {1,3}) = {B14,B15, B30,B31}
          union {
            i8x16 vec;
            int32_t i32[4];
          } b_reg_1, b_reg_2;

          // b_reg_1: from blo[0..1] and bhi[0..1]
          b_reg_1.i32[0] = __builtin_bit_cast(int, __builtin_shufflevector(blo[0], bhi[0], 0, 2));
          b_reg_1.i32[1] = __builtin_bit_cast(int, __builtin_shufflevector(blo[0], bhi[0], 1, 3));
          b_reg_1.i32[2] = __builtin_bit_cast(int, __builtin_shufflevector(blo[1], bhi[1], 0, 2));
          b_reg_1.i32[3] = __builtin_bit_cast(int, __builtin_shufflevector(blo[1], bhi[1], 1, 3));

          // b_reg_2: from blo[2..3] and bhi[2..3]
          b_reg_2.i32[0] = __builtin_bit_cast(int, __builtin_shufflevector(blo[2], bhi[2], 0, 2));
          b_reg_2.i32[1] = __builtin_bit_cast(int, __builtin_shufflevector(blo[2], bhi[2], 1, 3));
          b_reg_2.i32[2] = __builtin_bit_cast(int, __builtin_shufflevector(blo[3], bhi[3], 0, 2));
          b_reg_2.i32[3] = __builtin_bit_cast(int, __builtin_shufflevector(blo[3], bhi[3], 1, 3));

          // SMFMAC 1: first 8 A bytes x b_reg_1
          // Even lane: K0*B0 + K1*B1 + K2*B2 + K3*B3 + K4*B4 + K5*B5 +
          //            K6*B6 + K7*B7
          // Odd lane:  K16*B16 + K17*B17 + ... + K23*B23
          a_reg.i32[0] = a_i32[0];
          a_reg.i32[1] = a_i32[1];
          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg_1.vec, acc[j], sparsity_idx, 0, 0);

          // SMFMAC 2: next 8 A bytes x b_reg_2
          // Even lane: K8*B8 + K9*B9 + ... + K15*B15
          // Odd lane:  K24*B24 + K25*B25 + ... + K31*B31
          a_reg.i32[0] = a_i32[2];
          a_reg.i32[1] = a_i32[3];
          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg_2.vec, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: same 2-way reduction as baseline
    for (int j = 0; j < NS / 4; ++j) {
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      d[0] = d[0] + d[1]; // row0 complete
      d[1] = d[2] + d[3]; // row1 complete
    }

    constexpr int N_STRIDE = NS * 16;
    constexpr int TILE_ELEMS = 8 * N_STRIDE;

    const int m_pair = lane / 16;
    const int m0 = m_pair * 2;
    const int m1 = m0 + 1;
    const int n_col_out = lane % 16;

    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      TILE_ELEMS * (N_outer * blockIdx.x + blockIdx.y);

    for (int j = 0; j < NS / 4; ++j) {
      const int n = (j * 4 + warp) * 16 + n_col_out;
      auto *d = reinterpret_cast<int32_t *>(&acc[j]);
      C_tile[m0 * N_STRIDE + n] = d[0];
      C_tile[m1 * N_STRIDE + n] = d[1];
    }
  }
};

// Unique-lane-data variant of the F16 SMFMAC sparse trick kernel.
// Instead of even/odd lanes reading the SAME 128-bit A data from LDS,
// each lane reads UNIQUE data: even lane gets K_lo (K0..K7 as f16),
// odd lane gets K_hi (K8..K15). This doubles K_tile (KS * 64 vs
// KS * 32) and issues 2 SMFMAC instructions per k_bank per j.
//
// The key innovation is a BFI (bit-field-insert) shuffle that
// interleaves f16 elements from K_lo and K_hi at the 16-bit level
// within each 32-bit VGPR. This is applied to both A and B operands.
//
// Data flow:
//   1. Each lane loads 128b of unique A data from LDS
//   2. ds_bpermute exchanges all 4 A VGPRs with partner lane
//   3. BFI constructs interleaved A regs from local + partner data
//   4. Each lane loads two 128b B chunks (blo for K_lo, bhi for K_hi)
//   5. BFI constructs interleaved b_regs from each B chunk
//   6. SMFMAC 1 covers all K_lo positions, SMFMAC 2 covers all K_hi
//
// BFI shuffle for B (and similarly A):
//   b_reg[0] = (B_VGPR0 & LO) | (B_VGPR2 & HI)
//   b_reg[1] = (B_VGPR2 & LO) | (B_VGPR0 & HI)
//   b_reg[2] = (B_VGPR1 & LO) | (B_VGPR3 & HI)
//   b_reg[3] = (B_VGPR3 & LO) | (B_VGPR1 & HI)
//   where LO = 0x0000FFFF, HI = 0xFFFF0000
//
// Expected assembly:
//   s_mov_b32 s0, 0x0000ffff
//   v_bfi_b32 b_reg0, s0, v0, v2
//   v_bfi_b32 b_reg1, s0, v2, v0
//   v_bfi_b32 b_reg2, s0, v1, v3
//   v_bfi_b32 b_reg3, s0, v3, v1
//
// Correctness proof (one k_bank, lane group 0):
//   Even lane loads A[K_lo]: {K0,K1,K2,K3,K4,K5,K6,K7}
//   Odd lane loads A[K_hi]: {K8,K9,K10,K11,K12,K13,K14,K15}
//
//   SMFMAC 1 (B from blo = K_lo B data):
//     b_reg = BFI(blo) = {{B0,B5},{B4,B1},{B2,B7},{B6,B3}}
//     Even A (normal BFI on local): {K0,K5, K2,K7}
//       -> K0*B0 + K5*B5 + K2*B2 + K7*B7
//     Odd A (swapped BFI on partner=K_lo): {K4,K1, K6,K3}
//       -> K4*B4 + K1*B1 + K6*B6 + K3*B3
//     Combined: K0*B0..K7*B7 (all 8 K_lo positions)
//
//   SMFMAC 2 (B from bhi = K_hi B data):
//     b_reg = BFI(bhi) = {{B8,B13},{B12,B9},{B10,B15},{B14,B11}}
//     Even A (normal BFI on partner=K_hi): {K8,K13, K10,K15}
//       -> K8*B8 + K13*B13 + K10*B10 + K15*B15
//     Odd A (swapped BFI on local): {K12,K9, K14,K11}
//       -> K12*B12 + K9*B9 + K14*B14 + K11*B11
//     Combined: K8*B8..K15*B15 (all 8 K_hi positions)
//
//   Total: all 16 K positions covered exactly once.
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  static constexpr int SPARSITY_EVEN = 0x00000044;
  static constexpr int SPARSITY_ODD = 0x000000EE;

  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; } // 2x baseline

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 64) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  // BFI helper: (mask & src1) | (~mask & src2) = lo16(src1) | hi16(src2)
  // Produces v_bfi_b32 with a constant mask of 0x0000ffff.
  __device__ static int32_t bfi16(int32_t src1, int32_t src2) {
    constexpr uint32_t LO = 0x0000FFFF;
    return (int32_t)((((uint32_t)src1) & LO) | (((uint32_t)src2) & ~LO));
  }

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using f16x4 =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using f16x8 =
        __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    union {
      f16x4 vec;
      int32_t i32[2];
    } a_reg;

    f32x4 acc[NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    int warp_id = tid / 64;

    // A tile: 8 rows x KS*64 f16 values = 8*KS*8 vec16 elements
    // (each vec16 = 8 f16 = 16 bytes)
    constexpr int A_tile_size_in_vec16 = 8 * KS * 8;
    // B tile: NS n-blocks x KS k-banks x 128 vec16 elements
    constexpr int B_tile_size_in_vec16 = NS * KS * 128;

    const f16x8 *A_global = static_cast<const f16x8 *>(A_data) +
                            m_outer * K_outer * A_tile_size_in_vec16;
    const f16x8 *B_global = static_cast<const f16x8 *>(B_data) +
                            n_outer * K_outer * B_tile_size_in_vec16;
    const f16x8 *B_global_ptr = B_global + tid;

    __shared__ f16x8 A_shared[A_tile_size_in_vec16];
    __shared__ f16x8 B_shared[B_tile_size_in_vec16];
    auto *A_shared_bytes = reinterpret_cast<int8_t *>(A_shared);

    int a_m_row = (lane_id % 16) / 2;
    int odd_lane = lane_id % 2;
    int a_k_range = lane_id / 16;
    // Unique data: even/odd lanes read from different 16-byte regions
    int a_lds_base = a_m_row * (KS * 128) + a_k_range * 32 + odd_lane * 16;

    int n_col = lane_id % 16;
    int k_slice = lane_id / 16;

    int sparsity_idx = (lane_id & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    // ds_bpermute index: partner lane = lane ^ 1, index = partner * 4
    int partner_idx = (lane_id ^ 1) * 4;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      // Load A tile: global -> shared
      const f16x8 *A_ptr = A_global + k_outer * A_tile_size_in_vec16;
      for (int i = tid; i < A_tile_size_in_vec16; i += 256) {
        A_shared[i] = A_ptr[i];
      }

      // Load B tile: global -> shared
      for (int i = 0; i < B_tile_size_in_vec16; i += 256) {
        B_shared[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      for (int k_bank = 0; k_bank < KS; ++k_bank) {
        // --- A: load 128b unique data (16 bytes = 8 f16 = 4 int32) ---
        // Even lane: a_local = {K0,K1, K2,K3, K4,K5, K6,K7}
        // Odd lane:  a_local = {K8,K9, K10,K11, K12,K13, K14,K15}
        auto *a_local_ptr = reinterpret_cast<int32_t *>(
            A_shared_bytes + a_lds_base + k_bank * 128);
        int32_t a_local[4];
        a_local[0] = a_local_ptr[0];
        a_local[1] = a_local_ptr[1];
        a_local[2] = a_local_ptr[2];
        a_local[3] = a_local_ptr[3];

        // --- ds_bpermute: exchange A VGPRs with partner lane ---
        // Even lane gets odd lane's K_hi data, odd lane gets even's K_lo.
        int32_t a_partner[4];
        a_partner[0] = __builtin_amdgcn_ds_bpermute(partner_idx, a_local[0]);
        a_partner[1] = __builtin_amdgcn_ds_bpermute(partner_idx, a_local[1]);
        a_partner[2] = __builtin_amdgcn_ds_bpermute(partner_idx, a_local[2]);
        a_partner[3] = __builtin_amdgcn_ds_bpermute(partner_idx, a_local[3]);

        // --- A BFI construction ---
        // For SMFMAC 1 (K_lo range):
        //   Even lane uses "normal" BFI on local data:
        //     a_reg = {lo16(local[0])|hi16(local[2]),
        //              lo16(local[1])|hi16(local[3])}
        //           = {K0,K5, K2,K7}
        //   Odd lane uses "swapped" BFI on partner (=K_lo) data:
        //     a_reg = {lo16(partner[2])|hi16(partner[0]),
        //              lo16(partner[3])|hi16(partner[1])}
        //           = {K4,K1, K6,K3}
        //
        // For SMFMAC 2 (K_hi range):
        //   Even lane uses "normal" BFI on partner (=K_hi) data:
        //     a_reg = {K8,K13, K10,K15}
        //   Odd lane uses "swapped" BFI on local data:
        //     a_reg = {K12,K9, K14,K11}
        //
        // Unifying: SMFMAC 1 src = even?local:partner,
        //           SMFMAC 2 src = even?partner:local
        // Even lane: normal BFI (src[0],src[2])
        // Odd lane:  swapped BFI (src[2],src[0])
        int32_t *a_src1 = odd_lane ? a_partner : a_local;
        int32_t *a_src2 = odd_lane ? a_local : a_partner;

        // BFI A for SMFMAC 1 (covers K_lo range)
        int a1_v0 = odd_lane ? bfi16(a_src1[2], a_src1[0])
                             : bfi16(a_src1[0], a_src1[2]);
        int a1_v1 = odd_lane ? bfi16(a_src1[3], a_src1[1])
                             : bfi16(a_src1[1], a_src1[3]);

        // BFI A for SMFMAC 2 (covers K_hi range)
        int a2_v0 = odd_lane ? bfi16(a_src2[2], a_src2[0])
                             : bfi16(a_src2[0], a_src2[2]);
        int a2_v1 = odd_lane ? bfi16(a_src2[3], a_src2[1])
                             : bfi16(a_src2[1], a_src2[3]);

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp_id;

          // --- B: two 128-bit loads (K_lo and K_hi B data) ---
          int b_byte_offset = n_block * (KS * 2048) + n_col * (KS * 128) +
                              k_bank * 128 + k_slice * 32;
          auto *blo_i32 = reinterpret_cast<int32_t *>(
              reinterpret_cast<int8_t *>(B_shared) + b_byte_offset);
          auto *bhi_i32 = reinterpret_cast<int32_t *>(
              reinterpret_cast<int8_t *>(B_shared) + b_byte_offset + 16);

          // --- B BFI shuffle ---
          // Each 128-bit B load (4 VGPRs: v0,v1,v2,v3) is shuffled via:
          //   b_reg[0] = bfi16(v0, v2) = lo16(v0) | hi16(v2)
          //   b_reg[1] = bfi16(v2, v0) = lo16(v2) | hi16(v0)
          //   b_reg[2] = bfi16(v1, v3) = lo16(v1) | hi16(v3)
          //   b_reg[3] = bfi16(v3, v1) = lo16(v3) | hi16(v1)
          //
          // For blo = {B0,B1, B2,B3, B4,B5, B6,B7}:
          //   b_reg_1 = {{B0,B5}, {B4,B1}, {B2,B7}, {B6,B3}}
          //   Group 0 pos 0,1 = B0,B5; pos 2,3 = B4,B1
          //   Group 1 pos 0,1 = B2,B7; pos 2,3 = B6,B3
          //
          // Even (pos 0,1): K0*B0 + K5*B5 + K2*B2 + K7*B7
          // Odd (pos 2,3):  K4*B4 + K1*B1 + K6*B6 + K3*B3
          // Combined: all K_lo products.
          union {
            f16x8 vec;
            int32_t i32[4];
          } b_reg_1, b_reg_2;

          b_reg_1.i32[0] = bfi16(blo_i32[0], blo_i32[2]);
          b_reg_1.i32[1] = bfi16(blo_i32[2], blo_i32[0]);
          b_reg_1.i32[2] = bfi16(blo_i32[1], blo_i32[3]);
          b_reg_1.i32[3] = bfi16(blo_i32[3], blo_i32[1]);

          b_reg_2.i32[0] = bfi16(bhi_i32[0], bhi_i32[2]);
          b_reg_2.i32[1] = bfi16(bhi_i32[2], bhi_i32[0]);
          b_reg_2.i32[2] = bfi16(bhi_i32[1], bhi_i32[3]);
          b_reg_2.i32[3] = bfi16(bhi_i32[3], bhi_i32[1]);

          // SMFMAC 1: K_lo range -- A set 1 x b_reg_1
          a_reg.i32[0] = a1_v0;
          a_reg.i32[1] = a1_v1;
          acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
              a_reg.vec, b_reg_1.vec, acc[j], sparsity_idx, 0, 0);

          // SMFMAC 2: K_hi range -- A set 2 x b_reg_2
          a_reg.i32[0] = a2_v0;
          a_reg.i32[1] = a2_v1;
          acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
              a_reg.vec, b_reg_2.vec, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: same 2-way reduction as baseline
    for (int j = 0; j < NS / 4; ++j) {
      float *regD = reinterpret_cast<float *>(&acc[j]);
      regD[0] = regD[0] + regD[1];
      regD[1] = regD[2] + regD[3];
    }

    constexpr int N_stride = NS * 16;
    int m_pair = lane_id / 16;
    int m0 = m_pair * 2;
    int m1 = m_pair * 2 + 1;

    constexpr int tile_size = 8 * NS * 16;
    float *C_tile = static_cast<float *>(C_data) +
                    tile_size * (N_outer * m_outer + n_outer);

    for (int j = 0; j < NS / 4; ++j) {
      int n_block = j * 4 + warp_id;
      int n = n_block * 16 + n_col;

      float *regD = reinterpret_cast<float *>(&acc[j]);
      C_tile[m0 * N_stride + n] = regD[0];
      C_tile[m1 * N_stride + n] = regD[1];
    }
  }
};

// Unique-lane-data v2 of the F16 SMFMAC sparse trick kernel.
//
// Eliminates BFI from BOTH A and B register construction compared to v1.
// A: direct dword select from local/partner via v_cndmask (no BFI).
// B: direct passthrough from blo/bhi (0 VALU).
// Total VALU: 4 v_cndmask.  Total LDS: 4 ds_bpermute.
//
// Same tile dimensions, LDS layout, and sparsity indices as uniquedata v1.
//
// Correctness proof (one k_bank, lane group 0):
//   Even lane loads A[K_lo]: {K0,K1,K2,K3,K4,K5,K6,K7}
//   Odd lane loads A[K_hi]: {K8,K9,K10,K11,K12,K13,K14,K15}
//
//   SMFMAC 1 (B = blo = K_lo B data, direct passthrough):
//     Even lane (sparsity 0x44 = pos 0,1):
//       A = {local[0], local[2]} = {{K0,K1}, {K4,K5}}
//       -> K0*B0 + K1*B1 + K4*B4 + K5*B5
//     Odd lane (sparsity 0xEE = pos 2,3):
//       A = {partner[1], partner[3]} = {{K2,K3}, {K6,K7}}
//       -> K2*B2 + K3*B3 + K6*B6 + K7*B7
//     Combined: K0*B0..K7*B7 (all 8 K_lo positions)
//
//   SMFMAC 2 (B = bhi = K_hi B data, direct passthrough):
//     Even lane (pos 0,1):
//       A = {partner[0], partner[2]} = {{K8,K9}, {K12,K13}}
//       -> K8*B8 + K9*B9 + K12*B12 + K13*B13
//     Odd lane (pos 2,3):
//       A = {local[1], local[3]} = {{K10,K11}, {K14,K15}}
//       -> K10*B10 + K11*B11 + K14*B14 + K15*B15
//     Combined: K8*B8..K15*B15 (all 8 K_hi positions)
//
//   Total: all 16 K positions covered exactly once.
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v2
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  static constexpr int SPARSITY_EVEN = 0x00000044;
  static constexpr int SPARSITY_ODD = 0x000000EE;

  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; } // 2x baseline

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 64) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using f16x4 =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using f16x8 =
        __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    union {
      f16x4 vec;
      int32_t i32[2];
    } a_reg;

    f32x4 acc[NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    int warp_id = tid / 64;

    // A tile: 8 rows x KS*64 f16 values = 8*KS*8 vec16 elements
    // (each vec16 = 8 f16 = 16 bytes)
    constexpr int A_tile_size_in_vec16 = 8 * KS * 8;
    // B tile: NS n-blocks x KS k-banks x 128 vec16 elements
    constexpr int B_tile_size_in_vec16 = NS * KS * 128;

    const f16x8 *A_global = static_cast<const f16x8 *>(A_data) +
                            m_outer * K_outer * A_tile_size_in_vec16;
    const f16x8 *B_global = static_cast<const f16x8 *>(B_data) +
                            n_outer * K_outer * B_tile_size_in_vec16;
    const f16x8 *B_global_ptr = B_global + tid;

    __shared__ f16x8 A_shared[A_tile_size_in_vec16];
    __shared__ f16x8 B_shared[B_tile_size_in_vec16];
    auto *A_shared_bytes = reinterpret_cast<int8_t *>(A_shared);

    int a_m_row = (lane_id % 16) / 2;
    int odd_lane = lane_id % 2;
    int a_k_range = lane_id / 16;
    // Unique data: even/odd lanes read from different 16-byte regions
    int a_lds_base = a_m_row * (KS * 128) + a_k_range * 32 + odd_lane * 16;

    int n_col = lane_id % 16;
    int k_slice = lane_id / 16;

    int sparsity_idx = (lane_id & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    // ds_bpermute index: partner lane = lane ^ 1, index = partner * 4
    int partner_idx = (lane_id ^ 1) * 4;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      // Load A tile: global -> shared
      const f16x8 *A_ptr = A_global + k_outer * A_tile_size_in_vec16;
      for (int i = tid; i < A_tile_size_in_vec16; i += 256) {
        A_shared[i] = A_ptr[i];
      }

      // Load B tile: global -> shared
      for (int i = 0; i < B_tile_size_in_vec16; i += 256) {
        B_shared[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      for (int k_bank = 0; k_bank < KS; ++k_bank) {
        // --- A: load 128b unique data (16 bytes = 8 f16 = 4 int32) ---
        // Even lane: a_local = {K0,K1, K2,K3, K4,K5, K6,K7}
        // Odd lane:  a_local = {K8,K9, K10,K11, K12,K13, K14,K15}
        auto *a_local_ptr = reinterpret_cast<int32_t *>(
            A_shared_bytes + a_lds_base + k_bank * 128);
        int32_t a_local[4];
        a_local[0] = a_local_ptr[0];
        a_local[1] = a_local_ptr[1];
        a_local[2] = a_local_ptr[2];
        a_local[3] = a_local_ptr[3];

        // --- ds_bpermute: exchange A VGPRs with partner lane ---
        // Even lane gets odd lane's K_hi data, odd lane gets even's K_lo.
        int32_t a_partner[4];
        a_partner[0] = __builtin_amdgcn_ds_bpermute(partner_idx, a_local[0]);
        a_partner[1] = __builtin_amdgcn_ds_bpermute(partner_idx, a_local[1]);
        a_partner[2] = __builtin_amdgcn_ds_bpermute(partner_idx, a_local[2]);
        a_partner[3] = __builtin_amdgcn_ds_bpermute(partner_idx, a_local[3]);

        // --- A: direct dword select (v_cndmask, no BFI) ---
        // SMFMAC 1 (K_lo range):
        //   Even lane: {local[0], local[2]} = {{K0,K1}, {K4,K5}}
        //   Odd lane:  {partner[1], partner[3]} = {{K2,K3}, {K6,K7}}
        int a1_v0 = odd_lane ? a_partner[1] : a_local[0];
        int a1_v1 = odd_lane ? a_partner[3] : a_local[2];

        // SMFMAC 2 (K_hi range):
        //   Even lane: {partner[0], partner[2]} = {{K8,K9}, {K12,K13}}
        //   Odd lane:  {local[1], local[3]} = {{K10,K11}, {K14,K15}}
        int a2_v0 = odd_lane ? a_local[1] : a_partner[0];
        int a2_v1 = odd_lane ? a_local[3] : a_partner[2];

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp_id;

          // --- B: direct passthrough (0 VALU) ---
          int b_byte_offset = n_block * (KS * 2048) + n_col * (KS * 128) +
                              k_bank * 128 + k_slice * 32;
          auto *b_ptr = reinterpret_cast<f16x8 *>(
              reinterpret_cast<int8_t *>(B_shared) + b_byte_offset);

          // b_reg_1 = blo (first 16 bytes), b_reg_2 = bhi (next 16 bytes)
          f16x8 b_reg_1 = b_ptr[0];
          f16x8 b_reg_2 = b_ptr[1];

          // SMFMAC 1: K_lo range -- A set 1 x b_reg_1
          a_reg.i32[0] = a1_v0;
          a_reg.i32[1] = a1_v1;
          acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
              a_reg.vec, b_reg_1, acc[j], sparsity_idx, 0, 0);

          // SMFMAC 2: K_hi range -- A set 2 x b_reg_2
          a_reg.i32[0] = a2_v0;
          a_reg.i32[1] = a2_v1;
          acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
              a_reg.vec, b_reg_2, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: same 2-way reduction as baseline
    for (int j = 0; j < NS / 4; ++j) {
      float *regD = reinterpret_cast<float *>(&acc[j]);
      regD[0] = regD[0] + regD[1];
      regD[1] = regD[2] + regD[3];
    }

    constexpr int N_stride = NS * 16;
    int m_pair = lane_id / 16;
    int m0 = m_pair * 2;
    int m1 = m_pair * 2 + 1;

    constexpr int tile_size = 8 * NS * 16;
    float *C_tile = static_cast<float *>(C_data) +
                    tile_size * (N_outer * m_outer + n_outer);

    for (int j = 0; j < NS / 4; ++j) {
      int n_block = j * 4 + warp_id;
      int n = n_block * 16 + n_col;

      float *regD = reinterpret_cast<float *>(&acc[j]);
      C_tile[m0 * N_stride + n] = regD[0];
      C_tile[m1 * N_stride + n] = regD[1];
    }
  }
};

// Unique-lane-data v3 of the F16 SMFMAC sparse trick kernel.
//
// Eliminates ds_bpermute entirely — completely thread-local computation.
// A: direct sequential load (0 VALU, 0 ds_bpermute).
// B: dword-level interleave of blo/bhi (4 v_mov_b32 per j).
// Total per (kb, j): 0 cross-lane ops, ~4 v_mov_b32.
//
// Key insight: instead of rearranging A to match B's natural layout (v2),
// rearrange B to match A's natural layout.  With direct A loading:
//   SMFMAC 1: A.v0={K0,K1}, A.v1={K2,K3}  (even)
//             A.v0={K8,K9}, A.v1={K10,K11} (odd)
//   SMFMAC 2: A.v0={K4,K5}, A.v1={K6,K7}  (even)
//             A.v0={K12,K13}, A.v1={K14,K15} (odd)
//
// Even lane (sparsity 0x44) reads B.v0 and B.v2.
// Odd  lane (sparsity 0xEE) reads B.v1 and B.v3.
//
// B construction: dword interleave = {blo[i], bhi[i], blo[i+1], bhi[i+1]}
//   (equivalent to shufflevector on i32x2 with mask {0, 2, 1, 3})
//
// Correctness proof (one k_bank, lane group 0):
//   Even lane loads A[K_lo]: {K0,K1,K2,K3,K4,K5,K6,K7}
//   Odd lane loads A[K_hi]: {K8,K9,K10,K11,K12,K13,K14,K15}
//
//   SMFMAC 1 (B = {blo[0], bhi[0], blo[1], bhi[1]}):
//     Even (pos 0,1): A.v0={K0,K1} x B.v0=blo[0] + A.v1={K2,K3} x B.v2=blo[1]
//       -> K0*BK0 + K1*BK1 + K2*BK2 + K3*BK3
//     Odd (pos 2,3): A.v0={K8,K9} x B.v1=bhi[0] + A.v1={K10,K11} x B.v3=bhi[1]
//       -> K8*BK8 + K9*BK9 + K10*BK10 + K11*BK11
//     Combined: 8 K products
//
//   SMFMAC 2 (B = {blo[2], bhi[2], blo[3], bhi[3]}):
//     Even: K4*BK4 + K5*BK5 + K6*BK6 + K7*BK7
//     Odd:  K12*BK12 + K13*BK13 + K14*BK14 + K15*BK15
//     Combined: 8 K products
//
//   Total: all 16 K positions covered exactly once.
//
// Cost comparison:
//   v2: 4 ds_bpermute + 4 v_cndmask (per kb), 0 B VALU (per j)
//   v3: 0 cross-lane, 0 A VALU (per kb), ~4 v_mov_b32 (per j)
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v3
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  static constexpr int SPARSITY_EVEN = 0x00000044;
  static constexpr int SPARSITY_ODD = 0x000000EE;

  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; } // 2x baseline

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 64) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using f16x4 =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using f16x8 =
        __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    union {
      f16x4 vec;
      int32_t i32[2];
    } a_reg;

    f32x4 acc[NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    int warp_id = tid / 64;

    // A tile: 8 rows x KS*64 f16 values = 8*KS*8 vec16 elements
    // (each vec16 = 8 f16 = 16 bytes)
    constexpr int A_tile_size_in_vec16 = 8 * KS * 8;
    // B tile: NS n-blocks x KS k-banks x 128 vec16 elements
    constexpr int B_tile_size_in_vec16 = NS * KS * 128;

    const f16x8 *A_global = static_cast<const f16x8 *>(A_data) +
                            m_outer * K_outer * A_tile_size_in_vec16;
    const f16x8 *B_global = static_cast<const f16x8 *>(B_data) +
                            n_outer * K_outer * B_tile_size_in_vec16;
    const f16x8 *B_global_ptr = B_global + tid;

    __shared__ f16x8 A_shared[A_tile_size_in_vec16];
    __shared__ f16x8 B_shared[B_tile_size_in_vec16];
    auto *A_shared_bytes = reinterpret_cast<int8_t *>(A_shared);

    int a_m_row = (lane_id % 16) / 2;
    int a_k_range = lane_id / 16;
    int odd_lane = lane_id % 2;
    // Unique data: even/odd lanes read from different 16-byte regions
    int a_lds_base = a_m_row * (KS * 128) + a_k_range * 32 + odd_lane * 16;

    int n_col = lane_id % 16;
    int k_slice = lane_id / 16;

    int sparsity_idx = (lane_id & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      // Load A tile: global -> shared
      const f16x8 *A_ptr = A_global + k_outer * A_tile_size_in_vec16;
      for (int i = tid; i < A_tile_size_in_vec16; i += 256) {
        A_shared[i] = A_ptr[i];
      }

      // Load B tile: global -> shared
      for (int i = 0; i < B_tile_size_in_vec16; i += 256) {
        B_shared[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      for (int k_bank = 0; k_bank < KS; ++k_bank) {
        // --- A: direct sequential load (0 VALU, 0 ds_bpermute) ---
        // Even lane: a_i32 = {K0K1, K2K3, K4K5, K6K7}
        // Odd lane:  a_i32 = {K8K9, K10K11, K12K13, K14K15}
        //
        // SMFMAC 1: a_reg = {a_i32[0], a_i32[1]}
        // SMFMAC 2: a_reg = {a_i32[2], a_i32[3]}
        auto *a_i32 = reinterpret_cast<int32_t *>(
            A_shared_bytes + a_lds_base + k_bank * 128);

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp_id;

          // --- B: dword-level interleave of blo/bhi ---
          // blo = first 16 bytes (4 dwords), bhi = next 16 bytes (4 dwords)
          // Interleave: {blo[i], bhi[i], blo[i+1], bhi[i+1]}
          // This places K_lo and K_hi B data in alternating VGPRs,
          // matching the VGPR-level sparsity partition (0x44/0xEE).
          int b_byte_offset = n_block * (KS * 2048) + n_col * (KS * 128) +
                              k_bank * 128 + k_slice * 32;
          auto *blo_i32 = reinterpret_cast<int32_t *>(
              reinterpret_cast<int8_t *>(B_shared) + b_byte_offset);
          auto *bhi_i32 = reinterpret_cast<int32_t *>(
              reinterpret_cast<int8_t *>(B_shared) + b_byte_offset + 16);

          union {
            f16x8 vec;
            int32_t i32[4];
          } b_reg_1, b_reg_2;

          // b_reg_1 = {blo[0], bhi[0], blo[1], bhi[1]}
          b_reg_1.i32[0] = blo_i32[0];
          b_reg_1.i32[1] = bhi_i32[0];
          b_reg_1.i32[2] = blo_i32[1];
          b_reg_1.i32[3] = bhi_i32[1];

          // b_reg_2 = {blo[2], bhi[2], blo[3], bhi[3]}
          b_reg_2.i32[0] = blo_i32[2];
          b_reg_2.i32[1] = bhi_i32[2];
          b_reg_2.i32[2] = blo_i32[3];
          b_reg_2.i32[3] = bhi_i32[3];

          // SMFMAC 1: first 4 f16 of A x b_reg_1
          a_reg.i32[0] = a_i32[0];
          a_reg.i32[1] = a_i32[1];
          acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
              a_reg.vec, b_reg_1.vec, acc[j], sparsity_idx, 0, 0);

          // SMFMAC 2: next 4 f16 of A x b_reg_2
          a_reg.i32[0] = a_i32[2];
          a_reg.i32[1] = a_i32[3];
          acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
              a_reg.vec, b_reg_2.vec, acc[j], sparsity_idx, 0, 0);
        }
        __syncthreads();
      }
    }

    // Post-processing: same 2-way reduction as baseline
    for (int j = 0; j < NS / 4; ++j) {
      float *regD = reinterpret_cast<float *>(&acc[j]);
      regD[0] = regD[0] + regD[1];
      regD[1] = regD[2] + regD[3];
    }

    constexpr int N_stride = NS * 16;
    int m_pair = lane_id / 16;
    int m0 = m_pair * 2;
    int m1 = m_pair * 2 + 1;

    constexpr int tile_size = 8 * NS * 16;
    float *C_tile = static_cast<float *>(C_data) +
                    tile_size * (N_outer * m_outer + n_outer);

    for (int j = 0; j < NS / 4; ++j) {
      int n_block = j * 4 + warp_id;
      int n = n_block * 16 + n_col;

      float *regD = reinterpret_cast<float *>(&acc[j]);
      C_tile[m0 * N_stride + n] = regD[0];
      C_tile[m1 * N_stride + n] = regD[1];
    }
  }
};

// M-padded MFMA kernel for skinny GEMMs (M=8->16).
// Uses the dense mfma_i32_16x16x32_i8 instruction with zero-padding
// on the A operand for lanes corresponding to M-rows 8-15.
// This serves as a benchmark comparison against the smfmac sparse
// trick kernel above, which achieves the same M=8 matmul using
// smfmac_i32_16x16x64_i8.
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_mfma_i32_16x16x32i8_zeropad_shared
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; }

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 64) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int8x8_t = int64_t;
    using int32x4_t = __attribute__((__vector_size__(16))) int32_t;
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;

    int32x4_t acc[NS / 4] = {{0}};

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;

    constexpr int A_TILE_VEC16 = 8 * KS * 4;
    constexpr int B_TILE_VEC16 = NS * KS * 64;

    const auto *A_base = static_cast<const i8x16 *>(A_data) +
                         blockIdx.x * K_outer * A_TILE_VEC16;
    const auto *B_base = static_cast<const i8x16 *>(B_data) +
                         blockIdx.y * K_outer * B_TILE_VEC16;
    const auto *B_global_ptr = B_base + tid;

    __shared__ i8x16 A_smem[A_TILE_VEC16];
    __shared__ i8x16 B_smem[B_TILE_VEC16];
    auto *A_smem_bytes = reinterpret_cast<int8_t *>(A_smem);

    const int a_m_row = lane % 16;
    const int a_k_half = lane / 16;
    const int b_cons_n_col = lane % 16;
    const int b_cons_k_slice = lane / 16;

    const int a_lds_base = a_m_row * (KS * 64) + a_k_half * 8;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      if (tid < A_TILE_VEC16) {
        const i8x16 *A_ptr = A_base + k_outer * A_TILE_VEC16;
        A_smem[tid] = A_ptr[tid];
      }

      for (int i = 0; i < B_TILE_VEC16; i += 256) {
        B_smem[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_TILE_VEC16;
      __syncthreads();

      for (int kb = 0; kb < KS; ++kb) {
        for (int kh = 0; kh < 2; ++kh) {
          // Load A: zero-pad for M-rows >= 8 (padding region)
          int8x8_t a_reg = (a_m_row < 8)
              ? *reinterpret_cast<int8x8_t *>(
                    A_smem_bytes + a_lds_base + kb * 64 + kh * 32)
              : 0;

          for (int j = 0; j < NS / 4; ++j) {
            int n_block = j * 4 + warp;

            auto *b_ptr = reinterpret_cast<int8x8_t *>(
                reinterpret_cast<int8_t *>(B_smem) + n_block * (KS * 1024) +
                b_cons_n_col * (KS * 64) + kb * 64 + kh * 32 +
                b_cons_k_slice * 8);
            int8x8_t b_reg = *b_ptr;

            acc[j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                a_reg, b_reg, acc[j], 0, 0, 0);
          }
        }
        __syncthreads();
      }
    }

    // Output: d[p] = C[4*(lane/16) + p][lane%16]
    // Only lane/16 < 2 (rows 0-7) contain real results.
    constexpr int N_STRIDE = NS * 16;
    constexpr int TILE_ELEMS = 8 * N_STRIDE;
    const int m_group = lane / 16;
    const int n_col_out = lane % 16;

    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      TILE_ELEMS * (N_outer * blockIdx.x + blockIdx.y);

    if (m_group < 2) {
      const int m_base = m_group * 4;
      for (int j = 0; j < NS / 4; ++j) {
        const int n = (j * 4 + warp) * 16 + n_col_out;
        auto *d = reinterpret_cast<int32_t *>(&acc[j]);
        C_tile[(m_base + 0) * N_STRIDE + n] = d[0];
        C_tile[(m_base + 1) * N_STRIDE + n] = d[1];
        C_tile[(m_base + 2) * N_STRIDE + n] = d[2];
        C_tile[(m_base + 3) * N_STRIDE + n] = d[3];
      }
    }
  }
};

// FP16 variant of the M-padded MFMA kernel for skinny GEMMs (M=8->16).
// Uses the dense mfma_f32_16x16x16_f16 instruction with zero-padding
// on the A operand for lanes corresponding to M-rows 8-15.
//
// For M=8: lanes with lane%16 >= 8 feed A=0 (zero-padding).
// Output rows 0-7 are real (lane/16 < 2), rows 8-15 are discarded.
// Since mfma processes K=16 vs smfmac's K=32, the inner loop runs
// KS*2 iterations (nested kb=0..KS-1, kh=0..1).
template <int NS, int KS>
class MmtKernel_256t_NSxKS_amdgcn_mfma_f32_16x16x16f16_zeropad_shared
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 32; }

  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m * (KS * 32) + k; };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 512) + n_col * (KS * 32) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return m * (NS * 16) + n; };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using f16x4 =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using f16x8 =
        __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    f32x4 acc[NS / 4] = {{0}};

    const int tid = threadIdx.x;
    const int lane = tid % 64;
    const int warp = tid / 64;

    constexpr int A_TILE_VEC16 = 8 * KS * 4;
    constexpr int B_TILE_VEC16 = NS * KS * 64;

    const auto *A_base = static_cast<const f16x8 *>(A_data) +
                         blockIdx.x * K_outer * A_TILE_VEC16;
    const auto *B_base = static_cast<const f16x8 *>(B_data) +
                         blockIdx.y * K_outer * B_TILE_VEC16;
    const auto *B_global_ptr = B_base + tid;

    __shared__ f16x8 A_smem[A_TILE_VEC16];
    __shared__ f16x8 B_smem[B_TILE_VEC16];
    auto *A_smem_bytes = reinterpret_cast<int8_t *>(A_smem);

    const int a_m_row = lane % 16;
    const int a_k_half = lane / 16;
    const int b_cons_n_col = lane % 16;
    const int b_cons_k_slice = lane / 16;

    const int a_lds_base = a_m_row * (KS * 64) + a_k_half * 8;

    const f16x4 zero_a = {0, 0, 0, 0};

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      if (tid < A_TILE_VEC16) {
        const f16x8 *A_ptr = A_base + k_outer * A_TILE_VEC16;
        A_smem[tid] = A_ptr[tid];
      }

      for (int i = 0; i < B_TILE_VEC16; i += 256) {
        B_smem[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_TILE_VEC16;
      __syncthreads();

      for (int kb = 0; kb < KS; ++kb) {
        for (int kh = 0; kh < 2; ++kh) {
          // Load A: zero-pad for M-rows >= 8 (padding region)
          f16x4 a_reg = (a_m_row < 8)
              ? *reinterpret_cast<f16x4 *>(
                    A_smem_bytes + a_lds_base + kb * 64 + kh * 32)
              : zero_a;

          for (int j = 0; j < NS / 4; ++j) {
            int n_block = j * 4 + warp;

            auto *b_ptr = reinterpret_cast<f16x4 *>(
                reinterpret_cast<int8_t *>(B_smem) + n_block * (KS * 1024) +
                b_cons_n_col * (KS * 64) + kb * 64 + kh * 32 +
                b_cons_k_slice * 8);
            f16x4 b_reg = *b_ptr;

            acc[j] = __builtin_amdgcn_mfma_f32_16x16x16f16(
                a_reg, b_reg, acc[j], 0, 0, 0);
          }
        }
        __syncthreads();
      }
    }

    // Output: d[p] = C[4*(lane/16) + p][lane%16]
    // Only lane/16 < 2 (rows 0-7) contain real results.
    constexpr int N_STRIDE = NS * 16;
    constexpr int TILE_ELEMS = 8 * N_STRIDE;
    const int m_group = lane / 16;
    const int n_col_out = lane % 16;

    float *C_tile = static_cast<float *>(C_data) +
                    TILE_ELEMS * (N_outer * blockIdx.x + blockIdx.y);

    if (m_group < 2) {
      const int m_base = m_group * 4;
      for (int j = 0; j < NS / 4; ++j) {
        const int n = (j * 4 + warp) * 16 + n_col_out;
        auto *d = reinterpret_cast<float *>(&acc[j]);
        C_tile[(m_base + 0) * N_STRIDE + n] = d[0];
        C_tile[(m_base + 1) * N_STRIDE + n] = d[1];
        C_tile[(m_base + 2) * N_STRIDE + n] = d[2];
        C_tile[(m_base + 3) * N_STRIDE + n] = d[3];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 64; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 1024 * mo + 256 * ((k % 32) / 8) + 16 * mi + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 1024 * no + 256 * ((k % 32) / 8) + 16 * ni + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 4 * (16 * (mi / 4) + ni) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int64x2_t = __attribute__((__vector_size__(8 * 2))) int64_t;
    using int32x4_t = __attribute__((__vector_size__(4 * 4))) int32_t;
    int32x4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec16 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec16 = NS * 16 * 4;

    const int64x2_t *A_global = static_cast<const int64x2_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec16;
    const int64x2_t *B_global = static_cast<const int64x2_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec16;
    const int64x2_t *A_global_ptr = A_global + tid;
    const int64x2_t *B_global_ptr = B_global + tid;

    __shared__ int64x2_t A_shared[A_tile_size_in_vec16];
    __shared__ int64x2_t B_shared[B_tile_size_in_vec16];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec16; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec16; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec16;
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          for (int k = 0; k < 2; ++k) {
            acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                A_shared[64 * i + (tid % 64)][k], B_shared[256 * j + tid][k],
                acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    int32x4_t *C_ptr = static_cast<int32x4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipelineload
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 64; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 1024 * mo + 256 * ((k % 32) / 8) + 16 * mi + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 1024 * no + 256 * ((k % 32) / 8) + 16 * ni + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 4 * (16 * (mi / 4) + ni) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int64x2_t = __attribute__((__vector_size__(8 * 2))) int64_t;
    using int32x4_t = __attribute__((__vector_size__(4 * 4))) int32_t;
    int32x4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec16 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec16 = NS * 16 * 4;
    constexpr int num_threads = 256;

    const int64x2_t *A_global = static_cast<const int64x2_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec16;
    const int64x2_t *B_global = static_cast<const int64x2_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec16;

    __shared__ int64x2_t A_shared[A_tile_size_in_vec16];
    __shared__ int64x2_t B_shared[B_tile_size_in_vec16];

    int64x2_t A_vgpr0[A_tile_size_in_vec16 / num_threads];
    int64x2_t B_vgpr0[B_tile_size_in_vec16 / num_threads];

    auto global_to_vgpr0 = [&]() {
      for (int i = 0; i < A_tile_size_in_vec16 / num_threads; ++i) {
        A_vgpr0[i] = A_global[i * num_threads + tid];
      }
      for (int j = 0; j < B_tile_size_in_vec16 / num_threads; ++j) {
        B_vgpr0[j] = B_global[j * num_threads + tid];
      }
      A_global += A_tile_size_in_vec16;
      B_global += B_tile_size_in_vec16;
    };

    auto vpgr0_to_shared = [&]() {
      for (int i = 0; i < A_tile_size_in_vec16 / num_threads; ++i) {
        A_shared[i * num_threads + tid] = A_vgpr0[i];
      }
      for (int j = 0; j < B_tile_size_in_vec16 / num_threads; ++j) {
        B_shared[j * num_threads + tid] = B_vgpr0[j];
      }
    };

    auto mfma = [&]() {
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          for (int k = 0; k < 2; ++k) {
            acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                A_shared[64 * i + (tid % 64)][k], B_shared[256 * j + tid][k],
                acc[i][j], 0, 0, 0);
          }
        }
      }
    };

    global_to_vgpr0();
    vpgr0_to_shared();
    if (K_outer >= 2) {
      global_to_vgpr0();
      for (int k_outer = 0; k_outer < K_outer - 2; ++k_outer) {
        // Wait for all shared memory to be written by all subgroups.
        __syncthreads();
        mfma();
        // Wait for all subgroups to finish reading from shared memory, so that
        // we can overwrite it.
        __syncthreads();
        vpgr0_to_shared();
        global_to_vgpr0();
      }
      // Wait for all shared memory to be written by all subgroups.
      __syncthreads();
      mfma();
      // Wait for all subgroups to finish reading from shared memory, so that
      // we can overwrite it.
      __syncthreads();
      vpgr0_to_shared();
    }
    // Wait for all shared memory to be written by all subgroups.
    __syncthreads();
    mfma();

    int32x4_t *C_ptr = static_cast<int32x4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipeline_v3
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 64; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 1024 * mo + 256 * ((k % 32) / 8) + 16 * mi + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 1024 * no + 256 * ((k % 32) / 8) + 16 * ni + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 4 * (16 * (mi / 4) + ni) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int64x2_t = __attribute__((__vector_size__(8 * 2))) int64_t;
    using int32x4_t = __attribute__((__vector_size__(4 * 4))) int32_t;
    int32x4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec16 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec16 = NS * 16 * 4;
    constexpr int num_threads = 256;

    const int64x2_t *A_global = static_cast<const int64x2_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec16;
    const int64x2_t *B_global = static_cast<const int64x2_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec16;

    __shared__ int64x2_t A_shared[A_tile_size_in_vec16];
    __shared__ int64x2_t B_shared[B_tile_size_in_vec16];

    int64x2_t A_vgpr0[A_tile_size_in_vec16 / num_threads];
    int64x2_t B_vgpr0[B_tile_size_in_vec16 / num_threads];

    int64x2_t A_block_vgpr1[MS];
    int64x2_t B_block_vgpr1[NS / 2];

    auto global_to_vgpr0 = [&]() {
      for (int i = 0; i < A_tile_size_in_vec16 / num_threads; ++i) {
        A_vgpr0[i] = A_global[i * num_threads + tid];
      }
      for (int j = 0; j < B_tile_size_in_vec16 / num_threads; ++j) {
        B_vgpr0[j] = B_global[j * num_threads + tid];
      }
      A_global += A_tile_size_in_vec16;
      B_global += B_tile_size_in_vec16;
    };

    auto vpgr0_to_shared = [&]() {
      for (int i = 0; i < A_tile_size_in_vec16 / num_threads; ++i) {
        A_shared[i * num_threads + tid] = A_vgpr0[i];
      }
      for (int j = 0; j < B_tile_size_in_vec16 / num_threads; ++j) {
        B_shared[j * num_threads + tid] = B_vgpr0[j];
      }
    };

    auto shared_to_vgpr = [&]() {
      for (int i = 0; i < MS; ++i)
        A_block_vgpr1[i] = A_shared[64 * i + (tid % 64)];
      for (int j = 0; j < NS / 4; ++j)
        B_block_vgpr1[j] = B_shared[256 * j + tid];
    };

    auto mfma = [&]() {
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          for (int k = 0; k < 2; ++k) {
            acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                A_block_vgpr1[i][k], B_block_vgpr1[j][k], acc[i][j], 0, 0, 0);
          }
        }
      }
    };

    auto sync = [] { __syncthreads(); };

    global_to_vgpr0();
    vpgr0_to_shared();
    if (K_outer >= 2) {
      global_to_vgpr0();
      sync();
      shared_to_vgpr();
      for (int k_outer = 0; k_outer < K_outer - 2; ++k_outer) {
        sync();
        vpgr0_to_shared();
        global_to_vgpr0();
        mfma();
        sync();
        shared_to_vgpr();
      }
      sync();
      vpgr0_to_shared();
      mfma();
    }
    sync();
    shared_to_vgpr();
    mfma();

    int32x4_t *C_ptr = static_cast<int32x4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

int main() {
  // std::printf("Best-performing kernels for each element types:\n\n");
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipeline_v3<
  //      8, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipelineload<
  //      12, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared<8, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<8, 8>());
  // test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 8>());

  // std::printf("\n\n\nOther kernels:\n\n");
  // test(MmtKernel_generic<Type::SI8, Type::SI8, Type::SI32, 3, 5, 2>());
  // test(MmtKernel_generic<Type::FP16, Type::FP16, Type::FP32, 3, 5, 2>());
  // test(MmtKernel_generic<Type::FP32, Type::FP32, Type::FP32, 3, 5, 2>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_32x32x16i8_shared<8, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_32x32x16i8_shared<4, 8>());
  // test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_rowmajor());
  // test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_directAB_rowmajorC());
  // test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct());
  // test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4());
  // test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4_unrollx4());
  // test(MmtKernel_128t_1x2_amdgcn_mfma_f32_16x16x4f32_direct());
  // test(MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_direct());
  // test(MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_shared());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<4, 4>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<4, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<8, 8>());
  // test(
  //     MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<8, 12>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<4, 4>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<4, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<8, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<8, 12>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<4, 4>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<4, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 12>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 16>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<16, 16>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_subgroup2x2<
  //      8, 8>());
  // test(
  //     MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_misguidednobankconflicts<
  //         8, 8>());
  // test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 12>());
  // test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 16>());
  // test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<16, 16>());
  // test(
  //     MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_naive<
  //         8, 8>());
  // test(
  //     MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_take2<
  //         8, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared<8, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2<8, 8>());
  // test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipelineload<
  //      8, 8>());

  // std::printf("\n\n\nSMFMAC sparse trick experiments:\n\n");
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared<4, 8>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared<8, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared<16, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle<4, 2>());
  test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle_v2<4, 2>());
  test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle_v2<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle_v2<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle_v3<4, 2>());
  test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle_v3<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_bswizzle_v3<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_i32_16x16x32i8_zeropad_shared<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_i32_16x16x32i8_zeropad_shared<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_i32_16x16x32i8_zeropad_shared<4, 8>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_i32_16x16x32i8_zeropad_shared<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_i32_16x16x32i8_zeropad_shared<8, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_i32_16x16x32i8_zeropad_shared<16, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared<4, 8>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared<8, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared<16, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_f32_16x16x16f16_zeropad_shared<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_f32_16x16x16f16_zeropad_shared<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_f32_16x16x16f16_zeropad_shared<4, 8>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_f32_16x16x16f16_zeropad_shared<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_f32_16x16x16f16_zeropad_shared<8, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_mfma_f32_16x16x16f16_zeropad_shared<16, 2>());

  // std::printf("\n\n\nSMFMAC unique-lane-data experiments:\n\n");
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata<4, 8>()); // LDS > 64KB
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata<8, 4>()); // LDS > 64KB
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata<16, 2>()); // LDS > 64KB
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata_v2<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata_v2<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_i32_16x16x64i8_shared_uniquedata_v2<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata<4, 8>()); // LDS > 64KB
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata<8, 4>()); // LDS > 64KB
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata<16, 2>()); // LDS > 64KB
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v2<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v2<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v2<8, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v3<4, 2>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v3<4, 4>());
  // test(MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v3<8, 2>());

  // std::printf("\n\n\nStream-K experiments:\n\n");
  // test(
  //     MmtKernel_StreamK_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<8,
  //                                                                         8>());
  // test(MmtKernel_StreamKTree_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<
  //      8, 8>());
  // test(
  //     MmtKernel_HybridStreamKTree_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<
  //         8, 8>());
}
