// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hip"

#include <cassert>
#include <cstdio>
#include <cxxabi.h>
#include <optional>
#include <random>
#include <typeinfo>
#include <vector>
#include <cstring>

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

    // M tile = MS * 16 = 128 
    // Total bytes in A tile (128 x 32)
    // M_tile x K_tile x sizeof(int8) = 128 * 32 * 1 = 4096 bytes
    // Each int8x8_t = 8 bytes
    // So A tile size in int8x8_t = 4096 / 8 = 512
    // this should really be MS * 
    // 1 * 16
    constexpr int A_tile_size_in_vec8 = MS * 16 * 4; // 8 * 16 * 4 = 512
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4; // 8 * 16 * 4 = 512

    const int8x8_t *A_global = static_cast<const int8x8_t *>(A_data) +
                               m_outer * K_outer * A_tile_size_in_vec8;
    const int8x8_t *B_global = static_cast<const int8x8_t *>(B_data) +
                               n_outer * K_outer * B_tile_size_in_vec8;
    const int8x8_t *A_global_ptr = A_global + tid;
    const int8x8_t *B_global_ptr = B_global + tid;

    // so we have enough shared memory for 8 MFMAs
    __shared__ int8x8_t A_shared[A_tile_size_in_vec8]; //512 * (64/8) = 4096 bytes / 8 = 512 per MFMA call which is 16 * 32 = correct
    __shared__ int8x8_t B_shared[B_tile_size_in_vec8];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) { 
        A_shared[i + tid] = A_global_ptr[i]; // 64bits load // 256 * 8 = 2048 bytes total, 2 iterations = 4096
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();

      // in total would run 16 times/16 MFMAs per tile
      for (int i = 0; i < MS; ++i) { 
        for (int j = 0; j < NS / 4; ++j) { 
            // i = 0 -> 8 gives us 512
            // j = 0 -> 2 gives us 512
            /*
            i = 0
              tid = 0 -> 64 * 0 + (0 % 64) = 0
              tid = 1 -> 64 * 0 + (1 % 64) = 1
              tid = 254 -> 64 * 0 + (254 % 64) = 62
              tid = 255 -> 64 * 0 + (255 % 64) = 63
            i = 1
              tid = 0 -> 64 * 1 + (0 % 64) = 64
              tid = 1 -> 64 * 1 + (1 % 64) = 65
              tid = 254 -> 64 * 1 + (254 % 64) = 126
              tid = 255 -> 64 * 1 + (255 % 64) = 127
            i = 2
              tid = 0 -> 64 * 2 + (0 % 64) = 128
              tid = 1 -> 64 * 2 + (1 % 64) = 129
              tid = 254 -> 64 * 2 + (254 % 64) = 190
              tid = 255 -> 64 * 2 + (255 % 64) = 191
            j = 0
              tid = 0 -> 256 * 0 + 0 = 0
              tid = 1 -> 256 * 0 + 1 = 1
              tid = 254 -> 256 * 0 + 254 = 254
              tid = 255 -> 256 * 0 + 255 = 255
            j = 1
              tid = 0 -> 256 * 1 + 0 = 256
              tid = 1 -> 256 * 1 + 1 = 257
              tid = 254 -> 256 * 1 + 254 = 510
              tid = 255 -> 256 * 1 + 255 = 511
            */
            /*
            j hits 4 MFMAs ( 256 / 64 = 4 subgroups) 
            // i broadcasts value to j each time
            */
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

template <int NS, int KS>
class MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x64i8_shared
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

   // Sparsity indices for the sparse trick
  static constexpr int SPARSITY_EVEN = 0x00004444;  // Selects positions 0,1 from each group of 4
  static constexpr int SPARSITY_ODD  = 0x0000EEEE;  // Selects positions 2,3 from each group of 4
  static constexpr int NB_BANKS = 32;

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 64; }  

  virtual tile_layout_func_t A_tile_layout() const override {
    // A tile: 8 rows × (KS * 64) K bytes
    // Row-major layout: A[m][k] at offset m * K_tile + k
    return [](int m, int k) {
      return m * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    // B tile: NS N-blocks × 16 N-cols × (KS * 64) K bytes
    // Layout: [n_block][n_col][k] where n = n_block * 16 + n_col
    // Offset = n_block * (KS * 1024) + n_col * (KS * 64) + k
    //        = (n / 16) * (KS * 1024) + (n % 16) * (KS * 64) + k
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 1024) + n_col * (KS * 64) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    // Row-major layout within the tile: C[m][n] at offset m * N_tile + n
    return [](int m, int n) {
      return m * (NS * 16) + n;
    };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using i8x8 = __attribute__((__vector_size__(8))) int8_t;    // 2 VGPRs for A
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;  // 4 VGPRs for B
    using i32x4 = __attribute__((__vector_size__(16))) int32_t; // 4 VGPRs for D
    union {
      i8x8 vec;
      int32_t i32[2];
    } a_reg;

    i32x4 acc[NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    int warp_id = tid / 64;

    constexpr int A_tile_size_in_vec16 = 8 * KS * 4; 
    constexpr int B_tile_size_in_vec16 = NS * KS * 64;

    const i8x16 *A_global = static_cast<const i8x16 *>(A_data) +
                            m_outer * K_outer * A_tile_size_in_vec16;
    const i8x16 *B_global = static_cast<const i8x16 *>(B_data) +
                            n_outer * K_outer * B_tile_size_in_vec16;
    const i8x16 *B_global_ptr = B_global + tid;

    __shared__ int8_t A_shared[KS * 512];
    __shared__ i8x16 B_shared[B_tile_size_in_vec16];

    int a_row_prod = tid / (KS * 4);
    int a_k_chunk = tid % (KS * 4);
    int a_bank = a_k_chunk / 4;
    int k_chunk_in_bank = a_k_chunk % 4;
    int a_col_prod = a_row_prod + k_chunk_in_bank * 8;

    int32_t *A_lds = reinterpret_cast<int32_t *>(A_shared);
    int a_m_row = (lane_id % 16) / 2;
    int a_is_odd = lane_id % 2;
    int a_k_range = lane_id / 16;
    int a_col_cons = a_m_row + a_k_range * 8;

    int n_col = lane_id % 16;
    int k_slice = lane_id / 16;
    int sparsity_idx = (lane_id & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      const i8x16 *A_tile_base = A_global + k_outer * A_tile_size_in_vec16;

      if (tid < A_tile_size_in_vec16) {
        auto global_idx = a_row_prod * (KS * 4) + a_k_chunk;
        i8x16 a_loaded = A_tile_base[a_row_prod * (KS * 4) + a_k_chunk];

        // Swizzle: reorder so even K positions are in first 8 bytes, odd in last 8
        // Input:  reg[0]=K0,K1,K2,K3  reg[1]=K4,K5,K6,K7
        //         reg[2]=K8,K9,K10,K11  reg[3]=K12,K13,K14,K15
        // Output: reg[0]=K0,K1,K4,K5  reg[1]=K8,K9,K12,K13
        //         reg[2]=K2,K3,K6,K7  reg[3]=K10,K11,K14,K15
        int32_t reg0 = reinterpret_cast<int32_t *>(&a_loaded)[0];
        int32_t reg1 = reinterpret_cast<int32_t *>(&a_loaded)[1];
        int32_t reg2 = reinterpret_cast<int32_t *>(&a_loaded)[2];
        int32_t reg3 = reinterpret_cast<int32_t *>(&a_loaded)[3];

        asm volatile(
            "v_pack_b32_f16 %0, %4, %5 op_sel:[0,0]\n\t"  // reg0 = lo(in0) | lo(in1)<<16
            "v_pack_b32_f16 %1, %6, %7 op_sel:[0,0]\n\t"  // reg1 = lo(in2) | lo(in3)<<16
            "v_pack_b32_f16 %2, %4, %5 op_sel:[1,1]\n\t"  // reg2 = hi(in0) | hi(in1)<<16
            "v_pack_b32_f16 %3, %6, %7 op_sel:[1,1]\n\t"  // reg3 = hi(in2) | hi(in3)<<16
            : "=&v"(reg0), "=&v"(reg1), "=&v"(reg2), "=&v"(reg3)
            : "v"(reinterpret_cast<int32_t *>(&a_loaded)[0]),
              "v"(reinterpret_cast<int32_t *>(&a_loaded)[1]),
              "v"(reinterpret_cast<int32_t *>(&a_loaded)[2]),
              "v"(reinterpret_cast<int32_t *>(&a_loaded)[3]));

        // Store to LDS
        // LDS layout: KS banks × 4 lines × 32 columns × 4 bytes
        // bank_base = A_lds + a_bank * 128 (128 int32s = 512 bytes per bank)
        int32_t *bank_base = A_lds + a_bank * 128;

        bank_base[0 * NB_BANKS + a_col_prod] = reg0;  // Line 0: even K bytes 0-3
        bank_base[1 * NB_BANKS + a_col_prod] = reg1;  // Line 1: even K bytes 4-7
        bank_base[2 * NB_BANKS + a_col_prod] = reg2;  // Line 2: odd K bytes 0-3
        bank_base[3 * NB_BANKS + a_col_prod] = reg3;  // Line 3: odd K bytes 4-7
      }

      // B tile: NS × KS × 1024 bytes = NS × KS × 64 vec16 elements
      for (int i = 0; i < B_tile_size_in_vec16; i += 256) {
        B_shared[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      // SMFMAC Loop: KS banks × (NS/4) N-block iterations
      for (int k_bank = 0; k_bank < KS; ++k_bank) {
        // A consumer load (8 bytes = 2 VGPRs) - same for all 4 wavefronts (broadcast)
        // Pointer formula:
        //   a_cons = A_shared + k_bank * 512 + a_col_cons * 4 + a_is_odd * 256
        // Justification:
        //   - k_bank * 512: bank offset (512 bytes per bank)
        //   - a_col_cons * 4: column offset (4 bytes per column)
        //   - a_is_odd * 256: line offset (lines 0-1 vs 2-3, 256 = 2 lines × 128 bytes/line)
        int8_t *a_cons = A_shared + k_bank * 512 + a_col_cons * 4 + a_is_odd * 256;
        int32_t *a_ptr = reinterpret_cast<int32_t *>(a_cons);
        a_reg.i32[0] = a_ptr[0];        // Line 0 or 2
        a_reg.i32[1] = a_ptr[NB_BANKS]; // Line 1 or 3

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp_id;

          // B consumer load (16 bytes = 4 VGPRs) - distributed per wavefront
          // B layout assumption: [n_block][n_col][k] (N-major for contiguous K access)
          // Pointer formula:
          //   b_cons = B_shared + n_block * (KS * 1024) + n_col * (KS * 64) + k_bank * 64 + k_slice * 16
          // Justification:
          //   - n_block * (KS * 1024): per N-block offset
          //   - n_col * (KS * 64): per N-column offset (each column has KS×64 K values)
          //   - k_bank * 64: which 64-K chunk within the N-column
          //   - k_slice * 16: which 16-K slice within the 64-K chunk
          int8_t *b_cons = reinterpret_cast<int8_t *>(B_shared)
                         + n_block * (KS * 1024)
                         + n_col * (KS * 64)
                         + k_bank * 64
                         + k_slice * 16;
          i8x16 b_reg = *reinterpret_cast<i8x16 *>(b_cons);

          int b_cons_offset = n_block * (KS * 1024) + n_col * (KS * 64) +
                              k_bank * 64 + k_slice * 16;
          // SMFMAC call
          // cbsz=0, abid=0: select bits [15:0] of sparsity index
          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(
              a_reg.vec, b_reg, acc[j], sparsity_idx, 0, 0);
        }
      }
      __syncthreads(); //is this necessary?
    }

    // Post-processing: Fuse partial sums from even/odd K positions
    //
    // With sparse trick, the SMFMAC computes:
    //   D[0][col] = physical_row_0 * B[col] using EVEN K positions (sparsity 0x4444)
    //   D[1][col] = physical_row_0 * B[col] using ODD K positions (sparsity 0xEEEE)
    //   D[2][col] = physical_row_1 * B[col] using EVEN K positions
    //   D[3][col] = physical_row_1 * B[col] using ODD K positions
    //
    // After fuse:
    //   d0 = D[0] + D[1] = complete result for physical_row_0, this lane's column
    //   d1 = D[2] + D[3] = complete result for physical_row_1, this lane's column
    for (int j = 0; j < NS / 4; ++j) {
      int32_t *regD = reinterpret_cast<int32_t *>(&acc[j]);
      regD[0] = regD[0] + regD[1];
      regD[1] = regD[2] + regD[3];
    }

    constexpr int N_stride = NS * 16;
    int m_pair = lane_id / 16;
    int m0 = m_pair * 2;
    int m1 = m_pair * 2 + 1;

    constexpr int tile_size = 8 * NS * 16;
    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      tile_size * (N_outer * m_outer + n_outer);

    for (int j = 0; j < NS / 4; ++j) {
      int n_block = j * 4 + warp_id;
      int n = n_block * 16 + n_col;

      int32_t *regD = reinterpret_cast<int32_t *>(&acc[j]);
      C_tile[m0 * N_stride + n] = regD[0];
      C_tile[m1 * N_stride + n] = regD[1];
    }
  }
};

// ==========================================================================
// CDNA4 I8 SMFMAC Sparse Trick Kernel: V_SMFMAC_I32_16X16X128_I8
//
// CDNA4 variant of the sparse trick kernel above, targeting the doubled K-depth
// instruction V_SMFMAC_I32_16X16X128_I8 (opcode 58, gfx950).
//
// The sparse trick is unchanged: 8 physical M-rows are mapped onto 16 logical
// SMFMAC M-rows by pairing adjacent lanes (even/odd) for the same physical row
// with complementary K positions (K%4 in {0,1} vs K%4 in {2,3}).
//
// Key differences from the CDNA3 16x16x64 variant:
//   - K per SMFMAC: 128 (vs 64). Each SMFMAC covers twice the K range.
//   - A operand: 4 VGPRs per lane (vs 2). The ISA index layout references
//     V0-V3, confirmed by A[16][64] compressed storage (ISA manual line 2644).
//   - B operand: 8 VGPRs per lane (vs 4). The ISA B layout table (line 2648)
//     shows v0-v7 covering K=0-127 in two non-contiguous 64-element blocks.
//   - D operand: 4 VGPRs per lane (unchanged, 16x16 output).
//   - Sparsity index: full 32-bit single set. CBSZ/ABID are ignored for
//     16x16x128 IU8 variants (ISA manual line 2500).
//   - Each pass covers TWO non-contiguous K blocks of 16 (e.g., pass 0 handles
//     K=0-15 AND K=64-79), per the B layout table. This requires the A LDS
//     layout to store low-K and high-K halves in separate line pairs.
//
// ISA reference: amdgpu-isa-manuals/cdna4/README.md
//   - Section 7.5.1.2 (line 2548): 8-bit A/B matrix layouts
//   - IU8 B layout table (line 2648): B[128][16] register mapping
//   - IU8 index layout (line 2673): 16 x 2-bit indices = 32-bit single set
//   - Line 2500: CBSZ/ABID ignored for 16x16x128 and 32x32x64 IU8 variants
// ==========================================================================
template <int NS, int KS>
class MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x128i8_shared
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  // KS <= 4 because A_tile_size_in_vec16 = KS*64 must fit in 256 threads.
  // KS >= 1 is valid since a single SMFMAC already covers K=128.
  static_assert(KS >= 1 && KS <= 4);

  // Sparsity indices for the sparse trick (CDNA4: full 32-bit, single set)
  //
  // Each 2-bit index selects which position within a group of 4 K-elements
  // the corresponding compressed byte represents. 16 indices x 2 bits = 32 bits.
  //
  // EVEN lanes: select K%4 in {0,1} -> indices [0,1,0,1,...] = 0x44444444
  //   Bit pattern: 01_00_01_00_01_00_01_00_01_00_01_00_01_00_01_00
  //
  // ODD lanes: select K%4 in {2,3} -> indices [2,3,2,3,...] = 0xEEEEEEEE
  //   Bit pattern: 11_10_11_10_11_10_11_10_11_10_11_10_11_10_11_10
  //
  // Compare CDNA3 (16-bit): EVEN=0x00004444, ODD=0x0000EEEE (8 indices in [15:0])
  static constexpr int SPARSITY_EVEN = 0x44444444;
  static constexpr int SPARSITY_ODD  = 0xEEEEEEEE;
  static constexpr int NB_BANKS = 32;

  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 128; }

  virtual tile_layout_func_t A_tile_layout() const override {
    // A tile: 8 rows x (KS * 128) K bytes (dense storage, sparse trick applied at LDS level)
    // Row-major: A[m][k] at element offset m * K_tile + k
    return [](int m, int k) {
      return m * (KS * 128) + k;
    };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    // B tile: NS N-blocks x 16 N-cols x (KS * 128) K bytes
    // Layout: [n_block][n_col][k] where n = n_block * 16 + n_col
    // N-block stride = 16 cols * KS * 128 = KS * 2048 elements
    // N-col stride = KS * 128 elements
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 2048) + n_col * (KS * 128) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    // Row-major within tile (unchanged from CDNA3)
    return [](int m, int n) {
      return m * (NS * 16) + n;
    };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {

    // Vector types for V_SMFMAC_I32_16X16X128_I8 operands:
    //   A: 4 VGPRs = 16 bytes (i32x4). ISA index layout references V0-V3,
    //      each VGPR holds 4 compressed i8 values. 16 values total per lane.
    //   B: 8 VGPRs = 32 bytes (i32x8). ISA B layout shows v0-v7 covering
    //      K=0-127 in two non-contiguous blocks per pass.
    //   D: 4 VGPRs = 16 bytes (i32x4). 16x16 output, same as CDNA3.
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;
    using i32x4 = __attribute__((__vector_size__(16))) int32_t;
    using i32x8 = __attribute__((__vector_size__(32))) int32_t;

    // A register: 4 VGPRs. V0-V1 hold low-K block data, V2-V3 hold high-K block data.
    union {
      i32x4 vec;
      int32_t i32[4];
    } a_reg;

    // B register: 8 VGPRs. halves[0] = v0-v3 (low-K), halves[1] = v4-v7 (high-K).
    union {
      i32x8 vec;
      i8x16 halves[2];
    } b_reg;

    i32x4 acc[NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;
    int lane_id = tid % 64;
    int warp_id = tid / 64;

    // A tile: 8 rows x KS banks x 8 chunks/bank x 16 bytes/chunk = KS * 1024 bytes
    // Each vec16 chunk is 16 bytes. Total chunks = 8 rows * KS * 8 = KS * 64.
    constexpr int A_tile_size_in_vec16 = 8 * KS * 8;
    // B tile: NS blocks x KS banks x 128 vec16 elements = NS * KS * 2048 bytes
    constexpr int B_tile_size_in_vec16 = NS * KS * 128;

    const i8x16 *A_global = static_cast<const i8x16 *>(A_data) +
                            m_outer * K_outer * A_tile_size_in_vec16;
    const i8x16 *B_global = static_cast<const i8x16 *>(B_data) +
                            n_outer * K_outer * B_tile_size_in_vec16;
    const i8x16 *B_global_ptr = B_global + tid;

    // Shared memory for A: KS banks x 1024 bytes/bank = KS * 1024 bytes
    // Each bank: 8 lines x 32 columns x 4 bytes
    //   Lines 0-1: EVEN K low-half  (V0, V1 data)
    //   Lines 2-3: EVEN K high-half (V2, V3 data)
    //   Lines 4-5: ODD K low-half
    //   Lines 6-7: ODD K high-half
    __shared__ int8_t A_shared[KS * 1024];
    __shared__ i8x16 B_shared[B_tile_size_in_vec16];

    // --- Producer coordinates for A ---
    // Each thread loads one 16-byte chunk (i8x16) from dense A in global memory.
    // With K=128 per SMFMAC and 8 chunks per bank:
    //   a_row_prod: which of 8 physical M-rows (0..7)
    //   a_k_chunk:  which 16-byte chunk along K (0..KS*8-1)
    //   a_bank:     which K-bank (0..KS-1), one bank per SMFMAC iteration
    //   k_chunk_in_bank: position within the bank (0..7)
    //
    // The mapping k_range = k_chunk_in_bank % 4, k_half = k_chunk_in_bank / 4
    // reflects the ISA's non-contiguous per-pass K coverage:
    //   Per the B layout table (ISA line 2648), pass 0 covers K=0-15 AND K=64-79.
    //   Chunks 0-3 (K=0-63) are the low-K blocks for passes 0-3.
    //   Chunks 4-7 (K=64-127) are the high-K blocks for passes 0-3.
    //   So chunk 0 -> pass 0 low, chunk 4 -> pass 0 high,
    //      chunk 1 -> pass 1 low, chunk 5 -> pass 1 high, etc.
    int a_row_prod = tid / (KS * 8);
    int a_k_chunk = tid % (KS * 8);
    int a_bank = a_k_chunk / 8;
    int k_chunk_in_bank = a_k_chunk % 8;
    int k_range = k_chunk_in_bank % 4;  // which pass (0..3)
    int k_half = k_chunk_in_bank / 4;   // low(0) or high(1) K block
    int a_col_prod = a_row_prod + k_range * 8;

    int32_t *A_lds = reinterpret_cast<int32_t *>(A_shared);

    // --- Consumer coordinates for A (same pattern as CDNA3) ---
    // Lane pairing for sparse trick:
    //   a_m_row: physical M-row (0..7), shared by adjacent even/odd lanes
    //   a_is_odd: 0 for EVEN K (K%4 in {0,1}), 1 for ODD K (K%4 in {2,3})
    //   a_k_range: which pass/lane-group (0..3), determines K coverage
    //   a_col_cons: LDS column encoding both M-row and pass
    int a_m_row = (lane_id % 16) / 2;
    int a_is_odd = lane_id % 2;
    int a_k_range = lane_id / 16;
    int a_col_cons = a_m_row + a_k_range * 8;

    // B consumer coordinates (same as CDNA3)
    int n_col = lane_id % 16;
    int k_slice = lane_id / 16;
    int sparsity_idx = (lane_id & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      const i8x16 *A_tile_base = A_global + k_outer * A_tile_size_in_vec16;

      // --- Load A tile: swizzle and store to LDS ---
      // Each thread loads 16 contiguous i8 values from global A, then uses
      // v_pack_b32_f16 to separate even K positions (K%4 in {0,1}) from odd
      // (K%4 in {2,3}), storing them to different LDS line groups.
      //
      // The swizzle is identical to CDNA3 since the fundamental unit
      // (16 i8 values -> 4 groups of 4 -> separate EVEN/ODD halves) is unchanged.
      if (tid < A_tile_size_in_vec16) {
        i8x16 a_loaded = A_tile_base[a_row_prod * (KS * 8) + a_k_chunk];

        // v_pack_b32_f16 swizzle: separate even/odd K positions
        // Input:  reg[0]=[K0,K1,K2,K3]  reg[1]=[K4,K5,K6,K7]
        //         reg[2]=[K8,K9,K10,K11] reg[3]=[K12,K13,K14,K15]
        // Output: reg0=[K0,K1,K4,K5]   reg1=[K8,K9,K12,K13]   (EVEN: K%4 in {0,1})
        //         reg2=[K2,K3,K6,K7]   reg3=[K10,K11,K14,K15] (ODD: K%4 in {2,3})
        int32_t reg0, reg1, reg2, reg3;
        asm volatile(
            "v_pack_b32_f16 %0, %4, %5 op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %1, %6, %7 op_sel:[0,0]\n\t"
            "v_pack_b32_f16 %2, %4, %5 op_sel:[1,1]\n\t"
            "v_pack_b32_f16 %3, %6, %7 op_sel:[1,1]\n\t"
            : "=&v"(reg0), "=&v"(reg1), "=&v"(reg2), "=&v"(reg3)
            : "v"(reinterpret_cast<int32_t *>(&a_loaded)[0]),
              "v"(reinterpret_cast<int32_t *>(&a_loaded)[1]),
              "v"(reinterpret_cast<int32_t *>(&a_loaded)[2]),
              "v"(reinterpret_cast<int32_t *>(&a_loaded)[3]));

        // Store to LDS: 8 lines per bank (1024 bytes = 256 int32s per bank)
        //
        // Line assignment (k_half selects within the EVEN or ODD group):
        //   k_half=0 (low-K): EVEN -> lines 0,1; ODD -> lines 4,5
        //   k_half=1 (high-K): EVEN -> lines 2,3; ODD -> lines 6,7
        //
        // This ensures the consumer reads 4 consecutive lines for 4 VGPRs:
        //   EVEN consumer: lines 0,1,2,3 -> V0(low), V1(low), V2(high), V3(high)
        //   ODD consumer:  lines 4,5,6,7 -> V0(low), V1(low), V2(high), V3(high)
        int32_t *bank_base = A_lds + a_bank * 256;  // 256 int32 = 1024 bytes per bank

        bank_base[(k_half * 2 + 0) * NB_BANKS + a_col_prod] = reg0;  // EVEN: line 0 or 2
        bank_base[(k_half * 2 + 1) * NB_BANKS + a_col_prod] = reg1;  // EVEN: line 1 or 3
        bank_base[(4 + k_half * 2 + 0) * NB_BANKS + a_col_prod] = reg2;  // ODD: line 4 or 6
        bank_base[(4 + k_half * 2 + 1) * NB_BANKS + a_col_prod] = reg3;  // ODD: line 5 or 7
      }

      // B tile: NS x KS x 2048 bytes = NS * KS * 128 vec16 elements (dense, no swizzle)
      for (int i = 0; i < B_tile_size_in_vec16; i += 256) {
        B_shared[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      // --- SMFMAC Loop: KS banks x (NS/4) N-block iterations ---
      for (int k_bank = 0; k_bank < KS; ++k_bank) {
        // A consumer load: 4 VGPRs from 4 consecutive LDS lines
        //
        // Pointer formula:
        //   a_cons = A_shared + k_bank * 1024 + a_col_cons * 4 + a_is_odd * 512
        //
        // Justification:
        //   - k_bank * 1024: bank offset (1024 bytes per bank, doubled from CDNA3's 512)
        //   - a_col_cons * 4: column offset (4 bytes per int32 column)
        //   - a_is_odd * 512: selects EVEN (lines 0-3) or ODD (lines 4-7)
        //     512 = 4 lines * 32 cols * 4 bytes = 4 * 128 bytes
        //
        // Reads 4 VGPRs from consecutive lines at stride NB_BANKS (= 32 int32 = 128 bytes):
        //   a_reg.i32[0] = line 0 (or 4) -> V0 (low-K swizzled data)
        //   a_reg.i32[1] = line 1 (or 5) -> V1 (low-K swizzled data)
        //   a_reg.i32[2] = line 2 (or 6) -> V2 (high-K swizzled data)
        //   a_reg.i32[3] = line 3 (or 7) -> V3 (high-K swizzled data)
        //
        // Verification for EVEN lane at k_range=0 (pass 0):
        //   V0 <- [K0,K1,K4,K5]       (low-K EVEN, from chunk 0)
        //   V1 <- [K8,K9,K12,K13]     (low-K EVEN, from chunk 0)
        //   V2 <- [K64,K65,K68,K69]   (high-K EVEN, from chunk 4)
        //   V3 <- [K72,K73,K76,K77]   (high-K EVEN, from chunk 4)
        //   Sparsity 0x44444444: each index = [0,1,0,1,...] confirming K%4 in {0,1}
        //   Matches B pass 0 coverage: K=0-15 and K=64-79
        int8_t *a_cons = A_shared + k_bank * 1024 + a_col_cons * 4 + a_is_odd * 512;
        int32_t *a_ptr = reinterpret_cast<int32_t *>(a_cons);
        a_reg.i32[0] = a_ptr[0 * NB_BANKS];
        a_reg.i32[1] = a_ptr[1 * NB_BANKS];
        a_reg.i32[2] = a_ptr[2 * NB_BANKS];
        a_reg.i32[3] = a_ptr[3 * NB_BANKS];

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp_id;

          // B consumer load: 8 VGPRs via two 16-byte loads
          //
          // B is stored K-contiguous per N-column: B[n_block][n_col][k]
          // Each SMFMAC needs 128 K values per N-column (128 bytes per bank).
          //
          // Per the ISA B layout table (line 2648), each pass covers two
          // non-contiguous 16-K blocks:
          //   v0-v3 (16 bytes): K in [k_slice*16, k_slice*16+15] (low block, K=0-63)
          //   v4-v7 (16 bytes): K in [64+k_slice*16, 64+k_slice*16+15] (high block, K=64-127)
          //
          // Pointer formula:
          //   b_cons = B_shared + n_block*(KS*2048) + n_col*(KS*128) + k_bank*128 + k_slice*16
          //   b_lo = b_cons (16 bytes for v0-v3)
          //   b_hi = b_cons + 64 (16 bytes for v4-v7, separated by 64 bytes = K=64 boundary)
          int8_t *b_cons = reinterpret_cast<int8_t *>(B_shared)
                         + n_block * (KS * 2048)
                         + n_col * (KS * 128)
                         + k_bank * 128
                         + k_slice * 16;
          b_reg.halves[0] = *reinterpret_cast<i8x16 *>(b_cons);       // v0-v3: low-K block
          b_reg.halves[1] = *reinterpret_cast<i8x16 *>(b_cons + 64);  // v4-v7: high-K block

          // V_SMFMAC_I32_16X16X128_I8 (opcode 58, gfx950)
          //   D = A(sparse 16x128) * B(128x16) + D(16x16)
          //   A: i32x4 (4 VGPRs, 16 compressed i8 values)
          //   B: i32x8 (8 VGPRs, 32 dense i8 values)
          //   D: i32x4 (4 VGPRs, 16x16 output accumulated)
          //   sparsity_idx: int32 (16 x 2-bit indices, single set)
          //   cbsz=0, abid=0: ignored for 16x16x128 IU8 (ISA line 2500)
          acc[j] = __builtin_amdgcn_smfmac_i32_16x16x128_i8(
              a_reg.vec, b_reg.vec, acc[j], sparsity_idx, 0, 0);
        }
      }
      __syncthreads();
    }

    // Post-processing: Fuse partial sums from even/odd K positions
    //
    // With the sparse trick, the SMFMAC output layout is (same as CDNA3):
    //   D[0][col] = physical_row_0 * B[col] using EVEN K (sparsity 0x44444444)
    //   D[1][col] = physical_row_0 * B[col] using ODD K  (sparsity 0xEEEEEEEE)
    //   D[2][col] = physical_row_1 * B[col] using EVEN K
    //   D[3][col] = physical_row_1 * B[col] using ODD K
    //
    // Fuse: add even and odd partial sums for complete dense results
    for (int j = 0; j < NS / 4; ++j) {
      int32_t *regD = reinterpret_cast<int32_t *>(&acc[j]);
      regD[0] = regD[0] + regD[1];  // complete result for physical_row_0
      regD[1] = regD[2] + regD[3];  // complete result for physical_row_1
    }

    // C matrix store (identical to CDNA3)
    constexpr int N_stride = NS * 16;
    int m_pair = lane_id / 16;
    int m0 = m_pair * 2;
    int m1 = m_pair * 2 + 1;

    constexpr int tile_size = 8 * NS * 16;
    int32_t *C_tile = static_cast<int32_t *>(C_data) +
                      tile_size * (N_outer * m_outer + n_outer);

    for (int j = 0; j < NS / 4; ++j) {
      int n_block = j * 4 + warp_id;
      int n = n_block * 16 + n_col;

      int32_t *regD = reinterpret_cast<int32_t *>(&acc[j]);
      C_tile[m0 * N_stride + n] = regD[0];
      C_tile[m1 * N_stride + n] = regD[1];
    }
  }
};

// ==========================================================================
// F16 SMFMAC Sparse Trick Kernel: V_SMFMAC_F32_16x16x32_F16
//
// This kernel uses the same "sparse trick" as the I8 SMFMAC variant above:
//   - 8 physical M-rows are mapped onto 16 logical SMFMAC M-rows by pairing
//     adjacent lanes: even-lane-ID threads handle EVEN K positions, odd-lane-ID
//     threads handle ODD K positions for the *same* physical row.
//   - The SMFMAC instruction's built-in 4:2 structured sparsity support
//     selects the appropriate K positions via compression indices.
//   - After the SMFMAC loop, partial sums from even/odd K halves are fused
//     (added) to produce the complete dense result.
//
// Key differences from the I8 variant (smfmac_i32_16x16x64_i8):
//
//   1. K per SMFMAC: 32 (not 64). But f16=2 bytes vs i8=1 byte, so
//      byte counts are identical: 32*2 = 64*1 = 64 bytes per K-range.
//      This means all shared memory sizes, tile sizes in vec16 units,
//      and byte-level pointer arithmetic remain unchanged.
//
//   2. Sparsity indices: 8-bit (0x44/0xEE) not 16-bit (0x4444/0xEEEE).
//      For V_SMFMAC_F32_16x16x32_F16 with CBSZ=0, ABID=0, the hardware
//      reads bits[7:0] of the index VGPR. Each 4-bit nibble encodes two
//      2-bit selectors choosing 2-of-4 K positions in a group.
//      (For I8, bits[15:0] are read — 4 nibbles for 4 groups of 4.)
//
//   3. A swizzle is trivial: no v_pack_b32_f16 assembly needed.
//      For f16, each 32-bit register holds exactly 2 f16 values, which
//      are naturally aligned to even/odd K-position boundaries:
//        Input:  reg[0]=[K0,K1]  reg[1]=[K2,K3]  reg[2]=[K4,K5]  reg[3]=[K6,K7]
//        EVEN:   reg[0]=[K0,K1], reg[2]=[K4,K5]
//        ODD:    reg[1]=[K2,K3], reg[3]=[K6,K7]
//      The "swizzle" is just storing registers in order [0, 2, 1, 3].
//
//   4. Accumulator and output are f32 (not i32).
//
//   5. Tile layout functions use f16-element offsets:
//        A: m*(KS*32)+k     (vs m*(KS*64)+k for i8)
//        B: n_blk*(KS*512)+n_col*(KS*32)+k  (vs n_blk*(KS*1024)+n_col*(KS*64)+k)
//
// Register layout (verified via matrix_calculator.py for CDNA3):
//
//   A (sparse, 2 VGPRs = f16x4):
//     Lane i (0..15):  v0 = A[i][0..3], v1 = A[i][4..7]   (K-range 0..7)
//     Lane i+16:       v0 = A[i][8..11], v1 = A[i][12..15] (K-range 8..15)
//     Lane i+32:       v0 = A[i][16..19], v1 = A[i][20..23] (K-range 16..23)
//     Lane i+48:       v0 = A[i][24..27], v1 = A[i][28..31] (K-range 24..31)
//     Each VGPR holds 2 f16 values; sparsity selects which 2-of-4 they represent.
//
//   B (dense, 4 VGPRs = f16x8):
//     Lane j (0..15):  v0=[B[0][j],B[1][j]], v1=[B[2][j],B[3][j]], ...
//                      v3=[B[6][j],B[7][j]]  (K 0..7, N-col j)
//     Lane j+16:       K 8..15, N-col j
//     Lane j+32:       K 16..23, N-col j
//     Lane j+48:       K 24..31, N-col j
//
//   D (4 VGPRs = f32x4):
//     Lane j (0..15):  v0=D[0][j], v1=D[1][j], v2=D[2][j], v3=D[3][j]
//     Lane j+16:       D[4..7][j]
//     Lane j+32:       D[8..11][j]
//     Lane j+48:       D[12..15][j]
//
//   Compression index (8 bits per lane, CBSZ=0, ABID=0 → bits[7:0]):
//     bits[3:0]: 2-of-4 selector for K group {k, k+1, k+2, k+3}
//     bits[7:4]: 2-of-4 selector for K group {k+4, k+5, k+6, k+7}
//     0x44 = 0100_0100: positions {0,1} from each group (EVEN)
//     0xEE = 1110_1110: positions {2,3} from each group (ODD)
// ==========================================================================
template <int NS, int KS>
class MmtKernel_256t_MSxNS_amdgcn_smfmac_f32_16x16x32f16_shared
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  static_assert(KS >= 2 && KS <= 8 && !(KS % 2));

  // Sparsity indices for the sparse trick (F16: 8-bit compression indices).
  //
  // For V_SMFMAC_F32_16x16x32_F16 with CBSZ=0, ABID=0:
  //   Hardware reads bits[7:0] of the sparsity index register.
  //   Each 4-bit nibble contains two 2-bit selectors for one group of 4 K-values.
  //     nibble 0x4 = 01_00 → selects positions {0, 1}
  //     nibble 0xE = 11_10 → selects positions {2, 3}
  //
  // EVEN: both nibbles = 0x4 → 0x44 → selects K0,K1 and K4,K5
  // ODD:  both nibbles = 0xE → 0xEE → selects K2,K3 and K6,K7
  static constexpr int SPARSITY_EVEN = 0x00000044;
  static constexpr int SPARSITY_ODD  = 0x000000EE;
  static constexpr int NB_BANKS = 32;

  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 8; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return KS * 32; }

  virtual tile_layout_func_t A_tile_layout() const override {
    // A tile: 8 rows × (KS * 32) f16 K-elements, row-major.
    // Element offset: A[m][k] = m * K_tile + k
    return [](int m, int k) {
      return m * (KS * 32) + k;
    };
  }

  virtual tile_layout_func_t B_tile_layout() const override {
    // B tile: NS N-blocks × 16 N-cols × (KS * 32) f16 K-elements.
    // Layout: [n_block][n_col][k] where n = n_block * 16 + n_col
    //
    // Element offset: n_block * (KS * 512) + n_col * (KS * 32) + k
    //   where KS * 512 = 16 cols × KS × 32 K-elements/col
    //
    // In bytes: n_block * (KS * 1024) + n_col * (KS * 64) + k * 2
    // Note that the byte-level layout is identical to the I8 variant because
    // 32 f16 × 2 bytes = 64 bytes = 64 i8 × 1 byte per K-range.
    return [](int n, int k) {
      int n_block = n / 16;
      int n_col = n % 16;
      return n_block * (KS * 512) + n_col * (KS * 32) + k;
    };
  }

  virtual tile_layout_func_t C_tile_layout() const override {
    // Row-major layout within the tile: C[m][n] at offset m * N_tile + n
    return [](int m, int n) {
      return m * (NS * 16) + n;
    };
  }

  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };

  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    // Vector types for V_SMFMAC_F32_16x16x32_F16 operands:
    //   A: f16x4 = 4 × _Float16 = 8 bytes = 2 VGPRs (sparse compressed input)
    //   B: f16x8 = 8 × _Float16 = 16 bytes = 4 VGPRs (dense input)
    //   D: f32x4 = 4 × float = 16 bytes = 4 VGPRs (accumulator)
    using f16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using f16x8 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    // i8x16: 16-byte vector for type-agnostic global/shared memory loads.
    // The SMFMAC instruction interprets the register contents as f16 regardless
    // of the pointer type used to load them.
    using i8x16 = __attribute__((__vector_size__(16))) int8_t;

    // Union for A register: load as int32 words from LDS, pass as f16x4
    // to the SMFMAC intrinsic.
    //   i32[0] = v0 = 2 packed f16 values (first half of K-range)
    //   i32[1] = v1 = 2 packed f16 values (second half of K-range)
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

    // Tile sizes in 16-byte (vec16) units.
    //
    // These are byte-identical to the I8 kernel because:
    //   A: 8 rows × KS × 32 f16 × 2 bytes = KS × 512 bytes = KS × 32 vec16
    //   B: NS × 16 cols × KS × 32 f16 × 2 bytes = NS × KS × 1024 bytes
    //      = NS × KS × 64 vec16
    constexpr int A_tile_size_in_vec16 = 8 * KS * 4;
    constexpr int B_tile_size_in_vec16 = NS * KS * 64;

    const i8x16 *A_global = static_cast<const i8x16 *>(A_data) +
                            m_outer * K_outer * A_tile_size_in_vec16;
    const i8x16 *B_global = static_cast<const i8x16 *>(B_data) +
                            n_outer * K_outer * B_tile_size_in_vec16;
    const i8x16 *B_global_ptr = B_global + tid;

    // Shared memory layout (byte counts identical to I8 kernel):
    //   A_shared: KS banks × 4 lines × 32 columns × 4 bytes = KS × 512 bytes
    //     Each bank stores one SMFMAC's worth of A data (32 K-values × 8 rows).
    //     Lines 0-1: EVEN K data (positions {0,1} and {4,5} from each group of 4)
    //     Lines 2-3: ODD K data (positions {2,3} and {6,7} from each group of 4)
    //   B_shared: NS × KS × 64 vec16 = NS × KS × 1024 bytes
    __shared__ int8_t A_shared[KS * 512];
    __shared__ i8x16 B_shared[B_tile_size_in_vec16];

    // === A tile producer coordinates ===
    // Each producer thread loads one vec16 (16 bytes = 8 f16 values) from global
    // memory and stores 4 swizzled int32 words to A_shared.
    //
    // Thread mapping (identical to I8):
    //   a_row_prod: which of the 8 physical M-rows (0..7)
    //   a_k_chunk: which 16-byte chunk along K (0..KS*4-1)
    //   a_bank: which K-bank this chunk belongs to (0..KS-1)
    //   k_chunk_in_bank: position within the bank (0..3)
    //   a_col_prod: LDS column index = a_row_prod + k_chunk_in_bank * 8
    //
    // Only threads 0..A_tile_size_in_vec16-1 participate in A production.
    int a_row_prod = tid / (KS * 4);
    int a_k_chunk = tid % (KS * 4);
    int a_bank = a_k_chunk / 4;
    int k_chunk_in_bank = a_k_chunk % 4;
    int a_col_prod = a_row_prod + k_chunk_in_bank * 8;

    int32_t *A_lds = reinterpret_cast<int32_t *>(A_shared);

    // === A tile consumer coordinates ===
    // The sparse trick maps pairs of adjacent lanes to the same physical M-row:
    //   a_m_row = (lane_id % 16) / 2: physical row (0..7)
    //   a_is_odd = lane_id % 2: 0 for EVEN K, 1 for ODD K
    //   a_k_range = lane_id / 16: which K-range group (0..3)
    //
    // The intrinsic sees 16 logical rows:
    //   Logical row 2*r+0 = physical row r, EVEN K
    //   Logical row 2*r+1 = physical row r, ODD K
    int a_m_row = (lane_id % 16) / 2;
    int a_is_odd = lane_id % 2;
    int a_k_range = lane_id / 16;
    int a_col_cons = a_m_row + a_k_range * 8;

    // B consumer coordinates:
    //   n_col = lane_id % 16: which N-column within an N-block
    //   k_slice = lane_id / 16: which 8-K-value slice (0..3)
    int n_col = lane_id % 16;
    int k_slice = lane_id / 16;

    // Sparsity index: even-numbered lanes use EVEN, odd-numbered lanes use ODD.
    int sparsity_idx = (lane_id & 1) ? SPARSITY_ODD : SPARSITY_EVEN;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      const i8x16 *A_tile_base = A_global + k_outer * A_tile_size_in_vec16;

      if (tid < A_tile_size_in_vec16) {
        auto global_idx = a_row_prod * (KS * 4) + a_k_chunk;
        i8x16 a_loaded = A_tile_base[a_row_prod * (KS * 4) + a_k_chunk];

        // ==================================================================
        // F16 A swizzle: simple register reorder (NO v_pack assembly needed).
        //
        // Each 32-bit register holds exactly 2 packed f16 values.
        // The 16-byte load yields 8 consecutive f16 K-values in 4 registers:
        //
        //   reg[0] = [K0, K1]   (2 f16, positions 0-1 of first group of 4)
        //   reg[1] = [K2, K3]   (2 f16, positions 2-3 of first group of 4)
        //   reg[2] = [K4, K5]   (2 f16, positions 0-1 of second group of 4)
        //   reg[3] = [K6, K7]   (2 f16, positions 2-3 of second group of 4)
        //
        // EVEN sparsity (0x44) selects positions {0,1} from each group of 4,
        // so the hardware expects:
        //   v0 = [K0, K1] = reg[0]     (from group 0..3)
        //   v1 = [K4, K5] = reg[2]     (from group 4..7)
        //
        // ODD sparsity (0xEE) selects positions {2,3} from each group of 4:
        //   v0 = [K2, K3] = reg[1]
        //   v1 = [K6, K7] = reg[3]
        //
        // LDS store order: line 0 = reg[0], line 1 = reg[2],
        //                  line 2 = reg[1], line 3 = reg[3]
        //
        // Compare to I8 where v_pack_b32_f16 was needed because each i32
        // holds 4 i8 values and even/odd separation required extracting
        // alternating 16-bit halves across registers.
        // ==================================================================
        int32_t reg0 = reinterpret_cast<int32_t *>(&a_loaded)[0];  // [K0, K1]
        int32_t reg1 = reinterpret_cast<int32_t *>(&a_loaded)[1];  // [K2, K3]
        int32_t reg2 = reinterpret_cast<int32_t *>(&a_loaded)[2];  // [K4, K5]
        int32_t reg3 = reinterpret_cast<int32_t *>(&a_loaded)[3];  // [K6, K7]

        // Store to LDS with even/odd separation.
        // LDS layout: KS banks × 4 lines × 32 columns × 4 bytes
        // bank_base = A_lds + a_bank * 128 (128 int32s = 512 bytes per bank)
        int32_t *bank_base = A_lds + a_bank * 128;

        bank_base[0 * NB_BANKS + a_col_prod] = reg0;  // Line 0: EVEN [K0, K1]
        bank_base[1 * NB_BANKS + a_col_prod] = reg2;  // Line 1: EVEN [K4, K5]
        bank_base[2 * NB_BANKS + a_col_prod] = reg1;  // Line 2: ODD  [K2, K3]
        bank_base[3 * NB_BANKS + a_col_prod] = reg3;  // Line 3: ODD  [K6, K7]
      }

      // B tile: NS × KS × 1024 bytes = NS × KS × 64 vec16 elements.
      // Loaded identically to I8 (same byte count per tile).
      for (int i = 0; i < B_tile_size_in_vec16; i += 256) {
        B_shared[i + tid] = B_global_ptr[i];
      }
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      // SMFMAC Loop: KS banks × (NS/4) N-block iterations
      for (int k_bank = 0; k_bank < KS; ++k_bank) {
        // A consumer load (8 bytes = 2 VGPRs = 4 packed f16 values).
        // Pointer formula (byte-identical to I8):
        //   k_bank * 512: bank offset (512 bytes per bank)
        //   a_col_cons * 4: column offset (4 bytes per int32 column)
        //   a_is_odd * 256: line offset (lines 0-1 for EVEN, lines 2-3 for ODD)
        //
        // Loads two int32 words separated by NB_BANKS (one line stride):
        //   i32[0] = line 0 or 2 → SMFMAC v0 (first 2 f16 of compressed K)
        //   i32[1] = line 1 or 3 → SMFMAC v1 (second 2 f16 of compressed K)
        int8_t *a_cons = A_shared + k_bank * 512 + a_col_cons * 4 + a_is_odd * 256;
        int32_t *a_ptr = reinterpret_cast<int32_t *>(a_cons);
        a_reg.i32[0] = a_ptr[0];        // Line 0 or 2
        a_reg.i32[1] = a_ptr[NB_BANKS]; // Line 1 or 3

        for (int j = 0; j < NS / 4; ++j) {
          int n_block = j * 4 + warp_id;

          // B consumer load (16 bytes = 4 VGPRs = 8 packed f16 values).
          // Byte-level formula identical to I8:
          //   n_block * (KS * 1024): per N-block offset
          //   n_col * (KS * 64): per N-column offset (KS × 32 f16 × 2 = KS × 64 bytes)
          //   k_bank * 64: which 32-K-value bank (32 × 2 = 64 bytes)
          //   k_slice * 16: which 8-K-value slice (8 × 2 = 16 bytes)
          int8_t *b_cons = reinterpret_cast<int8_t *>(B_shared)
                         + n_block * (KS * 1024)
                         + n_col * (KS * 64)
                         + k_bank * 64
                         + k_slice * 16;
          f16x8 b_reg = *reinterpret_cast<f16x8 *>(b_cons);

          // V_SMFMAC_F32_16x16x32_F16: sparse f16 matmul with f32 accumulation.
          // cbsz=0, abid=0: hardware reads bits[7:0] of sparsity_idx.
          acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(
              a_reg.vec, b_reg, acc[j], sparsity_idx, 0, 0);
        }
      }
      __syncthreads(); //is this necessary?
    }

    // Post-processing: Fuse partial sums from even/odd K positions.
    //
    // With the sparse trick, the SMFMAC computes (per lane):
    //   D[0][col] = physical_row_0 × B[col] using EVEN K positions (sparsity 0x44)
    //   D[1][col] = physical_row_0 × B[col] using ODD K positions  (sparsity 0xEE)
    //   D[2][col] = physical_row_1 × B[col] using EVEN K positions
    //   D[3][col] = physical_row_1 × B[col] using ODD K positions
    //
    // After fuse (float addition for f32 accumulators):
    //   d0 = D[0] + D[1] = complete result for physical_row_0, this lane's column
    //   d1 = D[2] + D[3] = complete result for physical_row_1, this lane's column
    for (int j = 0; j < NS / 4; ++j) {
      float *regD = reinterpret_cast<float *>(&acc[j]);
      regD[0] = regD[0] + regD[1];
      regD[1] = regD[2] + regD[3];
    }

    // C store: write fused results to global memory (row-major, float).
    //
    // m_pair = lane_id / 16 determines which pair of physical M-rows:
    //   m_pair 0 → rows 0,1; m_pair 1 → rows 2,3; etc.
    // This is because the D layout maps 4 consecutive rows to each 16-lane group,
    // and the sparse trick pairs them as (even_row, odd_row).
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
  std::printf("Best-performing kernels for each element types:\n\n");
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipeline_v3<
       8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipelineload<
       12, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<8, 8>());
  test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 8>());

  std::printf("\n\n\nOther kernels:\n\n");
  test(MmtKernel_generic<Type::SI8, Type::SI8, Type::SI32, 3, 5, 2>());
  test(MmtKernel_generic<Type::FP16, Type::FP16, Type::FP32, 3, 5, 2>());
  test(MmtKernel_generic<Type::FP32, Type::FP32, Type::FP32, 3, 5, 2>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_32x32x16i8_shared<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_32x32x16i8_shared<4, 8>());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_rowmajor());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_directAB_rowmajorC());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4_unrollx4());
  test(MmtKernel_128t_1x2_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_shared());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<4, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<8, 8>());
  test(
      MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<8, 12>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<4, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<8, 12>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<4, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 12>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 16>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<16, 16>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_subgroup2x2<
       8, 8>());
  test(
      MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_misguidednobankconflicts<
          8, 8>());
  test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 12>());
  test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 16>());
  test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<16, 16>());
  test(
      MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_naive<
          8, 8>());
  test(
      MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_take2<
          8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipelineload<
       8, 8>());

  std::printf("\n\n\nSMFMAC sparse trick INT8 experiments:\n\n");
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x64i8_shared<4, 2>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x64i8_shared<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x64i8_shared<4, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x64i8_shared<8, 2>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x64i8_shared<8, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x64i8_shared<16, 2>());

  std::printf("\n\n\nSMFMAC sparse trick INT8 CDNA4 (16x16x128) experiments:\n\n");
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x128i8_shared<4, 1>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x128i8_shared<4, 2>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x128i8_shared<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x128i8_shared<8, 2>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_i32_16x16x128i8_shared<16, 2>());

  std::printf("\n\n\nSMFMAC sparse trick FP16 experiments:\n\n");
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_f32_16x16x32f16_shared<4, 2>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_f32_16x16x32f16_shared<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_f32_16x16x32f16_shared<4, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_f32_16x16x32f16_shared<8, 2>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_f32_16x16x32f16_shared<8, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_smfmac_f32_16x16x32f16_shared<16, 2>());

  std::printf("\n\n\nStream-K experiments:\n\n");
  test(
      MmtKernel_StreamK_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<8,
                                                                          8>());
  test(MmtKernel_StreamKTree_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<
       8, 8>());
  test(
      MmtKernel_HybridStreamKTree_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<
          8, 8>());
}
