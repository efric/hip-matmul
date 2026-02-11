#include "common.hip"

#include <cstdint>
#include <numeric>

constexpr int kSubgroupSize = 32;
constexpr int kBytesPerPair = 16;
constexpr int kNumPairs = kSubgroupSize / 2;
constexpr int kInputElements = kNumPairs * kBytesPerPair;   // 256 bytes
constexpr int kBytesPerThread = 8;
constexpr int kOutputElements = kSubgroupSize * kBytesPerThread; // 256 bytes

// Swizzle: deinterleave 16-bit pairs across a thread pair.
//   T_even: K0  K1  K4  K5  K8  K9  K12 K13   (lo16 of each dword)
//   T_odd:  K2  K3  K6  K7  K10 K11 K14 K15   (hi16 of each dword)
__global__ void load_i8x16(const int8_t *__restrict__ in,
                           int8_t *__restrict__ out) {
  int tid = threadIdx.x;
  int pair_id = tid / 2;
  int lane_in_pair = tid & 1;

  // Each thread loads int2 (8 bytes). Even lane: dwords 0-1, odd lane: dwords 2-3.
  int load_offset = pair_id * kBytesPerPair + lane_in_pair * 8;
  int2 my_data = *reinterpret_cast<const int2 *>(in + load_offset);

  // Deinterleave 16-bit halves: view 8 bytes as <4 x i16>, shuffle even/odd.
  using u16x4 = uint16_t __attribute__((ext_vector_type(4)));
  using u16x2 = uint16_t __attribute__((ext_vector_type(2)));
  auto v = __builtin_bit_cast(u16x4, my_data);
  int pack_lo = __builtin_bit_cast(int, __builtin_shufflevector(v, v, 0, 2));
  int pack_hi = __builtin_bit_cast(int, __builtin_shufflevector(v, v, 1, 3));

  // Even sends pack_hi (odd needs it), odd sends pack_lo (even needs it).
  // DPP quad_perm [1,0,3,2]: swap adjacent lanes within each group of 4.
  // Executes as a VALU source modifier — no separate data path.
  constexpr int kQuadPermSwap = 0xB1; // (1<<0)|(0<<2)|(3<<4)|(2<<6)
  int to_send = (lane_in_pair == 0) ? pack_hi : pack_lo;
  int received = __builtin_amdgcn_mov_dpp(to_send, kQuadPermSwap,
                                          /*row_mask=*/0xF, /*bank_mask=*/0xF,
                                          /*bound_ctrl=*/false);

  // Even: {own pack_lo, received pack_lo}  = lo16 of all 4 dwords
  // Odd:  {received pack_hi, own pack_hi}  = hi16 of all 4 dwords
  int2 result;
  if (lane_in_pair == 0) {
    result = make_int2(pack_lo, received);
  } else {
    result = make_int2(received, pack_hi);
  }

  *reinterpret_cast<int2 *>(out + tid * kBytesPerThread) = result;
}

// --- Variant 1: Separate DPP movs + select (compiler may fold into cndmask_dpp) ---
//
// Instead of selecting to_send then doing one DPP swap, we DPP-swap both halves
// independently and let the ternary select own vs. neighbor.  LLVM's GCNDPPCombine
// pass can recognise mov_dpp feeding src0 of a VOP2 and fold it into a single
// v_cndmask_b32_dpp — but only when the DPP result lands in src0 (the VCC=0 path).
// One of the two selects has the DPP result in src1, so at best we expect one fold.
__global__ void load_i8x16_builtin(const int8_t *__restrict__ in,
                                    int8_t *__restrict__ out) {
  int tid = threadIdx.x;
  int pair_id = tid / 2;
  int lane_in_pair = tid & 1;

  int load_offset = pair_id * kBytesPerPair + lane_in_pair * 8;
  int2 my_data = *reinterpret_cast<const int2 *>(in + load_offset);

  using u16x4 = uint16_t __attribute__((ext_vector_type(4)));
  auto v = __builtin_bit_cast(u16x4, my_data);
  int pack_lo = __builtin_bit_cast(int, __builtin_shufflevector(v, v, 0, 2));
  int pack_hi = __builtin_bit_cast(int, __builtin_shufflevector(v, v, 1, 3));

  constexpr int kQuadPermSwap = 0xB1;
  int neighbor_lo = __builtin_amdgcn_mov_dpp(pack_lo, kQuadPermSwap, 0xF, 0xF, true);
  int neighbor_hi = __builtin_amdgcn_mov_dpp(pack_hi, kQuadPermSwap, 0xF, 0xF, true);

  int lo = (lane_in_pair == 0) ? pack_lo    : neighbor_hi;
  int hi = (lane_in_pair != 0) ? pack_hi    : neighbor_lo;

  *reinterpret_cast<int2 *>(out + tid * kBytesPerThread) = make_int2(lo, hi);
}

// --- Variant 2: Inline asm — guaranteed v_cndmask_b32_dpp encoding ---
//
// 4 VALU + 1 SALU (free):
//   v_cmp_ne_u32_e32   vcc, 0, lip          ; VCC: even=0, odd=1
//   v_cndmask_b32_dpp  hi, plo, phi, [vcc]  ; even→DPP(plo)=nbr_lo  odd→phi
//   s_not_b64          vcc, vcc             ; VCC: even=1, odd=0
//   v_cndmask_b32_dpp  lo, phi, plo, [vcc]  ; even→plo              odd→DPP(phi)=nbr_hi
__global__ void load_i8x16_asm(const int8_t *__restrict__ in,
                                int8_t *__restrict__ out) {
  int tid = threadIdx.x;
  int pair_id = tid / 2;
  int lane_in_pair = tid & 1;

  int load_offset = pair_id * kBytesPerPair + lane_in_pair * 8;
  int2 my_data = *reinterpret_cast<const int2 *>(in + load_offset);

  using u16x4 = uint16_t __attribute__((ext_vector_type(4)));
  auto v = __builtin_bit_cast(u16x4, my_data);
  int pack_lo = __builtin_bit_cast(int, __builtin_shufflevector(v, v, 0, 2));
  int pack_hi = __builtin_bit_cast(int, __builtin_shufflevector(v, v, 1, 3));

  int lo, hi;
  asm volatile(
      "v_cmp_ne_u32_e32 vcc, 0, %[lip]\n\t"
      "v_cndmask_b32_dpp %[hi], %[plo], %[phi], vcc"
      " quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
      "s_not_b64 vcc, vcc\n\t"
      "v_cndmask_b32_dpp %[lo], %[phi], %[plo], vcc"
      " quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
      : [lo] "=&v" (lo), [hi] "=&v" (hi)
      : [plo] "v" (pack_lo), [phi] "v" (pack_hi), [lip] "v" (lane_in_pair)
      : "vcc"
  );

  *reinterpret_cast<int2 *>(out + tid * kBytesPerThread) = make_int2(lo, hi);
}

int main() {
  std::vector<int8_t> h_in(kInputElements);
  std::iota(h_in.begin(), h_in.end(), int8_t{0});

  std::vector<int8_t> h_out(kOutputElements, 0);

  int8_t *d_in = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_in, kInputElements));
  HIP_CHECK(hipMalloc(&d_out, kOutputElements));
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), kInputElements,
                       hipMemcpyHostToDevice));

  // Expected swizzle indices into the 16-byte pair block.
  const int even_idx[] = {0, 1, 4, 5, 8, 9, 12, 13};
  const int odd_idx[]  = {2, 3, 6, 7, 10, 11, 14, 15};

  auto validate = [&](const char *name) {
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, kOutputElements,
                         hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    bool ok = true;
    for (int t = 0; t < kSubgroupSize; ++t) {
      int pair_id = t / 2;
      const int *idx = (t & 1) == 0 ? even_idx : odd_idx;
      for (int i = 0; i < kBytesPerThread; ++i) {
        int8_t expected = h_in[pair_id * kBytesPerPair + idx[i]];
        int8_t got = h_out[t * kBytesPerThread + i];
        if (got != expected) {
          printf("  MISMATCH thread %d byte %d: expected %d (K%d), got %d\n",
                 t, i, expected, idx[i], got);
          ok = false;
        }
      }
    }
    printf("[%-8s] %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
  };

  bool all_ok = true;

  HIP_CHECK(hipMemset(d_out, 0, kOutputElements));
  load_i8x16<<<1, kSubgroupSize>>>(d_in, d_out);
  all_ok &= validate("original");

  HIP_CHECK(hipMemset(d_out, 0, kOutputElements));
  load_i8x16_builtin<<<1, kSubgroupSize>>>(d_in, d_out);
  all_ok &= validate("builtin");

  HIP_CHECK(hipMemset(d_out, 0, kOutputElements));
  load_i8x16_asm<<<1, kSubgroupSize>>>(d_in, d_out);
  all_ok &= validate("asm");

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  return all_ok ? 0 : 1;
}

