# GFX950 SMFMAC Sparse Trick Notes

This note documents the gfx950/CDNA4 ports of the sparse SMFMAC
"dense M=8 via sparse M=16" trick in `matmul.hip`. It covers both the
16x16x64 F16/BF16 path and the 16x16x128 FP8_FP8 path.

The implemented kernels are:

- `MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x64f16_shared_uniquedata_gfx950`
- `MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x64bf16_shared_uniquedata_gfx950`
- `MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x128fp8_fp8_shared_uniquedata_gfx950`

They are the CDNA4 counterparts of:

- `MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x32f16_shared_uniquedata_v3`

## Problem Statement

The CDNA3 kernel uses `V_SMFMAC_F32_16X16X32_F16` to compute a dense M=8
matmul by treating two physical sparse rows as one dense row:

- even lanes carry the low half of the dense row and use sparse positions 0,1
- odd lanes carry the high half of the dense row and use sparse positions 2,3
- the destination rows are reduced after SMFMAC: `D0 += D1`, `D2 += D3`

On CDNA3, one dense 16-element K slice takes two `16x16x32_f16` SMFMACs.
On CDNA4/gfx950, `V_SMFMAC_F32_16X16X64_F16` and
`V_SMFMAC_F32_16X16X64_BF16` consume four A dwords and eight B dwords, so the
same logical 64-wide K bank can be covered with one SMFMAC per innermost loop.

The nontrivial part is the B register shuffle. The CDNA4 B mapping is not a
simple "CDNA3 but twice as wide" extension for this sparse trick.

## What Was Implemented

`common.hip` now has BF16 and FP8 type support:

- `Type::BF16` and `Type::FP8`
- `str(Type::BF16)` and `str(Type::FP8)`
- `type_size(Type::BF16) = 2` and `type_size(Type::FP8) = 1`
- `CType<Type::BF16> = __bf16`
- `CType<Type::FP8> = __hip_fp8_e4m3`
- BF16 and FP8 random-buffer generation

`matmul.hip` now has:

- BF16 reference checking: `HANDLE_CASE(BF16, BF16, FP32)`
- FP8 reference checking: `HANDLE_CASE(FP8, FP8, FP32)`
- a shared F16/BF16 gfx950 base template
- F16 and BF16 wrapper classes
- a gfx950 `16x16x128_fp8_fp8` kernel
- active tests for `<4,2>`, `<4,4>`, and `<8,2>` for F16, BF16, and FP8

The old CDNA3 `16x16x32_f16_shared_uniquedata_v3` active tests are commented
out in `main()` because they fail correctness on gfx950. That is expected: the
kernel bakes in the CDNA3 sparse register layout.

## F16/BF16 Sparse Index Constants

The F16/BF16 gfx950 kernels use:

```c++
static constexpr int SPARSITY_EVEN = 0x00004444;
static constexpr int SPARSITY_ODD = 0x0000EEEE;
```

The important semantic is the repeated 4-bit sparse group selection:

- `0x4` means select positions 0 and 1 within each group of 4
- `0xE` means select positions 2 and 3 within each group of 4

The repeated byte/halfword spelling keeps the intended pattern explicit across
the packed halves used by the CDNA4 instruction. The runtime probes and final
correctness checks were run with these exact constants.

## F16/BF16 Final A Packing

For one output lane, the kernel computes:

```c++
int a_m_row = (lane_id % 16) / 2;
int a_k_range = lane_id / 16;
int odd_lane = lane_id % 2;

int a_elem_base = a_m_row * (KS * 64) + k_bank * 64 +
                  a_k_range * 8 + odd_lane * 32;
for (int i = 0; i < 8; ++i) {
  a_reg.h[i] = A_shared_elems[a_elem_base + i];
}
```

Interpretation for a single 64-wide dense K bank:

- even physical sparse row loads `K = 8*a_k_range + slot`
- odd physical sparse row loads `K = 32 + 8*a_k_range + slot`

After the SMFMAC, the kernel reduces the physical rows back to dense M=8:

```c++
regD[0] = regD[0] + regD[1];
regD[1] = regD[2] + regD[3];
```

## F16/BF16 Final B Shuffle

The final B register fill is:

```c++
int b_elem_base = n_block * (KS * 1024) + n_col * (KS * 64) +
                  k_bank * 64 + k_slice * 4;

b_reg.h[0]  = B_shared_elems[b_elem_base + 0];
b_reg.h[1]  = B_shared_elems[b_elem_base + 1];
b_reg.h[2]  = B_shared_elems[b_elem_base + 32];
b_reg.h[3]  = B_shared_elems[b_elem_base + 33];
b_reg.h[4]  = B_shared_elems[b_elem_base + 2];
b_reg.h[5]  = B_shared_elems[b_elem_base + 3];
b_reg.h[6]  = B_shared_elems[b_elem_base + 34];
b_reg.h[7]  = B_shared_elems[b_elem_base + 35];
b_reg.h[8]  = B_shared_elems[b_elem_base + 16];
b_reg.h[9]  = B_shared_elems[b_elem_base + 17];
b_reg.h[10] = B_shared_elems[b_elem_base + 48];
b_reg.h[11] = B_shared_elems[b_elem_base + 49];
b_reg.h[12] = B_shared_elems[b_elem_base + 18];
b_reg.h[13] = B_shared_elems[b_elem_base + 19];
b_reg.h[14] = B_shared_elems[b_elem_base + 50];
b_reg.h[15] = B_shared_elems[b_elem_base + 51];
```

Equivalently, for `base = 4*k_slice`, the B operand slots are:

```text
{base+0,  base+1,  base+32, base+33,
 base+2,  base+3,  base+34, base+35,
 base+16, base+17, base+48, base+49,
 base+18, base+19, base+50, base+51}
```

This is the key gfx950-specific learning.

## Why The First F16/BF16 Port Failed

The first implementation assumed the CDNA4 B operand could be formed as the
CDNA3 layout with an added high 32-element half:

```text
{K0,K1, K32,K33, K2,K3, K34,K35,
 K4,K5, K36,K37, K6,K7, K38,K39}
```

That was incomplete. It handled the low/high split but missed the way gfx950
pairs A source slots and B lane groups for `16x16x64_f16/bf16`.

Observed failures during bring-up:

- original CDNA3 v3 active test on gfx950 failed immediately:
  `actual(7) != expected(26)` for the first checked tile
- the first gfx950 port also failed with the same first-tile mismatch
- after changing only the low/high stride, the first tile changed to
  `actual(57) != expected(26)`, which proved that part of the mapping changed
  but the B slot/lane-group shuffle was still wrong

The final shuffle above resolves those failures.

## What This Means For The ISA Table And Matrix Calculator

This does not imply that the CDNA4 ISA table or `amd_matrix_instruction_calculator`
are simply wrong.

The table and calculator describe the architectural/logical operand layout for
the instruction. This kernel is using the sparse instruction in a non-canonical
way:

1. The A operand is not a normal sparse A matrix from the original problem.
2. It is a packed representation of two dense M=8 row halves.
3. Sparse indices are chosen to select positions 0,1 or 2,3.
4. The destination rows are reduced to recover one dense row.
5. Therefore B must be packed in the inverse order that makes those selected
   products match dense K order.

The calculator's `--output-calculation` output is useful for understanding the
logical source terms. It did not directly answer the bring-up question:

```text
Given these concrete packed A slots, this concrete sparse index value, and this
post-SMFMAC row reduction, which physical B source slots and lane groups must be
fed to recover dense K order?
```

That question was answered with direct one-hot probes on gfx950.

## F16/BF16 Probe Methodology

Temporary HIP probes were compiled in `/tmp` on the gfx950 machine. They were
not committed to this directory.

The probes used `__builtin_amdgcn_smfmac_f32_16x16x64_f16` directly with
one-hot A and B operands, then copied the first lane's accumulator back to the
host. This avoids guessing from aggregate matmul failures.

The decisive probe varied:

- A source slot `0..7`
- A source lane group `threadIdx.x / 16`
- B source slot `0..15`
- B source lane group `threadIdx.x / 16`

The observed mapping for the first two destination components was:

```text
D0 A0/G0 <- B0/G0    D1 A0/G0 <- B2/G0
D0 A0/G1 <- B0/G2    D1 A0/G1 <- B2/G2
D0 A0/G2 <- B8/G0    D1 A0/G2 <- B10/G0
D0 A0/G3 <- B8/G2    D1 A0/G3 <- B10/G2

D0 A1/G0 <- B1/G0    D1 A1/G0 <- B3/G0
D0 A1/G1 <- B1/G2    D1 A1/G1 <- B3/G2
D0 A1/G2 <- B9/G0    D1 A1/G2 <- B11/G0
D0 A1/G3 <- B9/G2    D1 A1/G3 <- B11/G2

D0 A2/G0 <- B4/G0    D1 A2/G0 <- B6/G0
D0 A2/G1 <- B4/G2    D1 A2/G1 <- B6/G2
D0 A2/G2 <- B12/G0   D1 A2/G2 <- B14/G0
D0 A2/G3 <- B12/G2   D1 A2/G3 <- B14/G2

D0 A3/G0 <- B5/G0    D1 A3/G0 <- B7/G0
D0 A3/G1 <- B5/G2    D1 A3/G1 <- B7/G2
D0 A3/G2 <- B13/G0   D1 A3/G2 <- B15/G0
D0 A3/G3 <- B13/G2   D1 A3/G3 <- B15/G2

D0 A4/G0 <- B0/G1    D1 A4/G0 <- B2/G1
D0 A4/G1 <- B0/G3    D1 A4/G1 <- B2/G3
D0 A4/G2 <- B8/G1    D1 A4/G2 <- B10/G1
D0 A4/G3 <- B8/G3    D1 A4/G3 <- B10/G3

D0 A5/G0 <- B1/G1    D1 A5/G0 <- B3/G1
D0 A5/G1 <- B1/G3    D1 A5/G1 <- B3/G3
D0 A5/G2 <- B9/G1    D1 A5/G2 <- B11/G1
D0 A5/G3 <- B9/G3    D1 A5/G3 <- B11/G3

D0 A6/G0 <- B4/G1    D1 A6/G0 <- B6/G1
D0 A6/G1 <- B4/G3    D1 A6/G1 <- B6/G3
D0 A6/G2 <- B12/G1   D1 A6/G2 <- B14/G1
D0 A6/G3 <- B12/G3   D1 A6/G3 <- B14/G3

D0 A7/G0 <- B5/G1    D1 A7/G0 <- B7/G1
D0 A7/G1 <- B5/G3    D1 A7/G1 <- B7/G3
D0 A7/G2 <- B13/G1   D1 A7/G2 <- B15/G1
D0 A7/G3 <- B13/G3   D1 A7/G3 <- B15/G3
```

Compressing that table gives the final B shuffle formula used by the kernel.

## F16/BF16 Verification Evidence

The final unfiltered harness was run with reduced benchmark duration:

```bash
BENCHMARK_MIN_MS=1 ./build_and_test.sh
```

Before the FP8 path was added, all six active F16/BF16 gfx950 kernels printed `Checking correctness... OK`:

```text
MmtKernel_...16x16x64f16...<4, 2>   OK
MmtKernel_...16x16x64f16...<4, 4>   OK
MmtKernel_...16x16x64f16...<8, 2>   OK
MmtKernel_...16x16x64bf16...<4, 2>  OK
MmtKernel_...16x16x64bf16...<4, 4>  OK
MmtKernel_...16x16x64bf16...<8, 2>  OK
```

The generated gfx950 assembly was checked after `-save-temps=obj`:

```bash
grep -c "v_smfmac_f32_16x16x64_f16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
grep -c "v_smfmac_f32_16x16x64_bf16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
grep -c "v_smfmac_f32_16x16x32_f16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
```

Observed counts:

```text
v_smfmac_f32_16x16x64_f16   10
v_smfmac_f32_16x16x64_bf16  10
v_smfmac_f32_16x16x32_f16   0
```

This proves the active gfx950 path lowers to the intended CDNA4 sparse
instructions and no active CDNA3 F16 sparse instruction remains in the generated
code.


## FP8 16x16x128 Path

The gfx950 FP8 path uses:

```c++
__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8
```

The implemented kernel is:

```text
MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x128fp8_fp8_shared_uniquedata_gfx950
```

It has the same high-level tile shape as the F16/BF16 sparse trick:

- `M_tile = 8`
- `N_tile = NS * 16`
- `K_tile = KS * 128`
- 256 threads
- output type `FP32`

The instruction uses four A VGPRs, eight B VGPRs, and four D VGPRs. Since FP8
is one byte per element, the operands are naturally modeled as raw byte vectors:

```c++
using u8x16 = __attribute__((__vector_size__(16))) unsigned char;
using u8x32 = __attribute__((__vector_size__(32))) unsigned char;
```

A uses `u8x16`, B uses `u8x32`, and D uses the same `f32x4` accumulator shape as
the F16/BF16 path.

### FP8 Encoding Learning

`common.hip` now has `Type::FP8`, with:

```c++
CType<Type::FP8> = __hip_fp8_e4m3;
```

This is intentionally the non-FNUZ OCP e4m3 type.

A temporary host probe showed:

```text
__hip_fp8_e4m3_fnuz(1.0f).__x = 0x40
__hip_fp8_e4m3(1.0f).__x      = 0x38
```

A one-hot SMFMAC probe then showed that raw byte `0x40` produced a product of
`4`, while raw byte `0x38` produced a product of `1`. That means the gfx950
`fp8_fp8` SMFMAC builtin is interpreting operand bytes as the non-FNUZ/OCP e4m3
encoding for this path. Using `__hip_fp8_e4m3_fnuz` would make the reference
buffer encode `1.0` as hardware `2.0`, so it would be the wrong test type for
this instruction.

This is a separate issue from the sparse-layout question. The matrix calculator
labels this type as `amd_fp8`, but the executable evidence from the builtin on
gfx950 is that `0x38` is the byte encoding that behaves as `1.0`.

### FP8 Sparse Constants

The FP8 instruction covers twice as many K positions as the F16/BF16 16x16x64
instruction, so the sparse selector pattern is repeated across all eight
nibbles:

```c++
static constexpr int SPARSITY_EVEN = 0x44444444;
static constexpr int SPARSITY_ODD = 0xEEEEEEEE;
```

The semantic is unchanged:

- `0x4` selects positions 0 and 1 in each 4-wide sparse group
- `0xE` selects positions 2 and 3 in each 4-wide sparse group

### FP8 A Packing

For one 128-wide dense K bank, the A packing is:

```c++
int a_m_row = (lane_id % 16) / 2;
int a_k_range = lane_id / 16;
int odd_lane = lane_id % 2;

int a_elem_base = a_m_row * (KS * 128) + k_bank * 128 +
                  a_k_range * 16 + odd_lane * 64;
for (int i = 0; i < 16; ++i) {
  a_reg.u8[i] = A_shared_elems[a_elem_base + i];
}
```

Interpretation:

- even physical sparse row loads `K = 16*a_k_range + slot`
- odd physical sparse row loads `K = 64 + 16*a_k_range + slot`

As with F16/BF16, post-processing reduces the two physical sparse rows back to
one dense row:

```c++
regD[0] = regD[0] + regD[1];
regD[1] = regD[2] + regD[3];
```

### FP8 One-Hot Mapping

The decisive one-hot probe varied A byte slot `0..15`, A source lane group
`0..3`, B byte slot `0..31`, and B source lane group `0..3`. It restricted A
and B to specific source lanes so that the output row/column identified the
actual cross-lane SMFMAC source relationship.

For the even sparse selector, the observed mapping compresses to:

```text
even_b = 16 * floor(ag / 2) + 4 * floor((a % 8) / 2) + (a % 2)
bg     = 2 * (ag % 2) + floor(a / 8)
```

where `a` is the A byte slot, `ag` is the A source lane group, `even_b` is the B
byte slot, and `bg` is the B source lane group.

For the odd sparse selector, the B byte slot is shifted by two:

```text
odd_b = even_b + 2
bg    = 2 * (ag % 2) + floor(a / 8)
```

Representative raw probe rows with byte `0x38` showed exact products of `1`:

```text
EVEN:
D0 A00/G0 <- B00/G0    D0 A00/G1 <- B00/G2
D0 A00/G2 <- B16/G0    D0 A00/G3 <- B16/G2
D0 A01/G0 <- B01/G0    D0 A01/G1 <- B01/G2
D0 A01/G2 <- B17/G0    D0 A01/G3 <- B17/G2
D0 A08/G0 <- B00/G1    D0 A08/G1 <- B00/G3
D0 A08/G2 <- B16/G1    D0 A08/G3 <- B16/G3

ODD:
D0 A00/G0 <- B02/G0    D0 A00/G1 <- B02/G2
D0 A00/G2 <- B18/G0    D0 A00/G3 <- B18/G2
D0 A01/G0 <- B03/G0    D0 A01/G1 <- B03/G2
D0 A01/G2 <- B19/G0    D0 A01/G3 <- B19/G2
D0 A08/G0 <- B02/G1    D0 A08/G1 <- B02/G3
D0 A08/G2 <- B18/G1    D0 A08/G3 <- B18/G3
```

The D component did not change the byte-slot formula for this probe; rows D0
through D3 used the same A/B slot relationship.

### FP8 Final B Shuffle

Inverting the one-hot table gives the B operand order for the actual dense
matmul packing. For source B lane group `g`:

```text
base = 8*g

{base+0,  base+1,  base+64,  base+65,
 base+2,  base+3,  base+66,  base+67,
 base+4,  base+5,  base+68,  base+69,
 base+6,  base+7,  base+70,  base+71,
 base+32, base+33, base+96,  base+97,
 base+34, base+35, base+98,  base+99,
 base+36, base+37, base+100, base+101,
 base+38, base+39, base+102, base+103}
```

The implemented B register fill uses that exact order:

```c++
int b_elem_base = n_block * (KS * 2048) + n_col * (KS * 128) +
                  k_bank * 128 + k_slice * 8;

b_reg.u8[0]  = B_shared_elems[b_elem_base + 0];
b_reg.u8[1]  = B_shared_elems[b_elem_base + 1];
b_reg.u8[2]  = B_shared_elems[b_elem_base + 64];
b_reg.u8[3]  = B_shared_elems[b_elem_base + 65];
...
b_reg.u8[30] = B_shared_elems[b_elem_base + 102];
b_reg.u8[31] = B_shared_elems[b_elem_base + 103];
```

This is the byte-width analogue of the gfx950 F16/BF16 shuffle. The larger
constant offsets come from the 128-wide dense K bank: the odd sparse row starts
at `+64`, and the second half of the B lane-group inversion starts at `+32`.

### FP8 Verification Evidence

Filtered FP8 verification was run with:

```bash
FILTER=16x16x128fp8_fp8 BENCHMARK_MIN_MS=1 ./build_and_test.sh
```

All three active FP8 shapes printed `Checking correctness... OK`:

```text
MmtKernel_...16x16x128fp8_fp8...<4, 2>  OK  tile MxNxK=8x64x256
MmtKernel_...16x16x128fp8_fp8...<4, 4>  OK  tile MxNxK=8x64x512
MmtKernel_...16x16x128fp8_fp8...<8, 2>  OK  tile MxNxK=8x128x256
```

The unfiltered active harness was then rerun:

```bash
BENCHMARK_MIN_MS=1 ./build_and_test.sh
```

All nine active gfx950 kernels printed `Checking correctness... OK`: the three
F16 kernels, the three BF16 kernels, and the three FP8 kernels.

Generated gfx950 assembly was checked after `-save-temps=obj`:

```bash
grep -c "v_smfmac_f32_16x16x128_fp8_fp8" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
```

Observed count:

```text
v_smfmac_f32_16x16x128_fp8_fp8  10
```

The same assembly file still had ten F16 and ten BF16 CDNA4 sparse instructions:

```text
v_smfmac_f32_16x16x64_f16   10
v_smfmac_f32_16x16x64_bf16  10
```


## Current Technical Debt

The final B register construction is element-explicit for both the F16/BF16 and FP8 paths. That was intentional for
bring-up: it documents the proved mapping and avoids hiding the key layout in a
clever dword-level shuffle.

A follow-up optimization can repack the B operand with fewer dword moves or
vector loads. For F16/BF16, the invariant to preserve is exactly this logical slot order:

```text
{base+0,  base+1,  base+32, base+33,
 base+2,  base+3,  base+34, base+35,
 base+16, base+17, base+48, base+49,
 base+18, base+19, base+50, base+51}
```

For FP8, the invariant is the 32-byte order documented in the FP8 section.

Do not rewrite either path back to a contiguous or CDNA3-style interleave unless
a new gfx950 one-hot probe proves the replacement equivalent.

## Checklist For Future Changes

When modifying this path:

1. Keep the CDNA3 and gfx950 sparse kernels separate. Their B shuffles are not
   interchangeable.
2. Preserve the F16/BF16 A low/high split: even rows use `+0`, odd rows use `+32` within
   the 64-wide K bank.
3. Preserve the FP8 A low/high split: even rows use `+0`, odd rows use `+64` within
   the 128-wide K bank.
4. Preserve the gfx950 B slot orders above.
5. Run F16, BF16, and FP8 filtered correctness:

   ```bash
   FILTER=16x16x64f16 BENCHMARK_MIN_MS=1 ./build_and_test.sh
   FILTER=16x16x64bf16 BENCHMARK_MIN_MS=1 ./build_and_test.sh
   FILTER=16x16x128fp8_fp8 BENCHMARK_MIN_MS=1 ./build_and_test.sh
   ```

6. Run the unfiltered active harness:

   ```bash
   BENCHMARK_MIN_MS=1 ./build_and_test.sh
   ```

7. Check assembly for the intended instructions:

   ```bash
   grep -c "v_smfmac_f32_16x16x64_f16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
   grep -c "v_smfmac_f32_16x16x64_bf16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
   grep -c "v_smfmac_f32_16x16x128_fp8_fp8" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
   grep -c "v_smfmac_f32_16x16x32_f16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
   ```

The expected last count is `0` for the active gfx950 harness.
