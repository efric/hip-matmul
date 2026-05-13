# GFX950 SMFMAC 16x16x64 Sparse Trick Notes

This note documents the gfx950/CDNA4 port of the existing CDNA3 F16 sparse
SMFMAC "dense M=8 via sparse M=16" trick in `matmul.hip`.

The implemented kernels are:

- `MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x64f16_shared_uniquedata_gfx950`
- `MmtKernel_256t_NSxKS_amdgcn_smfmac_f32_16x16x64bf16_shared_uniquedata_gfx950`

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

`common.hip` now has BF16 type support:

- `Type::BF16`
- `str(Type::BF16)`
- `type_size(Type::BF16)`
- `CType<Type::BF16> = __bf16`
- BF16 random-buffer generation

`matmul.hip` now has:

- BF16 reference checking: `HANDLE_CASE(BF16, BF16, FP32)`
- a shared F16/BF16 gfx950 base template
- F16 and BF16 wrapper classes
- active tests for `<4,2>`, `<4,4>`, and `<8,2>` for both F16 and BF16

The old CDNA3 `16x16x32_f16_shared_uniquedata_v3` active tests are commented
out in `main()` because they fail correctness on gfx950. That is expected: the
kernel bakes in the CDNA3 sparse register layout.

## Sparse Index Constants

The gfx950 kernels use:

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

## Final A Packing

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

## Final B Shuffle

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

## Why The First Port Failed

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

## Probe Methodology

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

## Verification Evidence

The final unfiltered harness was run with reduced benchmark duration:

```bash
BENCHMARK_MIN_MS=1 ./build_and_test.sh
```

All six active gfx950 kernels printed `Checking correctness... OK`:

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

## Current Technical Debt

The final B register construction is element-explicit. That was intentional for
bring-up: it documents the proved mapping and avoids hiding the key layout in a
clever dword-level shuffle.

A follow-up optimization can repack the B operand with fewer dword moves or
vector loads. The invariant to preserve is exactly this logical slot order:

```text
{base+0,  base+1,  base+32, base+33,
 base+2,  base+3,  base+34, base+35,
 base+16, base+17, base+48, base+49,
 base+18, base+19, base+50, base+51}
```

Do not rewrite this back to a contiguous-8 or CDNA3-style interleave unless a
new gfx950 one-hot probe proves the replacement equivalent.

## Checklist For Future Changes

When modifying this path:

1. Keep the CDNA3 and gfx950 sparse kernels separate. Their B shuffles are not
   interchangeable.
2. Preserve the A low/high split: even rows use `+0`, odd rows use `+32` within
   the 64-wide K bank.
3. Preserve the gfx950 B slot order above.
4. Run both F16 and BF16 filtered correctness:

   ```bash
   FILTER=16x16x64f16 BENCHMARK_MIN_MS=1 ./build_and_test.sh
   FILTER=16x16x64bf16 BENCHMARK_MIN_MS=1 ./build_and_test.sh
   ```

5. Run the unfiltered active harness:

   ```bash
   BENCHMARK_MIN_MS=1 ./build_and_test.sh
   ```

6. Check assembly for the intended instructions:

   ```bash
   grep -c "v_smfmac_f32_16x16x64_f16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
   grep -c "v_smfmac_f32_16x16x64_bf16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
   grep -c "v_smfmac_f32_16x16x32_f16" build/matmul-hip-amdgcn-amd-amdhsa-gfx950.s
   ```

The expected last count is `0` for the active gfx950 harness.
