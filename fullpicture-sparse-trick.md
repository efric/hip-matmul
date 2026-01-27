To help understand the implementation details of the sparse trick, the following is a detailed walkthrough of what happens for one tile, ignoring warp specialization, in the HF skinny GEMM kernel. The focus is on the A tile (how we pack it to feed the SMFMAC instruction) and the post-intrinsic handling.

**Important note on OPS/K depth:** for OP_M = 8, the kernel actually compiles with **OPS = 4 or 8** (see `skinny_gemm_caller.cu`). That means **WARPTILE_K = 64 * OPS = 256 or 512**. The A producer loads data in **8×128 chunks** (two 8×128 halves per loop iteration: reg0 + reg1), so the diagrams below show one such 8×128 *half-tile* for clarity. If OPS=4, the full A tile is 8×256 (two halves). If OPS=8, it is 8×512 (four halves).

First, one 8 × 128 half-tile (1024 bytes) is loaded per subgroup from A. The thread layout looks like this:

```
K        0   16   32   48   64   80   96  112
row 0:  t0   t1   t2   t3   t4   t5   t6   t7
row 1:  t8   t9   t10  t11  t12  t13  t14  t15
row 2:  t16  t17  t18  t19  t20  t21  t22  t23
row 3:  t24  t25  t26  t27  t28  t29  t30  t31
row 4:  t32  t33  t34  t35  t36  t37  t38  t39
row 5:  t40  t41  t42  t43  t44  t45  t46  t47
row 6:  t48  t49  t50  t51  t52  t53  t54  t55
row 7:  t56  t57  t58  t59  t60  t61  t62  t63
```

For each thread, the 16 FP8 elements are contiguous along K:

```
K0 K1 K2 K3 K4 K5 K6 K7 K8 K9 K10 K11 K12 K13 K14 K15
```

These are then swizzled to:

```
K0 K1 K4 K5 | K8 K9 K12 K13 | K2 K3 K6 K7 | K10 K11 K14 K15
```

Each 4 element group is 32 bits and is stored in one of two 4×32-bit LDS banks. There are two banks because each SMFMAC consumes only **512 bytes (8×64)**. So an **8×128 half‑tile** is split into two 8×64 banks: one bank stores K spanning 0–63, and the other stores 64–127 (following the layout above). For OPS=4, you have two such half‑tiles (K 0–127 and K 128–255). For OPS=8, you have four.

The shared-memory arrangement looks like:

(shared memory, bank 0 - K 0..63)
```
line\col:   0         1         2         3        ...       31
line 0:   t0.reg0  t8.reg0  t16.reg0  t24.reg0     ...     t56.reg0
line 1:   t0.reg1  t8.reg1  t16.reg1  t24.reg1     ...     t56.reg1
line 2:   t0.reg2  t8.reg2  t16.reg2  t24.reg2     ...     t56.reg2
line 3:   t0.reg3  t8.reg3  t16.reg3  t24.reg3     ...     t56.reg3
```

Where, for each lane:

```
reg0 = K0 K1 K4 K5
reg1 = K8 K9 K12 K13
reg2 = K2 K3 K6 K7
reg3 = K10 K11 K14 K15
```

The second bank is the same structure, but for threads spanning the second half of the K tile.

---

## B tile load (producer) and LDS layout

For OP_M = 8, the dense B tile consumed by one SMFMAC is **16 x 64** (K x N), which is **1024 bytes**. With OPS = 4, the producer loads **4 of these tiles**, stacked in LDS, for a total of **4096 bytes**.

The B producer uses `produce_n_full_tiles` (`hf-rocm-kernels/csrc/op_src/skinny_gemm/producers/n_full_tiles.cu`). For OP_M = 8:

```
OP_K = 64
OP_N = 16
E_PER_THREAD = 16 (bytes)
THREADS_PER_LD = 4
THREADS_PER_AD = 16
```

Per lane:

```
curr_ld = (lane_id % 4) * 16   // K chunk: 0,16,32,48
curr_ad = lane_id / 4          // N column: 0..15
```

So the per-lane B mapping looks like:

```
K chunk ->   0     16    32    48
N=0:        L0    L1    L2    L3
N=1:        L4    L5    L6    L7
N=2:        L8    L9    L10   L11
N=3:        L12   L13   L14   L15
...
N=15:       L60   L61   L62   L63
```

Each lane loads 16 contiguous K values for a single N column.

### B LDS layout (one 16x64 tile)

The producer swizzles B into LDS as **two panels**, each panel being 4 lines x 32 columns:

- **Panel 0** = K 0..31
- **Panel 1** = K 32..63

Within a panel:
- cols 0..15  -> K chunk 0 (or 2)
- cols 16..31 -> K chunk 1 (or 3)

#### Panel 0 (K 0..31)

```
line\col:   0     1     2     3    ...   15    16    17    18   ...   31
line 0:    L0    L4    L8    L12   ...   L60   L1    L5    L9   ...   L61
line 1:    L0    L4    L8    L12   ...   L60   L1    L5    L9   ...   L61
line 2:    L0    L4    L8    L12   ...   L60   L1    L5    L9   ...   L61
line 3:    L0    L4    L8    L12   ...   L60   L1    L5    L9   ...   L61
```

#### Panel 1 (K 32..63)

```
line\col:   0     1     2     3    ...   15    16    17    18   ...   31
line 0:    L2    L6    L10   L14   ...   L62   L3    L7    L11  ...   L63
line 1:    L2    L6    L10   L14   ...   L62   L3    L7    L11  ...   L63
line 2:    L2    L6    L10   L14   ...   L62   L3    L7    L11  ...   L63
line 3:    L2    L6    L10   L14   ...   L62   L3    L7    L11  ...   L63
```

(Each lane writes four lines = 16 bytes total. The repeated lane IDs indicate the 4 lines for that lane.)

### Consumer-side B load (lane mapping)

The consumer relocates B as:

```
B_buffer += (lane_id % 32) * 4;
B_buffer += (lane_id / 32) * 4 * 4 * NB_BANKS;   // = 512 bytes per panel
```

So each lane picks:

```
panel = lane_id / 32
col   = lane_id % 32
```

This yields the following lane-to-(N,K) mapping per SMFMAC:

```
L0  -> N0  K 0..15
L1  -> N1  K 0..15
...
L15 -> N15 K 0..15

L16 -> N0  K 16..31
L17 -> N1  K 16..31
...
L31 -> N15 K 16..31

L32 -> N0  K 32..47
L33 -> N1  K 32..47
...
L47 -> N15 K 32..47

L48 -> N0  K 48..63
L49 -> N1  K 48..63
...
L63 -> N15 K 48..63
```

This exactly matches the dense B layout expected by `v_smfmac_f32_16x16x64_fp8_fp8`.

With OPS = 4, there are **4 such 16x64 tiles**, stacked sequentially in LDS (each 1024 bytes), and the consumer loops `op = 0..3`, loading B and issuing **4 SMFMACs** per K block.

---

The consumer then loads from shared memory into registers for the intrinsic in the following lane pairing pattern:

```
col:      0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31
row 0:   L0    L2    L4    L6    L8   L10   L12   L14   L16   L18   L20   L22   L24   L26   L28   L30   L32   L34   L36   L38   L40   L42   L44   L46   L48   L50   L52   L54   L56   L58   L60   L62
row 1:   L0    L2    L4    L6    L8   L10   L12   L14   L16   L18   L20   L22   L24   L26   L28   L30   L32   L34   L36   L38   L40   L42   L44   L46   L48   L50   L52   L54   L56   L58   L60   L62
row 2:   L1    L3    L5    L7    L9   L11   L13   L15   L17   L19   L21   L23   L25   L27   L29   L31   L33   L35   L37   L39   L41   L43   L45   L47   L49   L51   L53   L55   L57   L59   L61   L63
row 3:   L1    L3    L5    L7    L9   L11   L13   L15   L17   L19   L21   L23   L25   L27   L29   L31   L33   L35   L37   L39   L41   L43   L45   L47   L49   L51   L53   L55   L57   L59   L61   L63
```

So each even lane holds:

```
K0 K1 K4 K5 K8 K9 K12 K13
```

and each odd lane holds:

```
K2 K3 K6 K7 K10 K11 K14 K15
```

This matches the sparsity indices we use:

odd lanes: 0x0000EEEE (0111011101110111)
even lanes: 0x00004444 (0100010001000100)

As a reminder, every 2 bits encodes the non-zero positions of the “sparse” matrix that the instruction interprets.

In other words, the data loaded from LDS looks exactly like:

```
lane 0: K0 K1 _  _  K4 K5 _  _  K8  K9  _   _  K12 K13 _   _
lane 1: _  _  K2 K3 _  _  K6 K7 _   _  K10 K11 _   _  K14 K15
```

Together (as an even/odd pair), these precisely reconstruct the dense rows. The intrinsic `__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8` is then called with the “fake” sparse A tile and the dense B tile (which is loaded normally, without tricks), once per **512‑byte segment** along K. So for OPS=4 (K=256) there are **4 SMFMAC calls**, and for OPS=8 (K=512) there are **8 SMFMAC calls**.

After the intrinsic, the destination register layout looks like the following (extracted from [AMD matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator).) 

```
Architecture: CDNA3
Instruction: V_SMFMAC_F32_16X16X64_FP8_FP8
|   lane | v0        | v1        | v2        | v3        |
|--------|-----------|-----------|-----------|-----------|
|      0 | D[0][0]   | D[1][0]   | D[2][0]   | D[3][0]   |
|      1 | D[0][1]   | D[1][1]   | D[2][1]   | D[3][1]   |
|      2 | D[0][2]   | D[1][2]   | D[2][2]   | D[3][2]   |
|      3 | D[0][3]   | D[1][3]   | D[2][3]   | D[3][3]   |
|      4 | D[0][4]   | D[1][4]   | D[2][4]   | D[3][4]   |
|      5 | D[0][5]   | D[1][5]   | D[2][5]   | D[3][5]   |
|      6 | D[0][6]   | D[1][6]   | D[2][6]   | D[3][6]   |
|      7 | D[0][7]   | D[1][7]   | D[2][7]   | D[3][7]   |
|      8 | D[0][8]   | D[1][8]   | D[2][8]   | D[3][8]   |
|      9 | D[0][9]   | D[1][9]   | D[2][9]   | D[3][9]   |
|     10 | D[0][10]  | D[1][10]  | D[2][10]  | D[3][10]  |
|     11 | D[0][11]  | D[1][11]  | D[2][11]  | D[3][11]  |
|     12 | D[0][12]  | D[1][12]  | D[2][12]  | D[3][12]  |
|     13 | D[0][13]  | D[1][13]  | D[2][13]  | D[3][13]  |
|     14 | D[0][14]  | D[1][14]  | D[2][14]  | D[3][14]  |
|     15 | D[0][15]  | D[1][15]  | D[2][15]  | D[3][15]  |
|     16 | D[4][0]   | D[5][0]   | D[6][0]   | D[7][0]   |
|     17 | D[4][1]   | D[5][1]   | D[6][1]   | D[7][1]   |
|     18 | D[4][2]   | D[5][2]   | D[6][2]   | D[7][2]   |
|     19 | D[4][3]   | D[5][3]   | D[6][3]   | D[7][3]   |
|     20 | D[4][4]   | D[5][4]   | D[6][4]   | D[7][4]   |
|     21 | D[4][5]   | D[5][5]   | D[6][5]   | D[7][5]   |
|     22 | D[4][6]   | D[5][6]   | D[6][6]   | D[7][6]   |
|     23 | D[4][7]   | D[5][7]   | D[6][7]   | D[7][7]   |
|     24 | D[4][8]   | D[5][8]   | D[6][8]   | D[7][8]   |
|     25 | D[4][9]   | D[5][9]   | D[6][9]   | D[7][9]   |
|     26 | D[4][10]  | D[5][10]  | D[6][10]  | D[7][10]  |
|     27 | D[4][11]  | D[5][11]  | D[6][11]  | D[7][11]  |
|     28 | D[4][12]  | D[5][12]  | D[6][12]  | D[7][12]  |
|     29 | D[4][13]  | D[5][13]  | D[6][13]  | D[7][13]  |
|     30 | D[4][14]  | D[5][14]  | D[6][14]  | D[7][14]  |
|     31 | D[4][15]  | D[5][15]  | D[6][15]  | D[7][15]  |
|     32 | D[8][0]   | D[9][0]   | D[10][0]  | D[11][0]  |
|     33 | D[8][1]   | D[9][1]   | D[10][1]  | D[11][1]  |
|     34 | D[8][2]   | D[9][2]   | D[10][2]  | D[11][2]  |
|     35 | D[8][3]   | D[9][3]   | D[10][3]  | D[11][3]  |
|     36 | D[8][4]   | D[9][4]   | D[10][4]  | D[11][4]  |
|     37 | D[8][5]   | D[9][5]   | D[10][5]  | D[11][5]  |
|     38 | D[8][6]   | D[9][6]   | D[10][6]  | D[11][6]  |
|     39 | D[8][7]   | D[9][7]   | D[10][7]  | D[11][7]  |
|     40 | D[8][8]   | D[9][8]   | D[10][8]  | D[11][8]  |
|     41 | D[8][9]   | D[9][9]   | D[10][9]  | D[11][9]  |
|     42 | D[8][10]  | D[9][10]  | D[10][10] | D[11][10] |
|     43 | D[8][11]  | D[9][11]  | D[10][11] | D[11][11] |
|     44 | D[8][12]  | D[9][12]  | D[10][12] | D[11][12] |
|     45 | D[8][13]  | D[9][13]  | D[10][13] | D[11][13] |
|     46 | D[8][14]  | D[9][14]  | D[10][14] | D[11][14] |
|     47 | D[8][15]  | D[9][15]  | D[10][15] | D[11][15] |
|     48 | D[12][0]  | D[13][0]  | D[14][0]  | D[15][0]  |
|     49 | D[12][1]  | D[13][1]  | D[14][1]  | D[15][1]  |
|     50 | D[12][2]  | D[13][2]  | D[14][2]  | D[15][2]  |
|     51 | D[12][3]  | D[13][3]  | D[14][3]  | D[15][3]  |
|     52 | D[12][4]  | D[13][4]  | D[14][4]  | D[15][4]  |
|     53 | D[12][5]  | D[13][5]  | D[14][5]  | D[15][5]  |
|     54 | D[12][6]  | D[13][6]  | D[14][6]  | D[15][6]  |
|     55 | D[12][7]  | D[13][7]  | D[14][7]  | D[15][7]  |
|     56 | D[12][8]  | D[13][8]  | D[14][8]  | D[15][8]  |
|     57 | D[12][9]  | D[13][9]  | D[14][9]  | D[15][9]  |
|     58 | D[12][10] | D[13][10] | D[14][10] | D[15][10] |
|     59 | D[12][11] | D[13][11] | D[14][11] | D[15][11] |
|     60 | D[12][12] | D[13][12] | D[14][12] | D[15][12] |
|     61 | D[12][13] | D[13][13] | D[14][13] | D[15][13] |
|     62 | D[12][14] | D[13][14] | D[14][14] | D[15][14] |
|     63 | D[12][15] | D[13][15] | D[14][15] | D[15][15] |
```

Interpreting that as a 16×16 output tile (each element annotated by lane ID), we get:

```
row\col      0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
0          L0   L1   L2   L3   L4   L5   L6   L7   L8   L9  L10  L11  L12  L13  L14  L15
1          L0   L1   L2   L3   L4   L5   L6   L7   L8   L9  L10  L11  L12  L13  L14  L15
2          L0   L1   L2   L3   L4   L5   L6   L7   L8   L9  L10  L11  L12  L13  L14  L15
3          L0   L1   L2   L3   L4   L5   L6   L7   L8   L9  L10  L11  L12  L13  L14  L15
4         L16  L17  L18  L19  L20  L21  L22  L23  L24  L25  L26  L27  L28  L29  L30  L31
5         L16  L17  L18  L19  L20  L21  L22  L23  L24  L25  L26  L27  L28  L29  L30  L31
6         L16  L17  L18  L19  L20  L21  L22  L23  L24  L25  L26  L27  L28  L29  L30  L31
7         L16  L17  L18  L19  L20  L21  L22  L23  L24  L25  L26  L27  L28  L29  L30  L31
8         L32  L33  L34  L35  L36  L37  L38  L39  L40  L41  L42  L43  L44  L45  L46  L47
9         L32  L33  L34  L35  L36  L37  L38  L39  L40  L41  L42  L43  L44  L45  L46  L47
10        L32  L33  L34  L35  L36  L37  L38  L39  L40  L41  L42  L43  L44  L45  L46  L47
11        L32  L33  L34  L35  L36  L37  L38  L39  L40  L41  L42  L43  L44  L45  L46  L47
12        L48  L49  L50  L51  L52  L53  L54  L55  L56  L57  L58  L59  L60  L61  L62  L63
13        L48  L49  L50  L51  L52  L53  L54  L55  L56  L57  L58  L59  L60  L61  L62  L63
14        L48  L49  L50  L51  L52  L53  L54  L55  L56  L57  L58  L59  L60  L61  L62  L63
15        L48  L49  L50  L51  L52  L53  L54  L55  L56  L57  L58  L59  L60  L61  L62  L63
```

In terms of contents, D[0][0] (owned by lane 0) contains partial results for dense row 0 using contributions from lanes 0, 16, 32, 48 (i.e., $\sum_{k=0}^{63} a[0][k_i] * b[k_i][0]$ where $i \in {0, 16, 32, 48}$). Similarly, D[1][0] (owned by lane 0) holds the analogous partial results for dense row 0 using contributions from lanes 1, 17, 33, 49. In other words, each register owns partial results for the dense row.

These partials are then fused with:

```
D[0] = D[0] + D[1]
D[1] = D[2] + D[3]
```

At this point, the per even/odd lane-pair output (now a 2×2 tile derived from the original 4×2) looks like:

```
even.d0   odd.d0
even.d1   odd.d1
```

This is then shuffled to:

```
even.d0  even.d1
odd.d0   odd.d1
```

So the final logical row-to-thread mapping becomes:

```
       0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
0     L0  L0  L2  L2  L4  L4  L6  L6  L8  L8  L10 L10 L12 L12 L14 L14 L16
1     L1  L1  L3  L3  L5  L5  L7  L7  L9  L9  L11 L11 L13 L13 L15 L15 L17
2    L16 L16 L18 L18 L20 L20 L22 L22 L24 L24 L26 L26 L28 L28 L30 L30 L32
3    L17 L17 L19 L19 L21 L21 L23 L23 L25 L25 L27 L27 L29 L29 L31 L31 L33
4    L32 L32 L34 L34 L36 L36 L38 L38 L40 L40 L42 L42 L44 L44 L46 L46 L48
5    L33 L33 L35 L35 L37 L37 L39 L39 L41 L41 L43 L43 L45 L45 L47 L47 L49
6    L48 L48 L50 L50 L52 L52 L54 L54 L56 L56 L58 L58 L60 L60 L62 L62 L64
7    L49 L49 L51 L51 L53 L53 L55 L55 L57 L57 L59 L59 L61 L61 L63 L63 L65
```

This is then cast into `__half2` fragments per thread and written back using `global_atomic_pk_add_f16` (likely to support split-K accumulation), completing the processing of the one tile. 

Note: in the real kernel for OP_M=8 they load at least an **8×256** A tile per iteration (reg0 + reg1 per lane) and perform **OPS SMFMACs** per K block. The diagrams above show one 8×128 half‑tile for clarity, but the logic is identical at larger K.
