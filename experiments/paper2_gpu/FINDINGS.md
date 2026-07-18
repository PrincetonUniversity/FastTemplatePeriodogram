# GPU prototype — first measured FTP throughput (2026-07-18)

xp-agnostic (numpy/CuPy) batched multiband `floating_offsets` FTP, batched over
BOTH sources and frequencies (the Paper-2 throughput regime: many sparse griz
light curves). `gpu_ftp_proto.py` runs unchanged on CPU (`xp=numpy`) or GPU
(`xp=cupy`); `validate_vs_package.py` checks it against `ftperiodogram`.

## Correctness
Prototype vs the `ftperiodogram` package (floating_offsets scan), griz K=4 H=8:
median |ΔP| = 5.6e-17, identical argmax, identical peak power. The max |ΔP|
(~0.13) is confined to the ~4% deep-|MM|-dip frequencies the prototype's
simplified scan omits (no dip-candidate seeding, no exact-root fallback). A
production port routes those rare rows through a batched Aberth–Ehrlich rooter
(validated separately: 3.2× vs eigvals at H=8, ~2e-15 accuracy, GPU-portable —
the only rooter that batches, since CuPy has no batched non-symmetric eig).

## Throughput — first-ever FTP GPU run
Hardware: RunPod NVIDIA **A100-SXM4-80GB** (9.7 TFLOPS FP64), CuPy 14.1, FP64.
Baseline: same prototype, `xp=numpy`, one Apple M5 P-core.
Config: K=4 griz, N=15/band, nfreq=8000, FP64 throughout.

| H | CPU (M5 1 core) | A100 (this prototype) | speedup | 1e7 LC on 1 A100 |
|---|---|---|---|---|
| 8 | 97.7 µs/LC/freq (1.3 LC/s) | **2.6 µs/LC/freq (48 LC/s)** | **~38×/core** | **~58 GPU-hr (~2.4 days)** |
| 4 | 28.8 µs/LC/freq (4.3 LC/s) | 1.9 µs/LC/freq (65 LC/s) | ~15×/core | ~43 GPU-hr |

Consistent to <1% across repeats (the one-off 20 LC/s reading was a cold-clock
transient; idle SM clock 210 MHz boosts under load).

## Reading it honestly
- This is a **conservative floor**, not the achievable GPU ceiling. The cupy
  prototype is **kernel-launch / memory-bound**: `_horner` issues ~ncoef small
  kernels per call and the Newton polish repeats that 8×. A one-shot
  precomputed-phi-powers variant was *slower* (the (nB,Cmax,ncoef) powers
  tensor is ~8 GB/chunk → memory-bound). Neither cupy-level form is the answer.
- GPU_FEASIBILITY.md's roofline (~100–400×/core, central ~200×) assumes a
  **fused custom CUDA kernel** keeping the whole per-candidate Newton polish in
  registers. The measured ~38× vs that ~200× is exactly the fusion gap plus the
  doc's own "first measurement could move 2–3×" caveat — i.e. the measurement
  is consistent with, and refines, the feasibility estimate rather than
  contradicting it.
- Batch-over-sources saturates the A100 by B≈128 (44→48 LC/s from B=32→512), so
  the survey grid (~8k freqs) needs source-batching to fill the device — as the
  doc predicted.

## For Paper 2 (cost, WP P0.6)
At the (conservative) prototype rate, a full 1e7-LC griz H=8 pass ≈ 58 A100-hr ≈
$60–110 at RunPod A100 pricing (~$1.0–1.9/hr community/secure). A fused kernel
would cut that several-fold. The CPU scan-default is the pilot-scale path
(12–14× faster than before on this branch); GPU is the 1e7-source production
lever. First real GPU number in hand — no FTP GPU kernel had ever run.

Reproduce: `python bench_throughput.py --backend {numpy,cupy} --B 256 --H 8
--nfreq 8000`. RunPod A100 pod, `pip install cupy-cuda12x`, scp the two .py
files. Spend for this run: ~$0.87.
