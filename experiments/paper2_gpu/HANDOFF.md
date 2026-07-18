# Paper-2 FTP engine — handoff (2026-07-18)

Canonical engine/compute handoff for the PS1 3π faint-RRL template search. Read
this first. Science plan = `PAPER2_BRIEF.md` (parent dir); GPU feasibility =
`GPU_FEASIBILITY.md` (this repo); this doc is the **compute engine + cost + next
steps**. All work is on branch **`paper2-fast-multiband`** (off `dev`, 3 commits,
NOT merged, `master` untouched). Remote: `origin` (PrincetonUniversity) — branch
is LOCAL only, push before any cloud/remote session.

---

## 0. TL;DR
- **Delivered (measured, bit-identical): batched multiband `floating_offsets` = 12–14× CPU** per object at H=8 (up to 36× at H=4), plus an exact NFFT-free direct-summation path (faster for sparse N), plus the **first-ever FTP GPU run** (~38×/core on an A100, launch-bound floor).
- **Locked design** (this planning session): all bands always; **staged H** (H≈4 RRab / H≈2 RRc detection → H=8 refine at candidates); **purity by generative model-comparison at the candidate** (RRL vocab + EW + DRW + constant) + physical priors (dereddened color / instability strip / period box / Bailey / RRab asymmetry) — NOT a discriminative vocabulary, so the full-grid vocabulary stays RRL-only and small.
- **Revised full-search cost: ~$30–70** (lossless) vs the ~$1–2k planning basis — ~20–40× cheaper. Confirmed part is the 12–14× batching; the rest is the H/staging lever (projected, measurable) + near-free purity.
- **Next session = two tracks**: (A) fused-kernel GPU optimization, (B) the 5 measurements that convert "projected" → "confirmed" (§4).

---

## 1. What's done (branch `paper2-fast-multiband`)
| Commit | What | Status |
|---|---|---|
| c85ba5a | `multiband.multiband_power_spectrum_batched` — batched-over-frequency floating_offsets solve; deep-dip deferral (cond<3e-3) to per-freq reference | measured 12–14× @H8, parity 6.7e-16 vs eigvals, **full suite 494 pass** |
| c85ba5a | `summations.direct_summations_batched` — exact NFFT-free direct sums | 1.4–2.4× vs NFFT for N≲25/band; more valuable at PS1 `f_min/df≈9000` (NFFT wastes the [0,f_min] grid) |
| ec88db6 | `experiments/paper2_gpu/gpu_ftp_proto.py` — xp-agnostic (numpy/CuPy) batched pipeline + `bench_throughput.py` + `validate_vs_package.py` | validated on CPU (median ΔP 5.6e-17); A100 run 2.6 µs/LC/freq @H8 |
| ce41e4f | `experiments/paper2_gpu/rootfind/` — validated batched Aberth-Ehrlich rooter | 3.2× vs eigvals @H8, ~2e-15, GPU rooter (CuPy has no batched non-symm eig) |

Only `floating_offsets` is batched (the validated model). Fixed-offset `sesar`
mode extends easily (same cost) but needs the deferral tuned for its between-band
cancellation — do it only if the science switches to fixed colors.

## 2. Locked design decisions (this session)
1. **Model**: `floating_offsets` (per-band mean floats; robust to reddening). The
   "SesarOracle" is a floating-offset fit — *not* fixed offsets. Fixed-offset
   (`sesar`) mode is an option, more sensitive only if griz colors are trusted.
2. **Bands**: always all available (griz; z/y fold in when present). SNR adds in
   quadrature (~0.6 mag deeper vs 1 band); the phase-max runs on ONE combined
   polynomial so K_bands is nearly free; K≥3 also conditions MM' well (fewer
   fallbacks). Never drop bands to save compute.
3. **H (harmonics) — STAGED, lossless**:
   - Detection scan over the full grid at **H≈4 (RRab), H≈2 (RRc)**, regularized.
     RRab power saturates by ~H4–5 (~3–5% SNR loss vs H8); RRc ~all in H≤2.
   - **Precision refine** at candidate peaks at **H=8, unregularized, fine local
     grid** → exact period + unbiased amplitude/phase. Peak *height* (detection)
     saturates at low H; peak *width* (precision) recovered at candidates.
   - Cost ∝ ~H^2.3 (grid ∝ H, per-freq ∝ H^1.3), so H8→4 ≈ 5× on the dominant
     stage. NB: the b8 headline grid-starved H=8 (peak width ~Rayleigh/H); the
     fair recovery-vs-H comparison uses EACH H at its OWN converged grid (§4.1).
4. **Vocabulary size (the "K" cost lever)**: keep the DETECTION vocabulary
   RRL-only and small (k-medoids/Procrustes from bright anchors, `catalog_builder.py`
   / `phase3_vocab_design.md`). Contaminant discrimination is NOT extra detection
   templates — it's model-comparison at candidates (below). Recovery-vs-vocab-size
   is a measurable lever (§4.3).
5. **Purity — generative model-comparison at the candidate + physical priors**
   (NOT a discriminative vocabulary; label-scarce faint regime):
   - At each candidate period, compare RRL-vocab fit vs a couple **contact-EW**
     templates vs **DRW/QSO** (H1) vs **constant** (H0) → winning class + FAP
     (H2 vs H0) + DRW-LR (H2 vs H1). This is `PAPER2_BRIEF.md` §4c.
   - Physical priors (all free / lossless when boxed from known RRL): **dereddened
     color → RRL instability-strip box + standard-candle M_r≈+0.6** (strongest
     discriminant; per-band offsets ARE the colors, deredden via dust maps);
     **period box 0.2–1.0 d + Bailey (amplitude–period, griz amplitude ratios)**;
     **RRab sawtooth asymmetry** (free from the FTP phase fit — separates RRab
     from symmetric EWs).
   - Cost: ×~1.2–1.4 on detection, candidate-localized. No new full-grid templates.
6. **Staging is the efficiency principle**: full-grid work is cheap (low-H,
   regularized, RLL-only vocab); expensive work (H=8, unregularized, EW/DRW,
   fine grid) happens only at the handful of candidates per object.

## 3. Cost estimate (revised) — **VOID per the 2026-07-18 audit**

> **2026-07-18 audit (`audit_20260718/AUDIT_2026-07-18.md`):** this table's
> anchors are nfreq=8000 and single-template; the real PS1 RRL grid is ~4–20×
> larger and detection runs over K_vocab templates — neither multiplier is in
> the chain, so do **not** quote $30–70. Countervailing: the audit measured
> staging saving ~8.8× (not 5×), and the vetted efficiency hunt banked a
> further ~2–2.5× SAFE CPU stack + SCIENCE-gated ~4× cascade + injection-
> campaign localization. Rebuild bottom-up once Track-B #5 pins nfreq /
> candidate count / vocab size.
| Step | Factor | Status |
|---|---|---|
| Planning basis (per-freq scan, H=8, full grid) | $1–2k | prior session |
| Batched multiband CPU (bit-identical) | ÷12–14 | **MEASURED** |
| → ~$100–170 | | |
| Staged low-H detection (H4/H2 + matched grid) | ÷~3–5 | projected (§4.1) |
| → ~$25–55 | | |
| Purity (candidate-localized) | ×~1.2–1.4 | negligible on volume |
| **→ full lossless search + purity ≈ $30–70** | | |
| Injection completeness campaign (~10× volume) | | ~$300–700 CPU, or few-hundred fused-GPU |

CPU pilot is ready TODAY at these rates; GPU is the injection-volume lever.

## 4. Next session — Track B: measurements to lock the "projected" pieces
Harness: `ftperiodogram/recovery.py`, `baselines.FTPEstimator`/`SesarOracleEstimator`,
`experiments/phase3_recovery/`. Run on REAL PS1 faint cadences (WP P0.2 cache) where
possible, synthetic first for shape.
1. **Recovery vs H at each H's converged grid** (RRab & RRc separately). Confirms
   H4/H2 loses ≤ few % vs H8. Biggest cost lever. ← do first.
2. **Regularization recall**: does the ridge first-pass preserve the peak? (Predict
   neutral-to-better via false-peak suppression; MUST measure before relying on it.)
   Then re-fit unregularized at the peak for amplitude.
3. **Vocabulary size vs recall**: how many RLL templates before recall saturates.
4. **Deep-dip fallback rate on REAL PS1 cadences** (GPU_FEASIBILITY risk #2 — only
   measured on synthetic so far; sets whether the Aberth rooter is load-bearing).
5. **Exact PS1 cost inputs from WP P0.2**: candidate count after |b|+point-source+
   variability+color cuts; `nfreq` from the 5.5-yr baseline (df≤0.2/T); vocab size.
   Plug into §3 for the final $.

## 4b. Next session — Track A: fused-kernel GPU optimization

> **2026-07-18 audit re-ordering:** run the Stage-2 FP64 accuracy spike FIRST
> on the pod (GPU_FEASIBILITY §8), and bake into the fused-kernel design: the
> K.18 positive-amplitude filter (mandatory — see FINDINGS corrections),
> candidate compaction, shared-memory coef staging, device sums-reuse across
> K/subtypes, Chebyshev trig. Do-not-dos (vetted): CUDA graphs, H2D overlap;
> Aberth wiring deprioritized pending the B4 real-cadence dip rate.

The cupy prototype (§1) is a **launch/memory-bound floor** (honest ~20–30×/core,
see FINDINGS corrections), NOT the ceiling. `GPU_FEASIBILITY.md` roofline is ~200×/core with fused kernels. Targets,
in priority order (validate each bit-vs the frozen CPU scan; FP64 throughout;
datacenter cards only):
1. **Fuse the Newton-polish kernel** (Stage 3, ~77–90% of cost; the launch-bound
   part). One kernel per (frequency,candidate): the 6 Horner evals × 8 steps
   register-resident. The cupy `_horner` loop issues ~ncoef kernels/call; a
   one-shot precomputed-phi-powers variant was *slower* (an (nB,Cmax,ncoef) ≈8 GB
   tensor → memory-bound) — so neither cupy-level form works; a RawKernel/custom
   CUDA kernel is required.
2. **Batched direct-summation kernel** (Stage 1), one block per source,
   register-resident (right structure for 10⁷ tiny sources; NFFT per tiny source
   has bad overhead — literature-confirmed). Handles the [0,f_min] waste for free.
3. **Stage-2 O(H²) assembly kernel** in FP64 — the new numerically load-bearing
   piece (moment-sum cancellation under concentrated weights; validate first, per
   GPU_FEASIBILITY §7 risk #1).
4. **Wire the Aberth rooter** (`rootfind/`) for deep-|MM|-dip rows (rate from §4.4).
5. Reuse cuvarbase infra (batched cuFFT `_cufft.py`, memory classes, batch-over-
   objects) — the eventual home is cuvarbase, not this package (GPU_FEASIBILITY §5).
6. Benchmark on A100 AND H100 (H100 ~1.6× FP64); report per-node-vs-per-GPU.

RunPod: REST API `rest.runpod.io/v1` + Bearer `RUNPOD_API_KEY` (GraphQL is
Cloudflare-blocked); A100-SXM4-80GB $1.49/hr, `pip install cupy-cuda12x`; create
pod with `PUBLIC_KEY` env for SSH; **always DELETE the pod when done** (this
session: created/measured/torn-down, spend ~$0.87). GPU IDs & pod schema in the
git history of this dir.

## 5. Future sessions — running the search (pipeline shape)
1. Load candidate LCs (PS1 forced warp photometry, WP P0.2; flat (t,y,bands,dy)).
2. **Detection**: batched multiband floating_offsets FTP over the RLL vocab
   (H4 RRab / H2 RRc), regularized, full grid → power spectrum + candidate peaks
   + FAP (Track-D null machinery).
3. **Refine** at top candidates: H=8, unregularized, fine local grid → precise
   period + params.
4. **Classify** (per candidate): RRL vs EW vs DRW vs constant model-comparison +
   physical-prior box (color/dust/period/Bailey/asymmetry).
5. **Output** catalog + run the injection campaign for the completeness map.

## 6. Artifact index
- CPU engine: `ftperiodogram/multiband.py` (`multiband_power_spectrum_batched`,
  `_BATCHED_SCAN_MODES`, `_BATCHED_DEFER_RTOL`), `ftperiodogram/summations.py`
  (`direct_summations_batched`).
- GPU prototype: `experiments/paper2_gpu/gpu_ftp_proto.py`, `bench_throughput.py`,
  `validate_vs_package.py`, `FINDINGS.md`.
- Rooter: `experiments/paper2_gpu/rootfind/` (`rootfinders.py`, README).
- This handoff + `PAPER2_BRIEF.md` + `GPU_FEASIBILITY.md` + `phase3_vocab_design.md`.
