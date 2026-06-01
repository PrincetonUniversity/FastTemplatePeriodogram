# Phase 3 production runs — next-session runbook

Goal: full publication-scale runs for the Paper 1 recovery story — deliverables (i)
recovery-vs-K, (ii) recovery-vs-N_epochs (the sparse-regime headline), (iii)
cost-vs-accuracy vs the Sesar oracle — plus the comparison-method baselines and a
greedy-vs-PAM vocabulary comparison. Run scaled work on **RunPod** (many-core CPU,
`--n-jobs -1`); use **Workflows / multi-agent orchestration** for the build + adversarial
verification (user has opted into heavy compute — token cost is not a constraint).

## Current state (start of next session)
- Branch **`phase3.2-figures`** (off `phase3.2-recovery-harness`), ~10 commits, suite
  213+ pass / 2 known pre-existing `test_slow_template_modeler` fails. Features 1–6 done:
  N_epochs downsample+sweep, batched `source_masks`, greedy `pam:M`, optional `figures.py`,
  Sesar experiment CLI, and the **parallel** scorer (`RecoveryScorer(n_jobs=-1)`, ~8× on 10
  cores, mask-identical to serial).
- Draft results committed under `experiments/phase3_recovery/output_draft/` (headline
  N_epochs panel) and `output_sparse_k/` (sparse-master K-sweep).
- Comparison baselines scoped in `COMPARISON_BASELINES_SCOPE.md` (not built).

## 0. Prereqs / infra (do first)
1. **Merge PR #39** (harness) into `dev`; **rebase `phase3.2-figures` onto `dev`** (and
   open its own PR if desired). Confirm `pytest ftperiodogram/tests -q` green.
2. **RunPod** (no CLI/creds set up locally yet — wire this first): get an API key, or use
   the **`cloudrouter` skill** to provision a remote VM. Pick a **many-core CPU** pod
   (recovery is CPU-bound and parallel — a GPU does nothing here; cuvarbase is the GPU
   path, a different stack). Sync the repo, create the venv (numpy/scipy/nfft + matplotlib),
   warm the Sesar cache (`fetch_sesar_templates()`), run with `--n-jobs -1`, **tear down
   after** (see [[runpod-compute]]).

## 1. Build comparison baselines (CODE — must precede the comparison runs)
Per `COMPARISON_BASELINES_SCOPE.md`: new `ftperiodogram/baselines.py`, numpy/scipy only
(**no gatspy/astroML**):
- **GLS** (have, FTP@H=1) · **MHLS** free-shape upper bound · **multiband-LS** competitor
  (VanderPlas & Ivezić 2015) · **Sesar oracle** (scipy.optimize, slow, *timed* — drives
  deliverable iii) · optional **BLS**.
- Refactor `RecoveryScorer` to score a **pluggable estimator** `estimator(t,y,bands,dy,freqs)
  -> P_rec`, so every method (FTP, GLS, MHLS, …) runs on the *identical* frozen population +
  grid — the fairest comparison and a single sweep overlays all curves.
- **Suggested orchestration:** a Workflow that fans out one agent per baseline (implement +
  adversarially verify each against a brute-force `<1e-6` reference on a clean injection),
  then synthesizes `baselines.py` + tests. Linear LS baselines must match the brute-force
  reference; never loosen the oracle tolerance.

## 2. Production recovery runs (RunPod, `--n-jobs -1`)
Explicit `[f_min,f_max]=[1.0,5.0]` (RR Lyrae 0.2–1.0 d). Size `n_freq` to the baseline
(3-yr ZTF-like → ~15–20k for peak resolution). `n_sources` 512–1024 with ≥3 seeds for error
bars. Universe: full 98-template Sesar ugriz (decide whether to add the ~280-template
Baeza-Villagra 2025 set). Runs:
- **(a) recovery-vs-K, dense master** (~40–60 epochs) — knee + saturation, all baselines overlaid.
- **(b) recovery-vs-K, sparse master** (~6–8 epochs) — does K>1 help when data is scarce?
  *(This session's sparse draft, 6 epochs/band, 96 sources, `output_sparse_k/`: FTP@K=1
  **0.91** vs GLS **0.25**; K=1→2→4→8 = 0.91/0.92/0.89/0.89 → K=1–2 suffices and larger K
  marginally hurts via catalog-max alias peaks. Confirms at scale that the win is shape vs
  sinusoid, not vocabulary size — worth re-checking with the augmented 280-template universe
  and the comparison baselines overlaid.)*
- **(c) recovery-vs-N_epochs at fixed K\*** — the headline sparse-regime story, all baselines.
- **(d) cost-vs-accuracy (deliverable iii)** — FTP vs the *timed* Sesar oracle at matched
  recovery; the ~10³× speedup claim.
- **(e) greedy-vs-PAM vocabulary** — recovery-driven greedy (`candidate_pool='pam:M'`) vs PAM
  at each K, showing recovery-driven selection's value.

## 3. Stratify (design §2)
Stratify recovery by magnitude/SNR, amplitude, band coverage (scorer `amplitude`/`mean_mag`/
`band_amplitudes`/`band_offsets`). Report **exact vs exact-or-harmonic** as two curves
(`harmonic_aware`).

## 4. Figures + numbers
Extend `figures.py` to overlay all baselines; save publication PDFs + `results.json/.npz`
per run with seeds/config for reproducibility.

## Locked rules (carry over)
Headline = 1% fractional; δφ_max=0.5 secondary; explicit `[f_min,f_max]` (never
`nyquist_factor`); greedy cost = honest `1−scorer(vocab)`; brute-force oracle `<1e-6` never
loosened; numpy/scipy core, matplotlib optional; small/early commits; tear down RunPod after.

## Open decisions to resolve at session start
- Universe: Sesar 98 only, or + Baeza-Villagra 280?
- `n_sources` + number of seeds (error bars)?
- RunPod path: direct `runpod` (needs API key) vs the `cloudrouter` skill?
- Greedy `pam:M` value `M` at production scale?

## Suggested session shape
- **Phase A (code, Workflow):** build `baselines.py` (parallel agents + adversarial verify) →
  rebase/merge.
- **Phase B (compute, RunPod):** provision → run the matrix (`--n-jobs -1`) → pull results →
  tear down.
- **Phase C (figures):** synthesize publication figures + numbers.
