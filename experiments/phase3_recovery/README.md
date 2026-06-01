# Phase 3.2 — Sesar recovery experiment

`run_sesar_recovery.py` is the first paper-quality recovery driver (PLAN §3.2
deliverables i + ii): it fetches the Sesar 2010 ugriz templates, builds a
PAM-learned vocabulary, and sweeps period recovery over one frozen, seeded
synthetic multiband population — both vs vocabulary size **K** and, at the K-sweep
knee, vs per-band epoch count **N_epochs** (the sparse-regime story). The FTP@H=1
single-cosine (== GLS) baseline is included as the reference. Headline criterion is
the hard 1% fractional rule; the search grid is an explicit `[f_min, f_max]` band.

## Running

```bash
# tiny end-to-end wiring check (~1 min)
python run_sesar_recovery.py --smoke

# bounded-local draft on 10 cores (~19 min; what produced output_draft/)
python run_sesar_recovery.py \
    --n-sources 96 --n-master-epochs 20 --baseline-days 365 --n-freq 3000 \
    --k-values 1,2,4,8 --n-epochs-values 4,8,12,16,20 --n-jobs 10 \
    --outdir output_draft
```

The committed `output_draft/` (full 98-template ugriz universe, 96 sources, 1-year
baseline, `--n-jobs 10`) makes the **recovery-vs-N_epochs** panel the headline: a
single learned RRab template (the knee is K=1) recovers far more than the GLS sinusoid
in the sparse regime — N=4: **0.375 vs 0.104**, N=8: **0.958 vs 0.604** — converging by
N=16. Remember N is **epochs per band**, so with g+r the x-axis spans 8→40 total
points per light curve.

The **recovery-vs-K** panel is flat at 1.0 here, and that is a real result rather than
saturation alone: at ≥~12 epochs/band over a one-year baseline the period is
recoverable by any method, and where it *is* hard (the sparse N-sweep) K=1 already
captures the whole shape-prior gain — RRab shapes are homogeneous enough that
vocabulary *size* adds little; the win is template *shape* vs. sinusoid. A
discriminating recovery-vs-K panel needs a **sparse** master cadence (e.g.
`--n-master-epochs 6`), where it tests whether K>1 helps when data is scarce.

`output_sparse_k/` is exactly that run (6 epochs/band, 96 sources): the recovery-vs-K
panel is now discriminating — FTP holds **~0.88–0.92** across K=1–8, far above the GLS
line at **0.25** — and shows K=1–2 already suffices (larger K marginally hurts via
catalog-max alias peaks). Together the two dirs give deliverable (ii) (`output_draft/`,
the N_epochs headline) and deliverable (i) (`output_sparse_k/`, the K panel).

Outputs: `results.json` + `results.npz` (raw numbers) and, when matplotlib is
installed, `figures/` (recovery-vs-K, recovery-vs-N_epochs, and the two-panel
figure).

## Scale / compute note

The script **defaults** are the full bounded-ugriz config (98 templates,
`n_sources=128`, `n_freq=10000`, 3-year baseline). The per-frequency multiband solve
is ~1 ms, so cost scales as `sum(K) x n_freq x n_sources`. The single biggest lever is
`--n-jobs`: the per-source loop is embarrassingly parallel, so `--n-jobs -1` gives
roughly an Nx speedup on an N-core box (this draft saw ~8x on 10 cores). Even so the
full default config is ~1.5-2.5 hr on 10 cores (the 10k-freq grid dominates) — for the
true publication-resolution run, a many-core cloud box (`--n-jobs -1`, torn down after)
brings it to ~10-15 min. The committed `output_draft/` is the bounded-but-parallel
config above; treat its absolute numbers as a draft, the full-resolution run as the
publication figure.
