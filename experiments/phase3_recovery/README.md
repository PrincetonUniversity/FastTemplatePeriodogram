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

# bounded-local draft (~20-30 min on a laptop; what produced output_draft/)
python run_sesar_recovery.py \
    --n-sources 40 --n-master-epochs 80 --baseline-days 180 --n-freq 1200 \
    --k-values 1,2,4,8 --n-epochs-values 5,10,20,40,80 \
    --outdir output_draft
```

Outputs: `results.json` + `results.npz` (raw numbers) and, when matplotlib is
installed, `figures/` (recovery-vs-K, recovery-vs-N_epochs, and the two-panel
figure).

## Scale / compute note

The script **defaults** are the full bounded-ugriz config (98 templates,
`n_sources=128`, `n_freq=10000`, 3-year baseline). Because the per-frequency
multiband solve is ~1 ms, that full config is a multi-hour local job — run it on a
RunPod instance (spin up small, **tear down after**), not locally. The committed
`output_draft/` uses the faster sub-config above (single ~180-day season, coarser
grid, `n_sources=40`) so the first draft figures could be produced locally this
session; treat its absolute numbers as a draft, the full-resolution run as the
publication figure.
