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

# bounded-local draft (~6 min on a laptop; what produced output_draft/)
python run_sesar_recovery.py \
    --n-sources 28 --n-master-epochs 16 --baseline-days 180 --n-freq 600 \
    --k-values 1,2,4 --n-epochs-values 4,8,12,16 --max-templates 60 \
    --outdir output_draft
```

The committed `output_draft/` uses a deliberately *sparse* master cadence (16
epochs/band over one 180-day season) so the K-sweep sits in a discriminating regime
rather than saturating at 1.0 — at that density the headline result is the
**recovery-vs-N_epochs** panel, where FTP's learned shape prior beats GLS most at the
sparsest epoch counts (N=4: 0.14 vs 0.04; converging by N=16). Treat these absolute
numbers as a small-population draft (28 sources → ~0.036 granularity).

Outputs: `results.json` + `results.npz` (raw numbers) and, when matplotlib is
installed, `figures/` (recovery-vs-K, recovery-vs-N_epochs, and the two-panel
figure).

## Scale / compute note

The script **defaults** are the full bounded-ugriz config (98 templates,
`n_sources=128`, `n_freq=10000`, 3-year baseline). Because the per-frequency
multiband solve is ~1 ms, that full config is a multi-hour local job — run it on a
RunPod instance (spin up small, **tear down after**), not locally. The committed
`output_draft/` uses the faster sub-config above so the first draft figures could be
produced locally this session; treat its absolute numbers as a draft, the
full-resolution run as the publication figure.
