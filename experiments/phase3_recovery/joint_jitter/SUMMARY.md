# Phase 3.4 joint-vs-pipeline gap -- sesar universe

K=2, 64 sources, H=4, max_iter=8, seed=0, bands=gr, jitter=0.15, mean_mag=19, T=60d

| N_epochs | pipeline | joint | gap | GLS |
|---|---|---|---|---|
| 4 | 0.656 | 0.656 | +0.000 | 0.281 |
| 6 | 0.984 | 0.984 | +0.000 | 0.547 |
| 8 | 1.000 | 1.000 | +0.000 | 0.812 |
| 12 | 1.000 | 1.000 | +0.000 | 0.984 |
| 20 | 1.000 | 1.000 | +0.000 | 1.000 |

Hypothesis: gap > 0 in the sparse regime, -> 0 as N_epochs grows.
