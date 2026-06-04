# Phase 3.4 joint-vs-pipeline gap -- sesar universe

K=2, 64 sources, H=4, max_iter=8, seed=0, bands=gr, jitter=0, mean_mag=15, T=60d

| N_epochs | pipeline | joint | gap | GLS |
|---|---|---|---|---|
| 4 | 0.828 | 0.797 | -0.031 | 0.188 |
| 6 | 1.000 | 1.000 | +0.000 | 0.609 |
| 8 | 1.000 | 1.000 | +0.000 | 0.938 |
| 12 | 1.000 | 1.000 | +0.000 | 1.000 |
| 20 | 1.000 | 1.000 | +0.000 | 1.000 |

Hypothesis: gap > 0 in the sparse regime, -> 0 as N_epochs grows.
