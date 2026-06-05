# Phase 3.4 joint-vs-pipeline gap -- sesar universe

K=2, 32 sources, H=4, max_iter=6, seed=0, bands=gr

| N_epochs | pipeline | joint | gap | GLS |
|---|---|---|---|---|
| 6 | 0.906 | 0.906 | +0.000 | 0.500 |
| 10 | 1.000 | 1.000 | +0.000 | 0.969 |
| 18 | 1.000 | 1.000 | +0.000 | 1.000 |
| 30 | 1.000 | 1.000 | +0.000 | 1.000 |

Hypothesis: gap > 0 in the sparse regime, -> 0 as N_epochs grows.
