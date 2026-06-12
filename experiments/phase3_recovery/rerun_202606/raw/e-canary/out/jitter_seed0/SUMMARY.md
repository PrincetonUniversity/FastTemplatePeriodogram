# Phase 3.4 joint-vs-pipeline gap -- sesar universe

K=2, 256 sources, H=8, max_iter=8, seed=0, bands=gr, jitter=0.15, mean_mag=19, amplitude=0.5, T=60d

| N_epochs | rec pipe | rec joint | rec gap | assign pipe | assign joint | mech gap | n_sub (p/j) | null (p/j) | GLS | EM best/n |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.672 | 0.672 | +0.000 | 0.721 | 0.721 | +0.000 | 172/172 | 0.61/0.61 | 0.207 | 0/3 |
| 8 | 1.000 | 1.000 | +0.000 | 0.855 | 0.855 | +0.000 | 256/256 | 0.61/0.61 | 0.863 | 0/3 |
| 12 | 1.000 | 1.000 | +0.000 | 0.918 | 0.926 | +0.008 | 256/256 | 0.61/0.60 | 0.988 | 1/4 |
| 20 | 1.000 | 1.000 | +0.000 | 0.941 | 0.941 | +0.000 | 256/256 | 0.61/0.61 | 1.000 | 0/3 |
| 40 | 1.000 | 1.000 | +0.000 | 0.961 | 0.973 | +0.012 | 256/256 | 0.61/0.61 | 1.000 | 1/4 |

rec gap = period-recovery (1%) joint-minus-pipeline; mech gap = correct-template-assignment on the correct-period subset (the shape mechanism, unclipped by period recovery). n_sub = subset size; null = majority-target null on that subset (the honest chance bar, not 1/K). EM best/n = accepted best iteration / iterations run (early stop on the continuous power-margin val signal).
Hypothesis: gap > 0 in the sparse regime, -> 0 as N_epochs grows.
