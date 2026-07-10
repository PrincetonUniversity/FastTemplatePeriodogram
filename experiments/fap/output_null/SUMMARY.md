# WP D1 -- measured null / FAP calibration (SUMMARY)

All numbers below are computed by `measure_null.py` from this run's own Monte-Carlo output (no quoted constants).

## Configuration (production synthetic cadence + error model)

- `n_real` = 3000
- `n_obs` = 40
- `n_freq` = 10000
- `f_min` = 1.0
- `f_max` = 5.0
- `baseline_days` = 1095.75
- `season_length_days` = 270.0
- `season_period_days` = 365.25
- `mean_mag` = 15.0
- `harmonics` = [1, 3, 6, 12]
- `vocab_h` = 8
- `k_values` = [1, 2, 4, 8]
- `seed` = 0
- `seed_base` = 1000000
- median per-point sigma dy = 0.01003 mag (constant; amplitude=0)
- realized N_obs (mean/min/max) = 40.0 / 40 / 40

## Sanity anchor (audit): pure-noise max power ~0.5 at N=40

| statistic | mean | median | 99th pct |
|-----------|------|--------|----------|
| FTP H=1 | 0.4075 | 0.4013 | 0.5347 |
| FTP H=3 | 0.4404 | 0.4350 | 0.5509 |
| FTP H=6 | 0.4425 | 0.4361 | 0.5507 |
| FTP H=12 | 0.4426 | 0.4367 | 0.5527 |
| GLS | 0.4075 | 0.4013 | 0.5347 |

## FAP-vs-threshold: power thresholds at fixed FAP (single template)

| statistic | FAP=0.1 (emp / GEV) | FAP=0.05 (emp / GEV) | FAP=0.01 (emp / GEV) | FAP=0.001 (emp / GEV) |
|---|---|---|---|---|
| FTP H=1 | 0.462 / 0.464 | 0.484 / 0.485 | 0.535 / 0.529 | 0.578 / 0.583 |
| FTP H=3 | 0.491 / 0.491 | 0.511 / 0.510 | 0.551 / 0.551 | 0.599 / 0.601 |
| FTP H=6 | 0.494 / 0.493 | 0.513 / 0.513 | 0.551 / 0.555 | 0.610 / 0.608 |
| FTP H=12 | 0.495 / 0.494 | 0.513 / 0.514 | 0.553 / 0.556 | 0.608 / 0.609 |
| GLS | 0.462 / 0.464 | 0.484 / 0.485 | 0.535 / 0.529 | 0.578 / 0.583 |

## GEV tail fit parameters (scipy.stats.genextreme; c=-xi)

| statistic | c (shape) | loc | scale |
|-----------|-----------|-----|-------|
| ftp_H1 | 0.0746 | 0.3892 | 0.0360 |
| ftp_H3 | 0.0682 | 0.4239 | 0.0321 |
| ftp_H6 | 0.0583 | 0.4257 | 0.0321 |
| ftp_H12 | 0.0591 | 0.4258 | 0.0323 |
| cat_K1 | 0.0773 | 0.4041 | 0.0354 |
| cat_K2 | 0.0826 | 0.4242 | 0.0345 |
| cat_K4 | 0.0825 | 0.4407 | 0.0335 |
| cat_K8 | 0.0665 | 0.4499 | 0.0325 |
| gls | 0.0746 | 0.3892 | 0.0360 |

## null-max vs K (catalog-max over the PAM vocab, order 8)

| K | mean | median | 99th pct | FAP=0.01 thr (emp / GEV) |
|---|------|--------|----------|--------------------------|
| 1 | 0.4219 | 0.4166 | 0.5403 | 0.540 / 0.541 |
| 2 | 0.4414 | 0.4366 | 0.5606 | 0.561 / 0.556 |
| 4 | 0.4574 | 0.4522 | 0.5695 | 0.570 / 0.569 |
| 8 | 0.4666 | 0.4615 | 0.5760 | 0.576 / 0.579 |

## Timing (this machine, 8 workers)

- template build: 0.8 s
- Monte-Carlo: 3468.2 s wall for 3000 realizations (1.156 s/realization wall; 9.249 CPU-s/realization)
- host: macOS-26.0-arm64-arm-64bit, python 3.9.6, numpy 2.0.2

