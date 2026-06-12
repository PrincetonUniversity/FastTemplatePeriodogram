# Robustness arm b-xuniv-0

**cross-universe: BV truth, sesar library** (N-sweep @K=2, 256 src/seed):

| method | N=4 | N=8 | N=12 | N=16 | N=24 | N=40 |
|---|---|---|---|---|---|---|
| ftp_pam | 0.102 | 0.695 | 0.973 | 0.996 | 1.000 | 1.000 |
| ftp_greedy | 0.109 | 0.887 | 0.988 | 1.000 | 1.000 | 1.000 |
| gls | 0.086 | 0.672 | 0.926 | 1.000 | 1.000 | 1.000 |
| mhls | 0.086 | 0.586 | 0.734 | 0.773 | 0.809 | 0.820 |
| mbls | 0.035 | 0.590 | 0.840 | 0.977 | 1.000 | 1.000 |
| ce | 0.000 | 0.137 | 0.492 | 0.840 | 0.984 | 1.000 |

FTP(PAM)>=GLS at every N: False. arms metadata: {"truth_universe": "bv", "library_holdout_frac": 0.0, "band_amp_ratio": 1.0, "n_truth": 560, "n_library": 98, "band_amplitudes": null}
