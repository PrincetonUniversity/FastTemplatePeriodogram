# Robustness arm b-holdout-0

**holdout(0.5): truth shapes held out of the library** (N-sweep @K=2, 256 src/seed):

| method | N=4 | N=8 | N=12 | N=16 | N=24 | N=40 |
|---|---|---|---|---|---|---|
| ftp_pam | 0.164 | 0.891 | 0.965 | 0.996 | 1.000 | 1.000 |
| ftp_greedy | 0.152 | 0.879 | 0.973 | 0.996 | 1.000 | 1.000 |
| gls | 0.047 | 0.449 | 0.898 | 0.988 | 1.000 | 1.000 |
| mhls | 0.047 | 0.566 | 0.750 | 0.805 | 0.836 | 0.887 |
| mbls | 0.004 | 0.289 | 0.746 | 0.961 | 1.000 | 1.000 |
| ce | 0.000 | 0.109 | 0.527 | 0.871 | 0.984 | 0.996 |

FTP(PAM)>=GLS at every N: True. arms metadata: {"truth_universe": null, "library_holdout_frac": 0.5, "band_amp_ratio": 1.0, "n_truth": 49, "n_library": 49, "band_amplitudes": null}
