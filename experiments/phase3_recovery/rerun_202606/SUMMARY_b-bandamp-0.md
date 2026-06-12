# Robustness arm b-bandamp-0

**band-amp ratio 1.4 (g brighter)** (N-sweep @K=2, 256 src/seed):

| method | N=4 | N=8 | N=12 | N=16 | N=24 | N=40 |
|---|---|---|---|---|---|---|
| ftp_pam | 0.082 | 0.898 | 0.988 | 1.000 | 1.000 | 1.000 |
| ftp_greedy | 0.031 | 0.793 | 0.988 | 1.000 | 1.000 | 1.000 |
| gls | 0.047 | 0.328 | 0.867 | 0.992 | 1.000 | 1.000 |
| mhls | 0.047 | 0.422 | 0.715 | 0.711 | 0.801 | 0.836 |
| mbls | 0.012 | 0.223 | 0.652 | 0.922 | 0.992 | 1.000 |
| ce | 0.004 | 0.078 | 0.496 | 0.840 | 0.984 | 0.992 |

FTP(PAM)>=GLS at every N: True. arms metadata: {"truth_universe": null, "library_holdout_frac": 0.0, "band_amp_ratio": 1.4, "n_truth": 98, "n_library": 98, "band_amplitudes": {"g": 1.4, "r": 1.0}}
