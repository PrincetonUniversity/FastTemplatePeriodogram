# Phase 3.2 production headline -- sesar universe (H=8, 256 src x 3 seeds)

Full RunPod fleet result (3 seeds, mean across seeds). Figures in this dir.

## Recovery vs N_epochs @ knee K=[2, 2, 2] (the headline)
| epochs/band | 4 | 8 | 12 | 16 | 24 | 40 |
|---|---|---|---|---|---|---||
| FTP(PAM) | 0.197 | 0.816 | 0.961 | 0.991 | 1.000 | 1.000 |
| FTP(greedy) | 0.155 | 0.809 | 0.962 | 0.995 | 1.000 | 1.000 |
| GLS | 0.029 | 0.378 | 0.857 | 0.988 | 1.000 | 1.000 |
| MHLS | 0.003 | 0.003 | 0.341 | 0.663 | 0.816 | 0.854 |
| multiband-LS | 0.018 | 0.217 | 0.698 | 0.940 | 1.000 | 1.000 |

## Recovery vs K
- sparse (N=6): FTP(PAM) ['0.613', '0.651', '0.643', '0.643'] ; GLS 0.150 ; MHLS 0.003 ; mb-LS 0.079
- dense (N=60): FTP(PAM) ['1.000', '1.000', '1.000', '1.000'] ; GLS 1.000 ; MHLS 0.859 ; mb-LS 1.000

## Cost vs accuracy (FTP vs Sesar-style non-linear oracle)
- FTP recovery 0.438 vs oracle recovery 0.438 (gold-standard equivalence)
- speedup 2.5x measured in-harness (oracle n_tau=128; the ratio is grid-stable and grows with N_obs)
