# Phase 3.2 production headline -- bv universe (H=8, 256 src x 1 seed)

RunPod fleet result (1 seed, mean across seeds). Figures in this dir.

## Recovery vs N_epochs @ knee K=[2] (the headline)
| epochs/band | 4 | 8 | 12 | 16 | 24 | 40 |
|---|---|---|---|---|---|---||
| FTP(PAM) | 0.133 | 0.832 | 0.980 | 0.992 | 1.000 | 1.000 |
| FTP(greedy) | 0.082 | 0.801 | 0.984 | 0.996 | 1.000 | 1.000 |
| GLS | 0.078 | 0.645 | 0.910 | 0.988 | 1.000 | 1.000 |
| MHLS | 0.000 | 0.000 | 0.332 | 0.688 | 0.785 | 0.801 |
| multiband-LS | 0.012 | 0.496 | 0.816 | 0.953 | 1.000 | 1.000 |

## Recovery vs K
- sparse (N=6): FTP(PAM) ['0.441', '0.625', '0.621', '0.625'] ; GLS 0.395 ; MHLS 0.000 ; mb-LS 0.215
- dense (N=60): FTP(PAM) ['1.000', '1.000', '1.000', '1.000'] ; GLS 1.000 ; MHLS 0.801 ; mb-LS 1.000

## Cost vs accuracy (FTP vs Sesar-style non-linear oracle)
- FTP recovery 0.375 vs oracle recovery 0.375 (gold-standard equivalence)
- speedup 2.5x measured in-harness (oracle n_tau=128; the ratio is grid-stable and grows with N_obs)
