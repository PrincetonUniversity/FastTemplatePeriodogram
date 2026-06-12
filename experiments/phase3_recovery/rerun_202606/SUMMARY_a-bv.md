# Arm (a) bv -- 1 seed

**BV griz** (N-sweep @K=2, 256 src/seed):

| method | N=4 | N=8 | N=12 | N=16 | N=24 | N=40 |
|---|---|---|---|---|---|---|
| ftp_pam | 0.148 | 0.895 | 0.988 | 1.000 | 1.000 | 1.000 |
| ftp_greedy | 0.105 | 0.879 | 0.984 | 1.000 | 1.000 | 1.000 |
| gls | 0.086 | 0.672 | 0.926 | 1.000 | 1.000 | 1.000 |
| mhls | 0.086 | 0.586 | 0.734 | 0.773 | 0.809 | 0.820 |
| mbls | 0.035 | 0.590 | 0.840 | 0.977 | 1.000 | 1.000 |
| ce | 0.000 | 0.137 | 0.492 | 0.840 | 0.984 | 1.000 |

Sparse K-sweep PAM: 0.512 0.703 0.727 0.719 -- real K=1->2 jump (+0.19), knee=2: the diverse universe rewards vocabulary; contrast with flat sesar curve.
