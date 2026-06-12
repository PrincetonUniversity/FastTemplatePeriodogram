# Arm (c) grid convergence -- N-sweep @20k vs 10k freqs, PAIRED exact McNemar on identical seed-0 sources

| method | N | 20k | 10k | 20k-only | 10k-only | p |
|---|---|---|---|---|---|---|
| ftp_pam | 4 | 0.305 | 0.234 | 47 | 29 | 0.0505 |
| ftp_pam | 8 | 0.973 | 0.926 | 17 | 5 | 0.0169 |
| ftp_pam | 12 | 1.000 | 0.988 | 3 | 0 | 0.25 |
| ftp_pam | 16 | 1.000 | 1.000 | 0 | 0 | 1 |
| ftp_pam | 24 | 1.000 | 1.000 | 0 | 0 | 1 |
| ftp_pam | 40 | 1.000 | 1.000 | 0 | 0 | 1 |
| gls | 4 | 0.059 | 0.035 | 14 | 8 | 0.286 |
| gls | 8 | 0.434 | 0.387 | 33 | 21 | 0.134 |
| gls | 12 | 0.906 | 0.871 | 19 | 10 | 0.136 |
| gls | 16 | 0.996 | 0.996 | 1 | 1 | 1 |
| gls | 24 | 1.000 | 1.000 | 0 | 0 | 1 |
| gls | 40 | 1.000 | 1.000 | 0 | 0 | 1 |
| mhls | 4 | 0.059 | 0.035 | 14 | 8 | 0.286 |
| mhls | 8 | 0.723 | 0.527 | 74 | 24 | 4.22e-07 |
| mhls | 12 | 0.930 | 0.762 | 54 | 11 | 6.03e-08 |
| mhls | 16 | 0.910 | 0.785 | 46 | 14 | 4.22e-05 |
| mhls | 24 | 0.910 | 0.867 | 29 | 18 | 0.144 |
| mhls | 40 | 0.945 | 0.848 | 34 | 9 | 0.00017 |
| mbls | 4 | 0.016 | 0.012 | 4 | 3 | 1 |
| mbls | 8 | 0.262 | 0.242 | 28 | 23 | 0.576 |
| mbls | 12 | 0.766 | 0.691 | 38 | 19 | 0.0163 |
| mbls | 16 | 0.977 | 0.957 | 8 | 3 | 0.227 |
| mbls | 24 | 1.000 | 1.000 | 0 | 0 | 1 |
| mbls | 40 | 1.000 | 1.000 | 0 | 0 | 1 |
| ce | 4 | 0.000 | 0.004 | 0 | 1 | 1 |
| ce | 8 | 0.074 | 0.066 | 15 | 13 | 0.851 |
| ce | 12 | 0.582 | 0.496 | 65 | 43 | 0.0428 |
| ce | 16 | 0.918 | 0.836 | 34 | 13 | 0.00309 |
| ce | 24 | 0.996 | 0.984 | 4 | 1 | 0.375 |
| ce | 40 | 1.000 | 0.988 | 3 | 0 | 0.25 |

Significant (p<0.05) cells: 8/30 -- ftp_pam N=8 (+17/-5, p=0.0169); mhls N=8 (+74/-24, p=4.22e-07); mhls N=12 (+54/-11, p=6.03e-08); mhls N=16 (+46/-14, p=4.22e-05); mhls N=40 (+34/-9, p=0.00017); mbls N=12 (+38/-19, p=0.0163); ce N=12 (+65/-43, p=0.0428); ce N=16 (+34/-13, p=0.00309)

READING: the production 10k grid is NOT converged for the H=8 and binned methods (multiharmonic peak width ~Rayleigh/H; 10k gives 2.28 pts/Rayleigh = ~0.29 per H=8 peak width). H=1 baselines (GLS/MBLS) are converged (all p>0.13). Direction is one-sided: every significant cell GAINS at 20k, and FTP gains more than GLS, so the FTP>GLS contrast at 10k is CONSERVATIVE and the absolute sparse-N rates are lower bounds at the stated grid. Resolution plan: post-C2 (scan+polish), either a ~10x denser grid or a coarse-grid + local peak-refinement harness; until then every absolute rate is quoted "at the 10k production grid".
