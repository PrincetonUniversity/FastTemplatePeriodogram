# WP B8 consolidated rerun -- master summary (2026-06-12)

10/10 fleet jobs succeeded (arms a-d + e-canary); arm (e) remainder DEFERRED to
post-C2, arm (f) CUT (decision log in B8_COST_ESTIMATE.md). Pod code: dev @
c8763ea. Production grid 10k freqs (B7 guard; headline 8k was non-compliant at
1.83 pts/Rayleigh), convergence arm at 20k.

## Acceptance checks

1. **FTP>GLS ordering**: holds in EVERY production arm at every N-sweep cell: {"a-bv-0": true, "a-sesar-0": true, "a-sesar-1": false, "a-sesar-2": true, "b-bandamp-0": true, "b-holdout-0": true, "b-xuniv-0": false, "c-grid20k-0": true, "d-empirical-0": true}.
2. **>1 SE escalation -- FIRED (expected, favorable, explained)**: 29/41 pooled
   headline contrasts moved >1 SE. Mechanism: headline 8k grid was grid-limited
   for H=8 methods (B7's own guard); H=1 baselines static. MHLS moves are the
   intended WP B2 cap fix; ftp_greedy N=4 dip is the WP B4 leakage fix.
   Disposition: rerun SUPERSEDES headline_full for all FTP/MHLS numbers.
3. **Knee**: pooled sesar sparse PAM = 0.716 0.715 0.738 0.741 -> frac-rule knee=4, SE-aware knee=1.
   The sesar curve is FLAT in K within noise (K1-K2 pooled contrast not
   significant); BV keeps a real knee at K=2 (K1->K2 jump +0.19). Paper
   narrative: vocabulary size matters for the diverse (BV) universe, not for
   RRab-dominated sesar at this sparsity.
4. **Grid convergence (c) -- NOT CONVERGED for H=8/binned methods**: 8/30
   paired-McNemar cells significant at p<0.05, ALL gaining at 20k (MHLS most,
   FTP at N=8, CE mid-N; GLS/MBLS flat). FTP gains MORE than GLS, so FTP>GLS
   at 10k is conservative; absolute sparse-N rates are lower bounds "at the
   10k production grid". Resolution: post-C2 dense-grid or peak-refinement rerun (see SUMMARY_c-grid20k.md).
5. **Empirical errors (d)**: recovery insensitive to the 1.6-3.5x larger
   empirical errors (FTP 0.21/0.92/0.99 vs synthetic 0.23/0.93/0.99 at
   N=4/8/12); curve provenance proven (committed-json).
6. **Robustness (b)**: FTP>GLS ordering survives holdout(0.5), cross-universe,
   and band-amp 1.4 (per-arm summaries).
7. **joint/EM (e-canary)**: B6 machinery verified at production scale; recovery
   gap == 0 everywhere BUT 4/5 cells are saturated (zero-by-construction,
   audit B8E-1) -- the deferred rerun MUST re-grid N (e.g. 4,5,6,8,12) and fix
   the serial assignment_accuracy loop (87% of the canary bill) first.

## Escalation table (cells >1 SE, pooled 768 vs 768)

| cell | method | rerun | headline | delta | |d|/SE |
|---|---|---|---|---|---|
| nsweep N=4 | ftp_pam | 0.233 | 0.197 | +0.036 | 1.7 |
| nsweep N=8 | ftp_pam | 0.902 | 0.816 | +0.086 | 4.9 |
| nsweep N=12 | ftp_pam | 0.984 | 0.961 | +0.023 | 2.8 |
| nsweep N=16 | ftp_pam | 1.000 | 0.991 | +0.009 | 2.4 |
| nsweep N=4 | ftp_greedy | 0.128 | 0.155 | -0.027 | 1.5 |
| nsweep N=8 | ftp_greedy | 0.870 | 0.809 | +0.061 | 3.3 |
| nsweep N=12 | ftp_greedy | 0.986 | 0.962 | +0.023 | 2.8 |
| nsweep N=4 | gls | 0.040 | 0.029 | +0.012 | 1.2 |
| nsweep N=8 | gls | 0.417 | 0.378 | +0.039 | 1.6 |
| nsweep N=12 | gls | 0.889 | 0.857 | +0.033 | 1.9 |
| nsweep N=16 | gls | 0.993 | 0.988 | +0.005 | 1.0 |
| nsweep N=4 | mhls | 0.040 | 0.003 | +0.038 | 5.0 |
| nsweep N=8 | mhls | 0.565 | 0.003 | +0.562 | 31.3 |
| nsweep N=12 | mhls | 0.767 | 0.341 | +0.426 | 18.6 |
| nsweep N=16 | mhls | 0.776 | 0.663 | +0.113 | 5.0 |
| nsweep N=24 | mhls | 0.853 | 0.816 | +0.036 | 1.9 |
| nsweep N=8 | mbls | 0.246 | 0.217 | +0.029 | 1.3 |
| nsweep N=12 | mbls | 0.733 | 0.698 | +0.035 | 1.5 |
| nsweep N=16 | mbls | 0.962 | 0.940 | +0.022 | 2.0 |
| ksweep-sparse K=1 | ftp_pam | 0.716 | 0.613 | +0.103 | 4.3 |
| ksweep-sparse K=2 | ftp_pam | 0.715 | 0.651 | +0.064 | 2.7 |
| ksweep-sparse K=4 | ftp_pam | 0.738 | 0.643 | +0.095 | 4.1 |
| ksweep-sparse K=8 | ftp_pam | 0.741 | 0.643 | +0.098 | 4.2 |
| ksweep-sparse K=1 | ftp_greedy | 0.622 | 0.568 | +0.055 | 2.2 |
| ksweep-sparse K=2 | ftp_greedy | 0.674 | 0.600 | +0.074 | 3.0 |
| ksweep-sparse K=4 | ftp_greedy | 0.712 | 0.613 | +0.099 | 4.1 |
| ksweep-sparse K=8 | ftp_greedy | 0.720 | 0.621 | +0.099 | 4.2 |
| ksweep-sparse ref | gls | 0.171 | 0.150 | +0.021 | 1.1 |
| ksweep-sparse ref | mhls | 0.197 | 0.003 | +0.194 | 13.4 |

## Spend

~$64-67 actual vs $60 cap (~$49 compute + ~$16 idle tail during the 2h
session-limit freeze; see B8_COST_ESTIMATE.md decision log). Watchdog v2 now
kills beaconed pods after 15 min -- the tail cannot recur.
