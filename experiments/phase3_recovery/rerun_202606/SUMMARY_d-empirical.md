# Arm (d) empirical-error revalidation (1 seed, ALL stages)

err_model: `empirical_ztf(binned-median;committed-json(ztf_error_curve.json))` (provenance proven in-log).

**empirical ZTF errors** (N-sweep @K=4, 256 src/seed):

| method | N=4 | N=8 | N=12 | N=16 | N=24 | N=40 |
|---|---|---|---|---|---|---|
| ftp_pam | 0.215 | 0.918 | 0.988 | 1.000 | 1.000 | 1.000 |
| ftp_greedy | 0.152 | 0.918 | 0.988 | 1.000 | 1.000 | 1.000 |
| gls | 0.043 | 0.379 | 0.859 | 0.992 | 1.000 | 1.000 |
| mhls | 0.043 | 0.516 | 0.758 | 0.777 | 0.859 | 0.852 |
| mbls | 0.027 | 0.234 | 0.684 | 0.957 | 1.000 | 1.000 |
| ce | 0.004 | 0.051 | 0.480 | 0.844 | 0.980 | 1.000 |

CAVEAT (audit B8F-6): --fixed-k was not pinned; the un-pinned knee derived K=4, so the N-sweep ran at K=4 vs K=2 elsewhere. The sesar sparse curve is flat in K (0.715 0.699 0.742 0.727), so the contrast with the synthetic arms is unaffected at the quoted precision.

Cost panel: FTP rec 0.344 == oracle rec 0.344, in-harness speedup 2.39x.
