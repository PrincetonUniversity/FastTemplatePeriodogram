# SAFE CPU stack — before/after bench (2026-07-18, rev 2 post-FILTER-DIP-1 fix)

Median of 5, nfreq=8000, single-thread pinned BLAS, Apple M5 P-core.
`before` = pre-stack commit 1529c33 (files-only checkout — its JSON's
`rev` field records HEAD at run time, not the measured code; from rev 2
on, every JSON also records `pkg_sha`, a content hash of the measured
package). `after_fix` = full stack items 1-6 WITH the FILTER-DIP-1
correctness fix (filter restricted to dip-free rows, 4x margin, 2x
densify headroom — see PARITY_REPORT.md).
Raw: output/bench_before.json, bench_after.json (pre-fix, superseded),
bench_after_fix.json.

| row | before (s) | after_fix (s) | net speedup |
|---|---|---|---|
| sb_scan_h2 | 0.044 | 0.036 | 1.22x |
| sb_scan_h4 | 0.096 | 0.068 | 1.41x |
| sb_scan_h8 | 0.253 | 0.155 | 1.63x |
| mb_batched_h2 (K=1) | 0.090 | 0.063 | 1.43x |
| mb_batched_h4 (K=1) | 0.194 | 0.128 | 1.52x |
| mb_batched_h8 (K=1) | 0.489 | 0.317 | 1.54x |
| mb_catalog_h2 (K=4) | (no before row) | 0.175 | -- |
| **mb_catalog_h4 (K=4)** | 1.031 | **0.358** | **2.88x** |
| **mb_catalog_h8 (K=4)** | 2.816 | **0.797** | **3.53x** |
| mb_direct_h4 (fast=False) | 0.167 | 0.080 | 2.09x |
| mb_sparse2_h4 (deferral-heavy) | 0.109 | 0.081 | 1.35x |

The pre-fix stack measured 3.43x/4.10x on the catalog rows; the
correctness fix returns 8-19% of that (dip-carrying rows now polish all
candidates; 4x margin keeps more candidates on dip-free rows; densified
rescans sized 2x). Correct beats fast.

Real-grid anchors (median of 3, single LC, K=4 catalog, post-fix) — the
B5 per-candidate inputs; ~9% superlinear vs scaling the 8k rows, so B5
uses THESE, not linear extrapolation:
- RRab H4, nfreq=48,000: **2.341 s/LC**
- RRc H2, nfreq=45,120: **1.130 s/LC**

Per-unit CPU floors (K=4-amortized, post-fix, from the 8k rows):
H8 24.9 us/LC/f/T, H4 11.2 us, H2 5.5 us.
