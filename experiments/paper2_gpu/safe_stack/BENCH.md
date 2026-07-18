# SAFE CPU stack — before/after bench (2026-07-18)

Median of 5, nfreq=8000, single-thread pinned BLAS, Apple M5 P-core.
`before` = pre-stack commit 1529c33; `after` = full stack (items 1-6).
Raw: output/bench_before.json, output/bench_after.json (bench.py).

| row | before (s) | after (s) | speedup |
|---|---|---|---|
| sb_scan_h2 | 0.044 | 0.033 | 1.33x |
| sb_scan_h4 | 0.096 | 0.054 | 1.78x |
| sb_scan_h8 | 0.253 | 0.122 | 2.07x |
| mb_batched_h2 (K=1) | 0.090 | 0.057 | 1.58x |
| mb_batched_h4 (K=1) | 0.194 | 0.115 | 1.69x |
| mb_batched_h8 (K=1) | 0.489 | 0.298 | 1.64x |
| **mb_catalog_h4 (K=4 vocab)** | 1.031 | **0.301** | **3.43x** |
| **mb_catalog_h8 (K=4 vocab)** | 2.816 | **0.686** | **4.10x** |
| mb_catalog_h2 (K=4 vocab) | (not in before-run) | 0.162 | -- |
| mb_direct_h4 (fast=False sums) | 0.167 | 0.070 | 2.39x |
| mb_sparse2_h4 (deferral-heavy) | 0.109 | 0.074 | 1.47x |

Takeaways
- The production detection shape (K=4 vocab catalog, floating_offsets)
  gets **3.4x @H4 / 4.1x @H8** -- the audit's predicted compound
  ~2-2.5x is exceeded because the sums hoist (item 5) multiplies with
  the polish-side wins on catalogs.
- New per-unit CPU costs (K=4-amortized, single M5 P-core):
  H8 21.4 us/LC/freq/template, H4 9.4 us, H2 5.1 us.
  The audit's "best single-thread CPU floor ~78-81 us @H8" is now
  ~21 us catalog-amortized (~37 us single-template).
- GPU implication (surface, don't decide): the A100 kernel's measured
  0.554 us @H8/K=4 is now ~39x the M5 core (was ~135x), i.e. ~0.6x of
  a hypothetical 64 M5-core node. M5 P-cores are faster than cloud
  vCPUs, so the GPU is still competitive per dollar, but the SAFE stack
  materially strengthens the playbook's CPU-only Paper-2 decision.
- The sparse deferral-heavy row improves least (1.47x): its cost is
  dominated by per-frequency reference recomputation at the (unchanged)
  3e-3 deferral threshold, not by the scan.
