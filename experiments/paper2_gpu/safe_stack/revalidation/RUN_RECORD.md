# FILTER-DIP-1 revalidation — committed artifact record

The 2026-07-18 both-sessions re-audit confirmed (major) that the
FILTER-DIP-1 fix's quantitative revalidation basis existed only in a
session scratchpad — prose-only in PARITY_REPORT rev 2 / VERIFICATION
errata / commit f7e8bbe. This directory makes it reproducible:

- probe_hunt.py — 450k-row adversarial parity hunt (current scan vs
  pre-change scan `core_base.py` vs per-row eigvals reference) over
  clump/sparse/eclipse cadence families, H in {2,4,8,12}, both
  positive_amplitude arms. PASS criterion: 0 findings (no row where
  current < pre-change - 1e-12, no miss vs eigvals where pre-change
  agreed, no K.18 feasibility regressions).
- probe_hunt2.py — 234k-row deep-band-weighted variant (168,674 rows
  with r <= gate0). PASS: 0 misses.
- probe_dense_attr.py — re-checks the specific rows the pre-fix code
  regressed on (dense-band FILTER@DENSE class). PASS: all match eigvals
  to ~1e-12.
- skeptic1_more_cases.py / skeptic1_math_repro.py — the verification
  panel's independent counterexample reproducers (seeds 6002 exemplar +
  fresh 20002/22002/8002).
- core_base.py — bit-verified copy of pre-stack (1529c33) core.py used
  as the pre-change reference implementation.

First run (fix working tree, later committed as f7e8bbe; Apple M5,
single-thread BLAS): probe_hunt "rows scanned: 450000 ... findings: 0";
probe_hunt2 "counts: {'rows': 234000, 'deep': 168674, ...} misses: 0";
probe_dense_attr: previously-regressed rows all equal eigvals (e.g.
0.608054473984 across eig/cur/newdense/olddense columns); skeptic
counterexamples: deficits <= 5.7e-14 typical, <= 1.1e-11 corner-regime.

The committed *.out.txt files in this directory are a fresh rerun at or
after fc73099 (same machine, single-thread pinned); pass criteria as
above. If any log is missing, the rerun was interrupted — rerun the
script; each is standalone (imports the repo package + core_base.py
from this directory).
