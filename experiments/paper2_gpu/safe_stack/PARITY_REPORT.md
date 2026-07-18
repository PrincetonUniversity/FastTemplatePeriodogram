# SAFE CPU stack — per-item bit-parity report (2026-07-18)

Audit §2 items 1–6 (AUDIT_2026-07-18.md), implemented on `paper2-fast-multiband`.
Parity harness: `fixtures.py` (7 deterministic fixtures: griz H8/H4, RRc H2,
sparse 2-band ZTF g/r, single-band H8/H2, phase-clumped dip H8; nfreq=2000)
+ `golden.py` (captures batched scan powers on NFFT and direct paths, catalog
max-over-vocab, and an eigvals reference subsample). "base" = pre-stack
commit 1529c33. All runs: Apple M5, repo `.venv` (numpy 2.0.2/Accelerate).

| # | Item | Commit | Parity class | Measured |
|---|------|--------|--------------|----------|
| 1 | Newton polish early-exit (`_SCAN_NEWTON_XTOL=1e-12` rad, active-set compaction; WP C2 floor met as "6 steps or convergence") | c6eb799 | numerically different, equivalent | max abs dP **1.2e-15** vs base across all fixtures; 0 argmax moves |
| 2 | Bernstein candidate filter (vetted k18 arm; dip candidates exempt; positive-seed protection under K.18) | 1da80d2 | numerically different, equivalent | **no additional change** over item 1 on any fixture (diff list identical); drops **42%** of polish candidates on griz H8 |
| 3 | Bernstein-gated deep-dip triage (gate at base grid → one densified rescan with gate recomputed at dense M → eigvals only if still failing / >2^20 angles; max-merge) | a6d523d | numerically different, equivalent | cumulative max rel dP **2.4e-15** vs base; 0 argmax moves; eigvals calls on phase-clumped H8 fixture **192 → 0** (124 gate-pass + 68 densified ≤1024 angles); both monkeypatch mechanism pins (`_SCAN_EXACT_RTOL=0`, `_SCAN_DIP_RTOL=0`) still load-bearing |
| 6 | Powers-only tail skip + deferral reads the scan's own circle extrema | 617413a | **bit-identical** | max abs dP = 0.0 exactly (all fixtures); deferral decisions bit-identical (same grid, same comparison) |
| 4 | Fused trig recurrence in `_direct_stacked_sums_chunk` (one exp on fundamental, `Zcur *= Z` per harmonic, fused matvecs; freq-cumprod variant REJECTED per hunt, ~1e-8 drift) | e84da62 | numerically different, equivalent (direct-sums paths only) | NFFT paths **bit-identical (0.0)**; direct paths max abs 4.6e-11 / max rel 2.2e-10 on dip-heavy fixtures (conditioning-amplified ulp); batched-vs-per-frequency-reference on the adversarial sparse fixture: **1.79e-9 new vs 1.93e-9 pre — no regression**; deferred rows still bit-match the reference (per-frequency ref sums untouched); 0 argmax moves |
| 5 | Sums hoist across K-vocab catalogs (multiband `multiband_power_spectra_batched` chunk-outer/set-inner; modeler `autopower` sums-per-distinct-H, verified bitwise vs internal path) | 85f3993 | **bit-identical** | max abs dP = 0.0 exactly (all fixtures incl. catalog captures) |

Notes
- Ordering of implementation was 1→2→3→6→4→5; parity rows above give each
  item's incremental effect at its own commit.
- Item 4's errstate guard covers a numpy-on-Accelerate quirk: complex matmul
  leaves spurious FP status flags (divide/overflow/invalid) on finite
  inputs/outputs (minimal reproduction verified); genuine non-finite data
  still raises downstream at the scan's coefficient check.
- Full-suite and before/after bench results: see BENCH.md (same directory)
  and the VERIFICATION.md addendum.

---

## Rev 2 — FILTER-DIP-1 (found by adversarial verification, fixed same day)

The 33-agent verification workflow (5 lenses x 2-refuter panels) run on
the stack CONFIRMED a critical defect in item 2 as shipped: the
Bernstein improvement bound assumes the smooth degree-2H trig scale, but
P = Re(YM^2/MM) is RATIONAL — near a circle dip of |MM| its curvature
inflates beyond (2H)^2, so the unrestricted filter could drop a
candidate whose polish wins the argmax. Reproduced independently by two
skeptics against brute-force dense-circle ground truth (exemplar: seed
6002 clustered cadence, H=2, r=0.020 — shipped 0.131661 vs true
0.131793, silent 1.3e-4 deficit; worst observed 2.6e-3; incidence
~1e-4 of rows, phase-clustered/rank-deficient cadences only; the
7-fixture golden harness lacked this cadence class). A second (major)
confirmed finding: the stage-2 densified rescan was sized at the
Bernstein equality, so accepted rows sat exactly at the recomputed gate
in the same regime.

**Fix** (uncommitted-at-verification, now in the same branch):
1. Filter restricted to dip-free rows (min|MM| >= _SCAN_DIP_RTOL
   max|MM| — exactly the rows with no dip candidates, where the smooth
   Bernstein scale is the C2-validated regime); every candidate of a
   dip-carrying row is polished. Dense rescan rows are all deep, so the
   filter self-disables there.
2. Margin raised 2x -> 4x on the eligible rows (headroom for the
   residual <= 1/r^2 <= 4 curvature inflation at r ~ 0.5).
3. Stage-2 M_need sized with 2x headroom.

**Revalidation** (the workflow's own adversarial probes, rerun on the
fix): probe_hunt 450k rows — 0 findings; probe_hunt2 234k rows (168,674
deep-band) — 0 misses; probe_dense_attr — all previously-regressed rows
now match eigvals (residuals ~1e-12); skeptic fresh-seed counterexamples
(20002/22002/8002) — deficits <= 5.7e-14, per-case max 1.1e-11 (inside
the corner-regime evaluation-noise bar). Suite: 209 targeted + full
suite rerun. Bench deltas: see BENCH.md rev 2 (net 2.88x/3.53x catalog).

Also actioned from the same verification run: bench provenance
(pkg_sha content hash recorded per run; mb_catalog_h2 row added to
bench.py), B5 real-grid anchors re-measured post-fix (superlinearity
corrected, B5_RESULT.md updated), VERIFICATION.md speedup ranges
corrected, and five coverage regression tests added
(tests/test_safe_stack_regressions.py) for: trig-recurrence accuracy at
long grids, the batched deferral contract, stage-2 recheck + angle-cap
branches, mixed-H modeler hoist, multi-chunk hoist staleness.
