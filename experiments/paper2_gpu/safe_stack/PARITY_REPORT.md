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
