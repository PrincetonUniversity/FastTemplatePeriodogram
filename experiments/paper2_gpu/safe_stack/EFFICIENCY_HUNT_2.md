# Post-stack efficiency hunt #2 (2026-07-19) — COMPLETE, NOT YET IMPLEMENTED

Fresh 5-lens hunt on the POST-SAFE-stack code, every proposal
skeptic-vetted with its own thread-pinned interleaved A/B probe. Run
wf_8afd047f-74b: launched under Fable, 12 vet/finder agents died on a
usage-credit limit, **resumed under Opus and completed 18/19 agents**
(1 schema-retry failure, non-substantive). Per the "recover and persist,
don't go further" instruction NONE of this was applied to the engine
code — this is the implementation menu for a future session.

Result: **11 SOLID verdicts → 6 distinct mechanisms** (several found by
two lenses independently), 1 PLAUSIBLE, 2 REJECT. All gains are e2e on
the production shape (multiband floating_offsets, K=4 griz, N=8/band,
scan, powers_only) at nfreq 8k–40k, independently reproduced by a
skeptic's own probe (not the finder's). Probes: scratchpad/hunt2/.

## The 6 distinct wins (dedup'd, ranked by safety × gain)

### A. Gate the dip-candidate machinery to r < _SCAN_EXACT_RTOL (0.15)
- **Seam:** `core.py::_scan_pass` step 3b — add `ismin &= (mm_min <
  _SCAN_EXACT_RTOL * mm_max)[:, None]`. (Found by BOTH the `scanpath`
  and `python` lenses — same one-line change.)
- **Why safe:** on well-conditioned griz, `_SCAN_DIP_RTOL=0.5` flags the
  NATURAL oscillation minima of the degree-4H |MM| polynomial on 65–92%
  of rows (~2.3 spurious dip seeds/row), each Newton-refined and
  re-polished — but the code's own `_SCAN_DIP_RTOL` comment already
  states the (0.15, 0.5)-band dips are "a deliberate conservative hedge,
  not load-bearing on any tested valid input." Gating to r<0.15 keeps
  every load-bearing dip (the triage zone) and drops the rest.
- **Measured (skeptic, isolated one-block patch):** +19–25% H4, +16–18%
  H8, +15–23% H2 e2e; +24.6% on the adversarial sparse2 deep-dip
  fixture. **Biggest single win.** Safety class: needs-gate (must
  re-golden + confirm no argmax/deferral flips; expected ulp-or-better
  since dropped candidates never win).

### B. Rework `_scan_pass` inner loops (fused triple-Horner + skip-gather)
- **Seam:** `core.py::_scan_pass` both Newton loops; ref impl
  hunt2/opt_scan.py. Fuse p/p'/p''/2 into one Horner recurrence per
  family (6 loops→2), skip the fancy-index gather while the active set
  is full, slice-compare instead of np.roll, reuse the row max.
- **Measured (skeptic, median-of-9 interleaved):** +14–17% H4, +12–14%
  H8, +8–11% H2 e2e. Safety class: **ulp-equivalent** — the p-value is
  bit-identical to stock, mm_min/mm_max and all filter/gate/deferral
  decisions bit-preserved; only the fused Y1/Y2 perturb theta at ~1e-13
  (maxrel ≤ 2.2e-15, argmax identical on all 12 fixtures incl. deep-dip).

### C. Fused diagonal-segment-sum assembly + hoist kernels across the catalog
- **Seam:** `multiband.py::_combine_stacked_shared_amp` /
  `core.py::batched_YM_MM_from_sums`; ref impl
  revalidation/hunt2_probe_e2e.py::fast_combine. (Found by `assembly`
  and `python` lenses; the SOLID item from the partial run, re-confirmed.)
- **Mechanism:** each flattened (r,c) covariance entry maps to one MM
  diagonal → replace 3·(2H−1) np.trace + stacks + copies with hoisted
  template-independent kernels (Pcc, Pcs, once per chunk per band, reused
  across K templates) + one `np.add.reduceat` segment sum per matrix;
  SS never formed (`SS_diags = conj(CC_diags)[:, ::-1]`, exact).
  reduceat's fixed within-segment order PRESERVES the pinned chunk-size
  bitwise invariance.
- **Measured (skeptic, median-of-18):** +8–13% e2e (H8 best). Safety
  class: **ulp-equivalent** (reassociates diagonal/band sums; e2e maxRel
  ≤1.4e-15, argmax identical, chunk 64-vs-4096 np.array_equal True).
  Bit-identical fallback (UU/VV hoist only, no reassociation) = ~2–3% if
  the ulp change is refused.

### D. fast=False exact direct summations for the sparse production shape
- **Seam:** `multiband.py::_prepare_band_transforms` / `_run_batched` —
  an N-per-band-adaptive switch (conservative threshold N_band ≤ ~16), or
  just flip the production harness to fast=False. (Found by BOTH `profile`
  and `sums` lenses.)
- **Why:** at N=8/band the adjoint-NFFT cost (two ~128k Bluestein iFFTs
  per band, no plan cache, paid fresh per LC) is N-independent and
  dominates the fused-trig direct stacked sums (cost ∝ N, hoisted across
  K); fast=False is the EXACT reference path (removes the 5e-10–2e-8 NFFT
  approximation — strictly more accurate).
- **Measured (skeptic, median-of-7):** +11–22% at 8k, +21–27% at 40k
  (production). Safety class: **needs-gate** — powers move 5e-10 (8k) to
  2e-8 (40k) vs the current NFFT baseline, so a knife-edge catalog
  argmax/deferral row could flip: re-golden + one B-series recovery
  spot-check. Crossover N unmeasured → keep the adaptive threshold.
- **SUBSTITUTES with E** (below): if D is adopted, the NFFT path is off
  the production shape and E's padding is moot there (still helps any
  retained-NFFT / large-N use).

### E. Pad the adjoint-NFFT to 5-smooth lengths (next_fast_len)
- **Seam:** `summations.py::_nfft_grid_coefficients` — inflate
  nf_nfft_u/nf_nfft_w so sigma·N hits a 5-smooth size, killing pocketfft's
  Bluestein path. (Found by `scanpath` and `sums` lenses.)
- **Measured (skeptic):** +7–13% e2e where the NFFT path is retained
  (K=4 catalog; ~+20pts single-template K=1). Safety class: needs-gate
  (changes NFFT grid → tiny power move). **Only relevant if D is NOT
  adopted** (or for large-N inputs).

### F. Drop the scan grid floor 128→64 angles for H2 (RRc shape only)
- **Seam:** `core.py::_SCAN_MIN_ANGLES` 128→64, i.e. `floor = max(64,
  32H)` — leaves H≥4 (128 = 32·4) untouched, only H≤3 changes.
- **Measured (skeptic):** +12% on the H2 RRc catalog alone. Safety class:
  needs-gate — 64 angles must still bracket every circular max at H2
  (32·2=64 satisfies the ≥32H rule, but the C2 bracketing proof and the
  narrow-peak fixtures must be re-run at the reduced floor before adoption).

## PLAUSIBLE (1) — do not adopt without more measurement
- **Z-scatter diagonal sums in batched_YM_MM_from_sums, H8-only:** safety
  claim correct (ulp, chunk-invariant, bit-identical at H2) but magnitude
  ~2× overstated — skeptic measured ~2.7–3.6% e2e at H8 vs the claimed
  6.5% (a median-of-5 lucky draw). Subsumed by C anyway.

## REJECT (2) — recorded so they are not re-proposed
- **Extend the Bernstein filter to DIP-CARRYING rows (science-gated
  variant):** census "23–43% droppable" inflated 5–8× (counted drops
  before the K.18 constraint); rejected.
- **chunk_size retune to 8192:** slower than 4096 at nfreq=32k (the 8k
  "win" was a single-chunk boundary artifact).

## ⚠ SPECIAL-HANDLING win — Bernstein filter extension to r ≥ 0.15
Verdict was SOLID and the skeptic measured it **bit-identical
(max|dP|=0.0)** with the FILTER-DIP-1 failure mode NOT recurring (the
row-wise `max(4, 1/r²)` margin is the correct curvature bound; 63–83% of
rows sit in the newly-active (0.15, 0.5) band), gaining +10 points at
H4/H8. **BUT this re-touches the exact code path that carried the
critical FILTER-DIP-1 bug.** Do NOT adopt on the single vet — it must go
through the full FILTER-DIP-1 adversarial ensemble
(safe_stack/revalidation/probe_hunt*, probe_dense_attr, the skeptic
counterexamples) before it is allowed near production. High reward,
highest scrutiny.

## Recommended adoption order (future session)
Independent + ulp/bit-safe first, cheapest to gate:
1. **A** (dip-gate, ~20%) + **B** (scan rework, ~13%) + **C** (assembly
   fusion, ~10%) — all touch scan/assembly, largely stack, mostly
   ulp-equivalent. Re-golden once for all three. Est. combined ~1.4–1.5×.
2. **D** (fast=False, ~20% more) — separate gate (recovery spot-check),
   makes E redundant on the production shape.
3. **F** (H2 floor) — needs the bracketing-proof re-verification.
4. **Bernstein r≥0.15 extension** — only after the full FILTER-DIP-1
   ensemble passes.
Combined realistic ceiling (A+B+C+D, NOT simply multiplicative;
substitutions accounted): finder/skeptic combos reached ~1.55× H8,
~1.32× H4, ~1.36× H2 — i.e. roughly HALVING the per-candidate CPU cost
again on top of the SAFE stack. Each adoption re-runs the parity harness,
the 6 SAFE regression tests, and the full suite; commit as
ulp-equivalent or needs-gate SAFE-adjacent items with parity-report
entries.
