# Verification records (C-track)

Independent adversarial verification records for the high-care solver work packages
(EXECUTION_PLAN.md Track C). Per John's standing delegation (2026-06-12), a recorded
adversarial multi-agent verification with zero confirmed critical/major defects stands
in lieu of human review for the C-track sign-off gates.

## WP C1 — batched coefficient assembly (`c0d3cb2`) — VERIFIED 2026-06-12

**Scope**: commit `c0d3cb2` (`method='batched'` stacked assembly behind the unchanged
`'eigvals'` default). **Protocol**: the WP C3 verifier protocol applied early — 64 agents:
5 line-level reviewers (independent re-derivation of the YM/MM broadcast math and the
stationarity convolution, NFFT indexing/code-motion audit, whole-diff safety, test-adequacy
with mutation spot-checks), 4 empirical probes writing their own scripts with fresh seeds
(never reading the committed tests), a 3-lens refutation panel per finding (re-derivation /
reproduction / code-context), a completeness critic, and 3 critic-directed gap probes;
plus a final single-agent verification of the hardening fixes below.

### Core results (all pass)

- `_nfft_grid_coefficients` refactor is byte-identical code motion; all NFFT defaults,
  grid-sizing formulae, and phase corrections preserved.
- Fancy-index extraction (`idx = outer(freq_index + dnf, arange(1, 2H+1))`) is **bitwise
  identical** to the per-frequency loop, including across chunk boundaries and ragged
  final chunks.
- `batched_YM_MM_from_sums` re-derived independently: trace-offset semantics, broadcast
  orientations, and conjugation conventions all match the reference; YM and AC bitwise,
  MM to ≤1.6e-16 relative.
- `batched_stationarity_coefs` exactly equals `2·MM·YM′ − MM′·YM` minus its analytically-zero
  top coefficient (integer-exact test: bitwise; Gaussian draws: ≤9.1e-16 relative, two
  independent reference implementations).
- Default (`eigvals`) path **bitwise identical** pre- vs post-commit (powers AND params),
  24 configurations.
- Fresh-seed equivalence sweep (250 configs, H∈{1..10} × 5 seeds × N∈{8..300}, heteroscedastic
  errors): max |ΔP| = 7.6e-14, 0/250 argmax mismatches.
- Production scale: 99,001-point grid (25 chunks, ragged tail) max |ΔP| ≤ 8.9e-16, identical
  argmax; 16,384-point grids at H∈{8,10} ≤ 1.1e-15; gauge-aware fitted-curve agreement
  ≤ 1.0e-13. Informational speedup ~2.1× (H=3, 99k grid) before WP C2.
- Adversarial inputs (constant y, N=3, weight ratios 1e8/1e12, rank-deficient N∈{8,12} with
  H∈{8,10}, spiky templates, duplicate timestamps, NaN): equivalent or exception-parity.
- Full suite at gate: 361 passed / 1 skipped / 2 xfailed / 0 failed (matches commit claim);
  21 new tests confirmed non-vacuous (the power gate is enforced against the true default path).

### Confirmed findings → all fixed in the follow-up hardening commit

1. **[major, test gap]** No test compared `best_fit_params` between methods; the NFFT-batched
   C/S → AC → θ₃ reconstruction was value-checked nowhere. → Added params-equivalence tests
   (H∈{2,3,5,8}), an H=1 gauge-invariant fitted-curve test, and a bitwise field-level test of
   `fast_summations_batched` vs per-frequency `fast_summations` (all 7 fields incl. C/S).
   Mutation-checked: an injected `conj(AC)` bug invisible to the power gate (4.4e-16) trips
   the params gate at 0.83.
2. **[major]** `len(summations) ≠ len(freqs)` silently truncated the batched output at a
   chunk boundary (output length depended on `chunk_size`). → Now raises `ValueError`
   (stricter than the eigvals quirk of silently returning `len(summations)` powers; documented).
3. **[minor]** Negative `chunk_size` silently returned an empty periodogram. → `chunk_size`
   now validated (positive integer; bool rejected) at both entry points, eagerly.
4. **[minor]** "Chunking bitwise-invariant" was false for chunk lengths ≳1024: `np.trace` on
   layout-dependent (non-contiguous) stacks reordered the diagonal-sum reduction by 1 ulp.
   → CC/CS/SS stacks are materialized C-contiguous before `np.trace`; bitwise invariance now
   holds for all chunk sizes and is pinned by a regression test in the drift regime (nf=1999,
   chunks {1, 256, 1365, 1999, 4096}; pre-fix drift 1.25e-16 reproduced, post-fix bitwise).
5. **[minor]** Double-trim asymmetry: the precomputed `stationarity=` path went through
   `trim_zero_leading_coef` a second time and could (in a contrived |c| ≤ 1e-9·scale corner,
   0/240 realistic frequencies) drop the *genuine* degree-(6H−2) coefficient. → Trim skipped
   for precomputed polynomials (they arrive pre-trimmed); docstrings updated.

### Documented benign divergences (no fix; pinned or recorded)

- **H=1 parameter gauge**: at exactly-tied φ/−φ maxima the methods can return
  (a,b,sgn) ↔ (−a,−b,−sgn) — identical fitted curve (≤4e-15) and power (≤3.3e-16). A
  representation tie-break, not a model change; pinned via the curve-equivalence test.
- **Empty `freqs` (fast=False)**: eigvals raises (pre-existing quirk), batched returns an
  empty periodogram — the saner contract; pinned by test.
- Refuted-finding classes (9 refuted by the panel) and full probe transcripts live in the
  session workflow record (`wf_fbb9b727-3f6`).

### Sign-off

Zero confirmed critical/major defects in the math or the default path; the two majors were
a test blind spot and an out-of-contract input handler, both closed by the hardening commit.
Post-fix verification (single adversarial agent, checks (a)–(f) incl. default-path diff audit,
call-site sweep, polyroots degenerate-input probe): no defect. Full suite after hardening:
373 passed / 1 skipped / 2 xfailed / 0 failed.

**WP C1 signed off 2026-06-12 (adversarial verification in lieu of human review, per
delegation). C2 is unblocked.**

## WP C2 — scan+polish maximizer (`a8fd646` + dip/exact-fallback hardening) — VERIFIED 2026-06-13

**Scope**: `method='scan'` (FFT circle-scan + Newton-polish maximizer over phase; default
`'eigvals'` unchanged), the in-session dip-candidate / exact-root-fallback hardening for narrow
peaks at deep `|MM|` dips, and the CRIT-1 / MB-2 / TA1 closeouts committed alongside.
**Protocol** (WP C3 verifier protocol applied as the C2 sign-off, per John's 2026-06-12
delegation): two adversarial workflows — (1) 9 independent agents = 4 line-level reviewers
(derivative-math re-derivation, dip/fallback logic + bookkeeping, multiband scan parity,
test-adequacy/mutation) + 5 fresh-seed empirical probes that wrote their own ≥2^18 brute-force
circle oracles and never read the WP tests (broad sparse/rank-deficient sweep, threshold-boundary
adversary, multiband all-modes, production-scale argmax stability, degenerate-weights) + a
completeness critic; (2) a bounded 3-lens refutation panel (re-derive / reproduce / code-context)
over every non-info finding + a fix-evaluation critic.

### Core results (all pass)

- **Single-band scan maximizer is correct.** Across ~180k+ independent evaluations (5 probes, own
  oracles), ZERO maximizer misses (scan < oracle true max − 1e-10) above the 0.15 exact-fallback
  threshold. The central safety claim — a sub-grid-width P spike *requires* min|MM| < ~0.08·max|MM|
  (Bernstein), so the 0.15 fallback is safe — is empirically confirmed and shown conservative:
  the largest conditioning hosting a real sub-grid spike was ~0.003 (H=4) / ~0.0013 (H=8), an
  order of magnitude inside the fallback zone; count of sub-grid spikes escaping the fallback = 0.
- **Derivatives correct.** `_scan_dP_d2P` (single source of truth for the Newton polish) and the
  `|MM|²`-minimum dip-refinement derivatives re-derived from scratch and checked vs
  Richardson-extrapolated finite differences (H∈{1,2,3,5,8}) and the complex analytic dG/dθ
  (Romberg, 1.6e-10); the multiband `W_k`-weighted accumulation verified over H×K grids.
- **Gates 1–5 pass**: gates 1–4 are asserted by test_scan_polish.py — (1) scan≡eigvals
  max|ΔP| ≤ ~1e-15 over H∈{1..10}×5 seeds×N∈{30,300}, identical argmax; (2) Issue-#33 fixtures
  on the scan path; (3) adversarial rank-deficient vs the 2^16 oracle; (4) weight-conditioning
  within 2× of eigvals. Gate 5's speedup half — **26.5× @H=2, 20.2× @H=8 (≥15× required),
  22.6× @H=10**, max|ΔP| ~1e-15 — is measured by
  `experiments/benchmarks/benchmark_scan_polish.py` (a one-off benchmark script, not collected
  by pytest; the test-file docstring says so). *[Amended 2026-07-04: the original wording
  attributed all five gates to test_scan_polish.py — recheck defect 2. The speedup numbers are
  unregressed one-off measurements.]*
- **Multiband: all four modes** match eigvals to ≤1.3e-15 (powers + a/b/c/sgn params).
  shared_phase scan independently verified to attain the true max of its objective
  F(θ)=Σ_k W_k Re(YM_k²/MM_k) via a brute-force oracle that never calls the eigvals G-polynomial
  root-finder (new `test_multiband_shared_phase_scan_vs_independent_F_oracle`).
- **Argmax stable** to ~4 orders of grid margin at nf=30000; chunk_size bitwise-invariant;
  fast=True/False both match their references.
- **Degenerate inputs**: NaN/inf/dy=0 fail loudly (ValueError, parity with eigvals LinAlgError);
  constant y / single obs / empty freqs handled (scan is the more robust path on empty freqs).
- Full suite at gate: 459 passed / 1 skipped / 2 xfailed / 0 failed.

### Confirmed findings → dispositions

1. **[claimed major → REFUTED] MB-1 / P1-2** "shared_phase fallback under-reports power by ~5e-3
   at deep `|MM|` dips". The 3-lens panel refuted it: the original probes' "true max" oracle used
   a *free-per-band-phase* (or differently-normalized) LS, which has strictly more DOF than the
   shared_phase model permits, so it spuriously "explained more variance". Against a correct
   oracle matching the shared_phase objective (one common phase, YY_combined normalization,
   unconstrained amplitudes), the eigvals reference is the exact global maximizer (matches a
   200k-point brute force to ~1e-11) and the scan matches it bit-for-bit. **No defect.**
2. **[minor] MB-2** the multiband-scan gate only compared scan-vs-eigvals → couldn't catch a
   deficit shared by both. → **FIXED**: added an independent brute-force-F-oracle gate for the
   shared_phase scan (3 seeds) that isolates the maximizer from the eigvals root-finder.
3. **[minor] TA1** the dip-candidate machinery in the (0.15, 0.5) band is never load-bearing on
   valid data (real sub-grid spikes only occur below ~0.003, inside the 0.15 exact-fallback zone);
   disabling it leaves the suite green. → **DOCUMENTED** in core.py as a deliberate conservative
   hedge (the exact-root fallback below 0.15 is the actual correctness net); the refinement path
   is exercised by the deep-dip fixtures and now also gated by the MB-2 oracle.
4. **[minor] CRIT-1** `scan_polish_from_coefs` honored an `n_angles` override below the
   max(128,32H) floor (internal helper only; no public exposure), which could undersample the
   circle. → **FIXED**: the override is clamped up to the floor (resolution may only increase);
   pinned by `test_scan_n_angles_floor_clamps_unsafe_override`.
5. **[minor] P5-4** empty freqs: eigvals crashes, scan returns empty — a benign reverse-asymmetry
   (scan is the more robust path), already pinned. No action.
6. **[minor → refuted] TA2** corner-test docstrings; refuted (rationale adequate). No action.

### Pre-verified OPTIONAL follow-up (NOT adopted in C2 — would change the contract)

The fix-evaluation critic proved that replacing the verbatim exact-root fallback with
`max(scan_own_best, fallback)` is **strictly safe** (both evaluate the true objective at real
phases ⇒ both are valid lower bounds ⇒ their max never overshoots; matched an independent refined
oracle to ~1e-13 across 50+ deep-dip fixtures). It would make scan ≥ eigvals everywhere and remove
the rare (~2% of fallback fixtures, deficit ≤~1e-2, never argmax-flipping) cases where the
degree-(8HK−2) G-polynomial root-finder is the weaker maximizer. **Not adopted** because it
changes the WP's defined scan≡eigvals equivalence contract (and C4's scan≡eigvals pinning);
recorded as a pre-verified enhancement for a deliberate future decision.

*[Amended 2026-07-04: the "~2%, ≤1e-2" characterization was REMEASURED during WP C3 (the
2026-07-04 recheck had advised retract-or-remeasure after failing to reproduce it). Outcome: the
underlying defect is real and recipe-conditional — ~2% incidence reproduces on moderate deep-dip
fixture families, but on every-band-deep recipes the incidence and magnitude are far larger (up
to 8.4e-2 power / 26.3% raw χ²). See WP C3 finding MB-DIP-1 below; the fix direction sketched
here (scan-first + max(scan, root path)) was independently validated by the C3 root-cause agent.
The C2 MB-1 refutation itself REMAINS CORRECT — the original MB-1 probes' oracle had excess
degrees of freedom exactly as the panel found; surfacing the real defect required raw-data
oracles on every-band-deep fixtures, which neither the MB-1 probes nor the recheck adjudicator's
recipe produced.]*

### Verification-process note

The first workflow's refutation phase partially STALLED (several "reproduce" agents ran unbounded
2^20-point oracle sweeps with no time budget; the runtime retried each 6× over ~18 min before
dropping it to null). The 9 review+probe reports and the completeness critic completed; the
refutation was re-run BOUNDED (≤2^16 grids, <30 s scripts) to closure — which is how the
plausible-but-wrong MB-1/P1-2 "major" findings were caught as oracle artifacts. Lesson for future
verification workflows: give empirical-probe/refutation agents explicit grid-size and wall-time
budgets.

### Sign-off

Zero confirmed critical/major defects in the scan maximizer (single-band or multiband); the two
"major" correctness findings were oracle artifacts refuted by the panel. Two minor findings (MB-2
test gap, CRIT-1 robustness) fixed; one (TA1) documented as intended conservative behavior. Full
suite after the closeout: 459 passed / 1 skipped / 2 xfailed / 0 failed.

**WP C2 signed off 2026-06-13 (adversarial verification in lieu of human review, per delegation).
C3 — the dedicated independent-verifier WP — is unblocked.**

## WP C3 — dedicated independent verification (`8dad841` + the C3 record commit) — 2026-07-04

**Scope**: the dedicated independent-verifier WP for the scan+polish maximizer. Base records:
the WP C2 sign-off (above) and `AUDIT_2026-07-04_C2_RECHECK.md`; this record extends both and
lands the recheck's four deferred fixes (`8dad841`). **Protocol**: 12-agent adversarial workflow
(`wf_bb84e7c1-b84`, ~1.23M subagent tokens): 8 wave-1 agents (2 single-band probes, 3
multiband-mode probes with mode-matched oracles, 1 root-cause agent, 1 skeptic tasked with
refuting the orchestrator's probes, 1 fix-verification agent) + completeness critic + 3
critic-directed follow-up probes. All probes bounded (≤2^16 circle grids / ≤2^13 raw-data phase
sweeps, <30 s scripts), fresh seeds, never reading the repo tests (fix-verification agent
excepted). **New vs C2**: RAW-DATA weighted-LS oracles — dense phase-shift sweep + closed-form
per-shift LS on the raw photometry, immune to the polynomial-formula `Re(YM²/MM)` conditioning —
built independently for the single band and each of the four multiband model classes, each
validated ≤1e-9 against the code on well-conditioned fixtures before use (achieved 8.9e-16 …
6.3e-15). **Criterion note**: the raw-data oracles gate at `oracle > code + 1e-9` (not the
brief's 1e-10) because the shared evaluation-noise floor of code-vs-raw-truth is ~3e-10; the
1e-10 criterion is enforced only in the polynomial-oracle modality (P1).

### Gate result: the scan+polish maximizer itself PASSES everywhere it runs

- **P1 (single band, polynomial oracle w/ dip refinement)**: 104 fixtures (4 template families ×
  H 1..10 × N {6..300} × hetero/weight-concentrated dy) — 6,272 full-grid scan-vs-eigvals
  comparisons, 1,093 refined-oracle points: **0 misses** at the 1e-10 bar (worst oracle−scan
  2.84e-11, at conditioning 9.3e-7, where eigvals sits 3.3e-11 above scan too — shared sums
  noise), **0 argmax mismatches**. Oracle positive control: dip refinement improved the raw 2^16
  grid by up to 8.4e-5 at 106 points and the scan matched the refined values.
- **P2 (single band, raw-data oracle, extreme conditioning)**: 540 rows (H {8,10,12} × N {5,6,8}
  × clustered/duplicate-times/weight-ratio-to-1e12): **no maximizer deficit** — every flagged row
  (5/540, worst 2.16e-8, all at circle conditioning ≤1e-7) adjudicated with 50-digit mpmath: the
  returned phases are within ~1e-11 of the true optimum; the discrepancy is *value* evaluation
  noise through the ill-conditioned sums (finding 4 below).
- **M1 (floating_offsets — the production headline mode)**: 746 rows; scan-proper (non-fallback)
  worst deficit **7.2e-14**; scan-vs-eigvals parity ≤8.9e-16; K.18 positivity verified live
  (a 9× stronger negative-amplitude root correctly filtered); no overstatement (max +4.6e-11).
- **F1 (shared_phase scan-proper — critic-directed; zero coverage before)**: 96 verified
  non-fallback cells (H up to 10): worst oracle deficit **1.9e-15**; argmax 48/48 fixtures.
- **F3 (H=10 all remaining modes)**: scan-proper worst **6.0e-15**; parity ≤2.5e-16 everywhere.
- **M2/M3/S1**: scan ≡ eigvals bitwise on 100% of deep-dip fallback rows (the C2 contract).

### CONFIRMED findings — all in the shared reference/exact-root path at deep |MM| dips; none
### are scan regressions; all pre-date C1/C2

1. **MB-DIP-1 [major] — shared_phase G-root path returns suboptimal phases at every-band-deep
   dips.** Confirmed by 4 independent chains: orchestrator probes (raw-data oracle validated at
   1e-14 + explicit-lstsq achievability + normalization-free raw-χ² of the code's own returned
   fit: 7.946 vs 7.553 = −4.95% at the anchor case), skeptic S1 (every refutation vector failed;
   model-class bijection s = (−θ₂/2π) mod 1 proven, curves reproduced both directions ≤3.9e-14;
   the returned phase is not even a stationary point of χ²(s): dχ²/ds = +600.6), independent
   reproduction M3 (own fixtures, seeds 90000+: worst **8.42e-2 power / 26.3% raw χ²**;
   incidence >{1e-9,1e-6,1e-4,1e-3,1e-2} = {53.2, 38.5, 24.5, 14.2, 4.0}% of recipe-class cells;
   no overstatement anywhere), and root-cause R1. **Mechanism (R1)**: when the F-maximizer sits
   where ALL bands' |MM_k| dip to ≲1e-4..1e-5 of max, the FP-constructed
   G ~ Π MM_l² · dF (degree 8HK−2) falls 180–1460× BELOW its own coefficient-rounding noise
   floor at the peak — G is pure noise there, no computed root lands within 6e-3..2.5e-2 rad of
   the true maximizer. Eigensolver, F-evaluation-at-roots, projection, and the finite-F guard
   are all exonerated (backward errors at/below the noise floor; the code maximizes faithfully
   over the roots it gets). **Incidence is recipe-conditional**: ~2% of seed-freq pairs on the
   original 40000-family (R1 — consistent with the old "~2%" record), 14.2% >1e-3 on M3's
   deeper every-band recipe; turnoff mapped at dip ratios ≳1e-4 (deficit ≤6.4e-8 at ≥2e-5,
   ≤5.8e-15 at ≥2.7e-4). **Practical impact nil on realistic cadence**: uniform-random sparse
   N/band {4..12}: fallback fires on 100% of rows yet zero deficits >1e-6 (worst 2.9e-8);
   F1's benign arms worst 2.75e-11; requires adversarial phase-clumped geometry. Production
   experiments use floating_offsets. **The 0.15 exact fallback is causally INVERTED for this
   mode (R1-F3)**: deep dips are precisely where G is noise-blinded while F itself stays smooth
   (no sub-grid spike — peak width ~1e-3 cycles) and trivially scannable, yet the fallback hands
   these rows verbatim to the broken root path. **Fix validated (R1-F4)**: the code's own
   `_shared_phase_scan_fit` machinery run WITHOUT the fallback branch + `max(scan, root-path)`
   recovers the true power to <2e-14 on all analyzed cases; merely polishing from the code's
   chosen root is NOT sufficient. Pinned by
   `test_multiband_shared_phase_deepdip_{bounded_vs,attains}_F_oracle` (bounded 6e-3 green;
   tight 1e-9 xfail(strict=False), XPASSes when fixed).
2. **SESAR-DIP-1 [major] — sesar mode returns WRONG peak power at deep dips, both signs,
   including overstatement (false-alarm direction).** F3 (mp-verified): errors at the code's own
   returned phase up to **+1.186e-4 overstatement** and −4.8e-6 understatement; 15/96 deep-dip
   cells >1e-6; **flat in H** (H=6/8/10 all ~1e-4). Mechanism: the sesar-only global-recentring
   correction in `combine_band_summations` (MM′ = Σ W_k MM_k + Σ W_k M̄_k² − M̄_comb² terms) has
   O(0.1) coefficient mass that cancels to ~1e-13 on the circle — MM′ is wrong by −4.8e-4
   relative at the dip while YM′ stays accurate to 5e-10 and floating_offsets' MM at identical
   conditioning is accurate to 1.3e-9. Identical in eigvals and scan (all such cells route
   through the fallback). Only at conditioning ≲1e-7 with phase-clumped ~5-pt/band sampling.
   The earlier "sesar ceiling 3.7e-11" (M2, H≤8) was fixture-limited, not a real bound.
3. **FO-TRIM-1 [confirmed defect; magnitude minor (≤3.9e-8), mechanism actionable] —
   `trim_zero_leading_coef` eats the GENUINE degree-(6H−2) coefficient.** F2 (50-digit mp
   adjudication of M1's two worst floating_offsets rows): when the analytic degree-(6H−1) zero
   cancels to EXACT FP zero, numpy's trimseq removes it first, and the conditional trim
   `|coef[-1]| ≤ 1e-9·scale` then fires on the real c_{6H−2} (genuine at 2.3e5–4.3e6× above
   construction noise; one case fired at |c|/thresh = 0.95 — a whisker). Root displaced by
   exactly the predicted |c·r^{6H−2}/G′| (3-digit agreement); deficits 3.4e-8/3.9e-8 are REAL in
   exact arithmetic. Both proposed mechanisms REFUTED: eigensolver backward-stable (untrimmed FP
   roots land 2.3e-9 from the oracle phase; deficit collapses to 1.6e-13), construction noise
   1.3–12× eps·max (37–8700× too small). **Affected surface**: every `stationarity=None` trim —
   single-band eigvals, multiband eigvals (fo/sesar/independent), and the scan's exact-fallback
   rows; the batched path is immune (`batched_stationarity_coefs` keeps c_{6H−2}
   unconditionally). **Fix is a one-line guard** (length-gate the trim: only drop when the
   polynomial still carries its nominal degree 6H−1). This realizes in practice the "contrived
   corner" flagged as C1-hardening finding 5. Extended-precision rooting NOT warranted.
4. **[minor] Value-evaluation noise scope at extreme conditioning (shared by both paths).**
   (a) P2-F1: at circle conditioning ≤~1e-7 both methods mis-evaluate the power of their own
   (near-optimal) returned model by up to 2.1e-8 below / 1.0e-9 above raw truth (3/540 rows;
   exceeds the previously documented ~5e-10 floor, which holds above ~1e-8 conditioning).
   (b) P2-F2: inside the single-band exact fallback, scan ≠ eigvals bitwise —
   `batched_YM_MM_from_sums` (np.trace) vs per-frequency `get_diags` differ by 1 ulp in MM,
   amplified ~6.6e7× at conditioning 7.2e-10 to a 6.04e-9 power difference. The fallback
   guarantees equal *treatment*, not equal values (multiband fallback re-enters per-frequency
   assembly and IS bitwise). `template_periodogram`'s docstring re-scoped accordingly (this
   commit). **C4 design note**: the scan≡eigvals pin must scope conditioning (tolerate ~1e-8
   below min|MM|/max|MM| ~1e-8, or pin on conditioning-screened fixtures).

### Deferred-fix verification (V1) + record corrections

All `8dad841` numbers reproduce against the actual a8fd646 worktree (bitwise stable across
processes; mechanism-pin assert margins 2.8–2.9×; a fallback-less mutant fails both pins;
deep-dip regime guard proven live — de-clustered fixtures fail the trigger assert). V1-1
(sampled-row provenance of the "3.6e-3" figure) fixed in this commit: row 21 — the
clump-degenerate f=1.15, adjudicated per-row by S1 — is now sampled, and docstrings state the
recipe-conditional magnitudes.

### Operational notes (for C4/C5)

- Fallback incidence is high off the well-conditioned regime: 75% of shared_phase cells overall
  (67% even at uniform 40 pts/band; 100% for eclipse H≥8 K=3), 98% of independent-mode per-band
  solves at H=10/40-per-band (0% at 100/band), ~80% of the adversarial floating_offsets mix.
  The scan's speed advantage accrues on well-conditioned data; C5's timing narrative must not
  assume scan-proper everywhere, and the (HK)³ shared_phase cost saving is rarely realized at
  H≥8 with N~40/band.

### Sign-off status

**The scan+polish maximizer is verified clean** — the WP C3 gate criterion ("scan < oracle −
1e-10 or argmax differs from eigvals") produced ZERO confirmed hits across ~7,900 gated
comparisons in six independent modalities. However, the verification CONFIRMED three unfixed
defects in the shared reference path (MB-DIP-1, SESAR-DIP-1 major; FO-TRIM-1 minor). Per the WP
C3 text ("ESCALATE if it finds one") and the 2026-06-12 delegation (ESCALATE on confirmed
defect), **C3 is NOT signed off; ESCALATED to John 2026-07-04** with a validated fix package
(scan-first + max(scan, root) for shared_phase; one-line trim length-gate; sesar recentring
reformulation to be designed). **C4 (default flip) not started** — strict C-track order.

## WP C3.5 — deep-dip reference-path fixes (`dd632f0`, `decfe08`, `20bf034`) — VERIFIED 2026-07-05

**Scope**: John approved option A of the C3 escalation (2026-07-04): fix the three confirmed
defects, then C4. **Fixes**: FO-TRIM-1 — `trim_zero_leading_coef` length-gated by the nominal
degree (6H−1 band / 8HK−1 shared-phase G), so a genuine degree-(6H−2) coefficient is never
eaten after numpy has removed the exactly-cancelled analytic zero. SESAR-DIP-1 — the sesar
recentring variance term is assembled in centered form Σ W_k (M̄_k − M̄_comb)² (the
between-band variance definition; second-order-insensitive to the ~2-ulp deviation of the
float W_k sum from 1). MB-DIP-1 — `_shared_phase_scan_fit` no longer delegates deep-dip rows
verbatim to the G-root path: it escalates the scan grid to the Bernstein bracketing bound
Δθ ≤ √(2 r_min)/(2H) (cap `_SCAN_DEEP_MAX_ANGLES` = 2^20 after panel hardening) and returns
max(scan, root path) — strictly safe, both being true-objective evaluations at genuine
phases. **Deliberate contract change**: shared_phase `method='scan'` is now ONE-SIDED at deep
dips (scan ≥ eigvals, recovering the reference's deficit); eigvals stays the pure reference.

### Fix results (orchestrator, re-running the C3 harnesses on fixed code)

- M3's independent 400-cell MB-DIP-1 sweep: worst (oracle − scan) 8.4e-2 → **7.1e-10** at the
  final 2^20 cap (zero cells >1e-9; overshoot ≤9.1e-10 = shared floor; scan ≥ eigvals on
  400/400). The default-density max-merge alone left up to 2.0e-2 (68/400) — the Bernstein
  escalation was REQUIRED, not optional.
- F3's 96-cell sesar sweep: worst |error| 1.186e-4 → **5.9e-10** (both signs bounded).
- F3's floating_offsets / independent sweeps: 7.4e-10 / 6.8e-9 (unchanged micro-levels).
- F2's adjudicated FO-TRIM-1 cells: gains +3.93e-8 / +3.42e-8 reproduced exactly.
- Realistic-cadence probe: worst 2.9e-8 → 3.0e-12. Full suite 470 passed / 1 skipped /
  2 xfailed (the 3 MB-DIP-1 xfails now PASS as fix gates; +4 new regression tests).

### Adversarial panel (workflow `wf_2f97265f-95b`, 8 agents; A6 refutation agent stalled and
### was dropped — its three checks were covered by A1/A3/A4 (overshoot vs 50-digit truth),
### A2 (Bernstein-transfer + cap-zone analysis), and an orchestrator probe (one-sidedness:
### 200 fresh deep-dip cells, 0 violations, 31 bitwise root-wins / 169 scan recoveries)

- **A1 (MB-DIP-1, harsher fixtures than C3)**: 360 cells (clump widths 3e-4 AND 3e-5, K≤4,
  H≤10, r_min to 4.6e-11), own validated raw-data oracle (3.5e-14): zero genuine misses on
  non-cap-bound cells (worst TRUE miss +2.4e-10 after 50-digit adjudication of the two
  apparent gate hits); overshoot vs f64 oracle ≤1.1e-11; eig − scan ≤ 0 EXACTLY everywhere;
  pre-fix worktree deficits up to 2.1e-2 removed (non-vacuity). Cap-zone (then 2^17)
  quantified: worst true miss 1.472e-4, recovered to ~1e-6 by 2^20 → adopted in `20bf034`.
  Notable scope facts: eclipse-like templates at H≥8 route through the deep-dip merge even on
  benign dense data (r_min < 0.15 is ROUTINE there), and at r_min ~3e-8 a 1-ulp input
  perturbation moves the result by ~1.2e-7 — float64 is input-limited in that zone.
- **A3 + follow-up F1 (SESAR-DIP-1)**: centered form non-perturbing (1.1e-15 vs uncentered on
  well-conditioned cells; code vs raw oracle 5.0e-14 there); fresh defect-geometry fixtures
  (identical clump centers, clump-degenerate trial frequency — the load-bearing lever)
  re-excited the pre-fix defect to **6.36e-3** (58/576 rows ≥1e-5, both signs) while the fixed
  tree stayed within [−1.1e-8, +5.7e-9] of the oracle; pre-fix worktree ≡ inline uncentered
  rebuild BIT-EXACTLY, isolating the delta to the fix.
- **A4 (FO-TRIM-1)**: single-band well-conditioned = strict bitwise no-op (the gate corner
  fired 0/4800 cells); where the gate changes answers the fixed value is the better one
  (164/164 corner cells adjudicated); batched path bit-identical pre/post.
- **A5 (global no-regression, both trees)**: nothing outside the three fix surfaces moved;
  sesar well-conditioned shifts ≤~1e-13 (assembly rounding), everything else bitwise or
  ≤1e-15. Timing is config-dependent: sparse all-deep rows cost up to 2.5× eigvals at H=2
  (G cheap) vs ~1.0–1.2× at H=8; dense keeps ~1.7–2× scan advantage at H=8, ~1× at H=3.
- **A2 (adversarial diff review)**: structural proof the length gate can never eat a genuine
  coefficient; sesar YM′ correction's analogous uncancelled assembly contributes only ~1e-10
  (recorded, no action); escalation arithmetic total and safe (r_min = 0/NaN handled; flat-band
  semantics preserved). Doc over-claims it flagged (W_k-sum wording, cap-coverage comment,
  stale 3.562e-3 magnitude — the G-trim gate alone recovered +2.92e-3 of that cell's eigvals
  deficit, residual ~4.5e-4) fixed in `20bf034`.
- **Follow-up F2 (fast=True/NFFT surface)**: all parity gates pass; fix signatures realize
  identically under NFFT sums (max-merge recovered +2.7e-2 at an extreme dip with the cap
  exercised); NFFT floor 3.3e-12 well-conditioned, conditioning-amplified at extreme dips as
  documented.

### Residual scope notes (recorded, no action)

- Shared float64 value-evaluation noise at extreme dips (r_min ≲ 1e-7): reported power can
  deviate from exact-arithmetic truth by ~±1e-8 class (worst measured overshoot +5.8e-9,
  deficit −1.1e-8; supersedes the earlier "+1.0e-9" upper-side figure). Pre-existing, shared
  by eigvals/scan/any float64 oracle — not a maximizer defect.
- Cap-bound best-effort zone now r_min < ~(4πH/2^20)²/2 (≈4.6e-10 at H=8): no committed
  fixture exercises it; behavior there is max(scan, root) best-effort (panel evidence only).
- Test-authoring follow-ons for a later WP: a committed cap-zone regression fixture; a
  deep-regime guard on the one-sided assert in `test_narrow_peak_multiband_clustered`.

### Sign-off

Zero unfixed critical/major findings; all panel findings were minor/info and are fixed in
`20bf034` or recorded above. **WP C3.5 signed off 2026-07-05 (adversarial verification in
lieu of human review, per delegation). The C3 escalation is RESOLVED (option A); C4 is
unblocked.**

## WP C4 — scan+polish is the default (`bbdf0ca`) — 2026-07-06

Defaults flipped (core / multiband / modeler); `method='eigvals'` retained permanently as the
reference/validation mode. Anti-drift pins in `test_default_method.py`: default ≡ scan
(bitwise, single band + all four multiband modes + modeler autopower); default ≡ eigvals to
1e-12 with identical argmax on conditioning-screened fixtures (two-sided scope per WP C3
finding 4b); shared_phase deep-dip one-sidedness (default ≥ eigvals, ≥1e-5 recovery
non-vacuity) per the C3.5 contract. 27 pre-existing reference comparisons made explicitly
`method='eigvals'` (they would otherwise have become scan-vs-scan tautologies). Default-path
smoke benchmark: 26.3× @H=2 / 19.8× @H=8 / 22.1× @H=10 vs eigvals, max|ΔP| ~1e-15
(N_obs=300, N_freq=20000). Full suite 483 / 1 / 2 / 0. The production experiment harness
(FTPEstimator, greedy source_masks, joint-EM E-step) now runs the scan maximizer by default —
relevant to B8-closure and C5 timing.

## WP D1 — measured null / FAP calibration (`experiments/fap/`) — VERIFIED 2026-07-10

Collection/finalization of the detached 3000-realization pure-noise Monte-Carlo launched
2026-07-10 (marker `output_null/DONE.marker`; mc wall 3468.2 s ≈ 58 min; local, unpaid).
All numbers below re-measured in THIS session from `output_null/null_maxpower.npz`, not quoted.

**Acceptance gate (COLLECT_D1.md Step 2), re-measured — PASS.** Config in the npz:
n_real=3000, n_obs=40, n_freq=10000. Pure-noise null max-power at N=40:
FTP H=1 mean 0.4075 / median 0.4013 / p99 0.5347; H=3 0.4404/0.4350/0.5509;
H=6 0.4425/0.4361/0.5507; H=12 0.4426/0.4367/0.5527; GLS ≡ FTP H=1. Invariants:
FTP H=1 == GLS max|diff| = 1.20e-9 (< 1e-8); catalog-max `cat_K8` == max over the 8 vocab
templates exactly (0.0); catalog-max nested/monotone in K∈{1,2,4,8} (True); all values
finite. Anchor is in the audit's "~0.5" ballpark (canary was mean ~0.42-0.44) — not
wildly off, no ESCALATE. Both figures render sensibly: FTP null sits modestly above GLS
(extra nonlinear-phase DOF), GEV dashed tails track the empirical curves, catalog extra-
trials penalty grows monotonically with K.

**Adversarial verifier (1 bounded agent, blind — headline-bearing result).** Single claim:
independently confirm (A) the pure-noise FTP H=1 max-power null at N=40 is ~0.4-0.5 and
(B) FTP H=1 == floating-mean Lomb-Scargle to <1e-8 per-frequency, using its OWN hand-rolled
weighted-LS χ² oracle (NOT `GLSEstimator`), its OWN fresh pure-noise LCs/seeds, and WITHOUT
reading `measure_null.py` or `output_null/`. Budget: 30 realizations, ≥4000 freqs, <30 s.
**Verdict: CONFIRMED (with one quantified caveat).**
- (B) identity: CONFIRMED decisively — its from-scratch oracle matched FTP H=1 to 5.3e-10
  across ~10⁴ freqs × 30 realizations (round-off, not a modeling gap); the P=1−χ²/χ²₀
  normalizations coincide. Independent confirmation of the 1.20e-9 gate number above.
- (A) anchor: CONFIRMED — its independent production-density run gave mean 0.402 /
  median 0.397 / p99 0.515, within ~1% of the re-measured 0.4075 / 0.4013 / 0.5347.
  **Caveat (recorded honestly, not overstated in any artifact):** the null-max *center*
  pins to ~0.40 — the LOW edge of "~0.4-0.5" — and is grid-density-sensitive (~0.38 at a
  ~5k grid; a 3-yr baseline supports only ~4400 independent frequencies, analytic
  median-of-max ≈ 0.377, and oversampling to 10k nudges the center up to ~0.40). Treating
  0.45 as the typical null max would be slightly optimistic; SUMMARY.md quotes the actual
  measured mean/median/p99, so no artifact overstates this. Zero confirmed defects → no
  ESCALATE; D1 finalized to DONE.

## WP C5 — timing narrative + performance figures (paper repo `h-vs-accuracy`) — VERIFIED 2026-07-10

Fresh timing benchmark `paper/…/timing_v5/timing_c5.py` (never legacy `scripts/`; recorded
`SEED=1`) **re-run in THIS session** on the submission machine (D1 compute finished → quiet
machine); `timing_results.json` + the three figures (`plots/timing_vs_nharm`,
`timing_vs_ndata`, `timing_vs_ndata_const_freq`) regenerated at 16:58. Compares the default
scan+polish maximiser (`method='scan'`, C4 default) vs the permanent reference root path
(`method='eigvals'`) vs full non-linear optimisation (`SlowTemplatePeriodogram`, `nguesses=10`),
on **identical seeds/data per configuration** (WP MUST iii). **Every number written into
paper_v5.tex was re-measured from this session's run**, not quoted from the interrupted prior
session (an earlier queue iteration produced the same edits but was killed by the harness
background-wait ceiling before commit; its numbers were discarded and independently reproduced
to within timing noise).

**Re-measured numbers (this session, min-of-reps):**
- Single-band scan-vs-eigvals speedup vs H (N=200, Nf=6000): 34.9/29.2/26.6/25.6/25.3/25.7/
  26.9/30.2/29.6/31.3/34.3/34.9× for H=1..12 → **range 25.3–34.9×** (TeX "25–35"). U-shaped,
  never below 25.
- H=8 grid-stability vs Nf (N=300): Nf=2000 → 31.9× (eigvals 1.52 s → scan 0.048 s);
  8000 → 29.3× (6.10 → 0.208); 20000 → 28.3× (15.27 → 0.540). TeX "≈32×, ≈29×, ≈28×".
- Log-log slopes over H=8..12: **eigvals 2.29, scan 1.86** (polyfit) → TeX "≈2.3 / ≈1.9".
- Multiband `shared_phase` (K=2, N=200, Nf=2000): H=3 → 1.34× (1.84 → 1.38 s), H=8 → 5.48×
  (19.52 → 3.56 s). TeX "1.3× / 5.4×" (both round DOWN from measured — conservative).
- FTP(scan) vs non-linear (nguesses=10): const-cadence (Nf=12N) N=15 → 13.7×, N=250 → 479×;
  const-baseline (Nf=2000) N=15 → 14.6×, N=250 → 498×. TeX "~14–15×" and "~480–500×"
  (brackets both regimes) and const-baseline "≳480×" (498 ≥ 480 ✓).

**Complexity (settled from code, not doc):** the reference eigvals path roots a degree-(6H−2)
companion matrix → O((6H)³) = O(H³) per trial frequency; the default scan path's per-freq cost
is O(H²) (coefficient assembly O(H²); polish of ≤4H bracketed candidates × constant Newton
steps × O(H) Horner = O(H²); circle-eval FFT subdominant O(H log H)), with the adjoint-NFFT
summations shared across the grid at O(HN_f log HN_f). So the TeX's default
**O(HN_f log HN_f + H²N_f)** / reference **O(H³N_f)** (and the ×H resolving-power variants
H³N_f / H⁴N_f) are correct. Measured slopes 2.29/1.86 at H≤12 sit below the 3/2 asymptotes
because the lower-order NFFT and constant terms still contribute in that range — the paper's
"interpolates between O(H²) throughout the practical H≲15 and O(H³) as the deep-|MM| exact-root
fallback incidence grows" hedge is honest (and conservative — real fallbacks are rare on
well-conditioned data).

**Acceptance gate — PASS.** PDF builds (`make all`, exit 0) → **19 pages, 0 undefined
references/citations, no new or appendix overfull hbox** (3 pre-existing body overfulls only,
all outside the C5 diff hunks). Figures regenerate from the committed fresh script with the
recorded seed. Retired headroom claims removed everywhere as live text (10³×/`~5%`/`1–2 orders`/
`25–100×`/`factor of 3–20`/`10^120` survive only inside `%` comments); eigvals labelled "the
reference Python implementation"; precision-floor caveat untouched (not claimed retired, per WP).

**Adversarial panel (workflow `wf_a5f3df23-e2e`; 3 independent verifiers, own oracles/scripts,
effort max/xhigh).** Zero confirmed critical/major.
- **complexity-derivation (max; read ONLY core.py, NOT the timing script/JSON/tex): CONFIRMED.**
  Independently derived scan = O(H²N_f + HN_f log HN_f), reference = O(H³N_f) "exactly as
  claimed"; interpolation hedge "honest … conservative". 1 *minor* nuance: the scan carries a
  second, subdominant per-frequency circle FFT (`_eval_polys_on_circle`, O(N_f H log H)) not
  named in the decomposition — dominated by the O(H²N_f) assembly and the shared NFFT term, so
  the stated leading order is unaffected. No edit required.
- **timing-repro (xhigh; own fresh script, did NOT read `timing_c5.py`/JSON): CONFIRMED band.**
  Independent measurement 34.9×/29.6×/37.0× at H=1/8/12 (25–35× band; H=12 within ±20% timing
  tolerance) and FTP-vs-nonlinear growing to a few-hundred× by N=250. Surfaced that the
  FTP-vs-nonlinear factor scales ~linearly with the optimiser's restart count → **fix applied:
  fig:timingndata caption now states "10 random phase restarts per frequency".** Its "N=60 is
  ~112× not 10–20×" finding was against the orchestrator's brief ballpark, **not** any paper
  claim (the paper makes no N=60 claim — grep-confirmed by the critic).
- **tex-audit (max; cross-checked EVERY TeX number vs the JSON): CONFIRMED.** "All substantive
  claims supported; none materially overstate the data" (max deviation 2.4%, all nearest-
  rounding). 29×/28×, 1.3×/5.4× round DOWN (conservative). Flagged the one bare (non-`~`)
  round-up — "32×" for measured 31.888× — → **fix applied: now "≈32×, ≈29×, ≈28×".**
- **completeness-critic:** the workflow synthesis agent FAILED TWICE on harness/serialization
  issues (first the sandbox lacks `JSON.stringify`; on resume, the StructuredOutput retry cap
  was hit — the `findings` array kept being rejected, a payload/special-char formatting failure
  after ~8.5 min), NOT on substance. Its transcript recorded a complete CONFIRMED assessment,
  adopted here by the orchestrator (precedent: C3.5 A6 refutation-agent stall): both applied
  fixes verified live in the tex; no `N_obs=60` claim exists; the H⁴ reference term is fully &
  consistently derived (reference O(H³N_f) × the δf∝1/H peak-resolution factor, labelled
  reference-only); the only C5-adjacent number left unrefreshed — the introduction's
  Lomb–Scargle NFFT-vs-direct baseline timings (lines ~342–352) — is **out of C5 scope** and
  already self-caveats "will be refreshed for the final submission" (recorded for a later pass,
  no action this WP).

**Sign-off (adversarial multi-agent verification in lieu of human review, per delegation
2026-06-12).** Zero unfixed critical/major; the two actionable findings (nguesses provenance,
bare "32×" round-up) both FIXED this session; no measured number changed. C5 SIGNED OFF. No
ESCALATE trigger fired (acceptance gate passed first attempt; no headline shift — the re-run
reproduced the prior interrupted run to within timing noise; no paid compute; branches
`h-vs-accuracy`/`dev`, `master` untouched). Suite unaffected (paper-only + docs).

**Addendum 2026-07-10 — reconciling 28.3×@H8 (this WP) vs 19.8–20.2×@H8 (C2 gate / C4
smoke) at the nominally identical H=8 / Nf=20000 / N_obs=300 cell.** Both numbers are
correct; they measure different fixtures. Re-measured back-to-back this session (same venv,
min-of-3 per method, 1-min loadavg 1.9–2.5 throughout — mildly contended, flagged):
the C4-smoke fixture (`benchmark_scan_polish.py`: random N(0,1)-coefficient template,
seed 0, 10-d baseline, heteroskedastic dy 0.05–0.10) gave eigvals 15.16 s / scan 0.743 s =
**20.4×** (scan min-of-10 floor 0.699 s → 21.7×), while the `timing_c5.py` block-D fixture
(1/n-decaying template cos(0.6n)/n, 250-d baseline, dy = 0.02, spp = 5) gave eigvals
15.45 s / scan 0.599 s = **25.8×** (scan min-of-10 floor 0.559 s → 27.7×, vs the committed
28.3× — residual is scan-side scheduling noise on a ~0.55 s denominator). The eigvals
reference cost is fixture-independent (15.16–15.48 s, i.e. ±1%, across all four cells
measured); the entire ratio difference lives in the template-dependent scan cost. Isolation
toggle confirms: fitting the C5 1/n template on the C4 fixture's own data+grid moves the
ratio 20.4× → **26.1×** (eigvals 15.48 s / scan 0.594 s), and the reverse swap (C4 random
template on C5 data) drops it to 22.6× — the fitted template's harmonic spectrum is the
dominant knob (flat-spectrum random coefficients make the scanned objective more oscillatory
and the scan+polish stage ~1.25× slower), with the data/grid contributing a small secondary
effect (≲10%). The reps asymmetry (eigvals min-of-2 in timing_c5.py vs min-of-3 in the smoke
benchmark) and BLAS threading (env unset in both) were checked and are immaterial. **Quote
~20× for the C2/C4 random-template smoke fixture and ≈28× for the paper's timing_c5 config;
neither supersedes the other.**

## WP F1 — multiband appendix derivations (paper repo `d825d69`) — VERIFIED 2026-07-10, defects closed 2026-07-10

**Original work (2026-07-10, unattended queue session; paper repo `h-vs-accuracy`
commit `d825d69`).** Retitled the multiband appendix, added the sharing-hierarchy
Table 2 (all four modes: shared-vs-free parameters, reference variance $V_0$, per-frequency
root cost), the `floating_offsets` derivation (the default headline mode), a `shared_phase`
subsection, and a hierarchy-wide cost subsection. Every claim was re-measured in that
session against `ftperiodogram/multiband.py` / `core.py`. In-run sign-off = **2-agent
adversarial panel @max** (independent re-derivation SIGN_OFF, 14/14 CONFIRMED; adversarial
TeX audit, 14/14 CONFIRMED, its 3 flags fixed in-session), recorded in the PLAN.md Status
ledger. No VERIFICATION.md entry was written at the time — this section closes that gap.

**Independent audit finding (2026-07-10, `AUDIT_2026-07-10_QUEUE_RUN.md`, f1-math
verifier).** The audit re-derived the floating_offsets algebra from the code and confirmed
the substance (its record: end-to-end FO power vs brute force 8.7e-16; polynomial degrees
16 = 6H−2 and 46 = 8HK−2; all four Table 2 rows verified behaviorally, including $V_0$ per
mode to ~1e-15). It also found **two real TeX defects that the in-run 14/14×2 panel
missed**:

1. **Hatted-macro / centred-moment inconsistency.** The new `app:floatingoffsets`
   subsection reused the hatted macros `\YMh`/`\MMh` — which the paper's own definitions
   (single-band appendix; sesar subsection) fix as *uncentred* moments — for *centred*
   per-band quantities, so `eq:foaverage` was wrong under the paper's own symbol
   definitions and correct only via the prose gloss ("about that band's own weighted
   mean"); the neighbouring `app:sharedphase` writes the same centred objects unhatted.
   The primes on the band-combined `YM'`/`MM'`/`YY'` were also used without the
   ψ-rescaling that primes carry everywhere else in the derivation appendices.
2. **shared_phase cost-sentence contradiction.** `app:sharedphase` called `shared_phase`
   "the only member of the hierarchy whose per-frequency cost grows with $K$ beyond the
   summations", contradicting Table 2's own $O(KH^3)$ row for the `independent` mode
   (linear in $K$).

**Policy significance.** This is the **first measured false-negative for the
sign-off-delegation policy** (2026-06-12: adversarial multi-agent verification in lieu of
human review): a 2-agent panel returned 14/14 CONFIRMED twice over a work product that
contained two real (TeX-internal, non-substantive) defects. Calibration point recorded:
panel CONFIRMED counts bound *substance* well (the audit found zero math/code defects)
but under-detect notation-consistency defects that require cross-referencing symbol
definitions from distant sections; future paper-WP panels should include an explicit
notation-consistency check against the paper's own macro definitions.

**Fixes (this session, 2026-07-10; paper repo `h-vs-accuracy`).**
- **`8eb6b28`** — closes both audit defects. (1) `app:floatingoffsets` now uses unhatted
  centred per-band moments $YM^{(k)} \equiv \widehat{YM}^{(k)} - \bar{y}^{(k)}\overline{M}^{(k)}$,
  $MM^{(k)} \equiv \widehat{MM}^{(k)} - (\overline{M}^{(k)})^2$ (matching `app:sharedphase`),
  with overlined $\overline{YM}/\overline{MM}/\overline{YY}$ for the plain $W^{(k)}$-weighted
  band averages and an explicit remark reserving hats for uncentred moments and primes for
  ψ-rescaled polynomials; verified this session against the implementation ground truth
  (`multiband.py::combine_band_summations` — plain $W_k$-weighted average of per-band
  `YM`/`MM` built by `core.YM_MM_from_sums` from the *centred* sums of `summations.py`,
  normalised by `YY_combined` $= \sum_k W_k YY^{(k)}$). (2) The `shared_phase` sentence now
  states the true distinction — the only mode whose phase maximisation *couples* the bands
  into a single polynomial of $K$-dependent degree (superlinear $(HK)^3$), vs the
  `independent` mode's $K$ repetitions of the single-band solve — and `app:mbcost`'s
  "no larger than a single-band periodogram's" was tightened to acknowledge the $K$-fold
  NFFT summations. Build gate re-run: `make all` exit 0, **19 pp, 0 undefined references,
  no new overfull boxes** (the 3 pre-existing body overfulls only).
- **`25ae62c`** (same session, same audit run, c5-consistency verifier finding) — the
  stale abstract speed claim ("a factor of a few faster for $N_{\rm obs}\lesssim 10$", a
  pre-C5 leftover with no supporting measurement) replaced with ratios recomputed this
  session from the committed `timing_v5/timing_results.json`: 13.7×/14.6×
  (const-cadence/const-baseline) at $N_{\rm obs}=15$ rising to 479×/498× at
  $N_{\rm obs}=250$ → abstract now says "≈14× at $N_{\rm obs}=15$ … ≳480× by
  $N_{\rm obs}=250$, an advantage that grows without bound".

Both commits pushed to `h-vs-accuracy`; `master` untouched in both repos.

## WP D2 — significance/FAP paper subsection (paper repo `768b36f`) — DONE 2026-07-10

New subsection **3.2 "Significance and false-alarm calibration"** (`\label{sec:fap}`) in
`paper_v5.tex`, placed in §3 (Implementation) after the non-linear-optimization comparison
and before the Discussion. **Every number in the TeX was recomputed in THIS session from
committed artifacts** — never quoted from SUMMARY.md, memory, or the superseded
`headline_full/`:

- From `experiments/fap/output_null/null_maxpower.npz` (D1; copied into paper repo
  `fap_v1/`): FTP H=1 == GLS max|diff| **1.2e-9** over 3000 pure-noise maxima; null-max
  means 0.4075/0.4404/0.4425/0.4426 (H=1/3/6/12) → TeX 0.408/0.440/0.443/0.443; empirical
  FAP=0.01 thresholds 0.535/0.551/0.551/0.553 (inflation < +0.02, saturating at H≥3);
  catalog-max means 0.4219→0.4666 and FAP=0.01 thresholds 0.540/0.561/0.570/0.576 for
  K=1/2/4/8; single-template FAP at the K=8 threshold 0.0030 → Bonferroni (0.024)
  conservative by ×2.4; between-template spread of per-template null-max means **0.0251**
  (the "~0.025" bound in caveat b).
- From `experiments/phase3_recovery/rerun_202606/raw/a-sesar-*/out/per_source/
  seed*__ftp_pam__ksweep-sparse-K{1,2,4,8}.npz` (B8 rerun, NOT headline_full): pooled
  recovery 550/549/567/569 of 768 → 0.716/0.715/0.738/0.741 with Wilson 95% intervals
  [0.683,0.747]/[0.682,0.746]/[0.706,0.768]/[0.709,0.771] (recomputed from raw
  `recovered` arrays; matches `aggregate_sesar.json` lo/hi). K-sweep wobble (range 0.026,
  overlapping intervals) framed as comparable to the measured extra-trials cost.
- V0-per-sharing-mode sentences verified against `ftperiodogram/multiband.py` ground truth
  (`_prepare_bands`: `YY_global` on λ-subtracted data for `sesar`, else
  `YY_combined = Σ_k W_k YY^(k)`; used at the `template_periodogram` power normalisation
  and the `independent`/`shared_phase` paths), consistent with Table 2's V0 column.

**Figures**: fresh `fap_v1/make_fap_figures.py` (paper repo; deterministic from the
committed npz, no seed needed; CVD-validated fixed-order palette) regenerates
`plots/fap_vs_threshold.pdf` + `plots/nullmax_vs_K.pdf`; both included as Fig. 7.

**Mandatory audit caveats (AUDIT_2026-07-10_QUEUE_RUN.md d1-scope findings) are in the
TeX**: (a) single-band N_obs=40 null is not transferable across N/band structure/grid
without recalibration; (b) nested-prefix null K-axis vs production per-K vocabulary
rebuild, bounded by the 0.025 between-template spread. **Scope guard**: significance
calibration only — no completeness/purity sweep (Paper-2 territory).

**Acceptance gate (EXECUTION_PLAN D2 = "PDF builds") — PASS**: `make all` exit 0,
**20 pages, 0 undefined references, no new overfull boxes** (3 pre-existing only, all
outside the diff). No verifier subagent: D2 carries no marked verifier in
EXECUTION_PLAN.md and every quoted number was recomputed in-session directly from the
committed artifacts above. Commit `768b36f` pushed to `h-vs-accuracy`; `master` untouched
in both repos.

## WP E2-acquisition — sample verification (`66e5dd3`) — VERIFIED 2026-07-10

Independent verifier (this session, artifacts-only — fetch agent's reasoning not read).
Verdict: **PASS** (minor findings only). Checks:

1. **Manifest vs approved spec — PASS.** 58 science / 15 controls; per-field 486=22
   (20–25), 686=18 (15–20), 786=18 (15–20); RRab/RRc 11-11 / 9-9 / 10-8 (balanced to
   availability). Every science star has Chen `period_truth_days` + Type
   (`period_source: chen2020_Per`); every control has `control: true`, a Gaia period
   (`gaia_dr3_pf` for RRab, `gaia_dr3_p1_o` for RRc), and a `chen_overlap` key.
   `rejects[]` = 5, all with reasons (0-epoch prescreen, no r-match <1", <40 epochs);
   `n_failed_promoted` = 5; `problems` = [].
2. **Manifest vs disk — PASS.** 73/73 npz exist, 0 orphans, 0 missing; per-band npz
   epoch counts == manifest `ngoodobs` for all 73×2; all ≥40; all `sep_arcsec` < 1;
   all `{g,r}_{mjd,hjd,mag,magerr}` keys present and finite.
3. **Data quality (seed 42; 4 sci + 2 ctl) — PASS.** Spans 7.1–7.6 yr both bands;
   magerr ∈ [0.011, 0.028] (inside 0.005–0.3); median mags match manifest within
   0.04 mag; zero duplicate timestamps.
4. **Truth sanity (seed 43; 3 sci + 2 ctl) — PASS 5/5 exact.** Hand-rolled
   floating-mean GLS, g band, 20k freqs over explicit [1,5] c/d (no nyquist_factor):
   every top peak = truth fundamental within tolerance (~7e-4 c/d); no alias needed,
   RRab included (2f subdominant). Per-star: f_truth/f_found (c/d) 3.158057/3.158108,
   2.015496/2.015251, 1.561889/1.562028, 4.041174/4.041152, 1.627180/1.627231.
5. **Circularity guard — PASS.** Controls come from a genuine ESA TAP ADQL query
   (`gaiadr3.vari_rrlyrae JOIN gaiadr3.gaia_source_lite`), raw CSVs cached on disk
   (651/958/32 rows for 486/686/786); selection brightest-first by `phot_g_mean_mag`
   from the Gaia list; truth is Gaia's, with `chen_per` + `chen_sep_arcsec` recorded
   for the 12 overlap stars; all 12 Gaia-vs-Chen periods agree to <0.1%.
6. **Git — PASS.** `66e5dd3` on `dev`, contained in `origin/dev`, exactly the 3
   claimed files; no phase4 LC npz anywhere in git history.

**Minor findings (no action required):**
- The 12 `chen_overlap` controls are exactly the 12 controls sharing ZTF oids with
  science-sample stars — same physical stars, byte-identical npz under two `star_id`s
  (verified on one pair). The 73 manifest entries cover **61 unique stars**. Expected
  from dual brightest-first selection in the same fields and flagged in the manifest,
  but downstream analyses must dedupe by oid (or use overlap controls for
  truth-validation only) — never treat the 73 as independent.
- Field 786 Gaia SOS pool is thin (31 candidates vs 651/958); 5/5 controls still filled.
- Reject reason `"prescreen ngoodobsrel g=45 r=39 < 40"` reads loosely (only r fails);
  behaviour is correct.

## WP E3 — false-alarm run on real photometry (`experiments/phase4_demo/e3_false_alarm/`) — DONE, verified 2026-07-11

**Verdict: CONFIRMED** (independent adversarial verification; artifacts-only — all numbers
below recomputed by the verifier's own scripts from `e3_results.npz`, `screen_table.json`,
the per-unit `output/*.npz`, the raw cached photometry, and the D1 null
`experiments/fap/output_null/null_maxpower.npz`; the run agents' reasoning was not read).

**Supersession note:** an earlier "IN PROGRESS, checkpoint 2026-07-11" version of this
section (commit `8117d80`) quoted placeholder text — including a bogus
"**Verifier: CONFIRMED**" line that merely described monitoring being armed — left behind
by a killed workflow. That checkpoint was NOT a verification sign-off; it is superseded by
this record, which is the actual verification.

**What ran (commits `4525c35`, `52ea39e`, `767fde8`, `cc94423`, `a5cea8e`, all on `dev`,
pushed):** phase-1 variability screen of the 99 cached ZTF field objects (81 clean / 18
flagged; 155 band-LCs = 126 clean + 29 screened, 0 skipped for N<40), then
`run_e3.py` — (a) PRIMARY: per band-LC seeded N=40 subsample (catflags==0, seeds
500,000+) scanned on the identical D1 10k-freq [1,5] cyc/day grid and compared to the D1
n=3000 null; (b) SECONDARY: full-N runs on per-object df≤0.2/T grids compared ONLY to
(c) a fresh N/T/grid-matched synthetic null per band stratum (g: N=626, T=2743.8 d,
n_freq=54,877; r: N=947, T=2730.7 d, n_freq=54,615; 500 realizations each, seeds
3,000,000+/3,500,000+). Statistics: GLS, FTP H∈{1,3}, nested catalog-max K∈{1,2,4,8}
(seed-0 H=8 PAM vocab, identical to D1), all with the C4 `method='scan'` default.
Deliverables: `E3_MEMO.md`, `e3_results.npz`, `fig_e3_primary.png`, `fig_e3_secondary.png`,
`check_hits.py`, `make_e3_figures.py`.

**Verifier recomputations (all MATCH the memo):**
1. *Primary exceedances (126 clean band-LCs, expected 1.26/statistic at p99):* thresholds
   recomputed from the D1 npz — p99 0.5347/0.5509/0.5403/0.5606/0.5695/0.5760 and GEV-99
   0.5293/0.5508/0.5410/0.5563/0.5689/0.5789 for gls(=ftp_H1)/ftp_H3/cat_K1/K2/K4/K8;
   clean counts >p99 = 3/2/4/3/3/3 (GEV: 3/2/4/3/3/2), binomial P(≥obs) 0.13/0.36/0.04/
   0.13/0.13/0.13. `prim_gls` vs `prim_ftp_H1` agree to 2.8e-9. Union: D1 7-statistic
   union rate 2.20% → expected 2.77, observed 5 (binomial P(≥5)=0.146), and the 5
   exceeders are exactly g00014 r, g00029 g, g00037 g, g00044 g, g00082 r. KS clean-vs-D1
   for cat_K8 = 0.140 (p=0.016), clean median 0.4588 vs D1 0.4615 (slightly below, tail
   heavier). Largest clean tail event g00037 g GLS d1-p = 6.7e-4.
2. *Determinism:* for g00014 r the seed formula (500000 + 10·i_obj + band_idx → 500141)
   re-derives the recorded `prim_idx` exactly; `frequency_grid(1,5,10000)` is bit-identical
   to the stored D1 grid; all 7 primary statistics recomputed from the raw cached
   photometry match both the per-unit npz and `e3_results.npz` to 0.0 (exact float match).
3. *Screened hits (verifier's own peak extraction from raw data):* g00017 g full-N peak
   f=4.31343 = 2·f_orb(Chen P=0.46366 d) − 7e-5; N=40 peak 3.31073 = 2·f_orb − 1.00277
   (sidereal-day alias). g00015 g full-N peak f=1.65726 = 2·f_orb(P=1.2068 d) − 1e-5;
   N=40 peak 3.65997 = 2·f_orb + 2.00269. g00052 r N=40 peak f=2.00220 (catK8 and GLS),
   on the ~2 cyc/day diurnal-alias comb. The other 26 screened band-LCs are below the p99
   thresholds in all 7 statistics (screened union = exactly those 3 hits).
4. *Secondary:* full-N powers compared ONLY to the fresh matched null (verified in
   `run_e3.py` aggregate, `make_e3_figures.py`, and the memo table — D1 thresholds are
   never applied to `sec_*`). Recomputed table matches every cell: g p99
   0.0470/0.0505/0.0481/0.0521 with clean exceedances 52/51/50/51 of 53 (KS 0.97/0.96/
   0.96/0.96); r p99 0.0300/0.0348/0.0314/0.0366 with 61/60/61/61 of 73 (KS 0.85/0.86/
   0.84/0.86). The memo's operational conclusion is supported: synthetic Gaussian nulls
   (even N/T/grid-matched) must NOT set full-depth real-data thresholds — the whole real
   sample, clean included, sits far above them (known ZTF excess variance; GLS displaced
   equally with the catalog statistic, so not an FTP/K-axis artifact).
5. *Caveats & metadata:* all four mandatory caveats present in `E3_MEMO.md` — (i) n=99 ⇒
   ~1% FAP floor; (ii) only 8 same_star physical pairs vs 48 same_field_cadence cadence
   nulls, with measured pair separations (verifier recomputed: median 0.249°, max 0.847°,
   matching the memo's 0.25°/0.85°); (iii) D1 single-band N=40 non-transferability, with
   the residual ~0.95 vs 2.28 pts/Rayleigh trials mismatch quantified (verifier: prim_T
   median 2630.8 d → 0.950; D1 1095.75 d → 2.281); (iv) nested-vs-per-K vocabulary bound
   ~0.025. Grid metadata (df, explicit [f_min,f_max]=[1,5], pts/Rayleigh) present for all
   five run configs and recomputed to match (secondary df median 7.324e-5, min 7.281e-5,
   max 1.892e-4; ppr 5.00; n_freq median 54,615, max 54,935). Seed ranges verified unique
   and mutually disjoint (primary 500,000–500,980; matched null 3,000,000–3,000,499 g /
   3,500,000–3,500,499 r; D1 1,000,000+).
6. *Git:* all E3 commits on `dev` and contained in `origin/dev`; committed npz sizes sane
   (`e3_results.npz` 105 KB; `output/` 2.0 MB total, 155 real units + 100 null chunks).

**Minor findings (prose-level, no action required; tables/figures/conclusions unaffected):**
- Memo hit table quotes g00017 g cat_K8 = 0.688; the npz (and verifier recompute) give
  0.6863 → "0.686" (the 0.688 belongs to g00052 r). The quoted d1-p range is consistent.
- Memo secondary prose says "96–98% of clean g LCs"; the actual per-statistic range is
  94.3–98.1% (cat_K1 is 50/53 = 94.3%). The memo's own table is correct.
- `fig_e3_secondary.png` reuses the g-stratum legend counts (n=53/11) figure-wide; the r
  panels' in-panel counts (…/73, …/18) are correct.

## WP F2 — vocabulary-method + sim-validation paper sections (paper repo `h-vs-accuracy`) — DONE 2026-07-11

**Deliverable:** paper commit `95abf57` (pushed). Two new body sections in `paper_v5.tex` —
"Template vocabularies for heterogeneous signal classes" (sec:vocab) and "Validation on
simulated sparse multi-band surveys" (sec:simval, 5 subsections) — plus the alias-breakdown
appendix (app:aliases, Table 4), a robustness-arms table, two new figures
(`plots/recovery_vs_nepochs_sesar.pdf`, `plots/recovery_vs_k_sparse.pdf`), three
ADS/arXiv-verified new refs (Baeza-Villagra+2025 A&A 694 A72 / arXiv:2501.03813;
Kaufman & Rousseeuw 1990 doi:10.1002/9780470316801; Simon & Lee 1981 ApJ 248 291), and
three cross-reference stitches (sec:choiceH forward ref, sec:fap K-wobble sentence,
Discussion item 3 rewritten to point at the new sections).

**Methods text source:** the F2 line-level ground-truth read of `catalog_builder.py` /
`recovery.py` / the `validation.py` scorer seam (unit-Fourier-energy normalization; orbit
distance d^2 = 1 - max CC on the alias-free oversampled FFT grid + Newton polish;
reflections NOT quotiented; PAM BUILD+SWAP on unsquared distances, n_init=10, medoids
physical; greedy = set cover on standalone masks with end-to-end re-score, farthest-point
padding, pam:M pool = lower bound, disjoint selection split; loaders 98 sesar g,r / 560 BV
g,r shapes at common H=8). Code behavior cited over design docs throughout.

**Number provenance (hard rule honored):** every quoted number recomputed THIS session from
the committed `experiments/phase3_recovery/rerun_202606/raw/*/out/{results.json,per_source/*.npz}`
artifacts by the fresh committed script `simval_v5/make_simval_figures.py` (paper repo);
`simval_v5/derived_numbers.json` is the single machine-readable source for the TeX. Key
recomputed values: N-sweep pooled (768/cell, K=2) FTP(PAM) 0.233/0.902/0.984/1.000/0.999/1.000
vs GLS 0.040/0.417/0.889/0.993/1.000/1.000; McNemar FTP-vs-GLS N=8 = 392/19 (p~1e-91), N=4 =
169/21; MHLS==GLS at N=4 mask-identical (B2 cap); sesar sparse PAM K-curve 0.716/0.715/0.738/0.741,
K1-vs-K2 discordant 34/33 (p=1.0), SE-aware knee=1 (frac-rule 4); BV 0.512/0.703/0.727/0.719,
K1->K2 +0.191 (13/62, p=8.4e-9), knee=2; robustness arms per raw jsons incl. xuniv N=16
exception (PAM 0.996 vs GLS 1.000) and greedy adaptivity (0.887 vs 0.695 at N=8); empirical
arm d 0.215/0.918/0.988 at K=4, error-curve ratio recomputed 1.35-3.54x over mags 14.3-18.5;
cost 23.54 s vs 56.25 s = 2.39x at identical recovery 0.34375 (32 src, 1500 freq, n_tau=128);
grid 2.281 pts/Rayleigh (10k) vs 4.563 (20k), paired McNemar 8/30 significant, all gaining,
GLS all p>=0.13, FTP-GLS margin 0.199->0.246 (N=4) / 0.539 both (N=8). Alias table = fresh
`rescore_aliases` phase-coherence pooled re-scoring of the 3 sesar seeds.

**Grid caveat:** isolated in ONE comment-fenced paragraph ("Grid-resolution caveat", end of
sec:simval-nsweep; the only other touchpoint is one cross-reference sentence in the
fig:nsweep caption) — written to be updated in place by the B8-closure rerun.

**Gates:** `make all` exit 0; 24 pp; 0 undefined references/citations; overfull boxes = the
3 pre-existing only (none new). Pages 13-16 + 24 visually inspected.

**Discrepancies found by recomputation (carried, not silently adopted):**
1. `SUMMARY_c-grid20k.md` prose says "GLS/MBLS converged (all p>0.13)" — its own table (and
   this recompute) has MBLS N=12 significant at p=0.0163. Paper text says GLS-only flat and
   counts MBLS among the significant gainers.
2. `SUMMARY.md` acceptance item 1 headline ("FTP>GLS ordering holds in EVERY arm at every
   cell") is contradicted by its own JSON dict (a-sesar-1, b-xuniv-0 false). Paper reports
   pooled contrasts and states the b-xuniv N=16 one-source exception explicitly.
3. Canonical "~2.5x" cost figure: the committed rerun artifact gives 2.389x -> paper quotes
   "measured ~2.4x". The unartifacted "~5x @ n_tau=256" / "~96x @ 3840 obs" docstring figures
   were NOT quoted (no committed artifact to recompute from); replaced by the structural
   n_tau*N_obs argument + cross-ref to the C5-measured single-band timings.
4. `EMPIRICAL_ERROR_REVALIDATION.md` says ratio "1.6-3.5x"; recomputation from the committed
   `ztf_error_curve.json` vs `exp_mag_error` gives 1.35-3.54x -> paper quotes "1.4-3.5".
   [Superseded by numbers-lens finding N2 below: fixed to "1.3-3.5" in `25d1fd1`.]
5. Baseline PDF was already 20 pp before F2; the WP's "~19-20pp" gate figure is stale — F2's
   required content adds 4 pp (24 pp total).

**Adversarial verification (three lenses, sign-off delegated per 2026-06-12 policy):**

*Lens 1 — methods-vs-code (independent agent, journal `wf_11877fc6-7ae`): ISSUES, all fixed.*
Verified against `catalog_builder.py`/`recovery.py`/`template.py`/`validation.py`/
`baselines.py`/`multiband.py`/`simulate.py`: orbit metric (CC of z_k = c_k - i s_k, FFT grid
>= 8H+1 -> power of two, Newton polish of every cyclic local max, reflections not identified),
normalization/ingestion, PAM objective/seeding/restarts, greedy selection seam (tie-break,
stop rules, union-as-signal + end-to-end re-score, disjoint split seed offset 100003), catalog
argmax-over-stack consumption, all harness flags vs committed JSONs, grid-caveat direction.
Findings: (V1, moderate) "g,r subset ... 98 shapes" mis-described the Sesar library — the run
used the FULL ugriz 98 (n_library=98 in `raw/a-sesar-0/out/seed_0.json`; a true g,r subset
would be 45), observations simulated in g,r only; (V2, minor) norm clause "unit-energy shape
has ||f|| = 1" contradicted d^2 = 1 - max CC (plain L^2([0,1]) norm has ||f||^2 = 1/2; the
equation matches the code); (V3, moderate) the paper described the library's farthest-point
saturation padding, but the production K-sweep driver (`run_production_matrix.greedy_order`)
pads post-saturation greedy picks by ASCENDING INDEX (committed greedy_order arrays prove
saturation at 2-3 picks), affecting only the greedy K>knee points of fig:ksweep; plus two
notes (grid-caveat scope slightly narrow — MBLS also gained a cell; +/-2/day beats tested but
unmentioned, zero everywhere — cosmetic, not fixed).

*Lens 2 — completeness-vs-F2-brief (independent agent, same journal): COMPLETE, no gaps.*
All nine required F2 elements present (N-sweep+Wilson+McNemar, K-sweep, robustness arms,
A4-compliant cost wording with the "we quote only the like-for-like in-harness ratio as a
measured result" injunction honored, alias appendix, points-per-Rayleigh, MHLS cap formula,
CE + VdP&I bracketing, isolated grid caveat). Notes: cost is a prose subsection (exceeds a
panel); ~2.4x vs A4's verbatim ~2.5x documented as discrepancy 3 above; 24 pp documented as
discrepancy 5 above.

*Lens 3 — numbers (this session, 2026-07-11; the interrupted verifier re-run from scratch).*
EVERY quantitative claim in the F2 diff recomputed with fresh scripts directly from
`rerun_202606/raw/*/out/per_source/*.npz` + `results.json`/`seed_0.json` (NOT via
`derived_numbers.json`, NOT from `headline_full/` — the paper figure script reads
`rerun_202606` exclusively; its only headline_full mention is the "supersedes" docstring).
Confirmed exact: all pooled N-sweep rates and the five quoted Wilson CIs; McNemar 392/19
p=9.9e-92, 169/21 p=6.4e-30, 5/0 p=0.0625, PAM-vs-greedy 124/43 p=2.7e-10 (and N=8 45/20
p=0.0026, supporting the plural "sparsest cells"); MHLS==GLS at N=4 mask- AND period-identical;
K-sweep 0.716/0.715/0.738/0.741 (34/33 p=1.0; smallest-CI-overlap K=1; 98%-rule K=4; K1->K8
gain +0.0247 <= "~0.03"), BV 0.512/0.703/0.727/0.719 (+0.1914, 13/62 p=8.4e-9, flat after:
p=0.24/0.54), sparse baselines 0.171/0.197/0.090 + BV GLS 0.422; all 27 robustness-table
values exact at 3dp + xuniv N=16 one-source exception (0.9961 vs 1.0000); empirical arm
0.215/0.918/0.988 at fixed_k=4; cost 23.5398/56.2464 s = 2.3894x, recovery 0.34375 both,
32 src/1500 freq/n_tau=128, N_obs<=120 = 60-epoch dense master x 2 bands, "<1e-6" backed by
`ftperiodogram/tests/test_baselines.py:311` (atol=1e-6); grid caveat 8/30 cells (FTP=PAM
counting) significant all gaining, MHLS max +0.168 at N=12, FTP N=8 0.9258->0.9727 +17/-5
p=0.0169, CE 2 / MBLS 1 / GLS 0 (min p=0.1337), margins 0.199->0.246 (N=4) and 0.5391 both
(N=8), grid arithmetic 4.0004e-4 / 2.2813 / 4.5629 / 0.285; MHLS cap formula == 
`ftperiodogram/baselines.py:200-201` exactly; all 18 rows of tab:aliases reproduced exactly
by an independent phase-coherence re-scorer (|df| T < 0.5, named-alias assignment), incl.
689/768 = 0.8971, 0 aliased among FTP's 79 N=8 phase-coherent misses, MHLS N=40 2P+3P =
99/768 and 0.868 -> 0.9974 if credited, |df| T = 3.000 for the year beat.
Findings beyond honest rounding: (N1, minor claim-scope) "year-beat contamination is at most
one source per cell" is FALSE at N=4 — GLS and MHLS each have 3 year-beat recoveries there
(ftp_greedy 1; day beats: 1 at N=4, 0 at N>=8); holds at N>=8. (N2, minor rounding) empirical
error ratio quoted "1.4-3.5" but the committed-curve minimum is 1.3475 (m=16.30) -> "1.3-3.5".
(N3, nit bound-direction) year-beat fractional period error "at most 0.27%" — true sup is
0.2745% at f=1 -> bumped to 0.28% (a stated bound must not round down).

*Fixes:* paper commit `25d1fd1` (h-vs-accuracy, pushed): V1 (full-ugriz library sentence),
V2 (CC(0)=1 normalisation + ||f||^2 = 1/2 clause), V3 (production index-padding disclosed in
sec:simval-ksweep, knee + PAM points unaffected), the lens-1 scope note (grid-caveat lower
bounds "for every method except GLS"), N1 (year-beat claim scoped: <=1 at N>=8, <=3 at N=4),
N2 (1.3-3.5), N3 (0.28%). Rebuild: `make all` exit 0, 24 pp, 0 undefined refs, same 3
pre-existing overfull boxes (none new), all 7 edits render exactly once.

**Grid-caveat anchor (pending B8-closure):** the single point of update remains the
comment-fenced "Grid-resolution caveat" paragraph at the end of sec:simval-nsweep (banner
`% GRID-RESOLUTION CAVEAT -- single point of update`), plus the one cross-reference sentence
in the fig:nsweep caption. The B8-closure dense-grid rerun must update the numbers in that
paragraph only; every absolute sparse-cell rate in sec:simval is quoted "at the 10^4-frequency
production grid".

**Leftover (non-blocking):** the +/-2/day-beat cosmetic omission in app:aliases prose (zero in
every cell, so no number is wrong); "p ~ 1e-29" for the N=4 McNemar is conservative
(recomputed 6.4e-30); sesar K1->K8 "at most ~0.03" is a bound on +0.0247 (fair).

## WP E4 — known-period recovery on the Phase-4 ZTF RRL sample (`experiments/phase4_demo/e4_recovery/`) — VERIFIED 2026-07-11

**Independent adversarial verification (artifacts only — writer scripts not consulted for the
recomputation; scorer, PDM/χ² refold, and Newcombe/Wilson implemented from scratch from the stated
criterion).** Inputs judged: `e4_results.npz` (438 = 73×6 rows), `e4_adjudication.json`,
`e4_adjudication_evidence.json`, `folds/*.png`, `E4_MEMO.md`, `e4_scores.csv` (comparison only),
manifest + raw `~/.ftperiodogram_data/phase4_rrl_sample/lc/*.npz`.

**What ran (writer):** phase 1 prep `7117955`, phase 2 full run `cbd7422`, phase 3
score/adjudication/memo `7be3fd6` — all on `dev`, `dev == origin/dev` at `7be3fd6` (pushed ✓).
All E4 artifacts are tracked; only raw per-block `output/` is untracked (by design).

**(1) Rate recomputation (own scorer: |f−f_t|/f_t < 1% AND |Δf|·T_grid < 0.5, class order
exact → 2f → f/2 → ±1, ±1/365.25, ±1/354.37 c/d → miss; truth = Chen for science, Gaia for
controls; Wilson 95%).** EVERY cell of the memo's pre-adjudication table (§1), post-adjudication
table (§1), class-breakdown table (§4), strata table (§3, with r-medians and joint-epoch counts
recomputed from the RAW LC npz, tercile edges 14.43/15.20 and 1497/1826 reproduced), exact-only
note (mhls 42/58), dual-truth section (§5: median |ΔP|/P 3.61e−5, max 4.79e−4, the 12-value
|Δf|·T list 0.04…3.64, 11/12 < 0.5, 18/72 flips in exactly 3 stars with the stated directions:
181921 0/6→6/6, 095443 0/6→6/6, 100903 6/6→0/6, 094954 both-miss) and both Newcombe intervals
(§2: pre +4.6 pp [−11.5, +33.4]; post +8.3 pp [−0.9, +35.4]) reproduce EXACTLY.
**Per-row comparison vs `e4_scores.csv`: 0 disagreements in 438 verdicts/classes.**
Post-adjudication rule ambiguity (re-score-vs-adjudicated vs additive credit) is moot: both
readings give identical cells everywhere. Fractional-only 1% cross-check confirmed
(ftp_mb 58/58, 11/12, 72/73 pre-adjudication = its post-adjudication cells; ce 72/73, mbls 71/73).
All 438 truth frequencies inside [1,5] ✓. Miss decomposition confirmed: 9 ftp_mb non-recovered
rows = exactly the 9-star adjudication queue (8 near-truth, frac 1.1e−4–4.7e−4; 1 depth failure
frac 0.335); mhls extra subharmonics = 4 non-queued science (3× f/3, 1× f/4); outside-queue
wrong peaks gls ×1 (0.57 cyc), ftp_1band ×3, mbls_h1 ×1, ce ×0; CE's extra science recovery is
0.4983 cyc on ZTFJ194003 (threshold luck, as stated).

**(2) Science vs anti-join** re-derived (above, exact). Hardness claims verified from raw LCs:
anti-join median joint epochs 118 vs science 1758; anti-join r_med max 20.2; science r_med
12.9–17.8; failure star g/r = 20.4/20.2. Manifest dedupe verified independently: 85 entries,
73 unique byte-contents, exactly 12 byte-identical control↔science pairs (md5), science types
30 RRab/28 RRc, anti-join 11 RRab/1 RRc, 55/58 science Gaia cross-periods.

**(3) Adjudication spot-checks (4 calls; the record contains 8 lawful + 1 failure, so 3 lawful +
the single failure were checked — a 2nd failure does not exist):**
* `chen_f486_ZTFJ180708.63+013433.4` (lawful, Chen-error-Gaia-confirmed) — **refolded from the raw
  npz by the verifier**: own weighted 25-bin PDM θ = 0.7272 (Chen) / 0.0387 (recovered) / 0.0451
  (Gaia pf) and fold χ²_red = 56.83 / 3.03 / 3.52 match the writer's evidence to every quoted
  digit; own fold figure reproduces the committed one (truth smeared, recovered = textbook RRab,
  recovered ≡ Gaia pf within 0.007 cyc). Call sound.
* `chen_f786_ZTFJ094954.90+514422.2` (lawful, Blazhko/period-change) — top-5 close-pair
  Δf = 5.37e−4 c/d → P_mod ≈ 5.07 yr recomputed from npz top-5 ✓; fold plot shows clean RRab at
  the consensus period with max-light scatter vs both catalog folds smeared (θ 0.249 vs
  0.916/0.898); Chen–Gaia mutually 0.31 cyc but both ~0.86 cyc from consensus ✓. Call sound.
* `gaia_f486_4276113046713707904` (lawful, Gaia-truth precision) — 6-method best-freq spread
  7.3e−5 c/d (unanimous) ✓; fold contrast θ 0.368 vs 0.860 evident in the committed plot. Sound.
* `gaia_f786_1020098816344918656` (failure, photometric depth) — all 6 methods on the window comb
  (best freqs 1.0002–1.0025; top-5 = 1.0002/1.0025/2.0030/1.0031/2.0025) ✓; truth fold pure noise
  (θ 0.922); mags 20.4/20.2 verified from raw LC. Correctly charged as a genuine failure.

**(4) Grid metadata** — f_min = 1, f_max = 5 on all 438 rows (explicit bounds, never
nyquist_factor); df = 1/ceil(1/(0.2/T_grid)) EXACTLY on all rows (snap ≤ 0.2/T everywhere);
df ∈ [7.22e−5, 1.115e−4] (memo's 1.11e−4 = correct 3-s.f. rounding of max 1.1148e−4);
T_grid ∈ [1794, 2769] d; pts/Rayleigh ∈ [3.66, 5.00] with ppr = 1/(df·T_band) on single-band rows
and 1/(df·T_joint) on joint rows ✓; worst df·T_joint = 0.273 cyc < 0.5 ✓. T_grid (per-band /
shorter-band), T_joint (union span) and n_obs verified against the raw LC npz for **all 73 stars /
438 rows: 0 mismatches** after accounting for the sample's single non-finite epoch (one NaN-HJD
r-band point in ZTFJ181921.70+032211.3, correctly dropped by the runner; not mentioned in the
memo — harmless).

**(5) Commits** — `7117955` / `cbd7422` / `7be3fd6` on `dev`, pushed (dev == origin/dev) ✓.

**ESCALATE rule outcome:** recomputed pre-adjudication best multiband on science = CE 89.7%
(ftp_mb & mbls_h1 87.9%, all joint g+r methods) ≥ 85% ⇒ **no escalation**, matching the writer.

**Scoring disagreements: NONE (0/438).** **Verdict: CONFIRMED.** Non-blocking notes: (a) the
verification brief's "2 failure spot-checks" was unsatisfiable (1 failure exists); (b) the NaN-HJD
epoch drop above; (c) memo intro says baselines "T ≈ 4.9–7.6 yr" (grid T range) while the sample
doc quotes 7.1–7.6 yr joint spans — both true, different definitions (shortest single-band grid
baseline vs joint span).


---

## WP B8e-prep — E-step acceleration verification (2026-07-12)

**Scope.** Adversarial verification of the two E-step accelerations in `ftperiodogram/joint_em.py` (commits 5b8196b sums-cache, e791d3d warm-start refine, 37b3069 driver plumbing, 04a4c76 tests): (1) SUMS-CACHE, claimed EXACT/bitwise; (2) WARM-START REFINE (`estep_refine`, default OFF), an approximation for speed. Method: independent from-scratch oracles driving ONLY the public multiband API (`compute_band_summations`/`multiband_template_periodogram`/`solve_over_frequencies`), never certifying via the committed `test_estep_cache.py`. Env pinned `VECLIB/OMP/OPENBLAS/MKL=1` throughout; NFFT-valid grids via `frequency_grid`.

**Overall verdict: ESCALATE.** Two approximation claims (RF-4, RF-5) survived independent refutation. The EXACT sums-cache — the accelerator that UNCONDITIONALLY gates arm-a — is fully CONFIRMED. Both surviving defects sit on the `estep_refine` path, which is OFF by default and, on arm-a, additionally screened by the driver being hardwired to `floating_offsets` (the one mode where even refine-on holds bitwise). Neither surviving defect is known to touch the default shipped vocab; they are escalated (not self-signed-off) per the standing C-track delegation because defects survived refutation and the real launch-command flag state is not confirmable from the repo.

### Claim ledger — per-claim verdict (worst numbers)

**SUMS-CACHE (exact / bitwise) — CONFIRMED, unconditional arm-a gate**
- SC-1 CONFIRMED. `_compute_source_sums(source, freqs, mode, H)` (joint_em.py:149) takes no template and no iteration arg; forwards to `compute_band_summations(t,y,bands,freqs,H,dy,mode)` which reads only data+(freqs,mode,H). Lens: after mutating the template bank 3x, cached `Summations` bytes byte-identical and the mutated-bank E-step matched the from-scratch oracle bitwise. Worst: 0 / bitwise.
- SC-2 CONFIRMED. `Summations` is an immutable namedtuple; `YM_MM_from_sums` (core.py:23) is read-only (only `+=` targets a locally-allocated MM). Lens: byte-snapshot of a cached `Summations` unchanged across iterations with changing templates; `id()` reused. Worst: 0 mutations / 0 byte drift.
- SC-3 CONFIRMED. cache on vs off bitwise-identical. Lens: 45 e2e configs (3 modes × H{1,4,8} × 5 geometries) 0 fails; production scale nf=2000 and one nf=4000 cache-parity NONE. Oracle validity: NFFT-fast vs direct two-pass agree 8.4e-11 (H=1)…1.4e-12 (H=8) with zero argmax disagreement; executor E-step == oracle bitwise (max|dpower|=0). **My re-measurement:** cache on vs off, max_iter=3, modes {floating_offsets, shared_phase, independent}: assign/freq/power hist + val_signal bitwise-equal, vocab_dev=0.0, cache_hits 40(on)/0(off) each. Worst: 0 / bitwise.
- SC-4 CONFIRMED. Serial vs n_jobs=2 (spawn per-worker cache) bitwise-equal traces; parallel cache hits == serial; val_exec is a distinct object when val≠train (joint_em.py:862), transductive val (val=None) shares train deliberately and remains bitwise. Worst: 0 diffs.
- SC-5 CONFIRMED. `build_joint_em_catalog` default `cache_sums=True` (verified via `inspect.signature`); driver never passes `cache_sums`, so the cache is active on the release path (call at run_joint_vs_pipeline.py:118-122). CLI artifact: `em_cache_hits>0`, `em_cache_misses=2*n_sources`, counters reproduced exactly from the touch schedule.
- RF-1 CONFIRMED (window-subset within-grid exactness). `_subset_sumlists` (joint_em.py:156) shares the same `Summations` objects at positions `idx` (no copy/mutate); `solve_over_frequencies` reads frequency-independent stats. Lens: real-solver subset vs full[idx] bitwise across floating_offsets/shared_phase/independent × H{4,8}, plus 72 real-power `_refine_pair_solve` cases with 46 edge-escapes/10 fallbacks firing, |power(returned) − full[returned]| < 1e-12. Worst: 0 / bitwise.

**WARM-START REFINE (approximation, default OFF) — structural guards CONFIRMED; two approximation claims DEFECT**
- RF-2 CONFIRMED (last recorded E-step always full-grid). `refine_it = estep_refine and 2<=it<max_iter and prev_states`; in-loop final it==max_iter is full-grid, and an early stop on a windowed iteration triggers the post-loop full-grid rerun (joint_em.py:964-967). Across 18 configs `solve_fraction[-1]==1.0`, `refine_used[-1]==False`, incl. an early-stop-on-windowed run with `final_full_grid_rerun=True` at n_iter=3/max_iter=4. Worst: 0 violations.
- RF-3 CONFIRMED (rerun uses the correct stopped-iteration bank). `last_estep_vocab = vocab` set at the top of each iteration (line 902) before the M-step overwrites `vocab` (line 943). Lens captured (deep-copied) the exact bank of the final recorded E-step; independent oracle reproduced `assign_hist[-1]`/`freq_hist[-1]` bitwise and `power_hist[-1]` to max|Δ|=0 in all 18 configs. Worst: 0 mismatches.
- RF-6 CONFIRMED (degenerate inputs fall back to full grid). `if not (T>0) or prev_peaks.size==0: fallback=1; return _full()`. Lens: T≤0 and empty peaks both solve the whole grid and return the true argmax; a 20000-case fuzz confirmed the initial window mask is never empty for T>0 with nonempty grid-index peaks (so the `idx.size==0` branch is unreachable dead code); 10/10 real-power fallbacks returned the global peak. Worst: 0 bad.
- RF-7 CONFIRMED (refine opt-in / off by default). `--estep-refine` is `action='store_true'` with no default (run_joint_vs_pipeline.py:315), passed `estep_refine=args.estep_refine` (line 121); signature default `estep_refine=False` (verified via `inspect.signature`). CLI without the flag: `em_solve_fraction` all 1.0, `em_edge_escapes/em_fallbacks` all 0, `em_final_full_grid_rerun=false`. Driver also hardwires the scorer mode to `floating_offsets` (line 87, no mode flag exists). Worst: 0 mismatches.
- **RF-5 DEFECT (approximation claim 2c overstated).** Guards prevent silent global-peak misses ONLY when the windowed argmax lands on a window BOUNDARY. An interior local max inside a warm-start window, with the true global peak outside all windows, is returned with edge_escape=0, fallback=0. Reproduced on the REAL solve path (not injected): silent-miss rate 197/400=49.3% (floating_offsets), 218/400 (shared_phase), 213/400 (independent), worst power gap 0.3275; injected control-flow miss gap 0.40 (44% of the true peak) at n_solved=12/400. Classification: documentation-accuracy — the production docstring wording "prevent silently missing the global peak" must be corrected to "reduce but do not eliminate." Not wrong-science under refine-off; escalated because it survived refutation.
- **RF-4 DEFECT (approximation claim 2b false outside floating_offsets).** "refine=on FINAL assignments+freqs identical to refine=off, powers ≤1e-12, vocab matches" holds only if every intermediate windowed E-step reproduces the full-grid best per source so the gated M-step/vocab/val/best_iter trajectory never diverges — which fails when the M-step moves a template and a pair's global peak leaves all windows (an RF-5 interior silent miss). **My independent re-measurement (jitter=0.30, max_iter=patience=5, nfreq=128, H=4):** `floating_offsets` vocab_dev=0.0, final_assign_eq=True, final|dpow|=0.0 (claim holds); `independent` final_assign_eq=**False**, final|dpow|=**2.071e-3** (≫ claimed 1e-12) — claim 2b FALSE. Lens/refutation corroboration: `shared_phase` vocab_dev=0.3126 (deterministic on repeat), best_iter 0(on) vs 3(off), final|dpow|=6.97e-3; `independent` vocab_dev=0.539, best_iter 5 vs 3, 5/7 seeds violate. The mandatory final rerun makes the last RECORDED trace full-grid on an already-divergent bank, so the narrow docstring survives but 2b does not. Escalated: do NOT certify claim 2b as a four-mode statement; do NOT sign off any `--estep-refine` run in shared_phase/independent (or untested sesar).

### Modes / H / geometries / parallelism covered
- Modes: floating_offsets, shared_phase, independent exercised at cache/refine/EM level. **sesar NOT exercised** (arm-a-unreachable: driver hardwires floating_offsets and the recovery scorer supplies no relative_offsets for sesar).
- H ∈ {1,4,8}; grids 128 (unit gates) up to nf=2000/4000 (cache spot-check); max_iter 2–6; geometries dense N=12, sparse N=6, single-band, low-amp 0.05–0.12, intrinsic_jitter 0.20–0.30; parallelism n_jobs=1 and n_jobs=2 (spawn per-worker cache).
- My session re-measurements: cache on/off bitwise parity max_iter=3 in 3 modes (vocab_dev=0, cache_hits 40/0); RF-4 refine on/off in 3 modes (floating_offsets bitwise-identical; independent divergent |dpow|=2.071e-3); code defaults `cache_sums=True`/`estep_refine=False` via `inspect.signature`; driver plumbing lines 87/121/315 read directly.

### Residuals / caveats
- **shared_phase one-sidedness (pre-existing WP C3.5):** shared_phase carries a documented scan≥eigvals contract at deep |MM| dips. The RF-4 refutation confirmed the divergences it found are NOT that residual — refine on/off use the identical shared_phase scan fit (common-mode, cancels); the only difference is which frequencies the windowed search evaluates. Not misattributed.
- The oracles' only apparent sub-1e-15 gaps traced to Template re-normalization to unit Fourier energy when re-wrapping `c_n/s_n`; the no-rewrap oracle gives exactly 0/0.
- Diagnostics array-length coupling (DIAG) is real but benign and refuted as a gating defect: `val_signal` length n_iter+1 (index 0 = pipeline-init, documented joint_em.py:73-75), `n_gated/n_reverted` length n_iter, estep arrays length n_iter (+1 iff final_full_grid_rerun — trailing element is the mandatory rerun, not a distinct iteration); `solve_fraction` is a WORK ratio and can exceed 1.0 for a fallback-heavy windowed iteration. No committed consumer misaligns these. Recommend a one-line F3 methods note.

### Bottom line
- **DEFAULT arm-a release path (cache_sums=True, estep_refine=False, mode=floating_offsets): correctness fully established.** The sums-cache is bitwise-exact across every tested mode/scale/parallelism and unconditionally gates arm-a; refine is off; even were refine on, floating_offsets holds bitwise.
- **ESCALATE for John's disposition:** (i) confirm the RunPod launch omits `--estep-refine` (not visible in-repo); (ii) RF-4 — do not certify claim 2b for shared_phase/independent/sesar and do not sign off any refine-on run in those modes; (iii) RF-5 — fix the "prevent silently missing the global peak" wording to "reduce but do not eliminate." `arm_a_cleared=false` is set per the mandatory rule (defects survived + launch flag unconfirmed), not because the default path is known-broken.

### Disposition status (2026-07-13)

**ESCALATE fired -> NOT signed off; arm-a paid release NOT cleared to launch pending John's disposition** (standing C-track delegation). The DEFAULT arm-a path is correctness-clean; the block is procedural (2 defects survived refutation on the refine path + the operational launch flag is not in-repo).

**Disposition items:**

1. RF-4 (approximation claim 2b) SURVIVES refutation. Claim 'estep_refine=on -> FINAL assignments+freqs IDENTICAL to refine=off and powers match to <=1e-12 (and vocab matches)' is FALSE outside floating_offsets. I independently reproduced a divergence in mode='independent' (jitter=0.30, max_iter=patience=5, nfreq=128, H=4): final recorded assignments DIFFER between refine on/off and final |dpow|=2.071e-3, which is ~9 orders of magnitude above the claimed 1e-12. The refutation lens additionally recorded shared_phase vocab_dev=0.3126 (deterministic), best_iter 0(on) vs 3(off), final|dpow|=6.97e-3, and independent vocab_dev=0.539 with 5/7 seeds violating. Root cause: warm-start windows are seeded from the PREVIOUS iteration's peaks; when the M-step moves a template the current global peak of a (source,template) pair can fall outside all windows with an interior windowed argmax, so no edge-escape/fallback fires (an RF-5 silent miss) and the trajectory/gated M-step/val curve/best_iter/returned vocab diverge. The mandatory final full-grid rerun makes the last RECORDED trace full-grid on an already-divergent bank, so the narrow docstring wording ('never come from a stale window') holds but the stronger 2b ('identical to refine=off') does not. DISPOSITION NEEDED: do NOT certify claim 2b as a four-mode statement; do NOT sign off any --estep-refine run in shared_phase/independent (or untested sesar). floating_offsets holds bitwise (my run: vocab_dev=0.0, assign_eq=True, dpow=0.0), so the arm-a default mode is unaffected.

2. RF-5 (approximation claim 2c) SURVIVES refutation. The handoff/docstring wording 'edge-escape + fallback prevent the warm-start from silently missing the global peak' is OVERSTATED/false. The edge-escape guard fires ONLY when the windowed argmax lands on a window BOUNDARY; a genuine interior local maximum inside a warm-start window is returned with edge_escape=0, fallback=0 even when a strictly larger global peak lies entirely outside every window (top-k prev peaks + 2f/0.5f aliases). Two independent lenses reproduced this on the REAL solve path (not just injected solvers): real-periodogram silent-miss rate 197/400=49.3% (floating_offsets), 218/400 (shared_phase), 213/400 (independent), worst power gap 0.3275; injected control-flow case missed a peak with power gap 0.40 (44% of the true peak) at n_solved=12/400. DISPOSITION NEEDED: this is a documentation-accuracy defect (production CODE docstrings that claim only the final-rescan exactness are accurate). Correct the wording to 'reduce but do not eliminate.' Not wrong-science under the default (refine-off) path, but it survived refutation so it is escalated per the standing delegation rather than self-signed-off.

**Completeness gaps (recorded for honesty):**

COMPLETENESS GAPS a rigorous sign-off should note (factored honestly into the verdict):

1) sesar MODE NEVER EXERCISED. The brief lists four modes to exercise (floating_offsets, shared_phase, sesar, independent) and multiband.MODES includes 'sesar', but no lens ran cache-parity or refine-parity or an EM at the 'sesar' setting. Scope/mitigation: 'sesar' is NOT reachable on the arm-a path — the driver hardwires make_recovery_scorer with no mode arg (run_joint_vs_pipeline.py:87 -> scorer default floating_offsets; there is no CLI flag to change the mode), and make_recovery_scorer never supplies the relative_offsets that mode='sesar' requires (validation.py:397, multiband.py:231-233), so sesar cannot be driven through the recovery seam. So sesar-cache/sesar-refine parity is UNVERIFIED but also arm-a-unreachable. Given RF-4 breaks refine parity in the two OTHER non-default modes, a hypothetical future sesar+refine run should be treated as unverified/plausibly divergent.

2) ACTUAL RunPod LAUNCH FLAG STATE NOT IN-REPO. The real arm-a invocation is not committed, so no verifier can confirm whether the operator passes --estep-refine. This is THE determinant of whether RF-4/RF-5 are release-gating. By code default refine is OFF and the arm-a driver only enables it via that opt-in flag, but the launch command itself must be confirmed operationally before any refine-on run is trusted. This unconfirmed flag state is part of why arm_a_cleared is not self-certified.

3) RF-4 PRODUCTION-SCALE MULTI-MODE MAGNITUDE UNCHARACTERIZED. Divergence is CONFIRMED at modest scale (nfreq=128, H=4), but the RF-4 lens's full 120-config sweep did not complete and refine-on-vs-off parity for shared_phase/independent at production scale (H=8, nf=2000-4000) was not measured. Moot for the verdict (claim 2b already fails), but the divergence magnitude at scale is unquantified.

4) PARALLELISM ONLY AT n_jobs=2. Per-worker spawn-pool cache invariance and serial==parallel bitwise were confirmed at 2 workers only; higher worker counts / alternate chunking untested. Low risk (per-worker cache, independent sources, order-preserving pool.map).

5) NO ARM-A-SCALE CLI CACHE ON/OFF BITWISE DIFF. The production CLI exposes no cache toggle; cache parity was verified at module level (with an nf=2000/4000 spot-check), not via the exact production CLI params in a single cache-on/off comparison. Low risk (the cache only skips a deterministic recompute).

6) DIAG-LENGTH-COUPLING oracle did not complete a live run; its array-length relationships were derived from reading the append sites in joint_em.py and corroborated by the committed assert (test_estep_cache.py:198). Benign (refuted as a gating defect); recommend a one-line note in the F3 methods text that em_solve_fraction/edge_escapes/fallbacks length = n_iter (+1 iff final_full_grid_rerun, the trailing element being the mandatory rerun, not a distinct iteration), val_signal length = n_iter+1, and that solve_fraction is a WORK ratio that can exceed 1.0 for a fallback-heavy windowed iteration.

### Disposition RESOLVED (2026-07-13, John)

John chose the **prose-only Paper-1 finish**: the paid **arm-a joint/EM release
is NOT run**, so the two surviving refine defects are moot for the paper (refine
is default-off; arm-a would have used `floating_offsets`, where it is bitwise
anyway). Escalation closed on the code side:
- **RF-4** — `build_joint_em_catalog` now RAISES `ValueError` for
  `estep_refine=True` outside `mode='floating_offsets'` (`joint_em.py`; new test
  `test_refine_refused_outside_floating_offsets`; `test_estep_cache` 10/10).
- **RF-5** — window-geometry comment corrected to "reduces but does not
  eliminate" + points to the mandatory final full-grid rescan (RF-2) as the
  correctness guarantee. No production docstring over-claimed (the over-claim
  was in the verification brief/handoff, not the code).
- **Sums-cache** (the unconditional gate) stands CONFIRMED bitwise and is
  untouched. F3 will report joint/EM as the honest negative-result deferral
  (Paper 2 = full joint), so no `--estep-refine` run is gated on this.
