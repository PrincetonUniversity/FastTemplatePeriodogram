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
