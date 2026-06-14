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
- **Gates 1–5 pass** (test_scan_polish.py): (1) scan≡eigvals max|ΔP| ≤ ~1e-15 over H∈{1..10}×5
  seeds×N∈{30,300}, identical argmax; (2) Issue-#33 fixtures on the scan path; (3) adversarial
  rank-deficient vs the 2^16 oracle; (4) weight-conditioning within 2× of eigvals; (5) speedup
  **26.5× @H=2, 20.2× @H=8 (≥15× required), 22.6× @H=10**, max|ΔP| ~1e-15.
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
