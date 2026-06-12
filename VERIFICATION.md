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
