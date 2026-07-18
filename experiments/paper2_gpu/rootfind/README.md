# Batched Aberth–Ehrlich rooter — validated, ready for the GPU port

Batched, vectorized replacement for the per-frequency companion-matrix
`numpy.polynomial.Polynomial.roots()` (LAPACK `eigvals`) used in
`ftperiodogram.core.roots_from_YM_MM` / `_exact_root_fallback`. Fully vectorized
over the polynomial axis (Python loops only over degree, never over polynomials).

## Validation (see rootfinders.py / hybrid.py / retime.py)
On real FTP stationarity polynomials (degree D = 6H−2), FP64, Apple M5:
- **Speed**: ~3.2× vs the per-poly numpy loop at H=8 (D=46); grows with degree.
  Batched-eigvals-on-a-stack gives ~1× (numpy already loops in C internally);
  Durand–Kerner is ~8× slower and unreliable — both rejected.
- **Accuracy**: selected phase/power matches numpy roots to ~2e-15 on every
  case incl. rank-deficient N<2H+1 data; 100% convergence, 0 fallbacks on real
  FTP polynomials. `batched_roots` keeps a per-row numpy fallback for any
  non-converged row (provably bitwise-identical to today when it fires).
- Key tuning: seed the initial root circle OFF |z|=1 (inflate=1.2) — halves the
  iteration count (FTP roots cluster near the unit circle).

## Why it is the GPU rooter, and NOT yet wired into the CPU fallback
- **GPU (decisive)**: CuPy/cuSOLVER has **no batched non-symmetric eigensolver**,
  so `cupy.linalg.eigvals` loops per matrix. Aberth is the only candidate that
  batches cleanly (elementwise complex arithmetic + reductions). The GPU port's
  deep-|MM|-dip rows must route through this.
- **CPU (deferred, on purpose)**: the CPU fallback fires precisely at deep-|MM|
  dips, where the *conditional* leading-coefficient trim (C3 finding FO-TRIM-1,
  see VERIFICATION.md) is load-bearing. That per-row conditional trim makes the
  stationarity polynomials variable-degree, which is fundamentally incompatible
  with fixed-degree batched rooting — padding the trimmed zero back reintroduces
  the spurious huge-modulus root the trim exists to remove. Wiring Aberth into
  `_exact_root_fallback` would risk reintroducing FO-TRIM-1 for a ~20% gain on
  the well-conditioned griz target (which defers ~0% anyway). Not worth it on
  the verified CPU path; essential and clean on the fresh GPU path.

## Integration entry point (GPU)
`batched_roots(coefs)` — coefs (nf, D+1) increasing order (as
`core.batched_stationarity_coefs` returns) -> roots (nf, D). Feed straight into
the existing project-to-unit-circle + argmax selection.
