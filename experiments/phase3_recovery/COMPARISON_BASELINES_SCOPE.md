# Next unit — comparison-method baselines (scope, not yet built)

The recovery-vs-K / recovery-vs-N_epochs figures need the comparator curves from the
`phase3_vocab_design.md` §2 "comparison methods" table drawn alongside FTP. Two are
already in hand; the rest need **new** implementations because gatspy / astroML /
astropy were intentionally dropped from the dependency set. This document scopes that
unit and flags the dependency decision — it does **not** implement anything.

## The rigidity spectrum (design §2)

| Method | Shape DOF | Role | Status |
|---|---|---|---|
| **GLS** (Zechmeister & Kürster 2009) | H=1 sinusoid + floating mean | sinusoidal **lower bound** | ✅ have — `Template([1.0],[0.0])` via the FTP path |
| **FTP (K templates)** (this work) | 3 (phase, amp, offset)/template | the method | ✅ have — `RecoveryScorer` |
| **MHLS / multiharmonic LS** (Schwarzenberg-Czerny 1996) | 2H free amps+phases | free-shape **upper bound** (overfits sparse data) | ❌ new |
| **Multiband LS** (VanderPlas & Ivezić 2015) | per-band Fourier, shared period | the *fair* sparse-multiband LS **competitor** | ❌ new |
| **Sesar-style non-linear fit** (Sesar 2017; Stringer & Long 2019) | same templates, slow optimizer | **gold standard** (the <1e-6 brute-force oracle) | ❌ new |
| **BLS** (Kovács et al. 2002) | boxcar | off-Fourier-axis control / EB flag | ❌ new (optional / deferrable) |

## Dependency decision (the flag)

**Recommendation: implement all baselines fresh in a new `ftperiodogram/baselines.py`
using only the existing numpy/scipy stack — do NOT pull gatspy/astroML/astropy.**
Rationale:

- GLS, MHLS, multiband LS are **linear** least-squares: build a per-frequency design
  matrix of sin/cos harmonics (+ per-band offset columns for multiband) and solve via
  `numpy.linalg.lstsq`; the periodogram is the explained-variance fraction. No
  external dependency, and it matches the harness's "stdlib + numpy/scipy core" rule.
- The Sesar oracle uses `scipy.optimize.least_squares` (scipy is already a core dep).
- gatspy's `LombScargleMultiband` would re-introduce a dropped, partly-unmaintained
  dependency for ~40 lines of `lstsq` we can own and test. Not worth it.

This is a **non-trivial new module** (~300–400 lines + tests + a validation that the
linear baselines match a brute-force `<1e-6` reference on a clean injection), so it is
gated behind explicit go-ahead, not bundled into the current figures unit.

## Suggested integration seam

Refactor `RecoveryScorer` to score a **pluggable period estimator** over its frozen
population: `estimator(t, y, bands, dy, freqs) -> P_rec`. Then *every* method —
including FTP and GLS — becomes one estimator scored on the **identical** frozen
sources and grid, which is the fairest possible comparison and lets the figure
helpers overlay all curves from one sweep. Sketch:

```python
# ftperiodogram/baselines.py
def gls_estimator(...)            # H=1 LS (cross-check vs current FTP@H=1 baseline)
def mhls_estimator(n_harmonics)   # free 2H amps+phases -> upper bound
def multiband_ls_estimator(...)   # shared-period per-band Fourier (VdP & Ivezic 2015)
def sesar_oracle_estimator(templates)  # scipy.optimize per template; SLOW, timed
def bls_estimator(...)            # boxcar; optional

# validation.py (the seam)
class RecoveryScorer:
    def score_estimator(self, estimator) -> (rate, mask): ...
```

Cost/value notes:
- **Sesar oracle** is deliberately slow (per-frequency, per-template non-linear fit) —
  it is what deliverable (iii)'s cost-vs-accuracy panel times FTP against (the ~10³×
  speedup claim). Keep it brute-force; never loosen its tolerance.
- **MHLS** is the headline foil: FTP's learned shape prior should beat free-shape MHLS
  *especially at low N_epochs* (the recovery-vs-N_epochs panel).
- **BLS** is a contaminant/EB control, not central to recovery-vs-K — safe to defer to
  the variable-TYPE confusion-matrix unit.
