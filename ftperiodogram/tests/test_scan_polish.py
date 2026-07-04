"""Acceptance gates + regression tests for the scan+polish maximizer (WP C2).

``method='scan'`` replaces the per-frequency stationarity-polynomial
root-finding with an FFT circle scan over ``max(128, 32H)`` uniform angles
plus Newton polish of every bracketed circular local maximum, on the
batched coefficient stacks of WP C1. The WP C2 acceptance gates pinned
here:

1. scan vs eigvals: max |dP| <= 1e-12 over H in {1..10} x 5 seeds x
   N in {30, 300}, AND identical argmax frequency;
2. the Issue-#33 regression fixtures pass on the scan path;
3. adversarial set (rank-deficient N in {8,12,25} x H in {8,10},
   spiky/sawtooth templates) vs a 65,536-point brute-force circle oracle:
   zero misses, |dP| <= 1e-12;
4. concentrated-weight cases (1e4/1e8/1e12 single-dominant-point): the
   scan path's error vs exact GLS is within 2x of the eigvals path's
   (the precision floor lives in the sums; the scan must not worsen it).

The speedup half of gate 5 is measured by
``experiments/benchmarks/benchmark_scan_polish.py`` (numbers recorded in
the WP commit message), not asserted here -- wall-clock asserts are flaky
under CI load.
"""
import numpy as np
import numpy.polynomial as pol
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar

from ..baselines import GLSEstimator
from ..core import (YM_MM_from_sums, scan_polish_YM_MM, scan_polish_from_coefs,
                    roots_from_YM_MM, template_periodogram)
from ..modeler import FastTemplatePeriodogram, TemplateModel
from ..multiband import FastMultibandTemplatePeriodogram
from ..summations import direct_summations
from ..template import Template
from ..utils import weights
from .test_modeler import _PRECISION_FLOOR_CASES, direct_periodogram
from .test_weight_conditioning import (H1_CN, H1_SN, dominant_point_data,
                                       regular_grid)

GATE_TOL = 1e-12
GATE_HARMONICS = list(range(1, 11))
GATE_SEEDS = [0, 1, 2, 3, 4]
GATE_NDATA = [30, 300]


def _simulate(H, N, seed, nf=64):
    """Random template + noisy template signal + a regular frequency grid
    (same generator as test_batched_assembly, kept local for independence)."""
    rng = np.random.RandomState(seed)
    template = Template(rng.randn(H), rng.randn(H))

    T = 10.0
    period = 0.77
    t = np.sort(T * rng.rand(N))
    dy = 0.05 * (1 + rng.rand(N))
    y = 1.5 * template((t / period) % 1.0) + dy * rng.randn(N)

    df = 0.05
    freqs = df * (10 + np.arange(nf))   # dnf = 10; spans ~0.5 - 3.65 c/d
    return t, y, dy, template, freqs


# ----------------------------------------------------------------------
# Gate 1: scan vs eigvals over the full (H, seed, N) matrix
# ----------------------------------------------------------------------
@pytest.mark.parametrize('N', GATE_NDATA)
@pytest.mark.parametrize('H', GATE_HARMONICS)
def test_gate1_scan_matches_eigvals(H, N):
    """max |P_scan - P_eigvals| <= 1e-12 AND identical argmax frequency."""
    for seed in GATE_SEEDS:
        t, y, dy, template, freqs = _simulate(H, N, seed)
        p_ref, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                        freqs, fast=True)
        p_scan, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                         freqs, fast=True, method='scan')
        assert float(np.max(np.abs(p_ref - p_scan))) <= GATE_TOL
        assert int(np.argmax(p_ref)) == int(np.argmax(p_scan))


@pytest.mark.parametrize('H', [1, 3, 8])
def test_scan_matches_eigvals_direct_sums(H):
    """Same equivalence on the direct (fast=False) summation path."""
    for seed in (0, 1):
        t, y, dy, template, freqs = _simulate(H, 30, seed)
        p_ref, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                        freqs, fast=False)
        p_scan, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                         freqs, fast=False, method='scan')
        assert float(np.max(np.abs(p_ref - p_scan))) <= GATE_TOL
        assert int(np.argmax(p_ref)) == int(np.argmax(p_scan))


# ----------------------------------------------------------------------
# Gate 2: Issue-#33 regression fixtures on the scan path
# ----------------------------------------------------------------------
@pytest.mark.parametrize('label,c_n,s_n,period', _PRECISION_FLOOR_CASES)
def test_gate2_issue33_fixtures_on_scan_path(label, c_n, s_n, period):
    """The #33 precision-floor fixtures (test_modeler.py), evaluated through
    the scan path against the same brute-force truth, at the same 1e-6 bar.
    Do not silently widen this tolerance."""
    rng = np.random.RandomState(42)
    N = 80
    T = 20.0
    yerr_val = 0.02
    t = np.sort(rng.uniform(0, T, N))
    omega = 2 * np.pi / period
    y = sum(c * np.cos((n + 1) * omega * t) + s * np.sin((n + 1) * omega * t)
            for n, (c, s) in enumerate(zip(c_n, s_n)))
    y = y + yerr_val * rng.randn(N)
    yerr = np.full_like(y, yerr_val)

    tmpl = Template(c_n, s_n)
    test_freqs = np.linspace(0.4, 2.5, 12)
    # linspace freqs are not on a df grid: use the direct-sums scan path
    p_scan, _ = template_periodogram(t, y, yerr, tmpl.c_n, tmpl.s_n,
                                     test_freqs, fast=False, method='scan')
    truth = np.array([direct_periodogram(f, tmpl, (t, y, yerr))
                      for f in test_freqs])
    assert_allclose(p_scan, truth, atol=1e-6, rtol=0,
                    err_msg=f"Issue #33 scan-path regression for {label}")


# ----------------------------------------------------------------------
# Gate 3: adversarial cases vs a 65,536-point brute-force circle oracle
# ----------------------------------------------------------------------
def _circle_oracle_power(YM, MM, YY, n_angles=1 << 16):
    """Brute-force max of Re(YM(phi)^2 / MM(phi)) / YY over the unit circle:
    a dense 2^16-point grid, then bounded scalar refinement of the best
    bracket AND of every deep local minimum of |MM| (narrow peaks of P hide
    inside |MM| dips and can be missed by any uniform grid -- the C2
    adversarial verification caught the original best-bracket-only oracle
    sharing the scan's blindness)."""
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    phi = np.exp(1j * theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.real(YM(phi) ** 2 / MM(phi)) / YY
    P[~np.isfinite(P)] = -np.inf
    i = int(np.argmax(P))
    if not np.isfinite(P[i]):
        return 0.0

    absM = np.abs(MM(phi))
    ismin = ((absM <= np.roll(absM, 1)) & (absM <= np.roll(absM, -1)) &
             (absM < 0.5 * absM.max()))
    brackets = [i] + list(np.where(ismin)[0])

    def negP(th):
        ph = np.exp(1j * th)
        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.real(YM(ph) ** 2 / MM(ph)) / YY
        return -val if np.isfinite(val) else np.inf

    dth = 2 * np.pi / n_angles
    best = float(P[i])
    for c in brackets:
        res = minimize_scalar(negP, bounds=(theta[c] - 2 * dth,
                                            theta[c] + 2 * dth),
                              method='bounded', options={'xatol': 1e-15})
        best = max(best, float(-res.fun))
    return best


def _adversarial_template(kind, H):
    n = np.arange(1, H + 1)
    if kind == 'sawtooth':
        return Template(np.zeros(H), 1.0 / n)
    # 'spiky': slowly-decaying, sign-alternating harmonics
    return Template(((-1.0) ** n) / np.sqrt(n), 1.0 / np.sqrt(n))


@pytest.mark.parametrize('kind', ['spiky', 'sawtooth'])
@pytest.mark.parametrize('H', [8, 10])
@pytest.mark.parametrize('N', [8, 12, 25])
def test_gate3_adversarial_vs_circle_oracle(N, H, kind):
    """Rank-deficient (N < 2H+1) and sparse cases with harsh templates:
    the scan must match the refined brute-force circle oracle to 1e-12 at
    every frequency (zero misses). Identical summations feed both, so the
    comparison isolates the maximizer."""
    rng = np.random.RandomState(1000 * H + 10 * N + (kind == 'spiky'))
    template = _adversarial_template(kind, H)
    t = np.sort(10.0 * rng.rand(N))
    dy = 0.05 * (1 + rng.rand(N))
    y = template((t / 0.77) % 1.0) + dy * rng.randn(N)
    freqs = 0.13 * (3 + np.arange(24))

    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar) ** 2)
    sums = direct_summations(t, y, w, freqs, H)

    p_scan, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                     freqs, summations=sums, method='scan')
    for i in range(len(freqs)):
        YM, MM, _ = YM_MM_from_sums(template.c_n, template.s_n, sums[i])
        p_oracle = _circle_oracle_power(YM, MM, YY)
        assert abs(p_scan[i] - p_oracle) <= GATE_TOL, \
            (kind, H, N, i, p_scan[i], p_oracle)


# ----------------------------------------------------------------------
# Gate 4: concentrated weights -- scan must not worsen the sums' floor
# ----------------------------------------------------------------------
@pytest.mark.parametrize('weight_ratio', [1e4, 1e8, 1e12])
def test_gate4_scan_weight_conditioning_within_2x_of_eigvals(weight_ratio):
    """Single-dominant-point fixtures at H=1 (where exact GLS is the truth):
    the scan path's error vs GLS must be within 2x the eigvals path's."""
    t, y, dy = dominant_point_data(weight_ratio)
    freqs = regular_grid(t)

    p_ref = GLSEstimator().power_spectrum(t, y, None, dy, freqs)
    p_eig, _ = template_periodogram(t, y, dy, H1_CN, H1_SN, freqs, fast=False)
    p_scan, _ = template_periodogram(t, y, dy, H1_CN, H1_SN, freqs,
                                     fast=False, method='scan')

    err_eig = float(np.max(np.abs(p_eig - p_ref)))
    err_scan = float(np.max(np.abs(p_scan - p_ref)))
    assert err_scan <= 2 * err_eig


# ----------------------------------------------------------------------
# Multiband: all four modes flow through the scan seam
# ----------------------------------------------------------------------
def _simulate_multiband(H, n_per_band, seed, K=2):
    rng = np.random.RandomState(seed)
    tmpl = Template(rng.randn(H), rng.randn(H))
    labels = ['g', 'r', 'i'][:K]
    t = np.concatenate([np.sort(10.0 * rng.rand(n_per_band))
                        for _ in range(K)])
    bands = np.array(sum(([b] * n_per_band for b in labels), []))
    dy = 0.05 * (1 + rng.rand(K * n_per_band))
    amp = {b: 0.8 + 0.4 * j for j, b in enumerate(labels)}
    off = {b: 15.0 - 0.5 * j for j, b in enumerate(labels)}
    y = (np.array([amp[b] for b in bands]) * tmpl((t / 0.77) % 1.0)
         + np.array([off[b] for b in bands]) + dy * rng.randn(K * n_per_band))
    freqs = 0.05 * (10 + np.arange(64))
    return t, y, bands, dy, tmpl, freqs, labels


@pytest.mark.parametrize('mode', ['independent', 'shared_phase',
                                  'floating_offsets', 'sesar'])
@pytest.mark.parametrize('H', [2, 5])
def test_multiband_scan_matches_eigvals(mode, H):
    """Scan and eigvals agree on all four sharing modes (powers + argmax)."""
    for seed in (0, 1):
        t, y, bands, dy, tmpl, freqs, labels = _simulate_multiband(H, 40, seed)
        kw = ({'relative_offsets': {b: 0.5 * j for j, b in enumerate(labels)}}
              if mode == 'sesar' else {})
        m = FastMultibandTemplatePeriodogram(templates=tmpl, mode=mode,
                                             **kw).fit(t, y, bands, dy)
        p_e = m.power(freqs, fast=True, save_best_model=False)
        p_s = m.power(freqs, fast=True, save_best_model=False, method='scan')
        assert float(np.max(np.abs(p_e - p_s))) <= GATE_TOL, (mode, H, seed)
        assert int(np.argmax(p_e)) == int(np.argmax(p_s))


def test_multiband_scan_three_bands_shared_phase():
    """K=3 exercises the M = max(128, 32 H K) grid and the K-band Newton
    accumulation in the shared-phase scan."""
    t, y, bands, dy, tmpl, freqs, _ = _simulate_multiband(3, 30, 2, K=3)
    m = FastMultibandTemplatePeriodogram(templates=tmpl,
                                         mode='shared_phase').fit(t, y, bands, dy)
    p_e = m.power(freqs, fast=True, save_best_model=False)
    p_s = m.power(freqs, fast=True, save_best_model=False, method='scan')
    assert float(np.max(np.abs(p_e - p_s))) <= GATE_TOL


def test_multiband_unknown_method_raises():
    t, y, bands, dy, tmpl, freqs, _ = _simulate_multiband(2, 20, 0)
    m = FastMultibandTemplatePeriodogram(templates=tmpl).fit(t, y, bands, dy)
    with pytest.raises(ValueError):
        m.power(freqs[:4], method='nope')


# ----------------------------------------------------------------------
# Parameter reconstruction equivalence
# ----------------------------------------------------------------------
@pytest.mark.parametrize('H', [2, 3, 5, 8])
def test_scan_params_match_eigvals(H):
    """best_fit_params agree between maximizers (power alone is blind to the
    theta_1/theta_2/theta_3 reconstruction). H = 1 is excluded: its exactly
    tied +/-phi maxima are a parametrization gauge (see curve test below)."""
    t, y, dy, template, freqs = _simulate(H, 30, 5)
    _, prm_ref = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                      freqs)
    _, prm_scan = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                       freqs, method='scan')
    ref = np.array([[p.a, p.b, p.c, p.sgn] for p in prm_ref])
    scan = np.array([[p.a, p.b, p.c, p.sgn] for p in prm_scan])
    assert float(np.max(np.abs(ref - scan))) <= 1e-9


def test_scan_h1_fitted_curves_match():
    """H = 1: gauge-invariant equivalence of the fitted models themselves."""
    t, y, dy, template, freqs = _simulate(1, 30, 5)
    _, prm_ref = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                      freqs)
    _, prm_scan = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                       freqs, method='scan')
    t_dense = np.linspace(t.min(), t.max(), 500)
    for freq, p_r, p_s in zip(freqs, prm_ref, prm_scan):
        y_r = TemplateModel(template, frequency=freq, parameters=p_r)(t_dense)
        y_s = TemplateModel(template, frequency=freq, parameters=p_s)(t_dense)
        assert float(np.max(np.abs(y_r - y_s))) <= 1e-9


def test_scan_polish_seam_matches_roots_seam():
    """scan_polish_YM_MM is a drop-in for roots_from_YM_MM at a single
    frequency, including under positive_amplitude (the multiband seam)."""
    H = 4
    t, y, dy, template, freqs = _simulate(H, 30, 9, nf=16)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar) ** 2)
    for sums in direct_summations(t, y, w, freqs, H):
        YM, MM, AC = YM_MM_from_sums(template.c_n, template.s_n, sums)
        for pa in (False, True):
            pr, powr, phir = roots_from_YM_MM(YM, MM, AC, H, ybar, YY,
                                              positive_amplitude=pa)
            ps, pows, phis = scan_polish_YM_MM(YM, MM, AC, H, ybar, YY,
                                               positive_amplitude=pa)
            assert abs(powr - pows) <= GATE_TOL
            assert abs(phir - phis) <= 1e-9
            assert abs(pr.a - ps.a) <= 1e-9


# ----------------------------------------------------------------------
# Degenerate inputs, plumbing, and bookkeeping invariances
# ----------------------------------------------------------------------
def test_scan_constant_y_zero_power():
    """Constant y: finite ~zero power, ~zero amplitude, offset = ybar, no
    crash (the A3 bar, np.allclose as in test_degenerate_inputs) -- and
    identical to the eigvals path on the same input."""
    t = np.linspace(0, 10, 40)
    y = np.full(40, 3.5)
    dy = np.full(40, 0.1)
    freqs = 0.1 * (5 + np.arange(16))
    p, prm = template_periodogram(t, y, dy, [0.7, 0.2], [0.1, -0.1], freqs,
                                  fast=False, method='scan')
    p_eig, prm_eig = template_periodogram(t, y, dy, [0.7, 0.2], [0.1, -0.1],
                                          freqs, fast=False)
    assert np.all(np.isfinite(p)) and np.allclose(p, 0.0)
    assert all(np.allclose(q.a, 0.0) and np.allclose(q.c, 3.5) for q in prm)
    # both maximizers sit on the same ~1e-33 residue landscape (not bitwise:
    # they land on different residue-level noise)
    assert np.all(np.isfinite(p_eig)) and float(np.max(np.abs(p - p_eig))) <= 1e-15


def test_scan_single_observation_flat():
    p, prm = template_periodogram(np.array([1.0]), np.array([2.0]),
                                  np.array([0.1]), [1.0], [0.0],
                                  np.array([0.5, 1.0]), fast=False,
                                  method='scan')
    assert np.all(p == 0.0)


def test_scan_empty_freqs_returns_empty():
    t, y, dy, template, _ = _simulate(2, 30, 0)
    p, prm = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                  np.array([]), fast=False, method='scan')
    assert len(p) == 0 and prm == []


def test_scan_chunk_size_invariant():
    """Chunking is bookkeeping only on the scan path too."""
    t, y, dy, template, freqs = _simulate(5, 30, 3, nf=100)
    p_one, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                    freqs, method='scan', chunk_size=10000)
    p_many, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                     freqs, method='scan', chunk_size=16)
    assert np.array_equal(p_one, p_many)


def test_scan_reachable_from_modeler_autopower():
    t, y, dy, template, _ = _simulate(3, 30, 11)
    ftp = FastTemplatePeriodogram(template=template).fit(t, y, dy)
    kw = dict(minimum_frequency=0.5, maximum_frequency=3.0,
              samples_per_peak=3)
    f_ref, p_ref = ftp.autopower(**kw)
    f_scan, p_scan = ftp.autopower(method='scan', **kw)
    assert np.array_equal(f_ref, f_scan)
    assert float(np.max(np.abs(p_ref - p_scan))) <= GATE_TOL


def test_scan_with_user_provided_summations():
    t, y, dy, template, freqs = _simulate(3, 30, 4)
    w = weights(dy)
    sums = direct_summations(t, y, w, freqs, len(template.c_n))
    p_ref, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                    freqs, summations=sums)
    p_scan, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                     freqs, summations=sums, method='scan')
    assert float(np.max(np.abs(p_ref - p_scan))) <= GATE_TOL


def test_scan_newton_derivatives_match_finite_differences():
    """core._scan_dP_d2P -- the PRODUCTION derivative code driving the
    Newton polish (single-band and per-band shared-phase) -- is checked
    against central finite differences of the directly-evaluated power at
    generic angles. (The C2 verification flagged the original version of
    this test for validating a local reimplementation instead.)"""
    from ..core import _scan_dP_d2P

    H = 4
    t, y, dy, template, freqs = _simulate(H, 30, 13, nf=4)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar) ** 2)
    sums = direct_summations(t, y, w, freqs, H)
    YM, MM, _ = YM_MM_from_sums(template.c_n, template.s_n, sums[2])

    def P(th):
        ph = np.exp(1j * th)
        return np.real(YM(ph) ** 2 / MM(ph)) / YY

    kY = np.arange(len(YM.coef))
    kM = np.arange(len(MM.coef))
    h = 1e-5
    for th in (0.3, 1.1, 2.9, 4.7, 5.9):
        ph = np.exp(1j * np.array([th]))
        Y = YM(ph)
        Y1 = np.polynomial.polynomial.polyval(ph, kY * YM.coef)
        Y2 = np.polynomial.polynomial.polyval(ph, kY * kY * YM.coef)
        Mm = MM(ph)
        M1 = np.polynomial.polynomial.polyval(ph, kM * MM.coef)
        M2 = np.polynomial.polynomial.polyval(ph, kM * kM * MM.coef)
        dP_arr, d2P_arr = _scan_dP_d2P(Y, Y1, Y2, Mm, M1, M2)
        dP = float(dP_arr[0]) / YY
        d2P = float(d2P_arr[0]) / YY
        fd1 = (P(th + h) - P(th - h)) / (2 * h)
        fd2 = (P(th + h) - 2 * P(th) + P(th - h)) / h ** 2
        assert abs(dP - fd1) <= 1e-6 * max(1.0, abs(fd1))
        assert abs(d2P - fd2) <= 1e-4 * max(1.0, abs(fd2))


def test_scan_positive_amplitude_filter_applies():
    """With an anti-correlated H=2 signal the unconstrained best amplitude is
    negative (at H >= 2 a half-phase flip cannot absorb the sign, unlike the
    H=1 +/-phi gauge); positive_amplitude must pick the best non-negative
    candidate, exactly as the root path does."""
    rng = np.random.RandomState(21)
    H = 2
    tmpl = Template([0.8, 0.4], [0.1, -0.2])
    t = np.sort(10.0 * rng.rand(60))
    y = -tmpl((1.3 * t) % 1.0) + 0.05 * rng.randn(60)
    dy = np.full(60, 0.05)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar) ** 2)
    sums = direct_summations(t, y, w, [1.3], H)
    YM, MM, AC = YM_MM_from_sums(tmpl.c_n, tmpl.s_n, sums[0])
    for pa in (False, True):
        pr, powr, _ = roots_from_YM_MM(YM, MM, AC, H, ybar, YY,
                                       positive_amplitude=pa)
        ps, pows, _ = scan_polish_YM_MM(YM, MM, AC, H, ybar, YY,
                                        positive_amplitude=pa)
        assert abs(powr - pows) <= GATE_TOL
        assert np.sign(pr.a) == np.sign(ps.a)
    # and the filter genuinely changed the answer on this fixture (the
    # eigvals path agrees, asserted via the sign match above)
    p_unc, _, _ = scan_polish_YM_MM(YM, MM, AC, H, ybar, YY)
    p_pos, _, _ = scan_polish_YM_MM(YM, MM, AC, H, ybar, YY,
                                    positive_amplitude=True)
    assert p_unc.a < 0 <= p_pos.a


# ----------------------------------------------------------------------
# Narrow-peak / near-singular-MM regime (C2 adversarial verification).
# The original scan missed sub-grid-width peaks of P hiding inside deep
# dips of |MM| on the circle (rank-deficient or phase-clustered sampling;
# silent power deficits up to ~0.7 incl. a realistic nightly cadence at
# alias frequencies). Fixed by dip candidates at deep |MM| minima plus an
# exact root-path fallback for frequencies below _SCAN_EXACT_RTOL.
# ----------------------------------------------------------------------
def _eclipse_template(H, w=0.08):
    """Box-dip (eclipse-like) template: slow sin(pi n w)/(pi n) decay --
    the harmonic content that drives MM near-singular at sparse N."""
    n = np.arange(1, H + 1)
    return Template(np.sin(np.pi * n * w) / (np.pi * n), np.zeros(H))


@pytest.mark.parametrize('H,N', [(8, 8), (8, 15), (10, 8), (8, 30)])
def test_narrow_peak_eclipse_sparse_matches_eigvals(H, N):
    """Eclipse template at sparse N: exact-fallback / dip-candidate
    territory, parity to 1e-12 with identical argmax. Honest attribution
    (re-measured on a8fd646, 2026-07-04 audit): of these parametrizations
    only (H, N) = (10, 8) contains a genuine pre-fix miss (seed 10007,
    deficit 0.139 -- the mechanism pin lives in
    test_scan_dip_candidates_are_load_bearing); the other three pass
    pre-fix and pin parity in the same regime."""
    for seed in range(10000, 10010):
        rng = np.random.default_rng([seed, H, N])
        tmpl = _eclipse_template(H)
        t = np.sort(10.0 * rng.random(N))
        dy = 0.05 * (1 + rng.random(N))
        y = tmpl((t / 0.77) % 1.0) + dy * rng.standard_normal(N)
        freqs = np.linspace(0.2, 3.0, 40)
        p_e, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                      fast=False)
        p_s, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                      fast=False, method='scan')
        assert float(np.max(np.abs(p_e - p_s))) <= GATE_TOL, (seed, H, N)
        assert int(np.argmax(p_e)) == int(np.argmax(p_s))


def test_narrow_peak_extreme_corner_conditioning():
    """H=12, N=6 (dof = 26 >> N): the most extreme rank-deficient corner.
    Here the power evaluation itself is conditioned at ~1e-10 near small
    |MM| -- scan, eigvals, AND a refined dense oracle disagree pairwise at
    that level, in BOTH directions. Pinned: agreement to 5e-10 (was 0.29
    pre-fix), no argmax flips. Do not tighten to 1e-12: the residual is
    two-sided evaluation noise shared with the reference, not a miss."""
    worst = 0.0
    for seed in range(10000, 10015):
        rng = np.random.default_rng([seed, 12, 6])
        tmpl = _eclipse_template(12)
        t = np.sort(10.0 * rng.random(6))
        dy = 0.05 * (1 + rng.random(6))
        y = tmpl((t / 0.77) % 1.0) + dy * rng.standard_normal(6)
        freqs = np.linspace(0.2, 3.0, 40)
        p_e, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                      fast=False)
        p_s, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                      fast=False, method='scan')
        worst = max(worst, float(np.max(np.abs(p_e - p_s))))
        assert int(np.argmax(p_e)) == int(np.argmax(p_s))
    assert worst <= 5e-10


def test_narrow_peak_clustered_cadence():
    """Phase-clustered sampling (two tight clumps): pre-fix (a8fd646) the
    scan missed on 3/15 of these seeds, by up to 0.13 (seed 21007;
    re-measured 2026-07-04 -- the C2 verification's ~0.7 deficits were on
    other clustered draws, not these). 5e-12 allows the shared
    sums-conditioning noise of this regime (verified two-sided vs a
    refined oracle)."""
    for seed in range(21000, 21015):
        rng = np.random.default_rng(seed)
        N, H = 6, 12
        t = np.sort(np.concatenate([0.05 * rng.random(3),
                                    5.0 + 0.05 * rng.random(3)]))
        n = np.arange(1, H + 1)
        tmpl = Template(((-1.0) ** n) / np.sqrt(n), 1.0 / np.sqrt(n))
        dy = 10 ** rng.uniform(-1, 0, N)
        y = tmpl((t * 0.62) % 1.0) + 0.1 * rng.standard_normal(N)
        freqs = np.linspace(0.3, 1.5, 30)
        p_e, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                      fast=False)
        p_s, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                      fast=False, method='scan')
        assert float(np.max(np.abs(p_e - p_s))) <= 5e-12, seed


def test_narrow_peak_nightly_cadence_alias():
    """Realistic 12-night cadence probed at trial frequencies near 1 c/d,
    where nightly sampling phase-clusters: the alias frequencies that
    matter for alias-vs-true discrimination. Pre-fix (a8fd646) deficit on
    these committed seeds: 1/10, up to 4.1e-4 (re-measured 2026-07-04; the
    C2 probes' 0.0077-in-18/20-seeds figure was for other nightly draws)."""
    for seed in range(22000, 22010):
        rng = np.random.default_rng(seed)
        H = 6
        nights = rng.choice(120, 12, replace=False)
        t = np.sort(np.concatenate(
            [nn + 0.6 + 0.05 * rng.random(3) for nn in nights[:10]] +
            [nights[10:] + 0.6 + 0.05 * rng.random(2)]))
        n = np.arange(1, H + 1)
        tmpl = Template(1.0 / n, 0.3 / n)
        dy = np.full(len(t), 0.08)
        y = tmpl((t / 0.51) % 1.0) + dy * rng.standard_normal(len(t))
        freqs = np.linspace(0.9995, 1.0005, 21)
        p_e, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                      fast=False)
        p_s, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                      fast=False, method='scan')
        assert float(np.max(np.abs(p_e - p_s))) <= GATE_TOL, seed


def test_narrow_peak_multiband_clustered():
    """Two phase-clumped bands at H=8. Honest attribution (re-measured on
    a8fd646, 2026-07-04 audit): on these committed fixtures the genuine
    pre-fix miss is in `independent` mode (6/8 seeds, up to 0.083 at seed
    26002); shared_phase and floating_offsets pass pre-fix here (the C2
    verification's 0.174 shared_phase miss was on other fixtures).
    Contracts: independent / floating_offsets pin scan == eigvals to the
    gate; shared_phase, post-C3.5, is ONE-SIDED at deep dips -- the
    max(scan, root-path) merge may recover micro-deficits of the G-root
    reference (measured up to ~1.3e-8 here, seed 26007) but must never
    fall below it; the 1e-6 cap on the recovery bounds gross F-evaluation
    blowups without pinning the reference path's exact deficit."""
    for mode in ('shared_phase', 'independent', 'floating_offsets'):
        for seed in range(26000, 26008):
            rng = np.random.default_rng(seed)
            H = 8
            t1 = np.concatenate([0.03 * rng.random(4),
                                 3.0 + 0.03 * rng.random(3)])
            t2 = np.concatenate([1.5 + 0.03 * rng.random(4),
                                 4.5 + 0.03 * rng.random(3)])
            t = np.concatenate([np.sort(t1), np.sort(t2)])
            bands = np.array(['g'] * 7 + ['r'] * 7)
            n = np.arange(1, H + 1)
            tmpl = Template(1.0 / n, 0.2 / n)
            dy = 0.05 * (1 + rng.random(14))
            y = tmpl((t * 1.0) % 1.0) + dy * rng.standard_normal(14)
            m = FastMultibandTemplatePeriodogram(
                templates=tmpl, mode=mode).fit(t, y, bands, dy)
            freqs = np.linspace(0.8, 1.2, 25)
            p_e = m.power(freqs, fast=False, save_best_model=False)
            p_s = m.power(freqs, fast=False, save_best_model=False,
                          method='scan')
            if mode == 'shared_phase':
                assert np.all(p_s >= p_e - 1e-15), (mode, seed)
                assert float(np.max(p_s - p_e)) <= 1e-6, (mode, seed)
            else:
                assert float(np.max(np.abs(p_e - p_s))) <= GATE_TOL, \
                    (mode, seed)


def test_narrow_peak_concentrated_weights_high_H():
    """Concentrated weights at H=8 (2-3 points carrying ~all weight collapse
    the effective N): the verified diff-safety regime, beyond gate 4's
    H=1-only coverage."""
    for ratio in (1e4, 1e6):
        for seed in (31, 32, 33):
            rng = np.random.default_rng(seed)
            N, H = 40, 8
            n = np.arange(1, H + 1)
            tmpl = Template(1.0 / n, 0.3 / n)
            t = np.sort(10.0 * rng.random(N))
            dy = np.full(N, 0.05)
            dy[[10, 25]] = 0.05 / np.sqrt(ratio)
            y = tmpl((t / 0.77) % 1.0) + dy * rng.standard_normal(N)
            freqs = np.linspace(0.2, 3.0, 40)
            p_e, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n,
                                          freqs, fast=False)
            p_s, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n,
                                          freqs, fast=False, method='scan')
            assert float(np.max(np.abs(p_e - p_s))) <= GATE_TOL, (ratio, seed)


def _corner_fixture(seed):
    """H=12, N=6 eclipse fixture (the extreme rank-deficient corner)."""
    rng = np.random.default_rng([seed, 12, 6])
    tmpl = _eclipse_template(12)
    t = np.sort(10.0 * rng.random(6))
    dy = 0.05 * (1 + rng.random(6))
    y = tmpl((t / 0.77) % 1.0) + dy * rng.standard_normal(6)
    return t, y, dy, tmpl, np.linspace(0.2, 3.0, 40)


def test_scan_exact_fallback_is_load_bearing(monkeypatch):
    """Mechanism pin for the exact root-path fallback, replacing a
    tautological predecessor (test_scan_dip_machinery_recovers_subgrid_spike
    compared the fallback to itself; caught by the 2026-07-04 audit).

    On this fixture (corner seed 10004; pre-fix a8fd646 deficit 0.286,
    re-measured against the actual commit) disabling ONLY the exact
    fallback -- dip candidates left active -- loses >= 0.1 of power vs the
    eigvals reference, so the dip-Newton refinement alone demonstrably
    cannot resolve this dip and the fallback specifically is the
    correctness net. Production settings must match eigvals to the corner
    regime bar. Fails by construction on any mutant that deletes or
    disables the fallback."""
    from .. import core

    t, y, dy, tmpl, freqs = _corner_fixture(10004)
    p_eig, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                    fast=False)
    p_prod, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                     fast=False, method='scan')
    assert float(np.max(np.abs(p_eig - p_prod))) <= 5e-10
    assert int(np.argmax(p_eig)) == int(np.argmax(p_prod))

    monkeypatch.setattr(core, '_SCAN_EXACT_RTOL', 0.0)
    p_mut, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                    fast=False, method='scan')
    assert float(np.max(p_eig - p_mut)) >= 0.1


def test_scan_dip_candidates_are_load_bearing(monkeypatch):
    """Mechanism pin for the deep-|MM|-dip candidates: on the eclipse
    (H=10, N=8) seed-10007 fixture (the one genuine pre-fix miss of that
    family; a8fd646 deficit 0.139) the grid+polish scan with BOTH the dip
    candidates and the exact fallback disabled loses >= 0.05 of power,
    while the dip candidates alone (exact fallback still disabled) fully
    recover the peak. Fails by construction on any mutant that deletes the
    dip-candidate machinery."""
    from .. import core

    rng = np.random.default_rng([10007, 10, 8])
    tmpl = _eclipse_template(10)
    t = np.sort(10.0 * rng.random(8))
    dy = 0.05 * (1 + rng.random(8))
    y = tmpl((t / 0.77) % 1.0) + dy * rng.standard_normal(8)
    freqs = np.linspace(0.2, 3.0, 40)

    p_eig, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                    fast=False)
    p_prod, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                     fast=False, method='scan')
    assert float(np.max(np.abs(p_eig - p_prod))) <= GATE_TOL

    monkeypatch.setattr(core, '_SCAN_EXACT_RTOL', 0.0)
    p_dips_only, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n,
                                          freqs, fast=False, method='scan')
    assert float(np.max(np.abs(p_eig - p_dips_only))) <= GATE_TOL

    monkeypatch.setattr(core, '_SCAN_DIP_RTOL', 0.0)
    p_mut, _ = template_periodogram(t, y, dy, tmpl.c_n, tmpl.s_n, freqs,
                                    fast=False, method='scan')
    assert float(np.max(p_eig - p_mut)) >= 0.05


# ----------------------------------------------------------------------
# Hardening items from the C2 verification (exception parity, dead paths)
# ----------------------------------------------------------------------
def test_scan_nonfinite_input_raises():
    """NaN-poisoned input must fail loudly on the scan path (the root path
    raises LinAlgError from np.roots); pre-fix the scan silently returned
    an all-zero periodogram."""
    t, y, dy, template, freqs = _simulate(3, 30, 0)
    y_bad = y.copy()
    y_bad[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        template_periodogram(t, y_bad, dy, template.c_n, template.s_n,
                             freqs, fast=False, method='scan')
    # multiband shared_phase scan: same loud failure
    bands = np.array(['g'] * 15 + ['r'] * 15)
    m = FastMultibandTemplatePeriodogram(
        templates=template, mode='shared_phase').fit(t, y_bad, bands, dy)
    with pytest.raises(ValueError, match="non-finite"):
        m.power(freqs[:3], fast=False, method='scan')


def test_scan_positive_amplitude_no_positive_fallback():
    """K.18 all-negative fallback (dead path under the original suite):
    ym(theta) = -(2 + cos theta) < 0 everywhere, so NO candidate has
    theta_1 >= 0 and both maximizers must return the global power maximizer
    with a < 0."""
    YM = pol.Polynomial(np.array([-0.5, -2.0, -0.5], dtype=complex))
    MM = pol.Polynomial(np.array([0.0, 0.5, 3.0, 0.5, 0.0], dtype=complex))
    AC = np.zeros(1)
    pr, powr, _ = roots_from_YM_MM(YM, MM, AC, 1, 0.0, 1.0,
                                   positive_amplitude=True)
    ps, pows, _ = scan_polish_YM_MM(YM, MM, AC, 1, 0.0, 1.0,
                                    positive_amplitude=True)
    assert pr.a < 0 and ps.a < 0
    assert abs(powr - pows) <= GATE_TOL


def test_scan_plateau_cap_at_real_kcap():
    """Zero-YM seam input: P is exactly constant (0) on the circle, every
    grid angle is a tied local max, and the real 4H candidate cap must fire
    without disturbing the zero-power result (dead path pre-verification)."""
    H = 2
    YM = pol.Polynomial(np.zeros(1, dtype=complex))
    MM = pol.Polynomial(np.array([0.1, 0.2, 1.0, 0.2, 0.1], dtype=complex))
    ps, pows, phis = scan_polish_YM_MM(YM, MM, np.zeros(2), H, 5.0, 1.0)
    assert pows == 0.0
    assert ps.a == 0.0 and ps.c == 5.0


def test_eval_polys_on_circle_guard_and_n_angles_override():
    """_eval_polys_on_circle must reject n_angles < ncoef (np.fft.ifft
    silently crops); a legal n_angles override must not change a smooth
    case beyond the parity bar."""
    from ..core import _eval_polys_on_circle
    with pytest.raises(ValueError):
        _eval_polys_on_circle(np.ones((1, 9), dtype=complex), 8)

    t, y, dy, template, freqs = _simulate(3, 30, 2, nf=16)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar) ** 2)
    for s in direct_summations(t, y, w, freqs, 3)[::4]:
        YM, MM, AC = YM_MM_from_sums(template.c_n, template.s_n, s)
        _, p_def, _ = scan_polish_YM_MM(YM, MM, AC, 3, ybar, YY)
        _, p_ovr, _ = scan_polish_from_coefs(
            YM.coef[np.newaxis], MM.coef[np.newaxis],
            np.asarray(AC)[np.newaxis], 3, ybar, YY, n_angles=4096)
        assert abs(p_def - float(p_ovr[0])) <= GATE_TOL


def test_scan_n_angles_floor_clamps_unsafe_override():
    """CRIT-1: an n_angles override below the max(128, 32H) floor would
    undersample the circle and could silently underestimate power; the floor
    clamps it up, so a deliberately tiny override returns the same powers as
    the default grid (resolution may only increase, never decrease)."""
    H = 8
    t, y, dy, template, freqs = _simulate(H, 30, 5, nf=24)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar) ** 2)
    for s in direct_summations(t, y, w, freqs, H)[::4]:
        YM, MM, AC = YM_MM_from_sums(template.c_n, template.s_n, s)
        ymc, mmc, acc = (YM.coef[np.newaxis], MM.coef[np.newaxis],
                         np.asarray(AC)[np.newaxis])
        _, p_default, _ = scan_polish_from_coefs(ymc, mmc, acc, H, ybar, YY)
        # 16 << 32*H = 256: must be clamped up to the floor, not honored, so
        # the result is bitwise identical to the default grid
        _, p_floored, _ = scan_polish_from_coefs(ymc, mmc, acc, H, ybar, YY,
                                                 n_angles=16)
        assert np.array_equal(p_default, p_floored)


# ----------------------------------------------------------------------
# MB-2 closeout (C2 verification): gate the multiband shared_phase scan
# against an INDEPENDENT brute-force max of F(theta), not just eigvals, so a
# deficit shared by the scan and the eigvals root-finder could not hide.
# ----------------------------------------------------------------------
def _shared_phase_F_oracle_power(template_dict, per_band_sums, stats,
                                 n_angles=1 << 16):
    """True max over the unit circle of the shared-phase objective
    F(theta) = sum_k W_k Re(YM_k(phi)^2 / MM_k(phi)), normalized by
    YY_combined -- assembled from the SAME per-band YM/MM the code builds
    (via _per_band_YM_MM) but maximized by brute force + scalar refinement
    of the best bracket AND of every deep local minimum of any band's
    |MM_k| (narrow F spikes hide inside per-band |MM| dips), NOT by the
    eigvals G-polynomial root-finder. This isolates the maximizer."""
    from ..multiband import _per_band_YM_MM
    per_band = _per_band_YM_MM(template_dict, per_band_sums, stats.bands)

    def F_at(ph):
        with np.errstate(divide='ignore', invalid='ignore'):
            return sum(stats.W[k] * np.real(per_band[k][0](ph) ** 2
                                            / per_band[k][1](ph))
                       for k in stats.bands)

    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    phi = np.exp(1j * theta)
    F = F_at(phi)
    F[~np.isfinite(F)] = -np.inf
    i = int(np.argmax(F))
    if not np.isfinite(F[i]):
        return 0.0
    brackets = [i]
    for k in stats.bands:
        absM = np.abs(per_band[k][1](phi))
        ismin = ((absM <= np.roll(absM, 1)) & (absM <= np.roll(absM, -1)) &
                 (absM < 0.5 * absM.max()))
        brackets += list(np.where(ismin)[0])
    dth = 2 * np.pi / n_angles

    def negF(th):
        v = F_at(np.exp(1j * np.atleast_1d(th)))[0]
        return -v if np.isfinite(v) else np.inf

    best = float(F[i])
    for c in brackets:
        res = minimize_scalar(negF, bounds=(theta[c] - 2 * dth,
                                            theta[c] + 2 * dth),
                              method='bounded', options={'xatol': 1e-14})
        best = max(best, float(-res.fun))
    return best / stats.YY_combined


@pytest.mark.parametrize('seed', [3, 7, 11])
def test_multiband_shared_phase_scan_vs_independent_F_oracle(seed):
    """The shared_phase scan must attain the true max of F(theta) (the
    objective the code optimizes), measured by an oracle that never calls the
    eigvals root-finder. Closes MB-2: a deficit shared by scan AND eigvals
    would be invisible to the scan-vs-eigvals gate but not to this one."""
    from ..multiband import build_template_set, compute_band_summations
    H = 4
    t, y, bands, dy, tmpl, freqs, labels = _simulate_multiband(H, 40, seed)
    bands_ = np.unique(bands)
    template_dict = build_template_set(tmpl, bands_)
    per_band_sumlists, stats = compute_band_summations(
        t, y, bands, freqs, H, dy=dy, mode='shared_phase', fast=False)

    m = FastMultibandTemplatePeriodogram(
        templates=tmpl, mode='shared_phase').fit(t, y, bands, dy)
    p_scan = m.power(freqs, fast=False, save_best_model=False, method='scan')
    p_eig = m.power(freqs, fast=False, save_best_model=False)

    for i in range(0, len(freqs), 4):
        per_band_sums = {b: per_band_sumlists[b][i] for b in stats.bands}
        p_oracle = _shared_phase_F_oracle_power(template_dict, per_band_sums,
                                                stats)
        # the scan must not silently underestimate the true max of F ...
        assert p_scan[i] >= p_oracle - 1e-9, (seed, i, p_scan[i], p_oracle)
        # ... and on these well-conditioned fixtures it attains it
        assert abs(p_scan[i] - p_oracle) <= 1e-7, (seed, i, p_scan[i], p_oracle)
    # scan still matches the eigvals reference to the gate
    assert float(np.max(np.abs(p_scan - p_eig))) <= GATE_TOL


# ----------------------------------------------------------------------
# MB-2 deep-dip extension (2026-07-04 audit, defect 4): oracle-gate the
# fallback regime itself. On rank-deficient phase-clustered fixtures every
# frequency trips the shared_phase deep-dip exact fallback, so scan ==
# eigvals is tautological there (the scan delegates to _shared_phase_fit
# verbatim) and only an independent F-oracle can catch a regression -- or,
# as it turned out, the pre-existing deficit of the G-root path itself.
# ----------------------------------------------------------------------
def _deepdip_multiband_fixture(seed, H=8, K=3, per_band=5):
    """Rank-deficient (at the committed per_band=5), phase-clustered K-band
    fixture (the 2026-07-04 adjudicator's recipe): every frequency of the
    test grid trips the shared_phase deep-dip exact fallback (min-band |MM|
    ratios ~5e-7..1e-4 on the scan circle; phase clustering, not point
    count, drives MM near-singular)."""
    rng = np.random.default_rng(seed)
    labels = ['g', 'r', 'i', 'z'][:K]
    n = np.arange(1, H + 1)
    tmpl = Template(1.0 / n, 0.2 / n)
    ts, bs = [], []
    for j in range(K):
        n1 = per_band // 2 + (per_band % 2)
        clump1 = 0.9 * j + 0.03 * rng.random(n1)
        clump2 = 3.0 + 0.7 * j + 0.03 * rng.random(per_band - n1)
        ts.append(np.sort(np.concatenate([clump1, clump2])))
        bs += [labels[j]] * per_band
    t = np.concatenate(ts)
    bands = np.array(bs)
    dy = 0.05 * (1 + rng.random(K * per_band))
    y = tmpl((t * 1.0) % 1.0) + dy * rng.standard_normal(K * per_band)
    return t, y, bands, dy, tmpl


def _deepdip_oracle_rows(seed, rows=(0, 8, 14, 21)):
    """(p_scan, p_eig, [(row, p_oracle), ...]) on a deep-dip fixture, with
    the fallback trigger asserted for every sampled row (regime guard).
    Rows 8 and 21 are the known deficit-carrying rows (C3 verification,
    per-row adjudication on seed 40005: +3.838e-4 at row 8, +3.562e-3 at
    row 21 = f=1.15, the clump-degenerate frequency); 0 and 14 are clean
    contrast rows."""
    from ..multiband import (build_template_set, compute_band_summations,
                             _per_band_YM_MM)
    from .. import core

    H = 8
    t, y, bands, dy, tmpl = _deepdip_multiband_fixture(seed)
    freqs = np.linspace(0.8, 1.2, 25)
    bands_ = np.unique(bands)
    template_dict = build_template_set(tmpl, bands_)
    per_band_sumlists, stats = compute_band_summations(
        t, y, bands, freqs, H, dy=dy, mode='shared_phase', fast=False)

    m = FastMultibandTemplatePeriodogram(
        templates=tmpl, mode='shared_phase').fit(t, y, bands, dy)
    p_scan = m.power(freqs, fast=False, save_best_model=False, method='scan')
    p_eig = m.power(freqs, fast=False, save_best_model=False)

    M_ang = max(core._SCAN_MIN_ANGLES,
                core._SCAN_ANGLES_PER_H * H * len(bands_))
    out = []
    for i in rows:
        per_band_sums = {b: per_band_sumlists[b][i] for b in stats.bands}
        pb = _per_band_YM_MM(template_dict, per_band_sums, stats.bands)
        Mco = np.array([pb[k][1].coef for k in stats.bands])
        absMg = np.abs(core._eval_polys_on_circle(Mco, M_ang))
        assert np.any(np.min(absMg, axis=1) <
                      core._SCAN_EXACT_RTOL * np.max(absMg, axis=1)), \
            "fixture drifted out of the deep-dip fallback regime"
        out.append((i, _shared_phase_F_oracle_power(template_dict,
                                                    per_band_sums, stats)))
    return p_scan, p_eig, out


@pytest.mark.parametrize('seed', [40000, 40005, 40007])
def test_multiband_shared_phase_deepdip_bounded_vs_F_oracle(seed):
    """Deep-dip regime, eigvals-side net: the exact G-root REFERENCE path
    retains its documented MB-DIP-1 deficit (the FP-constructed G drowns
    below its coefficient noise floor at the maximizer; worst on THESE
    committed fixtures 3.562e-3 at seed 40005 row 21; up to 8.4e-2 on
    deeper every-band-dip recipes -- see VERIFICATION.md WP C3), so it is
    gated by a bounded catastrophic-regression net vs the independent
    F-oracle, which no scan-vs-eigvals comparison can provide. The scan,
    post-C3.5, must never fall below the eigvals reference (max-merge
    safety)."""
    p_scan, p_eig, rows = _deepdip_oracle_rows(seed)
    assert np.all(p_scan >= p_eig - 1e-15)   # max(scan, root) >= root
    for i, p_oracle in rows:
        assert p_eig[i] >= p_oracle - 6e-3, (seed, i, p_eig[i], p_oracle)


@pytest.mark.parametrize('seed', [40000, 40005, 40007])
def test_multiband_shared_phase_deepdip_attains_F_oracle(seed):
    """C3.5 fix gate for MB-DIP-1 (was xfail while the defect was open):
    with the deep-dip max(scan, root-path) merge, method='scan' must attain
    the independent F-oracle max to 1e-9 on the rank-deficient
    phase-clustered fixtures where the G-root path alone loses up to
    3.6e-3 (committed seeds) / 8.4e-2 (deeper recipes)."""
    p_scan, _, rows = _deepdip_oracle_rows(seed)
    for i, p_oracle in rows:
        assert p_scan[i] >= p_oracle - 1e-9, (seed, i, p_scan[i], p_oracle)
