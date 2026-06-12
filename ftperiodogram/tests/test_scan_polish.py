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
    bracket (the raw grid max is only ~1e-7-accurate in power; refinement
    makes the oracle meaningful at the 1e-12 gate)."""
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    phi = np.exp(1j * theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.real(YM(phi) ** 2 / MM(phi)) / YY
    P[~np.isfinite(P)] = -np.inf
    i = int(np.argmax(P))
    if not np.isfinite(P[i]):
        return 0.0

    def negP(th):
        ph = np.exp(1j * th)
        with np.errstate(divide='ignore', invalid='ignore'):
            val = np.real(YM(ph) ** 2 / MM(ph)) / YY
        return -val if np.isfinite(val) else np.inf

    dth = 2 * np.pi / n_angles
    res = minimize_scalar(negP, bounds=(theta[i] - dth, theta[i] + dth),
                          method='bounded', options={'xatol': 1e-14})
    return max(float(P[i]), float(-res.fun))


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
    """The analytic dP/dtheta and d2P/dtheta2 driving the Newton polish are
    checked against central finite differences of the directly-evaluated
    power at generic angles."""
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
        B = 2.0 * Y * Y1 * Mm - Y * Y * M1
        dP = float(np.real(1j * B / (Mm * Mm))[0]) / YY
        d2P = float(np.real((-2.0 * (Y1 * Y1 + Y * Y2) * Mm + Y * Y * M2)
                            / (Mm * Mm) + 2.0 * B * M1 / (Mm * Mm * Mm))[0]) / YY
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
