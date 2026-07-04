"""Tests for the multiband fast template periodogram (ftperiodogram.multiband).

The decisive test is the K>=2 brute-force comparison: it is the only check that
can tell the floating-offsets (C) and sesar (D) models apart, since at K=1 they
both collapse to the single-band path. The brute force is an independent oracle
(fine phase scan + weighted least squares + local refinement), not a
reimplementation of the polynomial solver.
"""
import numpy as np
import numpy.polynomial as pol
import pytest
from scipy.optimize import minimize_scalar

from ftperiodogram.template import Template
from ftperiodogram.modeler import FastTemplatePeriodogram
from ftperiodogram.utils import weights
from ftperiodogram import core, summations
from ftperiodogram.multiband import (FastMultibandTemplatePeriodogram,
                                     MultibandTemplateModel,
                                     MultibandModelFitParams,
                                     multiband_template_periodogram,
                                     build_template_set)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
def _make_template(nharmonics=3, seed=0):
    rng = np.random.RandomState(seed)
    n = 64
    ph = np.arange(n) / n
    y = (np.cos(2 * np.pi * ph) + 0.4 * np.cos(2 * np.pi * 2 * ph + 0.7)
         + 0.2 * np.sin(2 * np.pi * 3 * ph))
    return Template.from_sampled(y, nharmonics=nharmonics)


def _simulate(template, f0=0.123, amp=2.0, band_offsets=(0.0,), n_per_band=60,
              noise=0.03, seed=1):
    """Simulate shared-amplitude, shared-phase multiband data with per-band
    constant offsets (mean magnitudes)."""
    rng = np.random.RandomState(seed)
    t, y, bands, dy = [], [], [], []
    for k, off in enumerate(band_offsets):
        tk = np.sort(rng.rand(n_per_band) * 80)
        sigk = noise * (1 + 0.5 * k)
        yk = amp * template((f0 * tk) % 1.0) + off + sigk * rng.randn(n_per_band)
        t.append(tk)
        y.append(yk)
        bands.append(np.full(n_per_band, k, dtype=int))
        dy.append(np.full(n_per_band, sigk))
    # interleave so the global array is NOT sorted (exercises per-band sorting)
    t = np.concatenate(t)
    y = np.concatenate(y)
    bands = np.concatenate(bands)
    dy = np.concatenate(dy)
    order = np.random.RandomState(seed + 7).permutation(len(t))
    return t[order], y[order], bands[order], dy[order]


def _global_weights(dy, n):
    err = np.ones(n) if dy is None else \
        np.broadcast_to(np.asarray(dy, float), (n,)).astype(float)
    return weights(err)


def _single_band_positive_power(t, y, dy, template, freqs):
    """Single-band power computed WITH the positivity filter (the reference the
    multiband K=1 path must match exactly)."""
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar) ** 2)
    sums = summations.direct_summations(t, y, w, freqs, len(template.c_n))
    H = len(template.c_n)
    out = np.empty(len(freqs))
    for i, s in enumerate(sums):
        YM, MM, AC = core.YM_MM_from_sums(template.c_n, template.s_n, s)
        _, pw, _ = core.roots_from_YM_MM(YM, MM, AC, H, ybar, YY,
                                         positive_amplitude=True)
        out[i] = pw
    return out


def _brute_power(t, y, bands, dy, template, freq, mode, relative_offsets=None,
                 nphase=2000):
    """Independent brute-force multiband power at a single frequency.

    Scans the phase on a fine grid, solving a weighted linear least squares for
    the (shared/per-band) amplitudes and offsets at each phase, then refines the
    best phase. Enforces the same positivity (theta_1 >= 0) constraint as the
    FTP solver: the maximum is taken only over phases yielding a non-negative
    amplitude. Returns the maximized power 1 - chi2/chi2_0.
    """
    n = len(t)
    w = _global_weights(dy, n)
    band_labels = np.unique(bands)

    y_work = y.astype(float).copy()
    if mode == 'sesar':
        for k in band_labels:
            y_work[bands == k] -= relative_offsets[k]

    # reference chi2_0
    if mode in ('floating_offsets', 'independent'):
        chi2_0 = 0.0
        for k in band_labels:
            m = bands == k
            ybark = np.dot(w[m], y_work[m]) / w[m].sum()
            chi2_0 += np.dot(w[m], (y_work[m] - ybark) ** 2)
    else:  # sesar -> single global mean
        ybar_g = np.dot(w, y_work)
        chi2_0 = np.dot(w, (y_work - ybar_g) ** 2)

    def design(ph):
        Mvals = template((freq * t - ph) % 1.0)
        if mode == 'sesar':
            return np.column_stack([Mvals, np.ones(n)])
        # floating_offsets: shared amplitude, per-band offsets
        cols = [Mvals] + [(bands == k).astype(float) for k in band_labels]
        return np.column_stack(cols)

    def chi2_amp_shared(ph):
        X = design(ph)
        XtW = X.T * w
        beta = np.linalg.solve(XtW @ X, XtW @ y_work)
        resid = y_work - X @ beta
        return np.dot(w, resid ** 2), beta[0]      # beta[0] is the amplitude

    if mode == 'independent':
        # each band independently; combine as the global fraction of variance
        # explained (sum_k W_k (YY_k - chi2_k) / sum_k W_k YY_k).
        explained = 0.0
        YY_C = 0.0
        for k in band_labels:
            m = bands == k
            wk = w[m] / w[m].sum()
            yk = y_work[m]
            tk = t[m]
            ybark = np.dot(wk, yk)
            chi2_0_k = np.dot(wk, (yk - ybark) ** 2)
            Wk = w[m].sum()

            def chi2_amp_k(ph, tk=tk, yk=yk, wk=wk):
                Mv = template((freq * tk - ph) % 1.0)
                X = np.column_stack([Mv, np.ones(len(tk))])
                XtW = X.T * wk
                beta = np.linalg.solve(XtW @ X, XtW @ yk)
                return np.dot(wk, (yk - X @ beta) ** 2), beta[0]

            best = _scan_and_refine(chi2_amp_k, nphase=nphase)
            explained += Wk * (chi2_0_k - best)
            YY_C += Wk * chi2_0_k
        return explained / YY_C

    best = _scan_and_refine(chi2_amp_shared, nphase=nphase)
    return 1 - best / chi2_0


def _brute_power_shared_phase(t, y, bands, dy, template, freq, nphase=2000):
    """Brute-force model-B power: scan a SHARED phase, per-band independent
    linear fits, combine W_k-weighted explained variance. No positivity
    constraint (model B does not enforce per-band positivity)."""
    n = len(t)
    w = _global_weights(dy, n)
    band_labels = np.unique(bands)

    # per-band within-band weights, means, variances, weight totals
    info = {}
    YY_C = 0.0
    for k in band_labels:
        m = bands == k
        wk = w[m] / w[m].sum()
        yk = y[m].astype(float)
        ybark = np.dot(wk, yk)
        YYk = np.dot(wk, (yk - ybark) ** 2)
        info[k] = (t[m], yk, wk, w[m].sum())
        YY_C += w[m].sum() * YYk

    def neg_power(ph):
        explained = 0.0
        for k in band_labels:
            tk, yk, wk, Wk = info[k]
            X = np.column_stack([template((freq * tk - ph) % 1.0),
                                 np.ones(len(tk))])
            XtW = X.T * wk
            beta = np.linalg.solve(XtW @ X, XtW @ yk)
            chi2_k = np.dot(wk, (yk - X @ beta) ** 2)
            ybark = np.dot(wk, yk)
            YYk = np.dot(wk, (yk - ybark) ** 2)
            explained += Wk * (YYk - chi2_k)
        return -explained / YY_C

    grid = np.linspace(0, 1, nphase, endpoint=False)
    vals = np.array([neg_power(p) for p in grid])
    i = int(np.argmin(vals))
    best = vals[i]
    lo, hi = grid[(i - 1) % nphase], grid[(i + 1) % nphase]
    if lo < hi:
        res = minimize_scalar(neg_power, bounds=(lo, hi), method='bounded')
        best = min(best, res.fun)
    return -best


def _scan_and_refine(chi2_amp_fn, nphase=2000):
    """Minimize chi2 over phase subject to amplitude >= 0.

    ``chi2_amp_fn(ph)`` returns ``(chi2, amplitude)``.
    """
    grid = np.linspace(0, 1, nphase, endpoint=False)
    out = [chi2_amp_fn(p) for p in grid]
    chi2s = np.array([o[0] for o in out])
    amps = np.array([o[1] for o in out])
    masked = np.where(amps >= 0, chi2s, np.inf)
    i = int(np.argmin(masked))
    best = masked[i]

    lo = grid[(i - 1) % nphase]
    hi = grid[(i + 1) % nphase]
    if lo < hi:
        def obj(ph):
            c, a = chi2_amp_fn(ph)
            return c if a >= 0 else np.inf
        res = minimize_scalar(obj, bounds=(lo, hi), method='bounded')
        if res.fun < best:
            best = res.fun
    return best


TEMPLATE = _make_template()
FREQS = np.linspace(0.05, 0.30, 300)


# ----------------------------------------------------------------------
# K = 1 exact reduction to the single band path
# ----------------------------------------------------------------------
@pytest.mark.parametrize('mode', ['floating_offsets', 'sesar', 'independent'])
def test_k1_reduces_to_single_band(mode):
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0,))
    ref = _single_band_positive_power(t, y, dy, TEMPLATE, FREQS)

    kw = dict(relative_offsets={0: 0.0}) if mode == 'sesar' else {}
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode=mode, **kw)
    mb.fit(t, y, bands, dy)
    power = mb.power(FREQS)

    assert np.allclose(power, ref, atol=1e-9), \
        "K=1 %s deviates from single-band(+positivity) by %.2e" % (
            mode, np.max(np.abs(power - ref)))


def test_k1_single_band_params_roundtrip():
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0,))
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='floating_offsets')
    mb.fit(t, y, bands, dy)
    f, p = mb.autopower()
    sb_params = mb.best_model.parameters.single_band(0)
    # the single-band projection is an ordinary ModelFitParams
    from ftperiodogram.utils import ModelFitParams
    assert isinstance(sb_params, ModelFitParams)
    assert sb_params.a >= 0


# ----------------------------------------------------------------------
# K >= 2 brute-force validation (the decisive test)
# ----------------------------------------------------------------------
def test_k2_floating_offsets_matches_brute_force():
    offs = (0.0, -0.6, 0.4)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='floating_offsets')
    mb.fit(t, y, bands, dy)
    for f in (0.123, 0.0917, 0.2013):
        p_ftp = float(mb.power(f))
        p_bf = _brute_power(t, y, bands, dy, TEMPLATE, f, 'floating_offsets')
        assert abs(p_ftp - p_bf) < 2e-3, \
            "C: f=%.4f ftp=%.5f brute=%.5f" % (f, p_ftp, p_bf)


def test_k2_sesar_matches_brute_force():
    offs = (0.0, -0.6, 0.4)
    lam = {0: 0.0, 1: -0.6, 2: 0.4}      # correct relative offsets
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='sesar',
                                          relative_offsets=lam)
    mb.fit(t, y, bands, dy)
    for f in (0.123, 0.0917, 0.2013):
        p_ftp = float(mb.power(f))
        p_bf = _brute_power(t, y, bands, dy, TEMPLATE, f, 'sesar',
                            relative_offsets=lam)
        assert abs(p_ftp - p_bf) < 2e-3, \
            "D: f=%.4f ftp=%.5f brute=%.5f" % (f, p_ftp, p_bf)


def test_k1_shared_phase_reduces_to_single_band():
    # model B does not enforce positivity, so it matches the (unfiltered)
    # single-band public power exactly at K=1.
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0,))
    order = np.argsort(t)   # single-band path requires globally-sorted t
    sb = FastTemplatePeriodogram(TEMPLATE).fit(t[order], y[order],
                                               dy[order]).power(FREQS)
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='shared_phase')
    mb.fit(t, y, bands, dy)
    p = mb.power(FREQS)
    assert np.allclose(p, sb, atol=1e-6), \
        "K=1 shared_phase deviates from single-band by %.2e" % np.max(np.abs(p - sb))


def test_k2_shared_phase_matches_brute_force():
    offs = (0.0, -0.6, 0.4)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='shared_phase')
    mb.fit(t, y, bands, dy)
    for f in (0.123, 0.0917, 0.2013):
        p_ftp = float(mb.power(f))
        p_bf = _brute_power_shared_phase(t, y, bands, dy, TEMPLATE, f)
        assert abs(p_ftp - p_bf) < 2e-3, \
            "B: f=%.4f ftp=%.5f brute=%.5f" % (f, p_ftp, p_bf)


def test_shared_phase_recovers_signal():
    offs = (0.0, -0.8, 0.5)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='shared_phase')
    mb.fit(t, y, bands, dy)
    f, p = mb.autopower(minimum_frequency=0.05, maximum_frequency=0.30)
    assert abs(f[np.argmax(p)] - 0.123) < 1e-3
    # all per-band amplitudes are positive for this genuine in-phase signal
    for pars in mb.best_model.parameters.params_by_band.values():
        assert pars.a > 0


def test_k2_independent_matches_brute_force():
    offs = (0.0, -0.6, 0.4)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='independent')
    mb.fit(t, y, bands, dy)
    for f in (0.123, 0.0917):
        p_ftp = float(mb.power(f))
        p_bf = _brute_power(t, y, bands, dy, TEMPLATE, f, 'independent')
        assert abs(p_ftp - p_bf) < 2e-3, \
            "A: f=%.4f ftp=%.5f brute=%.5f" % (f, p_ftp, p_bf)


def test_C_and_D_are_not_conflated():
    """With offset per-band means and WRONG (zero) relative offsets, the sesar
    model (D) must fit worse than floating-offsets (C). If the two were
    implemented identically (the bug all critics flagged), the powers would be
    equal."""
    offs = (0.0, -0.8, 0.5)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)

    c = FastMultibandTemplatePeriodogram(TEMPLATE, mode='floating_offsets')
    c.fit(t, y, bands, dy)
    d = FastMultibandTemplatePeriodogram(
        TEMPLATE, mode='sesar', relative_offsets={0: 0.0, 1: 0.0, 2: 0.0})
    d.fit(t, y, bands, dy)

    pc = float(c.power(0.123))
    pd = float(d.power(0.123))
    assert pc > pd + 0.05, "C (%.4f) should clearly beat misspecified D (%.4f)" % (pc, pd)


def test_sesar_correct_offsets_recovers_signal():
    offs = (0.0, -0.8, 0.5)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    d = FastMultibandTemplatePeriodogram(
        TEMPLATE, mode='sesar', relative_offsets={0: 0.0, 1: -0.8, 2: 0.5})
    d.fit(t, y, bands, dy)
    f, p = d.autopower(minimum_frequency=0.05, maximum_frequency=0.30)
    assert abs(f[np.argmax(p)] - 0.123) < 1e-3
    assert p.max() > 0.9


# ----------------------------------------------------------------------
# Positivity (K.18)
# ----------------------------------------------------------------------
@pytest.mark.parametrize('mode', ['floating_offsets', 'sesar', 'independent'])
def test_amplitudes_are_nonnegative(mode):
    offs = (0.0, -0.6, 0.4)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    kw = dict(relative_offsets={0: 0.0, 1: -0.6, 2: 0.4}) if mode == 'sesar' else {}
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode=mode, **kw)
    mb.fit(t, y, bands, dy)
    _, params_list = multiband_template_periodogram(
        t, y, bands, build_template_set(TEMPLATE, bands), FREQS, dy=dy,
        mode=mode, relative_offsets=kw.get('relative_offsets'), fast=False)
    for mbp in params_list:
        for band, pars in mbp.params_by_band.items():
            assert pars.a >= 0, "negative amplitude in band %r" % band


# ----------------------------------------------------------------------
# fast (NFFT) vs direct summations
# ----------------------------------------------------------------------
@pytest.mark.parametrize('mode', ['floating_offsets', 'sesar', 'independent'])
def test_fast_matches_direct(mode):
    offs = (0.0, -0.6, 0.4)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    kw = dict(relative_offsets={0: 0.0, 1: -0.6, 2: 0.4}) if mode == 'sesar' else {}
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode=mode, **kw)
    mb.fit(t, y, bands, dy)
    f_fast, p_fast = mb.autopower(fast=True, minimum_frequency=0.05,
                                  maximum_frequency=0.30)
    _, p_dir = mb.autopower(fast=False, minimum_frequency=0.05,
                            maximum_frequency=0.30)
    assert np.allclose(p_fast, p_dir, atol=1e-4), \
        "fast vs direct max|diff| = %.2e" % np.max(np.abs(p_fast - p_dir))


# ----------------------------------------------------------------------
# Model object
# ----------------------------------------------------------------------
def test_model_predicts_per_band_offsets():
    offs = (0.0, -0.8, 0.5)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs, noise=0.01)
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='floating_offsets')
    mb.fit(t, y, bands, dy)
    mb.autopower(minimum_frequency=0.05, maximum_frequency=0.30)
    model = mb.best_model
    yhat = model(t, bands)
    # a good fit at the recovered period should track the data
    assert np.corrcoef(yhat, y)[0, 1] > 0.95
    # single-band projection agrees with the masked multiband evaluation
    for k in np.unique(bands):
        m = bands == k
        assert np.allclose(model.predict_band(t[m], k), yhat[m])


# ----------------------------------------------------------------------
# Error / validation paths
# ----------------------------------------------------------------------
def test_sesar_without_offsets_raises():
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0, 0.3))
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='sesar')
    mb.fit(t, y, bands, dy)
    with pytest.raises(ValueError):
        mb.autopower()


def test_catalog_single_element_equals_single_set():
    offs = (0.0, -0.6, 0.4)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    single = FastMultibandTemplatePeriodogram(TEMPLATE, mode='floating_offsets')
    single.fit(t, y, bands, dy)
    f1, p1 = single.autopower(minimum_frequency=0.05, maximum_frequency=0.30)

    catalog = FastMultibandTemplatePeriodogram([TEMPLATE], mode='floating_offsets')
    catalog.fit(t, y, bands, dy)
    f2, p2 = catalog.autopower(minimum_frequency=0.05, maximum_frequency=0.30)

    assert np.allclose(p1, p2)
    assert catalog.best_model.template_set_index == 0


def test_catalog_picks_better_template():
    offs = (0.0, -0.6, 0.4)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=offs)
    other = _make_template(nharmonics=3, seed=99)  # a different shape
    sets = [other, TEMPLATE]

    p_each = []
    for tmpl in sets:
        mb = FastMultibandTemplatePeriodogram(tmpl, mode='floating_offsets')
        mb.fit(t, y, bands, dy)
        p_each.append(mb.autopower(minimum_frequency=0.05,
                                   maximum_frequency=0.30)[1])
    p_each = np.array(p_each)

    catalog = FastMultibandTemplatePeriodogram(sets, mode='floating_offsets')
    catalog.fit(t, y, bands, dy)
    f, p = catalog.autopower(minimum_frequency=0.05, maximum_frequency=0.30)

    # the catalog reports the per-frequency max over sets
    assert np.allclose(p, p_each.max(axis=0))
    # and records the argmax set at the recovered peak
    pk = int(np.argmax(p))
    assert catalog.best_model.template_set_index == int(np.argmax(p_each[:, pk]))
    assert abs(f[pk] - 0.123) < 1e-3


def test_unknown_mode_raises():
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0, 0.3))
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='nonsense')
    mb.fit(t, y, bands, dy)
    with pytest.raises(ValueError):
        mb.autopower()


def test_missing_band_template_raises():
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0, 0.3))
    mb = FastMultibandTemplatePeriodogram({0: TEMPLATE}, mode='floating_offsets')
    mb.fit(t, y, bands, dy)
    with pytest.raises(ValueError):
        mb.autopower()


@pytest.mark.parametrize('mode', ['floating_offsets', 'sesar', 'independent',
                                  'shared_phase'])
def test_flat_lightcurve_is_zero_power_not_nan(mode):
    # a perfectly flat (zero-variance) source has nothing to detect; the
    # periodogram must return finite, zero power rather than nan/inf or crash.
    rng = np.random.RandomState(3)
    n = 60
    t = np.sort(rng.rand(n) * 50)
    bands = np.array([0, 1, 2] * (n // 3))[:n]
    y = np.full(n, 7.0)                # constant in every band
    dy = np.full(n, 0.1)
    kw = dict(relative_offsets={0: 0.0, 1: 0.0, 2: 0.0}) if mode == 'sesar' else {}
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode=mode, **kw)
    mb.fit(t, y, bands, dy)
    f, p = mb.autopower(minimum_frequency=0.05, maximum_frequency=0.30)
    assert np.all(np.isfinite(p))
    assert np.allclose(p, 0.0, atol=1e-9)


def test_independent_mode_flat_band_does_not_crash():
    # A single perfectly flat band (zero variance) alongside a real variable
    # band used to crash independent mode: a constant band gives YM == 0, an
    # empty root set, and np.argmax on an empty sequence. It must now skip the
    # flat band (zero-amplitude fit, no explained variance) and still recover
    # the real band's signal.
    t0, y0, b0, dy0 = _simulate(TEMPLATE, f0=0.123, band_offsets=(0.0,))
    b0 = np.zeros(len(t0), dtype=int)                 # real signal in band 0
    rng = np.random.RandomState(5)
    tf = np.sort(rng.rand(50) * 80)
    yf = np.full(50, 7.0)                             # band 1: perfectly flat
    t = np.concatenate([t0, tf])
    y = np.concatenate([y0, yf])
    bands = np.concatenate([b0, np.ones(50, dtype=int)])
    dy = np.concatenate([dy0, np.full(50, 0.1)])
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='independent')
    mb.fit(t, y, bands, dy)
    f, p = mb.autopower(minimum_frequency=0.05, maximum_frequency=0.30)
    assert np.all(np.isfinite(p))
    assert p.max() > 0.0                             # real band still detected


def test_polynomial_coeffs_are_complex128():
    # The polynomial coefficients must stay complex128. complex64 degrades the
    # root-finding precision by ~500x, yet stays under every brute-force test
    # tolerance, so no numerical test catches a revert. Assert the dtype
    # directly so dropping to complex64 fails deterministically.
    t, y, b, dy = _simulate(TEMPLATE, band_offsets=(0.0,))
    w = weights(dy)
    sums = summations.direct_summations(t, y, w, np.array([0.1]),
                                        len(TEMPLATE.c_n))
    YM, MM, AC = core.YM_MM_from_sums(TEMPLATE.c_n, TEMPLATE.s_n, sums[0])
    assert YM.coef.dtype == np.complex128
    assert MM.coef.dtype == np.complex128


def test_autofrequency_honors_minimum_frequency():
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0, 0.3))
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='floating_offsets')
    mb.fit(t, y, bands, dy)
    f, p = mb.autopower(minimum_frequency=2.0, maximum_frequency=4.0)
    df = f[1] - f[0]
    # grid starts at the nearest point at/just below min_freq (was ~0 before fix)
    assert 2.0 - df <= f.min() <= 2.0 + df


def test_predict_unseen_band_is_nan():
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0, 0.3))
    mb = FastMultibandTemplatePeriodogram(TEMPLATE, mode='floating_offsets')
    mb.fit(t, y, bands, dy)
    mb.autopower(minimum_frequency=0.05, maximum_frequency=0.30)
    # band label 99 was never fit -> nan, not uninitialized garbage
    out = mb.best_model(np.array([1.0, 2.0]), np.array([0, 99]))
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_mismatched_harmonics_raises():
    t2 = _make_template(nharmonics=2)
    t3 = _make_template(nharmonics=3)
    t, y, bands, dy = _simulate(TEMPLATE, band_offsets=(0.0, 0.3))
    with pytest.raises(ValueError):
        build_template_set({0: t2, 1: t3}, bands)


# ----------------------------------------------------------------------
# C3 finding SESAR-DIP-1 (VERIFICATION.md WP C3 / C3.5): the sesar-only
# global-recentring variance term was assembled in the uncentered form
# sum_k W_k Mbar_k^2 - (sum_k W_k Mbar_k)^2, whose O(0.1) coefficient mass
# cancels to ~1e-13 on the circle when the bands' mean-template polynomials
# are similar -- corrupting MM' by ~5e-4 relative at deep |MM| dips and the
# returned peak power by ~1e-4 in BOTH directions (including OVERSTATEMENT,
# the false-alarm direction). combine_band_summations now assembles the
# algebraically identical centered form sum_k W_k (Mbar_k - Mbar_comb)^2.
#
# The fixture is the worst adjudicated cell of the C3 verification (F3,
# seed 96001 recipe: H=10 eclipse-like template, K=3, 5 pts/band in two
# ~3e-4-wide phase clumps): uncentered error vs the raw-data oracle was
# +1.186e-4 (overstatement); centered is 3.6e-10. Golden oracle value from
# the C3 raw-data weighted-LS oracle (2^13 shift grid + refinement).
# ----------------------------------------------------------------------
_SESAR_DIP_T = np.array([
    6.5947940084648416e-05, 2.0507639052717020e-04, 8.1935551429797678e-05,
    2.4676665726079294e+00, 2.4676200475208092e+00, 2.2603734193551384e-04,
    2.3652833671225965e-04, 4.1904105601952947e-05, 2.4676928296831537e+00,
    2.4676546638681818e+00, 8.2569951874920494e-05, 1.6000160834251593e-04,
    2.0831668423992129e-04, 2.4677384707458896e+00, 2.4675534114732689e+00])
_SESAR_DIP_Y = np.array([
    -2.436573625343343, -2.609919279486644, -2.822845698710106,
    -2.6984468383117606, -2.731749682405403, -2.933623044880887,
    -2.8656818873659486, -2.9180859527818885, -2.903921157092899,
    -2.8076172141370916, -2.218293653729429, -2.21051006985073,
    -2.3324702206991472, -2.040438259527455, -2.2148419209099215])
_SESAR_DIP_DY = np.array([
    0.12183737377127442, 0.1446486246012699, 0.13079587223685737,
    0.12354644648967136, 0.14141522432554846, 0.06564458953444063,
    0.06758106015148926, 0.07687534245474707, 0.09126503444120482,
    0.12592828815641843, 0.05295583000582459, 0.11763739002989276,
    0.0571207022186115, 0.13727680873738343, 0.12640351311193032])
_SESAR_DIP_BANDS = np.array(['a'] * 5 + ['b'] * 5 + ['c'] * 5)
_SESAR_DIP_CN = np.array([
    4.3636250153424664e-09, 6.9465339645364166e-01, 4.2877295288067980e-09,
    -5.1115347291167179e-01, 4.1430527420865291e-09, 3.1702829483339839e-01,
    3.9423423912021357e-09, -1.7200116008195929e-01, 3.7015313533088177e-09,
    8.3737577161999949e-02])
_SESAR_DIP_SN = np.array([
    -1.5391820517180577e-01, 4.1010577884974835e-10, 1.7574835857399859e-01,
    7.9978224230826565e-10, -1.8114200779218151e-01, 1.1516062894649659e-09,
    1.4916546913604969e-01, 1.4533406904715261e-09, -9.7130848384831922e-02,
    1.6988080176794876e-09])
_SESAR_DIP_LAM = {'a': 0.0, 'b': -0.2978243289731143, 'c': 0.4159798682813498}
_SESAR_DIP_F = 1.2158266519810017
_SESAR_DIP_ORACLE = 0.24529791670787773


def test_sesar_deepdip_centered_recentring_matches_oracle():
    """The centered MM' assembly must agree with the raw-data oracle on the
    adjudicated deep-dip cell (uncentered: +1.186e-4 overstatement)."""
    tmpl = Template(_SESAR_DIP_CN, _SESAR_DIP_SN)
    m = FastMultibandTemplatePeriodogram(
        templates=tmpl, mode='sesar',
        relative_offsets=_SESAR_DIP_LAM).fit(
            _SESAR_DIP_T, _SESAR_DIP_Y, _SESAR_DIP_BANDS, _SESAR_DIP_DY)
    freqs = np.array([_SESAR_DIP_F])
    p_eig = m.power(freqs, fast=False, save_best_model=False)
    p_scan = m.power(freqs, fast=False, save_best_model=False, method='scan')
    assert abs(float(p_eig[0]) - _SESAR_DIP_ORACLE) <= 5e-9
    assert abs(float(p_scan[0]) - _SESAR_DIP_ORACLE) <= 5e-9


def test_sesar_deepdip_uncentered_construction_still_fails():
    """Non-vacuity pin: the OLD uncentered construction, rebuilt inline from
    the same per-band pieces, still misses the oracle by ~1e-4 on this cell
    -- so the test above genuinely discriminates the constructions."""
    from ftperiodogram.multiband import (compute_band_summations,
                                         _per_band_YM_MM,
                                         _mean_template_poly)
    H = 10
    tmpl = Template(_SESAR_DIP_CN, _SESAR_DIP_SN)
    bands_ = np.unique(_SESAR_DIP_BANDS)
    td = build_template_set(tmpl, bands_)
    sl, stats = compute_band_summations(
        _SESAR_DIP_T, _SESAR_DIP_Y, _SESAR_DIP_BANDS,
        np.array([_SESAR_DIP_F]), H, dy=_SESAR_DIP_DY, mode='sesar',
        relative_offsets=_SESAR_DIP_LAM, fast=False)
    pb = _per_band_YM_MM(td, {b: sl[b][0] for b in stats.bands}, stats.bands)

    YM = MM = AC = None
    for b in stats.bands:
        YM_k, MM_k, AC_k = pb[b]
        wk = stats.W[b]
        YM = wk * YM_k if YM is None else YM + wk * YM_k
        MM = wk * MM_k if MM is None else MM + wk * MM_k
        a = wk * np.asarray(AC_k)
        AC = a if AC is None else AC + a
    YM_c = MM_c = Mb = None
    for b in stats.bands:
        AC_k = pb[b][2]
        wk = stats.W[b]
        M_k = _mean_template_poly(AC_k)
        term = wk * ((stats.ybar[b] - stats.ybar_global) * M_k)
        YM_c = term if YM_c is None else YM_c + term
        sq = wk * (M_k * M_k)                       # uncentered square
        MM_c = sq if MM_c is None else MM_c + sq
        wm = wk * M_k
        Mb = wm if Mb is None else Mb + wm
    YM_old = YM + YM_c
    MM_old = MM + MM_c - Mb * Mb                    # big - big cancellation

    _, pw_old, _ = core.roots_from_YM_MM(
        YM_old, MM_old, AC, H, stats.ybar_global, stats.YY_global,
        positive_amplitude=True)
    assert abs(pw_old - _SESAR_DIP_ORACLE) >= 5e-5   # measured 1.186e-4
