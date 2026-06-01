"""Comparison-method baselines (``ftperiodogram.baselines``).

Every linear baseline is checked against an INDEPENDENT brute-force reference
(weighted normal equations, a different linear-algebra path than the module's
whitened ``lstsq``) to ``< 1e-9`` on a clean injection, the GLS baseline is
checked to reproduce FTP@H=1 to machine precision, and the Sesar oracle is
checked to reproduce the analytic FTP optimum to ``< 1e-6`` -- the gold-standard
equivalence that makes it the cost reference.  All estimators are checked
picklable (the ``spawn`` parallel requirement) and the recovery-scorer seam is
checked to score FTP identically through ``score_estimator`` and ``__call__``.
"""
import pickle

import numpy as np
import pytest

from ftperiodogram.template import Template
from ftperiodogram.utils import weights
from ftperiodogram.multiband import FastMultibandTemplatePeriodogram
from ftperiodogram.baselines import (FTPEstimator, GLSEstimator, MHLSEstimator,
                                     MultibandLSEstimator, SesarOracleEstimator,
                                     _errs, _band_index, _chi2_0)
from ftperiodogram.simulate import SyntheticCadence, simulate_multiband_lightcurve
from ftperiodogram.validation import frequency_grid, make_recovery_scorer


# ----------------------------------------------------------------------
# Fixtures: one clean multiband injection + a search grid
# ----------------------------------------------------------------------
def _injection(seed=0, n_epochs=40, bands=('g', 'r'), period=0.55, h=4):
    rng = np.random.RandomState(seed)
    c = rng.randn(h)
    s = rng.randn(h)
    truth = Template(c, s)
    cadence = SyntheticCadence(n_epochs={b: n_epochs for b in bands},
                               bands=list(bands), baseline_days=400.0,
                               random_state=seed + 1)
    lc = simulate_multiband_lightcurve(
        truth, period, cadence, amplitude=0.6, mean_mag=15.0, tau=rng.rand(),
        band_offsets={b: 0.4 * i for i, b in enumerate(bands)},
        random_state=rng, add_noise=True, shuffle=True)
    return lc, truth


def _grid():
    # straddle the injected period (P=0.55 d  ->  f ~ 1.82 cyc/day)
    return frequency_grid(1.0, 3.0, 400)


# ----------------------------------------------------------------------
# Independent brute-force reference: weighted normal equations
# ----------------------------------------------------------------------
def _ref_chi2(X, y, w):
    """Weighted residual SS via a whitened SVD pseudo-inverse -- an independent
    linear-algebra path (different from the module's ``lstsq``), robust to a
    rank-deficient design at a degenerate frequency."""
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta = np.linalg.pinv(Xw, rcond=1e-10) @ yw    # truncate tiny singular values
    r = yw - Xw @ beta
    return float(r @ r)


def _ref_power_shared(t, y, bands, dy, freqs, H):
    """Reference MHLS/GLS power: shared Fourier(H) + per-band offset."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = t.size
    w = weights(_errs(dy, n))
    labels, codes = _band_index(bands, n)
    nb = len(labels)
    chi2_0 = _chi2_0(y, w, codes, nb)
    out = np.empty(freqs.size)
    for i, f in enumerate(freqs):
        cols = [(codes == b).astype(float) for b in range(nb)]
        for j in range(1, H + 1):
            cols.append(np.cos(2 * np.pi * j * f * t))
            cols.append(np.sin(2 * np.pi * j * f * t))
        out[i] = 1.0 - _ref_chi2(np.column_stack(cols), y, w) / chi2_0
    return out


def _ref_power_perband(t, y, bands, dy, freqs, H):
    """Reference multiband-LS power: independent per-band Fourier(H), shared
    period."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = t.size
    w = weights(_errs(dy, n))
    labels, codes = _band_index(bands, n)
    nb = len(labels)
    chi2_0 = _chi2_0(y, w, codes, nb)
    out = np.empty(freqs.size)
    for i, f in enumerate(freqs):
        per = 1 + 2 * H
        X = np.zeros((n, nb * per))
        for b in range(nb):
            m = (codes == b)
            X[m, b * per] = 1.0
            for j in range(1, H + 1):
                X[m, b * per + 2 * j - 1] = np.cos(2 * np.pi * j * f * t[m])
                X[m, b * per + 2 * j] = np.sin(2 * np.pi * j * f * t[m])
        out[i] = 1.0 - _ref_chi2(X, y, w) / chi2_0
    return out


# ----------------------------------------------------------------------
# GLS == FTP@H=1 (the anchor) ------------------------------------------
# ----------------------------------------------------------------------
def test_gls_matches_ftp_h1():
    lc, _ = _injection()
    freqs = _grid()
    gls = GLSEstimator().power_spectrum(lc.t, lc.y, lc.bands, lc.dy, freqs)

    ftp = FastMultibandTemplatePeriodogram([Template([1.0], [0.0])],
                                           mode='floating_offsets')
    ftp.fit(lc.t, lc.y, lc.bands, lc.dy)
    ftp_pow = ftp.power(freqs, fast=False, save_best_model=False)

    np.testing.assert_allclose(gls, ftp_pow, atol=1e-9, rtol=0)


def test_gls_equals_mhls_h1():
    lc, _ = _injection()
    freqs = _grid()
    gls = GLSEstimator().power_spectrum(lc.t, lc.y, lc.bands, lc.dy, freqs)
    mhls1 = MHLSEstimator(1).power_spectrum(lc.t, lc.y, lc.bands, lc.dy, freqs)
    np.testing.assert_allclose(gls, mhls1, atol=1e-12, rtol=0)


# ----------------------------------------------------------------------
# Linear baselines vs the independent normal-equations reference --------
# ----------------------------------------------------------------------
@pytest.mark.parametrize("H", [1, 3, 6])
def test_mhls_matches_reference(H):
    lc, _ = _injection()
    freqs = _grid()
    got = MHLSEstimator(H).power_spectrum(lc.t, lc.y, lc.bands, lc.dy, freqs)
    ref = _ref_power_shared(lc.t, lc.y, lc.bands, lc.dy, freqs, H)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0)


@pytest.mark.parametrize("H", [1, 2, 4])
def test_multiband_ls_matches_reference(H):
    lc, _ = _injection()
    freqs = _grid()
    got = MultibandLSEstimator(H).power_spectrum(lc.t, lc.y, lc.bands, lc.dy, freqs)
    ref = _ref_power_perband(lc.t, lc.y, lc.bands, lc.dy, freqs, H)
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0)


def test_powers_in_unit_interval():
    lc, _ = _injection()
    freqs = _grid()
    for est in (GLSEstimator(), MHLSEstimator(8), MultibandLSEstimator(2)):
        p = est.power_spectrum(lc.t, lc.y, lc.bands, lc.dy, freqs)
        assert np.all(p >= -1e-9) and np.all(p <= 1.0 + 1e-9)


# ----------------------------------------------------------------------
# Sesar oracle reproduces FTP (gold-standard equivalence) ---------------
# ----------------------------------------------------------------------
def test_sesar_oracle_matches_ftp_power():
    lc, truth = _injection(h=4)
    freqs = frequency_grid(1.0, 3.0, 120)         # coarse: the oracle is slow
    oracle = SesarOracleEstimator([truth], n_tau=256, polish=True)
    o_pow = oracle.power_spectrum(lc.t, lc.y, lc.bands, lc.dy, freqs)

    ftp = FastMultibandTemplatePeriodogram([truth], mode='floating_offsets')
    ftp.fit(lc.t, lc.y, lc.bands, lc.dy)
    f_pow = ftp.power(freqs, fast=False, save_best_model=False)

    np.testing.assert_allclose(o_pow, f_pow, atol=1e-6, rtol=0)


def test_sesar_oracle_recovers_injected_period():
    lc, truth = _injection(h=4, period=0.55)
    freqs = frequency_grid(1.0, 3.0, 400)
    P_rec = SesarOracleEstimator([truth], n_tau=256)(lc.t, lc.y, lc.bands,
                                                     lc.dy, freqs)
    assert abs(P_rec - 0.55) / 0.55 < 0.01


# ----------------------------------------------------------------------
# Picklability (the spawn-parallel requirement) -------------------------
# ----------------------------------------------------------------------
def test_estimators_picklable():
    truth = Template([1.0, 0.3], [0.0, 0.1])
    estimators = [FTPEstimator([truth]), GLSEstimator(), MHLSEstimator(8),
                  MultibandLSEstimator(2), SesarOracleEstimator([truth])]
    lc, _ = _injection()
    freqs = _grid()
    for est in estimators:
        clone = pickle.loads(pickle.dumps(est))
        a = est(lc.t, lc.y, lc.bands, lc.dy, freqs)
        b = clone(lc.t, lc.y, lc.bands, lc.dy, freqs)
        assert a == b


# ----------------------------------------------------------------------
# Single-band path (bands is None) --------------------------------------
# ----------------------------------------------------------------------
def test_single_band_linear_estimators_run():
    rng = np.random.RandomState(3)
    t = np.sort(rng.uniform(0, 100, 60))
    period = 0.7
    y = 15 + 0.5 * np.cos(2 * np.pi * (t / period)) + 0.02 * rng.randn(60)
    dy = np.full(60, 0.02)
    freqs = frequency_grid(1.0, 2.0, 500)
    for est in (GLSEstimator(), MHLSEstimator(3)):
        P_rec = est(t, y, None, dy, freqs)
        assert abs(P_rec - period) / period < 0.02


# ----------------------------------------------------------------------
# Scorer seam: FTP through score_estimator == native __call__ -----------
# ----------------------------------------------------------------------
def test_score_estimator_matches_call():
    truths = [Template([1.0, 0.25], [0.0, 0.15]),
              Template([1.0], [0.4])]
    cadence = SyntheticCadence(n_epochs={'g': 25, 'r': 25}, bands=['g', 'r'],
                               baseline_days=365.0, random_state=7)
    freqs = frequency_grid(1.0, 4.0, 1500)
    scorer = make_recovery_scorer(cadence, truths, freqs=freqs, n_sources=12,
                                  random_state=1)
    vocab = [truths[0]]
    rate_call, mask_call = scorer(vocab, return_mask=True)
    rate_est, mask_est = scorer.score_estimator(
        FTPEstimator(vocab, mode='floating_offsets'), return_mask=True)
    assert rate_call == rate_est
    np.testing.assert_array_equal(mask_call, mask_est)


def test_score_estimator_runs_baselines():
    truths = [Template([1.0, 0.25], [0.0, 0.15])]
    cadence = SyntheticCadence(n_epochs={'g': 30, 'r': 30}, bands=['g', 'r'],
                               baseline_days=365.0, random_state=2)
    freqs = frequency_grid(1.0, 4.0, 2000)
    scorer = make_recovery_scorer(cadence, truths, freqs=freqs, n_sources=10,
                                  random_state=0)
    for est in (GLSEstimator(), MHLSEstimator(6), MultibandLSEstimator(1)):
        rate = scorer.score_estimator(est)
        assert 0.0 <= rate <= 1.0
