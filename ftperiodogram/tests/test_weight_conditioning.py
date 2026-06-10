"""Weight-conditioning regression tests (WP A2).

Mechanism under test: catastrophic cancellation in the ``E[xy] - E[x]E[y]``
moment sums when the inverse-variance weight concentrates in fewer points than
~the model dof (a raw max/min weight ratio alone is harmless; concentration is
what matters -- see AUDIT_2026-06-10.md).  The direct path computes two-pass
*centered* covariance sums (moments of ``cos - C``, ``sin - S``), which avoids
the cancellation; the NFFT fast path cannot be centered the same way, so it
emits a ``UserWarning`` advising ``fast=False`` when the weights are
dangerously concentrated.
"""
import warnings

import numpy as np
import pytest

from ..baselines import GLSEstimator
from ..core import template_periodogram
from ..summations import (_direct_summations_single_freq_uncentered,
                          direct_summations, fast_summations)
from ..template import Template
from ..utils import weights

# At H=1 with a single cosine template, FTP spans exactly the GLS model space
# {1, cos, sin}, so the rank-revealing lstsq GLS power is an exact reference.
H1_CN = [1.0]
H1_SN = [0.0]

SUMMATION_FIELDS = ('C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS')


def dominant_point_data(weight_ratio, N=40, T=10.0, f0=1.7, rseed=7):
    """Homoscedastic data except one point carrying max(w)/min(w) = weight_ratio."""
    rand = np.random.RandomState(rseed)
    t = np.sort(T * rand.rand(N))
    y = (1.0 + 0.5 * np.cos(2 * np.pi * f0 * t)
             + 0.3 * np.sin(2 * np.pi * f0 * t))
    y += 0.05 * rand.randn(N)
    dy = np.full(N, 0.05)
    # w ~ dy^-2: a dy contrast of sqrt(ratio) gives the target weight ratio
    dy[N // 2] = 0.05 / np.sqrt(weight_ratio)
    return t, y, dy


def regular_grid(t, samples_per_peak=4, fmax=3.0):
    df = 1.0 / ((t.max() - t.min()) * samples_per_peak)
    nf = int(np.ceil(fmax / df))
    return df * (1 + np.arange(nf))


def uncentered_direct_powers(t, y, dy, freqs):
    """H=1 powers through the retained uncentered direct path."""
    w = weights(dy)
    sums = [_direct_summations_single_freq_uncentered(t, y, w, f, len(H1_CN))
            for f in freqs]
    p, _ = template_periodogram(t, y, dy, H1_CN, H1_SN, freqs,
                                summations=sums)
    return p


@pytest.mark.parametrize('weight_ratio', [1e8, 1e12])
def test_centered_direct_path_beats_uncentered_vs_gls(weight_ratio):
    """Single-dominant-point fixture: the centered direct path must cut the
    power error vs the exact GLS reference by >= 5x relative to the
    uncentered path."""
    t, y, dy = dominant_point_data(weight_ratio)
    freqs = regular_grid(t)

    p_ref = GLSEstimator().power_spectrum(t, y, None, dy, freqs)
    p_centered, _ = template_periodogram(t, y, dy, H1_CN, H1_SN, freqs,
                                         fast=False)
    p_uncentered = uncentered_direct_powers(t, y, dy, freqs)

    err_centered = np.max(np.abs(p_centered - p_ref))
    err_uncentered = np.max(np.abs(p_uncentered - p_ref))

    assert err_uncentered >= 5 * err_centered


@pytest.mark.parametrize('weight_ratio', [1e8, 1e12])
def test_fast_path_warns_on_concentrated_weights(weight_ratio):
    t, y, dy = dominant_point_data(weight_ratio)
    freqs = regular_grid(t)
    with pytest.warns(UserWarning, match="fast=False"):
        fast_summations(t, y, weights(dy), freqs, 1)


def test_fast_path_warns_on_ratio_trigger_alone():
    """max(w)/min(w) > 1e6 fires even when no single point holds ~all the
    weight (share trigger not met)."""
    t, y, dy = dominant_point_data(1e7)
    w = weights(dy)
    assert np.max(w) < 1 - 1e-6  # share trigger genuinely not met
    freqs = regular_grid(t)
    with pytest.warns(UserWarning, match="fast=False"):
        fast_summations(t, y, w, freqs, 1)


def test_fast_path_warning_through_template_periodogram():
    t, y, dy = dominant_point_data(1e8)
    freqs = regular_grid(t)
    with pytest.warns(UserWarning, match="fast=False"):
        template_periodogram(t, y, dy, H1_CN, H1_SN, freqs, fast=True)


def test_no_warning_for_benign_weights():
    t, y, dy = dominant_point_data(1.0)  # homoscedastic
    freqs = regular_grid(t)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fast_summations(t, y, weights(dy), freqs, 1)


def test_centered_matches_uncentered_on_benign_weights():
    """The centered rewrite is analytically identical to the uncentered sums;
    on well-conditioned weights they must agree to numerical noise."""
    t, y, dy = dominant_point_data(1.0)
    freqs = regular_grid(t)
    w = weights(dy)
    nh = 3
    centered = direct_summations(t, y, w, freqs, nh)
    for sums_c, f in zip(centered, freqs):
        sums_u = _direct_summations_single_freq_uncentered(t, y, w, f, nh)
        for field in SUMMATION_FIELDS:
            np.testing.assert_allclose(getattr(sums_c, field),
                                       getattr(sums_u, field),
                                       rtol=0, atol=1e-12)


def test_tol_plumbed_through_template_periodogram():
    """sigma/tol must reach fast_summations: a loose NFFT tolerance has to
    change the result, and a tight one has to track the direct path at least
    as well."""
    t, y, dy = dominant_point_data(1.0)
    freqs = regular_grid(t)

    p_direct, _ = template_periodogram(t, y, dy, H1_CN, H1_SN, freqs,
                                       fast=False)
    p_tight, _ = template_periodogram(t, y, dy, H1_CN, H1_SN, freqs,
                                      fast=True, tol=1e-12)
    p_loose, _ = template_periodogram(t, y, dy, H1_CN, H1_SN, freqs,
                                      fast=True, tol=1e-1)

    assert np.max(np.abs(p_loose - p_tight)) > 0  # kwarg actually plumbed
    err_tight = np.max(np.abs(p_tight - p_direct))
    err_loose = np.max(np.abs(p_loose - p_direct))
    assert err_tight <= err_loose


def test_tol_plumbed_through_modeler_autopower():
    from ..modeler import FastTemplatePeriodogram

    t, y, dy = dominant_point_data(1.0)
    model = FastTemplatePeriodogram(template=Template(H1_CN, H1_SN))
    model.fit(t, y, dy)

    kwargs = dict(minimum_frequency=0.1, maximum_frequency=3.0,
                  samples_per_peak=4, save_best_model=False)
    freq_t, p_tight = model.autopower(fast=True, tol=1e-12, **kwargs)
    freq_l, p_loose = model.autopower(fast=True, tol=1e-1, **kwargs)

    np.testing.assert_allclose(freq_t, freq_l)
    assert np.max(np.abs(p_loose - p_tight)) > 0
