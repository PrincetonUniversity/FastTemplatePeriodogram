"""API-hygiene regression tests (WP A7).

Three fixes, applied to both periodogram classes (single-band modeler and
multiband):

1. the ``nyquist_factor`` default frequency-grid path is deprecated -- the
   "average Nyquist frequency" is not meaningful for irregular sampling, so
   ``autofrequency``/``autopower`` warn unless ``maximum_frequency`` is
   explicit;
2. an explicit ``maximum_frequency`` is *included* in the grid (the grid ends
   at the first point >= f_max; previously the top bin was always excluded),
   and ``minimum_frequency`` is honored (the single-band modeler silently
   ignored it via a ``min``-for-``max`` bug);
3. ``fit()`` resets ``best_model``, so refitting on a new dataset can never
   return a stale best fit from the previous one.
"""
import warnings

import numpy as np
import pytest

from ..modeler import FastTemplatePeriodogram, FastMultiTemplatePeriodogram
from ..multiband import FastMultibandTemplatePeriodogram
from ..template import Template

TEMPLATE = Template([0.8, 0.15], [0.1, 0.05])


def _data(P=0.77, N=60, T=20.0, rseed=3):
    rand = np.random.RandomState(rseed)
    t = np.sort(T * rand.rand(N))
    t[0], t[-1] = 0.0, T  # pin the baseline so df = 1/(T*spp) exactly
    y = 12.0 + 0.6 * TEMPLATE((t / P) % 1.0) + 0.02 * rand.randn(N)
    dy = np.full(N, 0.02)
    return t, y, dy


def _multiband_data(P=0.77, N=80, T=20.0, rseed=4):
    t, y, dy = _data(P=P, N=N, T=T, rseed=rseed)
    bands = np.array(['g', 'r'])[np.arange(N) % 2]
    return t, y, bands, dy


def _single_band_model(**kwargs):
    t, y, dy = _data(**kwargs)
    return FastTemplatePeriodogram(template=TEMPLATE).fit(t, y, dy)


def _multiband_model(**kwargs):
    t, y, bands, dy = _multiband_data(**kwargs)
    model = FastMultibandTemplatePeriodogram(TEMPLATE, mode='floating_offsets')
    model.fit(t, y, bands, dy)
    return model


# ----------------------------------------------------------------------
# 1. nyquist_factor default path is deprecated
# ----------------------------------------------------------------------
def test_autofrequency_nyquist_default_path_warns():
    model = _single_band_model()
    with pytest.warns(FutureWarning, match="maximum_frequency"):
        model.autofrequency()


def test_multiband_autofrequency_nyquist_default_path_warns():
    model = _multiband_model()
    with pytest.warns(FutureWarning, match="maximum_frequency"):
        model.autofrequency()


def test_autofrequency_explicit_bounds_do_not_warn():
    model = _single_band_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.autofrequency(minimum_frequency=0.5, maximum_frequency=2.5)


def test_multiband_autofrequency_explicit_bounds_do_not_warn():
    model = _multiband_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model.autofrequency(minimum_frequency=0.5, maximum_frequency=2.5)


# ----------------------------------------------------------------------
# 2. explicit grid bounds are honored, f_max included
# ----------------------------------------------------------------------
@pytest.mark.parametrize('make_model', [_single_band_model, _multiband_model])
def test_autofrequency_includes_maximum_frequency(make_model):
    model = make_model()
    # T = 20, samples_per_peak = 5 -> df = 0.01; f_max = 2.5 is exactly on-grid
    freqs = model.autofrequency(samples_per_peak=5,
                                minimum_frequency=0.5, maximum_frequency=2.5)
    df = freqs[1] - freqs[0]
    np.testing.assert_allclose(df, 0.01, rtol=1e-10)
    np.testing.assert_allclose(freqs.max(), 2.5, rtol=1e-10)


@pytest.mark.parametrize('make_model', [_single_band_model, _multiband_model])
def test_autofrequency_off_grid_maximum_frequency_covered(make_model):
    model = make_model()
    # f_max = 2.5043 is off-grid: the grid must end at the first point >= f_max
    freqs = model.autofrequency(samples_per_peak=5,
                                minimum_frequency=0.5,
                                maximum_frequency=2.5043)
    df = freqs[1] - freqs[0]
    assert freqs.max() >= 2.5043
    assert freqs.max() < 2.5043 + df


@pytest.mark.parametrize('make_model', [_single_band_model, _multiband_model])
def test_autofrequency_honors_minimum_frequency(make_model):
    model = make_model()
    freqs = model.autofrequency(minimum_frequency=2.0, maximum_frequency=4.0)
    df = freqs[1] - freqs[0]
    # grid starts at the point at/just below f_min (the single-band modeler
    # used to silently start at df instead)
    assert 2.0 - df <= freqs.min() <= 2.0 + df
    assert freqs.max() >= 4.0


# ----------------------------------------------------------------------
# 3. fit() resets best_model (no stale cross-dataset fits)
# ----------------------------------------------------------------------
GRID = dict(minimum_frequency=0.5, maximum_frequency=2.5)


def test_fit_resets_best_model_single_band():
    model = _single_band_model(P=0.77, rseed=3)
    model.autopower(**GRID)
    assert model.best_model is not None

    t2, y2, dy2 = _data(P=1.31, rseed=11)
    model.fit(t2, y2, dy2)
    assert model.best_model is None

    model.autopower(**GRID)
    f_best = model.best_model.frequency
    # the new best fit reflects the SECOND dataset's frequency
    assert abs(f_best - 1.0 / 1.31) < abs(f_best - 1.0 / 0.77)


def test_fit_resets_best_model_multi_template():
    t, y, dy = _data(P=0.77, rseed=3)
    model = FastMultiTemplatePeriodogram(templates=[TEMPLATE]).fit(t, y, dy)
    model.autopower(**GRID)
    assert model.best_model is not None

    t2, y2, dy2 = _data(P=1.31, rseed=11)
    model.fit(t2, y2, dy2)
    assert model.best_model is None


def test_fit_resets_best_model_multiband():
    model = _multiband_model(P=0.77, rseed=4)
    model.autopower(**GRID)
    assert model.best_model is not None

    t2, y2, bands2, dy2 = _multiband_data(P=1.31, rseed=12)
    model.fit(t2, y2, bands2, dy2)
    assert model.best_model is None
