"""Tests for the light-curve simulator and cadence model (ftperiodogram.simulate).

Bounded and seeded (RandomState) -- small epoch counts, no network.  The cadence
tests assert the seasonal-gap / per-band-count structure; the simulator tests
assert the single-band sorted-t contract, an exact noise-free round trip, and that
shuffled multiband output is unsorted yet consumable by the multiband modeler.
"""
from collections import Counter

import numpy as np
import numpy.testing as npt
import pytest

from ftperiodogram.template import Template
from ftperiodogram import simulate as sim
from ftperiodogram import FastMultibandTemplatePeriodogram


def _template():
    return Template([1.0, 0.3], [0.0, 0.1])


# ----------------------------------------------------------------------
# error-vs-magnitude model
# ----------------------------------------------------------------------
def test_err_model_monotone_and_clipped():
    f = sim.exp_mag_error()
    assert float(f(21.0)) > float(f(15.0))         # fainter => larger sigma
    assert float(f(0.0)) == pytest.approx(0.01)    # clipped to floor at bright end
    assert float(f(100.0)) == pytest.approx(0.5)   # clipped to sigma_max at faint end


# ----------------------------------------------------------------------
# synthetic cadence
# ----------------------------------------------------------------------
def test_synthetic_cadence_seasonal_gaps_single_band():
    cad = sim.SyntheticCadence(n_epochs=80, season_length_days=200.0,
                               season_period_days=365.25,
                               baseline_days=3 * 365.25, random_state=0)
    s = cad.sample()
    assert s.bands is None
    assert s.t.size == 80
    assert np.all(np.diff(s.t) >= 0)               # globally sorted
    assert np.all(sim._in_season(s.t, 0.0, 200.0, 365.25))  # every epoch in-season
    assert cad.baseline == pytest.approx(3 * 365.25)


def test_synthetic_cadence_per_band_counts():
    cad = sim.SyntheticCadence(n_epochs={'g': 40, 'r': 30}, bands=['g', 'r'],
                               random_state=1)
    s = cad.sample()
    assert s.bands is not None
    counts = Counter(s.bands.tolist())
    assert counts == {'g': 40, 'r': 30}
    assert np.all(np.diff(s.t) >= 0)               # globally sorted across bands


def test_synthetic_cadence_is_deterministic():
    cad = sim.SyntheticCadence(n_epochs=40, random_state=7)
    npt.assert_array_equal(cad.sample().t, cad.sample().t)


def test_synthetic_cadence_rejects_dict_without_bands():
    with pytest.raises(ValueError):
        sim.SyntheticCadence(n_epochs={'g': 10})


def test_cadence_base_is_abstract():
    with pytest.raises(NotImplementedError):
        sim.Cadence().sample()


# ----------------------------------------------------------------------
# single-band simulator
# ----------------------------------------------------------------------
def test_simulate_lightcurve_sorted_and_noise_free_roundtrip():
    tmpl = _template()
    cad = sim.SyntheticCadence(n_epochs=50, random_state=2)
    lc = sim.simulate_lightcurve(tmpl, period=0.6, cadence=cad, amplitude=0.4,
                                 mean_mag=16.0, tau=0.2, random_state=3,
                                 add_noise=False)
    assert np.all(np.diff(lc.t) >= 0)              # single-band sorted-t contract
    assert np.all(lc.dy > 0)
    assert lc.period == 0.6
    npt.assert_allclose(lc.frequency, 1.0 / 0.6)
    expected = 16.0 + 0.4 * np.asarray(tmpl(lc.t / 0.6 - 0.2))
    npt.assert_allclose(lc.y, expected)            # noise-free == the exact model


def test_simulate_lightcurve_noise_scales_with_dy():
    tmpl = _template()
    cad = sim.SyntheticCadence(n_epochs=400, random_state=9)
    clean = sim.simulate_lightcurve(tmpl, 0.6, cad, random_state=4, add_noise=False)
    noisy = sim.simulate_lightcurve(tmpl, 0.6, cad, random_state=4, add_noise=True)
    resid = (noisy.y - clean.y) / noisy.dy          # standardized residuals ~ N(0,1)
    assert abs(np.std(resid) - 1.0) < 0.2


def test_simulate_lightcurve_rejects_multiband_cadence():
    cad = sim.SyntheticCadence(n_epochs=20, bands=['g', 'r'], random_state=0)
    with pytest.raises(ValueError):
        sim.simulate_lightcurve(_template(), 0.6, cad)


# ----------------------------------------------------------------------
# multiband simulator
# ----------------------------------------------------------------------
def test_simulate_multiband_floating_offsets_roundtrip():
    tmpl = _template()
    cad = sim.SyntheticCadence(n_epochs=30, bands=['g', 'r', 'i'], random_state=4)
    boff = {'g': 0.0, 'r': -0.6, 'i': 0.4}
    bamp = {'g': 1.0, 'r': 0.8, 'i': 0.6}
    lc = sim.simulate_multiband_lightcurve(
        tmpl, period=0.55, cadence=cad, amplitude=0.5, mean_mag=15.0, tau=0.1,
        band_amplitudes=bamp, band_offsets=boff, random_state=5,
        add_noise=False, shuffle=False)
    for b in ('g', 'r', 'i'):
        m = lc.bands == b
        phase = lc.t[m] / 0.55 - 0.1
        expected = 15.0 + boff[b] + 0.5 * bamp[b] * np.asarray(tmpl(phase))
        npt.assert_allclose(lc.y[m], expected)


def test_simulate_multiband_shuffle_unsorted_but_fits():
    tmpl = _template()
    cad = sim.SyntheticCadence(n_epochs=40, bands=['g', 'r'], random_state=6)
    lc = sim.simulate_multiband_lightcurve(tmpl, 0.5, cad, amplitude=0.5,
                                           random_state=8, shuffle=True)
    assert not np.all(np.diff(lc.t) >= 0)          # global order is shuffled
    model = FastMultibandTemplatePeriodogram([tmpl], mode='floating_offsets')
    model.fit(lc.t, lc.y, lc.bands, lc.dy)
    f, p = model.autopower(minimum_frequency=1.0, maximum_frequency=3.0)
    assert np.all(np.isfinite(p))


def test_simulate_multiband_missing_band_key_raises():
    cad = sim.SyntheticCadence(n_epochs=10, bands=['g', 'r'], random_state=0)
    with pytest.raises(ValueError):
        sim.simulate_multiband_lightcurve(_template(), 0.6, cad,
                                          band_offsets={'g': 0.0})  # missing 'r'


def test_simulate_multiband_rejects_single_band_cadence():
    cad = sim.SyntheticCadence(n_epochs=20, random_state=0)
    with pytest.raises(ValueError):
        sim.simulate_multiband_lightcurve(_template(), 0.6, cad)
