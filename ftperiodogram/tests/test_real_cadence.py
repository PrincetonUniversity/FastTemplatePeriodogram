"""Tests for RealZTFCadence (the real-data observing cadence).

Bounded, seeded, and NO network: the cadence is built from a few hand-written
per-band epoch arrays (a tiny in-repo fixture standing in for a real ZTF DR light
curve), and the ``from_cache`` path is exercised against a temp ``.npz`` written in
the test rather than a live IRSA pull.  We assert the Cadence ABC contract
(globally ascending-sorted ``t``, aligned ``bands``, a positive-sigma
``mag_err_model``, the correct ``baseline``) and a clean round trip through the
multiband simulator.
"""
import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

from ftperiodogram.template import Template
from ftperiodogram import simulate as sim
from ftperiodogram import RealZTFCadence, FastMultibandTemplatePeriodogram


# A tiny RR-Lyrae-like real-ish cadence: two yearly seasons with a winter gap,
# r sampled a bit denser than g (as ZTF really does), unsorted on input so the
# globally-sorted contract is actually exercised.
G_EPOCHS = np.array([58200.4, 58205.1, 58560.9, 58300.2, 58210.7, 58575.3])
R_EPOCHS = np.array([58201.1, 58202.9, 58206.4, 58559.8, 58301.0, 58576.1,
                     58211.2, 58212.8])


def _fixture_cadence(err_model=None):
    return RealZTFCadence({'g': G_EPOCHS, 'r': R_EPOCHS},
                          err_model=err_model, object_id='fixture0',
                          metadata=dict(ra=298.0, dec=29.9, group_mode='test'))


def _template():
    return Template([1.0, 0.4, 0.2], [0.0, 0.1, 0.05])


# ----------------------------------------------------------------------
# Construction + Cadence ABC contract
# ----------------------------------------------------------------------
def test_real_cadence_sample_sorted_and_bands_aligned():
    cad = _fixture_cadence()
    s = cad.sample()
    assert s.bands is not None
    assert len(s.bands) == len(s.t) == G_EPOCHS.size + R_EPOCHS.size
    assert np.all(np.diff(s.t) >= 0)                       # globally ascending
    # every epoch and its band label co-located after the global mergesort
    for band, src in (('g', G_EPOCHS), ('r', R_EPOCHS)):
        npt.assert_array_equal(np.sort(s.t[s.bands == band]), np.sort(src))


def test_real_cadence_baseline_is_span():
    cad = _fixture_cadence()
    expect = max(G_EPOCHS.max(), R_EPOCHS.max()) - min(G_EPOCHS.min(), R_EPOCHS.min())
    assert cad.baseline == pytest.approx(expect)
    assert cad.baseline > 300.0                            # spans >1 season


def test_real_cadence_epoch_counts():
    cad = _fixture_cadence()
    assert cad.epoch_counts() == {'g': G_EPOCHS.size, 'r': R_EPOCHS.size}


def test_real_cadence_default_err_model_positive():
    cad = _fixture_cadence()                               # err_model=None -> exp_mag_error
    s = cad.sample()
    sig = s.mag_err_model(np.array([14.0, 16.0, 18.5]))
    assert np.all(np.asarray(sig) > 0)


def test_real_cadence_custom_err_model_used():
    flat = lambda mag: np.full(np.shape(mag), 0.037)       # noqa: E731
    cad = _fixture_cadence(err_model=flat)
    s = cad.sample()
    npt.assert_allclose(np.asarray(s.mag_err_model([15.0, 17.0])), 0.037)


def test_real_cadence_is_deterministic_ignores_rng():
    cad = _fixture_cadence()
    npt.assert_array_equal(cad.sample(rng=0).t, cad.sample(rng=999).t)


def test_real_cadence_rejects_empty():
    with pytest.raises(ValueError):
        RealZTFCadence({})
    with pytest.raises(ValueError):
        RealZTFCadence({'g': np.array([])})


def test_real_cadence_rejects_nonfinite():
    with pytest.raises(ValueError):
        RealZTFCadence({'g': np.array([1.0, np.nan, 3.0])})


# ----------------------------------------------------------------------
# Round trip through the multiband simulator + modeler
# ----------------------------------------------------------------------
def test_real_cadence_roundtrips_through_simulator():
    tmpl = _template()
    cad = _fixture_cadence()
    lc = sim.simulate_multiband_lightcurve(tmpl, period=0.55, cadence=cad,
                                           amplitude=0.6, mean_mag=16.0,
                                           random_state=7)
    n = G_EPOCHS.size + R_EPOCHS.size
    assert lc.t.size == n
    assert set(lc.bands.tolist()) == {'g', 'r'}
    assert np.all(np.isfinite(lc.y))
    assert np.all(lc.dy > 0)
    # the simulated LC is consumable by the multiband modeler over an explicit band
    model = FastMultibandTemplatePeriodogram([tmpl], mode='floating_offsets')
    model.fit(lc.t, lc.y, lc.bands, lc.dy)
    f, p = model.autopower(minimum_frequency=1.0, maximum_frequency=3.0)
    assert np.all(np.isfinite(p))


def test_real_cadence_noise_free_roundtrip_exact():
    tmpl = _template()
    cad = _fixture_cadence(err_model=lambda m: np.full(np.shape(m), 0.02))
    lc = sim.simulate_multiband_lightcurve(
        tmpl, period=0.55, cadence=cad, amplitude=0.6, mean_mag=16.0, tau=0.1,
        random_state=3, add_noise=False, shuffle=False)
    for b in ('g', 'r'):
        m = lc.bands == b
        expected = 16.0 + 0.6 * np.asarray(tmpl(lc.t[m] / 0.55 - 0.1))
        npt.assert_allclose(lc.y[m], expected)


# ----------------------------------------------------------------------
# from_cache: read a temp .npz (no network)
# ----------------------------------------------------------------------
def _write_cache_npz(path, mjd, band, magerr=None, catflags=None, metadata=None):
    mjd = np.asarray(mjd, dtype=float)
    band = np.asarray(band)
    if magerr is None:
        magerr = np.full(mjd.size, 0.03)
    if catflags is None:
        catflags = np.zeros(mjd.size, dtype=int)
    np.savez(path, mjd=mjd, hjd=mjd, mag=np.full(mjd.size, 16.0),
             magerr=magerr, catflags=catflags, band=band,
             metadata=np.array(metadata or {}, dtype=object))


def test_from_cache_roundtrip_and_catflags_filter():
    with tempfile.TemporaryDirectory() as d:
        mjd = np.concatenate([G_EPOCHS, R_EPOCHS])
        band = np.array(['g'] * G_EPOCHS.size + ['r'] * R_EPOCHS.size)
        # flag one g epoch and one r epoch as bad
        catflags = np.zeros(mjd.size, dtype=int)
        catflags[0] = 1
        catflags[G_EPOCHS.size] = 32768
        _write_cache_npz(os.path.join(d, 'g00007.npz'), mjd, band,
                         catflags=catflags, metadata=dict(ra=1.0, dec=2.0))
        cad = RealZTFCadence.from_cache('g00007', data_home=d, bands=['g', 'r'],
                                        catflags_max=0)
        counts = cad.epoch_counts()
        assert counts == {'g': G_EPOCHS.size - 1, 'r': R_EPOCHS.size - 1}
        assert cad.object_id == 'g00007'
        assert cad.metadata['ra'] == 1.0
        s = cad.sample()
        assert np.all(np.diff(s.t) >= 0)


def test_from_cache_missing_file_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            RealZTFCadence.from_cache('nope', data_home=d)


def test_from_cache_band_filter_rejects_absent_band():
    with tempfile.TemporaryDirectory() as d:
        _write_cache_npz(os.path.join(d, 'g0.npz'), G_EPOCHS,
                         np.array(['g'] * G_EPOCHS.size))
        with pytest.raises(ValueError):
            RealZTFCadence.from_cache('g0', data_home=d, bands=['i'])
