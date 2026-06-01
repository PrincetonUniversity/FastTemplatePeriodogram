"""Keystone integration tests: simulate -> FTP -> recovery (Phase 3.2 harness).

These tie the three new harness layers together against the real modeler.  The
search grid is an explicit ``[f_min, f_max]`` (no nyquist_factor) with a documented
spacing ``df = 1/(T * samples_per_peak)`` fine enough that grid resolution is not
the limiting factor -- the noise-free test asserts recovery *to grid resolution*.
"""
import numpy as np
import numpy.testing as npt

from ftperiodogram.template import Template
from ftperiodogram import simulate as sim
from ftperiodogram import FastTemplatePeriodogram
from ftperiodogram.recovery import classify_recovery


def _rrl_like_template():
    # a non-sinusoidal, multi-harmonic shape (the regime where templates help)
    return Template([1.0, 0.4, 0.2], [0.0, 0.1, 0.05])


def test_noise_free_recovers_injected_period_to_grid_resolution():
    tmpl = _rrl_like_template()
    P_true = 0.6
    cad = sim.SyntheticCadence(n_epochs=60, baseline_days=365.25,
                               random_state=11)
    lc = sim.simulate_lightcurve(tmpl, P_true, cad, amplitude=0.8, mean_mag=15.0,
                                 random_state=11, add_noise=False)
    T = lc.t.max() - lc.t.min()

    spp = 8
    model = FastTemplatePeriodogram(template=tmpl).fit(lc.t, lc.y, lc.dy)
    freqs, power = model.autopower(minimum_frequency=0.5, maximum_frequency=2.5,
                                   samples_per_peak=spp, fast=True)
    df = freqs[1] - freqs[0]
    npt.assert_allclose(df, 1.0 / (T * spp), rtol=1e-6)   # documented grid spacing

    f_peak = freqs[int(np.argmax(power))]
    assert abs(f_peak - 1.0 / P_true) <= df               # recovered to grid resolution

    r = classify_recovery(1.0 / f_peak, P_true, baseline=T, criterion='fractional')
    assert r.exact is True


def test_simulate_then_classify_high_snr_noisy():
    tmpl = _rrl_like_template()
    P_true = 0.55
    cad = sim.SyntheticCadence(n_epochs=80, baseline_days=365.25,
                               random_state=21)
    lc = sim.simulate_lightcurve(tmpl, P_true, cad, amplitude=1.0, mean_mag=14.0,
                                 random_state=22, add_noise=True)
    T = lc.t.max() - lc.t.min()

    model = FastTemplatePeriodogram(template=tmpl).fit(lc.t, lc.y, lc.dy)
    freqs, power = model.autopower(minimum_frequency=0.5, maximum_frequency=2.5,
                                   samples_per_peak=8, fast=True)
    P_rec = 1.0 / freqs[int(np.argmax(power))]

    r = classify_recovery(P_rec, P_true, baseline=T)
    assert r.recovered is True
