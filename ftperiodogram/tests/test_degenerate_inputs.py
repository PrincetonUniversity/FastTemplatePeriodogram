"""Degenerate-input regression tests: constant y, single-point input, and
zero-YM / NaN-safe root selection must return P = 0, never crash
(AUDIT_2026-06-10 / WP A3: "argmax of an empty sequence", NaN argmax hijack).
"""
import numpy as np
import numpy.polynomial as pol
import pytest

from ..core import roots_from_YM_MM
from ..modeler import FastTemplatePeriodogram
from ..template import Template
from ..utils import ModelFitParams


@pytest.fixture
def template():
    return Template([0.5, 0.2], [0.3, 0.1])


@pytest.fixture
def freqs():
    return np.linspace(0.1, 1.0, 25)


def _times(n, rseed=0):
    t = np.sort(np.random.RandomState(rseed).rand(n) * 10)
    return t


def test_constant_y(template, freqs):
    # constant y => YM is the zero polynomial, no candidate roots
    t = _times(30)
    y = np.full_like(t, 5.0)
    dy = 0.1 * (1 + np.arange(len(t)) % 3)
    power = FastTemplatePeriodogram(template=template).fit(t, y, dy).power(freqs)
    assert np.all(np.isfinite(power))
    assert np.allclose(power, 0.0)


def test_constant_y_equal_dy(template, freqs):
    t = _times(30)
    y = np.full_like(t, 5.0)
    dy = np.full_like(t, 0.1)
    power = FastTemplatePeriodogram(template=template).fit(t, y, dy).power(freqs)
    assert np.all(np.isfinite(power))
    assert np.allclose(power, 0.0)


def test_single_point(template, freqs):
    power = FastTemplatePeriodogram(template=template).fit(
        np.array([0.5]), np.array([1.0]), np.array([0.1])).power(freqs)
    assert np.all(np.isfinite(power))
    assert np.allclose(power, 0.0)


def test_zero_ym_direct(template):
    # call the root selector directly with a zero YM: flat zero-power fit
    ybar, YY = 5.0, 1.0
    H = 2
    YM = pol.Polynomial([0.0])
    MM = pol.Polynomial([1.0, 0.5, 0.25, 0.5, 1.0])
    AC = np.zeros(2 * H)
    params, power, best_phi = roots_from_YM_MM(YM, MM, AC, H, ybar, YY)
    assert power == 0.0
    assert params == ModelFitParams(a=0.0, b=1.0, c=ybar, sgn=1.0)
    assert np.abs(best_phi) == pytest.approx(1.0)


def test_nan_power_does_not_hijack_argmax(template):
    # YY = 0 makes every candidate power non-finite; the guard must return
    # the flat zero-power fit, not select a NaN/inf root
    ybar, YY = 5.0, 0.0
    H = 2
    YM = pol.Polynomial([0.1, 0.2, 0.05, 0.2, 0.1])
    MM = pol.Polynomial([1.0, 0.5, 0.25, 0.5, 1.0])
    AC = np.zeros(2 * H)
    params, power, best_phi = roots_from_YM_MM(YM, MM, AC, H, ybar, YY)
    assert power == 0.0
    assert params.a == 0.0 and params.c == ybar


def test_nondegenerate_unchanged(template, freqs):
    # sanity: the guards must not perturb a normal fit
    rng = np.random.RandomState(42)
    t = _times(40, rseed=1)
    y = 10 + template(t * 0.4) + 0.1 * rng.randn(len(t))
    dy = np.full_like(t, 0.1)
    power = FastTemplatePeriodogram(template=template).fit(t, y, dy).power(freqs)
    assert np.all(np.isfinite(power))
    assert np.all(power > -1e-9) and np.all(power < 1 + 1e-9)
    assert power.max() > 0.5
