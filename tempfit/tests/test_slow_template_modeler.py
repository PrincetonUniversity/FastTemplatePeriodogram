import numpy as np

from numpy.testing import assert_allclose
import pytest

try:
    from astropy.stats import LombScargle as astropy_LombScargle
except ImportError:
    astropy_LombScargle = None

from ..modeler import SlowTemplateModeler, FastTemplateModeler
from ..template import Template


def generate_template(nharmonics, rseed=0):
    rng = np.random.RandomState(rseed)
    c_n, s_n = rng.randn(2, nharmonics) * 1. / np.arange(1, nharmonics + 1)
    return Template(c_n, s_n)


def generate_data(template, N, tmin, tmax, freq, dy=0.1,
                  phase=0, amp=1, offset=10, rseed=0):
    rng = np.random.RandomState(rseed)
    t = tmin + (tmax - tmin) * np.random.rand(N)
    t.sort()
    y = offset + amp * template(t * freq - phase) + dy * rng.randn(N)
    return t, y, dy


@pytest.mark.skipif(not astropy_LombScargle, reason="astropy is not installed")
def test_vs_lombscargle():
    # one-component template should be identical to a
    # Lomb-Scargle periodogram with floating-mean
    template = generate_template(1)
    t, y, dy = generate_data(template, N=100, tmin=0, tmax=100, freq=0.1)

    freq = np.linspace(0.01, 1, 10)
    power1 = SlowTemplateModeler(template=template).fit(t, y, dy).power(freq)
    power2 = astropy_LombScargle(t, y, dy, fit_mean=True).power(freq)

    assert_allclose(power1, power2)


@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4])
def test_zero_noise(nharmonics):
    # in the zero-noise perfect template case, the true frequency should
    # have power = 1
    template = generate_template(nharmonics)
    t, y, _ = generate_data(template, N=100, tmin=0, tmax=100, freq=0.1, dy=0)
    dy = None
    power = SlowTemplateModeler(template=template).fit(t, y, dy).power(0.1)
    assert_allclose(power, 1)
