import numpy as np

from gatspy.periodic import LombScargle

from numpy.testing import assert_allclose
import pytest

from ..slow_template_periodogram import SlowTemplatePeriodogram
from ..template import Template


def generate_template(nharmonics, rseed=0):
    rng = np.random.RandomState(rseed)
    c_n, s_n = rng.randn(2, nharmonics) * 1. / np.arange(1, nharmonics + 1)
    return Template(c_n, s_n)


def generate_data(template, N, tmin, tmax, freq, dy=0.1,
                  phase=0, amp=1, offset=10, rseed=0):
    rng = np.random.RandomState(rseed)
    t = tmin + (tmax - tmin) * np.random.rand(N)
    y = offset + amp * template(t * freq - phase) + dy * rng.randn(N)
    return t, y, dy


def test_vs_lombscargle():
    # one-component template should be identical to a
    # Lomb-Scargle periodogram with floating-mean
    template = generate_template(1)
    t, y, dy = generate_data(template, N=100, tmin=0, tmax=100, freq=0.1)

    freq = np.linspace(0.01, 1, 10)
    power1 = SlowTemplatePeriodogram(t, y, dy, template=template).power(freq)
    power2 = LombScargle(fit_offset=True).fit(t, y, dy).periodogram(1. / freq)

    assert_allclose(power1, power2)


@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4])
def test_zero_noise(nharmonics):
    # in the zero-noise perfect template case, the true frequency should
    # have power = 1
    template = generate_template(nharmonics)
    t, y, _ = generate_data(template, N=100, tmin=0, tmax=100, freq=0.1, dy=0)
    dy = None
    power = SlowTemplatePeriodogram(t, y, dy, template=template).power(0.1)
    assert_allclose(power, 1)
