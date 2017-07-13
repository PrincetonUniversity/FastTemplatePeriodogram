import numpy as np

from numpy.testing import assert_allclose, assert_array_less
import pytest

try:
    from astropy.stats import LombScargle as astropy_LombScargle
except ImportError:
    astropy_LombScargle = None

from ..modeler import SlowTemplatePeriodogram, FastTemplatePeriodogram
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
    dy = dy * np.ones_like(y)
    return t, y, dy


@pytest.mark.skipif(not astropy_LombScargle, reason="astropy is not installed")
def test_vs_lombscargle():
    # one-component template should be identical to a
    # Lomb-Scargle periodogram with floating-mean
    template = generate_template(1)
    t, y, dy = generate_data(template, N=50, tmin=0, tmax=100, freq=0.1)

    freq = np.linspace(0.01, 1, 10)
    power1 = SlowTemplatePeriodogram(template=template).fit(t, y, dy).power(freq)
    power2 = astropy_LombScargle(t, y, dy, fit_mean=True).power(freq)

    assert_allclose(power1, power2)


@pytest.mark.parametrize('nharmonics', [1, 2, 3])
@pytest.mark.parametrize('nguesses', [None, 2])
def test_zero_noise(nharmonics, nguesses):
    # in the zero-noise perfect template case, the true frequency should
    # have power = 1
    template = generate_template(nharmonics)
    t, y, _ = generate_data(template, N=50, tmin=0, tmax=100, freq=0.1, dy=0)
    dy = None
    power = SlowTemplatePeriodogram(nguesses=nguesses, template=template).fit(t, y, dy).power(0.1)
    assert_allclose(power, 1)


@pytest.mark.parametrize('nharmonics', [1, 2, 3])
@pytest.mark.parametrize('nguesses', [None, 2])
def test_slow_vs_fast(nharmonics, nguesses):
    # Slow periodogram has convergence issues, and will undershoot the power
    # in some cases. This means the fast periodogram should always be larger
    # than the slow periodogram, up to a small floating point error.
    template = generate_template(nharmonics)
    t, y, dy = generate_data(template, N=50, tmin=0, tmax=100, freq=0.1)
    freq = 0.01 * np.arange(1, 101)

    power_slow = SlowTemplatePeriodogram(template, nguesses=nguesses).fit(t, y, dy).power(freq)
    power_fast = FastTemplatePeriodogram(template).fit(t, y, dy).power(freq)

    assert(all(power_fast + 1E-5 >= power_slow))
    """
    inds = np.arange(len(freq))
    mask = power_slow > power_fast + 1E-4
    for i in inds[mask]:
        p_slow = power_slow[i]
        p_fast = power_fast[i]
        print("frequency %d (%.3e) is problematic because (p_slow = %e) > (p_fast = %e) + 1E-4"%((i, 
                     freq[i], p_slow, p_fast)))
        print("          p_slow is %.e times larger than p_fast at this frequency"%(p_slow/p_fast - 1))

    # occaisionally (and randomly) there will be one frequency for which
    # the slow periodogram finds a slightly better solution; I've hacked around
    # that for now since it seems like a relatively minor problem, but it may
    # hint at a larger issue...
    assert(all(np.sort(power_fast - power_slow)[1:] > -1E-4))
    #assert_array_less(power_slow, power_fast + 1E-4)
    """