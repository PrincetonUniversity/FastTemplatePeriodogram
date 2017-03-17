"""Tests fast summation utilities against direct sums"""
import numpy as np
from ..summations import fast_summations, shift_t_for_nfft, direct_summations
from ..utils import weights

from numpy.testing import assert_allclose
from scipy.interpolate import interp1d


import pytest

def template_function(phase,
                      c_n=[-0.181, -0.075, -0.020],
                      s_n=[-0.110, 0.000,  0.030]):
    n = 1 + np.arange(len(c_n))[:, np.newaxis]
    return (np.dot(c_n, np.cos(2 * np.pi * n * phase)) +
            np.dot(s_n, np.sin(2 * np.pi * n * phase)))


@pytest.fixture
def data(N=30, T=5, period=0.9, coeffs=(5, 10),
         yerr=0.1, rseed=42):
    rand = np.random.RandomState(rseed)
    t = T * rand.rand(N)
    y = coeffs[0] + coeffs[1] * template_function(t / period)
    y += yerr * rand.randn(N)
    return t, y, yerr * np.ones_like(y)

def get_frequencies(data, samples_per_peak, nyquist_factor):
  t, y, yerr = data
  df = 1. / (t.max() - t.min()) / samples_per_peak
  Nf = int(0.5 * samples_per_peak * nyquist_factor * len(t))
  return df * (1 + np.arange(Nf))


@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('samples_per_peak', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('nyquist_factor', [1, 2, 3, 4, 5])
def test_compute_summations_executes(nharmonics, data, samples_per_peak, nyquist_factor):
    t, y, yerr = data
    w = weights(yerr)
    freqs = get_frequencies(data, samples_per_peak, nyquist_factor)
    all_fast_sums  = fast_summations(t, y, w, freqs, nharmonics)


@pytest.mark.parametrize('nharmonics', [1])
#@pytest.mark.parametrize('samples_per_peak', [1, 2, 3, 4, 5])
#@pytest.mark.parametrize('nyquist_factor', [1, 2, 3, 4, 5])
def test_compute_summations_is_accurate(nharmonics, data):
    samples_per_peak, nyquist_factor = 1, 1
    t, y, yerr = data
    w = weights(yerr)
    freqs = get_frequencies(data, samples_per_peak, nyquist_factor)
    all_fast_sums =  fast_summations(t, y, w, freqs, nharmonics)

    for freq, fast_sums in zip(freqs, all_fast_sums):
        slow_sums = direct_summations(t, y, w, freq, nharmonics)

        assert_allclose(np.ravel(fast_sums.CC), np.ravel(slow_sums.CC))
        assert_allclose(np.ravel(fast_sums.CS), np.ravel(slow_sums.CS))
        assert_allclose(np.ravel(fast_sums.SS), np.ravel(slow_sums.SS))

        assert_allclose(fast_sums.YC, slow_sums.YC)
        assert_allclose(fast_sums.YS, slow_sums.YS)



