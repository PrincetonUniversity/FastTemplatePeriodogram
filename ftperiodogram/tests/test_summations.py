"""Tests fast summation utilities against direct sums"""
import numpy as np
from ..summations import fast_summations, direct_summations
from ..utils import weights

from numpy.testing import assert_allclose


import pytest

nharms_to_test = [ 1, 3, 5 ]
ndata_to_test  = [ 10, 30 ]
samples_per_peak_to_test = [ 1, 3 ]
nyquist_factors_to_test = [ 1, 2 ]
rseeds_to_test = [ 42 ]

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
    t = np.sort(T * rand.rand(N))
    y = coeffs[0] + coeffs[1] * template_function(t / period)
    y += yerr * rand.randn(N)
    return t, y, yerr * np.ones_like(y)

def get_frequencies(data, samples_per_peak, nyquist_factor):
    t, y, yerr = data
    df = 1. / (t.max() - t.min()) / samples_per_peak
    Nf = int(0.5 * samples_per_peak * nyquist_factor * len(t))
    return df * (1 + np.arange(Nf))


@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('samples_per_peak', samples_per_peak_to_test)
@pytest.mark.parametrize('nyquist_factor', nyquist_factors_to_test)
def test_fast_vs_slow(nharmonics, data, samples_per_peak, nyquist_factor):
    t, y, yerr = data
    w = weights(yerr)
    freqs = get_frequencies(data, samples_per_peak, nyquist_factor)

    all_fast_sums =   fast_summations(t, y, w, freqs, nharmonics)
    all_slow_sums = direct_summations(t, y, w, freqs, nharmonics)

    for freq, fast_sums, slow_sums in zip(freqs, all_fast_sums, all_slow_sums):

        assert_allclose(np.ravel(fast_sums.CC), np.ravel(slow_sums.CC))
        assert_allclose(np.ravel(fast_sums.CS), np.ravel(slow_sums.CS))
        assert_allclose(np.ravel(fast_sums.SS), np.ravel(slow_sums.SS))

        assert_allclose(fast_sums.YC, slow_sums.YC)
        assert_allclose(fast_sums.YS, slow_sums.YS)

        assert_allclose(fast_sums.C, slow_sums.C)
        assert_allclose(fast_sums.S, slow_sums.S)


@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('samples_per_peak', samples_per_peak_to_test)
@pytest.mark.parametrize('nyquist_factor', nyquist_factors_to_test)
def test_covariance_matrices_are_symmetric(nharmonics, data, samples_per_peak, nyquist_factor):
    t, y, yerr = data
    w = weights(yerr)
    freqs = get_frequencies(data, samples_per_peak, nyquist_factor)
    all_fast_sums =  fast_summations(t, y, w, freqs, nharmonics)

    for freq, fast_sums in zip(freqs, all_fast_sums):
        assert_allclose(fast_sums.CC, fast_sums.CC.T)
        assert_allclose(fast_sums.SS, fast_sums.SS.T)
