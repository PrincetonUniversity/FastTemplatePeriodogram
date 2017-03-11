"""Tests fast summation utilities against direct sums"""
import numpy as np
from ..fast_template_periodogram import compute_summations, shift_t_for_nfft
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

def weights(yerr):
    assert(all(yerr > 0))

    w = np.power(yerr, -2)
    return w / np.sum(w)

def direct_summations(freq, nharmonics, data):
    """ for testing against NFFT implementation """

    t0, y, yerr = data 

    # shift the data to [-1/2, 1/2); then scale so that
    # frequency * t gives expected phase
    t = shift_t_for_nfft(t0, 1) * (max(t0) - min(t0))

    w = weights(yerr)
    ybar = np.dot(w, y)


    YC = np.array([ np.dot(w, np.multiply(y-ybar, np.cos(2 * np.pi * freq * t * (h+1))))\
                                 for h in range(nharmonics) ])

    YS = np.array([ np.dot(w, np.multiply(y-ybar, np.sin(2 * np.pi * freq * t * (h+1))))\
                                 for h in range(nharmonics) ])

    C = np.array([ np.dot(w, np.cos(2 * np.pi * freq * t * (h+1)))\
                                 for h in range(nharmonics) ])

    S = np.array([ np.dot(w, np.sin(2 * np.pi * freq * t * (h+1)))\
                                 for h in range(nharmonics) ])

    CC = np.zeros((nharmonics, nharmonics))
    CS = np.zeros((nharmonics, nharmonics))
    SS = np.zeros((nharmonics, nharmonics))

    for h1 in range(nharmonics):
        for h2 in range(nharmonics):
            CC[h1][h2] = np.dot(w, np.multiply(np.cos(2 * np.pi * freq * t * (h1+1)), 
                                               np.cos(2 * np.pi * freq * t * (h2+1))))
                                          
            CS[h1][h2] = np.dot(w, np.multiply(np.cos(2 * np.pi * freq * t * (h1+1)), 
                                               np.sin(2 * np.pi * freq * t * (h2+1)))) 

            SS[h1][h2] = np.dot(w, np.multiply(np.sin(2 * np.pi * freq * t * (h1+1)), 
                                               np.sin(2 * np.pi * freq * t * (h2+1))))
                                          
            CC[h1][h2] -= C[h1] * C[h2]
            CS[h1][h2] -= C[h1] * S[h2]
            SS[h1][h2] -= S[h1] * S[h2]

    return YC, YS, CC, CS, SS

@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4, 5])
def test_compute_summations_executes(nharmonics, data):
    t, y, yerr = data


    omegas, all_fast_sums, YY, w, ybar \
           = compute_summations(t, y, yerr, nharmonics, ofac=1, hfac=1)


@pytest.mark.parametrize('nharmonics', [1 ])
def test_compute_summations_is_accurate(nharmonics, data):
    t, y, yerr = data


    omegas, all_fast_sums, YY, w, ybar \
           = compute_summations(t, y, yerr, nharmonics, ofac=1, hfac=1)

    for omega, fast_sums in zip(omegas, all_fast_sums):
        YC, YS, CC, CS, SS = direct_summations(omega / (2 * np.pi), nharmonics, data)

        assert_allclose(np.ravel(fast_sums.CC), np.ravel(CC))
        assert_allclose(np.ravel(fast_sums.CS), np.ravel(CS))
        assert_allclose(np.ravel(fast_sums.SS), np.ravel(SS))

        assert_allclose(fast_sums.YC, YC)
        assert_allclose(fast_sums.YS, YS)



