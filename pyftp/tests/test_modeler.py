"""Tests of high-level methods for revised modeler class"""
import numpy as np
from ..modeler import FastTemplateModeler
from ..modeler import Template
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
def template():
    phase = np.linspace(0, 1, 100, endpoint=False)
    return phase, template_function(phase)


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


def shift_template(template, tau):
    """
    returns new shifted template:
    M_shift(t) = M(t - tau)
    """
    phase, y_phase = template
    tfunc = interp1d(phase, y_phase)
    new_y = tfunc((phase - tau)%1.0)

    return (phase, new_y)

def get_amplitude_and_offset(freq, template, data):
    """ 
    Obtain optimal amplitude and offset from shifted template.
    amplitude = E[(y-ybar) * M(omega * t)] / Var(M(omega * t))
    offset    = ybar - amplitude * E[M(omega * t)]
    """

    phase, y_phase = template
    t, y, yerr = data

    tfunc = interp1d(phase, y_phase)
    w = weights(yerr)
    ybar = np.dot(w, y)

    M = tfunc((t*freq)%1.0)
    covym = np.dot(w, np.multiply(y - ybar, M))
    varm = np.dot(w, np.multiply(M, M))

    amplitude = covym / varm
    offset = ybar - amplitude * np.dot(w, M)

    return amplitude, offset

def chi2_template_fit(freq, template, data):
    """
    obtain the chi2 (weighted sum of squared residuals)
    for a given (shifted) template
    """

    phase, y_phase = template
    t, y, yerr = data
    amp, offset = get_amplitude_and_offset(freq, template, data)

    tfunc = interp1d(phase, y_phase)
    p = (t * freq)%1.0
    
    M = tfunc(p)

    w = weights(yerr)

    return np.dot(w, (y - amp * M - offset)**2)

def direct_periodogram(freq, template, data, nshifts=100):
    """
    computes periodogram at a given frequency directly, 
    using grid search for the best phase shift
    """

    taus = np.linspace(0, 1, nshifts)
    t, y, yerr = data
    w = weights(yerr)

    chi2_0 = np.dot(w, (y - np.dot(w, y))**2)


    chi2_tau = lambda tau : chi2_template_fit(freq, shift_template(template, tau), data)
    chi2s = [ chi2_tau(tau) for tau in taus ]

    return 1. - min(chi2s) / chi2_0

def truncate_template(phase, y, nharmonics):
    fft = np.fft.fft(y[::-1])
    c_n, s_n = zip(*[ (p.real/len(phase), p.imag/len(phase)) for i,p in enumerate(fft) \
                     if i > 0 and i <= int(nharmonics) ])

    return phase, template_function(phase, c_n=c_n, s_n=s_n)

@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4, 5])
def test_fast_template_method(nharmonics, template, data):
    t, y, yerr = data
    phase, y_phase = template

    phase_trunc, y_phase_trunc = truncate_template(phase, y_phase, nharmonics)

    template = Template(phase=phase, y=y_phase, nharmonics=nharmonics)
    template.precompute()

    model = FastTemplateModeler(templates=template)
    model.fit(t, y, yerr)
    freq_template, power_template = model.periodogram(ofac=1, hfac=1)

    if max(phase_trunc) < 1:
        phase_trunc = [ p for p in phase_trunc ]
        phase_trunc.append(1)
        y_phase_trunc = [ Y for Y in y_phase_trunc ]
        y_phase_trunc.append(y_phase_trunc[0])

    
    pdg = lambda freq : direct_periodogram(freq, (phase_trunc, y_phase_trunc), data)
    direct_power_template = np.array([ pdg(freq) for freq in freq_template ])

    dp = np.absolute(power_template - direct_power_template)
    print("max diff = %e, mean diff = %e, median diff = %e"%(max(dp), np.mean(dp), np.median(dp)))
    assert(np.median(dp) < 1E-2)

