"""Tests of high-level methods for revised modeler class"""
import numpy as np
from ..modeler import FastTemplateModeler, FastMultiTemplateModeler
from ..template import Template
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


def shift_template(template, tau):
    return lambda phase, tau=tau : template(phase - tau)


def get_amplitude_and_offset(freq, template, data):
    """ 
    Obtain optimal amplitude and offset from shifted template.
    amplitude = E[(y-ybar) * M(omega * t)] / Var(M(omega * t))
    offset    = ybar - amplitude * E[M(omega * t)]
    """
    t, y, yerr = data

    w = weights(yerr)
    ybar = np.dot(w, y)

    M = template((t*freq)%1.0)
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

    t, y, yerr = data
    amp, offset = get_amplitude_and_offset(freq, template, data)

    M = template((t * freq)%1.0)

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

    return c_n, s_n



@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4, 5])
def test_fast_template_method(nharmonics, template, data):
    t, y, yerr = data
    phase, y_phase = template
    w = weights(yerr)

    chi2_0 = np.dot(w, (y - np.dot(w, y))**2)

    c_n, s_n = truncate_template(phase, y_phase, nharmonics)

    temp = Template(c_n, s_n)
    temp.precompute()

    model = FastTemplateModeler(template=temp)
    model.fit(t, y, yerr)
    freq_template, power_template = model.autopower(samples_per_peak=10, nyquist_factor=1)

  
    pdg = lambda freq : direct_periodogram(freq, temp, data)
    direct_power_template = np.array([ pdg(freq) for freq in freq_template ])

    #assert_allclose(power_template, direct_power_template)

    dp = np.absolute(power_template - direct_power_template)
    from scipy.stats import pearsonr
    r, pval = pearsonr(power_template, direct_power_template)
    
    inds = np.argsort(-dp)

    nbad = 0
    dp_crit = 5E-2
    for frq, ptmp, pdir, dpval in zip(freq_template[inds], power_template[inds], direct_power_template[inds], dp[inds]):
        if dpval > dp_crit:
            nbad += 1
            #print("%.5e %.5e %.5e %.5e %.5e"%(frq, ptmp, pdir, dpval, chi2_0))

    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt 
    #f, ax = plt.subplots()
    #ax.plot(freq_template, power_template, color='r', label='ftp')
    #ax.plot(freq_template, direct_power_template, color='k', label='direct')
    #ax.set_xlim(min(freq_template), max(freq_template))
    #ax.legend(loc='best')
    #f.savefig('compare.png')

    #print("pearson r correlation = %e"%(r))
    #print("median err: %e, max err = %e, nbad = %d"%(np.median(dp), max(dp), nbad))

    assert(float(np.argmax(power_template) - np.argmax(direct_power_template)) / len(power_template) < 0.01)
    assert(max(dp) < 0.5)
    #assert(nbad < max([ int(0.05 * len(dp)), 2 ]))
    assert(np.median(dp) < 5E-3)


@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4, 5])
def test_best_model(nharmonics, template, data):
    samples_per_peak, nyquist_factor = 4, 1

    t, y, yerr = data
    phase, y_phase = template
    w = weights(yerr)
    ybar = np.dot(w, y)

    chi2_0 = np.dot(w, (y - ybar)**2)
    c_n, s_n = truncate_template(phase, y_phase, nharmonics)
    temp = Template(c_n, s_n)

    modeler = FastTemplateModeler(template=temp).fit(t, y, yerr)
    freq, P = modeler.autopower(samples_per_peak = samples_per_peak,
                                nyquist_factor = nyquist_factor,
                               save_best_model = True)

    y_model = modeler.best_model(t)
    chi2_m = np.dot(w, (y - y_model)**2)

    P_max = 1 - chi2_m / chi2_0

    assert( abs(P_max - max(P)) < 1E-3 * max(P) )


@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4, 5])
def test_multi_template_method(nharmonics, template, data):
    t, y, yerr = data
    phase, y_phase = template
    w = weights(yerr)

    c_n, s_n = truncate_template(phase, y_phase, nharmonics)
    c_n2, s_n2 = c_n + np.random.rand(len(c_n)), s_n + np.random.rand(len(c_n))


    temp = Template(c_n, s_n)
    temp2 = Template(c_n2, s_n2)

    temp.precompute()
    temp2.precompute()

    model1 = FastTemplateModeler(template=temp).fit(t, y, yerr)
    model2 = FastTemplateModeler(template=temp2).fit(t, y, yerr)

    freq, p1 = model1.autopower(samples_per_peak=10, nyquist_factor=1, save_best_model=False)
    freq, p2 = model2.autopower(samples_per_peak=10, nyquist_factor=1, save_best_model=False)

    model = FastMultiTemplateModeler(templates=[ temp, temp2 ])
    model.fit(t, y, yerr)
    freq_template, p_multi = model.autopower(samples_per_peak=10, nyquist_factor=1, save_best_model=False)

    #for P1, P2, Pmult in zip(p1, p2, p_multi):
    #    print("Pmult={0}, P1={1}, P2={2}".format(Pmult, P1, P2))
    #    assert(abs(Pmult - max([ P1, P2 ])) < 1E-3*Pmult)
