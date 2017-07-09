"""Tests of high-level methods for revised modeler class"""
import numpy as np

from ..modeler import FastTemplatePeriodogram, FastMultiTemplatePeriodogram, TemplateModel
from ..template import Template
from ..utils import weights, ModelFitParams
from ..core import fit_template

from numpy.testing import assert_allclose
import pytest

#nharms_to_test = [ 1, 2, 3, 4, 5 ]
nharms_to_test = [ 1, 3 ]
ndata_to_test  = [ 30 ]
samples_per_peak_to_test = [ 1, 2 ]
nyquist_factors_to_test = [ 1, 2 ]
rseeds_to_test = [ 42 ]
ndata0 = ndata_to_test[0]


def template_function(phase,
                      c_n=[-0.181, -0.1, -0.020],
                      s_n=[-0.110, 0.000,  0.030]):
    n = 1 + np.arange(len(c_n))[:, np.newaxis]
    return (np.dot(c_n, np.cos(2 * np.pi * n * phase)) +
            np.dot(s_n, np.sin(2 * np.pi * n * phase)))


@pytest.fixture
def template():
    phase = np.linspace(0, 1, 100, endpoint=False)
    return phase, template_function(phase)


@pytest.fixture
def data(N=ndata0, T=2, period=0.9, coeffs=(5, 10),
         yerr=0.0001, rseed=42):
    rand = np.random.RandomState(rseed)
    t = np.sort(T * rand.rand(N))
    y = coeffs[0] + coeffs[1] * template_function(t / period)
    y += yerr * rand.randn(N)
    return t, y, yerr * np.ones_like(y)

def data_from_template(template, parameters, N=ndata0, T=2,
                                 period=0.9, yerr=0.0001, rseed=42):

    rand = np.random.RandomState(rseed)
    model = TemplateModel(template, parameters=parameters, frequency=(1./period))

    t = np.sort(T * rand.rand(N))
    y = model(t) + yerr * rand.randn(N)

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


def direct_periodogram(freq, template, data, nshifts=50):
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

def convert_template(temp, nharmonics):
    phase, y_phase = temp
    c_n, s_n = truncate_template(phase, y_phase, nharmonics)

    template = Template(c_n, s_n)
    return template

def get_ftau(parameters):
    omega_tau = np.arccos(parameters.b)
    if parameters.sgn < 0:
        omega_tau = 2 * np.pi - omega_tau

    return (omega_tau / (2 * np.pi))%1.0


def pdg(data, y_fit):
    t, y, yerr = data
    w = weights(yerr)

    ybar = np.dot(w, y)

    chi2 = np.dot(w, (y - y_fit)**2)
    chi2_0 = np.dot(w, (y - ybar)**2)

    return 1 - chi2 / chi2_0


@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('samples_per_peak', samples_per_peak_to_test)
@pytest.mark.parametrize('nyquist_factor', nyquist_factors_to_test)
def test_fast_template_method(nharmonics, template, data, samples_per_peak, nyquist_factor):
    t, y, yerr = data
    phase, y_phase = template
    w = weights(yerr)

    chi2_0 = np.dot(w, (y - np.dot(w, y))**2)

    c_n, s_n = truncate_template(phase, y_phase, nharmonics)

    temp = Template(c_n, s_n)

    model = FastTemplatePeriodogram(template=temp)
    model.fit(t, y, yerr)
    freq_template, power_template = model.autopower(samples_per_peak=samples_per_peak,
                                                      nyquist_factor=nyquist_factor)

    pdg = lambda freq : direct_periodogram(freq, temp, data)
    direct_power_template = np.array([ pdg(freq) for freq in freq_template ])


    dp = power_template - direct_power_template

    assert(all(np.sort(dp) > -1E-2))


@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('samples_per_peak', samples_per_peak_to_test)
@pytest.mark.parametrize('nyquist_factor', nyquist_factors_to_test)
def test_fast_vs_slow(nharmonics, template, data, samples_per_peak, nyquist_factor):
    t, y, yerr = data
    phase, y_phase = template
    w = weights(yerr)

    chi2_0 = np.dot(w, (y - np.dot(w, y))**2)

    c_n, s_n = truncate_template(phase, y_phase, nharmonics)

    temp = Template(c_n, s_n)

    model = FastTemplatePeriodogram(template=temp)
    model.fit(t, y, yerr)
    freq_template, power_template_fast = model.autopower(samples_per_peak=samples_per_peak,
                                                      nyquist_factor=nyquist_factor, fast=True)

    freq_template, power_template_slow = model.autopower(samples_per_peak=samples_per_peak,
                                                      nyquist_factor=nyquist_factor, fast=False)

    assert_allclose(power_template_fast, power_template_slow)


@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('samples_per_peak', samples_per_peak_to_test)
@pytest.mark.parametrize('nyquist_factor', nyquist_factors_to_test)
def test_best_model(nharmonics, template, data, samples_per_peak, nyquist_factor):

    t, y, yerr = data
    phase, y_phase = template
    w = weights(yerr)
    ybar = np.dot(w, y)

    chi2_0 = np.dot(w, (y - ybar)**2)
    c_n, s_n = truncate_template(phase, y_phase, nharmonics)
    temp = Template(c_n, s_n)

    modeler = FastTemplatePeriodogram(template=temp).fit(t, y, yerr)

    freq, P = modeler.autopower(samples_per_peak = samples_per_peak,
                                  nyquist_factor = nyquist_factor,

                               save_best_model = True)

    y_model = modeler.best_model(t)
    chi2_m = np.dot(w, (y - y_model)**2)

    P_max = 1 - chi2_m / chi2_0

    assert( max(P) - P_max > -1E-4 )


@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('samples_per_peak', samples_per_peak_to_test)
@pytest.mark.parametrize('nyquist_factor', nyquist_factors_to_test)
@pytest.mark.parametrize('rseed', rseeds_to_test)
def test_multi_template_method(nharmonics, template, data, samples_per_peak, nyquist_factor, rseed):

    t, y, yerr = data
    phase, y_phase = template
    w = weights(yerr)

    rand = np.random if rseed is None else np.random.RandomState(rseed)

    c_n, s_n = truncate_template(phase, y_phase, nharmonics)
    c_n2, s_n2 = c_n + rand.rand(len(c_n)), s_n + rand.rand(len(c_n))


    temp = Template(c_n, s_n)
    temp2 = Template(c_n2, s_n2)

    model1 = FastTemplatePeriodogram(template=temp).fit(t, y, yerr)
    model2 = FastTemplatePeriodogram(template=temp2).fit(t, y, yerr)

    freq, p1 = model1.autopower(samples_per_peak=samples_per_peak,
                     nyquist_factor=nyquist_factor, save_best_model=False)
    freq, p2 = model2.autopower(samples_per_peak=samples_per_peak,
                     nyquist_factor=nyquist_factor, save_best_model=False)

    model = FastMultiTemplatePeriodogram(templates=[ temp, temp2 ])
    model.fit(t, y, yerr)
    freq_template, p_multi = model.autopower(samples_per_peak=samples_per_peak,
                     nyquist_factor=nyquist_factor, save_best_model=False)

    for P1, P2, Pmult in zip(p1, p2, p_multi):
        #print("Pmult={0}, P1={1}, P2={2}".format(Pmult, P1, P2))
        assert(abs(Pmult - max([ P1, P2 ])) < 1E-4)


@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('samples_per_peak', samples_per_peak_to_test)
@pytest.mark.parametrize('nyquist_factor', nyquist_factors_to_test)
@pytest.mark.parametrize('rseed', rseeds_to_test)
def test_autopower_and_power_are_consistent(nharmonics, template,
                           samples_per_peak, nyquist_factor,rseed,  data):
    t, y, yerr = data
    temp = convert_template(template, nharmonics)

    rand = np.random if rseed is None else np.random.RandomState(rseed)

    temp2 = Template(c_n = rand.rand(nharmonics),
                     s_n = rand.rand(nharmonics))

    modeler_single = FastTemplatePeriodogram(template=temp).fit(t, y, yerr)

    # MULTI-template
    modeler_multi = FastMultiTemplatePeriodogram(templates=[temp, temp2])
    modeler_multi.fit(t, y, yerr)

    for modeler in [ modeler_single, modeler_multi ]:
        freqs, p_auto = modeler.autopower(samples_per_peak=samples_per_peak,
                                            nyquist_factor=nyquist_factor)

        p_single = modeler.power(freqs)

        inds = np.argsort(p_auto)[::-1]
        assert_allclose(p_auto[inds], p_single[inds], atol=1E-5)

@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('samples_per_peak', samples_per_peak_to_test)
@pytest.mark.parametrize('nyquist_factor', nyquist_factors_to_test)
@pytest.mark.parametrize('ndata', ndata_to_test)
@pytest.mark.parametrize('rseed', rseeds_to_test)
def test_best_model_and_fit_model_are_consistent(nharmonics, template, ndata,
                              rseed, nyquist_factor, samples_per_peak):


    temp = convert_template(template, nharmonics)
    parameters = ModelFitParams(a=1.25, b=0.75, c=0.5, sgn=1)
    t, y, yerr = data_from_template(temp, parameters, N=ndata, T=5,
                                 period=1.0, yerr=0.0001, rseed=rseed)

    rand = np.random if rseed is None else np.random.RandomState(rseed)

    temp2 = Template(c_n = rand.rand(nharmonics),
                     s_n = rand.rand(nharmonics))

    modeler_single = FastTemplatePeriodogram(template = temp).fit(t, y, yerr)

    modeler_multi = FastMultiTemplatePeriodogram(templates = [temp, temp2])
    modeler_multi.fit(t, y, yerr)

    for modeler in [ modeler_single, modeler_multi ]:
        if modeler == modeler_multi and nharmonics == 1:
            continue

        freqs, p_auto = modeler.autopower(save_best_model=True,
                                 samples_per_peak=samples_per_peak,
                                   nyquist_factor=nyquist_factor)


        i = np.argmax(p_auto)
        fit_model_params = modeler.fit_model(freqs[i]).parameters._asdict()
        best_model_params = modeler.best_model.parameters._asdict()

        print(fit_model_params)
        print(best_model_params)
        print(parameters)
        for par in best_model_params.keys():

            if nharmonics == 1 and not par == 'c':
                assert(abs(abs(best_model_params[par]) \
                - abs(fit_model_params[par])) < 1E-5)

            else:
                assert(abs(best_model_params[par] \
                    - fit_model_params[par]) < 1E-5)

@pytest.mark.parametrize('nharmonics', nharms_to_test)
def test_errors_are_raised(nharmonics, template, data):
    t, y, yerr = data

    modeler = FastTemplatePeriodogram()

    with pytest.raises(ValueError):
        modeler.autopower()

    modeler.fit(t, y, yerr)

    with pytest.raises(ValueError):
        modeler.autopower()

    modeler.template = convert_template(template, nharmonics)

    freq, p = modeler.autopower()

    modeler = FastMultiTemplatePeriodogram()

    with pytest.raises(ValueError):
        modeler.autopower()

    modeler.fit(t, y, yerr)

    with pytest.raises(ValueError):
        modeler.autopower()

    #modeler.templates = [ convert_template(template, nharmonics) ]

    modeler.templates = []

    with pytest.raises(ValueError):
        modeler.autopower()

    modeler.templates.append(convert_template(template, nharmonics))

    freq, p = modeler.autopower()

@pytest.mark.parametrize('ndata', ndata_to_test)
@pytest.mark.parametrize('nharmonics', nharms_to_test)
@pytest.mark.parametrize('rseed', rseeds_to_test)
def test_inject_and_recover(nharmonics, ndata, rseed, period=1.2, tol=1E-2):

    rand = np.random if rseed is None else np.random.RandomState(rseed)

    a = 1 + rand.rand()
    b = 2 * rand.rand() - 1
    c = 1 + rand.rand()
    sgn   = np.sign(rand.rand() - 0.5)

    parameters = ModelFitParams(a=a, b=b, c=c, sgn=sgn)

    c_n, s_n = rand.rand(nharmonics), rand.rand(nharmonics)

    template = Template(c_n=c_n, s_n=s_n)

    t, y, yerr = data_from_template(template, parameters, period=period,
                                         yerr=0.0001, rseed=rseed)

    max_p, best_pars = fit_template(t, y, yerr, template.c_n, template.s_n, 1./period,
                                    allow_negative_amplitudes=True)

    # if nharmonics == 1 and best_pars.a < 0:
    #     best_pars = ModelFitParams(a=-best_pars.a, b=-best_pars.b, c=best_pars.c, sgn=-best_pars.sgn)


    signal = TemplateModel(template, parameters=parameters, frequency=1./period)
    model  = TemplateModel(template, parameters=best_pars, frequency=1./period)

    signal_power = pdg((t, y, yerr), signal(t))
    fit_power = pdg((t, y, yerr), model(t))

    # no absolute value here, its possible to find better fit with noise
    assert(signal_power - fit_power < tol)

    assert(abs(abs(best_pars.a) - abs(parameters.a)) / parameters.a < tol)
    assert(abs(best_pars.c - parameters.c) / parameters.c < tol)

    ftau     = get_ftau(parameters)
    ftau_fit = get_ftau(best_pars)

    dftau = ftau - ftau_fit
    # if abs(dftau) > 0.5:
    #     dftau -= np.sign(dftau) * 0.5
    assert(dftau < tol)
