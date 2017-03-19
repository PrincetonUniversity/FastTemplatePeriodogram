import pyftp.pseudo_poly as ppol
from pyftp.modeler import FastTemplateModeler, TemplateModel
from pyftp.utils import weights, ModelFitParams, Summations, Avec, Bvec
from pyftp.summations import direct_summations
from pyftp.template import Template
from pyftp.periodogram import get_a_from_b
import numpy as np 
import matplotlib.pyplot as plt

def get_poly(t, y, dy, template, freq, allow_negative_amplitudes=True):
    H = len(template.c_n)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar)**2)
    
    sums = direct_summations(t, y, weights(dy), freq, H)
    
    return ppol.get_poly_coeffs(template.ptensors, sums)

def pdg(data, y_fit):
    t, y, yerr = data
    w = weights(yerr)

    ybar = np.dot(w, y)

    chi2 = np.dot(w, (y - y_fit)**2)
    chi2_0 = np.dot(w, (y - ybar)**2)

    return 1 - chi2 / chi2_0



def get_poly_vs_pdg(data, template, freq, nharmonics):
    t, y, dy = data
    w = weights(dy)
    poly = get_poly(t, y, dy, template, freq)
    H = nharmonics

    bvals = np.linspace(-1, 1, 200)

    poly_vals = np.polyval(poly, bvals)
    pdg_vals_pos = []
    pdg_vals_neg = []

    sums = direct_summations(t, y, w, freq, nharmonics)

    fit_params_pos = []
    fit_params_neg = []
    for b in bvals:
        for sgn, pdg_vals, fit_params in zip([-1, 1], [pdg_vals_neg, pdg_vals_pos], 
                                                      [fit_params_neg, fit_params_pos]):
            A = Avec(b, template.c_n, template.s_n, sgn=sgn)
            B = Bvec(b, template.c_n, template.s_n, sgn=sgn)

            AYCBYS = np.dot(A, sums.YC[:H]) + np.dot(B, sums.YS[:H])
            ACBS   = np.dot(A, sums.C[:H])  + np.dot(B, sums.S[:H])

            # Obtain amplitude for a given b=cos(wtau) and sign(sin(wtau))
            a = get_a_from_b(b, template.c_n, template.s_n, sums, A=A, B=B, AYCBYS=AYCBYS)

            c = np.dot(w, y) - a * ACBS

            params = ModelFitParams(a=a, b=b, c=c, sgn=sgn)
            model = TemplateModel(template, frequency=freq, 
                               parameters=params)

            fit_params.append(params)
            y_fit = model(t)

            pdg_vals.append(pdg(data, y_fit))

    f, ax = plt.subplots()
    ax.plot(bvals, pdg_vals_pos, color='b', label='P(w) given b (sinwtau > 0)')
    ax.plot(bvals, pdg_vals_neg, color='r', label='P(w) given b (sinwtau < 0)')

    #poly_mad = np.median(np.absolute(poly_vals - np.median(poly_vals)))
    poly_mad = 1.
    ax.plot(bvals, poly_vals/poly_mad, color='k', label='polynomial')
    ax.axhline(0, ls=':', color='0.5')
    ax.legend(loc='best')

    

    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('$\\cos\\omega\\tau$')
    plt.show()

if __name__ == '__main__':
    ndata = 100
    nharmonics = 3
    sigma = 0.1
    freq = 1.0
    a = 1.0
    b = 0.5
    c = 0.0
    sgn = 1
    seed = 42


    
    rand = np.random if seed is None else np.random.RandomState()
    c_n = rand.rand(nharmonics)
    s_n = rand.rand(nharmonics)

    template = Template(c_n=c_n, s_n=s_n)
    template.precompute()

    params = ModelFitParams(a=a, b=b, c=c, sgn=sgn)
    model = TemplateModel(template, frequency=freq, parameters=params)


    t = np.sort(rand.rand(ndata))
    y = model(t) + sigma * rand.randn(ndata)
    dy = sigma * np.ones_like(y)

    get_poly_vs_pdg((t, y, dy), template, freq, nharmonics)





