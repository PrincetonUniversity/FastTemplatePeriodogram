import pyftp.pseudo_poly as ppol
from pyftp.modeler import FastTemplateModeler, TemplateModel
from pyftp.utils import weights, ModelFitParams, Summations, Avec, Bvec
from pyftp.summations import direct_summations
from pyftp.template import Template
from pyftp.periodogram import get_a_from_b
import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

import imageio


import numpy.polynomial.polynomial as pol

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

def convert_b_to_phi(b, sgn):
    wtau = np.arccos(b)
    if sgn < 0:
        wtau = 2 * np.pi - wtau

    return wtau / (2 * np.pi)

def convert_phi_to_b(phi):
    b = np.cos(2 * np.pi * phi)
    sgn = np.sign(np.sin(2 * np.pi * phi))
    return b, sgn

def nlmodel(data, template, freq, b=None, sgn=None):
    
    phi0 = None if b is None else convert_b_to_phi(b, sgn)

    if phi0 is None:

        func = lambda t, a, phi, c : a * template((t * freq - phi)%1.0) + c
        bounds = ([ -np.inf, 0, -np.inf ], [ np.inf, 1, np.inf])
        p0 = [ 1.0, 0.5, 0.0 ]
    
    else:
        func = lambda t, a, c : a * template((t * freq - phi0)%1.0) + c
        bounds = ([ -np.inf, -np.inf ], [ np.inf, np.inf])
        p0 = [ 1.0, 0.0 ]

    tvals, yvals, yerr = data 
    popt, pcov = curve_fit(func, tvals, yvals, sigma=yerr, absolute_sigma=True, bounds=bounds, p0=p0)

    if phi0 is None:
        b, sgn = convert_phi_to_b(popt[1])
        parameters = ModelFitParams(a=popt[0], b=b, c=popt[2], sgn=sgn)

    else:
        parameters = ModelFitParams(a=popt[0], b=b, c=popt[1], sgn=sgn)


    return TemplateModel(template, frequency=freq, parameters=parameters)


def animate3d(f, ax, filename='animation', steps=150, min_dist=5., max_dist=7., min_elev=10., max_elev=80.):
    

    # a viewing perspective is composed of an elevation, distance, and azimuth
    # define the range of values we'll cycle through for the distance of the viewing perspective
    #dist_range = np.arange(min_dist, max_dist, (max_dist-min_dist)/steps)
    dist_range = np.zeros(steps)
    dist_range[:steps/2] = np.linspace(min_dist, max_dist, steps/2)
    dist_range[steps/2:] = np.linspace(max_dist, min_dist, steps/2)

    # the range of values we'll cycle through for the elevation of the viewing perspective
    elev_range = np.zeros(steps)
    elev_range[:steps/2] = np.linspace(min_elev, max_elev, steps/2)
    elev_range[steps/2:] = np.linspace(max_elev, min_elev, steps/2)

    # now create the individual frames that will be combined later into the animation
    for azimuth in range(0, 360, int(360/steps)):
        
        # pan down, rotate around, and zoom out
        ax.azim = azimuth#float(azimuth/3.)
        ax.elev = elev_range[int(azimuth/(360./steps))]
        ax.dist = dist_range[int(azimuth/(360./steps))]
        
        # save each figure as a .png
        plt.savefig('{}_frame{:03d}.png'.format(filename, azimuth))
    

    gif_filepath = '{}.gif'.format(filename)
    images = []
    for frame in glob.glob("{}_frame*.png".format(filename)):
        images.append(imageio.imread(frame))
    #kargs = dict(duration=75)
    imageio.mimsave(gif_filepath, images, 'GIF')#, **kargs)
    
    # dont display static plot
    plt.close(f)



def get_poly_vs_pdg(data, template, freq, nharmonics):
    t, y, dy = data
    w = weights(dy)
    ybar = np.dot(w, y)
    poly = get_poly(t, y, dy, template, freq)
    H = nharmonics
    YY = np.dot(w, (y - ybar)**2)

    bvals = np.linspace(-1, 1, 500)

    poly_vals = np.polyval(poly, bvals)
    zeros = pol.polyroots(poly_vals)

    sums = direct_summations(t, y, w, freq, nharmonics)

    pdg_vals_pos = []
    pdg_vals_neg = []

    pdg_vals_nl_neg  = []
    pdg_vals_nl_pos  = []

    fit_params_pos = []
    fit_params_neg = []

    fit_params_nl_neg = []
    fit_params_nl_pos = []

    pdg_vals_frompars_pos = []
    pdg_vals_frompars_neg = []

    for b in bvals:
        for sgn, pdg_vals, pdg_vals_nl, fit_params, fit_params_nl in zip([-1, 1], 
                                            [pdg_vals_neg, pdg_vals_pos], 
                                            [pdg_vals_nl_neg, pdg_vals_nl_pos],
                                            [fit_params_nl_neg, fit_params_nl_pos],
                                            [fit_params_neg, fit_params_pos]):
            A = Avec(b, template.c_n, template.s_n, sgn=sgn)
            B = Bvec(b, template.c_n, template.s_n, sgn=sgn)

            AYCBYS = np.dot(A, sums.YC[:H]) + np.dot(B, sums.YS[:H])
            ACBS   = np.dot(A, sums.C[:H])  + np.dot(B, sums.S[:H])

            # Obtain amplitude for a given b=cos(wtau) and sign(sin(wtau))
            a = get_a_from_b(b, template.c_n, template.s_n, sums, A=A, B=B, AYCBYS=AYCBYS)

            c = ybar - a * ACBS

            pz = a * AYCBYS / YY

            params = ModelFitParams(a=a, b=-b, c=c, sgn=-sgn)

            model = TemplateModel(template, frequency=freq, 
                               parameters=params)

            fit_params.append(params)
            y_fit = model(t)

            model_nl = nlmodel(data, template, freq, b=b, sgn=sgn)
            fit_params_nl.append(model_nl.parameters)
            
            y_nl  = model_nl(t)
            pdg_vals_nl.append(pdg(data, y_nl))
            pdgval = pdg(data, y_fit)

            if sgn > 0:
                pdg_vals_frompars_pos.append(pz)
            else:
                pdg_vals_frompars_neg.append(pz)

            pdg_vals.append(pdg(data, y_fit))

    pdg_zeros_max  = None
    best_b_zeros   = None
    best_sgn_zeros = None

    bad_zero = lambda bz : bz.real < -1 or bz.real > 1 #or abs(bz.imag) / (b.imag ** 2 + b.real ** 2) > 0.01

    zeros = [ b.real for b in zeros if not bad_zero(b) ]
    print zeros

    for b in zeros:
        for sgn in [ -1, 1 ]:
            A = Avec(b, template.c_n, template.s_n, sgn=sgn)
            B = Bvec(b, template.c_n, template.s_n, sgn=sgn)

            AYCBYS = np.dot(A, sums.YC[:H]) + np.dot(B, sums.YS[:H])
            ACBS   = np.dot(A, sums.C[:H])  + np.dot(B, sums.S[:H])

            # Obtain amplitude for a given b=cos(wtau) and sign(sin(wtau))
            a = get_a_from_b(b, template.c_n, template.s_n, sums, A=A, B=B, AYCBYS=AYCBYS)

            c = np.dot(w, y) - a * ACBS

            params = ModelFitParams(a=a, b=-b, c=c, sgn=-sgn)

            model  = TemplateModel(template, frequency=freq, 
                               parameters=params)

            y_fit  = model(t)
            pdgval = pdg(data, y_fit)

            if pdg_zeros_max is None or pdgval > pdg_zeros_max:
                best_b_zeros = params.b
                best_sgn_zeros = params.sgn
                pdg_zeros_max = pdgval


    f, ax = plt.subplots()


    ax.plot(-bvals, pdg_vals_pos, color='b', label='P(w) given b (sinwtau > 0)', alpha=0.8)
    ax.plot(-bvals, pdg_vals_neg, color='b', ls='--', label='P(w) given b (sinwtau < 0)', alpha=0.8)
    ax.plot(-bvals, pdg_vals_frompars_neg, color='g', lw=2, label='P(w) given b (sinwtau > 0) (analytic)', alpha=0.8)
    ax.plot(-bvals, pdg_vals_frompars_pos, color='g', ls='--', lw=2, label='P(w) given b (sinwtau < 0) (analytic)', alpha=0.8)
    ax.plot(bvals, pdg_vals_nl_pos, color='m', label='P(w) (nonlin) sinwtau > 0', alpha=0.8)
    ax.plot(bvals, pdg_vals_nl_neg, color='m', ls='--', label='P(w) (nonlin) sinwtau < 0', alpha=0.8)




    best_b_nl_pos = bvals[np.argmax(pdg_vals_nl_pos)]
    best_b_nl_neg = bvals[np.argmax(pdg_vals_nl_neg)]

    best_b_pos = bvals[np.argmax(pdg_vals_pos)]
    best_b_neg = bvals[np.argmax(pdg_vals_neg)]


    print("best b vals:")
    print("   NL fit (sin > 0): %e"%(best_b_nl_pos))
    print("   pdg    (sin > 0): %e"%(best_b_pos))
    print("   NL fit (sin < 0): %e"%(best_b_nl_neg))
    print("   pdg    (sin < 0): %e"%(best_b_neg))


    print("   zeros           : %e"%(best_b_zeros))
    print("compare periodograms")
    print("   pdg (zeros)     : %e"%(pdg_zeros_max))
    print("   pdg (nonlin)    : %e"%(max([ max(pdg_vals_nl_pos), max(pdg_vals_nl_neg)])))
    print("   pdg (all)       : %e"%(max([ max(pdg_vals_pos), max(pdg_vals_neg)])))

    #poly_mad = np.median(np.absolute(poly_vals - np.median(poly_vals)))
    poly_mad = 1.
    ax.plot(bvals, poly_vals/poly_mad, color='k', label='polynomial')
    ax.axhline(0, ls=':', color='0.5')
    ax.legend(loc='best')

    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('$\\cos\\omega\\tau$')

    f2 = plt.figure()

    ax2 = f2.add_subplot(111, projection='3d')

    a_neg    = [ par.a for par in fit_params_pos ]
    a_pos    = [ par.a for par in fit_params_neg ]
    a_pos_nl = [ par.a for par in fit_params_nl_pos ]
    a_neg_nl = [ par.a for par in fit_params_nl_neg ]

    c_neg    = [ par.c for par in fit_params_pos ]
    c_pos    = [ par.c for par in fit_params_neg ]
    c_pos_nl = [ par.c for par in fit_params_nl_pos ]
    c_neg_nl = [ par.c for par in fit_params_nl_neg ]

    ax2.plot(-bvals, a_pos, c_pos, color='b')
    ax2.plot(-bvals, a_neg, c_neg, color='b', ls='--')

    ax2.plot(bvals, a_pos_nl, c_pos_nl, color='m')
    ax2.plot(bvals, a_neg_nl, c_neg_nl, color='m', ls='--')

    ax2.set_xlabel('$\\cos\\omega\\tau$')
    ax2.set_ylabel('Amplitude')
    ax2.set_zlabel('Offset')


    #animate3d(f2, ax2)

    plt.show()

if __name__ == '__main__':
    ndata = 100
    nharmonics = 1
    sigma = 0.0001
    freq = 1.0
    a = 0.1
    b = 0.2
    c = 0.0
    sgn = 1
    seed = 42


    
    rand = np.random if seed is None else np.random.RandomState()
    c_n = rand.rand(nharmonics)
    s_n = rand.rand(nharmonics)

    template = Template(c_n=c_n, s_n=s_n)
    template.precompute()

    params = ModelFitParams(a=a, b=b, c=c, sgn=sgn)
    model = TemplateModel(template, frequency=0.8 * freq, parameters=params)


    t = np.sort(rand.rand(ndata))
    t[0] = 0
    y = model(t) + sigma * rand.randn(ndata)
    dy = sigma * np.ones_like(y)

    get_poly_vs_pdg((t, y, dy), template, freq, nharmonics)





