"""
Core algorithm for the Fast Template Periodogram

(c) John Hoffman, Jake Vanderplas 2017
"""
from __future__ import print_function


import numpy as np
import numpy.polynomial as pol

from .summations import fast_summations, direct_summations

from .utils import ModelFitParams, weights

from time import time


def new_solution(cn, sn, sums, ybar, YY):
    H = len(cn)

    alpha = 0.5 * (np.asarray(cn) + 1j * np.asarray(sn))
    alpha_conj = np.conj(alpha)

    CC =  (sums.CC - sums.SS - 1j * (sums.CS + sums.CS.T)) * np.outer(alpha, alpha)
    CS =  2 * ((sums.CC + sums.SS + 1j * (sums.CS - sums.CS.T)) * np.outer(alpha, alpha_conj))#[:,::-1]
    SS = np.conj(CC)#[::-1, ::-1]
    
    YC = np.array(sums.YC - 1j * sums.YS, dtype=np.complex64)

    aYC = alpha * YC
    YM = pol.Polynomial(np.concatenate((np.conj(aYC)[::-1], [0], aYC)).astype(np.complex64))


    AC = alpha * (sums.C - 1j * sums.S)
    MM = np.zeros(4 * H + 1, dtype=np.complex64)

    #for k in range(0, 2 * H - 1):
    #    n0 = max([ 0, k - (H-1) ])
    #    m0 = min([ H-1, k ])

    #    inds = np.arange(k + 1)
    #    if k + 1 > H:
    #        inds = np.arange(2 * H - k - 1)


    #    MM[k + 2 * H + 2] = np.sum(CC[n0 + inds, m0 - inds])
    #    MM[k + H + 1]     = np.sum(CS[n0 + inds, m0 - inds])
    #    MM[k]             = np.sum(SS[n0 + inds, m0 - inds])
    for n in range(H):
        for m in range(H):
            MM[2*H + (n + 1) + (m + 1)] += CC[n][m]
            MM[2*H + (n + 1) - (m + 1)] += CS[n][m]
            MM[2*H - (n + 1) - (m + 1)] += SS[n][m]


    MM = pol.Polynomial(MM)

    
    alpha_phi = pol.Polynomial(np.concatenate(([0], AC)))

    #phiH = np.zeros(H+1)
    #phiH[-1] = 1.

    #YMphiH = pol.Polynomial(phiH) * YM

    p = 2 * MM * YM.deriv() - MM.deriv() * YM

    #dt = time() - t0
    #print("%.3e s to come up with arrays"%dt)

    #t0 = time()
    roots = np.exp(1j * np.imag(np.log(p.roots())))
    #dt = time() - t0
    #print("%.3e s to find roots"%dt)
    #roots = np.array([ np.imag(np.log(r))%(2 * np.pi) for r in roots ])

    #t0 = time()
    pdg_phi = np.real(YM(roots) ** 2 / MM(roots))

    i = np.argmax(pdg_phi)
    best_phi = roots[i]

    
    theta_1 = np.real(np.power(best_phi, H) * YM(best_phi) /  MM(best_phi))
    theta_2 = np.imag(np.log(best_phi)) % (2 * np.pi)

    mbar = 2 * np.real(alpha_phi(best_phi))
    theta_3 = ybar - mbar * theta_1

    best_params = ModelFitParams(a=theta_1, 
                                 b=np.cos(theta_2), 
                                 c=theta_3, 
                                 sgn=np.sign(np.sin(theta_2)))

    #dt = time() - t0
    #print("%.3e s to find best root and derive other parameters"%dt)
    return best_params, pdg_phi[i] / YY

def fit_template(t, y, dy, cn, sn, freq, sums=None,
                       allow_negative_amplitudes=True, zeros=None,
                       small=1E-7):
    r"""
    Fits periodic template to data at a single frequency

    Parameters
    ----------
    t : array_like
        Measurement times (must be monotonically increasing)
    y : array_like
        Measurement values at corresponding measurement times
    dy : array_like
        Measurement uncertainties
    cn : array_like
        Fourier (cosine) coefficients of the template
    sn : array_like
        Fourier (sine) coefficients of the template
    ptensors : np.ndarray, shape = (H, H, H, L)
        Polynomial coefficients from template; H is the number of
        harmonics, L is the (maximum) length of the polynomial
    freq : float
        Frequency at which to fit the template
    sums : Summations, optional
        Precomputed summations (C, S, CC, CS, SS, YC, YS). Default
        is None, which means the sums are computed directly (no NFFT)
    allow_negative_amplitudes : bool, optional, (default = True)
        Specifies whether or not negative amplitude solutions are allowed.
        They are automatically forbidden for H=1 (since this is equivalent
        to a phase shift). If no positive amplitude solutions are found and
        allow_negative_amplitudes = False, the periodogram is set to 0

    Returns
    -------
    power : float
        $(\chi^2_0 - \chi^2(fit)) / \chi^2_0$, where $\chi^2_0$ is for a
        flat model with $\hat{y}_0 = \bar{y}$, the weighted mean.
    params : ModelFitParams
        Best fit template parameters
    """
    nh   = len(cn)
    w    = weights(dy)
    ybar = np.dot(w, y)
    YY   = np.dot(w, np.power(y - ybar, 2))

    if sums is None:
        sums   = direct_summations(t, y, w, freq, nh)

    params, power = new_solution(cn, sn, sums, ybar, YY) 

    return power, params



def template_periodogram(t, y, dy, cn, sn, freqs,
                        summations=None, allow_negative_amplitudes=True,
                        fast=True):
    r"""
    Produces a template periodogram using a single template

    Parameters
    ----------
    t : array_like
        Measurement times (must be monotonically increasing)
    y : array_like
        Measurement values at corresponding measurement times
    dy : array_like
        Measurement uncertainties
    cn : array_like
        Fourier (cosine) coefficients of the template
    sn : array_like
        Fourier (sine) coefficients of the template
    ptensors : np.ndarray, shape = (H, H, H, L), optional
        Polynomial coefficients from template; H is the number of
        harmonics, L is the (maximum) length of the polynomial
    freqs : array_like
        Frequencies at which to fit the template
    summations : list of Summations, optional
        Precomputed summations (C, S, CC, CS, SS, YC, YS) at each frequency
        in freqs. Default is None, which means the sums are computed via
        direct summations (if `fast=False`) or via fast summations (NFFT, if
        `fast=True`)
    allow_negative_amplitudes : bool, optional, (default = True)
        Specifies whether or not negative amplitude solutions are allowed.
        They are automatically forbidden for H=1 (since this is equivalent
        to a phase shift). If no positive amplitude solutions are found and
        allow_negative_amplitudes = False, the periodogram is set to 0

    Returns
    -------
    powers : array_like
        $(\chi^2_0 - \chi^2(fit)) / \chi^2_0$ at each frequency in `freqs`,
        where $\chi^2_0$ is for a flat model with $\hat{y}_0 = \bar{y}$,
        the weighted mean.
    best_fit_params : list of `ModelFitParams`
        List of best-fit model parameters at each frequency in `freqs`
    """
    nh = len(cn)
    w = weights(dy)

    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(y - ybar, 2))
    
    if summations is None:
        # compute sums using NFFT
        if fast:
            summations = fast_summations(t, y, w, freqs, nh)
        else:
            summations = direct_summations(t, y, w, freqs, nh)

    best_fit_params, powers = zip(*[ new_solution(cn, sn, sums, ybar, YY) \
                                             for sums in summations ])

    return np.array(powers), best_fit_params