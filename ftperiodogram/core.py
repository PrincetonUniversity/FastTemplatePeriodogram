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

get_diags = lambda mat, shape : np.array([ sum(mat.diagonal(i)) for i in range(*shape) ])

def template_fit_from_sums(alpha, AA, Aa, sums, ybar, YY):
    r"""
    Finds optimal parameters given precomputed sums

    Parameters
    ----------
    alpha : array_like
        `0.5 * (cn + 1j * sn)`, where `cn` and `sn` are the cosine and
        sine Fourier coefficients of the template
    AA : shape = (H, H), ndarray
        outer product of `alpha` and `alpha`
    Aa : shape = (H, H), ndarray
        outer product of `alpha` and `conj(alpha)`
    sums : Summations
        Precomputed summations (C, S, CC, CS, SS, YC, YS).
    ybar : float
        Weighted mean of data
    YY : float
        Weighted variance of data

    Returns
    -------
    params : ModelFitParams
        Best fit template parameters
    power : float
        $(\chi^2_0 - \chi^2(fit)) / \chi^2_0$, where $\chi^2_0$ is for a
        flat model with $\hat{y}_0 = \bar{y}$, the weighted mean.
    """
    H = len(alpha)

    AC = alpha * (sums.C - 1j * sums.S)
    alpha_phi = pol.Polynomial(np.concatenate(([0], AC)))

    # compute YM
    aYC = alpha * (sums.YC - 1j * sums.YS)
    YM = pol.Polynomial(np.concatenate((np.conj(aYC)[::-1], [0], aYC)).astype(np.complex64))

    # compute MM
    CC = sums.CC - sums.SS - 1j * (sums.CS + sums.CS.T)
    CS = sums.CC + sums.SS + 1j * (sums.CS - sums.CS.T)

    CC_diags = get_diags(CC[::-1, :], (-H + 1, H))
    CS_diags = get_diags(CS.T, (0, 2*H))

    inds = np.arange(2 * H - 1)
    mm = np.zeros(2 * H + 1, dtype=np.complex64)

    mm[inds[:H]] += 2 * CS_diags[inds[:H]]
    mm[inds + 2] +=     CC_diags[inds]

    MM = pol.Polynomial(np.concatenate((np.conj(mm)[::-1], mm[1:])))

    # Polynomial math + root finding!
    p = 2 * MM * YM.deriv() - MM.deriv() * YM

    roots = p.roots()

    # only keep non-zero roots.
    roots = roots[np.absolute(roots) > 0]

    # ensure they are on the unit circle.
    roots /= np.absolute(roots)
    
    # Get periodogram values at each root
    pdg_phi = np.real(YM(roots) ** 2 /  MM(roots)) / YY

    # find root that maximizes periodogram
    i = np.argmax(pdg_phi)
    best_phi = roots[i]

    # get optimal model parameters
    mbar = 2 * np.real(alpha_phi(best_phi))

    theta_1 = np.real(np.power(best_phi, H) * YM(best_phi) /  MM(best_phi))
    theta_2 = np.imag(np.log(best_phi)) % (2 * np.pi)
    theta_3 = ybar - mbar * theta_1

    best_params = ModelFitParams(a=theta_1, 
                                 b=np.cos(theta_2), 
                                 c=theta_3, 
                                 sgn=np.sign(np.sin(theta_2)))

    return best_params, pdg_phi[i]

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

    alpha = 0.5 * (np.asarray(cn) + 1j * np.asarray(sn))

    AA = np.outer(alpha, alpha)
    Aa = np.outer(alpha, np.conj(alpha))

    if sums is None:
        sums   = direct_summations(t, y, w, freq, nh)

    params, power = template_fit_from_sums(alpha, AA, Aa, sums, ybar, YY) 

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

    alpha = 0.5 * (np.asarray(cn) + 1j * np.asarray(sn))

    AA = np.outer(alpha, alpha)
    Aa = np.outer(alpha, np.conj(alpha))
    
    if summations is None:
        # compute sums using NFFT
        if fast:
            summations = fast_summations(t, y, w, freqs, nh)
        else:
            summations = direct_summations(t, y, w, freqs, nh)

    best_fit_params, powers = zip(*[ template_fit_from_sums(alpha, AA, Aa, sums, ybar, YY) \
                                             for sums in summations ])

    return np.array(powers), best_fit_params