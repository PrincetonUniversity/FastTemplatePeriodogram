"""
Core algorithm for the Fast Template Periodogram

(c) John Hoffman, Jake Vanderplas 2017
"""
from __future__ import print_function


import numpy as np
import numpy.polynomial as pol

from .summations import fast_summations, direct_summations

from .utils import ModelFitParams, AltModelFitParams, weights, get_diags

from time import time


def _template_fit_from_sums(cn, sn, sums, ybar, YY,
                            return_alt_params=False, **kwargs):
    r"""
    Finds optimal parameters given precomputed sums

    Parameters
    ----------
    cn : array_like
        Fourier (cosine) coefficients of the template
    sn : array_like
        Fourier (sine) coefficients of the template
    sums : Summations
        Precomputed summations (C, S, CC, CS, SS, YC, YS).
    ybar : float
        Weighted mean of data
    YY : float
        Weighted variance of data
    return_alt_params : bool, default = False
        Return the best fit model parameters as an `AltModelFitParams`
        object instead of a `ModelFitParams` object. This uses
        `theta_1` (amplitude), `theta_2` (phase), `theta_3` (offset)
        instead of the `a`, `b`, `c`, `sgn` parameters.

    Returns
    -------
    params : `ModelFitParams` (or `AltModelFitParams`)
        Best fit template parameters
    power : float
        $(\chi^2_0 - \chi^2(fit)) / \chi^2_0$, where $\chi^2_0$ is for a
        flat model with $\hat{y}_0 = \bar{y}$, the weighted mean.
    """
    H = len(cn)

    alpha = 0.5 * (np.asarray(cn) + 1j * np.asarray(sn))

    # compute YM
    aYC = alpha * (sums.YC - 1j * sums.YS)
    YM = pol.Polynomial(np.concatenate((np.conj(aYC)[::-1], [0], aYC)))

    # compute MM
    UU = sums.CC + 1j * sums.CS
    VV = sums.SS + 1j * sums.CS.T

    CC = (np.conj(UU) - VV) * np.outer(alpha, alpha)
    CS = (UU + np.conj(VV)) * np.outer(alpha, np.conj(alpha))
    SS = np.conj(CC)

    # TODO : use numpy to speed this up.
    MM = np.zeros(4 * H + 1, dtype=np.complex64)

    CC = CC[::-1, :]
    CS = CS.T
    SS = SS[:, ::-1]

    CC_diags = get_diags(CC)
    CS_diags = get_diags(CS)
    SS_diags = get_diags(SS)

    inds = np.arange(2 * H - 1)

    MM[inds] += SS_diags[inds]
    MM[inds + H + 1] += 2 * CS_diags[inds]
    MM[inds + 2 * H + 2] += CC_diags[inds]

    # Polynomial math + root finding!
    MM = pol.Polynomial(MM)

    p = 2 * MM * YM.deriv() - MM.deriv() * YM

    roots = p.roots()

    # only keep non-zero roots.
    roots = roots[np.absolute(roots) > 0]

    # ensure they are on the unit circle.
    roots /= np.absolute(roots)

    # Get periodogram values at each root
    pdg_phi = np.real(YM(roots) ** 2 / MM(roots)) / YY

    # find root that maximizes periodogram
    i = np.argmax(pdg_phi)
    best_phi = roots[i]

    # get optimal model parameters
    AC = alpha * (sums.C - 1j * sums.S)
    alpha_phi = pol.Polynomial(np.concatenate(([0], AC)))
    mbar = 2 * np.real(alpha_phi(best_phi))

    theta_1 = np.real(np.power(best_phi, H) * YM(best_phi) / MM(best_phi))
    theta_2 = np.imag(np.log(best_phi)) % (2 * np.pi)
    theta_3 = ybar - mbar * theta_1

    best_params = ModelFitParams(a=theta_1,
                                 b=np.cos(theta_2),
                                 c=theta_3,
                                 sgn=np.sign(np.sin(theta_2)))\
        if not return_alt_params else\
        AltModelFitParams(theta_1=theta_1, theta_2=theta_2,
                          theta_3=theta_3)

    return best_params, pdg_phi[i]


def fit_template(t, y, dy, cn, sn, freq, sums=None, **kwargs):
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
    freq : float
        Frequency at which to fit the template
    sums : Summations, optional
        Precomputed summations (C, S, CC, CS, SS, YC, YS). Default
        is None, which means the sums are computed directly (no NFFT)
    ** kwargs : dict, optional
        Passed to ``template_fit_from_sums`` or ``direct_summations``
        function

    Returns
    -------
    power : float
        $(\chi^2_0 - \chi^2(fit)) / \chi^2_0$, where $\chi^2_0$ is for a
        flat model with $\hat{y}_0 = \bar{y}$, the weighted mean.
    params : ``ModelFitParams`` or ``AltModelFitParams``
        Best fit template parameters

    Notes
    -----
    ``allow_negative_amplitudes`` is deprecated as of ``v1.0.1``
    to simplify things. Earlier versions were inconsistent in
    whether or not they actually checked for negative amplitudes,
    since this was an experimental feature -- don't trust that
    earlier versions do anything useful with this parameter!
    """
    nh = len(cn)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(y - ybar, 2))

    if sums is None:
        sums = direct_summations(t, y, w, freq, nh, **kwargs)

    params, power = _template_fit_from_sums(cn, sn, sums, ybar, YY, **kwargs)

    return power, params


def template_periodogram(t, y, dy, cn, sn, freqs,
                         summations=None, fast=True,
                         **kwargs):
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
    freqs : array_like
        Frequencies at which to fit the template
    summations : list of Summations, optional
        Precomputed summations (C, S, CC, CS, SS, YC, YS) at each frequency
        in freqs. Default is ``None``, which means the sums are computed via
        direct summations (if ``fast=False``) or via NFFT if ``fast=True``
    **kwargs : dict, optional
        Passed to ``template_fit_from_sums`` and ``direct_summations``
        or ``fast_summations`` functions

    Returns
    -------
    powers : array_like
        $(\chi^2_0 - \chi^2(fit)) / \chi^2_0$ at each frequency in `freqs`,
        where $\chi^2_0$ is for a flat model with $\hat{y}_0 = \bar{y}$,
        the weighted mean.
    best_fit_params : list of `ModelFitParams` or `AltModelFitParams`
        List of best-fit model parameters at each frequency in `freqs`

    Notes
    -----
    ``allow_negative_amplitudes`` is deprecated as of ``v1.0.1``
    to simplify things. Earlier versions were inconsistent in
    whether or not they actually checked for negative amplitudes,
    since this was an experimental feature -- don't trust that
    earlier versions do anything useful with this parameter!
    """
    nh = len(cn)
    w = weights(dy)

    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(y - ybar, 2))

    if summations is None:
        # compute sums using NFFT
        if fast:
            summations = fast_summations(t, y, w, freqs, nh, **kwargs)
        else:
            summations = direct_summations(t, y, w, freqs, nh, **kwargs)

    best_fit_params, powers = zip(*[_template_fit_from_sums(cn, sn, sums,
                                                            ybar, YY, **kwargs)
                                    for sums in summations])

    return np.array(powers), best_fit_params
