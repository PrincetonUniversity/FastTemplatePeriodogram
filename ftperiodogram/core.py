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

get_diags = lambda mat : np.array([ sum(mat.diagonal(i)) for i in range(-mat.shape[0]+1,mat.shape[1]) ])


def YM_MM_from_sums(cn, sn, sums):
    r"""
    Assemble the ``YM`` and ``MM`` polynomials (in :math:`\phi = e^{i\theta_2}`)
    from precomputed summations for a single band/template.

    This is the polynomial-assembly half of :func:`template_fit_from_sums`,
    factored out so the multiband solver can build per-band ``YM``/``MM``
    polynomials and combine their coefficients before the (shared) root-finding
    step (see :mod:`ftperiodogram.multiband`).

    Parameters
    ----------
    cn : array_like
        Fourier (cosine) coefficients of the template
    sn : array_like
        Fourier (sine) coefficients of the template
    sums : Summations
        Precomputed summations (C, S, CC, CS, SS, YC, YS).

    Returns
    -------
    YM : numpy.polynomial.Polynomial
        The data-template correlation polynomial.
    MM : numpy.polynomial.Polynomial
        The template-variance polynomial.
    AC : ndarray
        ``alpha * (C - i S)``, the mean-template coefficients used to
        reconstruct the offset (``mbar``) at a root.
    """
    H = len(cn)

    alpha = 0.5 * (np.asarray(cn) + 1j * np.asarray(sn))

    # compute YM
    aYC = alpha * (sums.YC - 1j * sums.YS)
    YM = pol.Polynomial(np.concatenate((np.conj(aYC)[::-1], [0], aYC)).astype(np.complex128))

    # compute MM
    UU = sums.CC + 1j * sums.CS
    VV = sums.SS + 1j * sums.CS.T

    CC = (np.conj(UU) - VV) * np.outer(alpha, alpha)
    CS = (UU + np.conj(VV)) * np.outer(alpha, np.conj(alpha))
    SS = np.conj(CC)

    # TODO : use numpy to speed this up.
    MM = np.zeros(4 * H + 1, dtype=np.complex128)

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

    MM = pol.Polynomial(MM)

    # mean-template coefficients (for offset reconstruction)
    AC = alpha * (sums.C - 1j * sums.S)

    return YM, MM, AC


def roots_from_YM_MM(YM, MM, AC, H, ybar, YY, positive_amplitude=False):
    r"""
    Find the optimal phase root of the ``YM``/``MM`` polynomials and reconstruct
    the best-fit template parameters and periodogram power.

    This is the root-finding / parameter-reconstruction half of
    :func:`template_fit_from_sums`, factored out so the single-band and
    multiband solvers share one source of truth for root selection.

    Parameters
    ----------
    YM, MM : numpy.polynomial.Polynomial
        The data-template correlation and template-variance polynomials. For
        multiband these are the band-combined ``YM'``/``MM'``.
    AC : ndarray
        Mean-template coefficients (``alpha * (C - i S)``) used to reconstruct
        the offset.
    H : int
        Number of harmonics (the centering power for ``phi**H``).
    ybar : float
        Weighted mean used to reconstruct the offset ``theta_3``.
    YY : float
        Weighted variance used to normalize the power.
    positive_amplitude : bool, optional (default False)
        If True, restrict the root selection to roots that yield a non-negative
        amplitude ``theta_1`` and return the best-fitting such root. This is the
        constant-time positivity filter over the same root set (paper item
        K.18); the multiband solver uses it. The default (False) leaves the
        single-band behavior unchanged: the global power-maximizing root is
        returned regardless of the sign of ``theta_1``.

    Returns
    -------
    params : ModelFitParams
        Best fit template parameters
    power : float
        $(\chi^2_0 - \chi^2(fit)) / \chi^2_0$, where $\chi^2_0$ is for a
        flat model with $\hat{y}_0 = \bar{y}$, the weighted mean.
    best_phi : complex
        The optimal root :math:`\phi = e^{i\theta_2}` (exposed so the multiband
        solver can reconstruct per-band offsets).
    """
    # Polynomial math + root finding!
    p = 2 * MM * YM.deriv() - MM.deriv() * YM

    roots = p.roots()

    # only keep non-zero roots.
    roots = roots[np.absolute(roots) > 0]

    # ensure they are on the unit circle.
    roots /= np.absolute(roots)

    # Get periodogram values at each root
    pdg_phi = np.real(YM(roots) ** 2 /  MM(roots)) / YY

    if positive_amplitude:
        # K.18: among the roots, keep only those whose amplitude theta_1 >= 0,
        # then pick the best-fitting one. A half-phase flip would NOT preserve
        # the fitted curve for H > 1, so we filter the root set instead.
        theta_1_all = np.real(np.power(roots, H) * YM(roots) / MM(roots))
        positive = theta_1_all >= 0
        if np.any(positive):
            pdg_phi = np.where(positive, pdg_phi, -np.inf)

    # find root that maximizes periodogram
    i = np.argmax(pdg_phi)
    best_phi = roots[i]

    # get optimal model parameters
    alpha_phi = pol.Polynomial(np.concatenate(([0], AC)))
    mbar = 2 * np.real(alpha_phi(best_phi))

    theta_1 = np.real(np.power(best_phi, H) * YM(best_phi) /  MM(best_phi))
    theta_2 = np.imag(np.log(best_phi)) % (2 * np.pi)
    theta_3 = ybar - mbar * theta_1

    best_params = ModelFitParams(a=theta_1,
                                 b=np.cos(theta_2),
                                 c=theta_3,
                                 sgn=np.sign(np.sin(theta_2)))

    return best_params, pdg_phi[i], best_phi


def template_fit_from_sums(cn, sn, sums, ybar, YY):
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

    Returns
    -------
    params : ModelFitParams
        Best fit template parameters
    power : float
        $(\chi^2_0 - \chi^2(fit)) / \chi^2_0$, where $\chi^2_0$ is for a
        flat model with $\hat{y}_0 = \bar{y}$, the weighted mean.
    """
    H = len(cn)

    YM, MM, AC = YM_MM_from_sums(cn, sn, sums)
    best_params, power, _ = roots_from_YM_MM(YM, MM, AC, H, ybar, YY)

    return best_params, power

def fit_template(t, y, dy, cn, sn, freq, sums=None,
                       zeros=None, small=1E-7):
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

    params, power = template_fit_from_sums(cn, sn, sums, ybar, YY) 

    return power, params



def template_periodogram(t, y, dy, cn, sn, freqs,
                        summations=None, fast=True):
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
        in freqs. Default is None, which means the sums are computed via
        direct summations (if `fast=False`) or via fast summations (NFFT, if
        `fast=True`)

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

    best_fit_params, powers = zip(*[ template_fit_from_sums(cn, sn, sums, ybar, YY) \
                                             for sums in summations ])

    return np.array(powers), best_fit_params