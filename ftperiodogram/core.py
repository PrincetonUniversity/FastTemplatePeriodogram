"""
Core algorithm for the Fast Template Periodogram

(c) John Hoffman, Jake Vanderplas 2017
"""
from __future__ import print_function


import numpy as np
import numpy.polynomial as pol

from .summations import (fast_summations, direct_summations,
                         fast_summations_batched, stack_summations,
                         _validate_chunk_size)

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


def batched_YM_MM_from_sums(cn, sn, sums):
    r"""Vectorized :func:`YM_MM_from_sums` over a stacked ``Summations``.

    Parameters
    ----------
    cn : array_like
        Fourier (cosine) coefficients of the template
    sn : array_like
        Fourier (sine) coefficients of the template
    sums : Summations
        Stacked summations with a leading frequency axis (``C``/``S``/``YC``/
        ``YS`` of shape ``(nf, H)``; ``CC``/``CS``/``SS`` of ``(nf, H, H)``),
        e.g. one chunk from
        :func:`ftperiodogram.summations.fast_summations_batched`.

    Returns
    -------
    YM_coefs : ndarray, shape (nf, 2H+1)
        Stacked coefficients of the per-frequency ``YM`` polynomials.
    MM_coefs : ndarray, shape (nf, 4H+1)
        Stacked coefficients of the per-frequency ``MM`` polynomials.
    AC : ndarray, shape (nf, H)
        Per-frequency mean-template coefficients ``alpha * (C - i S)``.
    """
    H = len(cn)

    alpha = 0.5 * (np.asarray(cn) + 1j * np.asarray(sn))

    # compute YM
    aYC = alpha * (sums.YC - 1j * sums.YS)                      # (nf, H)
    nf = aYC.shape[0]
    YM_coefs = np.concatenate((np.conj(aYC)[:, ::-1],
                               np.zeros((nf, 1), dtype=np.complex128),
                               aYC), axis=1).astype(np.complex128)

    # compute MM
    UU = sums.CC + 1j * sums.CS                                 # (nf, H, H)
    VV = sums.SS + 1j * np.swapaxes(sums.CS, 1, 2)

    CC = (np.conj(UU) - VV) * np.outer(alpha, alpha)
    CS = (UU + np.conj(VV)) * np.outer(alpha, np.conj(alpha))
    SS = np.conj(CC)

    # materialize the reversed/transposed views with a fixed (C-contiguous)
    # layout: np.trace's reduction order -- and hence the last-ulp rounding
    # of the diagonal sums -- must not depend on how large the chunk is
    CC = np.ascontiguousarray(CC[:, ::-1, :])
    CS = np.ascontiguousarray(np.swapaxes(CS, 1, 2))
    SS = np.ascontiguousarray(SS[:, :, ::-1])

    # diagonal sums (get_diags) for the whole stack at once
    offsets = np.arange(-H + 1, H)
    CC_diags = np.stack([np.trace(CC, offset=o, axis1=1, axis2=2)
                         for o in offsets], axis=1)
    CS_diags = np.stack([np.trace(CS, offset=o, axis1=1, axis2=2)
                         for o in offsets], axis=1)
    SS_diags = np.stack([np.trace(SS, offset=o, axis1=1, axis2=2)
                         for o in offsets], axis=1)

    MM_coefs = np.zeros((nf, 4 * H + 1), dtype=np.complex128)

    inds = np.arange(2 * H - 1)

    MM_coefs[:, inds] += SS_diags
    MM_coefs[:, inds + H + 1] += 2 * CS_diags
    MM_coefs[:, inds + 2 * H + 2] += CC_diags

    # mean-template coefficients (for offset reconstruction)
    AC = alpha * (sums.C - 1j * sums.S)

    return YM_coefs, MM_coefs, AC


def batched_stationarity_coefs(YM_coefs, MM_coefs):
    r"""Stacked coefficients of the stationarity polynomial
    ``p = 2 MM YM' - MM' YM`` at fixed length ``6H - 1``.

    The two convolutions are evaluated as a vectorized shift-and-add over the
    (short) coefficient axis, so the work is ``O(H)`` array operations on the
    whole frequency stack at once. The nominal degree-``(6H - 1)`` leading
    coefficient cancels analytically (see :func:`trim_zero_leading_coef`); it
    is dropped here, consistent with the conditional trim on the
    per-frequency path, so the returned stack has ``6H - 1`` columns (degree
    at most ``6H - 2``).
    """
    nf, n_ym = YM_coefs.shape
    H = (n_ym - 1) // 2
    n_mm = MM_coefs.shape[1]

    dYM = YM_coefs[:, 1:] * np.arange(1, n_ym)      # YM' coefficients (nf, 2H)
    dMM = MM_coefs[:, 1:] * np.arange(1, n_mm)      # MM' coefficients (nf, 4H)

    p = np.zeros((nf, 6 * H), dtype=np.complex128)
    for s in range(n_ym - 1):                       # 2 * MM * YM'
        p[:, s:s + n_mm] += (2 * dYM[:, s:s + 1]) * MM_coefs
    for s in range(n_ym):                           # ... - MM' * YM
        p[:, s:s + n_mm - 1] -= YM_coefs[:, s:s + 1] * dMM

    return p[:, :6 * H - 1]


def trim_zero_leading_coef(p):
    r"""Drop the analytically-zero leading coefficient of a stationarity
    polynomial of the form ``2 MM YM' - MM' YM``.

    Because ``deg MM = 2 deg YM`` (4H vs 2H), the nominal top terms cancel
    exactly (``2 * mm * 2H * ym - 4H * mm * ym = 0``): the true degree is at
    most ``6H - 2``, and the stored leading coefficient is pure floating-point
    residue (~1e-19). Left in place it injects a spurious huge-modulus
    (~1e15) companion root -- harmless to the power after unit-circle
    projection, but a numerical wart. The same per-term cancellation applies
    to the multiband shared-phase polynomial ``G`` (true degree at most
    ``8HK - 2``).

    The trim is conditional because the cancellation is often EXACT in
    floating point, in which case numpy's polynomial arithmetic has already
    trimmed the zero and the stored leading coefficient is genuine: only a
    residue-scale leading coefficient is dropped.
    """
    coef = p.coef
    if len(coef) > 1:
        scale = np.max(np.abs(coef))
        if scale > 0 and np.abs(coef[-1]) <= 1e-9 * scale:
            return pol.Polynomial(coef[:-1])
    return p


def roots_from_YM_MM(YM, MM, AC, H, ybar, YY, positive_amplitude=False,
                     stationarity=None):
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
        K.18); the multiband solver uses it. If *no* root yields a non-negative
        amplitude (the data anti-correlate with the template at every candidate
        phase), the global power-maximizing root is returned as a fallback, so a
        negative ``theta_1`` is still possible in that degenerate case. The
        default (False) leaves the single-band behavior unchanged: the global
        power-maximizing root is returned regardless of the sign of ``theta_1``.
    stationarity : numpy.polynomial.Polynomial, optional
        Precomputed stationarity polynomial ``2 MM YM' - MM' YM`` (e.g. one
        row of :func:`batched_stationarity_coefs`); when given, the
        per-frequency polynomial arithmetic is skipped. It must already have
        the analytically-zero degree-``(6H - 1)`` coefficient dropped (as
        :func:`batched_stationarity_coefs` does) -- no conditional trim is
        applied to it here.

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
    if stationarity is None:
        # true degree <= 6H-2; the nominal leading coefficient is FP residue
        p = trim_zero_leading_coef(2 * MM * YM.deriv() - MM.deriv() * YM)
    else:
        # precomputed polynomials arrive with the analytically-zero top
        # coefficient already dropped (batched_stationarity_coefs); a second
        # conditional trim could eat the genuine degree-(6H-2) coefficient
        p = stationarity

    roots = p.roots()

    # only keep non-zero roots.
    roots = roots[np.absolute(roots) > 0]

    # Degenerate inputs (constant y, single observation) give YM ~ 0 and an
    # empty candidate set; report a flat zero-power fit instead of crashing
    # (mirrors the multiband flat-band guard).
    flat_params = ModelFitParams(a=0.0, b=1.0, c=ybar, sgn=1.0)
    if len(roots) == 0:
        return flat_params, 0.0, 1.0 + 0.0j

    # ensure they are on the unit circle.
    roots /= np.absolute(roots)

    # Get periodogram values at each root; map non-finite values (YY = 0, or
    # a candidate with MM ~ 0) to -inf so they cannot hijack the argmax.
    with np.errstate(divide='ignore', invalid='ignore'):
        pdg_phi = np.real(YM(roots) ** 2 /  MM(roots)) / YY
    pdg_phi[~np.isfinite(pdg_phi)] = -np.inf

    if positive_amplitude:
        # K.18: among the roots, keep only those whose amplitude theta_1 >= 0,
        # then pick the best-fitting one. A half-phase flip would NOT preserve
        # the fitted curve for H > 1, so we filter the root set instead.
        with np.errstate(divide='ignore', invalid='ignore'):
            theta_1_all = np.real(np.power(roots, H) * YM(roots) / MM(roots))
        positive = theta_1_all >= 0
        if np.any(positive):
            pdg_phi = np.where(positive, pdg_phi, -np.inf)

    if not np.any(pdg_phi > -np.inf):
        return flat_params, 0.0, 1.0 + 0.0j

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



def _iter_stacked_summations(t, y, w, freqs, nh, summations, fast, sigma, tol,
                             chunk_size):
    """Yield stacked ``Summations`` chunks for the batched assembly path,
    from precomputed per-frequency summations if given, else via the batched
    NFFT (`fast=True`) or stacked direct summations (`fast=False`)."""
    nf = len(freqs)
    if summations is not None:
        for i0 in range(0, nf, chunk_size):
            yield stack_summations(summations[i0:i0 + chunk_size])
    elif fast:
        for stacked in fast_summations_batched(t, y, w, freqs, nh,
                                               chunk_size=chunk_size,
                                               sigma=sigma, tol=tol):
            yield stacked
    else:
        for i0 in range(0, nf, chunk_size):
            yield stack_summations(
                direct_summations(t, y, w, freqs[i0:i0 + chunk_size], nh))


def _template_periodogram_batched(t, y, w, cn, sn, freqs, ybar, YY,
                                  summations=None, fast=True, sigma=2,
                                  tol=1E-7, chunk_size=4096):
    """Batched-assembly template periodogram (``method='batched'``).

    Coefficient assembly (sums -> YM/MM -> stationarity polynomial) is
    vectorized over frequency chunks; root-finding and root selection then
    reuse :func:`roots_from_YM_MM` per frequency on the precomputed
    coefficients, so the behavior matches the per-frequency path.
    """
    _validate_chunk_size(chunk_size)
    if summations is not None and len(summations) != len(freqs):
        # the chunk loop slices `summations` by frequency index, so a length
        # mismatch would silently truncate at a chunk boundary; fail loudly
        # (the per-frequency path silently iterates whatever it is given)
        raise ValueError("summations must have one entry per frequency; "
                         "got {0} for {1} frequencies".format(
                             len(summations), len(freqs)))
    H = len(cn)
    powers = []
    best_fit_params = []
    for sums in _iter_stacked_summations(t, y, w, freqs, H, summations, fast,
                                         sigma, tol, chunk_size):
        YM_coefs, MM_coefs, AC = batched_YM_MM_from_sums(cn, sn, sums)
        p_coefs = batched_stationarity_coefs(YM_coefs, MM_coefs)
        for i in range(YM_coefs.shape[0]):
            params, power, _ = roots_from_YM_MM(
                pol.Polynomial(YM_coefs[i]), pol.Polynomial(MM_coefs[i]),
                AC[i], H, ybar, YY,
                stationarity=pol.Polynomial(p_coefs[i]))
            powers.append(power)
            best_fit_params.append(params)

    return np.array(powers), best_fit_params


def template_periodogram(t, y, dy, cn, sn, freqs,
                        summations=None, fast=True, sigma=2, tol=1E-7,
                        method='eigvals', chunk_size=4096):
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
        `fast=True`). The batched path requires exactly one entry per
        frequency and raises ``ValueError`` otherwise.
    sigma : float, optional (default 2)
        NFFT oversampling factor; forwarded to `fast_summations` (only used
        when `fast=True` and `summations` is None).
    tol : float, optional (default 1e-7)
        NFFT kernel truncation tolerance; forwarded to `fast_summations`
        (only used when `fast=True` and `summations` is None).
    method : str, optional (default 'eigvals')
        'eigvals' is the reference per-frequency path (polynomial objects
        assembled and rooted one frequency at a time). 'batched' assembles
        the YM/MM/stationarity coefficients for whole chunks of frequencies
        as stacked arrays (vectorized over frequency) and then runs the same
        per-frequency root selection on the precomputed coefficients; it is
        numerically equivalent (powers agree to ~1e-15).
    chunk_size : int, optional (default 4096)
        Number of frequencies per assembly chunk on the batched path (bounds
        the memory of the ``(nf, H, H)`` covariance stacks); only used when
        `method='batched'`. Must be a positive integer; the powers are
        bitwise independent of its value.

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

    if method == 'batched':
        return _template_periodogram_batched(
            t, y, w, cn, sn, freqs, ybar, YY, summations=summations,
            fast=fast, sigma=sigma, tol=tol, chunk_size=chunk_size)
    if method != 'eigvals':
        raise ValueError("Unknown method {0!r}; must be 'eigvals' or "
                         "'batched'".format(method))

    if summations is None:
        # compute sums using NFFT
        if fast:
            summations = fast_summations(t, y, w, freqs, nh, sigma=sigma,
                                          tol=tol)
        else:
            summations = direct_summations(t, y, w, freqs, nh)

    best_fit_params, powers = zip(*[ template_fit_from_sums(cn, sn, sums, ybar, YY) \
                                             for sums in summations ])

    return np.array(powers), best_fit_params