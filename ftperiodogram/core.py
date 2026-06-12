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


# ----------------------------------------------------------------------
# Scan+polish maximizer (WP C2). Instead of finding all roots of the
# stationarity polynomial (a dense companion-matrix eigenproblem per
# frequency), evaluate the periodogram on a uniform circle grid for ALL
# frequencies at once (zero-padded inverse FFT on the stacked coefficient
# arrays), bracket every circular local maximum, and polish each with
# Newton steps on dP/dtheta using analytic first and second derivatives.
# ----------------------------------------------------------------------

# M = max(128, 32 H) uniform angles: the periodogram is a ratio of
# trigonometric polynomials of degree <= 2H, with at most 3H - 1 local
# maxima on the circle, so ~10 grid points per stationary-point spacing.
_SCAN_MIN_ANGLES = 128
_SCAN_ANGLES_PER_H = 32
# >= 6 Newton steps per the WP C2 spec; 8 gives margin at no real cost.
_SCAN_NEWTON_STEPS = 8
# keep at most 4H bracketed maxima per frequency (> 3H - 1, the analytic
# bound, so genuine maxima are never dropped; only degenerate plateaus --
# e.g. an exactly constant periodogram -- are trimmed)
_SCAN_MAX_CANDIDATES_PER_H = 4


def _eval_polys_on_circle(coefs, n_angles):
    r"""Evaluate stacked polynomials at the uniform circle grid
    ``phi_m = exp(2 pi i m / n_angles)``, ``m = 0 .. n_angles - 1``.

    For ``p(phi) = sum_k c_k phi^k``, ``p(phi_m) = n_angles *
    ifft(c, n=n_angles)[m]`` (the zero-padded inverse FFT), evaluated here
    for every row of ``coefs`` at once.

    Parameters
    ----------
    coefs : ndarray, shape (nf, ncoef)
        Stacked polynomial coefficients (one polynomial per row).
    n_angles : int
        Number of uniform circle angles; must be >= ncoef (``np.fft.ifft``
        silently crops longer inputs).

    Returns
    -------
    values : ndarray, shape (nf, n_angles)
        ``p_i(phi_m)`` for every row ``i`` and angle ``m``.
    """
    if coefs.shape[1] > n_angles:
        raise ValueError("n_angles ({0}) must be >= the number of polynomial "
                         "coefficients ({1})".format(n_angles, coefs.shape[1]))
    return n_angles * np.fft.ifft(coefs, n=n_angles, axis=1)


def _horner_eval(coefs, phi):
    """Horner evaluation of polynomials at per-candidate points.

    ``coefs`` is either ``(nc, ncoef)`` (one coefficient row per candidate)
    or ``(ncoef,)`` (one polynomial shared by all candidates); ``phi`` is
    the ``(nc,)`` complex evaluation points.
    """
    out = np.zeros(phi.shape, dtype=np.complex128)
    for j in range(coefs.shape[-1] - 1, -1, -1):
        out *= phi
        out += coefs[..., j]
    return out


def scan_polish_from_coefs(YM_coefs, MM_coefs, AC, H, ybar, YY,
                           positive_amplitude=False, n_angles=None,
                           n_newton=_SCAN_NEWTON_STEPS):
    r"""Maximize the periodogram over phase by circle scan + Newton polish,
    vectorized over a stack of frequencies.

    The drop-in scan+polish replacement for the root-finding step
    (:func:`roots_from_YM_MM`) on stacked coefficients (e.g. one chunk from
    :func:`batched_YM_MM_from_sums`):

    1. evaluate ``YM``/``MM`` at ``M = max(128, 32 H)`` uniform circle
       angles for all frequencies at once (zero-padded inverse FFT);
    2. ``P = Re(YM^2 / MM) / YY`` with non-finite values (``|MM| ~ 0``,
       ``YY = 0``) mapped to ``-inf``;
    3. bracket every circular local maximum (no top-3 cap; degenerate
       plateaus are trimmed at ``4H`` candidates per frequency);
    4. polish each candidate with ``n_newton`` Newton steps on
       ``dP/dtheta`` using analytic first and second derivatives (Horner
       on the k- and k^2-weighted coefficient rows), each iterate clamped
       to the bracketing interval ``theta_0 +/- 2 pi / M``;
    5. evaluate the true ``P`` at the polished angles (falling back to the
       grid angle if polishing did not improve) and take the argmax,
       applying the same positive-amplitude filter and the same
       ``theta_1``/``theta_2``/``theta_3`` reconstruction formulas as
       :func:`roots_from_YM_MM`.

    Derivatives: with ``Y(theta) = YM(e^{i theta})``, ``Y1 = sum_k k y_k
    phi^k`` and ``Y2 = sum_k k^2 y_k phi^k`` (same for ``MM``),

        ``dP/dtheta   = Re(i B / MM^2) / YY``,  ``B = 2 Y Y1 MM - Y^2 M1``
        ``d2P/dtheta2 = Re((-2 (Y1^2 + Y Y2) MM + Y^2 M2) / MM^2
                          + 2 B M1 / MM^3) / YY``.

    Parameters
    ----------
    YM_coefs : ndarray, shape (nf, 2H+1)
        Stacked ``YM`` polynomial coefficients.
    MM_coefs : ndarray, shape (nf, 4H+1)
        Stacked ``MM`` polynomial coefficients.
    AC : ndarray, shape (nf, H)
        Per-frequency mean-template coefficients (offset reconstruction).
    H, ybar, YY, positive_amplitude
        As in :func:`roots_from_YM_MM`.
    n_angles : int, optional
        Override the scan grid size ``M`` (default ``max(128, 32 H)``).
    n_newton : int, optional
        Newton polish steps (default 8; the spec floor is 6).

    Returns
    -------
    params_list : list of ModelFitParams, length nf
    powers : ndarray, shape (nf,)
    best_phis : ndarray of complex, shape (nf,)
    """
    YM_coefs = np.atleast_2d(np.asarray(YM_coefs, dtype=np.complex128))
    MM_coefs = np.atleast_2d(np.asarray(MM_coefs, dtype=np.complex128))
    AC = np.atleast_2d(np.asarray(AC))
    nf = YM_coefs.shape[0]

    if n_angles is None:
        n_angles = max(_SCAN_MIN_ANGLES, _SCAN_ANGLES_PER_H * H)
    M = int(n_angles)

    flat_params = ModelFitParams(a=0.0, b=1.0, c=ybar, sgn=1.0)
    if nf == 0:
        return [], np.zeros(0), np.zeros(0, dtype=np.complex128)

    # -- 1-2: grid scan ------------------------------------------------
    Yv = _eval_polys_on_circle(YM_coefs, M)                 # (nf, M)
    Mv = _eval_polys_on_circle(MM_coefs, M)
    with np.errstate(divide='ignore', invalid='ignore'):
        Pg = np.real(Yv * Yv / Mv) / YY
    Pg[~np.isfinite(Pg)] = -np.inf

    # -- 3: bracket all circular local maxima --------------------------
    ismax = ((Pg >= np.roll(Pg, 1, axis=1)) &
             (Pg >= np.roll(Pg, -1, axis=1)) & (Pg > -np.inf))
    kcap = _SCAN_MAX_CANDIDATES_PER_H * H
    counts = ismax.sum(axis=1)
    for row in np.where(counts > kcap)[0]:
        cols = np.where(ismax[row])[0]
        keep = cols[np.argsort(Pg[row, cols])[-kcap:]]
        ismax[row] = False
        ismax[row, keep] = True

    fidx, gidx = np.where(ismax)            # row-major: fidx nondecreasing
    if len(fidx) == 0:
        return ([flat_params] * nf, np.zeros(nf),
                np.full(nf, 1.0 + 0.0j))

    theta0 = (2 * np.pi / M) * gidx
    P0 = Pg[fidx, gidx]

    # -- 4: Newton polish on dP/dtheta ---------------------------------
    YMc = YM_coefs[fidx]                                    # (nc, 2H+1)
    MMc = MM_coefs[fidx]                                    # (nc, 4H+1)
    kY = np.arange(YM_coefs.shape[1])
    kM = np.arange(MM_coefs.shape[1])
    dY1c, dY2c = YMc * kY, YMc * (kY * kY)
    dM1c, dM2c = MMc * kM, MMc * (kM * kM)

    half_window = 2 * np.pi / M
    theta = theta0.copy()
    for _ in range(n_newton):
        phi = np.exp(1j * theta)
        Y = _horner_eval(YMc, phi)
        Y1 = _horner_eval(dY1c, phi)
        Y2 = _horner_eval(dY2c, phi)
        Mm = _horner_eval(MMc, phi)
        M1 = _horner_eval(dM1c, phi)
        M2 = _horner_eval(dM2c, phi)
        with np.errstate(divide='ignore', invalid='ignore'):
            B = 2.0 * Y * Y1 * Mm - Y * Y * M1
            dP = np.real(1j * B / (Mm * Mm))
            d2P = np.real((-2.0 * (Y1 * Y1 + Y * Y2) * Mm + Y * Y * M2)
                          / (Mm * Mm) + 2.0 * B * M1 / (Mm * Mm * Mm))
            step = dP / d2P
        step[~np.isfinite(step)] = 0.0
        theta = np.clip(theta - step,
                        theta0 - half_window, theta0 + half_window)

    # -- 5: evaluate true P at polished angles, grid fallback ----------
    phi = np.exp(1j * theta)
    Yp = _horner_eval(YMc, phi)
    Mp = _horner_eval(MMc, phi)
    with np.errstate(divide='ignore', invalid='ignore'):
        Pp = np.real(Yp * Yp / Mp) / YY
    use_grid = ~np.isfinite(Pp) | (Pp < P0)
    theta = np.where(use_grid, theta0, theta)
    phi = np.where(use_grid, np.exp(1j * theta0), phi)
    Yc = np.where(use_grid, Yv[fidx, gidx], Yp)
    Mc = np.where(use_grid, Mv[fidx, gidx], Mp)
    Pc = np.where(use_grid, P0, Pp)

    # theta_1 at every candidate (same formula as the root path)
    with np.errstate(divide='ignore', invalid='ignore'):
        th1 = np.real(phi ** H * Yc / Mc)

    Peff = Pc
    if positive_amplitude:
        # K.18: among candidates, keep only those with theta_1 >= 0 and
        # pick the best-fitting one; if no candidate is non-negative,
        # fall back to the global power maximizer (as roots_from_YM_MM).
        pos = th1 >= 0
        anyp = np.zeros(nf, dtype=bool)
        np.logical_or.at(anyp, fidx, pos)
        Peff = np.where(pos | ~anyp[fidx], Pc, -np.inf)

    best = np.full(nf, -np.inf)
    np.maximum.at(best, fidx, Peff)

    winner = np.full(nf, -1, dtype=np.int64)
    hit = np.where(Peff == best[fidx])[0]
    rows, first = np.unique(fidx[hit], return_index=True)
    winner[rows] = hit[first]

    # -- parameter reconstruction (core formulas, vectorized) ----------
    ok_rows = np.where(winner >= 0)[0]
    w_idx = winner[ok_rows]
    bphi = phi[w_idx]
    theta_2 = np.imag(np.log(bphi)) % (2 * np.pi)
    # mbar = 2 Re(alpha_phi(best_phi)), alpha_phi = Polynomial([0, AC...])
    mb = np.zeros(len(ok_rows), dtype=np.complex128)
    ACw = AC[ok_rows]
    for j in range(AC.shape[1] - 1, -1, -1):
        mb = (mb + ACw[:, j]) * bphi
    mbar = 2 * np.real(mb)
    theta_1 = th1[w_idx]
    theta_3 = ybar - mbar * theta_1
    b_arr = np.cos(theta_2)
    sgn_arr = np.sign(np.sin(theta_2))

    powers = np.zeros(nf)
    best_phis = np.full(nf, 1.0 + 0.0j)
    powers[ok_rows] = Pc[w_idx]
    best_phis[ok_rows] = bphi

    params_list = [flat_params] * nf
    for k, i in enumerate(ok_rows):
        params_list[i] = ModelFitParams(a=theta_1[k], b=b_arr[k],
                                        c=theta_3[k], sgn=sgn_arr[k])

    return params_list, powers, best_phis


def scan_polish_YM_MM(YM, MM, AC, H, ybar, YY, positive_amplitude=False):
    r"""Per-frequency scan+polish maximizer with the same signature and
    return contract as :func:`roots_from_YM_MM` (minus the precomputed
    ``stationarity``, which the scan never needs).

    This is the seam the multiband solver flows through: the band-combined
    ``YM'``/``MM'`` polynomials have the same circle structure as the
    single-band ones, so the scan applies verbatim.

    Returns ``(params, power, best_phi)``.
    """
    params_list, powers, best_phis = scan_polish_from_coefs(
        YM.coef[np.newaxis, :], MM.coef[np.newaxis, :],
        np.asarray(AC)[np.newaxis, :], H, ybar, YY,
        positive_amplitude=positive_amplitude)
    return params_list[0], float(powers[0]), best_phis[0]


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
                                  tol=1E-7, chunk_size=4096,
                                  maximizer='roots'):
    """Batched-assembly template periodogram (``method='batched'`` /
    ``method='scan'``).

    Coefficient assembly (sums -> YM/MM) is vectorized over frequency
    chunks. With ``maximizer='roots'`` (``method='batched'``) the
    stationarity polynomial is also assembled and root-finding/selection
    reuses :func:`roots_from_YM_MM` per frequency, so the behavior matches
    the per-frequency path. With ``maximizer='scan'`` (``method='scan'``)
    the whole chunk is maximized at once by
    :func:`scan_polish_from_coefs` (circle scan + Newton polish; no
    stationarity polynomial, no root-finding).
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
        if maximizer == 'scan':
            plist, pw, _ = scan_polish_from_coefs(YM_coefs, MM_coefs, AC,
                                                  H, ybar, YY)
            powers.extend(pw)
            best_fit_params.extend(plist)
            continue
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
        numerically equivalent (powers agree to ~1e-15). 'scan' uses the
        same batched assembly but replaces root-finding entirely with the
        scan+polish maximizer (:func:`scan_polish_from_coefs`): an FFT
        circle scan over ``max(128, 32 H)`` angles plus Newton polish of
        every bracketed maximum -- numerically equivalent (powers agree to
        ~1e-13 of the root path) and much faster at high ``H``.
    chunk_size : int, optional (default 4096)
        Number of frequencies per assembly chunk on the batched/scan paths
        (bounds the memory of the ``(nf, H, H)`` covariance stacks); only
        used when `method` is 'batched' or 'scan'. Must be a positive
        integer; the powers are bitwise independent of its value.

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

    if method in ('batched', 'scan'):
        return _template_periodogram_batched(
            t, y, w, cn, sn, freqs, ybar, YY, summations=summations,
            fast=fast, sigma=sigma, tol=tol, chunk_size=chunk_size,
            maximizer='scan' if method == 'scan' else 'roots')
    if method != 'eigvals':
        raise ValueError("Unknown method {0!r}; must be 'eigvals', "
                         "'batched', or 'scan'".format(method))

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