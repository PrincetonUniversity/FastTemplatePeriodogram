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


def trim_zero_leading_coef(p, nominal_degree=None):
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
    trimmed the zero and the stored leading coefficient is genuine.

    Parameters
    ----------
    p : numpy.polynomial.Polynomial
        The freshly-assembled stationarity polynomial.
    nominal_degree : int, optional
        The polynomial's nominal (pre-cancellation) degree: ``6H - 1`` for
        the single/combined-band stationarity polynomial, ``8HK - 1`` for
        the multiband shared-phase ``G``. When given, the residue trim is
        applied ONLY if the polynomial still carries that nominal degree.
        Without this length gate, a genuine degree-``(6H - 2)`` leading
        coefficient that happens to sit below the residue threshold (deep
        ``|MM|`` dips shrink the whole tail) was eaten after numpy's
        arithmetic had already removed the exactly-cancelled zero,
        displacing the true stationary root (measured power deficits up to
        ~4e-8; C3 finding FO-TRIM-1, see VERIFICATION.md). ``None``
        preserves the legacy length-blind behavior.
    """
    coef = p.coef
    if len(coef) > 1:
        if nominal_degree is not None and len(coef) != nominal_degree + 1:
            # the exactly-cancelled analytic zero is already gone: the
            # stored leading coefficient is genuine, however small
            return p
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
        # (the length gate keeps a genuine small c_{6H-2} when numpy already
        # removed the exactly-cancelled zero -- C3 finding FO-TRIM-1)
        p = trim_zero_leading_coef(2 * MM * YM.deriv() - MM.deriv() * YM,
                                   nominal_degree=6 * H - 1)
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
# The spec is satisfied as "n_newton steps OR convergence": a candidate
# whose applied (post-clamp) update falls below _SCAN_NEWTON_XTOL is
# converged -- its remaining steps would move theta by <~ xtol, changing
# the power only at second order (~|d2P| xtol^2 / 2 <~ 1e-24 P), far
# below float64 resolution -- and exits the polish loop early (the
# 2026-07-18 efficiency audit measured the knee at ~3 steps).
_SCAN_NEWTON_STEPS = 8
_SCAN_NEWTON_XTOL = 1e-12
# keep at most 4H bracketed maxima per frequency (> 3H - 1, the analytic
# bound, so genuine maxima are never dropped; only degenerate plateaus --
# e.g. an exactly constant periodogram -- are trimmed)
_SCAN_MAX_CANDIDATES_PER_H = 4
# P(theta) = ym^2/mm with ym, mm trig polynomials of degree <= H, 2H: away
# from near-zeros of mm, Bernstein's inequality bounds P's feature scale
# below by ~1/(2H) of the circle (>> the 2pi/(32H) grid step), so a spike
# narrower than the grid REQUIRES a deep dip of |MM| on the circle
# (rank-deficient or phase-clustered sampling): a sub-grid-width spike needs
# mm_min <~ mm_max * ((2H) * (2pi/M))^2 / 2 ~ 0.08 mm_max at M = 32H.
# Every circular local minimum of |MM| below this fraction of the row max
# becomes an extra polish candidate; 0.5 keeps a ~6x safety margin.
# NB (C2 adversarial verification): real sub-grid P spikes were never observed
# above conditioning ~0.003 mm_max -- i.e. always inside the _SCAN_EXACT_RTOL
# (=0.15) fallback zone -- so the dip candidates in the (0.15, 0.5) band are a
# deliberate conservative hedge, not load-bearing on any tested valid input;
# the exact-root fallback below 0.15 is the actual correctness net.
_SCAN_DIP_RTOL = 0.5
# Below this depth the dip can host structures the clamped Newton polish is
# NOT guaranteed to resolve (e.g. a zero of ym inside the dip displaces the
# spike away from the |MM| minimum and breaks the seeded-Newton geometry);
# such frequencies are handed to the exact root path (roots_from_YM_MM)
# verbatim. A sub-grid-width spike needs mm_min <~ (pi^2/512) mm_max
# ~ 0.02 mm_max at M = 32H (Bernstein curvature bound), and the observed
# polish-failure modes sit below ~0.1 mm_max; 0.15 adds margin while
# keeping the fallback rare on well-conditioned data.
_SCAN_EXACT_RTOL = 0.15
# Cap for the escalated deep-dip scan density of the multiband shared_phase
# maximizer (C3.5 / MB-DIP-1): at deep-dip frequencies the scan grid is
# densified until the Bernstein bracketing bound
# dtheta <= sqrt(2 r_min)/(2H) (r_min = worst band's min|MM|/max|MM| on the
# circle) is met, so no F spike a dip of that depth can host escapes the
# grid; the cap bounds the FFT length when r_min is extreme. Below
# r_min ~ (4 pi H / cap)^2 / 2 (~4.6e-10 at H = 8 with the 2^20 cap) the
# bound is unmet and the remaining mitigations -- dip-Newton candidate
# seeds and max(scan, exact root path) -- are BEST-EFFORT, not guaranteed
# (MB-DIP-1 is itself proof the exact path can also fail there); the
# C3.5 panel measured residual true misses up to ~1.5e-4 at the old 2^17
# cap in that zone, recovered to the ~1e-6 local float64 evaluation floor
# by 2^20 (its own adjudicated recommendation). Note the escalation sizes
# the grid from the coarse-grid r_min measured once, so a dip deeper than
# coarse-measured is also covered only by the same best-effort net.
_SCAN_DEEP_MAX_ANGLES = 1 << 20


def _scan_dP_d2P(Y, Y1, Y2, Mm, M1, M2):
    r"""Analytic first and second theta-derivatives of ``YY * P`` =
    ``Re(Y^2 / Mm)`` at unit-circle points, from the polynomial values and
    their k- and k^2-weighted sums (``Y1 = sum_k k y_k phi^k`` etc.).

    With ``Y(theta) = YM(e^{i theta})``: ``dY/dtheta = i Y1``,
    ``d2Y/dtheta2 = -Y2`` (same for ``Mm``), giving

        ``dP   = Re(i B / Mm^2)``,  ``B = 2 Y Y1 Mm - Y^2 M1``
        ``d2P  = Re((-2 (Y1^2 + Y Y2) Mm + Y^2 M2) / Mm^2 + 2 B M1 / Mm^3)``.

    The ``1/YY`` factor is omitted consistently from both, so it cancels in
    the Newton ratio ``dP / d2P``. This is the single source of truth for
    the Newton polish (single-band and per-band shared-phase).
    """
    B = 2.0 * Y * Y1 * Mm - Y * Y * M1
    dP = np.real(1j * B / (Mm * Mm))
    d2P = np.real((-2.0 * (Y1 * Y1 + Y * Y2) * Mm + Y * Y * M2)
                  / (Mm * Mm) + 2.0 * B * M1 / (Mm * Mm * Mm))
    return dP, d2P


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


def _exact_root_fallback(rows, YM_coefs, MM_coefs, AC, H, ybar, YY,
                         positive_amplitude, params_list, powers, best_phis):
    """Overwrite the flagged rows (deep-|MM|-dip frequencies, see
    :data:`_SCAN_EXACT_RTOL`) with the exact root-path solution, in place.
    This is verbatim :func:`roots_from_YM_MM` -- the reference 'eigvals'
    treatment -- so the scan is guaranteed-equal to the reference exactly
    where the grid scan has no bracketing guarantee."""
    for i in rows:
        params_list[i], powers[i], best_phis[i] = roots_from_YM_MM(
            pol.Polynomial(YM_coefs[i]), pol.Polynomial(MM_coefs[i]),
            AC[i], H, ybar, YY, positive_amplitude=positive_amplitude)


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
       plateaus are trimmed at ``4H`` candidates per frequency), PLUS one
       candidate per deep circular local minimum of ``|MM|`` -- refined to
       the true ``|MM|^2`` minimum by clamped Newton -- because narrow
       peaks of ``P`` hide inside deep ``|MM|`` dips (rank-deficient or
       phase-clustered sampling) where the uniform grid cannot bracket
       them (see :data:`_SCAN_DIP_RTOL`); grid maxima that provably
       cannot win the (possibly K.18-constrained) argmax under the
       Bernstein polish-improvement bound are then dropped before the
       polish (step 3c in the source);
    4. polish each candidate with up to ``n_newton`` Newton steps on
       ``dP/dtheta`` using analytic first and second derivatives (Horner
       on the k- and k^2-weighted coefficient rows; see
       :func:`_scan_dP_d2P`), each iterate clamped to ``+/- 2 pi / M``
       around its seed; a candidate whose applied update falls below
       :data:`_SCAN_NEWTON_XTOL` is converged and exits the loop early;
    5. evaluate the true ``P`` at the polished angles (falling back to the
       seed angle if polishing did not improve) and take the argmax,
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
        Maximum Newton polish steps (default 8; the WP C2 spec floor of 6
        is met as "6 steps or convergence": a candidate exits early once
        its applied update falls below :data:`_SCAN_NEWTON_XTOL`, beyond
        which further steps change the power only at second order,
        ``<~ 1e-24``; the typical converged step count is ~3).

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

    # garbage in (NaN/inf observations, zero dy) must fail loudly, as the
    # root path does (np.roots raises LinAlgError on non-finite input); a
    # silent all-zero periodogram would corrupt downstream statistics
    if not (np.all(np.isfinite(YM_coefs)) and np.all(np.isfinite(MM_coefs))):
        raise ValueError("non-finite YM/MM polynomial coefficients (NaN/inf "
                         "in t, y, dy, or the weights?)")

    # the scan needs at least max(128, 32H) angles to bracket every circular
    # local maximum of P; an override below that floor undersamples the circle
    # and can silently miss a peak even on well-conditioned data, so clamp up
    # rather than honor an unsafe smaller value (resolution may only increase).
    floor = max(_SCAN_MIN_ANGLES, _SCAN_ANGLES_PER_H * H)
    M = floor if n_angles is None else max(int(n_angles), floor)

    flat_params = ModelFitParams(a=0.0, b=1.0, c=ybar, sgn=1.0)
    if nf == 0:
        return [], np.zeros(0), np.zeros(0, dtype=np.complex128)

    # -- 1-2: grid scan ------------------------------------------------
    Yv = _eval_polys_on_circle(YM_coefs, M)                 # (nf, M)
    Mv = _eval_polys_on_circle(MM_coefs, M)
    with np.errstate(divide='ignore', invalid='ignore'):
        Pg = np.real(Yv * Yv / Mv) / YY
    Pg[~np.isfinite(Pg)] = -np.inf

    kcap = _SCAN_MAX_CANDIDATES_PER_H * H

    # -- 3a: bracket all circular local maxima of P --------------------
    ismax = ((Pg >= np.roll(Pg, 1, axis=1)) &
             (Pg >= np.roll(Pg, -1, axis=1)) & (Pg > -np.inf))
    counts = ismax.sum(axis=1)
    for row in np.where(counts > kcap)[0]:
        cols = np.where(ismax[row])[0]
        keep = cols[np.argsort(Pg[row, cols])[-kcap:]]
        ismax[row] = False
        ismax[row, keep] = True
    max_f, max_g = np.where(ismax)

    # -- 3b: dip candidates at deep circular local minima of |MM| ------
    # Narrow peaks of P hide inside deep dips of |MM| (rank-deficient or
    # phase-clustered sampling drives MM toward zero on the circle); the P
    # grid scan cannot bracket a spike narrower than the grid step, and a
    # bracketed near-pole spike can defeat the clamped Newton polish (C2
    # adversarial verification, probe:sweep/probe:oracle). Each deep dip is
    # refined to the true |MM|^2 minimum by clamped Newton and polished
    # from there. Frequencies whose dip is so deep that even the seeded
    # polish has no convergence guarantee are recorded now and handed to
    # the exact root path at the end (see _SCAN_EXACT_RTOL).
    absM = np.abs(Mv)
    exact_rows = np.where(np.min(absM, axis=1) <
                          _SCAN_EXACT_RTOL * np.max(absM, axis=1))[0]
    ismin = ((absM <= np.roll(absM, 1, axis=1)) &
             (absM <= np.roll(absM, -1, axis=1)) &
             (absM < _SCAN_DIP_RTOL * np.max(absM, axis=1, keepdims=True)))
    counts = ismin.sum(axis=1)
    for row in np.where(counts > kcap)[0]:
        cols = np.where(ismin[row])[0]
        keep = cols[np.argsort(absM[row, cols])[:kcap]]
        ismin[row] = False
        ismin[row, keep] = True
    dip_f, dip_g = np.where(ismin)

    half_window = 2 * np.pi / M
    kY = np.arange(YM_coefs.shape[1])
    kM = np.arange(MM_coefs.shape[1])

    if len(dip_f):
        # refine each dip: Newton on d|MM|^2/dtheta = 2 Re(i M1 conj(MM)),
        # with d2|MM|^2/dtheta2 = 2 (|M1|^2 - Re(conj(MM) M2))
        dip_theta0 = (2 * np.pi / M) * dip_g
        MMd = MM_coefs[dip_f]
        dM1d, dM2d = MMd * kM, MMd * (kM * kM)
        th = dip_theta0.copy()
        active = np.arange(th.shape[0])
        for _ in range(n_newton):
            ph = np.exp(1j * th[active])
            Mm = _horner_eval(MMd[active], ph)
            M1 = _horner_eval(dM1d[active], ph)
            M2 = _horner_eval(dM2d[active], ph)
            with np.errstate(divide='ignore', invalid='ignore'):
                q1 = 2.0 * np.real(1j * M1 * np.conj(Mm))
                q2 = 2.0 * (np.abs(M1) ** 2 - np.real(np.conj(Mm) * M2))
                step = q1 / q2
            step[~np.isfinite(step)] = 0.0
            th_new = np.clip(th[active] - step,
                             dip_theta0[active] - half_window,
                             dip_theta0[active] + half_window)
            moved = np.abs(th_new - th[active]) > _SCAN_NEWTON_XTOL
            th[active] = th_new
            active = active[moved]
            if active.size == 0:
                break
        dip_phi = np.exp(1j * th)
        Y_dip = _horner_eval(YM_coefs[dip_f], dip_phi)
        M_dip = _horner_eval(MMd, dip_phi)
        with np.errstate(divide='ignore', invalid='ignore'):
            P_dip = np.real(Y_dip * Y_dip / M_dip) / YY
        P_dip[~np.isfinite(P_dip)] = -np.inf

        # combined candidate set; restore row-major (fidx-sorted) order for
        # the per-frequency winner extraction below
        fidx = np.concatenate([max_f, dip_f])
        theta0 = np.concatenate([(2 * np.pi / M) * max_g, th])
        P0 = np.concatenate([Pg[max_f, max_g], P_dip])
        Yfb = np.concatenate([Yv[max_f, max_g], Y_dip])
        Mfb = np.concatenate([Mv[max_f, max_g], M_dip])
        isdip = np.concatenate([np.zeros(len(max_f), dtype=bool),
                                np.ones(len(dip_f), dtype=bool)])
        order = np.argsort(fidx, kind='stable')
        fidx, theta0, P0 = fidx[order], theta0[order], P0[order]
        Yfb, Mfb, isdip = Yfb[order], Mfb[order], isdip[order]
    else:
        fidx = max_f
        theta0 = (2 * np.pi / M) * max_g
        P0 = Pg[max_f, max_g]
        Yfb = Yv[max_f, max_g]
        Mfb = Mv[max_f, max_g]
        isdip = np.zeros(len(max_f), dtype=bool)

    if len(fidx) == 0:
        params_list = [flat_params] * nf
        powers = np.zeros(nf)
        best_phis = np.full(nf, 1.0 + 0.0j)
        _exact_root_fallback(exact_rows, YM_coefs, MM_coefs, AC, H, ybar, YY,
                             positive_amplitude, params_list, powers,
                             best_phis)
        return params_list, powers, best_phis

    # -- 3c: Bernstein candidate filter --------------------------------
    # A bracketed grid local maximum can improve under polishing by at
    # most ``gain * max|P|`` on the circle (the true peak lies within one
    # grid step of the seed; second-derivative Bernstein bound
    # ``(2H)^2 max|P|`` for the degree-<=2H trig structure away from |MM|
    # dips), so a candidate whose grid value trails the row's best
    # candidate by more than TWICE that bound (2x margin) provably cannot
    # win the argmax and is dropped before the polish. Dip candidates are
    # exempt (their near-pole structure voids the smooth Bernstein
    # scale), and under the K.18 positive-amplitude constraint every
    # positive-seed candidate within the bound of the best positive seed
    # is also retained: the constrained winner need not be the global
    # maximizer. (2026-07-18 efficiency hunt, skeptic-vetted.)
    gain = (np.pi / M) ** 2 * (2.0 * H) ** 2 / 2.0
    rowmax = np.full(nf, -np.inf)
    np.maximum.at(rowmax, fidx, P0)
    rowabs = np.where(np.isfinite(Pg), np.abs(Pg), 0.0).max(axis=1)
    keep = isdip | (P0 >= (rowmax - 2.0 * gain * rowabs)[fidx])
    if positive_amplitude:
        with np.errstate(divide='ignore', invalid='ignore'):
            th1g = np.real(np.exp(1j * theta0) ** H * Yfb / Mfb)
        pos = th1g >= 0
        rowmax_pos = np.full(nf, -np.inf)
        np.maximum.at(rowmax_pos, fidx[pos], P0[pos])
        keep |= pos & (P0 >= (rowmax_pos - 2.0 * gain * rowabs)[fidx])
    if not np.all(keep):
        fidx, theta0, P0 = fidx[keep], theta0[keep], P0[keep]
        Yfb, Mfb = Yfb[keep], Mfb[keep]

    # -- 4: Newton polish on dP/dtheta ---------------------------------
    YMc = YM_coefs[fidx]                                    # (nc, 2H+1)
    MMc = MM_coefs[fidx]                                    # (nc, 4H+1)
    dY1c, dY2c = YMc * kY, YMc * (kY * kY)
    dM1c, dM2c = MMc * kM, MMc * (kM * kM)

    theta = theta0.copy()
    active = np.arange(theta.shape[0])
    for _ in range(n_newton):
        phi = np.exp(1j * theta[active])
        Y = _horner_eval(YMc[active], phi)
        Y1 = _horner_eval(dY1c[active], phi)
        Y2 = _horner_eval(dY2c[active], phi)
        Mm = _horner_eval(MMc[active], phi)
        M1 = _horner_eval(dM1c[active], phi)
        M2 = _horner_eval(dM2c[active], phi)
        with np.errstate(divide='ignore', invalid='ignore'):
            dP, d2P = _scan_dP_d2P(Y, Y1, Y2, Mm, M1, M2)
            step = dP / d2P
        step[~np.isfinite(step)] = 0.0
        theta_new = np.clip(theta[active] - step,
                            theta0[active] - half_window,
                            theta0[active] + half_window)
        moved = np.abs(theta_new - theta[active]) > _SCAN_NEWTON_XTOL
        theta[active] = theta_new
        active = active[moved]
        if active.size == 0:
            break

    # -- 5: evaluate true P at polished angles, seed fallback ----------
    phi = np.exp(1j * theta)
    Yp = _horner_eval(YMc, phi)
    Mp = _horner_eval(MMc, phi)
    with np.errstate(divide='ignore', invalid='ignore'):
        Pp = np.real(Yp * Yp / Mp) / YY
    use_grid = ~np.isfinite(Pp) | (Pp < P0)
    theta = np.where(use_grid, theta0, theta)
    phi = np.where(use_grid, np.exp(1j * theta0), phi)
    Yc = np.where(use_grid, Yfb, Yp)
    Mc = np.where(use_grid, Mfb, Mp)
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

    # rows whose candidates are all -inf (e.g. YY = 0: dip candidates exist
    # but no finite power anywhere) fall through to the flat fit
    winner = np.full(nf, -1, dtype=np.int64)
    hit = np.where((Peff == best[fidx]) & (Peff > -np.inf))[0]
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

    _exact_root_fallback(exact_rows, YM_coefs, MM_coefs, AC, H, ybar, YY,
                         positive_amplitude, params_list, powers, best_phis)

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
                        method='scan', chunk_size=4096):
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
    method : str, optional (default 'scan')
        'scan' -- the default since WP C4 -- is the scan+polish maximizer;
        'eigvals' is retained permanently as the reference/validation mode
        (the per-frequency path: polynomial objects assembled and rooted
        one frequency at a time). 'batched' assembles the
        YM/MM/stationarity coefficients for whole chunks of frequencies
        as stacked arrays (vectorized over frequency) and then runs the same
        per-frequency root selection on the precomputed coefficients; it is
        numerically equivalent (powers agree to ~1e-15). 'scan' uses the
        same batched assembly but replaces root-finding with the
        scan+polish maximizer (:func:`scan_polish_from_coefs`): an FFT
        circle scan over ``max(128, 32 H)`` angles plus Newton polish of
        every bracketed maximum and every deep ``|MM|`` dip; frequencies
        where ``|MM|`` nearly vanishes on the circle (rank-deficient or
        phase-clustered sampling) automatically fall back to the exact
        root path, so the two methods agree to ~1e-12 on data with circle
        conditioning ``min|MM|/max|MM|`` above ~1e-6, while 'scan' is much
        faster at high ``H`` on well-conditioned data. Below that the
        agreement degrades with conditioning (~3e-11 measured at ~1e-6,
        up to ~6e-9 at ~1e-9): the fallback guarantees equal *treatment*,
        not bitwise-equal values, because the scan path assembles
        coefficients via the batched (``np.trace``) sums whose last-ulp
        differences from the per-frequency assembly are amplified by the
        shared sums conditioning (C3 verification, 2026-07-04).
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