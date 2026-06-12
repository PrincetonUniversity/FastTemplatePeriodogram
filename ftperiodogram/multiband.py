"""
Multiband Fast Template Periodogram.

A drop-in sibling of :class:`ftperiodogram.modeler.FastTemplatePeriodogram` for
time series observed in multiple photometric bands (e.g. ZTF *g/r/i*).  It
implements the multiband "sharing hierarchy" derived in the thesis appendix
(``chapter-appendices/multiband-template.tex``) and the paper's multiband
notebooks, parametrized by what is held in common across bands:

==================  =====================================  ====================
``mode``            shared across bands                    free per band
==================  =====================================  ====================
``independent``     nothing (VanderPlas & Ivezic 2015)     amplitude, phase, offset
``shared_phase``    phase                                  amplitude, offset
``floating_offsets``  amplitude, phase                     offset  (**default**)
``sesar``           amplitude, phase, offset; fixed lambda  -- (Sesar et al. 2016)
==================  =====================================  ====================

The two coupled-but-cheap members (``floating_offsets`` and ``sesar``) reduce to
weighted averages of the per-band single-band quantities and reuse the
single-band polynomial machinery in :mod:`ftperiodogram.core` verbatim, so they
cost essentially the same as a single-band fit.

NAMING: *Multiband* means multiple **bands**.  Do not confuse this with
:class:`ftperiodogram.modeler.FastMultiTemplatePeriodogram`, which fits multiple
candidate **templates** to a single band.

DATA CONVENTION: flat ``(t, y, bands, dy)`` -- one row per observation with a
band label -- matching ``astropy``/``gatspy`` ``LombScargleMultiband`` and a ZTF
detection-table dump.  Each band's NFFT is computed on that band's own times,
which are sorted internally; the *global* array need not be sorted (a documented
relaxation of the single-band monotonic-``t`` contract).
"""
from __future__ import print_function

import warnings
from collections import namedtuple, OrderedDict

import numpy as np
import numpy.polynomial as pol

from .template import Template
from .utils import weights, ModelFitParams
from .summations import fast_summations, direct_summations
from .modeler import TemplateModel
from . import core as pdg


# Sharing-hierarchy members. The string values are the accepted ``mode=``.
MODES = ('independent', 'shared_phase', 'floating_offsets', 'sesar')
DEFAULT_MODE = 'floating_offsets'

# Phase-maximizer methods. 'eigvals' is the reference root-finding path;
# 'scan' is the scan+polish maximizer (WP C2): floating_offsets / sesar /
# independent flow through core.scan_polish_YM_MM on the (band-combined)
# YM/MM polynomials, and shared_phase scans the total power
# F(phi) = sum_k W_k Re(YM_k^2 / MM_k) directly on max(128, 32 H K)
# circle angles -- the degree-8HK polynomial G is never formed.
METHODS = ('eigvals', 'scan')
DEFAULT_METHOD = 'eigvals'


def _validate_method(method):
    if method not in METHODS:
        raise ValueError("Unknown method {0!r}; must be one of {1}"
                         "".format(method, METHODS))


# ----------------------------------------------------------------------
# Parameter / model containers (multiband analogs of ModelFitParams /
# TemplateModel). The per-band parameters are ALWAYS stored as a
# {band: ModelFitParams} dict -- a single source of truth, uniform across
# modes, with no per-mode type switching.
# ----------------------------------------------------------------------
class MultibandModelFitParams(namedtuple('MultibandModelFitParams',
                                          ['params_by_band', 'mode'])):
    """Best-fit multiband parameters; the multiband analog of ``ModelFitParams``.

    Parameters
    ----------
    params_by_band : dict {band: ModelFitParams}
        The ordinary single-band ``ModelFitParams`` actually applied in each
        band. For ``floating_offsets``/``sesar`` the amplitude (``a``) and phase
        (``b``, ``sgn``) are identical across bands and only the offset (``c``)
        differs; for ``independent``/``shared_phase`` they may all differ. The
        offset ``c`` is the full additive constant in that band (for ``sesar``
        it already includes the fixed relative offset ``lambda``).
    mode : str
        The sharing mode that produced these parameters.
    """
    __slots__ = ()

    def single_band(self, band):
        """Return the ordinary single-band ``ModelFitParams`` for one band."""
        return self.params_by_band[band]


class MultibandTemplateModel(object):
    """Callable multiband best-fit model; the multiband analog of ``TemplateModel``.

    Evaluates, in each band ``k``,

        y_hat^(k)(t) = a^(k) * M^(k)(freq * (t - tau^(k))) + c^(k)

    reconstructing ``tau`` from each band's ``(b, sgn)`` exactly as
    :class:`ftperiodogram.modeler.TemplateModel` does.

    Parameters
    ----------
    templates : dict {band: Template}
        The resolved per-band templates.
    frequency : float, optional (default 1.0)
        Frequency of the signal.
    parameters : MultibandModelFitParams
        Best-fit multiband parameters.
    """
    def __init__(self, templates, frequency=1.0, parameters=None):
        self.templates = templates
        self.frequency = frequency
        self.parameters = parameters

    def single_band_model(self, band):
        """Project to an ordinary single-band ``TemplateModel`` for one band."""
        return TemplateModel(self.templates[band], frequency=self.frequency,
                             parameters=self.parameters.single_band(band))

    def predict_band(self, t, band):
        """Evaluate the model light curve in a single band over times ``t``."""
        return self.single_band_model(band)(np.asarray(t, dtype=float))

    def __call__(self, t, bands):
        """Evaluate y_hat for flat arrays ``(t, bands)``, dispatching per band.

        Observations whose band was not present at fit time are returned as
        ``nan`` (the model has no parameters for them).
        """
        t = np.asarray(t, dtype=float)
        bands = np.asarray(bands)
        out = np.full(t.shape, np.nan, dtype=float)
        for band in self.parameters.params_by_band:
            mask = (bands == band)
            if np.any(mask):
                out[mask] = self.single_band_model(band)(t[mask])
        return out


# Frequency-independent per-band statistics, computed once per data set.
_BandStats = namedtuple('_BandStats',
                        ['bands', 'W', 'ybar', 'ybar_global',
                         'YY_per_band', 'YY_combined', 'YY_global', 'H'])


# ----------------------------------------------------------------------
# Template-set normalization
# ----------------------------------------------------------------------
def build_template_set(templates, bands):
    """Normalize a template specification into a ``{band: Template}`` dict.

    Accepts a single shared ``Template``, a ``{band: Template}`` dict, or a
    list/tuple aligned to ``sorted(unique(bands))``. Validates that every band
    is covered and that all band templates share a common harmonic count ``H``.
    """
    band_labels = list(np.unique(bands))

    if isinstance(templates, Template):
        template_dict = {band: templates for band in band_labels}
    elif isinstance(templates, dict):
        template_dict = dict(templates)
    elif isinstance(templates, (list, tuple)):
        if len(templates) != len(band_labels):
            raise ValueError("template list length ({0}) does not match the "
                             "number of bands ({1})".format(len(templates),
                                                            len(band_labels)))
        template_dict = {band: tmpl
                         for band, tmpl in zip(band_labels, templates)}
    else:
        raise TypeError("templates must be a Template, a {band: Template} dict, "
                        "or a list/tuple of Templates")

    harmonics = set()
    for band in band_labels:
        if band not in template_dict:
            raise ValueError("No template provided for band {0!r}".format(band))
        tmpl = template_dict[band]
        if not isinstance(tmpl, Template):
            raise ValueError("Template for band {0!r} is not a Template "
                             "instance".format(band))
        harmonics.add(len(tmpl.c_n))

    if len(harmonics) > 1:
        raise ValueError("All band templates must share the same number of "
                         "harmonics (got {0})".format(sorted(harmonics)))

    return template_dict


# ----------------------------------------------------------------------
# Band bookkeeping
# ----------------------------------------------------------------------
def _prepare_bands(t, y, bands, dy, mode, relative_offsets, H):
    """Split flat data into per-band, time-sorted slices and summary statistics.

    Returns ``(band_data, stats)`` where ``band_data`` maps each band to its
    time-sorted ``(t_k, y_k, w_k)`` (with ``w_k`` renormalized to sum to 1
    within the band, as required for the per-band NFFT) and ``stats`` is a
    :class:`_BandStats` carrying the global/per-band means, variances, and band
    weight totals ``W_k`` (which sum to 1 over all bands).

    For ``mode='sesar'`` the fixed relative offsets ``lambda^(k)`` are subtracted
    from ``y`` up front, so the means/variances/sums are all computed on the
    offset-subtracted data.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    bands = np.asarray(bands)
    n = len(t)

    if dy is None:
        err = np.ones(n)
    else:
        err = np.broadcast_to(np.asarray(dy, dtype=float), (n,)).astype(float)
    w_global = weights(err)

    band_labels = list(np.unique(bands))

    y_work = y.copy()
    if mode == 'sesar':
        if relative_offsets is None:
            raise ValueError("mode='sesar' requires relative_offsets={band: "
                             "lambda}; none were provided.")
        for band in band_labels:
            if band not in relative_offsets:
                raise ValueError("relative_offsets is missing band {0!r}"
                                 "".format(band))
            y_work[bands == band] -= relative_offsets[band]

    ybar_global = float(np.dot(w_global, y_work))
    YY_global = float(np.dot(w_global, (y_work - ybar_global) ** 2))

    band_data = OrderedDict()
    W, ybar, YY_per_band = {}, {}, {}
    for band in band_labels:
        mask = (bands == band)
        order = np.argsort(t[mask], kind='mergesort')
        t_k = t[mask][order]
        y_k = y_work[mask][order]
        w_k_global = w_global[mask][order]

        W_k = float(w_k_global.sum())
        w_k = w_k_global / W_k                       # sum to 1 within the band
        ybar_k = float(np.dot(w_k, y_k))
        YY_k = float(np.dot(w_k, (y_k - ybar_k) ** 2))

        band_data[band] = (t_k, y_k, w_k)
        W[band] = W_k
        ybar[band] = ybar_k
        YY_per_band[band] = YY_k

    # Combined (model-C) variance: deviations from per-band means.
    YY_combined = float(sum(W[b] * YY_per_band[b] for b in band_labels))

    stats = _BandStats(bands=band_labels, W=W, ybar=ybar,
                       ybar_global=ybar_global, YY_per_band=YY_per_band,
                       YY_combined=YY_combined, YY_global=YY_global, H=H)
    return band_data, stats


def _mean_template_poly(AC):
    """Build the (phi**H-scaled) mean-template polynomial ``M_bar(phi)`` from AC."""
    AC = np.asarray(AC)
    coeffs = np.concatenate((np.conj(AC)[::-1], [0], AC)).astype(np.complex128)
    return pol.Polynomial(coeffs)


# ----------------------------------------------------------------------
# The new multiband math: combining per-band YM/MM into YM'/MM'.
# ----------------------------------------------------------------------
def combine_band_summations(per_band_YM_MM, W, ybar, ybar_global, mode):
    r"""Combine per-band ``YM``/``MM`` polynomials into the band-combined pair.

    The per-band polynomials come from :func:`core.YM_MM_from_sums`, which
    centers each band by its **own** weighted mean. For ``floating_offsets``
    (model C) each band keeps its own offset, so the combination is simply the
    ``W_k``-weighted average

        YM' = sum_k W_k YM^(k),    MM' = sum_k W_k MM^(k).

    For ``sesar`` (model D) there is a single shared offset, so the quantities
    must be re-centered on the **global** mean (thesis lines 76-78):

        YM' = sum_k W_k YM^(k) + sum_k W_k (ybar_k - ybar_global) Mbar^(k)
        MM' = sum_k W_k MM^(k) + sum_k W_k (Mbar^(k))^2 - (sum_k W_k Mbar^(k))^2

    where ``Mbar^(k)`` is band ``k``'s (phi**H-scaled) mean-template polynomial.
    At ``K = 1`` (``ybar_k == ybar_global``, ``W_k == 1``) every correction term
    vanishes and both modes collapse to the single-band ``YM``/``MM``.

    Returns ``(YM, MM, AC)`` where ``AC`` is the ``W_k``-weighted mean-template
    coefficient vector for the (global) offset reconstruction in the root tail.
    """
    bands = list(per_band_YM_MM)

    YM = MM = None
    AC = None
    for band in bands:
        YM_k, MM_k, AC_k = per_band_YM_MM[band]
        wk = W[band]
        YM = wk * YM_k if YM is None else YM + wk * YM_k
        MM = wk * MM_k if MM is None else MM + wk * MM_k
        AC_k = wk * np.asarray(AC_k)
        AC = AC_k if AC is None else AC + AC_k

    if mode == 'sesar':
        YM_corr = MM_corr = Mbar_comb = None
        for band in bands:
            AC_k = per_band_YM_MM[band][2]
            wk = W[band]
            Mbar_k = _mean_template_poly(AC_k)

            term = wk * ((ybar[band] - ybar_global) * Mbar_k)
            YM_corr = term if YM_corr is None else YM_corr + term

            sq = wk * (Mbar_k * Mbar_k)
            MM_corr = sq if MM_corr is None else MM_corr + sq

            wk_mbar = wk * Mbar_k
            Mbar_comb = wk_mbar if Mbar_comb is None else Mbar_comb + wk_mbar

        YM = YM + YM_corr
        MM = MM + MM_corr - Mbar_comb * Mbar_comb

    return YM, MM, AC


# ----------------------------------------------------------------------
# Per-frequency multiband solve (analog of core.template_fit_from_sums).
# ----------------------------------------------------------------------
def _mbar_at(AC, phi):
    """Mean-template value ``2 Re(sum_n AC_n phi^n)`` at ``phi`` (offset recon)."""
    return 2 * np.real(pol.Polynomial(np.concatenate(([0], AC)))(phi))


def _per_band_YM_MM(template_dict, per_band_sums, bands):
    """Build ``{band: (YM, MM, AC)}`` via :func:`core.YM_MM_from_sums` per band."""
    return {band: pdg.YM_MM_from_sums(template_dict[band].c_n,
                                      template_dict[band].s_n,
                                      per_band_sums[band])
            for band in bands}


def _flat_model(stats, mode):
    """Degenerate zero-amplitude, zero-power result for a flat (no-variance)
    light curve, used when the relevant reference variance is zero."""
    pbb = {band: ModelFitParams(a=0.0, b=1.0, c=stats.ybar[band], sgn=1.0)
           for band in stats.bands}
    return MultibandModelFitParams(pbb, mode), 0.0


# A light curve (or a single band) is treated as flat when its weighted variance
# is negligible relative to its mean-magnitude scale. The exact ``YY <= 0`` test
# only fires when the centering is bit-exact (true for constant input in our
# numpy, but not guaranteed across platforms/BLAS); this relative test is robust.
_FLAT_VARIANCE_RTOL = 1e-12


def _is_flat(YY, ref):
    """True if weighted variance ``YY`` is negligible vs the ``ref`` magnitude."""
    return YY <= _FLAT_VARIANCE_RTOL * max(ref * ref, 1.0)


def multiband_template_fit_from_sums(template_dict, per_band_sums, stats, mode,
                                     relative_offsets=None,
                                     method=DEFAULT_METHOD):
    """Solve the multiband template fit at one frequency from precomputed sums.

    Returns ``(MultibandModelFitParams, power)``. For every mode except
    ``shared_phase`` the better-fitting ``theta_1 >= 0`` solution is returned
    when one exists (paper item K.18); ``shared_phase`` shares the phase across
    bands and so reports per-band amplitudes as fitted. ``method`` selects the
    phase maximizer (see :data:`METHODS`); both are numerically equivalent.
    """
    _validate_method(method)
    solver = (pdg.scan_polish_YM_MM if method == 'scan'
              else pdg.roots_from_YM_MM)
    H = stats.H
    bands = stats.bands

    if mode in ('floating_offsets', 'sesar'):
        YY = stats.YY_global if mode == 'sesar' else stats.YY_combined
        if _is_flat(YY, stats.ybar_global):   # flat light curve: nothing to detect
            return _flat_model(stats, mode)

        per_band_YM_MM = _per_band_YM_MM(template_dict, per_band_sums, bands)
        YM, MM, AC = combine_band_summations(per_band_YM_MM, stats.W, stats.ybar,
                                             stats.ybar_global, mode)
        shared, power, best_phi = solver(
            YM, MM, AC, H, stats.ybar_global, YY, positive_amplitude=True)

        params_by_band = {}
        for band in bands:
            AC_k = per_band_YM_MM[band][2]
            if mode == 'sesar':
                # single shared offset, plus the fixed relative offset lambda^(k)
                c_k = shared.c + relative_offsets[band]
            else:
                # floating offset: theta_3^(k) = ybar_k - theta_1 * Mbar^(k)(phi)
                c_k = stats.ybar[band] - shared.a * _mbar_at(AC_k, best_phi)
            params_by_band[band] = ModelFitParams(a=shared.a, b=shared.b,
                                                  c=c_k, sgn=shared.sgn)

        return MultibandModelFitParams(params_by_band, mode), float(power)

    if mode == 'independent':
        if _is_flat(stats.YY_combined, stats.ybar_global):
            return _flat_model(stats, mode)
        per_band_YM_MM = _per_band_YM_MM(template_dict, per_band_sums, bands)
        params_by_band = {}
        explained = 0.0
        for band in bands:
            if _is_flat(stats.YY_per_band[band], stats.ybar[band]):
                # A flat band explains no variance and yields no roots to find;
                # skip the solver (which would argmax an empty root set) and
                # report a zero-amplitude fit. Its explained contribution is 0.
                params_by_band[band] = ModelFitParams(a=0.0, b=1.0,
                                                       c=stats.ybar[band], sgn=1.0)
                continue
            YM_k, MM_k, AC_k = per_band_YM_MM[band]
            params_k, power_k, _ = solver(
                YM_k, MM_k, AC_k, H, stats.ybar[band], stats.YY_per_band[band],
                positive_amplitude=True)
            params_by_band[band] = params_k
            # Accumulate explained variance and normalize once by the combined
            # variance, so the reported power is the global fraction of variance
            # explained -- comparable to the other modes. A plain W_k-weighted
            # average of per-band powers would over-weight low-variance bands.
            explained += stats.W[band] * power_k * stats.YY_per_band[band]
        power = explained / stats.YY_combined
        return MultibandModelFitParams(params_by_band, mode), float(power)

    if mode == 'shared_phase':
        if method == 'scan':
            return _shared_phase_scan_fit(template_dict, per_band_sums, stats)
        return _shared_phase_fit(template_dict, per_band_sums, stats)

    raise ValueError("Unknown mode {0!r}; must be one of {1}".format(mode, MODES))


def _shared_phase_fit(template_dict, per_band_sums, stats):
    r"""Solve model B (shared phase, per-band amplitude and offset) at one freq.

    For a fixed shared phase :math:`\phi` each band fits independently, so the
    total power is

        F(phi) = sum_k W_k (P_YM^(k)(phi))^2 / P_MM^(k)(phi)

    (normalized by the per-band-centered variance ``YY_combined``). Maximizing
    over phi, after clearing the common denominator ``prod_l (P_MM^(l))^2``
    (which is positive on the unit circle), gives the stationarity polynomial

        G(phi) = sum_k W_k P_YM^(k) q_k prod_{l != k} (P_MM^(l))^2 = 0,
        q_k = 2 P_MM^(k) (P_YM^(k))' - (P_MM^(k))' P_YM^(k),

    of degree at most ``8 H K - 2`` (each term's nominal ``8 H K - 1`` leading
    coefficient cancels exactly, by the same top-term cancellation as the
    single-band ``q_k``) -- a genuinely larger root problem than the
    single-band case (cost grows as ``(H K)^3`` per frequency).

    POSITIVITY: unlike the shared-amplitude modes, the phase here is shared, so
    the per-band amplitudes cannot each be independently forced positive; this
    routine returns the power-maximizing shared phase and reports the per-band
    amplitudes as fitted (they are >= 0 for a genuine in-phase multiband signal,
    but can be negative for a band that anti-correlates at the shared phase).
    """
    H = stats.H
    bands = stats.bands

    if _is_flat(stats.YY_combined, stats.ybar_global):   # flat: nothing to detect
        return _flat_model(stats, 'shared_phase')

    per_band = _per_band_YM_MM(template_dict, per_band_sums, bands)
    PMM = {band: per_band[band][1] for band in bands}

    G = None
    for k in bands:
        P_YM, P_MM, _ = per_band[k]
        q = 2 * P_MM * P_YM.deriv() - P_MM.deriv() * P_YM
        prod_others = pol.Polynomial([1.0 + 0j])
        for l in bands:
            if l != k:
                prod_others = prod_others * (PMM[l] * PMM[l])
        term = stats.W[k] * (P_YM * q * prod_others)
        G = term if G is None else G + term

    # true degree <= 8HK-2; the nominal leading coefficient is FP residue
    G = pdg.trim_zero_leading_coef(G)

    roots = G.roots()
    roots = roots[np.absolute(roots) > 0]
    if len(roots) == 0:                   # no stationary phase: degenerate
        return _flat_model(stats, 'shared_phase')
    roots /= np.absolute(roots)

    # total power at each candidate shared phase; guard against a root that is
    # also (near) a root of some band's P_MM, which would make a term blow up.
    with np.errstate(divide='ignore', invalid='ignore'):
        F = np.zeros(len(roots))
        for k in bands:
            P_YM, P_MM, _ = per_band[k]
            F += stats.W[k] * np.real(P_YM(roots) ** 2 / P_MM(roots))
    finite = np.isfinite(F)
    if not np.any(finite):
        return _flat_model(stats, 'shared_phase')
    F = np.where(finite, F, -np.inf)

    i = int(np.argmax(F))
    best_phi = roots[i]
    power = F[i] / stats.YY_combined

    theta_2 = np.imag(np.log(best_phi)) % (2 * np.pi)
    b = np.cos(theta_2)
    sgn = np.sign(np.sin(theta_2))

    params_by_band = {}
    for k in bands:
        P_YM, P_MM, AC_k = per_band[k]
        theta_1_k = np.real(np.power(best_phi, H) * P_YM(best_phi) / P_MM(best_phi))
        c_k = stats.ybar[k] - theta_1_k * _mbar_at(AC_k, best_phi)
        params_by_band[k] = ModelFitParams(a=theta_1_k, b=b, c=c_k, sgn=sgn)

    return MultibandModelFitParams(params_by_band, 'shared_phase'), float(power)


def _shared_phase_scan_fit(template_dict, per_band_sums, stats):
    r"""Scan+polish solve of model B (shared phase) at one frequency.

    Maximizes the total power

        F(phi) = sum_k W_k Re(P_YM^(k)(phi)^2 / P_MM^(k)(phi))

    directly on ``M = max(128, 32 H K)`` uniform circle angles (the per-band
    polynomials evaluated by zero-padded inverse FFT), then polishes every
    bracketed circular local maximum with Newton steps on ``dF/dtheta``
    using the analytic per-band derivatives of
    :func:`ftperiodogram.core.scan_polish_from_coefs`, summed over bands
    with weights ``W_k``. The degree-``8HK`` stationarity polynomial ``G``
    of :func:`_shared_phase_fit` is never formed, so the ``(HK)^3``
    per-frequency root-finding cost is avoided entirely.

    Parameter reconstruction (per-band ``theta_1``, offsets) is identical
    to :func:`_shared_phase_fit`.
    """
    H = stats.H
    bands = stats.bands
    K = len(bands)

    if _is_flat(stats.YY_combined, stats.ybar_global):   # flat: nothing to detect
        return _flat_model(stats, 'shared_phase')

    per_band = _per_band_YM_MM(template_dict, per_band_sums, bands)

    Yco = np.array([per_band[k][0].coef for k in bands])     # (K, 2H+1)
    Mco = np.array([per_band[k][1].coef for k in bands])     # (K, 4H+1)
    Wv = np.array([stats.W[k] for k in bands])

    M_ang = max(pdg._SCAN_MIN_ANGLES, pdg._SCAN_ANGLES_PER_H * H * K)

    # grid scan of F(theta)
    Yg = pdg._eval_polys_on_circle(Yco, M_ang)               # (K, M)
    Mg = pdg._eval_polys_on_circle(Mco, M_ang)
    with np.errstate(divide='ignore', invalid='ignore'):
        Fg = np.dot(Wv, np.real(Yg * Yg / Mg))               # (M,)
    Fg[~np.isfinite(Fg)] = -np.inf

    ismax = ((Fg >= np.roll(Fg, 1)) & (Fg >= np.roll(Fg, -1)) &
             (Fg > -np.inf))
    if not np.any(ismax):                 # no finite total power anywhere
        return _flat_model(stats, 'shared_phase')
    kcap = pdg._SCAN_MAX_CANDIDATES_PER_H * H * K
    cols = np.where(ismax)[0]
    if len(cols) > kcap:                  # degenerate plateau: trim
        cols = cols[np.argsort(Fg[cols])[-kcap:]]

    theta0 = (2 * np.pi / M_ang) * cols
    F0 = Fg[cols]

    # Newton polish on dF/dtheta (per-band derivative coefficient rows)
    kY = np.arange(Yco.shape[1])
    kM = np.arange(Mco.shape[1])
    half_window = 2 * np.pi / M_ang
    theta = theta0.copy()

    def _F_derivs(phi):
        """(dF/dtheta, d2F/dtheta2) at unit-circle points ``phi``."""
        dF = np.zeros(phi.shape)
        d2F = np.zeros(phi.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            for kb in range(K):
                Y = pdg._horner_eval(Yco[kb], phi)
                Y1 = pdg._horner_eval(Yco[kb] * kY, phi)
                Y2 = pdg._horner_eval(Yco[kb] * kY * kY, phi)
                Mm = pdg._horner_eval(Mco[kb], phi)
                M1 = pdg._horner_eval(Mco[kb] * kM, phi)
                M2 = pdg._horner_eval(Mco[kb] * kM * kM, phi)
                B = 2.0 * Y * Y1 * Mm - Y * Y * M1
                dF = dF + Wv[kb] * np.real(1j * B / (Mm * Mm))
                d2F = d2F + Wv[kb] * np.real(
                    (-2.0 * (Y1 * Y1 + Y * Y2) * Mm + Y * Y * M2)
                    / (Mm * Mm) + 2.0 * B * M1 / (Mm * Mm * Mm))
        return dF, d2F

    def _F_at(phi):
        """Total power F at unit-circle points ``phi``."""
        F = np.zeros(phi.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            for kb in range(K):
                Y = pdg._horner_eval(Yco[kb], phi)
                Mm = pdg._horner_eval(Mco[kb], phi)
                F = F + Wv[kb] * np.real(Y * Y / Mm)
        return F

    for _ in range(pdg._SCAN_NEWTON_STEPS):
        dF, d2F = _F_derivs(np.exp(1j * theta))
        with np.errstate(divide='ignore', invalid='ignore'):
            step = dF / d2F
        step[~np.isfinite(step)] = 0.0
        theta = np.clip(theta - step,
                        theta0 - half_window, theta0 + half_window)

    Fp = _F_at(np.exp(1j * theta))
    use_grid = ~np.isfinite(Fp) | (Fp < F0)
    theta = np.where(use_grid, theta0, theta)
    Fc = np.where(use_grid, F0, Fp)

    i = int(np.argmax(Fc))
    best_phi = np.exp(1j * theta[i])
    power = Fc[i] / stats.YY_combined

    # reconstruction identical to _shared_phase_fit
    theta_2 = np.imag(np.log(best_phi)) % (2 * np.pi)
    b = np.cos(theta_2)
    sgn = np.sign(np.sin(theta_2))

    params_by_band = {}
    for k in bands:
        P_YM, P_MM, AC_k = per_band[k]
        theta_1_k = np.real(np.power(best_phi, H) * P_YM(best_phi) / P_MM(best_phi))
        c_k = stats.ybar[k] - theta_1_k * _mbar_at(AC_k, best_phi)
        params_by_band[k] = ModelFitParams(a=theta_1_k, b=b, c=c_k, sgn=sgn)

    return MultibandModelFitParams(params_by_band, 'shared_phase'), float(power)


def compute_band_summations(t, y, bands, freqs, H, dy=None, mode=DEFAULT_MODE,
                            relative_offsets=None, fast=True):
    """Compute the per-band NFFT summations and per-band statistics.

    The returned ``per_band_sumlists`` (one ``Summations`` per frequency, per
    band) depend only on the data and ``(mode, relative_offsets, H)`` -- NOT on
    the template shape -- so they can be computed once and reused across a whole
    template catalog. Returns ``(per_band_sumlists, stats)``.
    """
    band_data, stats = _prepare_bands(t, y, bands, dy, mode,
                                      relative_offsets, H)
    freqs = np.asarray(freqs, dtype=float)

    per_band_sumlists = OrderedDict()
    for band, (t_k, y_k, w_k) in band_data.items():
        if fast:
            per_band_sumlists[band] = fast_summations(t_k, y_k, w_k, freqs, H)
        else:
            per_band_sumlists[band] = direct_summations(t_k, y_k, w_k, freqs, H)

    return per_band_sumlists, stats


def solve_over_frequencies(template_dict, per_band_sumlists, stats, nfreq,
                           mode=DEFAULT_MODE, relative_offsets=None,
                           method=DEFAULT_METHOD):
    """Run the per-frequency multiband solve given precomputed per-band sums.

    Pairs with :func:`compute_band_summations`; the template-dependent half of
    the periodogram. Returns ``(powers, params_list)``.
    """
    bands = list(per_band_sumlists)
    powers = np.empty(nfreq, dtype=float)
    params_list = []
    for i in range(nfreq):
        per_band_sums = {band: per_band_sumlists[band][i] for band in bands}
        mb_params, power = multiband_template_fit_from_sums(
            template_dict, per_band_sums, stats, mode, relative_offsets,
            method=method)
        powers[i] = power
        params_list.append(mb_params)

    return powers, params_list


def multiband_template_periodogram(t, y, bands, template_dict, freqs, dy=None,
                                   mode=DEFAULT_MODE, relative_offsets=None,
                                   fast=True, method=DEFAULT_METHOD):
    """Multiband template periodogram over flat data; analog of
    :func:`core.template_periodogram`.

    This is the functional entry point (no class construction), suited to
    batching many sources. Per-band NFFT summations are computed on each band's
    own time-sorted samples, then the per-frequency multiband solve runs at each
    frequency.

    Returns ``(powers, params_list)``: a power array and the list of
    :class:`MultibandModelFitParams` at each frequency.
    """
    H = len(next(iter(template_dict.values())).c_n)
    freqs = np.asarray(freqs, dtype=float)
    per_band_sumlists, stats = compute_band_summations(
        t, y, bands, freqs, H, dy=dy, mode=mode,
        relative_offsets=relative_offsets, fast=fast)
    return solve_over_frequencies(template_dict, per_band_sumlists, stats,
                                  len(freqs), mode, relative_offsets,
                                  method=method)


# ----------------------------------------------------------------------
# Public class (drop-in sibling of FastTemplatePeriodogram).
# ----------------------------------------------------------------------
class FastMultibandTemplatePeriodogram(object):
    """Multiband template periodogram.

    Fits a single (possibly per-band) template shape to multiband data under one
    of the sharing modes in :data:`MODES`.

    Parameters
    ----------
    templates : Template, dict {band: Template}, or list of those
        A shared template shape (one ``Template`` for every band) or explicit
        per-band shapes (a ``{band: Template}`` dict). A **list/tuple** is a
        *catalog* of template-sets (each element a ``Template`` or a
        ``{band: Template}`` dict); the periodogram then reports, at each
        frequency, the maximum power over the sets, and records the winning set
        index on ``best_model.template_set_index``. All templates share the
        harmonic count ``H``. (Per-band assignment uses a dict; a top-level list
        is always interpreted as a catalog.)
    mode : str, optional (default ``'floating_offsets'``)
        The sharing-hierarchy member; one of :data:`MODES`.
    relative_offsets : dict {band: float}, optional
        Fixed per-band offsets ``lambda^(k)``; **required** for ``mode='sesar'``
        and ignored otherwise.

    Notes
    -----
    Distinct from :class:`ftperiodogram.modeler.FastMultiTemplatePeriodogram`,
    which fits multiple candidate *templates* to a single band.
    """
    def __init__(self, templates=None, mode=DEFAULT_MODE, relative_offsets=None):
        self.templates = templates
        self.mode = mode
        self.relative_offsets = relative_offsets
        self.t, self.y, self.bands, self.dy = None, None, None, None
        self.bands_ = None
        self.best_model = None
        self._is_catalog = False
        self._template_dict = None
        self._template_sets = None

    # -- validation -----------------------------------------------------
    def _validate_mode(self):
        if self.mode not in MODES:
            raise ValueError("Unknown mode {0!r}; must be one of {1}"
                             "".format(self.mode, MODES))
        if self.mode == 'sesar' and self.relative_offsets is None:
            raise ValueError("mode='sesar' requires relative_offsets="
                             "{band: lambda}.")

    def _validate_templates(self):
        if self.templates is None:
            raise ValueError("No templates set.")
        if isinstance(self.templates, (list, tuple)):
            self._is_catalog = True
            self._template_sets = [build_template_set(s, self.bands_)
                                   for s in self.templates]
        else:
            self._is_catalog = False
            self._template_dict = build_template_set(self.templates,
                                                     self.bands_)

    def _validate_data(self):
        if any(x is None for x in (self.t, self.y, self.bands)):
            raise ValueError("fit(t, y, bands, dy) must be called first.")
        if not (len(self.t) == len(self.y) == len(self.bands)):
            raise ValueError("t, y, and bands must have equal lengths.")
        if self.dy is not None and np.ndim(self.dy) > 0 \
                and len(self.dy) != len(self.t):
            raise ValueError("dy must be scalar or the same length as t.")

    # -- fitting --------------------------------------------------------
    def fit(self, t, y, bands, dy=None):
        """Store flat ``(t, y, bands, dy)``; record the sorted band labels.

        Parameters
        ----------
        t, y : array_like
            Flat observation times and values (in any global order; each band's
            times are sorted internally for its NFFT).
        bands : array_like
            Per-observation band labels (a single hashable dtype, e.g.
            ``'g'``/``'r'``/``'i'`` or integers).
        dy : float or array_like, optional
            Observational uncertainties; ``None`` or a scalar weights all
            observations equally.

        Returns
        -------
        self
        """
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.bands = np.asarray(bands)
        self.dy = dy
        self.bands_ = np.unique(self.bands)
        # a best fit from previously-fit data would be stale for this dataset
        self.best_model = None
        return self

    def autofrequency(self, nyquist_factor=5, samples_per_peak=5,
                      minimum_frequency=None, maximum_frequency=None):
        """Determine a trial-frequency grid over the global observation baseline.

        For irregularly-sampled data the average-Nyquist heuristic is not
        meaningful; the recommended path is to set ``minimum_frequency`` and
        ``maximum_frequency`` explicitly (the grid then starts at the point
        at/just below the requested minimum and ends at the first point
        at/above the requested maximum, so both bounds are searched).
        ``nyquist_factor`` is deprecated and retained only for API parity with
        the single-band class; relying on it warns.
        """
        baseline = self.t.max() - self.t.min()
        n_samples = self.t.size

        df = 1. / (baseline * samples_per_peak)

        if minimum_frequency is not None:
            # start the grid at (or just above) the requested minimum frequency
            nf0 = max(1, int(np.floor(minimum_frequency / df)))
        else:
            nf0 = 1

        if maximum_frequency is not None:
            # end at the first grid point >= maximum_frequency, so the
            # requested maximum is included (the grid stays on multiples of df)
            Nf = int(np.ceil(maximum_frequency / df - nf0)) + 1
        else:
            warnings.warn(
                "Choosing the maximum frequency from nyquist_factor (the "
                "'average Nyquist frequency') is deprecated: the average "
                "Nyquist frequency is not meaningful for irregularly-sampled "
                "data. Pass explicit minimum_frequency and maximum_frequency "
                "bounds instead.", FutureWarning)
            Nf = int(0.5 * samples_per_peak * nyquist_factor * n_samples)

        return df * (nf0 + np.arange(Nf))

    def _run(self, freqs, fast, method=DEFAULT_METHOD):
        """Compute powers + best params over a frequency array, handling both a
        single template-set and a catalog (per-frequency max over sets).

        Returns ``(powers, params_list, winning_set)`` where ``winning_set`` is
        an int array of winning set indices for a catalog, else ``None``.
        """
        self._validate_mode()
        _validate_method(method)
        self._validate_templates()
        self._validate_data()

        if not self._is_catalog:
            powers, params = multiband_template_periodogram(
                self.t, self.y, self.bands, self._template_dict, freqs,
                dy=self.dy, mode=self.mode,
                relative_offsets=self.relative_offsets, fast=fast,
                method=method)
            return np.asarray(powers), params, None

        # Catalog: the per-band NFFT summations are template-independent, so
        # compute them ONCE and reuse them for every template-set (avoids an
        # N-fold redundant NFFT for an N-template catalog).
        nfreq = len(np.asarray(freqs))
        H = len(next(iter(self._template_sets[0].values())).c_n)
        per_band_sumlists, stats = compute_band_summations(
            self.t, self.y, self.bands, freqs, H, dy=self.dy, mode=self.mode,
            relative_offsets=self.relative_offsets, fast=fast)

        stack, param_sets = [], []
        for template_dict in self._template_sets:
            pw, pl = solve_over_frequencies(
                template_dict, per_band_sumlists, stats, nfreq,
                self.mode, self.relative_offsets, method=method)
            stack.append(np.asarray(pw))
            param_sets.append(pl)
        stack = np.array(stack)                       # (nsets, nfreq)
        winning_set = np.argmax(stack, axis=0)
        powers = stack[winning_set, np.arange(stack.shape[1])]
        params = [param_sets[winning_set[i]][i] for i in range(len(winning_set))]
        return powers, params, winning_set

    def _make_model(self, freq, params, set_index):
        templates = (self._template_sets[set_index] if self._is_catalog
                     else self._template_dict)
        model = MultibandTemplateModel(templates, frequency=float(freq),
                                       parameters=params)
        if self._is_catalog:
            model.template_set_index = int(set_index)
        return model

    def fit_model(self, freq):
        """Best-fit :class:`MultibandTemplateModel` at a single frequency."""
        powers, params, win = self._run([float(freq)], fast=False)
        return self._make_model(freq, params[0], None if win is None else win[0])

    def autopower(self, save_best_model=True, fast=True,
                  method=DEFAULT_METHOD, **kwargs):
        """Compute the multiband periodogram on the auto-determined grid.

        ``method`` selects the phase maximizer ('eigvals' root-finding
        reference, or the numerically-equivalent 'scan' scan+polish path).

        Returns ``(frequency, power)``.
        """
        frequency = self.autofrequency(**kwargs)
        powers, params, win = self._run(frequency, fast, method=method)

        if save_best_model:
            i = int(np.argmax(powers))
            self._save_best_model(self._make_model(
                frequency[i], params[i], None if win is None else win[i]))

        return frequency, powers

    def power(self, frequency, save_best_model=True, fast=False,
              method=DEFAULT_METHOD):
        """Compute multiband power at arbitrary (not necessarily gridded)
        frequencies. Output is reshaped to match the input. ``method``
        selects the phase maximizer (see :data:`METHODS`).
        """
        frequency = np.asarray(frequency, dtype=float)
        shape = frequency.shape
        frequency = np.atleast_1d(frequency.ravel())

        powers, params, win = self._run(frequency, fast, method=method)

        if save_best_model:
            i = int(np.argmax(powers))
            self._save_best_model(self._make_model(
                frequency[i], params[i], None if win is None else win[i]))

        return powers.reshape(shape)

    def _save_best_model(self, model, overwrite=False):
        if overwrite or self.best_model is None:
            self.best_model = model
            return
        # keep whichever model better fits the data
        y_fit = model(self.t, self.bands)
        y_best = self.best_model(self.t, self.bands)
        if _power_from_fit(self.y, self.dy, y_fit) > \
                _power_from_fit(self.y, self.dy, y_best):
            self.best_model = model


def _power_from_fit(y, dy, yfit):
    """Global unbiased-power of a fit, used only to break ties between saved
    models (mirrors ``modeler.power_from_fit``)."""
    n = len(y)
    err = np.ones(n) if dy is None else \
        np.broadcast_to(np.asarray(dy, dtype=float), (n,)).astype(float)
    w = weights(err)
    ybar = np.dot(w, y)
    chi2_0 = np.dot(w, (y - ybar) ** 2)
    chi2 = np.dot(w, (y - yfit) ** 2)
    return 1. - (chi2 / chi2_0)
