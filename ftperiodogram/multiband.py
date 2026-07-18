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
from .utils import weights, ModelFitParams, Summations
from .summations import (fast_summations, direct_summations,
                         direct_summations_single_freq,
                         _nfft_grid_coefficients, _batched_sums_from_nfft,
                         _direct_stacked_sums_chunk, _validate_chunk_size)
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
# circle angles -- the degree-8HK polynomial G is only formed at deep-|MM|
# dips, where the scan returns max(scan, root path) (C3.5 / MB-DIP-1:
# scan >= eigvals there, equal elsewhere). 'scan' is the default since
# WP C4; 'eigvals' is retained permanently as the reference/validation
# mode.
METHODS = ('eigvals', 'scan')
DEFAULT_METHOD = 'scan'


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
    The ``MM'`` variance correction is *assembled* in the centered form
    ``sum_k W_k (Mbar^(k) - sum_l W_l Mbar^(l))^2`` -- the definition of the
    between-band variance, algebraically equal to the uncentered form above
    when the ``W_k`` sum to 1 and insensitive (second order) to the actual
    ulp-level deviation of their float sum from 1 (C3 finding SESAR-DIP-1:
    the uncentered form loses ~12 digits to cancellation on the circle when
    the bands' mean-template polynomials are similar; C3.5 panel measured
    the centered assembly at 1.3e-21 abs error vs 4.6e-16 uncentered).
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
        YM_corr = Mbar_comb = None
        Mbar = {}
        for band in bands:
            AC_k = per_band_YM_MM[band][2]
            wk = W[band]
            Mbar[band] = _mean_template_poly(AC_k)

            term = wk * ((ybar[band] - ybar_global) * Mbar[band])
            YM_corr = term if YM_corr is None else YM_corr + term

            wk_mbar = wk * Mbar[band]
            Mbar_comb = wk_mbar if Mbar_comb is None else Mbar_comb + wk_mbar

        # The re-centering variance term is computed in CENTERED form,
        #     sum_k W_k (Mbar_k - Mbar_comb)^2
        # which equals sum_k W_k Mbar_k^2 - Mbar_comb^2 exactly (the W_k sum
        # to 1) but avoids its catastrophic cancellation: the uncentered
        # form carries O(0.1) coefficient mass that cancels to ~1e-13 on
        # the circle when the bands' mean-template polynomials are similar,
        # corrupting MM' by up to ~5e-4 relative at deep |MM| dips and the
        # peak power by ~1e-4 in BOTH directions (C3 finding SESAR-DIP-1,
        # VERIFICATION.md).
        MM_corr = None
        for band in bands:
            D_k = Mbar[band] - Mbar_comb
            sq = W[band] * (D_k * D_k)
            MM_corr = sq if MM_corr is None else MM_corr + sq

        YM = YM + YM_corr
        MM = MM + MM_corr

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
    # (length-gated: keep a genuine small leading coefficient when numpy
    # already removed the exactly-cancelled zero -- C3 finding FO-TRIM-1)
    G = pdg.trim_zero_leading_coef(G, nominal_degree=8 * H * len(bands) - 1)

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
    bracketed circular local maximum -- plus every deep dip of any band's
    ``|MM_k|``, refined to the ``|MM_k|^2`` minimum -- with Newton steps on
    ``dF/dtheta`` using the analytic per-band derivatives of
    :func:`ftperiodogram.core._scan_dP_d2P`, summed over bands with weights
    ``W_k``. At frequencies where any band's ``|MM_k|`` nearly vanishes on
    the circle (see :data:`ftperiodogram.core._SCAN_EXACT_RTOL`) the exact
    :func:`_shared_phase_fit` root path is ALSO run and the better of the
    two solutions is returned: narrow sub-grid spikes of ``F`` defeat any
    grid scan (the root path's guarantee), while in the same regime the
    root path's ``G`` polynomial can drown below its FP noise floor exactly
    where ``F`` peaks (the scan's guarantee; C3 finding MB-DIP-1). Both
    evaluate the true objective at genuine phases, so the max is a valid
    lower bound that never overshoots. Away from deep dips the
    degree-``8HK`` stationarity polynomial ``G`` is never formed, so the
    ``(HK)^3`` per-frequency root-finding cost is avoided.

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

    # garbage in must fail loudly (parity with the root path's LinAlgError)
    if not (np.all(np.isfinite(Yco)) and np.all(np.isfinite(Mco))):
        raise ValueError("non-finite per-band YM/MM polynomial coefficients "
                         "(NaN/inf in t, y, dy, or the weights?)")

    M_ang = max(pdg._SCAN_MIN_ANGLES, pdg._SCAN_ANGLES_PER_H * H * K)

    # grid scan of F(theta)
    Yg = pdg._eval_polys_on_circle(Yco, M_ang)               # (K, M)
    Mg = pdg._eval_polys_on_circle(Mco, M_ang)

    # Deep-dip regime: when ANY band's |MM_k| dips below _SCAN_EXACT_RTOL
    # of its circle max, neither maximizer dominates. F can carry a
    # sub-grid-width spike the grid scan + seeded polish has no bracketing
    # guarantee for (Bernstein: such a spike REQUIRES a deep dip) -- but in
    # the same regime the exact path's degree-(8HK-2) G polynomial can
    # drown below its own FP coefficient noise floor precisely where F
    # peaks, so G carries no root there while F itself stays smooth and
    # scannable (C3 finding MB-DIP-1: verbatim delegation cost up to
    # 8.4e-2 of power on rank-deficient phase-clustered fixtures). So run
    # BOTH maximizers and return the better one: each evaluates the true
    # objective at a genuine phase, so each is a valid lower bound and
    # their max never overshoots (C2 fix-evaluation critic; the C3
    # root-cause agent validated the scan machinery recovers the G-path
    # deficits to < 2e-14).
    absMg = np.abs(Mg)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_min = float(np.min(np.min(absMg, axis=1) / np.max(absMg, axis=1)))
    deep_dip = r_min < pdg._SCAN_EXACT_RTOL

    if deep_dip:
        # Escalate the scan density until the grid is guaranteed to bracket
        # the narrowest F spike a dip of this depth can host: a sub-grid
        # spike of a band's Re(YM_k^2/MM_k) term requires
        # min|MM_k| <= max|MM_k| ((2H) dtheta)^2 / 2 (the C2 Bernstein
        # bound), so dtheta <= sqrt(2 r_min)/(2H) suffices; capped at
        # core._SCAN_DEEP_MAX_ANGLES. Where the cap binds (see the
        # constant's comment) the bracketing bound is unmet and the
        # dip-Newton seeds plus the max(scan, root) merge below are
        # best-effort only. Validated on the C3 reproduction fixtures: the
        # default M misses genuine ~1e-3-cycle-wide peaks by up to 2e-2;
        # the escalated grid attains the raw-data oracle to <~1e-9.
        need = 2.0 * np.pi * 2.0 * H / np.sqrt(max(2.0 * r_min, 1e-30))
        M_deep = int(min(pdg._SCAN_DEEP_MAX_ANGLES,
                         max(M_ang, 1 << int(np.ceil(np.log2(need))))))
        if M_deep > M_ang:
            M_ang = M_deep
            Yg = pdg._eval_polys_on_circle(Yco, M_ang)
            Mg = pdg._eval_polys_on_circle(Mco, M_ang)
            absMg = np.abs(Mg)

    with np.errstate(divide='ignore', invalid='ignore'):
        Fg = np.dot(Wv, np.real(Yg * Yg / Mg))               # (M,)
    Fg[~np.isfinite(Fg)] = -np.inf

    kY = np.arange(Yco.shape[1])
    kM = np.arange(Mco.shape[1])
    half_window = 2 * np.pi / M_ang

    ismax = ((Fg >= np.roll(Fg, 1)) & (Fg >= np.roll(Fg, -1)) &
             (Fg > -np.inf))
    kcap = pdg._SCAN_MAX_CANDIDATES_PER_H * H * K
    cols = np.where(ismax)[0]
    if len(cols) > kcap:                  # degenerate plateau: trim
        cols = cols[np.argsort(Fg[cols])[-kcap:]]

    theta0 = (2 * np.pi / M_ang) * cols
    F0 = Fg[cols]

    # Dip candidates: a narrow spike of F arises when ANY band's |MM_k|
    # nearly vanishes on the circle (rank-deficient or phase-clustered
    # sampling in that band); the F grid scan cannot bracket sub-grid-width
    # spikes (same mechanism as the single-band scan, see
    # core._SCAN_DIP_RTOL). Refine each deep circular local minimum of
    # |MM_k| to the true |MM_k|^2 minimum by clamped Newton, then polish F
    # from there alongside the grid maxima.
    dip_seeds = []
    kcap_band = pdg._SCAN_MAX_CANDIDATES_PER_H * H
    for kb in range(K):
        rowM = np.abs(Mg[kb])
        ismin = ((rowM <= np.roll(rowM, 1)) & (rowM <= np.roll(rowM, -1)) &
                 (rowM < pdg._SCAN_DIP_RTOL * rowM.max()))
        dcols = np.where(ismin)[0]
        if len(dcols) == 0:
            continue
        if len(dcols) > kcap_band:
            dcols = dcols[np.argsort(rowM[dcols])[:kcap_band]]
        dth0 = (2 * np.pi / M_ang) * dcols
        th = dth0.copy()
        dM1 = Mco[kb] * kM
        dM2 = Mco[kb] * kM * kM
        for _ in range(pdg._SCAN_NEWTON_STEPS):
            ph = np.exp(1j * th)
            Mm = pdg._horner_eval(Mco[kb], ph)
            M1 = pdg._horner_eval(dM1, ph)
            M2 = pdg._horner_eval(dM2, ph)
            with np.errstate(divide='ignore', invalid='ignore'):
                q1 = 2.0 * np.real(1j * M1 * np.conj(Mm))
                q2 = 2.0 * (np.abs(M1) ** 2 - np.real(np.conj(Mm) * M2))
                step = q1 / q2
            step[~np.isfinite(step)] = 0.0
            th_new = np.clip(th - step, dth0 - half_window, dth0 + half_window)
            converged = np.all(np.abs(th_new - th) <= pdg._SCAN_NEWTON_XTOL)
            th = th_new
            if converged:
                break
        dip_seeds.append(th)

    def _F_derivs(phi):
        """(dF/dtheta, d2F/dtheta2) at unit-circle points ``phi``; the
        per-band terms come from :func:`ftperiodogram.core._scan_dP_d2P`
        (the single source of truth for the scan derivatives), summed with
        weights ``W_k``."""
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
                dPk, d2Pk = pdg._scan_dP_d2P(Y, Y1, Y2, Mm, M1, M2)
                dF = dF + Wv[kb] * dPk
                d2F = d2F + Wv[kb] * d2Pk
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

    if dip_seeds:
        dip_theta = np.concatenate(dip_seeds)
        F_dip = _F_at(np.exp(1j * dip_theta))
        F_dip[~np.isfinite(F_dip)] = -np.inf
        theta0 = np.concatenate([theta0, dip_theta])
        F0 = np.concatenate([F0, F_dip])

    if len(theta0) == 0 or not np.any(F0 > -np.inf):
        # no scan candidate with finite total power anywhere: degenerate
        # for the scan; in the deep-dip regime let the exact path decide
        # (it applies its own degenerate guards)
        if deep_dip:
            return _shared_phase_fit(template_dict, per_band_sums, stats)
        return _flat_model(stats, 'shared_phase')

    theta = theta0.copy()
    for _ in range(pdg._SCAN_NEWTON_STEPS):
        dF, d2F = _F_derivs(np.exp(1j * theta))
        with np.errstate(divide='ignore', invalid='ignore'):
            step = dF / d2F
        step[~np.isfinite(step)] = 0.0
        theta_new = np.clip(theta - step,
                            theta0 - half_window, theta0 + half_window)
        converged = np.all(np.abs(theta_new - theta) <= pdg._SCAN_NEWTON_XTOL)
        theta = theta_new
        if converged:
            break

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

    if deep_dip:
        # max(scan, exact root path): strictly safe (both are valid lower
        # bounds of the true max of F); ties go to the exact path, which
        # keeps the historical behavior wherever the root path is already
        # optimal (see the deep-dip comment above / MB-DIP-1)
        root_params, root_power = _shared_phase_fit(template_dict,
                                                    per_band_sums, stats)
        if root_power >= power:
            return root_params, float(root_power)

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


# ----------------------------------------------------------------------
# Batched (vectorized-over-frequency) solve. The per-frequency
# solve_over_frequencies re-implements a Python loop over frequencies, one
# tiny FFT / Horner / numpy.polynomial object per frequency; the batched path
# below assembles the per-band YM/MM/AC coefficients for whole frequency
# chunks (core.batched_YM_MM_from_sums), combines the bands with the SAME
# linear W_k-weighting (which commutes with batching exactly), and maximizes
# each chunk at once (core.scan_polish_from_coefs). It is a drop-in for the
# scan-method shared-amplitude modes (floating_offsets / sesar / independent)
# and returns identical powers to solve_over_frequencies (to float64 round-off;
# ~machine epsilon where the combined MM' is well conditioned). Since the
# recovery/detection use returns only the power spectrum (best model saved at
# a single frequency), the per-frequency MultibandModelFitParams objects are
# not materialized here -- reconstruction at the peak reuses the per-frequency
# path on one frequency.
# ----------------------------------------------------------------------
# Modes whose scan solve is a single (band-combined) YM/MM scan and so batch
# through core.scan_polish_from_coefs verbatim. Restricted to the Paper-2
# target (floating_offsets): its combined MM' = sum_k W_k MM_k is well
# conditioned on the unit circle (few deep dips), so the batched np.trace
# assembly matches the per-frequency reference to ~machine epsilon, and the
# rare deep-dip frequencies are DEFERRED to the per-frequency reference below.
# `sesar`/`independent`/`shared_phase` keep the (correct) per-frequency scan:
# sesar's between-band-variance cancellation amplifies assembly-order ulps
# catastrophically at deep dips (SESAR-DIP-1), and independent/shared_phase
# need per-band deep-dip handling.
_BATCHED_SCAN_MODES = ('floating_offsets',)

# Deferral threshold for the batched floating_offsets path: at frequencies whose
# combined |MM'| conditioning (min/max on the unit circle) falls below this, the
# power is recomputed by the per-frequency reference so it bit-matches
# solve_over_frequencies. floating_offsets' combined MM' = sum_k W_k MM_k has NO
# between-band cancellation (unlike sesar; SESAR-DIP-1), so batched-vs-reference
# divergence stays <= ~1e-12 even on adversarial phase-clumped data and is
# ~machine-epsilon on well-conditioned griz -- measured max 9.6e-13 at
# conditioning ~3e-5, 2.8e-16 on realistic K=4 griz. The 1e-4 threshold fires on
# ~0% of well-conditioned griz frequencies (so the target runs fully batched)
# while still deferring the deepest dips for a guaranteed reference match. Much
# deeper than core._SCAN_EXACT_RTOL (0.15, the scan's own exact-root fallback),
# because that fallback tolerates the batched-coefficient assembly whereas this
# guarantees the PER-FREQUENCY assembly at extreme conditioning. Set to 3e-3:
# batched-vs-reference divergence reaches ~1e-12 only near conditioning ~8e-4
# (measured on adversarial K=2 phase-clumped H=8 fixtures), so 3e-3 leaves a
# ~10x margin below the 1e-12 gate, while realistic griz (K>=3) has ~0% of
# frequencies this ill-conditioned so the target never pays the deferral.
_BATCHED_DEFER_RTOL = 3E-3


def _prepare_band_transforms(t, y, bands, dy, freqs, H, mode, relative_offsets,
                             fast=True, sigma=2, tol=1E-7):
    """Per-band, frequency-independent transforms for the batched solve.

    Returns ``(transforms, stats, freqs)`` where ``transforms`` maps each band
    to a ``(kind, payload)`` pair: ``('nfft', (f_hat_u, f_hat_w, dnf))`` for the
    adjoint-NFFT path or ``('direct', (t_k, y_k, w_k))`` for the exact direct
    path. Template-independent -- reusable across a whole template catalog.
    """
    band_data, stats = _prepare_bands(t, y, bands, dy, mode,
                                      relative_offsets, H)
    freqs = np.asarray(freqs, dtype=float)
    transforms = OrderedDict()
    for band, (t_k, y_k, w_k) in band_data.items():
        if fast:
            f_hat_u, f_hat_w, _nf, dnf = _nfft_grid_coefficients(
                t_k, y_k, w_k, freqs, H, sigma=sigma, tol=tol)
            transforms[band] = ('nfft', (f_hat_u, f_hat_w, dnf))
        else:
            transforms[band] = ('direct', (t_k, y_k, w_k))
    return transforms, stats, freqs


def _band_stacked_chunk(transform, H, i0, i1, freqs):
    """Stacked ``Summations`` for one band over grid frequencies ``[i0, i1)``."""
    kind, payload = transform
    if kind == 'nfft':
        f_hat_u, f_hat_w, dnf = payload
        return _batched_sums_from_nfft(f_hat_u, f_hat_w, H, dnf, i0, i1)
    t_k, y_k, w_k = payload
    ybar_k = np.dot(w_k, y_k)
    u_k = w_k * (y_k - ybar_k)
    return _direct_stacked_sums_chunk(t_k, u_k, w_k, freqs[i0:i1], H)


def _band_single_ref_sums(transform, H, gi, freqs):
    """Per-frequency reference ``Summations`` at grid index ``gi``, computed the
    same way the non-batched path would (so a deep-dip frequency deferred here
    bit-matches :func:`solve_over_frequencies`): NFFT-extracted for ``'nfft'``
    (identical to :func:`fast_summations` per frequency), or the centered
    two-pass :func:`direct_summations_single_freq` for ``'direct'``."""
    kind, payload = transform
    if kind == 'nfft':
        f_hat_u, f_hat_w, dnf = payload
        stacked = _batched_sums_from_nfft(f_hat_u, f_hat_w, H, dnf, gi, gi + 1)
        return Summations(*(getattr(stacked, f)[0] for f in stacked._fields))
    t_k, y_k, w_k = payload
    return direct_summations_single_freq(t_k, y_k, w_k, float(freqs[gi]), H)


def _combine_stacked_shared_amp(template_dict, transforms, stats, i0, i1, freqs,
                                mode):
    r"""Band-combined stacked ``YM'``/``MM'``/``AC'`` for a frequency chunk.

    Vectorized analog of :func:`combine_band_summations`: for
    ``floating_offsets`` the combination is the ``W_k``-weighted average of the
    per-band coefficients; for ``sesar`` the global-mean re-centering
    corrections are added (assembled in the same centered ``MM'`` form as the
    per-frequency path, so the SESAR-DIP-1 cancellation fix carries over).
    Also returns the per-band ``AC_k`` stack (for offset reconstruction).
    """
    bands = stats.bands
    H = stats.H
    YM = MM = AC = None
    AC_by_band = {}
    Mbar_by_band = {}
    Mbar_comb = None
    for band in bands:
        stacked = _band_stacked_chunk(transforms[band], H, i0, i1, freqs)
        YM_k, MM_k, AC_k = pdg.batched_YM_MM_from_sums(
            template_dict[band].c_n, template_dict[band].s_n, stacked)
        AC_by_band[band] = AC_k
        wk = stats.W[band]
        YM = wk * YM_k if YM is None else YM + wk * YM_k
        MM = wk * MM_k if MM is None else MM + wk * MM_k
        AC_wk = wk * AC_k
        AC = AC_wk if AC is None else AC + AC_wk

    if mode == 'sesar':
        # global-mean re-centering (thesis lines 76-78), batched. Mbar_k is the
        # (phi**H-scaled) mean-template polynomial of band k: coefs
        # [conj(AC_k)[::-1], 0, AC_k] -> shape (m, 2H+1).
        nfc = i1 - i0
        YM_corr = None
        for band in bands:
            AC_k = AC_by_band[band]
            Mbar_k = np.concatenate(
                (np.conj(AC_k)[:, ::-1],
                 np.zeros((nfc, 1), dtype=np.complex128), AC_k), axis=1)
            Mbar_by_band[band] = Mbar_k
            wk = stats.W[band]
            term = wk * ((stats.ybar[band] - stats.ybar_global) * Mbar_k)
            YM_corr = term if YM_corr is None else YM_corr + term
            wk_mbar = wk * Mbar_k
            Mbar_comb = wk_mbar if Mbar_comb is None else Mbar_comb + wk_mbar
        # MM' correction in CENTERED form: sum_k W_k (Mbar_k - Mbar_comb)^2,
        # each square a per-row polynomial self-convolution (degree 4H).
        MM_corr = None
        for band in bands:
            D_k = Mbar_by_band[band] - Mbar_comb            # (m, 2H+1)
            sq = _batched_polysquare(D_k)                   # (m, 4H+1)
            wk_sq = stats.W[band] * sq
            MM_corr = wk_sq if MM_corr is None else MM_corr + wk_sq
        YM = YM + YM_corr
        MM = MM + MM_corr

    return YM, MM, AC, AC_by_band


def _batched_polysquare(coefs):
    """Row-wise polynomial square: for coefs ``(m, n)`` return ``(m, 2n-1)``
    with each row the self-convolution of that row (``np.convolve`` per row,
    vectorized as a shifted outer-product accumulation)."""
    m, n = coefs.shape
    out = np.zeros((m, 2 * n - 1), dtype=coefs.dtype)
    for s in range(n):
        out[:, s:s + n] += coefs[:, s:s + 1] * coefs
    return out


def _batched_mbar_at(AC_k, phi):
    """``2 Re(sum_n AC_k[:, n] phi^n)`` per frequency (offset reconstruction),
    vectorized: ``AC_k`` is ``(nf, H)`` and ``phi`` is ``(nf,)``."""
    acc = np.zeros(phi.shape, dtype=np.complex128)
    for j in range(AC_k.shape[1] - 1, -1, -1):
        acc = (acc + AC_k[:, j]) * phi
    return 2.0 * np.real(acc)


def multiband_power_spectrum_batched(template_dict, transforms, stats, freqs,
                                     mode=DEFAULT_MODE, relative_offsets=None,
                                     chunk_size=4096):
    """Batched power spectrum for ``mode='floating_offsets'`` (the Paper-2
    target), vectorized over frequency chunks.

    Pairs with :func:`_prepare_band_transforms`. Returns just the power array
    (the recovery/detection workload); the best-fit model at any single
    frequency is reconstructed on demand by the per-frequency path.

    Deep-|MM'|-dip frequencies -- where assembly-order round-off is amplified by
    the near-singular phase optimization -- are DEFERRED to the per-frequency
    reference (:func:`multiband_template_fit_from_sums`), so the batched powers
    bit-match :func:`solve_over_frequencies` at exactly the frequencies where
    the batched ``np.trace`` assembly would otherwise diverge from it. On
    well-conditioned data (the griz target) the deferral fires on a few percent
    of frequencies; the rest run fully batched.
    """
    if mode not in _BATCHED_SCAN_MODES:
        raise ValueError("multiband_power_spectrum_batched supports mode in "
                         "{0}; got {1!r}".format(_BATCHED_SCAN_MODES, mode))
    _validate_chunk_size(chunk_size)
    H = stats.H
    bands = stats.bands
    nfreq = len(freqs)
    powers = np.zeros(nfreq)

    YY = stats.YY_combined
    if _is_flat(YY, stats.ybar_global):
        return powers

    n_ang = max(pdg._SCAN_MIN_ANGLES, pdg._SCAN_ANGLES_PER_H * H)
    for i0 in range(0, nfreq, chunk_size):
        i1 = min(i0 + chunk_size, nfreq)
        YM, MM, AC, _ACb = _combine_stacked_shared_amp(
            template_dict, transforms, stats, i0, i1, freqs, mode)
        _pl, pw, _bp = pdg.scan_polish_from_coefs(
            YM, MM, AC, H, stats.ybar_global, YY, positive_amplitude=True)
        powers[i0:i1] = pw

        # defer extreme-|MM'|-dip rows to the per-frequency reference
        absM = np.abs(pdg._eval_polys_on_circle(MM, n_ang))
        deep = np.where(np.min(absM, axis=1)
                        < _BATCHED_DEFER_RTOL * np.max(absM, axis=1))[0]
        for r in deep:
            gi = i0 + int(r)
            ref_sums = {b: _band_single_ref_sums(transforms[b], H, gi, freqs)
                        for b in bands}
            _mp, pr = multiband_template_fit_from_sums(
                template_dict, ref_sums, stats, mode, relative_offsets,
                method='scan')
            powers[gi] = pr
    return powers


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

    def _run(self, freqs, fast, method=DEFAULT_METHOD, powers_only=False):
        """Compute powers + best params over a frequency array, handling both a
        single template-set and a catalog (per-frequency max over sets).

        Returns ``(powers, params_list, winning_set)`` where ``winning_set`` is
        an int array of winning set indices for a catalog, else ``None``. When
        ``powers_only`` and the (mode, method) support it, the fast batched path
        is used and ``params_list`` is returned as ``None`` (the caller
        reconstructs the best-fit model at the single peak frequency on demand).
        """
        self._validate_mode()
        _validate_method(method)
        self._validate_templates()
        self._validate_data()

        if powers_only and method == 'scan' and self.mode in _BATCHED_SCAN_MODES:
            return self._run_batched(np.asarray(freqs, dtype=float), fast)

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

    def _run_batched(self, freqs, fast):
        """Fast batched power spectrum (scan method, shared-amplitude modes).

        Returns ``(powers, None, winning_set)``; per-frequency params are not
        materialized (see :func:`multiband_power_spectrum_batched`). For a
        catalog the template-independent per-band transforms are computed once
        and reused across every template-set.
        """
        if not self._is_catalog:
            H = len(next(iter(self._template_dict.values())).c_n)
            transforms, stats, freqs = _prepare_band_transforms(
                self.t, self.y, self.bands, self.dy, freqs, H, self.mode,
                self.relative_offsets, fast=fast)
            powers = multiband_power_spectrum_batched(
                self._template_dict, transforms, stats, freqs, mode=self.mode,
                relative_offsets=self.relative_offsets)
            return powers, None, None

        nfreq = len(freqs)
        H = len(next(iter(self._template_sets[0].values())).c_n)
        transforms, stats, freqs = _prepare_band_transforms(
            self.t, self.y, self.bands, self.dy, freqs, H, self.mode,
            self.relative_offsets, fast=fast)
        stack = [multiband_power_spectrum_batched(
                    td, transforms, stats, freqs, mode=self.mode,
                    relative_offsets=self.relative_offsets)
                 for td in self._template_sets]
        stack = np.array(stack)                       # (nsets, nfreq)
        winning_set = np.argmax(stack, axis=0)
        powers = stack[winning_set, np.arange(nfreq)]
        return powers, None, winning_set

    def _best_model_from_powers(self, freq, set_index):
        """Reconstruct the best-fit :class:`MultibandTemplateModel` at a single
        peak frequency via the per-frequency solve (used by the batched path,
        which does not keep per-frequency params). ``set_index`` is ``None`` for
        a single template-set."""
        template_dict = (self._template_sets[set_index] if self._is_catalog
                         else self._template_dict)
        H = len(next(iter(template_dict.values())).c_n)
        per_band_sumlists, stats = compute_band_summations(
            self.t, self.y, self.bands, np.atleast_1d(float(freq)), H,
            dy=self.dy, mode=self.mode,
            relative_offsets=self.relative_offsets, fast=False)
        _pw, pl = solve_over_frequencies(
            template_dict, per_band_sumlists, stats, 1, self.mode,
            self.relative_offsets, method='scan')
        return self._make_model(freq, pl[0], set_index)

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
        powers, params, win = self._run(frequency, fast, method=method,
                                        powers_only=True)

        if save_best_model:
            i = int(np.argmax(powers))
            set_index = None if win is None else int(win[i])
            if params is None:
                self._save_best_model(
                    self._best_model_from_powers(frequency[i], set_index))
            else:
                self._save_best_model(self._make_model(
                    frequency[i], params[i], set_index))

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

        powers, params, win = self._run(frequency, fast, method=method,
                                        powers_only=True)

        if save_best_model:
            i = int(np.argmax(powers))
            set_index = None if win is None else int(win[i])
            if params is None:
                self._save_best_model(
                    self._best_model_from_powers(frequency[i], set_index))
            else:
                self._save_best_model(self._make_model(
                    frequency[i], params[i], set_index))

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
