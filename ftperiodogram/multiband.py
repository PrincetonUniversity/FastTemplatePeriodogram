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
        """Evaluate y_hat for flat arrays ``(t, bands)``, dispatching per band."""
        t = np.asarray(t, dtype=float)
        bands = np.asarray(bands)
        out = np.empty(t.shape, dtype=float)
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
def multiband_template_fit_from_sums(template_dict, per_band_sums, stats, mode,
                                     relative_offsets=None):
    """Solve the multiband template fit at one frequency from precomputed sums.

    Returns ``(MultibandModelFitParams, power)``. Always returns the
    ``theta_1 >= 0`` solution (paper item K.18).
    """
    H = stats.H
    bands = stats.bands

    if mode in ('floating_offsets', 'sesar'):
        per_band_YM_MM = {band: pdg.YM_MM_from_sums(template_dict[band].c_n,
                                                    template_dict[band].s_n,
                                                    per_band_sums[band])
                          for band in bands}

        YM, MM, AC = combine_band_summations(per_band_YM_MM, stats.W, stats.ybar,
                                             stats.ybar_global, mode)

        YY = stats.YY_global if mode == 'sesar' else stats.YY_combined
        shared, power, best_phi = pdg.roots_from_YM_MM(
            YM, MM, AC, H, stats.ybar_global, YY, positive_amplitude=True)

        params_by_band = {}
        for band in bands:
            AC_k = per_band_YM_MM[band][2]
            if mode == 'sesar':
                # single shared offset, plus the fixed relative offset lambda^(k)
                c_k = shared.c + relative_offsets[band]
            else:
                # floating offset: theta_3^(k) = ybar_k - theta_1 * Mbar^(k)(phi)
                mbar_k = 2 * np.real(pol.Polynomial(
                    np.concatenate(([0], AC_k)))(best_phi))
                c_k = stats.ybar[band] - shared.a * mbar_k
            params_by_band[band] = ModelFitParams(a=shared.a, b=shared.b,
                                                  c=c_k, sgn=shared.sgn)

        return MultibandModelFitParams(params_by_band, mode), float(power)

    if mode == 'independent':
        params_by_band = {}
        power = 0.0
        for band in bands:
            YM_k, MM_k, AC_k = pdg.YM_MM_from_sums(template_dict[band].c_n,
                                                  template_dict[band].s_n,
                                                  per_band_sums[band])
            params_k, power_k, _ = pdg.roots_from_YM_MM(
                YM_k, MM_k, AC_k, H, stats.ybar[band], stats.YY_per_band[band],
                positive_amplitude=True)
            params_by_band[band] = params_k
            power += stats.W[band] * power_k
        return MultibandModelFitParams(params_by_band, mode), float(power)

    if mode == 'shared_phase':
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

    of degree ``8 H K - 1`` -- a genuinely larger root problem than the
    single-band case (cost grows as ``(H K)^3`` per frequency).

    POSITIVITY: unlike the shared-amplitude modes, the phase here is shared, so
    the per-band amplitudes cannot each be independently forced positive; this
    routine returns the power-maximizing shared phase and reports the per-band
    amplitudes as fitted (they are >= 0 for a genuine in-phase multiband signal,
    but can be negative for a band that anti-correlates at the shared phase).
    """
    H = stats.H
    bands = stats.bands

    per_band = {band: pdg.YM_MM_from_sums(template_dict[band].c_n,
                                          template_dict[band].s_n,
                                          per_band_sums[band])
                for band in bands}
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

    roots = G.roots()
    roots = roots[np.absolute(roots) > 0]
    roots /= np.absolute(roots)

    # total power at each candidate shared phase
    F = np.zeros(len(roots))
    for k in bands:
        P_YM, P_MM, _ = per_band[k]
        F += stats.W[k] * np.real(P_YM(roots) ** 2 / P_MM(roots))

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
        mbar_k = 2 * np.real(pol.Polynomial(np.concatenate(([0], AC_k)))(best_phi))
        c_k = stats.ybar[k] - theta_1_k * mbar_k
        params_by_band[k] = ModelFitParams(a=theta_1_k, b=b, c=c_k, sgn=sgn)

    return MultibandModelFitParams(params_by_band, 'shared_phase'), float(power)


def multiband_template_periodogram(t, y, bands, template_dict, freqs, dy=None,
                                   mode=DEFAULT_MODE, relative_offsets=None,
                                   fast=True):
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
    band_data, stats = _prepare_bands(t, y, bands, dy, mode,
                                      relative_offsets, H)
    freqs = np.asarray(freqs, dtype=float)

    per_band_sumlists = {}
    for band, (t_k, y_k, w_k) in band_data.items():
        if fast:
            per_band_sumlists[band] = fast_summations(t_k, y_k, w_k, freqs, H)
        else:
            per_band_sumlists[band] = direct_summations(t_k, y_k, w_k, freqs, H)

    powers = np.empty(len(freqs), dtype=float)
    params_list = []
    for i in range(len(freqs)):
        per_band_sums = {band: per_band_sumlists[band][i]
                         for band in band_data}
        mb_params, power = multiband_template_fit_from_sums(
            template_dict, per_band_sums, stats, mode, relative_offsets)
        powers[i] = power
        params_list.append(mb_params)

    return powers, params_list


# ----------------------------------------------------------------------
# Public class (drop-in sibling of FastTemplatePeriodogram).
# ----------------------------------------------------------------------
class FastMultibandTemplatePeriodogram(object):
    """Multiband template periodogram.

    Fits a single (possibly per-band) template shape to multiband data under one
    of the sharing modes in :data:`MODES`.

    Parameters
    ----------
    templates : Template or dict {band: Template} or list of Template
        A shared template shape, explicit per-band shapes, or a list aligned to
        ``sorted(unique(bands))``. All band templates must share the harmonic
        count ``H``.
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
        self._template_dict = None

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
        self._template_dict = build_template_set(self.templates, self.bands_)

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
        return self

    def autofrequency(self, nyquist_factor=5, samples_per_peak=5,
                      minimum_frequency=None, maximum_frequency=None):
        """Determine a trial-frequency grid over the global observation baseline.

        For irregularly-sampled data the average-Nyquist heuristic is not
        meaningful; the recommended path is to set ``minimum_frequency`` and
        ``maximum_frequency`` explicitly. ``nyquist_factor`` is retained only for
        API parity with the single-band class.
        """
        baseline = self.t.max() - self.t.min()
        n_samples = self.t.size

        df = 1. / (baseline * samples_per_peak)

        if minimum_frequency is not None:
            nf0 = min([1, np.floor(minimum_frequency / df)])
        else:
            nf0 = 1

        if maximum_frequency is not None:
            Nf = int(np.ceil(maximum_frequency / df - nf0))
        else:
            Nf = int(0.5 * samples_per_peak * nyquist_factor * n_samples)

        return df * (nf0 + np.arange(Nf))

    def fit_model(self, freq):
        """Best-fit :class:`MultibandTemplateModel` at a single frequency."""
        self._validate_mode()
        self._validate_templates()
        self._validate_data()
        _, params_list = multiband_template_periodogram(
            self.t, self.y, self.bands, self._template_dict, [float(freq)],
            dy=self.dy, mode=self.mode, relative_offsets=self.relative_offsets,
            fast=False)
        return MultibandTemplateModel(self._template_dict, frequency=float(freq),
                                      parameters=params_list[0])

    def autopower(self, save_best_model=True, fast=True, **kwargs):
        """Compute the multiband periodogram on the auto-determined grid.

        Returns ``(frequency, power)``.
        """
        self._validate_mode()
        self._validate_templates()
        self._validate_data()

        frequency = self.autofrequency(**kwargs)
        powers, params_list = multiband_template_periodogram(
            self.t, self.y, self.bands, self._template_dict, frequency,
            dy=self.dy, mode=self.mode, relative_offsets=self.relative_offsets,
            fast=fast)

        if save_best_model:
            i = int(np.argmax(powers))
            self._save_best_model(MultibandTemplateModel(
                self._template_dict, frequency=frequency[i],
                parameters=params_list[i]))

        return frequency, powers

    def power(self, frequency, save_best_model=True, fast=False):
        """Compute multiband power at arbitrary (not necessarily gridded)
        frequencies. Output is reshaped to match the input.
        """
        self._validate_mode()
        self._validate_templates()
        self._validate_data()

        frequency = np.asarray(frequency, dtype=float)
        shape = frequency.shape
        frequency = np.atleast_1d(frequency.ravel())

        powers, params_list = multiband_template_periodogram(
            self.t, self.y, self.bands, self._template_dict, frequency,
            dy=self.dy, mode=self.mode, relative_offsets=self.relative_offsets,
            fast=fast)

        if save_best_model:
            i = int(np.argmax(powers))
            self._save_best_model(MultibandTemplateModel(
                self._template_dict, frequency=frequency[i],
                parameters=params_list[i]))

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
