"""Comparison-method period estimators (Phase 3.2 bake-off).

The recovery harness (:mod:`ftperiodogram.validation`) scores a *pluggable period
estimator* over its frozen synthetic population::

    estimator(t, y, bands, dy, freqs) -> P_rec

so every method -- the FTP catalog periodogram and every comparison baseline --
is scored on the **identical** frozen sources and grid (the fairest possible
comparison).  This module supplies the estimator zoo spanning the design memo's
"rigidity spectrum" (``phase3_vocab_design.md`` S2), implemented fresh on the
existing numpy/scipy stack -- **no gatspy / astroML / astropy**:

==================  ===========================================  =================
estimator           shape degrees of freedom                     role
==================  ===========================================  =================
``GLSEstimator``    H=1 sinusoid, shared + per-band offset       sinusoidal LOWER bound
``FTPEstimator``    fixed K-template bank (3 dof / template)      the method
``MHLSEstimator``   free shared Fourier order H (2H coeffs)       free-shape reference (H capped vs n_obs)
``MultibandLS``     free per-band Fourier order H, shared period  fair sparse-multiband LS
``SesarOracle``     same templates, slow non-linear fit          GOLD standard (<1e-6 oracle)
==================  ===========================================  =================

All estimators are **picklable plain objects** (their state is templates / ints /
strings) so the harness can fan the per-source loop across a ``spawn`` process
pool.  Each returns the recovered period ``P_rec = 1 / freqs[argmax(power)]``;
``power_spectrum`` is also exposed for overlay/diagnostics.

Power convention
----------------
Every linear estimator reports the standard periodogram power

    power(f) = 1 - chi2(f) / chi2_0 ,

where ``chi2(f)`` is the weighted residual sum of squares of the per-frequency
weighted least-squares fit and ``chi2_0`` is the weighted sum of squares about the
**per-band weighted means** (the floating-offset baseline).  Weights are the
inverse-variance ``w_i = (1/sigma_i^2) / sum_j (1/sigma_j^2)`` of
:func:`ftperiodogram.utils.weights`.  Because every multiband model carries a
per-band offset column, ``chi2_0`` is exactly the FTP ``floating_offsets``
normalization (``YY_combined``), so ``GLSEstimator`` (H=1) reproduces FTP@H=1 to
machine precision -- the built-in cross-check in ``tests/test_baselines.py``.

The recovered period is invariant to ``chi2_0`` (it only rescales every power by a
constant), so a method's ``P_rec`` depends only on which design matrix maximizes
explained variance at each frequency.
"""
import numpy as np

from .template import Template
from .utils import weights
from .multiband import FastMultibandTemplatePeriodogram


# ----------------------------------------------------------------------
# Shared weighted-least-squares machinery
# ----------------------------------------------------------------------
def _errs(dy, n):
    """Broadcast ``dy`` (None / scalar / array) to a length-``n`` sigma array."""
    if dy is None:
        return np.ones(n, dtype=float)
    return np.broadcast_to(np.asarray(dy, dtype=float), (n,)).astype(float)


def _band_index(bands, n):
    """Return ``(labels, codes)``: unique band labels and a per-obs integer code.

    ``bands is None`` collapses to a single synthetic band (one offset column).
    """
    if bands is None:
        return [None], np.zeros(n, dtype=int)
    bands = np.asarray(bands)
    labels = list(np.unique(bands))
    codes = np.searchsorted(np.asarray(labels), bands)
    return labels, codes


def _offset_columns(codes, n_bands):
    """``(n, n_bands)`` per-band DC-offset indicator design block."""
    cols = np.zeros((len(codes), n_bands), dtype=float)
    cols[np.arange(len(codes)), codes] = 1.0
    return cols


def _weighted_chi2(X, y, w):
    """Minimum weighted residual sum of squares of ``y ~ X beta``.

    Solves the whitened system ``sqrt(w)[:,None] * X  beta = sqrt(w) * y`` with
    :func:`numpy.linalg.lstsq` (rank-revealing, so a rank-deficient design at a
    degenerate frequency yields the minimum-norm fit rather than blowing up) and
    returns ``sum_i w_i (y_i - X_i . beta)^2``.  ``w`` need not be normalized.
    """
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw.dot(beta)
    return float(resid.dot(resid))


def _chi2_0(y, w, codes, n_bands):
    """Weighted SS of ``y`` about the per-band weighted means (the FTP
    ``floating_offsets`` / ``YY_combined`` normalization)."""
    chi2_0 = 0.0
    for b in range(n_bands):
        m = (codes == b)
        if not np.any(m):
            continue
        wb, yb = w[m], y[m]
        ybar = wb.dot(yb) / wb.sum()
        chi2_0 += float((wb * (yb - ybar) ** 2).sum())
    return chi2_0


# ----------------------------------------------------------------------
# Linear least-squares periodogram estimators
# ----------------------------------------------------------------------
class _LinearLSEstimator(object):
    """Base for the linear-LS periodogram baselines.

    Subclasses build a per-frequency design matrix via :meth:`_design`; the base
    class handles weighting, the per-band-mean normalization, the power spectrum,
    and the ``P_rec = 1 / freqs[argmax]`` reduction common to all of them.
    """

    def _design(self, t, twopift, codes, n_bands):
        """Return the ``(n, p)`` design matrix at one frequency.

        ``twopift = 2*pi*f*t`` is precomputed (the only frequency-dependent input);
        ``codes`` / ``n_bands`` describe the per-band structure for offset columns.
        """
        raise NotImplementedError

    def power_spectrum(self, t, y, bands, dy, freqs):
        """Periodogram power over ``freqs`` (each in ``[0, 1]``)."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        n = t.size
        w = weights(_errs(dy, n))
        labels, codes = _band_index(bands, n)
        n_bands = len(labels)
        chi2_0 = _chi2_0(y, w, codes, n_bands)
        if chi2_0 <= 0:
            return np.zeros(freqs.size, dtype=float)

        powers = np.empty(freqs.size, dtype=float)
        two_pi_t = 2.0 * np.pi * t
        for i, f in enumerate(freqs):
            X = self._design(t, two_pi_t * f, codes, n_bands)
            powers[i] = 1.0 - _weighted_chi2(X, y, w) / chi2_0
        return powers

    def __call__(self, t, y, bands, dy, freqs):
        freqs = np.asarray(freqs, dtype=float)
        powers = self.power_spectrum(t, y, bands, dy, freqs)
        return 1.0 / freqs[int(np.argmax(powers))]


class GLSEstimator(_LinearLSEstimator):
    """Generalized Lomb-Scargle (Zechmeister & Kuerster 2009): the sinusoidal
    lower bound.

    One shared first-harmonic ``(cos, sin)`` pair across all bands plus a per-band
    floating offset -- the multiband ``floating_offsets`` model at ``H=1``.  Equal
    to ``MHLSEstimator(n_harmonics=1)`` and, by construction, to FTP with a single
    cosine template (``Template([1.0], [0.0])``) to machine precision.
    """

    def _design(self, t, twopift, codes, n_bands):
        return np.column_stack([_offset_columns(codes, n_bands),
                                np.cos(twopift), np.sin(twopift)])


class MHLSEstimator(_LinearLSEstimator):
    """Multiharmonic Lomb-Scargle (Schwarzenberg-Czerny 1996): the free-shape
    reference.

    A free shared Fourier series of order ``n_harmonics`` (``2H`` coefficients,
    one shape across bands) plus a per-band floating offset.  The requested
    order is capped per light curve so ``n_bands + 2H <= n_obs/2`` (at 4
    epochs/band the cap makes MHLS identical to GLS): below the cap the design
    is *rank-deficient* -- with ``p >= n_obs`` columns it interpolates every
    point exactly, power is identically 1, and the argmax is meaningless.
    That is degeneracy, not "overfitting"; genuine overfitting (chasing noise
    with positive residual dof) is what the capped estimator exhibits at
    moderate N (~12+ epochs/band), and is the failure mode FTP's learned shape
    prior should beat at low ``N_epochs``.
    """

    def __init__(self, n_harmonics=8):
        if int(n_harmonics) < 1:
            raise ValueError("n_harmonics must be >= 1")
        self.n_harmonics = int(n_harmonics)

    def _capped_order(self, n_obs, n_bands):
        """Largest order with ``n_bands + 2H <= n_obs/2``, clamped to
        ``[1, n_harmonics]`` (H=1 -- the GLS design -- is the floor)."""
        return max(1, min(self.n_harmonics,
                          (int(n_obs) - 2 * int(n_bands)) // 4))

    def _design(self, t, twopift, codes, n_bands):
        cols = [_offset_columns(codes, n_bands)]
        for j in range(1, self._capped_order(len(t), n_bands) + 1):
            cols.append(np.cos(j * twopift))
            cols.append(np.sin(j * twopift))
        return np.column_stack(cols)


class MultibandLSEstimator(_LinearLSEstimator):
    """Multiband least squares (VanderPlas & Ivezic 2015): the fair sparse-
    multiband LS competitor.

    Each band carries its **own** free Fourier series of order ``n_harmonics``
    (offset + ``2H`` coefficients), and only the period is shared across bands --
    the block-diagonal per-band-shape model.  At ``n_harmonics=1`` this is the
    canonical per-band floating-mean sinusoid with a shared period; larger orders
    let each band fit its own harmonic shape.  More flexible than the shared-shape
    MHLS, so it is the realistic LS baseline against which the FTP shape prior must
    earn its sparse-regime advantage.
    """

    def __init__(self, n_harmonics=1):
        if int(n_harmonics) < 1:
            raise ValueError("n_harmonics must be >= 1")
        self.n_harmonics = int(n_harmonics)

    def _design(self, t, twopift, codes, n_bands):
        n = len(t)
        H = self.n_harmonics
        per_band = 1 + 2 * H                       # offset + (cos, sin) x H
        X = np.zeros((n, n_bands * per_band), dtype=float)
        harmonics = np.arange(1, H + 1)
        cosines = np.cos(np.outer(twopift, harmonics))      # (n, H)
        sines = np.sin(np.outer(twopift, harmonics))
        for b in range(n_bands):
            m = (codes == b)
            base = b * per_band
            X[m, base] = 1.0
            X[np.ix_(m, base + 1 + np.arange(H))] = cosines[m]
            X[np.ix_(m, base + 1 + H + np.arange(H))] = sines[m]
        return X


# ----------------------------------------------------------------------
# FTP itself, as an estimator (so the harness scores it like any baseline)
# ----------------------------------------------------------------------
class FTPEstimator(object):
    """The Fast (multiband) Template Periodogram as a pluggable estimator.

    Wraps :class:`~ftperiodogram.multiband.FastMultibandTemplatePeriodogram` in
    catalog mode: at each frequency the power is the max over the template bank.
    This is the exact computation the recovery scorer's native ``__call__`` runs,
    re-expressed as an estimator so FTP is scored through the same
    ``score_estimator`` seam as every comparison method.
    """

    def __init__(self, templates, mode='floating_offsets'):
        self.templates = list(templates)
        self.mode = mode

    def power_spectrum(self, t, y, bands, dy, freqs):
        model = FastMultibandTemplatePeriodogram(self.templates, mode=self.mode)
        model.fit(t, y, bands, dy)
        return model.power(np.asarray(freqs, dtype=float), fast=True,
                           save_best_model=False)

    def __call__(self, t, y, bands, dy, freqs):
        freqs = np.asarray(freqs, dtype=float)
        powers = self.power_spectrum(t, y, bands, dy, freqs)
        return 1.0 / freqs[int(np.argmax(powers))]


# ----------------------------------------------------------------------
# Sesar-style non-linear template fit: the slow gold-standard oracle
# ----------------------------------------------------------------------
def _template_fit_chi2(template, twopift, y, w, codes, n_bands, n_tau, polish):
    """Best weighted chi2 of ``y ~ a * template(f t - tau) + offset_band`` over
    the phase shift ``tau`` (turns), with the **amplitude constrained ``a >= 0``**
    (a fixed-orientation template scaled by a positive amplitude -- the physical
    Sesar/Stringer template fit, and exactly FTP's positive-amplitude convention).

    ``a`` and the per-band offsets are solved by weighted least squares at each
    ``tau``; where the unconstrained amplitude is negative the constrained optimum
    is ``a = 0`` (the offsets-only fit, ``base_chi2``).  The global optimum over
    ``tau`` is found by a dense brute scan (``n_tau`` points -- the "oracle")
    refined by a local golden-section ``polish``, so it matches FTP's analytic
    phase optimum to ``< 1e-6``.
    """
    base = _offset_columns(codes, n_bands)
    base_chi2 = _weighted_chi2(base, y, w)         # a = 0 fallback (offsets only)
    phase0 = twopift / (2.0 * np.pi)               # = f * t, in turns
    sw = np.sqrt(w)
    base_w = base * sw[:, None]
    yw = y * sw

    def chi2_at(tau):
        """Constrained (a >= 0) weighted chi2 at phase ``tau``."""
        shape = template(phase0 - tau)             # template is 1-periodic in turns
        Xw = np.column_stack([base_w, shape * sw])
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        if beta[-1] < 0:                           # inverted template -> clamp a = 0
            return base_chi2
        resid = yw - Xw.dot(beta)
        return float(resid.dot(resid))

    taus = np.arange(n_tau) / float(n_tau)
    vals = np.array([chi2_at(tau) for tau in taus])
    k = int(np.argmin(vals))
    best_tau, best_chi2 = taus[k], float(vals[k])

    if polish:
        # golden-section search in the bracketing grid cell around the minimum
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = (k - 1) / float(n_tau), (k + 1) / float(n_tau)
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = chi2_at(c), chi2_at(d)
        for _ in range(60):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = chi2_at(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = chi2_at(d)
            if abs(b - a) < 1e-12:
                break
        tau_p = 0.5 * (a + b)
        chi2_p = chi2_at(tau_p)
        if chi2_p < best_chi2:
            best_tau, best_chi2 = tau_p, chi2_p
    return best_chi2, best_tau % 1.0


class SesarOracleEstimator(object):
    """Slow non-linear template fit -- the period-recovery gold standard.

    For every template in the bank and every trial frequency, the floating-offset
    template model ``y_b = a * M(f t - tau) + c_b`` is fit by optimizing the phase
    shift ``tau`` over a dense grid + local polish (``a`` and the per-band offsets
    ``c_b`` solved linearly at each ``tau``); the power is the best explained-
    variance fraction over templates.  This reproduces FTP's analytic optimum to
    ``< 1e-6`` -- so it is both the **correctness** oracle (FTP must match it) and
    the **cost** reference timed in deliverable (iii)'s cost-vs-accuracy panel.

    It is deliberately brute force; do not loosen ``n_tau`` to speed it up (that
    is the whole point of the comparison).  Run it on a subsample of sources.
    """

    def __init__(self, templates, mode='floating_offsets', n_tau=256, polish=True):
        if mode != 'floating_offsets':
            raise ValueError("SesarOracleEstimator currently fits the "
                             "floating_offsets model (shared shape + per-band "
                             "offset); got mode=%r" % (mode,))
        self.templates = list(templates)
        self.mode = mode
        self.n_tau = int(n_tau)
        self.polish = bool(polish)

    def power_spectrum(self, t, y, bands, dy, freqs):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        n = t.size
        w = weights(_errs(dy, n))
        labels, codes = _band_index(bands, n)
        n_bands = len(labels)
        chi2_0 = _chi2_0(y, w, codes, n_bands)
        if chi2_0 <= 0:
            return np.zeros(freqs.size, dtype=float)

        two_pi_t = 2.0 * np.pi * t
        best = np.full(freqs.size, np.inf, dtype=float)
        for tmpl in self.templates:
            for i, f in enumerate(freqs):
                chi2, _ = _template_fit_chi2(tmpl, two_pi_t * f, y, w, codes,
                                             n_bands, self.n_tau, self.polish)
                if chi2 < best[i]:
                    best[i] = chi2
        return 1.0 - best / chi2_0

    def __call__(self, t, y, bands, dy, freqs):
        freqs = np.asarray(freqs, dtype=float)
        powers = self.power_spectrum(t, y, bands, dy, freqs)
        return 1.0 / freqs[int(np.argmax(powers))]
