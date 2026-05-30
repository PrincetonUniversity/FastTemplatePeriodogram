"""Recovery scorer + K-sweep driver (Phase 3.2 harness orchestration).

This is the FTP-coupled layer that ties the simulator and the recovery metrics to
the periodogram and the vocabulary builder:

* :class:`RecoveryScorer` -- simulates a population of multiband light curves ONCE
  (frozen, seeded), then scores any candidate template set by the fraction of
  sources whose period it recovers.  Because the population is frozen, every
  template set is scored against the *same* injected light curves, the per-source
  recovered mask is aligned across calls, and the result is deterministic -- the
  three properties the greedy selector (catalog_builder, ``method='greedy'``)
  relies on.
* :func:`k_sweep_recovery` -- the headline recovery-vs-K curve: for each K it builds
  a size-K vocabulary with :func:`~ftperiodogram.catalog_builder.build_template_catalog`
  and scores it, with FTP@H=1 (a single-cosine template) as the GLS-equivalent
  lower-bound baseline, all on one explicit ``[f_min, f_max]`` grid.

The recovery scorer is the seam the K-sweep and the greedy selector share.
Search grids are explicit ``[f_min, f_max]`` -- never a Nyquist heuristic.
"""
from collections import namedtuple

import numpy as np

from .template import Template
from .multiband import FastMultibandTemplatePeriodogram
from .catalog_builder import build_template_catalog
from . import simulate as _sim
from . import recovery as _rec


def frequency_grid(f_min, f_max, n_freq):
    """Explicit uniform search grid over ``[f_min, f_max]`` with ``n_freq`` points.

    Refuses the Nyquist heuristic (all three bounds are required).  The grid is
    NFFT-aligned -- ``freqs[0]`` is an integer multiple of ``df`` -- which the fast
    summations require; ``f_min`` is snapped to the nearest such multiple (a shift
    of at most ``df/2``).
    """
    if f_min is None or f_max is None or n_freq is None:
        raise ValueError("explicit f_min, f_max, n_freq are required "
                         "(set the search band by [f_min, f_max], not nyquist_factor)")
    f_min, f_max, n_freq = float(f_min), float(f_max), int(n_freq)
    if not 0 < f_min < f_max:
        raise ValueError("require 0 < f_min < f_max")
    if n_freq < 2:
        raise ValueError("n_freq must be >= 2")
    df = (f_max - f_min) / (n_freq - 1)
    nf0 = int(round(f_min / df))
    return df * (nf0 + np.arange(n_freq))


def _align_grid(freqs):
    """Snap a uniform ascending grid so ``freqs[0]`` is an integer multiple of
    ``df`` (the NFFT fast-summation requirement).  Idempotent on grids from
    :func:`frequency_grid`."""
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.size < 2:
        raise ValueError("freqs must be a 1-D grid with >= 2 points")
    df = freqs[1] - freqs[0]
    if df <= 0 or not np.allclose(np.diff(freqs), df, rtol=1e-6, atol=0):
        raise ValueError("freqs must be a uniformly-spaced ascending grid")
    nf0 = int(round(freqs[0] / df))
    return df * (nf0 + np.arange(freqs.size))


# ----------------------------------------------------------------------
# Recovery scorer (frozen population)
# ----------------------------------------------------------------------
class RecoveryScorer(object):
    """Score candidate template sets by simulated period recovery.

    The population of ``n_sources`` multiband light curves is simulated once at
    construction (truth shapes drawn from ``truth_templates``, injected periods
    sampled in the grid band unless ``p_true`` is given) and frozen.  Calling the
    scorer runs the FTP multiband periodogram (catalog mode) for the candidate set
    over the explicit grid and returns the recovery rate -- the fraction of
    sources whose recovered period passes :func:`ftperiodogram.recovery.classify_recovery`.
    ``harmonic_aware`` selects exact (default) vs exact-or-harmonic recovery.
    """

    def __init__(self, cadence, truth_templates, *, freqs, n_sources=64,
                 p_true=None, mode='floating_offsets', criterion='fractional',
                 harmonic_aware=False, delta_phi_max=0.5, rtol=0.01,
                 amplitude=0.5, mean_mag=15.0, band_amplitudes=None,
                 band_offsets=None, random_state=0):
        self.freqs = _align_grid(freqs)        # NFFT-aligned (snaps if needed)
        self.n_sources = int(n_sources)
        self.mode = mode
        self.criterion = criterion
        self.harmonic_aware = bool(harmonic_aware)
        self.delta_phi_max = float(delta_phi_max)
        self.rtol = float(rtol)

        rng = _sim._as_rng(random_state)
        truth_templates = list(truth_templates)
        if not truth_templates:
            raise ValueError("truth_templates is empty")

        f_min, f_max = float(self.freqs.min()), float(self.freqs.max())
        if p_true is None:
            self.p_true = 1.0 / rng.uniform(f_min, f_max, size=self.n_sources)
        else:
            p = np.atleast_1d(np.asarray(p_true, dtype=float))
            if p.size == 1:
                self.p_true = np.full(self.n_sources, float(p[0]))
            elif p.size == self.n_sources:
                self.p_true = p
            else:
                raise ValueError("p_true must be a scalar or length n_sources")

        # Freeze the population: each source = one simulated multiband light curve.
        self._sources = []
        for i in range(self.n_sources):
            truth = truth_templates[rng.randint(len(truth_templates))]
            lc = _sim.simulate_multiband_lightcurve(
                truth, self.p_true[i], cadence, amplitude=amplitude,
                mean_mag=mean_mag, tau=rng.rand(), band_amplitudes=band_amplitudes,
                band_offsets=band_offsets, random_state=rng, add_noise=True,
                shuffle=True)
            baseline = float(lc.t.max() - lc.t.min())
            self._sources.append((lc.t, lc.y, lc.bands, lc.dy, self.p_true[i],
                                  baseline))

    def _recovered_one(self, templates, source):
        t, y, bands, dy, P_true, baseline = source
        model = FastMultibandTemplatePeriodogram(list(templates), mode=self.mode)
        model.fit(t, y, bands, dy)
        powers = model.power(self.freqs, fast=True, save_best_model=False)
        P_rec = 1.0 / self.freqs[int(np.argmax(powers))]
        result = _rec.classify_recovery(
            P_rec, P_true, baseline=baseline, criterion=self.criterion,
            rtol=self.rtol, delta_phi_max=self.delta_phi_max,
            count_harmonics=self.harmonic_aware)
        return bool(result.recovered)

    def __call__(self, templates, *, return_mask=False):
        """Recovery rate of ``templates`` over the frozen population.

        Returns ``rate`` (a float) or ``(rate, mask)`` with the per-source boolean
        recovered mask when ``return_mask`` -- the mask is what the greedy selector
        needs to target the not-yet-recovered subset.
        """
        mask = np.array([self._recovered_one(templates, s)
                         for s in self._sources], dtype=bool)
        rate = float(mask.mean()) if mask.size else 0.0
        return (rate, mask) if return_mask else rate

    def source_masks(self, templates):
        """Per-template standalone recovered masks, shape ``(len(templates), n_sources)``.

        Each row is the recovered mask of that single template alone; the greedy
        selector unions these (catalog power is the per-frequency max over the set,
        so a union of standalone masks exactly models the catalog's recovery)."""
        if not list(templates):
            return np.zeros((0, self.n_sources), dtype=bool)
        return np.array([self.__call__([tmpl], return_mask=True)[1]
                         for tmpl in templates], dtype=bool)


def make_recovery_scorer(cadence, truth_templates, *, freqs, n_sources=64,
                         p_true=None, mode='floating_offsets',
                         criterion='fractional', harmonic_aware=False,
                         delta_phi_max=0.5, rtol=0.01, amplitude=0.5,
                         mean_mag=15.0, band_amplitudes=None, band_offsets=None,
                         random_state=0):
    """Build a :class:`RecoveryScorer` over a frozen simulated population."""
    return RecoveryScorer(
        cadence, truth_templates, freqs=freqs, n_sources=n_sources, p_true=p_true,
        mode=mode, criterion=criterion, harmonic_aware=harmonic_aware,
        delta_phi_max=delta_phi_max, rtol=rtol, amplitude=amplitude,
        mean_mag=mean_mag, band_amplitudes=band_amplitudes,
        band_offsets=band_offsets, random_state=random_state)


# ----------------------------------------------------------------------
# K-sweep driver
# ----------------------------------------------------------------------
KSweepResult = namedtuple(
    'KSweepResult',
    ['k_values', 'recovery', 'recovery_by_k', 'baseline_recovery',
     'baseline_mask', 'n_sources', 'freq_grid', 'criterion', 'seed'])


def k_sweep_recovery(templates, cadence, k_values=(1, 2, 4, 8), *, scorer=None,
                     recovery_config=None, f_min=None, f_max=None, n_freq=None,
                     n_sources=64, p_true=None, mode='floating_offsets',
                     include_baseline=True, catalog_kwargs=None,
                     return_masks=False, random_state=0):
    """Recovery-vs-K curve for vocabularies built from ``templates``.

    For each K in ``k_values`` a size-K vocabulary is built with
    :func:`build_template_catalog` (``method='pam'``) and scored against the same
    frozen simulated population; ``include_baseline`` adds the FTP@H=1
    (single-cosine, GLS-equivalent) lower bound.  Either pass a prebuilt ``scorer``
    or let one be built from the explicit grid (``f_min``/``f_max``/``n_freq``
    required) and ``recovery_config``.  Returns a :class:`KSweepResult`.
    """
    templates = list(templates)
    k_values = sorted({int(k) for k in k_values})
    if not k_values:
        raise ValueError("k_values is empty")
    if k_values[0] < 1 or k_values[-1] > len(templates):
        raise ValueError("each K must be in [1, len(templates)]; got %r for %d "
                         "templates" % (k_values, len(templates)))

    catalog_kwargs = dict(catalog_kwargs or {})
    catalog_kwargs.setdefault('random_state', random_state)

    if scorer is None:
        freqs = frequency_grid(f_min, f_max, n_freq)
        scorer = make_recovery_scorer(
            cadence, templates, freqs=freqs, n_sources=n_sources, p_true=p_true,
            mode=mode, random_state=random_state, **dict(recovery_config or {}))

    baseline_recovery, baseline_mask = None, None
    if include_baseline:
        gls = [Template([1.0], [0.0])]                # H=1 single cosine == GLS
        if return_masks:
            baseline_recovery, baseline_mask = scorer(gls, return_mask=True)
        else:
            baseline_recovery = scorer(gls)

    recovery = np.empty(len(k_values), dtype=float)
    recovery_by_k = [] if return_masks else None
    for i, K in enumerate(k_values):
        vocab = build_template_catalog(templates, K, method='pam', **catalog_kwargs)
        if return_masks:
            rate, mask = scorer(vocab, return_mask=True)
            recovery_by_k.append(mask)
        else:
            rate = scorer(vocab)
        recovery[i] = rate

    return KSweepResult(
        k_values=np.asarray(k_values, dtype=int), recovery=recovery,
        recovery_by_k=recovery_by_k, baseline_recovery=baseline_recovery,
        baseline_mask=baseline_mask, n_sources=scorer.n_sources,
        freq_grid=(float(scorer.freqs.min()), float(scorer.freqs.max()),
                   int(scorer.freqs.size)),
        criterion=scorer.criterion, seed=int(random_state))
