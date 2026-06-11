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
from math import comb as _comb
import multiprocessing as _mp
import os
import zlib

import numpy as np

from .template import Template
from .multiband import (build_template_set, compute_band_summations,
                        solve_over_frequencies)
from .baselines import FTPEstimator
from .catalog_builder import build_template_catalog, _orbit_distance
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
# Honest uncertainty for recovery proportions (WP B1)
# ----------------------------------------------------------------------
def wilson_interval(k, n, z=1.959963984540054):
    """Wilson score confidence interval for a binomial proportion.

    ``k`` successes out of ``n`` trials (either may be an array; they broadcast)
    at confidence ``z`` (default two-sided 95%).  Returns ``(lo, hi)``, each
    clipped to ``[0, 1]``; scalars in, scalars out.  Unlike the normal ("Wald")
    interval it is never empty at k=0 or k=n -- the right band for recovery
    rates that saturate at 0 or 1.  ``n = 0`` yields the full-ignorance
    ``(0, 1)``.  Pool counts across seeds *before* calling (the sources are the
    independent trials; seeds only relabel them), rather than averaging
    per-seed intervals.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    scalar = (k.ndim == 0 and n.ndim == 0)
    n_safe = np.maximum(n, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k / n_safe
        denom = 1.0 + z ** 2 / n_safe
        center = (p + z ** 2 / (2.0 * n_safe)) / denom
        half = (z / denom) * np.sqrt(p * (1.0 - p) / n_safe
                                     + z ** 2 / (4.0 * n_safe ** 2))
    lo = np.where(n > 0, np.clip(center - half, 0.0, 1.0), 0.0)
    hi = np.where(n > 0, np.clip(center + half, 0.0, 1.0), 1.0)
    # at k=0 / k=n the bound is analytically exact; snap off the FP rounding
    lo = np.where(k <= 0, 0.0, lo)
    hi = np.where((n > 0) & (k >= n), 1.0, hi)
    return (float(lo), float(hi)) if scalar else (lo, hi)


def mcnemar_test(mask_a, mask_b):
    """Exact McNemar paired test on two recovered masks over the SAME sources.

    Returns ``(n_a_only, n_b_only, p_value)``: the discordant counts (sources
    method A recovers but B does not, and vice versa) and the exact two-sided
    binomial p-value under H0 "both methods recover equally well" (discordant
    outcomes ~ Binomial(n_a_only + n_b_only, 1/2)).  Concordant sources carry no
    information about the contrast and drop out -- this is the right paired test
    for two estimators scored on the identical frozen population, where the
    per-source outcomes are strongly correlated and two overlapping Wilson
    intervals say nothing.  No discordant pairs returns ``p = 1.0``.
    """
    a = np.asarray(mask_a, dtype=bool).ravel()
    b = np.asarray(mask_b, dtype=bool).ravel()
    if a.shape != b.shape:
        raise ValueError("masks must be the same length (paired sources); "
                         "got %d vs %d" % (a.size, b.size))
    n_a_only = int(np.sum(a & ~b))
    n_b_only = int(np.sum(b & ~a))
    n = n_a_only + n_b_only
    if n == 0:
        return n_a_only, n_b_only, 1.0
    m = min(n_a_only, n_b_only)
    cdf = sum(_comb(n, i) for i in range(m + 1)) / 2.0 ** n
    return n_a_only, n_b_only, min(1.0, 2.0 * cdf)


def _seed_int(random_state):
    """Coerce a seed-like value to a non-negative int for deterministic mixing."""
    if random_state is None:
        return 0
    if isinstance(random_state, (int, np.integer)):
        return int(random_state) & 0xffffffff
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, 2 ** 31 - 1))
    return int(random_state) & 0xffffffff


def _combine_seeds(base, source_index, band):
    """Deterministic per-(source, band) seed (stable across runs/processes).

    Uses ``zlib.crc32`` rather than ``hash`` so the value does not depend on
    ``PYTHONHASHSEED``; reused verbatim across N_epochs levels so a larger
    down-sample is a strict superset of a smaller one (nested thinning)."""
    key = "%d|%d|%r" % (int(base), int(source_index), band)
    return zlib.crc32(key.encode('utf-8')) & 0xffffffff


# ----------------------------------------------------------------------
# Per-source recovery (shared by the serial and parallel paths)
# ----------------------------------------------------------------------
# Limiting BLAS to one thread per worker avoids oversubscription when the source
# loop is fanned across processes; spawned children inherit these at interpreter
# startup (before numpy imports), so the parent sets them just around the pool.
_BLAS_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def _estimator_recovered(estimator, source, freqs, crit):
    """``(recovered, P_rec)`` for one frozen source under a pluggable estimator.

    ``estimator(t, y, bands, dy, freqs) -> P_rec`` is any period estimator -- the
    FTP catalog periodogram (:class:`~ftperiodogram.baselines.FTPEstimator`), a
    comparison baseline, or the Sesar oracle -- so the source is scored identically
    regardless of which produced ``P_rec``.  The recovered period is returned next
    to the bool so callers can persist it and re-classify under a different
    criterion post hoc.  Identical computation serial or parallel (the per-source
    estimates are independent), so fanning sources across processes is
    mask-identical to the serial loop; the estimator must be picklable for the
    ``spawn`` pool, which every :mod:`ftperiodogram.baselines` estimator is.
    """
    t, y, bands, dy, P_true, baseline = source
    P_rec = float(estimator(t, y, bands, dy, np.asarray(freqs, dtype=float)))
    result = _rec.classify_recovery(
        P_rec, P_true, baseline=baseline, criterion=crit['criterion'],
        rtol=crit['rtol'], delta_phi_max=crit['delta_phi_max'],
        count_harmonics=crit['harmonic_aware'])
    return bool(result.recovered), P_rec


_PAR = {}                                # per-worker fixed data, set by _par_init


def _par_init(estimator, freqs, crit, sources):
    _PAR.update(estimator=estimator, freqs=freqs, crit=crit, sources=sources)


def _par_recovered(i):
    return _estimator_recovered(_PAR['estimator'], _PAR['sources'][i],
                                _PAR['freqs'], _PAR['crit'])


def _parallel_map(n_jobs, n_items, initializer, initargs, worker):
    """Fan ``worker(i)`` for ``i in range(n_items)`` over a ``spawn`` pool, with
    BLAS pinned to one thread per worker (no oversubscription)."""
    if n_jobs < 0:
        n_jobs = _mp.cpu_count()
    n_jobs = max(1, min(int(n_jobs), n_items))
    prev = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
    for k in _BLAS_THREAD_VARS:
        os.environ[k] = '1'
    try:
        ctx = _mp.get_context('spawn')
        with ctx.Pool(n_jobs, initializer=initializer, initargs=initargs) as pool:
            return pool.map(worker, range(n_items))
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _source_mask_column(templates, source, freqs, mode, crit, H):
    """Per-template standalone recovered mask for ONE source (a mask column).

    The per-band NFFT summations are template-independent, so they are computed
    once per source and reused across the whole template list -- the fast path that
    lets greedy scale.  Identical computation serial or parallel."""
    t, y, bands, dy, P_true, baseline = source
    sumlists, stats = compute_band_summations(t, y, bands, freqs, H, dy=dy,
                                              mode=mode, fast=True)
    col = np.empty(len(templates), dtype=bool)
    for i, tmpl in enumerate(templates):
        template_dict = build_template_set(tmpl, bands)
        powers, _ = solve_over_frequencies(template_dict, sumlists, stats,
                                           freqs.size, mode=mode)
        P_rec = 1.0 / freqs[int(np.argmax(powers))]
        result = _rec.classify_recovery(
            P_rec, P_true, baseline=baseline, criterion=crit['criterion'],
            rtol=crit['rtol'], delta_phi_max=crit['delta_phi_max'],
            count_harmonics=crit['harmonic_aware'])
        col[i] = bool(result.recovered)
    return col


_PARSM = {}                              # per-worker fixed data for source_masks


def _parsm_init(templates, freqs, mode, crit, H, sources):
    _PARSM.update(templates=templates, freqs=freqs, mode=mode, crit=crit, H=H,
                  sources=sources)


def _parsm_col(j):
    return _source_mask_column(_PARSM['templates'], _PARSM['sources'][j],
                               _PARSM['freqs'], _PARSM['mode'], _PARSM['crit'],
                               _PARSM['H'])


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
                 band_offsets=None, intrinsic_jitter=0.0, random_state=0,
                 n_jobs=1):
        self.freqs = _align_grid(freqs)        # NFFT-aligned (snaps if needed)
        self.n_sources = int(n_sources)
        self.n_jobs = int(n_jobs)              # source-loop processes (1 = serial)
        self.intrinsic_jitter = float(intrinsic_jitter)
        self._set_scoring_params(mode=mode, criterion=criterion,
                                 harmonic_aware=harmonic_aware,
                                 delta_phi_max=delta_phi_max, rtol=rtol)

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
        self._truth = []                  # per-source generating shape (post-jitter)
        for i in range(self.n_sources):
            truth = truth_templates[rng.randint(len(truth_templates))]
            if self.intrinsic_jitter > 0:
                # Per-source relative (zero-mean) Fourier jitter: the population
                # shares a latent library shape but no fixed finite vocabulary is
                # exactly right, so joint refinement has legitimate headroom to
                # recover the cluster centroid that a frozen clean medoid cannot.
                z = np.asarray(truth.c_n, float) - 1j * np.asarray(truth.s_n, float)
                z = z + self.intrinsic_jitter * np.abs(z) * (
                    rng.randn(len(z)) + 1j * rng.randn(len(z)))
                truth = Template(np.real(z).copy(), (-np.imag(z)).copy(),
                                 template_id=truth.template_id)
            lc = _sim.simulate_multiband_lightcurve(
                truth, self.p_true[i], cadence, amplitude=amplitude,
                mean_mag=mean_mag, tau=rng.rand(), band_amplitudes=band_amplitudes,
                band_offsets=band_offsets, random_state=rng, add_noise=True,
                shuffle=True)
            baseline = float(lc.t.max() - lc.t.min())
            self._truth.append(truth)
            self._sources.append((lc.t, lc.y, lc.bands, lc.dy, self.p_true[i],
                                  baseline))

    def _crit(self):
        return dict(criterion=self.criterion, rtol=self.rtol,
                    delta_phi_max=self.delta_phi_max,
                    harmonic_aware=self.harmonic_aware)

    def _recovered_pairs_parallel(self, estimator, n_jobs):
        """Per-source ``(recovered, P_rec)`` pairs, fanned over a spawn pool
        (identical to the serial loop)."""
        return _parallel_map(
            n_jobs, self.n_sources, _par_init,
            (estimator, self.freqs, self._crit(), self._sources), _par_recovered)

    def score_estimator(self, estimator, *, return_mask=False,
                        return_periods=False, n_jobs=None):
        """Recovery rate of a pluggable period ``estimator`` over the frozen
        population -- the seam that scores every method on the identical sources.

        ``estimator(t, y, bands, dy, freqs) -> P_rec`` is any
        :mod:`ftperiodogram.baselines` estimator (FTP, GLS, MHLS, multiband LS, the
        Sesar oracle) or an equivalent picklable callable.  Returns ``rate``; with
        ``return_mask``, ``(rate, mask)``; with ``return_periods``,
        ``(rate, mask, P_rec)`` where ``P_rec`` is the per-source recovered-period
        array -- persisting it makes every recovery criterion (fractional,
        phase-coherence, alias breakdown) re-scorable post hoc without re-running
        the estimator.  ``n_jobs`` (default the scorer's) fans the per-source loop
        over that many ``spawn`` processes; sources are independent, so the
        parallel result is identical to the serial loop.
        """
        n_jobs = self.n_jobs if n_jobs is None else int(n_jobs)
        if n_jobs == 1 or self.n_sources <= 1:
            pairs = [_estimator_recovered(estimator, s, self.freqs, self._crit())
                     for s in self._sources]
        else:
            pairs = self._recovered_pairs_parallel(estimator, n_jobs)
        mask = np.array([rec for rec, _ in pairs], dtype=bool)
        rate = float(mask.mean()) if mask.size else 0.0
        if return_periods:
            periods = np.array([p for _, p in pairs], dtype=float)
            return rate, mask, periods
        return (rate, mask) if return_mask else rate

    def __call__(self, templates, *, return_mask=False, return_periods=False,
                 n_jobs=None):
        """Recovery rate of the FTP catalog ``templates`` over the frozen population.

        A thin wrapper over :meth:`score_estimator` with an
        :class:`~ftperiodogram.baselines.FTPEstimator` (catalog mode, the scorer's
        ``mode``): FTP is scored through the same seam as every comparison method.
        Returns ``rate``, ``(rate, mask)``, or ``(rate, mask, P_rec)`` exactly as
        :meth:`score_estimator`; the per-source boolean recovered mask is what the
        greedy selector needs to target the not-yet-recovered subset.
        """
        return self.score_estimator(FTPEstimator(list(templates), mode=self.mode),
                                     return_mask=return_mask,
                                     return_periods=return_periods, n_jobs=n_jobs)

    def source_masks(self, templates, *, n_jobs=None):
        """Per-template standalone recovered masks, shape ``(len(templates), n_sources)``.

        Each row is the recovered mask of that single template alone; the greedy
        selector unions these (catalog power is the per-frequency max over the set,
        so a union of standalone masks exactly models the catalog's recovery).

        Fast path: the per-band NFFT summations are template-independent, so for each
        source they are computed ONCE and reused across the whole template list (one
        NFFT per source instead of one per ``(template, source)`` -- the speedup that
        lets greedy scale past the Sesar universe).  The recovered mask is identical
        to the per-template path: both reduce to ``argmax`` of the *same* fast
        summations, and only sub-``1e-16`` power jitter sits below the argmax.  The
        fast path requires a common harmonic order (the summations are order-specific)
        and a non-``'sesar'`` mode (sesar needs ``relative_offsets`` the scorer never
        supplies); otherwise it falls back to the exact per-template path.

        ``n_jobs`` (default the scorer's) fans the per-source loop over a ``spawn``
        pool -- mask-identical to the serial loop, and the lever that makes the
        all-universe greedy precompute tractable at production scale (98 templates x
        1024 sources x 10^4 freq is hours serial, minutes on many cores)."""
        templates = list(templates)
        if not templates:
            return np.zeros((0, self.n_sources), dtype=bool)

        harmonics = {len(t.c_n) for t in templates}
        if len(harmonics) != 1 or self.mode == 'sesar':
            return np.array([self.__call__([tmpl], return_mask=True)[1]
                             for tmpl in templates], dtype=bool)

        H = harmonics.pop()
        crit = self._crit()
        n_jobs = self.n_jobs if n_jobs is None else int(n_jobs)
        if n_jobs == 1 or self.n_sources <= 1:
            cols = [_source_mask_column(templates, s, self.freqs, self.mode, crit, H)
                    for s in self._sources]
        else:
            cols = _parallel_map(
                n_jobs, self.n_sources, _parsm_init,
                (templates, self.freqs, self.mode, crit, H, self._sources),
                _parsm_col)
        return np.stack(cols, axis=1)                # (len(templates), n_sources)

    def assignment_accuracy(self, vocab, *, return_counts=False):
        """Correct-template-assignment rate on the correct-period subset.

        Among the frozen sources whose period this ``vocab`` recovers (the FTP
        catalog peak passes :func:`ftperiodogram.recovery.classify_recovery`), the
        fraction the FTP catalog assigns -- the argmax-power template at that peak --
        to the ``vocab`` template that is orbit-closest to the source's TRUE generating
        shape.  This is the *mechanism* metric for the joint-vs-pipeline study: period
        recovery saturates and clips shape gains, so this exposes whether a vocabulary
        matches the underlying shapes better even where period recovery cannot show it.

        Requires a scorer built with per-source truth shapes (``__init__`` /
        :meth:`downsample`); raises ``ValueError`` otherwise.  ``return_counts``
        returns ``(rate, n_correct, n_subset, null_rate)`` where ``null_rate`` is
        the MAJORITY-TARGET null on the same subset -- the accuracy of always
        predicting the subset's most common correct target.  Quote accuracies
        against this null (and with ``n_subset``), not against ``1/K``: targets
        are not uniform, so ``1/K`` overstates the skill.
        """
        from .joint_em import _estep_one              # lazy: avoids an import cycle
        if getattr(self, '_truth', None) is None:
            raise ValueError("assignment_accuracy needs per-source truth shapes; "
                             "build via RecoveryScorer / make_recovery_scorer")
        vocab = list(vocab)
        H = len(vocab[0].c_n)
        # truth -> orbit-nearest vocab template (the "correct" assignment target)
        target = [int(np.argmin([_orbit_distance(tr, v) for v in vocab]))
                  for tr in self._truth]
        crit = self._crit()
        n_correct, n_subset = 0, 0
        subset_targets = []
        for i, source in enumerate(self._sources):
            assigned, freq_rec, _ = _estep_one(source, vocab, self.freqs,
                                               self.mode, H)
            recovered = _rec.classify_recovery(
                1.0 / freq_rec, self.p_true[i], baseline=source[5],
                criterion=crit['criterion'], rtol=crit['rtol'],
                delta_phi_max=crit['delta_phi_max'],
                count_harmonics=crit['harmonic_aware']).recovered
            if recovered:
                n_subset += 1
                n_correct += int(assigned == target[i])
                subset_targets.append(target[i])
        rate = (n_correct / n_subset) if n_subset else 0.0
        null_rate = (max(np.bincount(subset_targets)) / n_subset
                     if n_subset else 0.0)
        return (rate, n_correct, n_subset, null_rate) if return_counts else rate

    def _set_scoring_params(self, *, mode, criterion, harmonic_aware,
                            delta_phi_max, rtol):
        """Set the scoring knobs shared by ``__init__`` and ``_from_sources``."""
        self.mode = mode
        self.criterion = criterion
        self.harmonic_aware = bool(harmonic_aware)
        self.delta_phi_max = float(delta_phi_max)
        self.rtol = float(rtol)

    @classmethod
    def _from_sources(cls, sources, *, freqs, p_true=None, truth=None,
                      mode='floating_offsets', criterion='fractional',
                      harmonic_aware=False, delta_phi_max=0.5, rtol=0.01,
                      n_jobs=1):
        """Build a scorer from an already-frozen population (no simulation).

        ``freqs`` is copied verbatim (assumed already NFFT-aligned -- the master
        scorer's grid); each source is ``(t, y, bands, dy, P_true, baseline)``.
        This backs :meth:`downsample` and any other re-freezing of the same
        injected truths under a different cadence realization."""
        self = cls.__new__(cls)
        self.freqs = np.asarray(freqs, dtype=float)       # already aligned; no re-snap
        self._sources = list(sources)
        self.n_sources = len(self._sources)
        self.n_jobs = int(n_jobs)
        self.intrinsic_jitter = 0.0      # already baked into the frozen sources
        self._truth = list(truth) if truth is not None else None
        self._set_scoring_params(mode=mode, criterion=criterion,
                                 harmonic_aware=harmonic_aware,
                                 delta_phi_max=delta_phi_max, rtol=rtol)
        if p_true is None:
            self.p_true = np.array([s[4] for s in self._sources], dtype=float)
        else:
            self.p_true = np.asarray(p_true, dtype=float)
        return self

    def downsample(self, n_epochs, *, random_state=0, freeze_baseline=True):
        """Return a sibling scorer with each source thinned to ``n_epochs`` per band.

        The thinning is a *nested*, seeded subset of the already-simulated rows --
        truth shapes and injected periods are unchanged -- so a recovery-vs-N_epochs
        sweep varies only the per-band epoch count (design A.4: "hold the LC fixed,
        down-sample it").  For a fixed ``random_state`` a larger ``n_epochs`` is a
        strict superset of a smaller one, because one frozen permutation per
        ``(source, band)`` is reused across N and the first ``n_epochs`` are kept.

        ``freeze_baseline`` (default ``True``) keeps each source's original baseline
        ``T`` so the phase-coherence criterion isolates epoch count from baseline
        (the headline 1% fractional criterion is baseline-independent regardless);
        set it ``False`` to recompute ``T`` from the thinned epochs.

        Requires ``3 <= n_epochs <=`` each source's master per-band count.
        """
        n_epochs = int(n_epochs)
        if n_epochs < 3:
            raise ValueError("n_epochs must be >= 3 per band (the fast NFFT path "
                             "is degenerate below that); got %r" % (n_epochs,))
        base = _seed_int(random_state)
        new_sources = []
        for si, (t, y, bands, dy, P_true, baseline) in enumerate(self._sources):
            band_labels = [None] if bands is None else list(np.unique(bands))
            keep_parts = []
            for b in band_labels:
                idx = (np.arange(len(t)) if bands is None
                       else np.flatnonzero(bands == b))
                if n_epochs > idx.size:
                    raise ValueError(
                        "n_epochs=%d exceeds source %d band %r master count %d"
                        % (n_epochs, si, b, idx.size))
                perm = np.random.RandomState(_combine_seeds(base, si, b))
                order = perm.permutation(idx.size)
                keep_parts.append(idx[order[:n_epochs]])
            keep = np.sort(np.concatenate(keep_parts))
            t_k, y_k = t[keep], y[keep]
            bands_k = None if bands is None else bands[keep]
            if dy is None or np.ndim(dy) == 0:
                dy_k = dy
            else:
                dy_k = np.asarray(dy)[keep]
            new_baseline = (baseline if freeze_baseline
                            else float(t_k.max() - t_k.min()))
            new_sources.append((t_k, y_k, bands_k, dy_k, P_true, new_baseline))
        return RecoveryScorer._from_sources(
            new_sources, freqs=self.freqs, p_true=self.p_true, mode=self.mode,
            truth=getattr(self, '_truth', None),
            criterion=self.criterion, harmonic_aware=self.harmonic_aware,
            delta_phi_max=self.delta_phi_max, rtol=self.rtol, n_jobs=self.n_jobs)


def make_recovery_scorer(cadence, truth_templates, *, freqs, n_sources=64,
                         p_true=None, mode='floating_offsets',
                         criterion='fractional', harmonic_aware=False,
                         delta_phi_max=0.5, rtol=0.01, amplitude=0.5,
                         mean_mag=15.0, band_amplitudes=None, band_offsets=None,
                         intrinsic_jitter=0.0, random_state=0, n_jobs=1):
    """Build a :class:`RecoveryScorer` over a frozen simulated population."""
    return RecoveryScorer(
        cadence, truth_templates, freqs=freqs, n_sources=n_sources, p_true=p_true,
        mode=mode, criterion=criterion, harmonic_aware=harmonic_aware,
        delta_phi_max=delta_phi_max, rtol=rtol, amplitude=amplitude,
        mean_mag=mean_mag, band_amplitudes=band_amplitudes,
        band_offsets=band_offsets, intrinsic_jitter=intrinsic_jitter,
        random_state=random_state, n_jobs=n_jobs)


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
                     return_masks=False, random_state=0, n_jobs=1):
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
            mode=mode, random_state=random_state, n_jobs=n_jobs,
            **dict(recovery_config or {}))

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


# ----------------------------------------------------------------------
# N_epochs stratification driver (the sparse-regime story, fixed K)
# ----------------------------------------------------------------------
NEpochsSweepResult = namedtuple(
    'NEpochsSweepResult',
    ['n_epochs_values', 'recovery', 'recovery_by_n', 'baseline_recovery',
     'baseline_mask', 'k', 'n_sources', 'freq_grid', 'criterion', 'seed'])


def n_epochs_sweep_recovery(master_scorer, templates, n_epochs_values, *, k,
                            catalog_kwargs=None, include_baseline=True,
                            return_masks=False, random_state=0,
                            freeze_baseline=True):
    """Recovery vs per-band epoch count at a fixed vocabulary size ``k``.

    The size-``k`` vocabulary is built once from ``templates`` with
    :func:`build_template_catalog` (``method='pam'``); ``master_scorer`` is then
    down-sampled (nested, seeded) to each value in ``n_epochs_values`` and the
    *same* vocabulary scored.  ``include_baseline`` also scores the FTP@H=1
    (single-cosine, GLS-equivalent) vocabulary at each N_epochs -- unlike the
    K-sweep this baseline varies with N, so ``baseline_recovery`` is a curve, not a
    scalar.  The headline criterion is the 1% fractional rule (baseline-independent);
    ``freeze_baseline`` keeps the master baseline so the phase-coherence secondary
    isolates epoch count from baseline T.  Returns a :class:`NEpochsSweepResult`.
    """
    templates = list(templates)
    n_epochs_values = sorted({int(n) for n in n_epochs_values})
    if not n_epochs_values:
        raise ValueError("n_epochs_values is empty")
    if not 1 <= k <= len(templates):
        raise ValueError("k must be in [1, len(templates)]; got %r for %d "
                         "templates" % (k, len(templates)))

    catalog_kwargs = dict(catalog_kwargs or {})
    catalog_kwargs.setdefault('random_state', random_state)
    vocab = build_template_catalog(templates, k, method='pam', **catalog_kwargs)
    gls = [Template([1.0], [0.0])]                     # H=1 single cosine == GLS

    recovery = np.empty(len(n_epochs_values), dtype=float)
    recovery_by_n = [] if return_masks else None
    base_rates = [] if include_baseline else None
    base_masks = ([] if (include_baseline and return_masks) else None)

    for i, N in enumerate(n_epochs_values):
        ds = master_scorer.downsample(N, random_state=random_state,
                                      freeze_baseline=freeze_baseline)
        if return_masks:
            rate, mask = ds(vocab, return_mask=True)
            recovery_by_n.append(mask)
        else:
            rate = ds(vocab)
        recovery[i] = rate
        if include_baseline:
            if return_masks:
                br, bm = ds(gls, return_mask=True)
                base_masks.append(bm)
            else:
                br = ds(gls)
            base_rates.append(br)

    return NEpochsSweepResult(
        n_epochs_values=np.asarray(n_epochs_values, dtype=int),
        recovery=recovery, recovery_by_n=recovery_by_n,
        baseline_recovery=(np.asarray(base_rates, dtype=float)
                           if include_baseline else None),
        baseline_mask=base_masks, k=int(k), n_sources=master_scorer.n_sources,
        freq_grid=(float(master_scorer.freqs.min()),
                   float(master_scorer.freqs.max()),
                   int(master_scorer.freqs.size)),
        criterion=master_scorer.criterion, seed=int(random_state))
