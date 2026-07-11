"""Joint (EM / alternating-minimization) template-vocabulary estimation (Phase 3.4).

The pipeline vocabulary (:func:`ftperiodogram.catalog_builder.build_template_catalog`)
clusters a fixed library of *clean* template shapes once, ahead of time.  The
**joint** estimator instead refines that vocabulary against the actual observed
population: it alternates between fitting every light curve with the current
template bank (the E-step) and re-estimating each template from the light curves
that fit it best (the M-step).  This is a multireference-alignment (MRA) problem --
estimating a shared latent *dictionary* of signals from many copies each corrupted
by an unknown phase shift / period and noise -- and MRA sample-complexity theory
(Perry, Weed, Bandeira, Rigollet & Singer 2019: ~``1/SNR^3`` observations at low
SNR) predicts the payoff is concentrated in the **sparse regime**: joint estimation
should help most when each light curve is individually uninformative and become
marginal once the data are dense.  That falsifiable *joint-minus-pipeline gap vs
N_epochs* is the Paper-1 deliverable :func:`build_joint_em_catalog` feeds.

Design (``phase3_vocab_design.md`` S.A.4)
-----------------------------------------
* **Init from k-medoids, never random.**  EM preserves the Fourier-phase geometry
  of its initialization (Balanov, Huleihel & Bendory 2025), so the joint run starts
  from the *exact* pipeline vocabulary and the gap measures only the value of
  refining on data.
* **E-step.**  Each source is fit against the current K-bank with the fast FTP
  multiband solve (one NFFT per source, reused across the bank -- the
  :func:`ftperiodogram.validation.RecoveryScorer.source_masks` fast path), yielding
  its best-matching template, recovered frequency, and fit quality (periodogram
  power = explained-variance fraction at the peak).  The per-band NFFT summations
  depend only on the data and ``(freqs, mode, H)`` -- never on the templates or
  the EM iteration -- so they are additionally cached **across iterations** (and
  across the val-margin scorer) by :class:`_SumsExecutor`, which is exact.
* **Gated, robust M-step.**  A template is updated only from the sources assigned to
  it whose fit quality clears a gate (a relative power quantile, optionally an
  absolute floor) -- a mis-estimated period (the dominant astronomical failure mode)
  produces a *low*-power fit and is gated out, so it cannot inject a garbage shape.
  Each gated source's empirical shape is re-estimated by a weighted-least-squares
  Fourier fit at its recovered frequency, **orbit-aligned** to the current medoid
  (U(1) circular-shift Procrustes, :func:`_max_cross_correlation`), then the cluster
  is aggregated by the **per-harmonic complex-coefficient median** -- robust to the
  residual wrong-period outliers that survive the gate.
* **Anti-collapse.**  Atom count K is fixed; any update that pushes two templates
  *closer* to within ``diversity_eps`` orbit distance is reverted (the offending
  template keeps its previous shape) and logged.  This is a *no-worsening* guard --
  it prevents atoms merging during refinement; it does not repair a k-medoids init
  that is already within ``diversity_eps`` (assumed diverse by construction).
* **Non-monotone convergence.**  The EM objective is *not* assumed to decrease
  monotonically toward the truth; we early-stop on a held-out recovery metric with
  patience and return the **best held-out** vocabulary, not the last -- the explicit
  guard against the "Ghost of Newton" finite-sample late-iteration instability
  (Balanov et al. 2025).

The returned vocabulary is a plain ``list`` of :class:`Template`, drop-in
interchangeable with a ``build_template_catalog`` result: both are scored through
the same :class:`~ftperiodogram.baselines.FTPEstimator` seam, so the only difference
the joint-minus-pipeline comparison sees is the vocabulary itself.
"""
from collections import namedtuple, OrderedDict
import multiprocessing as _mp
import os

import numpy as np

from .template import Template
from .utils import weights
from .baselines import _band_index, _offset_columns, _errs
from .catalog_builder import (build_template_catalog, _complex_coeffs, _pad,
                              _max_cross_correlation, _orbit_distance_matrix)
from .multiband import (build_template_set, compute_band_summations,
                        solve_over_frequencies)
from .validation import _BLAS_THREAD_VARS


#: Per-iteration trace returned by :func:`build_joint_em_catalog` with
#: ``return_diagnostics=True``.  ``val_signal`` is the held-out early-stop curve
#: over iterations (``val_signal[0]`` is the pipeline-init value) under the
#: criterion named by ``val_signal_name`` (``'power_margin'`` default, or
#: ``'recovery'``), ``best_iter`` the index whose vocabulary is returned,
#: ``min_pairwise_distance`` the orbit distance of the closest template pair each
#: iteration (collapse monitor), ``n_gated`` the per-iteration gated source count,
#: ``n_reverted`` the per-iteration count of anti-collapse reverts, and
#: ``stop_reason`` why the loop ended.  ``estep`` is a dict tracing the E-step
#: itself: per-iteration ``assign_hist`` / ``freq_hist`` / ``power_hist`` arrays,
#: the sums-cache counters (``cache_hits`` / ``cache_misses``), and -- with
#: ``estep_refine`` -- the per-iteration ``solve_fraction`` (solved grid points /
#: full-grid points), ``windowed_pairs``, ``edge_escapes``, ``fallbacks``,
#: ``refine_used`` flags, and ``final_full_grid_rerun`` (whether the stopped
#: iteration's E-step was re-run full-grid after an early stop so the last
#: recorded assignments/freqs/powers never come from a stale window).
EMDiagnostics = namedtuple(
    'EMDiagnostics',
    ['n_iter', 'best_iter', 'val_signal', 'val_signal_name', 'mean_fit_quality',
     'min_pairwise_distance', 'n_gated', 'n_reverted', 'stop_reason', 'estep'])


# ----------------------------------------------------------------------
# Single-source empirical shape fit (weighted-LS Fourier at a fixed frequency)
# ----------------------------------------------------------------------
def _fit_source_shape(t, y, bands, dy, freq, H):
    """Weighted-LS Fourier shape of one light curve folded at ``freq``.

    Fits ``y_b ~ offset_b + sum_{j=1..H} [c_j cos(2 pi j f t) + s_j sin(2 pi j f t)]``
    (a shared shape across bands + a per-band floating offset -- the
    ``floating_offsets`` model at a single known frequency), and returns the shape's
    complex coefficients ``z_j = c_j - i s_j`` (matching :class:`Template`).  Returns
    ``None`` if the fit is degenerate (no shape power), which the caller drops.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = t.size
    w = weights(_errs(dy, n))
    labels, codes = _band_index(bands, n)
    n_bands = len(labels)

    twopift = 2.0 * np.pi * freq * t
    cols = [_offset_columns(codes, n_bands)]
    for j in range(1, H + 1):
        cols.append(np.cos(j * twopift))
        cols.append(np.sin(j * twopift))
    X = np.column_stack(cols)

    sw = np.sqrt(w)
    beta, _, _, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    shape = beta[n_bands:]
    c = shape[0::2]                                  # cos coefficients c_1..c_H
    s = shape[1::2]                                  # sin coefficients s_1..s_H
    if not np.any(np.abs(c) + np.abs(s) > 0):
        return None
    return c - 1j * s                                # z_j = c_j - i s_j


def _template_from_coeffs(z):
    """Build a unit-energy :class:`Template` from complex coefficients ``z = c - i s``."""
    return Template(np.real(z).copy(), (-np.imag(z)).copy())


# ----------------------------------------------------------------------
# E-step: assign each source to its best template (+ recovered freq, fit quality)
# ----------------------------------------------------------------------
#: Warm-start (``estep_refine``) window geometry: windows are centered on the
#: previous iteration's top ``_REFINE_TOP_K`` peaks of each (source, template)
#: pair, half-width ``_REFINE_HALF_WIDTH_RAYLEIGH`` Rayleigh widths (1/T, with T
#: the source's observed time span); an argmax landing on a window edge expands
#: that window once by ``_REFINE_EXPAND_FACTOR`` before falling back to the
#: full grid.
_REFINE_TOP_K = 4
_REFINE_HALF_WIDTH_RAYLEIGH = 3.0
_REFINE_EXPAND_FACTOR = 3.0


def _compute_source_sums(source, freqs, mode, H):
    """One source's ``(per_band_sumlists, stats)`` -- iteration-invariant."""
    t, y, bands, dy = source[0], source[1], source[2], source[3]
    return compute_band_summations(t, y, bands, freqs, H, dy=dy,
                                   mode=mode, fast=True)


def _subset_sumlists(sumlists, idx):
    """Per-band sums restricted to grid indices ``idx`` (band order preserved --
    the per-frequency solve accumulates over bands in dict order)."""
    return OrderedDict((b, [sl[i] for i in idx]) for b, sl in sumlists.items())


def _top_peak_indices(idx, powers, top_k):
    """Grid indices of the top-``top_k`` local maxima of ``powers``.

    ``powers`` is sampled at the sorted (possibly non-contiguous) grid positions
    ``idx``; a point is a peak when its power is >= each solved neighbor within
    its contiguous segment (segment endpoints qualify on their single inner
    neighbor, so the global argmax is always a candidate).  Returned sorted by
    power descending."""
    idx = np.asarray(idx)
    powers = np.asarray(powers)
    n = idx.size
    if n == 0:
        return np.empty(0, dtype=int)
    if n == 1:
        return idx.astype(int).copy()
    gap = np.diff(idx) > 1
    ge_left = np.empty(n, dtype=bool)
    ge_left[0] = True
    ge_left[1:] = (powers[1:] >= powers[:-1]) | gap
    ge_right = np.empty(n, dtype=bool)
    ge_right[-1] = True
    ge_right[:-1] = (powers[:-1] >= powers[1:]) | gap
    peaks = np.flatnonzero(ge_left & ge_right)
    order = peaks[np.argsort(-powers[peaks], kind='stable')]
    return idx[order[:top_k]].astype(int)


def _window_mask(freqs, centers, half_width):
    """Boolean grid mask covering ``[c - half_width, c + half_width]`` for each
    center, implicitly clipped to the explicit ``[freqs[0], freqs[-1]]`` band
    (windows entirely outside the grid are dropped)."""
    mask = np.zeros(freqs.size, dtype=bool)
    for fc in centers:
        lo = np.searchsorted(freqs, fc - half_width, side='left')
        hi = np.searchsorted(freqs, fc + half_width, side='right')
        if hi > lo:
            mask[lo:hi] = True
    return mask


def _on_window_edge(mask, g):
    """True when grid index ``g`` sits on the edge of its solved window -- an
    unsolved neighbor on either side.  The physical grid boundaries are NOT
    window edges (there is nothing beyond them to escape into)."""
    if g > 0 and not mask[g - 1]:
        return True
    if g < mask.size - 1 and not mask[g + 1]:
        return True
    return False


def _refine_pair_solve(freqs, T, prev_peaks, prev_best, solve_fn,
                       top_k=_REFINE_TOP_K,
                       half_width_rayleigh=_REFINE_HALF_WIDTH_RAYLEIGH,
                       expand_factor=_REFINE_EXPAND_FACTOR):
    """Warm-started local re-solve for one (source, template) pair.

    Solves only on windows of +- ``half_width_rayleigh`` Rayleigh widths
    (``half_width_rayleigh / T``) around the previous iteration's ``prev_peaks``
    (grid indices) of THIS pair, plus windows at ``2 f`` and ``f / 2`` of the
    previous best ``prev_best`` (grid index), clipped to the explicit grid band.
    If the window argmax lands on a window edge, that window is expanded once
    (``expand_factor`` x) and the new points solved; a second edge landing falls
    back to the full grid for this pair.  ``solve_fn(idx) -> powers`` evaluates
    the periodogram at grid indices ``idx`` (injectable for unit tests).

    Returns ``(best_index, best_power, new_peaks, diag)`` with ``diag`` counting
    ``n_solved`` grid points actually solved (accumulated across the escape /
    fallback stages), ``edge_escape`` and ``fallback`` flags.
    """
    nf = freqs.size
    diag = dict(n_solved=0, edge_escape=0, fallback=0)

    def _full():
        idx = np.arange(nf)
        powers = solve_fn(idx)
        diag['n_solved'] += nf
        i = int(np.argmax(powers))
        return i, float(powers[i]), _top_peak_indices(idx, powers, top_k), diag

    prev_peaks = np.asarray(prev_peaks, dtype=int)
    if not (T > 0) or prev_peaks.size == 0:
        diag['fallback'] = 1
        return _full()

    w = half_width_rayleigh / float(T)
    f_best = float(freqs[int(prev_best)])
    centers = list(freqs[prev_peaks]) + [2.0 * f_best, 0.5 * f_best]
    mask = _window_mask(freqs, centers, w)
    idx = np.flatnonzero(mask)
    if idx.size == 0:                       # unreachable (peaks are grid points)
        diag['fallback'] = 1
        return _full()

    powers = solve_fn(idx)
    diag['n_solved'] += idx.size
    j = int(np.argmax(powers))
    g = int(idx[j])
    if _on_window_edge(mask, g):
        # expand the offending window once (x expand_factor) and re-solve the
        # newly-uncovered points only (the already-solved points are exact)
        diag['edge_escape'] = 1
        mask_new = _window_mask(freqs, [float(freqs[g])], expand_factor * w)
        new_idx = np.flatnonzero(mask_new & ~mask)
        mask = mask | mask_new
        if new_idx.size:
            new_powers = solve_fn(new_idx)
            diag['n_solved'] += new_idx.size
            order = np.argsort(np.concatenate([idx, new_idx]), kind='stable')
            idx = np.concatenate([idx, new_idx])[order]
            powers = np.concatenate([powers, new_powers])[order]
        j = int(np.argmax(powers))
        g = int(idx[j])
        if _on_window_edge(mask, g):
            diag['fallback'] = 1
            return _full()
    return g, float(powers[j]), _top_peak_indices(idx, powers, top_k), diag


def _estep_one_cached(source, sums, templates, freqs, mode, H,
                      prev_state=None, refine=False, top_k=_REFINE_TOP_K,
                      want_state=False):
    """Best ``(assignment, recovered_frequency, fit_quality)`` for one source,
    from precomputed per-band summations ``sums = (sumlists, stats)``.

    With ``refine`` (and a ``prev_state``), each (source, template) pair is
    re-solved only on warm-start windows via :func:`_refine_pair_solve`; the
    default is the exact full-grid solve per template.  ``want_state`` also
    returns the per-template ``(top_peaks, best_index)`` state seeding the next
    refined iteration.  Returns ``(best_k, best_freq, best_power, state, diag)``.
    """
    bands = source[2]
    sumlists, stats = sums
    nf = freqs.size
    track = want_state or refine
    state = [] if track else None
    diag = dict(n_solved=0, windowed_pairs=0, edge_escapes=0, fallbacks=0)

    best_power, best_k, best_i = -np.inf, 0, 0
    for k, tmpl in enumerate(templates):
        template_dict = build_template_set(tmpl, bands)
        if refine and prev_state is not None:
            def _solve(idx, _td=template_dict):
                powers, _ = solve_over_frequencies(
                    _td, _subset_sumlists(sumlists, idx), stats, len(idx),
                    mode=mode)
                return powers
            t = np.asarray(source[0], dtype=float)
            T = float(t.max() - t.min())
            peaks_k, prev_best_k = prev_state[k]
            i, p, new_peaks, pdiag = _refine_pair_solve(
                freqs, T, peaks_k, prev_best_k, _solve, top_k=top_k)
            diag['n_solved'] += pdiag['n_solved']
            diag['windowed_pairs'] += 1
            diag['edge_escapes'] += pdiag['edge_escape']
            diag['fallbacks'] += pdiag['fallback']
        else:
            powers, _ = solve_over_frequencies(template_dict, sumlists, stats,
                                               nf, mode=mode)
            i = int(np.argmax(powers))
            p = float(powers[i])
            diag['n_solved'] += nf
            new_peaks = (_top_peak_indices(np.arange(nf), powers, top_k)
                         if track else None)
        if track:
            state.append((new_peaks, i))
        if p > best_power:
            best_power, best_k, best_i = p, k, i
    return best_k, float(freqs[best_i]), best_power, state, diag


def _estep_one(source, templates, freqs, mode, H):
    """Best ``(assignment, recovered_frequency, fit_quality)`` for one source.

    Computes the per-band NFFT summations once and reuses them across the whole
    bank (template-independent), exactly the ``source_masks`` fast path; the fit
    quality is the periodogram power (explained-variance fraction) at the best
    template's peak frequency.
    """
    sums = _compute_source_sums(source, freqs, mode, H)
    k, f, p, _, _ = _estep_one_cached(source, sums, templates, freqs, mode, H)
    return k, f, p


def _margin_one_cached(source, p_true, sums, templates, freqs, mode, H):
    """:func:`_margin_one` from precomputed ``sums = (sumlists, stats)``."""
    bands = source[2]
    sumlists, stats = sums
    best = None
    for tmpl in templates:
        template_dict = build_template_set(tmpl, bands)
        powers, _ = solve_over_frequencies(template_dict, sumlists, stats,
                                           freqs.size, mode=mode)
        best = powers if best is None else np.maximum(best, powers)
    i_true = int(np.argmin(np.abs(freqs - 1.0 / p_true)))
    return float(best[i_true] - best.max())


def _margin_one(source, p_true, templates, freqs, mode, H):
    """Mean-max periodogram-power margin at the true period for ONE source.

    ``max_k P_k(f_nearest_truth) - max_f max_k P_k(f)`` -- continuous, <= 0,
    exactly 0 when the bank's global peak sits in the truth frequency bin.  The
    per-band summations are computed once and reused across the bank (the
    ``source_masks`` fast path)."""
    sums = _compute_source_sums(source, freqs, mode, H)
    return _margin_one_cached(source, p_true, sums, templates, freqs, mode, H)


# ----------------------------------------------------------------------
# Sums-cache executor: the E-step / val-margin fan-out with per-source
# band summations computed once and reused across EM iterations (FIX: the
# sums depend only on the data and (freqs, mode, H), never on the templates)
# ----------------------------------------------------------------------
_EXEC = {}                       # per-worker fixed data + sums cache


def _exec_init(sources, freqs, mode, H, cache_sums):
    _EXEC.update(sources=sources, freqs=freqs, mode=mode, H=H,
                 cache_sums=bool(cache_sums), sums={})


def _exec_get_sums(state, i):
    """``(sums, cache_hit)`` for source ``i`` against ``state``'s cache."""
    cache = state['sums']
    if state['cache_sums'] and i in cache:
        return cache[i], True
    sums = _compute_source_sums(state['sources'][i], state['freqs'],
                                state['mode'], state['H'])
    if state['cache_sums']:
        cache[i] = sums
    return sums, False


def _exec_task(state, task):
    kind, i, payload = task
    sums, hit = _exec_get_sums(state, i)
    source = state['sources'][i]
    if kind == 'estep':
        templates, prev_state_i, refine, top_k, want_state = payload
        k, f, p, st, diag = _estep_one_cached(
            source, sums, templates, state['freqs'], state['mode'], state['H'],
            prev_state=prev_state_i, refine=refine, top_k=top_k,
            want_state=want_state)
        return k, f, p, st, diag, hit
    if kind == 'margin':
        templates, p_true_i = payload
        m = _margin_one_cached(source, p_true_i, sums, templates,
                               state['freqs'], state['mode'], state['H'])
        return m, hit
    raise ValueError("unknown executor task kind %r" % (kind,))


def _exec_worker(task):
    return _exec_task(_EXEC, task)


class _SumsExecutor(object):
    """Per-population E-step / val-margin runner with cached band summations.

    The per-band NFFT summations of a source are **iteration-invariant** (they
    depend only on ``(t, y, bands, dy)`` and ``(freqs, mode, H)``), so an EM run
    that re-fits the same frozen population every iteration can compute them
    once per source and reuse them -- exactly, to the bit, since caching skips
    a recomputation of the identical value.

    Serial (``n_jobs == 1``): the cache is an in-process dict keyed by source
    index.  Parallel: ONE persistent ``spawn`` pool is created for the whole EM
    run (instead of a fresh pool per iteration) and each worker lazily computes
    and caches the sums of the source indices it processes, so a source's sums
    are computed at most once per worker that touches it.  Sources are
    independent and the map preserves order, so the fan-out is bit-for-bit the
    serial loop.

    Memory: the cached sums cost ~1-2 MB per source per band at ~2000 grid
    frequencies and H=8 (linear in both), held for the executor's lifetime
    (per worker, in parallel mode).  Callers with very large grids or
    populations can disable the cache with ``cache_sums=False`` (identical
    results; the sums are then recomputed per task as before).
    """

    def __init__(self, sources, freqs, mode, H, n_jobs=1, cache_sums=True):
        self.sources = list(sources)
        self.freqs = freqs
        self.n_freq = int(np.asarray(freqs).size)
        self.cache_hits = 0
        self.cache_misses = 0
        n_jobs = int(n_jobs)
        if n_jobs < 0:
            n_jobs = _mp.cpu_count()
        self.n_jobs = max(1, min(n_jobs, len(self.sources)))
        self._pool = None
        if self.n_jobs > 1:
            # BLAS pinned to one thread per worker (no oversubscription); the
            # children inherit the env at spawn, the parent env is restored.
            prev = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
            for k in _BLAS_THREAD_VARS:
                os.environ[k] = '1'
            try:
                ctx = _mp.get_context('spawn')
                self._pool = ctx.Pool(
                    self.n_jobs, initializer=_exec_init,
                    initargs=(self.sources, freqs, mode, H, cache_sums))
            finally:
                for k, v in prev.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        else:
            self._state = dict(sources=self.sources, freqs=freqs, mode=mode,
                               H=H, cache_sums=bool(cache_sums), sums={})

    def _run(self, tasks):
        if self._pool is None:
            return [_exec_task(self._state, t) for t in tasks]
        return self._pool.map(_exec_worker, tasks)

    def _count(self, hits):
        for h in hits:
            if h:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def estep(self, templates, prev_states=None, refine=False,
              top_k=_REFINE_TOP_K, want_state=False):
        """E-step over the population; ``(assign, freq_rec, power, states, diag)``.

        ``prev_states``/``refine`` select the warm-start windowed re-solve per
        (source, template) pair (:func:`_refine_pair_solve`); the default is the
        exact full-grid solve.  ``diag`` aggregates solved-point counts and the
        refine guard counters over the population."""
        templates = list(templates)
        tasks = [('estep', i,
                  (templates,
                   None if prev_states is None else prev_states[i],
                   refine, top_k, want_state))
                 for i in range(len(self.sources))]
        out = self._run(tasks)
        self._count(o[5] for o in out)
        assign = np.array([o[0] for o in out], dtype=int)
        freq_rec = np.array([o[1] for o in out], dtype=float)
        power = np.array([o[2] for o in out], dtype=float)
        states = [o[3] for o in out]
        n_pairs = len(self.sources) * len(templates)
        diag = dict(
            n_pairs=n_pairs,
            solved_points=int(sum(o[4]['n_solved'] for o in out)),
            full_points=int(n_pairs * self.n_freq),
            windowed_pairs=int(sum(o[4]['windowed_pairs'] for o in out)),
            edge_escapes=int(sum(o[4]['edge_escapes'] for o in out)),
            fallbacks=int(sum(o[4]['fallbacks'] for o in out)))
        return assign, freq_rec, power, states, diag

    def margins(self, templates, p_true):
        """Per-source power margins at truth (:func:`_margin_one`), in order."""
        templates = list(templates)
        tasks = [('margin', i, (templates, float(p_true[i])))
                 for i in range(len(self.sources))]
        out = self._run(tasks)
        self._count(o[1] for o in out)
        return [o[0] for o in out]

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def _estep(sources, templates, freqs, mode, H, n_jobs):
    """E-step over the whole population; ``(assign, freq_rec, power)`` arrays.

    Identical computation serial or parallel (sources are independent), so
    fanning over a ``spawn`` pool is bit-for-bit the serial loop.  One-shot
    (no cross-iteration sums cache); EM loops use :class:`_SumsExecutor`."""
    executor = _SumsExecutor(sources, freqs, mode, H, n_jobs=n_jobs,
                             cache_sums=False)
    try:
        assign, freq_rec, power, _, _ = executor.estep(list(templates))
    finally:
        executor.close()
    return assign, freq_rec, power


# ----------------------------------------------------------------------
# M-step: gated, robust, orbit-aligned per-cluster shape re-estimation
# ----------------------------------------------------------------------
def _align_to_template(t, y, bands, dy, freq, template, n_tau=128):
    """Best ``(tau, a, offsets, codes)`` aligning one source to ``template``.

    Fits ``y_b ~ a * template(f t - tau) + offset_b`` over the phase shift ``tau``
    (turns) with the amplitude constrained ``a >= 0`` (the physical, FTP
    positive-amplitude convention), ``a`` and the per-band offsets solved linearly
    at each ``tau`` on a dense grid.  Returns ``None`` if no ``tau`` admits ``a > 0``
    (the source is an anti-template -- excluded from the cluster's shape update).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = t.size
    w = weights(_errs(dy, n))
    labels, codes = _band_index(bands, n)
    n_bands = len(labels)
    phase0 = freq * t                                # f t, in turns
    sw = np.sqrt(w)
    base_w = _offset_columns(codes, n_bands) * sw[:, None]
    yw = y * sw

    best_chi2, best = np.inf, None
    for tau in np.arange(n_tau) / float(n_tau):
        shape = template(phase0 - tau)               # template is 1-periodic in turns
        Xw = np.column_stack([base_w, shape * sw])
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        if beta[-1] <= 0:                            # inverted template; skip
            continue
        resid = yw - Xw.dot(beta)
        chi2 = float(resid.dot(resid))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best = (float(tau), float(beta[-1]), beta[:n_bands].copy(), codes)
    return best


def _shape_design(phi, H):
    """``(n, 2H)`` design ``[cos2pi.phi, sin2pi.phi, cos4pi.phi, sin4pi.phi, ...]``.

    Interleaved so a least-squares solution ``beta`` reads off as ``c = beta[0::2]``,
    ``s = beta[1::2]`` -- the :class:`Template` ``(c_j, s_j)`` coefficients."""
    cols = np.empty((phi.size, 2 * H), dtype=float)
    for j in range(1, H + 1):
        cols[:, 2 * (j - 1)] = np.cos(2.0 * np.pi * j * phi)
        cols[:, 2 * (j - 1) + 1] = np.sin(2.0 * np.pi * j * phi)
    return cols


def _mstep_pooled(sources, assign, freq_rec, gated, vocab, H):
    """Pooled, amplitude-weighted MRA M-step; returns ``(new_vocab, n_updated)``.

    For each cluster the gated members are aligned to the current template (per-source
    ``tau``, ``a``, per-band offset from :func:`_align_to_template`), de-phased into the
    template's frame, and the offset-removed magnitudes from **all** members are pooled
    into a single weighted least-squares fit of one shared Fourier shape.  Scaling each
    member's design block by its amplitude ``a_i`` (and weighting by inverse variance)
    makes the pooled fit the exact joint minimizer of
    ``sum_i sum_p w_ip (y_ip - offset_ib - a_i * shape(phi_ip))^2`` over the shape --
    so the shape is informed by every epoch of every member, robust where each
    individual light curve is too sparse to fit a shape alone.  Clusters with no
    alignable member keep their previous template.
    """
    new_vocab = list(vocab)
    n_updated = 0
    for k in range(len(vocab)):
        members = np.flatnonzero((assign == k) & gated)
        if members.size == 0:
            continue
        Xparts, yparts = [], []
        for idx in members:
            t, y, bands, dy = sources[idx][:4]
            al = _align_to_template(t, y, bands, dy, freq_rec[idx], vocab[k])
            if al is None:
                continue
            tau, a, offsets, codes = al
            n = np.asarray(t).size
            w = weights(_errs(dy, n))
            sw = np.sqrt(w)
            phi = freq_rec[idx] * np.asarray(t, dtype=float) - tau
            X = a * _shape_design(phi, H)                  # scale by member amplitude
            target = np.asarray(y, dtype=float) - offsets[codes]
            Xparts.append(X * sw[:, None])
            yparts.append(target * sw)
        if not Xparts:
            continue
        Xall = np.vstack(Xparts)
        yall = np.concatenate(yparts)
        beta, _, _, _ = np.linalg.lstsq(Xall, yall, rcond=None)
        c, s = beta[0::2], beta[1::2]
        if not np.any(np.abs(c) + np.abs(s) > 0):
            continue
        new_vocab[k] = Template(c.copy(), s.copy())
        n_updated += 1
    return new_vocab, n_updated


def _mstep_median(sources, assign, freq_rec, gated, vocab, H):
    """Per-source-fit + robust-median M-step (the simpler MRA aggregator).

    Each gated member's shape is fit independently (:func:`_fit_source_shape`),
    orbit-aligned to the current medoid, and the cluster is aggregated by the
    per-harmonic complex-coefficient median.  Higher variance than the pooled fit
    when members are sparse, but makes no per-source amplitude/offset assumption;
    kept as a comparison aggregator (``m_step='median'``).
    """
    new_vocab = list(vocab)
    n_updated = 0
    kk = np.arange(1, H + 1)
    for k in range(len(vocab)):
        members = np.flatnonzero((assign == k) & gated)
        if members.size == 0:
            continue
        z_ref = _pad(_complex_coeffs(vocab[k]), H)
        aligned = []
        for idx in members:
            z = _fit_source_shape(sources[idx][0], sources[idx][1],
                                  sources[idx][2], sources[idx][3],
                                  freq_rec[idx], H)
            if z is None:
                continue
            z = _pad(z, H)
            _, delta = _max_cross_correlation(z_ref, z)     # align z to the medoid
            # _max_cross_correlation returns delta with z_k = z_ref_k e^{-2pi i k delta},
            # so undo it with +delta to bring the member onto the medoid's phase frame.
            aligned.append(z * np.exp(1j * 2.0 * np.pi * kk * delta))
        if not aligned:
            continue
        A = np.asarray(aligned)
        z_med = np.median(A.real, axis=0) + 1j * np.median(A.imag, axis=0)
        if not np.any(np.abs(z_med) > 0):
            continue
        new_vocab[k] = _template_from_coeffs(z_med)
        n_updated += 1
    return new_vocab, n_updated


_M_STEPS = {'pooled': _mstep_pooled, 'median': _mstep_median}


def _mstep(sources, assign, freq_rec, gated, vocab, H, m_step):
    try:
        return _M_STEPS[m_step](sources, assign, freq_rec, gated, vocab, H)
    except KeyError:
        raise ValueError("m_step must be 'pooled' or 'median'; got %r" % (m_step,))


# ----------------------------------------------------------------------
# Continuous held-out early-stop signal (WP B6)
# ----------------------------------------------------------------------
def _val_margin(val_scorer, vocab, H, n_jobs, executor=None):
    """Population-mean power margin at truth -- the CONTINUOUS early-stop signal.

    The held-out recovery rate is quantized at ``1/n_sources``, so on small val
    populations most EM iterations tie the init and the best-held-out return
    degenerates to the pipeline vocabulary verbatim.  The mean margin moves with
    every sub-threshold improvement, letting genuinely better vocabularies be
    accepted.  Supervised on the VAL truth only (training stays unsupervised);
    identical serial or parallel.  ``executor`` reuses an existing
    :class:`_SumsExecutor` over the val population (the EM loop's cross-iteration
    sums cache); otherwise a one-shot uncached executor is used."""
    p_true = np.asarray(val_scorer.p_true, dtype=float)
    if executor is not None:
        return float(np.mean(executor.margins(list(vocab), p_true)))
    one_shot = _SumsExecutor(val_scorer._sources, val_scorer.freqs,
                             val_scorer.mode, H, n_jobs=n_jobs,
                             cache_sums=False)
    try:
        out = one_shot.margins(list(vocab), p_true)
    finally:
        one_shot.close()
    return float(np.mean(out))


def _min_pairwise_distance(vocab):
    """Smallest orbit distance between any two templates (``inf`` if K < 2)."""
    if len(vocab) < 2:
        return float('inf')
    D = _orbit_distance_matrix(vocab)
    iu = np.triu_indices(len(vocab), k=1)
    return float(D[iu].min())


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_joint_em_catalog(init_templates, n_clusters, train_scorer, *,
                           val_scorer=None, max_iter=12, patience=3,
                           val_signal='power_margin',
                           fit_quality_quantile=0.5, fit_quality_floor=0.0,
                           min_members=3, diversity_eps=1e-3, m_step='pooled',
                           n_harmonics=None, random_state=None,
                           catalog_kwargs=None, n_jobs=1,
                           cache_sums=True, estep_refine=False,
                           refine_top_k=_REFINE_TOP_K,
                           return_diagnostics=False):
    """Refine a ``K``-template vocabulary jointly against an observed population.

    Initializes from the pipeline vocabulary
    (:func:`~ftperiodogram.catalog_builder.build_template_catalog`, k-medoids) and
    alternates a fast-FTP E-step with a gated/robust orbit-aligned M-step, early
    stopping on a held-out signal (continuous power margin at truth by default,
    see ``val_signal``) and returning the **best held-out** vocabulary
    (non-monotone-safe).  Drop-in interchangeable with a
    ``build_template_catalog`` result.

    Parameters
    ----------
    init_templates : sequence of Template
        Library shapes the pipeline vocabulary is built from (the EM init).
    n_clusters : int
        Vocabulary size ``K``.
    train_scorer : RecoveryScorer
        Frozen training population the templates are re-estimated from.  Only its
        light curves are used (the EM estimates each period); injected truth periods
        are *not* read, so training is unsupervised.  Supplies the search grid,
        ``mode`` and population via its public attributes.
    val_scorer : RecoveryScorer or None
        Held-out population for early stopping.  If ``None``, ``train_scorer`` is
        reused for validation (transductive; honest comparison still holds because
        pipeline and joint are both finally scored on a *separate* eval scorer, but
        a distinct ``val_scorer`` is recommended).
    max_iter : int
        Maximum EM iterations.
    patience : int
        Stop if the held-out signal has not improved for this many iterations.
    val_signal : {'power_margin', 'recovery'}
        Held-out early-stop signal.  ``'power_margin'`` (default) is the
        population-mean periodogram-power margin at the true period --
        CONTINUOUS, so sub-threshold improvements register; the held-out
        ``'recovery'`` rate is quantized at ``1/n_sources`` and on small val
        populations ties the init almost always, returning the pipeline
        vocabulary verbatim.
    fit_quality_quantile : float in [0, 1)
        Relative gate: drop sources whose E-step power is below this quantile of the
        population's powers that iteration (wrong-period fits have low power).
    fit_quality_floor : float
        Absolute power gate applied in addition to the quantile.
    min_members : int
        A cluster with fewer gated members than this keeps its previous template.
    diversity_eps : float
        Anti-collapse threshold: an update bringing two templates within this orbit
        distance is reverted.
    m_step : {'pooled', 'median'}
        Cluster aggregator.  ``'pooled'`` (default) jointly fits one shared shape over
        all gated members' epochs (amplitude-weighted MRA fit; robust in the sparse
        regime); ``'median'`` fits each member's shape and takes the orbit-aligned
        per-harmonic complex-coefficient median.
    n_harmonics : int or None
        Common harmonic order (passed to the pipeline init); ``None`` requires the
        inputs already share an order.
    random_state :
        Seed for the pipeline-init PAM.
    catalog_kwargs : dict or None
        Extra kwargs forwarded to ``build_template_catalog`` for the init.
    n_jobs : int
        Processes for the E-step source loop (1 = serial; <0 = all cores).
    cache_sums : bool
        Cache each source's per-band NFFT summations across EM iterations and
        the val-margin scorer (default).  EXACT -- the sums are iteration- and
        template-invariant, so the cache only skips recomputing identical
        values; results are bit-for-bit those of ``cache_sums=False``.  Memory:
        ~1-2 MB per source per band at ~2000 grid frequencies and H=8 (linear
        in both, held per worker in parallel mode); callers with very large
        grids/populations can set ``False`` to trade the recompute back.
    estep_refine : bool
        GUARDED APPROXIMATION, default off.  From iteration 2 on, each
        (source, template) pair is re-solved only on windows of +-3 Rayleigh
        widths around the previous iteration's top-``refine_top_k`` peaks of
        that pair (plus windows at ``2f`` and ``f/2`` of its previous best),
        with a one-shot window expansion on an edge landing and a full-grid
        fallback on a second (:func:`_refine_pair_solve`).  Iteration 1 is
        always the full grid, and the FINAL E-step is always re-run on the
        full grid (in-loop at ``max_iter``; after an early stop the stopped
        iteration's E-step is re-run full-grid against the same input bank),
        so reported assignments / recovered frequencies / powers never come
        from a stale window.
    refine_top_k : int
        Number of previous-iteration peaks seeding the ``estep_refine`` windows.
    return_diagnostics : bool
        If ``True``, also return an :class:`EMDiagnostics`.

    Returns
    -------
    list of Template, or (list of Template, EMDiagnostics)
    """
    catalog_kwargs = dict(catalog_kwargs or {})
    catalog_kwargs.setdefault('random_state', random_state)
    catalog_kwargs.setdefault('n_harmonics', n_harmonics)
    vocab = list(build_template_catalog(init_templates, n_clusters, method='pam',
                                        **catalog_kwargs))
    H = len(vocab[0].c_n)

    sources = train_scorer._sources
    freqs = train_scorer.freqs
    mode = train_scorer.mode
    val = val_scorer if val_scorer is not None else train_scorer
    n_jobs = int(n_jobs)

    if val_signal not in ('power_margin', 'recovery'):
        raise ValueError("val_signal must be 'power_margin' or 'recovery'; "
                         "got %r" % (val_signal,))

    train_exec = _SumsExecutor(sources, freqs, mode, H, n_jobs=n_jobs,
                               cache_sums=cache_sums)
    val_exec = None
    try:
        if val_signal == 'power_margin':
            # the val population is scored every iteration too -- share the
            # train executor when validating transductively, else its own
            if val is train_scorer:
                val_exec = train_exec
            else:
                val_exec = _SumsExecutor(val._sources, val.freqs, val.mode, H,
                                         n_jobs=n_jobs, cache_sums=cache_sums)

            def _val_signal_fn(v):
                return _val_margin(val, v, H, n_jobs, executor=val_exec)
        else:
            def _val_signal_fn(v):
                return float(val(v))

        best_vocab = [Template(t.c_n.copy(), t.s_n.copy(), t.template_id)
                      for t in vocab]
        best_val = _val_signal_fn(vocab)
        best_iter = 0
        val_hist = [best_val]
        fq_hist, dist_hist, gated_hist, revert_hist = [], [], [], []
        assign_hist, freq_hist, power_hist = [], [], []
        solved_hist, full_hist, winpair_hist = [], [], []
        edge_hist, fb_hist, refine_used = [], [], []
        stop_reason = 'max_iter'
        since_improved = 0
        prev_states = None
        final_rerun = False

        def _record_estep(assign, freq_rec, power, ediag, refined):
            assign_hist.append(assign)
            freq_hist.append(freq_rec)
            power_hist.append(power)
            solved_hist.append(int(ediag['solved_points']))
            full_hist.append(int(ediag['full_points']))
            winpair_hist.append(int(ediag['windowed_pairs']))
            edge_hist.append(int(ediag['edge_escapes']))
            fb_hist.append(int(ediag['fallbacks']))
            refine_used.append(bool(refined))

        last_estep_vocab = vocab
        for it in range(1, max_iter + 1):
            # iteration 1 seeds the windows; the FINAL in-loop iteration
            # (max_iter) is always the exact full grid
            refine_it = (estep_refine and it >= 2 and it < max_iter
                         and prev_states is not None)
            last_estep_vocab = vocab                 # the E-step's input bank
            assign, freq_rec, power, states, ediag = train_exec.estep(
                vocab, prev_states=prev_states if refine_it else None,
                refine=refine_it, top_k=refine_top_k,
                want_state=estep_refine)
            if estep_refine:
                prev_states = states
            _record_estep(assign, freq_rec, power, ediag, refine_it)
            fq_hist.append(float(np.mean(power)))

            thresh = max(float(fit_quality_floor),
                         float(np.quantile(power, fit_quality_quantile)))
            gated = power >= thresh
            # Enforce a per-cluster minimum: ungate clusters too sparse to trust.
            for k in range(len(vocab)):
                in_k = (assign == k) & gated
                if in_k.sum() < min_members:
                    gated = gated & (assign != k)
            gated_hist.append(int(gated.sum()))

            proposed, _ = _mstep(sources, assign, freq_rec, gated, vocab, H,
                                 m_step)

            # Anti-collapse: revert any template whose update collapses a pair.
            n_reverted = 0
            prev_min = _min_pairwise_distance(vocab)
            new_min = _min_pairwise_distance(proposed)
            if new_min < diversity_eps and new_min < prev_min:
                D = _orbit_distance_matrix(proposed)
                np.fill_diagonal(D, np.inf)
                changed = [k for k in range(len(vocab))
                           if not np.allclose(proposed[k].c_n, vocab[k].c_n)
                           or not np.allclose(proposed[k].s_n, vocab[k].s_n)]
                # revert the most-collapsed changed templates until the pair clears
                for k in sorted(changed, key=lambda k: D[k].min()):
                    if _min_pairwise_distance(proposed) >= diversity_eps:
                        break
                    proposed[k] = vocab[k]
                    n_reverted += 1
            revert_hist.append(n_reverted)

            vocab = proposed
            dist_hist.append(_min_pairwise_distance(vocab))

            cur_val = _val_signal_fn(vocab)
            val_hist.append(cur_val)
            if cur_val > best_val + 1e-12:
                best_val = cur_val
                best_iter = it
                best_vocab = [Template(t.c_n.copy(), t.s_n.copy(),
                                       t.template_id) for t in vocab]
                since_improved = 0
            else:
                since_improved += 1
                if since_improved >= patience:
                    stop_reason = 'early_stop'
                    break

        # MANDATORY refine guard: if the loop ended on a windowed E-step (an
        # early stop before max_iter), RE-RUN that final E-step on the exact
        # full grid -- same input bank, no windows -- so the last recorded
        # assignments/freqs/powers never come from a stale window.
        if estep_refine and refine_used and refine_used[-1]:
            assign, freq_rec, power, _, ediag = train_exec.estep(
                last_estep_vocab)
            _record_estep(assign, freq_rec, power, ediag, False)
            final_rerun = True
    finally:
        if val_exec is not None and val_exec is not train_exec:
            val_exec.close()
        train_exec.close()

    if not return_diagnostics:
        return best_vocab

    cache_hits = train_exec.cache_hits
    cache_misses = train_exec.cache_misses
    if val_exec is not None and val_exec is not train_exec:
        cache_hits += val_exec.cache_hits
        cache_misses += val_exec.cache_misses
    estep_diag = dict(
        cache_sums=bool(cache_sums), refine=bool(estep_refine),
        cache_hits=int(cache_hits), cache_misses=int(cache_misses),
        assign_hist=assign_hist, freq_hist=freq_hist, power_hist=power_hist,
        solved_points=solved_hist, full_points=full_hist,
        solve_fraction=[s / f if f else 1.0
                        for s, f in zip(solved_hist, full_hist)],
        windowed_pairs=winpair_hist, edge_escapes=edge_hist,
        fallbacks=fb_hist, refine_used=refine_used,
        final_full_grid_rerun=bool(final_rerun))

    diagnostics = EMDiagnostics(
        n_iter=len(val_hist) - 1, best_iter=best_iter,
        val_signal=np.asarray(val_hist, dtype=float),
        val_signal_name=val_signal,
        mean_fit_quality=np.asarray(fq_hist, dtype=float),
        min_pairwise_distance=np.asarray(dist_hist, dtype=float),
        n_gated=np.asarray(gated_hist, dtype=int),
        n_reverted=np.asarray(revert_hist, dtype=int),
        stop_reason=stop_reason,
        estep=estep_diag)
    return best_vocab, diagnostics
