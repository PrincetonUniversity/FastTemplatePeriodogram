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
  power = explained-variance fraction at the peak).
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
  within ``diversity_eps`` orbit distance is reverted (the offending template keeps
  its previous shape) and the event is logged, so atoms cannot merge.
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
from collections import namedtuple

import numpy as np

from .template import Template
from .utils import weights
from .baselines import _band_index, _offset_columns, _errs
from .catalog_builder import (build_template_catalog, _complex_coeffs, _pad,
                              _max_cross_correlation, _orbit_distance_matrix)
from .multiband import (build_template_set, compute_band_summations,
                        solve_over_frequencies)
from .validation import _parallel_map


#: Per-iteration trace returned by :func:`build_joint_em_catalog` with
#: ``return_diagnostics=True``.  ``val_recovery`` is the held-out recovery curve
#: over iterations (``val_recovery[0]`` is the pipeline-init recovery), ``best_iter``
#: the index whose vocabulary is returned, ``min_pairwise_distance`` the orbit
#: distance of the closest template pair each iteration (collapse monitor),
#: ``n_gated`` the per-iteration gated source count, ``n_reverted`` the per-iteration
#: count of anti-collapse reverts, and ``stop_reason`` why the loop ended.
EMDiagnostics = namedtuple(
    'EMDiagnostics',
    ['n_iter', 'best_iter', 'val_recovery', 'mean_fit_quality',
     'min_pairwise_distance', 'n_gated', 'n_reverted', 'stop_reason'])


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
def _estep_one(source, templates, freqs, mode, H):
    """Best ``(assignment, recovered_frequency, fit_quality)`` for one source.

    Computes the per-band NFFT summations once and reuses them across the whole
    bank (template-independent), exactly the ``source_masks`` fast path; the fit
    quality is the periodogram power (explained-variance fraction) at the best
    template's peak frequency.
    """
    t, y, bands, dy = source[0], source[1], source[2], source[3]
    sumlists, stats = compute_band_summations(t, y, bands, freqs, H, dy=dy,
                                              mode=mode, fast=True)
    best_power, best_k, best_i = -np.inf, 0, 0
    for k, tmpl in enumerate(templates):
        template_dict = build_template_set(tmpl, bands)
        powers, _ = solve_over_frequencies(template_dict, sumlists, stats,
                                           freqs.size, mode=mode)
        i = int(np.argmax(powers))
        if powers[i] > best_power:
            best_power, best_k, best_i = float(powers[i]), k, i
    return best_k, float(freqs[best_i]), best_power


_EM_PAR = {}                                 # per-worker fixed data for the E-step


def _em_par_init(templates, freqs, mode, H, sources):
    _EM_PAR.update(templates=templates, freqs=freqs, mode=mode, H=H,
                   sources=sources)


def _em_par_worker(i):
    return _estep_one(_EM_PAR['sources'][i], _EM_PAR['templates'],
                      _EM_PAR['freqs'], _EM_PAR['mode'], _EM_PAR['H'])


def _estep(sources, templates, freqs, mode, H, n_jobs):
    """E-step over the whole population; ``(assign, freq_rec, power)`` arrays.

    Identical computation serial or parallel (sources are independent), so fanning
    over a ``spawn`` pool is bit-for-bit the serial loop."""
    if n_jobs == 1 or len(sources) <= 1:
        out = [_estep_one(s, templates, freqs, mode, H) for s in sources]
    else:
        out = _parallel_map(n_jobs, len(sources), _em_par_init,
                            (templates, freqs, mode, H, sources), _em_par_worker)
    assign = np.array([o[0] for o in out], dtype=int)
    freq_rec = np.array([o[1] for o in out], dtype=float)
    power = np.array([o[2] for o in out], dtype=float)
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
            aligned.append(z * np.exp(-1j * 2.0 * np.pi * kk * delta))
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
                           fit_quality_quantile=0.5, fit_quality_floor=0.0,
                           min_members=3, diversity_eps=1e-3, m_step='pooled',
                           n_harmonics=None, random_state=None,
                           catalog_kwargs=None, n_jobs=1,
                           return_diagnostics=False):
    """Refine a ``K``-template vocabulary jointly against an observed population.

    Initializes from the pipeline vocabulary
    (:func:`~ftperiodogram.catalog_builder.build_template_catalog`, k-medoids) and
    alternates a fast-FTP E-step with a gated/robust orbit-aligned M-step, early
    stopping on a held-out recovery metric and returning the **best held-out**
    vocabulary (non-monotone-safe).  Drop-in interchangeable with a
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
        Stop if held-out recovery has not improved for this many iterations.
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

    def _val_recovery(v):
        return float(val(v))

    best_vocab = [Template(t.c_n.copy(), t.s_n.copy(), t.template_id) for t in vocab]
    best_val = _val_recovery(vocab)
    best_iter = 0
    val_hist = [best_val]
    fq_hist, dist_hist, gated_hist, revert_hist = [], [], [], []
    stop_reason = 'max_iter'
    since_improved = 0

    for it in range(1, max_iter + 1):
        assign, freq_rec, power = _estep(sources, vocab, freqs, mode, H, n_jobs)
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

        proposed, _ = _mstep(sources, assign, freq_rec, gated, vocab, H, m_step)

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

        cur_val = _val_recovery(vocab)
        val_hist.append(cur_val)
        if cur_val > best_val + 1e-12:
            best_val = cur_val
            best_iter = it
            best_vocab = [Template(t.c_n.copy(), t.s_n.copy(), t.template_id)
                          for t in vocab]
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                stop_reason = 'early_stop'
                break

    if not return_diagnostics:
        return best_vocab

    diagnostics = EMDiagnostics(
        n_iter=len(val_hist) - 1, best_iter=best_iter,
        val_recovery=np.asarray(val_hist, dtype=float),
        mean_fit_quality=np.asarray(fq_hist, dtype=float),
        min_pairwise_distance=np.asarray(dist_hist, dtype=float),
        n_gated=np.asarray(gated_hist, dtype=int),
        n_reverted=np.asarray(revert_hist, dtype=int),
        stop_reason=stop_reason)
    return best_vocab, diagnostics
