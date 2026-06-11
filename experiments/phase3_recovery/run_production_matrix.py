#!/usr/bin/env python
"""Phase 3.2 production recovery matrix (Paper-1 period-recovery deliverables).

One frozen, seeded synthetic multiband population per seed; every method scored on
the IDENTICAL sources and explicit ``[f_min, f_max]`` grid through the pluggable
estimator seam (:meth:`RecoveryScorer.score_estimator`).  Produces, per seed and
aggregated across seeds:

  (i)   recovery-vs-K, sparse + dense regimes -- FTP via PAM and via recovery-driven
        greedy (candidate_pool='all'), with the GLS / MHLS / multiband-LS baselines
        overlaid as K-independent reference lines;
  (ii)  recovery-vs-N_epochs at the K-sweep knee -- FTP(PAM), FTP(greedy), and the
        three LS baselines as curves vs per-band epoch count (the sparse-regime story);
  (iii) cost-vs-accuracy -- FTP vs the slow Sesar-style non-linear oracle on a source
        subsample.  FTP matches the oracle's recovery exactly, at a measured ~2.5x
        wall-time advantage in-harness (oracle n_tau=128; ~5x at the documented
        n_tau=256).  The ratio is grid-stable and grows with N_obs (~5x @60 obs ->
        ~96x @3840 measured); literature per-star pipelines (Sesar 2017, ~30 min/star)
        are 10^2-10^3x slower for external reasons.  Never print ~10^3x as a measured
        result.

The oracle's RECOVERY equals FTP's by construction (baselines.SesarOracleEstimator
reproduces FTP's analytic optimum to <1e-6; see tests/test_baselines.py), so it is
re-run only for the (iii) timing, not the full sweeps.

Search band is explicit RR-Lyrae [f_min, f_max] (default [1, 5] cyc/day, P in
[0.2, 1.0] d) -- never a Nyquist heuristic.  The per-source loop is embarrassingly
parallel: ``--n-jobs -1`` fans it across all cores (mask-identical to serial).
Results stream to ``--outdir`` per seed (checkpointed) and are aggregated at the end.

Robustness arms (each a separate run, recorded in the results.json config block):
``--library-holdout-frac`` injects truth from a held-out shape split so the
vocabulary never contains the generating shapes; ``--truth-universe`` crosses
universes (truth from one, library from ``--universe``); ``--band-amp-ratio``
breaks the equal-band-amplitude assumption (e.g. 1.4 for g/r).  Greedy selection
always runs on a DISJOINT simulated source split (selection seed != eval seed).
Every scored (seed, method, cell) also persists per-source ``(P_true, P_rec,
recovered)`` as one npz under ``--outdir/per_source/`` -- any recovery criterion is
re-scorable post hoc -- aggregation quotes pooled Wilson 95% CIs (not +/-std of a
few seed means), and the N-sweep records exact McNemar paired contrasts vs FTP(PAM).

    python run_production_matrix.py --smoke                 # local wiring check
    python run_production_matrix.py --n-jobs -1 --outdir output_prod   # full (RunPod)
"""
import argparse
import copy
import json
import os
import time

import numpy as np

from ftperiodogram.catalog_builder import (fetch_sesar_templates,
                                           fetch_baeza_villagra_templates)
from ftperiodogram.simulate import SyntheticCadence, exp_mag_error
from ftperiodogram.validation import (frequency_grid, make_recovery_scorer,
                                       RecoveryScorer, build_template_catalog,
                                       wilson_interval, mcnemar_test)
from ftperiodogram.baselines import (GLSEstimator, MHLSEstimator,
                                     MultibandLSEstimator, FTPEstimator,
                                     SesarOracleEstimator)
import ztf_error_model as zerr             # sibling experiment module


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _ints(text):
    return [int(x) for x in str(text).split(',') if str(x).strip()]


def _build_err_model(args, log):
    """Return ``(mag->sigma callable or None, label)`` for the cadence noise.

    ``synthetic`` (default) returns ``None`` so the cadence uses ``exp_mag_error``;
    ``empirical`` loads the binned-median ZTF error-vs-mag curve from the cached
    real cadence sample (1.6-3.5x larger than the synthetic default), falling back
    to synthetic with a warning if the cache is absent."""
    if getattr(args, 'err_model', 'synthetic') != 'empirical':
        return None, 'synthetic(exp_mag_error)'
    try:
        emp = zerr.make_empirical_error_model()
        log("  empirical ZTF error model: %d-bin median, faint sigma=%.3f"
            % (len(emp.curve[0]), emp.faint_sigma))
        return emp, 'empirical_ztf(binned-median)'
    except (FileNotFoundError, ValueError) as exc:
        log("  WARNING empirical error model unavailable (%s); using synthetic"
            % str(exc)[:80])
        return None, 'synthetic(exp_mag_error)[fallback]'


def knee_k(k_values, recovery, frac=0.98):
    """Smallest K reaching ``frac`` of the best observed recovery."""
    recovery = np.asarray(recovery, dtype=float)
    rmax = float(recovery.max()) if recovery.size else 0.0
    if rmax <= 0.0:
        return int(k_values[0])
    for k, r in zip(k_values, recovery):
        if r >= frac * rmax:
            return int(k)
    return int(k_values[-1])


def subsample_scorer(master, n, freqs=None):
    """A sibling scorer over the first ``n`` frozen sources (optionally on a coarser
    ``freqs`` grid) -- used both for the cost panel and the greedy-selection reduction.
    """
    n = min(int(n), master.n_sources)
    return RecoveryScorer._from_sources(
        master._sources[:n], freqs=(master.freqs if freqs is None else freqs),
        mode=master.mode, criterion=master.criterion,
        harmonic_aware=master.harmonic_aware, delta_phi_max=master.delta_phi_max,
        rtol=master.rtol, n_jobs=master.n_jobs)


# Greedy SELECTION sources are simulated with this seed offset so they are
# disjoint from the evaluation population (selection seed != eval seed) -- the
# selected vocabulary is never scored on the sources that chose it.
_SEL_SEED_OFFSET = 100003


def selection_scorers(truth_templates, args, seed, obs_bands, err_model,
                      band_amplitudes):
    """Dense + sparse scorers for greedy selection, DISJOINT from evaluation.

    The greedy forward-selection ORDER is a set-cover on per-template recovery and
    is robust to source count and grid resolution, so it is computed on
    ``--greedy-select-sources`` freshly simulated sources at
    ``--greedy-select-nfreq`` frequencies (``--greedy-select-sources 0`` =
    ``--n-sources`` sources on the full grid), then the vocabulary prefixes are
    scored on the untouched evaluation population at full resolution.
    """
    n_sel = int(args.greedy_select_sources) or int(args.n_sources)
    nf = (int(args.greedy_select_nfreq) if args.greedy_select_sources
          else int(args.n_freq))
    freqs = frequency_grid(args.f_min, args.f_max, nf)
    cad = SyntheticCadence(
        n_epochs={b: args.dense_master_epochs for b in obs_bands}, bands=obs_bands,
        baseline_days=args.baseline_days, err_model=err_model,
        random_state=seed + _SEL_SEED_OFFSET + 1)
    dense = make_recovery_scorer(cad, truth_templates, freqs=freqs,
                                 n_sources=n_sel,
                                 band_amplitudes=band_amplitudes,
                                 random_state=seed + _SEL_SEED_OFFSET,
                                 n_jobs=args.n_jobs)
    sparse = dense.downsample(args.sparse_master_epochs,
                              random_state=seed + _SEL_SEED_OFFSET)
    return dense, sparse


def band_amplitude_dict(obs_bands, ratio):
    """Per-band amplitude scalings spanning ``ratio`` from first to last band.

    ``None`` (the shared-shape default) when ratio is 1/unset or there is a single
    band; otherwise log-spaced from ``ratio`` down to 1.0 across the band list,
    e.g. ratio 1.4 with bands g,r -> {'g': 1.4, 'r': 1.0}.
    """
    ratio = float(ratio or 1.0)
    if ratio == 1.0 or len(obs_bands) < 2:
        return None
    if ratio <= 0:
        raise SystemExit("--band-amp-ratio must be positive; got %r" % ratio)
    n = len(obs_bands)
    return {b: float(ratio ** (1.0 - i / (n - 1.0)))
            for i, b in enumerate(obs_bands)}


def split_truth_library(templates, truth_pool, args, seed):
    """Resolve the (truth population, vocabulary library) pair for one seed.

    Cross-universe (``--truth-universe``): truth = the other universe, library =
    the ``--universe`` shapes.  Holdout (``--library-holdout-frac``): a per-seed
    disjoint split of the single universe, mirroring run_joint_vs_pipeline.py --
    the library never contains the generating shapes.  Default: truth == library.
    Returns ``(truth, library, arms_metadata_dict)``.
    """
    arms = {'truth_universe': args.truth_universe,
            'library_holdout_frac': float(args.library_holdout_frac),
            'band_amp_ratio': float(args.band_amp_ratio)}
    if truth_pool is not None:
        truth, library = truth_pool, templates
    elif args.library_holdout_frac > 0:
        idx = np.random.RandomState(seed).permutation(len(templates))
        n_pop = max(1, int(round(args.library_holdout_frac * len(templates))))
        if n_pop >= len(templates):
            raise SystemExit("--library-holdout-frac %.2f leaves an empty library "
                             "(%d templates)" % (args.library_holdout_frac,
                                                 len(templates)))
        truth = [templates[i] for i in idx[:n_pop]]
        library = [templates[i] for i in idx[n_pop:]]
        arms['holdout_truth_idx'] = [int(i) for i in idx[:n_pop]]
        arms['library_idx'] = [int(i) for i in idx[n_pop:]]
    else:
        truth = library = templates
    arms['n_truth'], arms['n_library'] = len(truth), len(library)
    return truth, library, arms


# ----------------------------------------------------------------------
# Per-source persistence (WP B1): every scored (seed, method, cell) writes one
# npz of (P_true, P_rec, recovered) so any recovery criterion -- fractional,
# phase-coherence, alias breakdown -- is re-scorable post hoc, and so paired
# (McNemar) contrasts and pooled Wilson CIs use real counts, not seed means.
# ----------------------------------------------------------------------
def make_saver(scorer, outdir, seed, cell_prefix):
    """Bind a frozen ``scorer`` + output location; the returned ``save`` scores an
    estimator, persists per-source arrays for ``(seed, method, cell)``, and hands
    back ``(rate, mask)`` so callers can pair masks across methods."""
    def save(method, cell_suffix, estimator):
        cell = '%s-%s' % (cell_prefix, cell_suffix)
        rate, mask, p_rec = scorer.score_estimator(estimator, return_periods=True)
        d = os.path.join(outdir, 'per_source')
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            os.path.join(d, 'seed%d__%s__%s.npz' % (seed, method, cell)),
            p_true=np.asarray(scorer.p_true, dtype=float), p_rec=p_rec,
            recovered=mask, rate=rate, seed=seed, method=method, cell=cell)
        return rate, mask
    return save


def ftp_est(scorer, vocab):
    return FTPEstimator(vocab, mode=scorer.mode)


# ----------------------------------------------------------------------
# Vocabulary curves (PAM vs recovery-driven greedy)
# ----------------------------------------------------------------------


def greedy_order(scorer, templates, max_k):
    """Recovery-driven greedy forward-selection ORDER (set-cover on the per-template
    standalone recovered masks), computed with ONE (parallel) ``source_masks``
    precompute and reused for every K -- the prefix of length K is the size-K greedy
    vocabulary on this cadence.

    Mirrors ``catalog_builder._greedy_select`` (union of standalone masks is the
    selection signal; ties broken by lowest template index) without re-running the
    expensive precompute per K.  The trailing pad (templates that add no new
    recoveries, ordered by index) only exists so prefixes of every requested K are
    defined; saturation before ``max_k`` is itself the recovery-vs-K knee.
    """
    masks = np.asarray(scorer.source_masks(templates), dtype=bool)  # (n_templ, n_src)
    covered = np.zeros(scorer.n_sources, dtype=bool)
    order, remaining = [], list(range(len(templates)))
    while len(order) < max_k and remaining and not covered.all():
        rem = np.array(remaining)
        gains = masks[rem][:, ~covered].sum(axis=1)
        if gains.max() == 0:
            break
        best = int(rem[int(np.lexsort((rem, -gains))[0])])
        order.append(best)
        covered |= masks[best]
        remaining.remove(best)
    for r in remaining:
        if len(order) >= max_k:
            break
        order.append(int(r))
    return order


def k_curves(scorer, templates, pam_vocabs, greedy_ord, k_values, baselines, save):
    """All recovery-vs-K curves on one ``scorer``: FTP(PAM), FTP(greedy prefixes),
    and the K-independent baseline reference levels.  Every point is persisted
    per-source through ``save`` (one npz per method and K; baselines once)."""
    pam = [save('ftp_pam', 'K%d' % K, ftp_est(scorer, pam_vocabs[K]))[0]
           for K in k_values]
    grd = [save('ftp_greedy', 'K%d' % K,
                ftp_est(scorer, [templates[i] for i in greedy_ord[:K]]))[0]
           for K in k_values]
    base = {name: save(name, 'ref', est)[0] for name, est in baselines.items()}
    return pam, grd, base


# ----------------------------------------------------------------------
# Per-seed run
# ----------------------------------------------------------------------
def run_seed(templates, truth_pool, args, seed, log):
    obs_bands = [b for b in str(args.obs_bands).split(',') if b]
    k_values = _ints(args.k_values)
    n_epochs_values = _ints(args.n_epochs_values)
    max_k = max(k_values)

    truth, library, arms = split_truth_library(templates, truth_pool, args, seed)
    band_amps = band_amplitude_dict(obs_bands, args.band_amp_ratio)
    arms['band_amplitudes'] = band_amps

    freqs = frequency_grid(args.f_min, args.f_max, args.n_freq)
    err_model, err_label = _build_err_model(args, log)
    cadence = SyntheticCadence(
        n_epochs={b: args.dense_master_epochs for b in obs_bands}, bands=obs_bands,
        baseline_days=args.baseline_days, err_model=err_model, random_state=seed + 1)
    master = make_recovery_scorer(cadence, truth, freqs=freqs,
                                  n_sources=args.n_sources,
                                  band_amplitudes=band_amps, random_state=seed,
                                  n_jobs=args.n_jobs)
    log("seed %d: frozen %d sources, bands=%s, dense=%d epochs/band, grid=[%.2f,%.2f]x%d"
        % (seed, args.n_sources, obs_bands, args.dense_master_epochs, args.f_min,
           args.f_max, args.n_freq))
    if (arms['truth_universe'] or arms['library_holdout_frac'] > 0
            or band_amps is not None):
        log("  arms: truth=%d shapes, library=%d shapes, truth_universe=%s, "
            "holdout=%.2f, band_amplitudes=%s"
            % (arms['n_truth'], arms['n_library'], arms['truth_universe'],
               arms['library_holdout_frac'], band_amps))

    baselines = {
        'gls': GLSEstimator(),
        'mhls': MHLSEstimator(args.mhls_h),
        'mbls': MultibandLSEstimator(args.mbls_h),
    }

    result = {'seed': seed, 'err_model': err_label, 'arms': arms}

    # PAM vocabularies are cadence-independent (they cluster template SHAPES), so
    # build them ONCE per K and reuse across both regimes and the N-sweep.
    # Vocabularies always come from the LIBRARY split (== truth unless an arm
    # separates them).
    pam_vocabs = {K: build_template_catalog(library, K, method='pam',
                                            random_state=seed) for K in k_values}
    # Greedy is recovery-driven, so its order is per-cadence: one source_masks
    # precompute on the dense selection scorer and one on its sparse downsample
    # (each parallel), on a DISJOINT simulated split (selection seed != eval seed).
    sparse = master.downsample(args.sparse_master_epochs, random_state=seed)
    t0 = time.time()
    sel_dense, sel_sparse = selection_scorers(truth, args, seed, obs_bands,
                                              err_model, band_amps)
    greedy_dense = greedy_order(sel_dense, library, max_k)
    greedy_sparse = greedy_order(sel_sparse, library, max_k)
    log("  greedy orders (dense+sparse, %d disjoint sel-src x %d sel-freq) in %.0fs"
        % (sel_dense.n_sources, sel_dense.freqs.size, time.time() - t0))

    # (i) recovery-vs-K, sparse + dense regimes ------------------------------------
    for regime, sc, order in (('sparse', sparse, greedy_sparse),
                             ('dense', master, greedy_dense)):
        t0 = time.time()
        save = make_saver(sc, args.outdir, seed, 'ksweep-%s' % regime)
        pam, grd, base = k_curves(sc, library, pam_vocabs, order, k_values,
                                  baselines, save)
        result['k_sweep_%s' % regime] = {
            'n_epochs': (args.sparse_master_epochs if regime == 'sparse'
                         else args.dense_master_epochs),
            'n_sources': sc.n_sources,
            'k_values': k_values, 'ftp_pam': pam, 'ftp_greedy': grd,
            'greedy_order': order, 'baselines': base}
        log("  K-sweep[%s] %.0fs  PAM=%s  greedy=%s  GLS=%.3f MHLS=%.3f MBLS=%.3f"
            % (regime, time.time() - t0, ['%.2f' % v for v in pam],
               ['%.2f' % v for v in grd], base['gls'], base['mhls'], base['mbls']))

    # knee from the sparse K-sweep PAM curve (the discriminating one)
    fixed_k = args.fixed_k or knee_k(k_values, result['k_sweep_sparse']['ftp_pam'])
    result['fixed_k'] = int(fixed_k)

    # (ii) recovery-vs-N_epochs at fixed K -----------------------------------------
    # FTP(PAM) and FTP(greedy) vocabularies fixed at the knee K (greedy learned on the
    # dense master), then evaluated across down-sampled epoch counts -- the sparse-
    # regime story. No per-N re-selection: the SAME vocab is scored at each N.
    pam_vocab = pam_vocabs[fixed_k]
    greedy_vocab = [library[i] for i in greedy_dense[:fixed_k]]
    t0 = time.time()
    n_curves = {'ftp_pam': [], 'ftp_greedy': [], 'gls': [], 'mhls': [], 'mbls': []}
    contrasts = []
    for N in n_epochs_values:
        ds = master.downsample(N, random_state=seed)
        save = make_saver(ds, args.outdir, seed, 'nsweep')
        masks = {}
        for method, est in (('ftp_pam', ftp_est(ds, pam_vocab)),
                            ('ftp_greedy', ftp_est(ds, greedy_vocab)),
                            *baselines.items()):
            rate, masks[method] = save(method, 'N%d' % N, est)
            n_curves[method].append(rate)
        # paired method contrasts on the identical sources (exact McNemar)
        for other in ('ftp_greedy', 'gls', 'mhls', 'mbls'):
            a_only, b_only, p = mcnemar_test(masks['ftp_pam'], masks[other])
            contrasts.append({'n_epochs': int(N), 'a': 'ftp_pam', 'b': other,
                              'a_only': a_only, 'b_only': b_only, 'p': p})
    result['n_epochs_sweep'] = {'n_epochs_values': n_epochs_values, 'k': int(fixed_k),
                                'n_sources': master.n_sources,
                                'mcnemar_vs_ftp_pam': contrasts, **n_curves}
    log("  N-sweep@K=%d %.0fs  FTP=%s  GLS=%s  MHLS=%s"
        % (fixed_k, time.time() - t0, ['%.2f' % v for v in n_curves['ftp_pam']],
           ['%.2f' % v for v in n_curves['gls']],
           ['%.2f' % v for v in n_curves['mhls']]))

    # (iii) cost-vs-accuracy: FTP vs the slow Sesar oracle on a subsample -----------
    if not args.no_cost:
        sub = subsample_scorer(master, args.cost_subsample)
        cost_freqs = frequency_grid(args.f_min, args.f_max, args.cost_n_freq)
        sub.freqs = cost_freqs                       # coarser grid; ratio is grid-stable
        cost_vocab = build_template_catalog(library, args.cost_k, method='pam',
                                            random_state=seed)
        save = make_saver(sub, args.outdir, seed, 'cost')
        fast = FTPEstimator(cost_vocab, mode=sub.mode)
        oracle = SesarOracleEstimator(cost_vocab, mode=sub.mode, n_tau=args.oracle_n_tau)
        t0 = time.time(); ftp_rec = save('ftp', 'K%d' % args.cost_k, fast)[0]; ftp_t = time.time() - t0
        t0 = time.time(); orc_rate = save('oracle', 'K%d' % args.cost_k, oracle)[0]; orc_t = time.time() - t0
        result['cost'] = {
            'n_sources': sub.n_sources, 'n_freq': int(args.cost_n_freq),
            'k': int(args.cost_k), 'oracle_n_tau': int(args.oracle_n_tau),
            'ftp_seconds': ftp_t, 'oracle_seconds': orc_t,
            'speedup': (orc_t / ftp_t if ftp_t > 0 else float('inf')),
            'ftp_recovery': ftp_rec, 'oracle_recovery': orc_rate}
        log("  cost@K=%d (%d src, %d freq): FTP %.2fs (rec=%.3f) | oracle %.1fs "
            "(rec=%.3f) | speedup %.0fx"
            % (args.cost_k, sub.n_sources, args.cost_n_freq, ftp_t, ftp_rec,
               orc_t, orc_rate, result['cost']['speedup']))
    return result


# ----------------------------------------------------------------------
# Aggregation across seeds.  Recovery curves get pooled Wilson 95% CIs (the
# sources are the independent trials; seeds only relabel them -- pool the counts,
# never average per-seed intervals or quote a +/-std of 2-3 seed means).  Only
# the timing ratio, which is not a proportion, keeps mean/std.
# ----------------------------------------------------------------------
def _stack_stat(per_seed_lists):
    a = np.asarray(per_seed_lists, dtype=float)
    return {'mean': a.mean(axis=0).tolist(), 'std': a.std(axis=0).tolist()}


def _pooled_wilson(per_seed_rates, n_per_seed):
    """Pool per-seed recovery rates back into counts; Wilson 95% CI per point.

    ``per_seed_rates`` is (n_seeds, n_points); each seed's rate was an exact
    count/n, so ``rint(rate*n)`` recovers the integer successes losslessly."""
    a = np.atleast_2d(np.asarray(per_seed_rates, dtype=float))
    n = np.asarray(n_per_seed, dtype=float)
    k = np.rint(a * n[:, None]).sum(axis=0)
    n_tot = float(n.sum())
    lo, hi = wilson_interval(k, n_tot)
    return {'mean': (k / n_tot).tolist(), 'lo': np.atleast_1d(lo).tolist(),
            'hi': np.atleast_1d(hi).tolist(), 'n_pooled': int(n_tot)}


def _block_n(seed_results, key, n_sources):
    """Per-seed source counts for one result block (with legacy-json fallback)."""
    ns = [r[key].get('n_sources', n_sources) for r in seed_results]
    if any(v is None for v in ns):
        raise ValueError("legacy per-seed results lack %r n_sources; pass "
                         "aggregate(..., n_sources=<count>)" % key)
    return ns


def aggregate(seed_results, n_sources=None):
    agg = {}
    for regime in ('sparse', 'dense'):
        key = 'k_sweep_%s' % regime
        ns_seed = _block_n(seed_results, key, n_sources)
        agg[key] = {
            'k_values': seed_results[0][key]['k_values'],
            'ftp_pam': _pooled_wilson([r[key]['ftp_pam'] for r in seed_results],
                                      ns_seed),
            'ftp_greedy': _pooled_wilson([r[key]['ftp_greedy'] for r in seed_results],
                                         ns_seed),
            'baselines': {b: _pooled_wilson([[r[key]['baselines'][b]]
                                             for r in seed_results], ns_seed)
                          for b in seed_results[0][key]['baselines']}}
    ns = 'n_epochs_sweep'
    ns_seed = _block_n(seed_results, ns, n_sources)
    agg[ns] = {'n_epochs_values': seed_results[0][ns]['n_epochs_values'],
               'k_per_seed': [r['fixed_k'] for r in seed_results]}
    for curve in ('ftp_pam', 'ftp_greedy', 'gls', 'mhls', 'mbls'):
        agg[ns][curve] = _pooled_wilson([r[ns][curve] for r in seed_results], ns_seed)
    if 'cost' in seed_results[0]:
        nc = [r['cost']['n_sources'] for r in seed_results]
        agg['cost'] = {
            'speedup': _stack_stat([[r['cost']['speedup']] for r in seed_results]),
            'ftp_recovery': _pooled_wilson(
                [[r['cost']['ftp_recovery']] for r in seed_results], nc),
            'oracle_recovery': _pooled_wilson(
                [[r['cost']['oracle_recovery']] for r in seed_results], nc)}
    return agg


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--universe', default='sesar',
                   help="'sesar' (98 ugriz, headline) or 'bv' (Baeza-Villagra 2025 "
                        "griz, robustness arm)")
    p.add_argument('--bv-bands', default='g,r',
                   help="DECam bands to keep for --universe bv (one band ~ 280 "
                        "shapes; comma-separated)")
    p.add_argument('--nharmonics', type=int, default=8)
    p.add_argument('--obs-bands', default='g,r')
    p.add_argument('--n-sources', type=int, default=1024)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--f-min', type=float, default=1.0)    # cyc/day (P=1.0 d)
    p.add_argument('--f-max', type=float, default=5.0)    # cyc/day (P=0.2 d)
    p.add_argument('--n-freq', type=int, default=15000)
    p.add_argument('--baseline-days', type=float, default=3 * 365.25)
    p.add_argument('--dense-master-epochs', type=int, default=60)
    p.add_argument('--sparse-master-epochs', type=int, default=6)
    p.add_argument('--k-values', default='1,2,4,8,16')
    p.add_argument('--n-epochs-values', default='4,8,12,16,24,40')
    p.add_argument('--fixed-k', type=int, default=None)
    p.add_argument('--mhls-h', type=int, default=8)
    p.add_argument('--mbls-h', type=int, default=1)
    p.add_argument('--greedy-select-sources', type=int, default=256,
                   help="sources for the greedy-order source_masks (0 = full); the "
                        "order is robust to subsampling, the curves score at full scale")
    p.add_argument('--greedy-select-nfreq', type=int, default=4000,
                   help="coarse grid for the greedy-order source_masks precompute")
    p.add_argument('--cost-subsample', type=int, default=48)
    p.add_argument('--cost-n-freq', type=int, default=2000)
    p.add_argument('--cost-k', type=int, default=2)
    p.add_argument('--oracle-n-tau', type=int, default=128)
    p.add_argument('--no-cost', action='store_true')
    p.add_argument('--max-templates', type=int, default=None)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), 'output_prod'))
    p.add_argument('--no-figures', action='store_true')
    p.add_argument('--err-model', choices=['synthetic', 'empirical'],
                   default='synthetic',
                   help="cadence noise: synthetic exp_mag_error (default) or the "
                        "empirical ZTF binned-median model from the cached real "
                        "cadence sample (1.6-3.5x larger errors)")
    p.add_argument('--library-holdout-frac', type=float, default=0.0,
                   help="robustness arm: per-seed fraction of the universe held "
                        "out as the truth POPULATION; the vocabulary library is "
                        "the disjoint remainder (0 = truth == library)")
    p.add_argument('--truth-universe', default=None, choices=['sesar', 'bv'],
                   help="robustness arm: simulate truth from THIS universe while "
                        "the vocabulary library comes from --universe "
                        "(cross-universe mismatch)")
    p.add_argument('--band-amp-ratio', type=float, default=1.0,
                   help="robustness arm: first-to-last band amplitude ratio "
                        "(e.g. 1.4 for g/r); 1.0 = shared shape across bands")
    p.add_argument('--smoke', action='store_true')
    return p.parse_args(argv)


def apply_smoke(args):
    args.nharmonics = 4
    args.n_sources = 16
    args.seeds = '0,1'
    args.n_freq = 1200
    args.baseline_days = 365.25
    args.dense_master_epochs = 30
    args.sparse_master_epochs = 6
    args.k_values = '1,2,4'
    args.n_epochs_values = '4,8,16'
    args.max_templates = 24
    args.greedy_select_sources = 8
    args.greedy_select_nfreq = 400
    args.cost_subsample = 6
    args.cost_n_freq = 500
    args.cost_k = 2
    args.oracle_n_tau = 64
    args.outdir = os.path.join(os.path.dirname(__file__), 'output_prod_smoke')
    return args


def load_universe(args):
    """Sesar 2010 ugriz (headline) or Baeza-Villagra 2025 griz (robustness arm)."""
    if args.universe == 'sesar':
        templates = fetch_sesar_templates(nharmonics=args.nharmonics)
    elif args.universe in ('bv', 'baeza_villagra'):
        bv_bands = [b for b in str(args.bv_bands).split(',') if b] or None
        templates = fetch_baeza_villagra_templates(
            bands=bv_bands, nharmonics=args.nharmonics)
    else:
        raise SystemExit("unknown --universe %r; expected 'sesar' or 'bv'"
                         % (args.universe,))
    if args.max_templates is not None:
        templates = templates[:args.max_templates]
    return templates


def load_truth_universe(args, log):
    """Templates for the ``--truth-universe`` cross-universe arm (None if unset).

    Mutually exclusive with ``--library-holdout-frac`` (the two arms answer
    different mismatch questions; composing them muddles both)."""
    if not args.truth_universe or args.truth_universe == args.universe:
        if args.truth_universe:
            log("--truth-universe == --universe; cross-universe arm disabled")
        return None
    if args.library_holdout_frac > 0:
        raise SystemExit("--truth-universe and --library-holdout-frac are "
                         "mutually exclusive; run them as separate arms")
    other = copy.copy(args)
    other.universe = args.truth_universe
    return load_universe(other)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args = apply_smoke(args)
    os.makedirs(args.outdir, exist_ok=True)
    seeds = _ints(args.seeds)
    t_start = time.time()

    def log(msg):
        print(msg, flush=True)

    templates = load_universe(args)
    truth_pool = load_truth_universe(args, log)
    log("universe: %d templates (H=%d), seeds=%s, n_jobs=%d"
        % (len(templates), len(templates[0].c_n), seeds, args.n_jobs))
    if truth_pool is not None:
        log("cross-universe arm: truth from %r (%d shapes), library from %r"
            % (args.truth_universe, len(truth_pool), args.universe))

    seed_results = []
    for seed in seeds:
        r = run_seed(templates, truth_pool, args, seed, log)
        seed_results.append(r)
        with open(os.path.join(args.outdir, 'seed_%d.json' % seed), 'w') as fh:
            json.dump(r, fh, indent=2)

    results = {
        'config': {k: getattr(args, k) for k in (
            'universe', 'nharmonics', 'obs_bands', 'n_sources', 'f_min', 'f_max',
            'n_freq', 'baseline_days', 'dense_master_epochs', 'sparse_master_epochs',
            'k_values', 'n_epochs_values', 'mhls_h', 'mbls_h', 'cost_subsample',
            'cost_n_freq', 'cost_k', 'oracle_n_tau', 'err_model',
            'library_holdout_frac', 'truth_universe', 'band_amp_ratio',
            'greedy_select_sources', 'greedy_select_nfreq')},
        'n_universe': len(templates), 'seeds': seeds,
        'per_seed': seed_results, 'aggregate': aggregate(seed_results),
        'wall_seconds': round(time.time() - t_start, 1)}
    with open(os.path.join(args.outdir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    log("wrote %s  (%.0fs total)"
        % (os.path.join(args.outdir, 'results.json'), time.time() - t_start))

    if not args.no_figures:
        try:
            import matplotlib
            matplotlib.use('Agg')
            from production_figures import make_figures
        except Exception as exc:                          # pragma: no cover
            log("figures skipped (%s)" % exc)
        else:
            make_figures(results, args.outdir)
            log("wrote figures to %s" % args.outdir)
    return results


if __name__ == '__main__':
    main()
