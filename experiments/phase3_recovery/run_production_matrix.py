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
        subsample: equal recovery at a large wall-time ratio (the ~10^3x speed claim).

The oracle's RECOVERY equals FTP's by construction (baselines.SesarOracleEstimator
reproduces FTP's analytic optimum to <1e-6; see tests/test_baselines.py), so it is
re-run only for the (iii) timing, not the full sweeps.

Search band is explicit RR-Lyrae [f_min, f_max] (default [1, 5] cyc/day, P in
[0.2, 1.0] d) -- never a Nyquist heuristic.  The per-source loop is embarrassingly
parallel: ``--n-jobs -1`` fans it across all cores (mask-identical to serial).
Results stream to ``--outdir`` per seed (checkpointed) and are aggregated at the end.

    python run_production_matrix.py --smoke                 # local wiring check
    python run_production_matrix.py --n-jobs -1 --outdir output_prod   # full (RunPod)
"""
import argparse
import json
import os
import time

import numpy as np

from ftperiodogram.catalog_builder import (fetch_sesar_templates,
                                           fetch_baeza_villagra_templates)
from ftperiodogram.simulate import SyntheticCadence
from ftperiodogram.validation import (frequency_grid, make_recovery_scorer,
                                       RecoveryScorer, build_template_catalog)
from ftperiodogram.baselines import (GLSEstimator, MHLSEstimator,
                                     MultibandLSEstimator, FTPEstimator,
                                     SesarOracleEstimator)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _ints(text):
    return [int(x) for x in str(text).split(',') if str(x).strip()]


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


def greedy_order_reduced(scorer, templates, max_k, args):
    """Greedy order computed on a REDUCED scorer (source subsample + coarse grid).

    The greedy forward-selection ORDER is a set-cover on per-template recovery and is
    robust to both source count and grid resolution, so computing it on
    ``--greedy-select-sources`` sources at ``--greedy-select-nfreq`` frequencies (then
    scoring the resulting vocabulary prefixes at FULL resolution) cuts the dominant
    ``source_masks`` cost ~20-40x with no measurable change to the order.  Pass
    ``--greedy-select-sources 0`` to disable the reduction (full-resolution greedy).
    """
    if not args.greedy_select_sources or args.greedy_select_sources >= scorer.n_sources:
        sel = scorer
    else:
        coarse = frequency_grid(args.f_min, args.f_max, args.greedy_select_nfreq)
        sel = subsample_scorer(scorer, args.greedy_select_sources, freqs=coarse)
    return greedy_order(sel, templates, max_k)


# ----------------------------------------------------------------------
# Vocabulary curves (PAM vs recovery-driven greedy)
# ----------------------------------------------------------------------
def ftp_rate(scorer, vocab):
    return scorer.score_estimator(FTPEstimator(vocab, mode=scorer.mode))


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


def k_curves(scorer, templates, pam_vocabs, greedy_ord, k_values, baselines):
    """All recovery-vs-K curves on one ``scorer``: FTP(PAM), FTP(greedy prefixes),
    and the K-independent baseline reference levels."""
    pam = [ftp_rate(scorer, pam_vocabs[K]) for K in k_values]
    grd = [ftp_rate(scorer, [templates[i] for i in greedy_ord[:K]]) for K in k_values]
    base = {name: scorer.score_estimator(est) for name, est in baselines.items()}
    return pam, grd, base


# ----------------------------------------------------------------------
# Per-seed run
# ----------------------------------------------------------------------
def run_seed(templates, args, seed, log):
    obs_bands = [b for b in str(args.obs_bands).split(',') if b]
    k_values = _ints(args.k_values)
    n_epochs_values = _ints(args.n_epochs_values)
    max_k = max(k_values)

    freqs = frequency_grid(args.f_min, args.f_max, args.n_freq)
    cadence = SyntheticCadence(
        n_epochs={b: args.dense_master_epochs for b in obs_bands}, bands=obs_bands,
        baseline_days=args.baseline_days, random_state=seed + 1)
    master = make_recovery_scorer(cadence, templates, freqs=freqs,
                                  n_sources=args.n_sources, random_state=seed,
                                  n_jobs=args.n_jobs)
    log("seed %d: frozen %d sources, bands=%s, dense=%d epochs/band, grid=[%.2f,%.2f]x%d"
        % (seed, args.n_sources, obs_bands, args.dense_master_epochs, args.f_min,
           args.f_max, args.n_freq))

    baselines = {
        'gls': GLSEstimator(),
        'mhls': MHLSEstimator(args.mhls_h),
        'mbls': MultibandLSEstimator(args.mbls_h),
    }

    result = {'seed': seed}

    # PAM vocabularies are cadence-independent (they cluster template SHAPES), so
    # build them ONCE per K and reuse across both regimes and the N-sweep.
    pam_vocabs = {K: build_template_catalog(templates, K, method='pam',
                                            random_state=seed) for K in k_values}
    # Greedy is recovery-driven, so its order is per-cadence: one source_masks
    # precompute on the dense master and one on the sparse scorer (each parallel).
    sparse = master.downsample(args.sparse_master_epochs, random_state=seed)
    t0 = time.time()
    greedy_dense = greedy_order_reduced(master, templates, max_k, args)
    greedy_sparse = greedy_order_reduced(sparse, templates, max_k, args)
    log("  greedy orders (dense+sparse, %d sel-src x %d sel-freq) in %.0fs"
        % (args.greedy_select_sources or master.n_sources,
           args.greedy_select_nfreq, time.time() - t0))

    # (i) recovery-vs-K, sparse + dense regimes ------------------------------------
    for regime, sc, order in (('sparse', sparse, greedy_sparse),
                             ('dense', master, greedy_dense)):
        t0 = time.time()
        pam, grd, base = k_curves(sc, templates, pam_vocabs, order, k_values, baselines)
        result['k_sweep_%s' % regime] = {
            'n_epochs': (args.sparse_master_epochs if regime == 'sparse'
                         else args.dense_master_epochs),
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
    greedy_vocab = [templates[i] for i in greedy_dense[:fixed_k]]
    t0 = time.time()
    n_curves = {'ftp_pam': [], 'ftp_greedy': [], 'gls': [], 'mhls': [], 'mbls': []}
    for N in n_epochs_values:
        ds = master.downsample(N, random_state=seed)
        n_curves['ftp_pam'].append(ftp_rate(ds, pam_vocab))
        n_curves['ftp_greedy'].append(ftp_rate(ds, greedy_vocab))
        for name, est in baselines.items():
            n_curves[name].append(ds.score_estimator(est))
    result['n_epochs_sweep'] = {'n_epochs_values': n_epochs_values, 'k': int(fixed_k),
                                **n_curves}
    log("  N-sweep@K=%d %.0fs  FTP=%s  GLS=%s  MHLS=%s"
        % (fixed_k, time.time() - t0, ['%.2f' % v for v in n_curves['ftp_pam']],
           ['%.2f' % v for v in n_curves['gls']],
           ['%.2f' % v for v in n_curves['mhls']]))

    # (iii) cost-vs-accuracy: FTP vs the slow Sesar oracle on a subsample -----------
    if not args.no_cost:
        sub = subsample_scorer(master, args.cost_subsample)
        cost_freqs = frequency_grid(args.f_min, args.f_max, args.cost_n_freq)
        sub.freqs = cost_freqs                       # coarser grid; ratio is grid-stable
        cost_vocab = build_template_catalog(templates, args.cost_k, method='pam',
                                            random_state=seed)
        ftp_est = FTPEstimator(cost_vocab, mode=sub.mode)
        oracle = SesarOracleEstimator(cost_vocab, mode=sub.mode, n_tau=args.oracle_n_tau)
        t0 = time.time(); ftp_rate = sub.score_estimator(ftp_est); ftp_t = time.time() - t0
        t0 = time.time(); orc_rate = sub.score_estimator(oracle); orc_t = time.time() - t0
        result['cost'] = {
            'n_sources': sub.n_sources, 'n_freq': int(args.cost_n_freq),
            'k': int(args.cost_k), 'oracle_n_tau': int(args.oracle_n_tau),
            'ftp_seconds': ftp_t, 'oracle_seconds': orc_t,
            'speedup': (orc_t / ftp_t if ftp_t > 0 else float('inf')),
            'ftp_recovery': ftp_rate, 'oracle_recovery': orc_rate}
        log("  cost@K=%d (%d src, %d freq): FTP %.2fs (rec=%.3f) | oracle %.1fs "
            "(rec=%.3f) | speedup %.0fx"
            % (args.cost_k, sub.n_sources, args.cost_n_freq, ftp_t, ftp_rate,
               orc_t, orc_rate, result['cost']['speedup']))
    return result


# ----------------------------------------------------------------------
# Aggregation across seeds
# ----------------------------------------------------------------------
def _stack_stat(per_seed_lists):
    a = np.asarray(per_seed_lists, dtype=float)
    return {'mean': a.mean(axis=0).tolist(), 'std': a.std(axis=0).tolist()}


def aggregate(seed_results):
    agg = {}
    for regime in ('sparse', 'dense'):
        key = 'k_sweep_%s' % regime
        agg[key] = {
            'k_values': seed_results[0][key]['k_values'],
            'ftp_pam': _stack_stat([r[key]['ftp_pam'] for r in seed_results]),
            'ftp_greedy': _stack_stat([r[key]['ftp_greedy'] for r in seed_results]),
            'baselines': {b: _stack_stat([[r[key]['baselines'][b]] for r in seed_results])
                          for b in seed_results[0][key]['baselines']}}
    ns = 'n_epochs_sweep'
    agg[ns] = {'n_epochs_values': seed_results[0][ns]['n_epochs_values'],
               'k_per_seed': [r['fixed_k'] for r in seed_results]}
    for curve in ('ftp_pam', 'ftp_greedy', 'gls', 'mhls', 'mbls'):
        agg[ns][curve] = _stack_stat([r[ns][curve] for r in seed_results])
    if 'cost' in seed_results[0]:
        agg['cost'] = {
            'speedup': _stack_stat([[r['cost']['speedup']] for r in seed_results]),
            'ftp_recovery': _stack_stat([[r['cost']['ftp_recovery']] for r in seed_results]),
            'oracle_recovery': _stack_stat([[r['cost']['oracle_recovery']] for r in seed_results])}
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
    log("universe: %d templates (H=%d), seeds=%s, n_jobs=%d"
        % (len(templates), len(templates[0].c_n), seeds, args.n_jobs))

    seed_results = []
    for seed in seeds:
        r = run_seed(templates, args, seed, log)
        seed_results.append(r)
        with open(os.path.join(args.outdir, 'seed_%d.json' % seed), 'w') as fh:
            json.dump(r, fh, indent=2)

    results = {
        'config': {k: getattr(args, k) for k in (
            'universe', 'nharmonics', 'obs_bands', 'n_sources', 'f_min', 'f_max',
            'n_freq', 'baseline_days', 'dense_master_epochs', 'sparse_master_epochs',
            'k_values', 'n_epochs_values', 'mhls_h', 'mbls_h', 'cost_subsample',
            'cost_n_freq', 'cost_k', 'oracle_n_tau')},
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
