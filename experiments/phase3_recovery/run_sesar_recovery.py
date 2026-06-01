#!/usr/bin/env python
"""Phase 3.2 first recovery experiment: learned Sesar vocabulary, recovery sweeps.

Pipeline (PLAN 3.2 deliverables i + ii):

    fetch_sesar_templates  ->  PAM-learned vocabulary (recovery-vs-K)
                           ->  N_epochs stratification at the K-sweep knee (fixed K)

Both sweeps score period recovery against ONE frozen, seeded synthetic multiband
population (the FTP@H=1 single-cosine == GLS baseline included as the reference).
The headline criterion is the hard 1% fractional rule (rtol=0.01); the search grid is
an explicit [f_min, f_max] band (RR Lyrae periods 0.2-1.0 d), never a Nyquist
heuristic.  Results (numbers + optional draft figures) are written under --outdir.

This is a fresh, self-contained driver (numpy + ftperiodogram core; matplotlib only
if available).  Run a tiny smoke first, then the bounded default:

    python run_sesar_recovery.py --smoke
    python run_sesar_recovery.py            # bounded ugriz default
"""
import argparse
import json
import os
import time

import numpy as np

from ftperiodogram.catalog_builder import fetch_sesar_templates
from ftperiodogram.simulate import SyntheticCadence
from ftperiodogram.validation import (frequency_grid, make_recovery_scorer,
                                       k_sweep_recovery, n_epochs_sweep_recovery)


def _ints(text):
    return [int(x) for x in str(text).split(',') if str(x).strip()]


def knee_k(k_values, recovery, frac=0.98):
    """Smallest K whose recovery reaches ``frac`` of the best observed recovery."""
    recovery = np.asarray(recovery, dtype=float)
    rmax = float(recovery.max()) if recovery.size else 0.0
    if rmax <= 0.0:
        return int(k_values[0])
    for k, r in zip(k_values, recovery):
        if r >= frac * rmax:
            return int(k)
    return int(k_values[-1])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--obs-bands', default='g,r',
                   help="observed photometric bands of the synthetic cadence")
    p.add_argument('--nharmonics', type=int, default=8)
    p.add_argument('--n-sources', type=int, default=128)
    p.add_argument('--n-master-epochs', type=int, default=80,
                   help="per-band epochs of the master (densest) population")
    p.add_argument('--k-values', default='1,2,4,8,16')
    p.add_argument('--n-epochs-values', default='5,10,20,40,80')
    p.add_argument('--fixed-k', type=int, default=None,
                   help="K for the N_epochs sweep (default: K-sweep knee)")
    p.add_argument('--f-min', type=float, default=1.0)      # cycles/day (P=1.0 d)
    p.add_argument('--f-max', type=float, default=5.0)      # cycles/day (P=0.2 d)
    p.add_argument('--n-freq', type=int, default=10000)
    p.add_argument('--baseline-days', type=float, default=3 * 365.25)
    p.add_argument('--max-templates', type=int, default=None,
                   help="cap the universe (for smoke runs)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-jobs', type=int, default=1,
                   help="processes for the per-source loop (-1 = all cores)")
    p.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), 'output'))
    p.add_argument('--no-figures', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help="tiny end-to-end run to validate wiring")
    return p.parse_args(argv)


def apply_smoke(args):
    args.n_sources = 8
    args.n_master_epochs = 20
    args.k_values = '1,2,4'
    args.n_epochs_values = '5,10,20'
    args.n_freq = 800
    args.max_templates = 24
    args.baseline_days = 365.25
    args.outdir = os.path.join(os.path.dirname(__file__), 'output_smoke')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args = apply_smoke(args)
    obs_bands = [b for b in str(args.obs_bands).split(',') if b]
    k_values = _ints(args.k_values)
    n_epochs_values = _ints(args.n_epochs_values)
    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()

    # 1. universe of Sesar shapes (full ugriz unless capped) ----------------------
    templates = fetch_sesar_templates(nharmonics=args.nharmonics)
    if args.max_templates is not None:
        templates = templates[:args.max_templates]
    print("universe: %d Sesar templates (H=%d)"
          % (len(templates), len(templates[0].c_n)))

    # 2. explicit RR-Lyrae search band + frozen synthetic population --------------
    freqs = frequency_grid(args.f_min, args.f_max, args.n_freq)
    cadence = SyntheticCadence(
        n_epochs={b: args.n_master_epochs for b in obs_bands}, bands=obs_bands,
        baseline_days=args.baseline_days, random_state=args.seed + 1)
    master = make_recovery_scorer(cadence, templates, freqs=freqs,
                                  n_sources=args.n_sources, random_state=args.seed,
                                  n_jobs=args.n_jobs)
    print("frozen population: %d sources, bands=%s, %d master epochs/band, "
          "grid=[%.3f, %.3f] x %d, n_jobs=%d"
          % (args.n_sources, obs_bands, args.n_master_epochs, args.f_min,
             args.f_max, args.n_freq, args.n_jobs))

    # 3. recovery-vs-K (deliverable i) -------------------------------------------
    t0 = time.time()
    ksweep = k_sweep_recovery(templates, cadence, k_values=k_values, scorer=master,
                              random_state=args.seed)
    print("K-sweep %ss in %.1fs:" % (list(ksweep.k_values), time.time() - t0))
    for k, r in zip(ksweep.k_values, ksweep.recovery):
        print("  K=%2d  recovery=%.3f" % (k, r))
    print("  baseline FTP@H=1 (GLS): %.3f" % ksweep.baseline_recovery)

    fixed_k = args.fixed_k or knee_k(ksweep.k_values, ksweep.recovery)
    print("fixed K for N_epochs sweep: %d (%s)"
          % (fixed_k, 'requested' if args.fixed_k else 'knee'))

    # 4. recovery-vs-N_epochs at fixed K (deliverable ii) ------------------------
    t0 = time.time()
    nsweep = n_epochs_sweep_recovery(master, templates, n_epochs_values, k=fixed_k,
                                     random_state=args.seed)
    print("N_epochs-sweep %ss in %.1fs:"
          % (list(nsweep.n_epochs_values), time.time() - t0))
    for n, r, b in zip(nsweep.n_epochs_values, nsweep.recovery,
                       nsweep.baseline_recovery):
        print("  N=%3d  FTP@K=%d=%.3f   GLS=%.3f" % (n, fixed_k, r, b))

    # 5. persist numbers ----------------------------------------------------------
    results = {
        'config': {
            'obs_bands': obs_bands, 'nharmonics': args.nharmonics,
            'n_sources': args.n_sources, 'n_master_epochs': args.n_master_epochs,
            'n_universe': len(templates), 'f_min': args.f_min, 'f_max': args.f_max,
            'n_freq': args.n_freq, 'baseline_days': args.baseline_days,
            'seed': args.seed, 'criterion': ksweep.criterion},
        'k_sweep': {
            'k_values': ksweep.k_values.tolist(),
            'recovery': ksweep.recovery.tolist(),
            'baseline_recovery': float(ksweep.baseline_recovery)},
        'fixed_k': int(fixed_k),
        'n_epochs_sweep': {
            'n_epochs_values': nsweep.n_epochs_values.tolist(),
            'recovery': nsweep.recovery.tolist(),
            'baseline_recovery': nsweep.baseline_recovery.tolist()},
        'wall_seconds': round(time.time() - t_start, 1)}
    json_path = os.path.join(args.outdir, 'results.json')
    with open(json_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    npz_path = os.path.join(args.outdir, 'results.npz')
    np.savez(npz_path, k_values=ksweep.k_values, k_recovery=ksweep.recovery,
             k_baseline=ksweep.baseline_recovery,
             n_epochs_values=nsweep.n_epochs_values, n_recovery=nsweep.recovery,
             n_baseline=nsweep.baseline_recovery, fixed_k=fixed_k)
    print("wrote %s and %s" % (json_path, npz_path))

    # 6. optional draft figures ---------------------------------------------------
    if not args.no_figures:
        try:
            import matplotlib
            matplotlib.use('Agg')
            from ftperiodogram import figures
        except ImportError:
            print("matplotlib unavailable; skipping figures")
        else:
            figdir = os.path.join(args.outdir, 'figures')
            os.makedirs(figdir, exist_ok=True)
            fig, _ = figures.plot_recovery_panels(ksweep, nsweep)
            panel_path = os.path.join(figdir, 'recovery_panels.pdf')
            fig.savefig(panel_path, bbox_inches='tight')
            ax_k = figures.plot_recovery_vs_k(ksweep)
            ax_k.figure.savefig(os.path.join(figdir, 'recovery_vs_k.pdf'),
                                bbox_inches='tight')
            ax_n = figures.plot_recovery_vs_nepochs(nsweep)
            ax_n.figure.savefig(os.path.join(figdir, 'recovery_vs_nepochs.pdf'),
                                bbox_inches='tight')
            print("wrote figures to %s" % figdir)

    print("done in %.1fs" % (time.time() - t_start))
    return results


if __name__ == '__main__':
    main()
