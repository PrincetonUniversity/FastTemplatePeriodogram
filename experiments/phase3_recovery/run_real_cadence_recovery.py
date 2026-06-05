#!/usr/bin/env python
"""Track B draft: recovery comparison on REAL ZTF cadence vs the synthetic cadence.

Drops a *real* ZTF DR observing cadence (cached by ``fetch_ztf_cadence.py``) into
the existing Phase-3.2 recovery harness and re-runs the recovery sweeps on the
real cadence vs the synthetic seasonal cadence we've been using, scoring the SAME
frozen sources/seeds so the ONLY difference is the sampling pattern (and,
optionally, the error-vs-mag model).

For both cadences and on one explicit ``[f_min, f_max]`` RR-Lyrae search band
(never a Nyquist heuristic -- irregular sampling), it runs:

* ``k_sweep_recovery``        -- recovery vs vocabulary size K (FTP PAM catalog),
                                 with the FTP@H=1 single-cosine == GLS lower bound;
* ``n_epochs_sweep_recovery`` -- recovery vs per-band epoch count at a fixed K;
* the comparison baselines GLS / MHLS / multiband-LS at the densest population,
  scored through the same per-source seam (``scorer.score_estimator``).

The real and synthetic cadences are matched on bands and (median) baseline and use
the *same* empirical ZTF error-vs-mag model, so a real-vs-synthetic recovery gap is
attributable to the real sampling structure (seasonal gaps + clumping), which is
exactly the sparse/clumpy regime where FTP's learned shape prior should help most.

LOCAL-SCALE draft (a few dozen sources, 1-2 seeds, a modest grid): a wiring +
sanity check, not the publication run.  Fresh, self-contained (numpy + ftperiodogram
core + the experiments-only ztf_error_model); matplotlib only if available.

    python run_real_cadence_recovery.py --smoke
    python run_real_cadence_recovery.py            # bounded real-vs-synthetic draft
"""
import argparse
import json
import os
import time

import numpy as np

from ftperiodogram.catalog_builder import fetch_sesar_templates, build_template_catalog
from ftperiodogram.simulate import SyntheticCadence, RealZTFCadence, exp_mag_error
from ftperiodogram.baselines import (GLSEstimator, MHLSEstimator,
                                     MultibandLSEstimator, FTPEstimator)
from ftperiodogram.validation import (frequency_grid, make_recovery_scorer,
                                      k_sweep_recovery, n_epochs_sweep_recovery)

# experiments-only empirical error model (sibling module)
import ztf_error_model as zerr


DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ftperiodogram_data",
                                 "ztf_cadence_sample")


def _ints(text):
    return [int(x) for x in str(text).split(',') if str(x).strip()]


def knee_k(k_values, recovery, frac=0.98):
    recovery = np.asarray(recovery, dtype=float)
    rmax = float(recovery.max()) if recovery.size else 0.0
    if rmax <= 0.0:
        return int(k_values[0])
    for k, r in zip(k_values, recovery):
        if r >= frac * rmax:
            return int(k)
    return int(k_values[-1])


# ----------------------------------------------------------------------
# Cadence selection
# ----------------------------------------------------------------------
def pick_real_cadences(cache_dir, obs_bands, err_model, *, n_cadences=1,
                       catflags_max=0, min_per_band=120, max_epochs_per_band=None):
    """Load up to ``n_cadences`` cached real ZTF objects that cover ``obs_bands``.

    Returns a list of (gid, RealZTFCadence).  Skips objects missing a requested
    band or too sparse in any band (so the N_epochs downsample has headroom).
    ``max_epochs_per_band`` evenly thins each loaded cadence (full baseline /
    seasonal structure preserved) so a dense ~1000-epoch full-survey ZTF light
    curve scores in seconds, not minutes -- the draft's tractability lever."""
    import glob
    cads = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "g*.npz"))):
        gid = os.path.basename(path)[:-4]
        # Eligibility on the RAW (uncapped) per-band counts, so the cap never
        # disqualifies an otherwise well-sampled object.
        try:
            raw = RealZTFCadence.from_cache(gid, err_model=err_model,
                                            bands=obs_bands, data_home=cache_dir,
                                            catflags_max=catflags_max)
        except (ValueError, FileNotFoundError):
            continue
        rc = raw.epoch_counts()
        if not (set(obs_bands).issubset(rc) and min(rc.values()) >= min_per_band):
            continue
        cad = RealZTFCadence.from_cache(
            gid, err_model=err_model, bands=obs_bands, data_home=cache_dir,
            catflags_max=catflags_max, max_epochs_per_band=max_epochs_per_band)
        cads.append((gid, cad))
        if len(cads) >= n_cadences:
            break
    return cads


def matched_synthetic_cadence(real_cad, obs_bands, err_model, *, n_master_epochs,
                              random_state):
    """A SyntheticCadence matched to the real one's baseline + bands, same error
    model -- so a real-vs-synthetic gap is the *sampling structure*, not T/noise."""
    return SyntheticCadence(
        n_epochs={b: n_master_epochs for b in obs_bands}, bands=obs_bands,
        baseline_days=real_cad.baseline, err_model=err_model,
        random_state=random_state)


# ----------------------------------------------------------------------
# One cadence's full sweep set
# ----------------------------------------------------------------------
def run_one_cadence(label, cadence, templates, args, freqs, seed):
    """K-sweep + N_epochs-sweep + comparison baselines for one cadence/seed."""
    t0 = time.time()
    master = make_recovery_scorer(
        cadence, templates, freqs=freqs, n_sources=args.n_sources,
        random_state=seed, n_jobs=args.n_jobs)

    ksweep = k_sweep_recovery(templates, cadence, k_values=_ints(args.k_values),
                              scorer=master, random_state=seed)
    fixed_k = args.fixed_k or knee_k(ksweep.k_values, ksweep.recovery)

    nsweep = n_epochs_sweep_recovery(
        master, templates, _ints(args.n_epochs_values), k=fixed_k,
        random_state=seed)

    # Comparison baselines at the densest (master) population, same seam.
    vocab_fixed = build_template_catalog(templates, fixed_k, method='pam',
                                         random_state=seed)
    comparisons = {
        'FTP_PAM_k%d' % fixed_k: master.score_estimator(
            FTPEstimator(vocab_fixed, mode=master.mode)),
        'GLS': master.score_estimator(GLSEstimator()),
        'MHLS_H8': master.score_estimator(MHLSEstimator(n_harmonics=8)),
        'MultibandLS_H1': master.score_estimator(MultibandLSEstimator(n_harmonics=1)),
    }

    return dict(
        label=label, seed=int(seed), fixed_k=int(fixed_k),
        baseline_days=float(cadence.baseline),
        epoch_counts=(cadence.epoch_counts()
                      if hasattr(cadence, 'epoch_counts') else None),
        k_sweep=dict(k_values=ksweep.k_values.tolist(),
                     recovery=ksweep.recovery.tolist(),
                     gls_baseline=float(ksweep.baseline_recovery)),
        n_epochs_sweep=dict(n_epochs_values=nsweep.n_epochs_values.tolist(),
                            recovery=nsweep.recovery.tolist(),
                            gls_baseline=nsweep.baseline_recovery.tolist()),
        comparisons=comparisons,
        wall_seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--obs-bands', default='g,r')
    p.add_argument('--nharmonics', type=int, default=8)
    p.add_argument('--n-sources', type=int, default=48)
    p.add_argument('--seeds', default='0,1')
    p.add_argument('--n-master-epochs', type=int, default=120,
                   help="per-band epochs of the synthetic master population")
    p.add_argument('--k-values', default='1,2,4,8')
    p.add_argument('--n-epochs-values', default='5,10,20,40,80')
    p.add_argument('--fixed-k', type=int, default=None)
    p.add_argument('--f-min', type=float, default=1.0)      # cycles/day (P=1.0 d)
    p.add_argument('--f-max', type=float, default=5.0)      # cycles/day (P=0.2 d)
    p.add_argument('--n-freq', type=int, default=6000)
    p.add_argument('--max-templates', type=int, default=None)
    p.add_argument('--n-real-cadences', type=int, default=1,
                   help="distinct cached real objects to use as cadences (per seed)")
    p.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    p.add_argument('--catflags-max', type=int, default=0)
    p.add_argument('--min-per-band', type=int, default=100)
    p.add_argument('--real-epochs-cap', type=int, default=None,
                   help="evenly thin each real cadence to <= this many epochs/band "
                        "(default: n_master_epochs, so real and synthetic are "
                        "epoch-matched and the dense real LC scores fast)")
    p.add_argument('--err-model', choices=['empirical', 'exp'], default='empirical',
                   help="error-vs-mag model shared by both cadences")
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--outdir',
                   default=os.path.join(os.path.dirname(__file__),
                                        'real_cadence_draft'))
    p.add_argument('--smoke', action='store_true')
    return p.parse_args(argv)


def apply_smoke(args):
    args.n_sources = 8
    args.seeds = '0'
    args.n_master_epochs = 60
    args.k_values = '1,2,4'
    args.n_epochs_values = '5,10,20'
    args.n_freq = 800
    args.max_templates = 24
    args.min_per_band = 40
    args.outdir = os.path.join(os.path.dirname(__file__), 'real_cadence_smoke')
    return args


def build_err_model(args, obs_bands):
    if args.err_model == 'exp':
        return exp_mag_error(), 'exp_mag_error()'
    try:
        emp = zerr.make_empirical_error_model(args.cache_dir,
                                              catflags_max=args.catflags_max)
        return emp, 'empirical_ztf(binned-median)'
    except (FileNotFoundError, ValueError) as exc:
        print("  empirical error model unavailable (%s); falling back to "
              "exp_mag_error" % str(exc)[:60])
        return exp_mag_error(), 'exp_mag_error()[fallback]'


def write_summary_md(path, payload):
    cfg = payload['config']
    lines = []
    lines.append("# Real ZTF cadence vs synthetic cadence -- recovery draft\n")
    lines.append("Generated by `run_real_cadence_recovery.py` "
                 "(Track B local-scale draft).\n")
    lines.append("\n## Config\n")
    lines.append("- bands: `%s`  |  sources: %d  |  seeds: %s\n"
                 % (cfg['obs_bands'], cfg['n_sources'], cfg['seeds']))
    lines.append("- universe: %d Sesar templates (H=%d)\n"
                 % (cfg['n_universe'], cfg['nharmonics']))
    lines.append("- grid: [%.3f, %.3f] cyc/d x %d (RR-Lyrae band; explicit "
                 "f_min/f_max, no Nyquist)\n"
                 % (cfg['f_min'], cfg['f_max'], cfg['n_freq']))
    lines.append("- error model (both cadences): %s\n" % cfg['err_model'])
    lines.append("- real cadence object(s): %s\n" % ", ".join(cfg['real_gids']))

    for seed_block in payload['results']:
        seed = seed_block['seed']
        real = seed_block['real']
        synth = seed_block['synthetic']
        lines.append("\n## seed %d\n" % seed)
        lines.append("real baseline %.0f d, epochs/band %s ; "
                     "synthetic baseline %.0f d, %d epochs/band\n"
                     % (real['baseline_days'], real['epoch_counts'],
                        synth['baseline_days'], cfg['n_master_epochs']))

        lines.append("\n### Recovery vs K (FTP PAM catalog)\n")
        lines.append("| K | real FTP | synth FTP | real GLS | synth GLS |\n")
        lines.append("|---|---|---|---|---|\n")
        for i, k in enumerate(real['k_sweep']['k_values']):
            lines.append("| %d | %.3f | %.3f | %.3f | %.3f |\n" % (
                k, real['k_sweep']['recovery'][i],
                synth['k_sweep']['recovery'][i],
                real['k_sweep']['gls_baseline'],
                synth['k_sweep']['gls_baseline']))

        lines.append("\n### Recovery vs N_epochs (fixed K real=%d synth=%d)\n"
                     % (real['fixed_k'], synth['fixed_k']))
        lines.append("| N/band | real FTP | synth FTP | real GLS | synth GLS |\n")
        lines.append("|---|---|---|---|---|\n")
        rn, sn = real['n_epochs_sweep'], synth['n_epochs_sweep']
        for i, n in enumerate(rn['n_epochs_values']):
            lines.append("| %d | %.3f | %.3f | %.3f | %.3f |\n" % (
                n, rn['recovery'][i], sn['recovery'][i],
                rn['gls_baseline'][i], sn['gls_baseline'][i]))

        lines.append("\n### Method comparison at densest population\n")
        lines.append("| method | real | synthetic |\n|---|---|---|\n")
        for m in real['comparisons']:
            lines.append("| %s | %.3f | %.3f |\n" % (
                m, real['comparisons'][m], synth['comparisons'].get(m, float('nan'))))
    with open(path, 'w') as fh:
        fh.write("".join(lines))


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args = apply_smoke(args)
    obs_bands = [b for b in str(args.obs_bands).split(',') if b]
    seeds = _ints(args.seeds)
    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()

    # 1. Sesar universe + shared error model ----------------------------------
    templates = fetch_sesar_templates(nharmonics=args.nharmonics)
    if args.max_templates is not None:
        templates = templates[:args.max_templates]
    print("universe: %d Sesar templates (H=%d)"
          % (len(templates), len(templates[0].c_n)))

    err_model, err_label = build_err_model(args, obs_bands)
    print("error-vs-mag model: %s" % err_label)

    # 2. real cadence(s) + matched synthetic cadence --------------------------
    # Cap the dense full-survey real LC to the synthetic master count by default,
    # so real vs synthetic are epoch-matched (the gap is sampling *structure*, not
    # epoch count) and the dense LC scores fast.
    epochs_cap = (args.n_master_epochs if args.real_epochs_cap is None
                  else args.real_epochs_cap)
    real_cads = pick_real_cadences(
        args.cache_dir, obs_bands, err_model, n_cadences=args.n_real_cadences,
        catflags_max=args.catflags_max, min_per_band=args.min_per_band,
        max_epochs_per_band=epochs_cap)
    print("real cadence epochs/band capped at %d (even thinning, full baseline kept)"
          % epochs_cap)
    if not real_cads:
        print("NO usable cached real ZTF cadence covering bands %s with >=%d "
              "epochs/band in %s. Run fetch_ztf_cadence.py first. Aborting."
              % (obs_bands, args.min_per_band, args.cache_dir))
        return 1
    real_gids = [gid for gid, _ in real_cads]
    print("real cadence object(s): %s" % real_gids)
    for gid, cad in real_cads:
        print("  %s baseline=%.0f d epochs/band=%s"
              % (gid, cad.baseline, cad.epoch_counts()))

    freqs = frequency_grid(args.f_min, args.f_max, args.n_freq)

    # 3. run real vs synthetic on the SAME sources/seeds ----------------------
    results = []
    for seed in seeds:
        gid, real_cad = real_cads[seed % len(real_cads)]
        synth_cad = matched_synthetic_cadence(
            real_cad, obs_bands, err_model,
            n_master_epochs=args.n_master_epochs, random_state=seed + 101)
        print("\n[seed %d] real=%s (T=%.0f d) vs synthetic (T=%.0f d, %d ep/band)"
              % (seed, gid, real_cad.baseline, synth_cad.baseline,
                 args.n_master_epochs))

        real_res = run_one_cadence('real:%s' % gid, real_cad, templates, args,
                                   freqs, seed)
        synth_res = run_one_cadence('synthetic', synth_cad, templates, args,
                                    freqs, seed)
        print("  real    K-sweep=%s GLS=%.3f"
              % ([round(x, 3) for x in real_res['k_sweep']['recovery']],
                 real_res['k_sweep']['gls_baseline']))
        print("  synth   K-sweep=%s GLS=%.3f"
              % ([round(x, 3) for x in synth_res['k_sweep']['recovery']],
                 synth_res['k_sweep']['gls_baseline']))
        print("  real    comparisons=%s"
              % {k: round(v, 3) for k, v in real_res['comparisons'].items()})
        print("  synth   comparisons=%s"
              % {k: round(v, 3) for k, v in synth_res['comparisons'].items()})
        results.append(dict(seed=int(seed), real_gid=gid, real=real_res,
                            synthetic=synth_res))

    # 4. persist --------------------------------------------------------------
    payload = dict(
        config=dict(
            obs_bands=','.join(obs_bands), nharmonics=args.nharmonics,
            n_sources=args.n_sources, seeds=args.seeds,
            n_master_epochs=args.n_master_epochs, n_universe=len(templates),
            f_min=args.f_min, f_max=args.f_max, n_freq=args.n_freq,
            err_model=err_label, real_gids=real_gids,
            cache_dir=args.cache_dir),
        results=results, wall_seconds=round(time.time() - t_start, 1))

    json_path = os.path.join(args.outdir, 'results.json')
    tmp = json_path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, json_path)
    write_summary_md(os.path.join(args.outdir, 'SUMMARY.md'), payload)
    print("\nwrote %s and SUMMARY.md (%.1fs total)"
          % (json_path, payload['wall_seconds']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
