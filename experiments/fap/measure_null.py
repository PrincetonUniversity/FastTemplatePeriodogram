#!/usr/bin/env python
"""WP D1 -- measured null / false-alarm-probability (FAP) calibration.

Fresh, self-contained (imports ONLY the ftperiodogram package, never any
sibling script under experiments/phase3_recovery/).  It:

  1. simulates PURE-NOISE single-band light curves at the *production* synthetic
     cadence + synthetic error model (SyntheticCadence, exp_mag_error defaults;
     3-yr baseline, yearly 270/365.25-d seasons, mean_mag=15, amplitude=0 ->
     homoscedastic sigma ~= exp_mag_error(15) ~ 0.010 mag);
  2. runs the FTP template periodogram on the *scan* path (the default since WP
     C4) at harmonic orders H in {1,3,6,12}, and a GLS baseline, over the
     explicit production RR-Lyrae grid [f_min,f_max]=[1,5] cyc/day, n_freq=10000
     (>= B7 grid-guard floor 8767; matches the rerun_202606 10k grid);
  3. records the MAX power over the grid per realization (the null max-power
     distribution), fits a Generalized-Extreme-Value (GEV) tail to each, and
     writes FAP-vs-threshold curves;
  4. builds a null-max-vs-K catalog curve: the catalog-max statistic max over
     K PAM-vocabulary templates AND frequencies, for K in {1,2,4,8}, using
     nested cumulative maxima over the order-8 PAM medoids (so K is a pure
     extra-trials axis -- exactly the null the paper's K-sweep wobble needs).

The FTP null uses a single *representative* order-H template per H: the global
PAM medoid (K=1) of the Sesar (2010) RRL library re-truncated to order H.  For
a pure-noise null the specific shape is second order (the null max-power law is
governed by the model DOF 2H+1, the grid size, and the cadence), but using a
real RRL medoid keeps it faithful to the production template family.

Every number written to SUMMARY.md is computed here from the run's own data.

Usage:
  measure_null.py --smoke                 # small canary (timing + sanity anchor)
  measure_null.py --n-real 2000           # full run
"""
import argparse
import json
import os
import platform
import time
import warnings
from multiprocessing import Pool

import numpy as np

from ftperiodogram.core import template_periodogram
from ftperiodogram.baselines import GLSEstimator
from ftperiodogram.template import Template
from ftperiodogram.catalog_builder import (fetch_sesar_templates,
                                           build_template_catalog)
from ftperiodogram.simulate import SyntheticCadence, simulate_lightcurve
from ftperiodogram.validation import frequency_grid

# ---------------------------------------------------------------------------
# Worker globals (set once per process by _init_worker; picklable via initargs).
# ---------------------------------------------------------------------------
_G = {}


def _init_worker(cfg, ftp_coeffs, vocab_coeffs, freqs):
    """Populate per-process globals: config, template coeff arrays, grid."""
    _G['cfg'] = cfg
    # ftp_coeffs: {H: (c_n, s_n)} representative order-H templates
    _G['ftp_coeffs'] = {int(H): (np.asarray(c), np.asarray(s))
                        for H, (c, s) in ftp_coeffs.items()}
    # vocab_coeffs: list of (c_n, s_n) order-vocab_h PAM medoids, length max(K)
    _G['vocab_coeffs'] = [(np.asarray(c), np.asarray(s)) for c, s in vocab_coeffs]
    _G['freqs'] = np.asarray(freqs)
    _G['gls'] = GLSEstimator()
    # amplitude=0 kills the template, but simulate_lightcurve still calls it:
    _G['dummy'] = Template(np.array([1.0]), np.array([0.0]))
    warnings.simplefilter('ignore')  # pure-noise |MM|~0 dips / conditioning notes


def _max_power(t, y, dy, c_n, s_n, freqs):
    p, _ = template_periodogram(t, y, dy, c_n, s_n, freqs)  # method='scan' default
    return float(np.max(p))


def run_one(idx):
    """One pure-noise realization -> dict of null max-power scalars."""
    cfg = _G['cfg']
    freqs = _G['freqs']
    seed = cfg['seed_base'] + idx
    cad = SyntheticCadence(n_epochs=cfg['n_obs'], bands=None,
                           baseline_days=cfg['baseline_days'],
                           season_length_days=cfg['season_length_days'],
                           season_period_days=cfg['season_period_days'],
                           err_model=None, random_state=seed)
    lc = simulate_lightcurve(_G['dummy'], 1.0, cad, amplitude=0.0,
                             mean_mag=cfg['mean_mag'], random_state=seed)
    t, y, dy = lc.t, lc.y, lc.dy

    out = {'n_obs': int(t.size), 'dy_med': float(np.median(dy))}
    # FTP single-template null at each H
    for H, (c_n, s_n) in _G['ftp_coeffs'].items():
        out['ftp_H%d' % H] = _max_power(t, y, dy, c_n, s_n, freqs)
    # GLS baseline
    out['gls'] = float(np.max(_G['gls'].power_spectrum(t, y, None, dy, freqs)))
    # Catalog-max over the PAM vocabulary: per-template max, then cumulative
    vt = np.array([_max_power(t, y, dy, c_n, s_n, freqs)
                   for (c_n, s_n) in _G['vocab_coeffs']])
    out['vocab_per_template'] = vt.tolist()
    for K in cfg['k_values']:
        out['cat_K%d' % K] = float(np.max(vt[:K]))
    return out


# ---------------------------------------------------------------------------
# Template construction (parent process, once).
# ---------------------------------------------------------------------------
def build_templates(harmonics, vocab_h, max_k, seed):
    """Representative order-H medoids (per H) + order-vocab_h PAM K=max_k vocab."""
    ftp_coeffs = {}
    for H in harmonics:
        lib_H = fetch_sesar_templates(nharmonics=H)
        medoid = build_template_catalog(lib_H, 1, method='pam', random_state=seed)[0]
        ftp_coeffs[int(H)] = (np.asarray(medoid.c_n), np.asarray(medoid.s_n))
    lib_V = fetch_sesar_templates(nharmonics=vocab_h)
    vocab = build_template_catalog(lib_V, max_k, method='pam', random_state=seed)
    vocab_coeffs = [(np.asarray(tm.c_n), np.asarray(tm.s_n)) for tm in vocab]
    return ftp_coeffs, vocab_coeffs


# ---------------------------------------------------------------------------
# Analysis / outputs.
# ---------------------------------------------------------------------------
def gev_fit(samples):
    """Fit scipy GEV; return dict(c, loc, scale) or None if degenerate."""
    from scipy import stats
    samples = np.asarray(samples, dtype=float)
    if samples.size < 8 or np.allclose(samples, samples[0]):
        return None
    c, loc, scale = stats.genextreme.fit(samples)
    return {'c': float(c), 'loc': float(loc), 'scale': float(scale)}


def gev_sf(thr, params):
    from scipy import stats
    return stats.genextreme.sf(thr, params['c'], params['loc'], params['scale'])


def empirical_threshold(samples, fap):
    """Non-parametric threshold at exceedance probability `fap` (upper quantile)."""
    return float(np.quantile(np.asarray(samples, float), 1.0 - fap))


def gev_threshold(params, fap):
    from scipy import stats
    return float(stats.genextreme.isf(fap, params['c'], params['loc'],
                                      params['scale']))


def make_figures(keys_H, keys_K, cols, gev, cfg, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    # --- Fig 1: FAP vs threshold, per H (+ GLS) ---
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    thr_grid = np.linspace(0.0, 1.0, 400)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(keys_H)))
    for col, key in zip(colors, keys_H):
        s = np.sort(cols[key])[::-1]
        emp_fap = (np.arange(1, s.size + 1)) / s.size
        label = 'FTP H=%s' % key.split('H')[-1] if key.startswith('ftp') else 'GLS'
        ax.step(s, emp_fap, where='post', color=col, lw=1.4, alpha=0.85, label=label)
        if gev.get(key):
            ax.plot(thr_grid, gev_sf(thr_grid, gev[key]), color=col, lw=1.0,
                    ls='--', alpha=0.9)
    # GLS baseline
    s = np.sort(cols['gls'])[::-1]
    ax.step(s, np.arange(1, s.size + 1) / s.size, where='post', color='k',
            lw=1.4, alpha=0.7, label='GLS')
    if gev.get('gls'):
        ax.plot(thr_grid, gev_sf(thr_grid, gev['gls']), color='k', lw=1.0,
                ls='--', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('max power threshold  (P = 1 - chi2/chi2_0)')
    ax.set_ylabel('false-alarm probability  P(max power > threshold)')
    ax.set_title('Measured null FAP (pure noise, N=%d, %d freqs, %d realizations)'
                 % (cfg['n_obs'], cfg['n_freq'], cfg['n_real']))
    ax.set_ylim(1.0 / cfg['n_real'] / 2, 1.0)
    ax.grid(alpha=0.25, which='both')
    ax.legend(title='solid=empirical, dashed=GEV', fontsize=8, ncol=2)
    fig.tight_layout()
    f1 = os.path.join(outdir, 'fap_vs_threshold.png')
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # --- Fig 2: null-max vs K (catalog-max over PAM vocab) ---
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    Ks = [int(k.split('K')[-1]) for k in keys_K]
    mean_ = [np.mean(cols[k]) for k in keys_K]
    med_ = [np.median(cols[k]) for k in keys_K]
    p99 = [np.quantile(cols[k], 0.99) for k in keys_K]
    thr01 = [gev_threshold(gev[k], 0.01) if gev.get(k)
             else empirical_threshold(cols[k], 0.01) for k in keys_K]
    axa.plot(Ks, mean_, 'o-', label='mean null max')
    axa.plot(Ks, med_, 's-', label='median null max')
    axa.plot(Ks, p99, '^-', label='99th-pct null max')
    axa.plot(Ks, thr01, 'D--', color='crimson', label='FAP=0.01 threshold (GEV)')
    axa.set_xscale('log', base=2)
    axa.set_xticks(Ks)
    axa.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axa.set_xlabel('catalog size K  (PAM vocab, order %d)' % cfg['vocab_h'])
    axa.set_ylabel('null max power')
    axa.set_title('Extra-trials penalty of the catalog-max statistic')
    axa.grid(alpha=0.25)
    axa.legend(fontsize=8)
    # FAP curves per K
    for K, key in zip(Ks, keys_K):
        s = np.sort(cols[key])[::-1]
        axb.step(s, np.arange(1, s.size + 1) / s.size, where='post', lw=1.4,
                 label='K=%d' % K)
    axb.set_yscale('log')
    axb.set_xlabel('max power threshold')
    axb.set_ylabel('false-alarm probability')
    axb.set_title('Catalog-max FAP vs K')
    axb.set_ylim(1.0 / cfg['n_real'] / 2, 1.0)
    axb.grid(alpha=0.25, which='both')
    axb.legend(fontsize=8)
    fig.tight_layout()
    f2 = os.path.join(outdir, 'nullmax_vs_K.png')
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    return f1, f2


def write_summary(cols, gev, cfg, keys_H, keys_K, timing, outdir):
    faps = [0.1, 0.05, 0.01, 0.001]
    lines = []
    lines.append('# WP D1 -- measured null / FAP calibration (SUMMARY)')
    lines.append('')
    lines.append('All numbers below are computed by `measure_null.py` from this '
                 "run's own Monte-Carlo output (no quoted constants).")
    lines.append('')
    lines.append('## Configuration (production synthetic cadence + error model)')
    lines.append('')
    for k in ('n_real', 'n_obs', 'n_freq', 'f_min', 'f_max', 'baseline_days',
              'season_length_days', 'season_period_days', 'mean_mag',
              'harmonics', 'vocab_h', 'k_values', 'seed', 'seed_base'):
        lines.append('- `%s` = %s' % (k, cfg[k]))
    lines.append('- median per-point sigma dy = %.5f mag (constant; amplitude=0)'
                 % np.median(cols['dy_med']))
    lines.append('- realized N_obs (mean/min/max) = %.1f / %d / %d'
                 % (np.mean(cols['n_obs']), int(np.min(cols['n_obs'])),
                    int(np.max(cols['n_obs']))))
    lines.append('')
    lines.append('## Sanity anchor (audit): pure-noise max power ~0.5 at N=40')
    lines.append('')
    lines.append('| statistic | mean | median | 99th pct |')
    lines.append('|-----------|------|--------|----------|')
    for key in keys_H + ['gls']:
        name = ('FTP H=%s' % key.split('H')[-1]) if key.startswith('ftp') else 'GLS'
        lines.append('| %s | %.4f | %.4f | %.4f |'
                     % (name, np.mean(cols[key]), np.median(cols[key]),
                        np.quantile(cols[key], 0.99)))
    lines.append('')
    lines.append('## FAP-vs-threshold: power thresholds at fixed FAP (single template)')
    lines.append('')
    hdr = '| statistic | ' + ' | '.join('FAP=%g (emp / GEV)' % f for f in faps) + ' |'
    lines.append(hdr)
    lines.append('|' + '---|' * (len(faps) + 1))
    for key in keys_H + ['gls']:
        name = ('FTP H=%s' % key.split('H')[-1]) if key.startswith('ftp') else 'GLS'
        cells = []
        for f in faps:
            emp = empirical_threshold(cols[key], f)
            g = gev_threshold(gev[key], f) if gev.get(key) else float('nan')
            cells.append('%.3f / %.3f' % (emp, g))
        lines.append('| %s | %s |' % (name, ' | '.join(cells)))
    lines.append('')
    lines.append('## GEV tail fit parameters (scipy.stats.genextreme; c=-xi)')
    lines.append('')
    lines.append('| statistic | c (shape) | loc | scale |')
    lines.append('|-----------|-----------|-----|-------|')
    for key in keys_H + keys_K + ['gls']:
        if gev.get(key):
            lines.append('| %s | %.4f | %.4f | %.4f |'
                         % (key, gev[key]['c'], gev[key]['loc'], gev[key]['scale']))
    lines.append('')
    lines.append('## null-max vs K (catalog-max over the PAM vocab, order %d)'
                 % cfg['vocab_h'])
    lines.append('')
    lines.append('| K | mean | median | 99th pct | FAP=0.01 thr (emp / GEV) |')
    lines.append('|---|------|--------|----------|--------------------------|')
    for key in keys_K:
        K = key.split('K')[-1]
        emp01 = empirical_threshold(cols[key], 0.01)
        g01 = gev_threshold(gev[key], 0.01) if gev.get(key) else float('nan')
        lines.append('| %s | %.4f | %.4f | %.4f | %.3f / %.3f |'
                     % (K, np.mean(cols[key]), np.median(cols[key]),
                        np.quantile(cols[key], 0.99), emp01, g01))
    lines.append('')
    lines.append('## Timing (this machine, %d workers)' % cfg['n_workers'])
    lines.append('')
    lines.append('- template build: %.1f s' % timing['build_s'])
    lines.append('- Monte-Carlo: %.1f s wall for %d realizations '
                 '(%.3f s/realization wall; %.3f CPU-s/realization)'
                 % (timing['mc_s'], cfg['n_real'],
                    timing['mc_s'] / cfg['n_real'],
                    timing['mc_s'] * cfg['n_workers'] / cfg['n_real']))
    lines.append('- host: %s, python %s, numpy %s'
                 % (platform.platform(), platform.python_version(), np.__version__))
    lines.append('')
    path = os.path.join(outdir, 'SUMMARY.md')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n-real', type=int, default=2000)
    ap.add_argument('--n-obs', type=int, default=40)          # audit sanity anchor
    ap.add_argument('--n-freq', type=int, default=10000)      # >= grid-guard floor
    ap.add_argument('--f-min', type=float, default=1.0)       # cyc/day (P=1.0 d)
    ap.add_argument('--f-max', type=float, default=5.0)       # cyc/day (P=0.2 d)
    ap.add_argument('--baseline-days', type=float, default=3 * 365.25)
    ap.add_argument('--season-length-days', type=float, default=270.0)
    ap.add_argument('--season-period-days', type=float, default=365.25)
    ap.add_argument('--mean-mag', type=float, default=15.0)
    ap.add_argument('--harmonics', type=str, default='1,3,6,12')
    ap.add_argument('--vocab-h', type=int, default=8)         # headline nharmonics
    ap.add_argument('--k-values', type=str, default='1,2,4,8')
    ap.add_argument('--seed', type=int, default=0)            # PAM build seed
    ap.add_argument('--seed-base', type=int, default=1000000) # per-realization seeds
    ap.add_argument('--n-workers', type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument('--outdir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'output_null'))
    ap.add_argument('--smoke', action='store_true',
                    help='canary: 24 realizations, 4 workers')
    args = ap.parse_args()

    if args.smoke:
        args.n_real = 24
        args.n_workers = min(args.n_workers, 4)
        args.outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'output_smoke')

    harmonics = [int(x) for x in args.harmonics.split(',')]
    k_values = [int(x) for x in args.k_values.split(',')]
    max_k = max(k_values)
    os.makedirs(args.outdir, exist_ok=True)

    cfg = dict(n_real=args.n_real, n_obs=args.n_obs, n_freq=args.n_freq,
               f_min=args.f_min, f_max=args.f_max,
               baseline_days=args.baseline_days,
               season_length_days=args.season_length_days,
               season_period_days=args.season_period_days,
               mean_mag=args.mean_mag, harmonics=harmonics, vocab_h=args.vocab_h,
               k_values=k_values, seed=args.seed, seed_base=args.seed_base,
               n_workers=args.n_workers)

    print('[fap] building templates (Sesar PAM medoids)...', flush=True)
    t0 = time.time()
    ftp_coeffs, vocab_coeffs = build_templates(harmonics, args.vocab_h, max_k,
                                               args.seed)
    build_s = time.time() - t0
    print('[fap]   built %d order-H medoids + %d-medoid order-%d vocab in %.1fs'
          % (len(ftp_coeffs), len(vocab_coeffs), args.vocab_h, build_s), flush=True)

    freqs = frequency_grid(args.f_min, args.f_max, args.n_freq)
    df = freqs[1] - freqs[0]
    rayleigh = 1.0 / args.baseline_days
    print('[fap]   grid: %d freqs [%.4f,%.4f] cyc/day, df=%.3g, %.2f pts/Rayleigh'
          % (freqs.size, freqs[0], freqs[-1], df, rayleigh / df), flush=True)

    print('[fap] Monte-Carlo: %d realizations x %d workers...'
          % (args.n_real, args.n_workers), flush=True)
    t0 = time.time()
    with Pool(args.n_workers, initializer=_init_worker,
              initargs=(cfg, ftp_coeffs, vocab_coeffs, freqs)) as pool:
        chunk = max(1, args.n_real // (args.n_workers * 8))
        results = pool.map(run_one, range(args.n_real), chunksize=chunk)
    mc_s = time.time() - t0
    print('[fap]   done in %.1fs (%.3f s/realization wall)'
          % (mc_s, mc_s / args.n_real), flush=True)

    # Collate columns
    keys_H = ['ftp_H%d' % H for H in harmonics]
    keys_K = ['cat_K%d' % K for K in k_values]
    cols = {}
    for key in keys_H + keys_K + ['gls', 'n_obs', 'dy_med']:
        cols[key] = np.array([r[key] for r in results], dtype=float)
    vocab_per_template = np.array([r['vocab_per_template'] for r in results],
                                  dtype=float)

    # GEV fits
    gev = {}
    for key in keys_H + keys_K + ['gls']:
        g = gev_fit(cols[key])
        if g:
            gev[key] = g

    # Persist raw artifact
    npz_path = os.path.join(args.outdir, 'null_maxpower.npz')
    np.savez_compressed(npz_path,
                        config_json=json.dumps(cfg),
                        gev_json=json.dumps(gev),
                        freqs=freqs,
                        vocab_per_template=vocab_per_template,
                        **{key: cols[key] for key in
                           keys_H + keys_K + ['gls', 'n_obs', 'dy_med']})
    print('[fap] wrote %s' % npz_path, flush=True)

    timing = dict(build_s=build_s, mc_s=mc_s)
    f1, f2 = make_figures(keys_H, keys_K, cols, gev, cfg, args.outdir)
    print('[fap] wrote %s , %s' % (f1, f2), flush=True)
    summ = write_summary(cols, gev, cfg, keys_H, keys_K, timing, args.outdir)
    print('[fap] wrote %s' % summ, flush=True)

    # Console sanity anchor
    print('[fap] sanity (N=%d) mean null max power:' % args.n_obs, flush=True)
    for key in keys_H + ['gls']:
        print('        %-8s mean=%.4f median=%.4f p99=%.4f'
              % (key, np.mean(cols[key]), np.median(cols[key]),
                 np.quantile(cols[key], 0.99)), flush=True)

    # Completion marker (LAST -- a later queue session polls for this)
    marker = os.path.join(args.outdir, 'DONE.marker')
    with open(marker, 'w') as fh:
        fh.write('n_real=%d n_obs=%d n_freq=%d mc_s=%.1f build_s=%.1f\n'
                 % (args.n_real, args.n_obs, args.n_freq, mc_s, build_s))
    print('[fap] wrote marker %s' % marker, flush=True)


if __name__ == '__main__':
    main()
