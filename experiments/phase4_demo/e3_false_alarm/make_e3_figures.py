#!/usr/bin/env python
"""WP E3 phase 3 -- memo figures for the real-sample false-alarm comparison.

Reads the aggregate produced by ``run_e3.py --stage aggregate``
(``e3_results.npz``) plus the D1 null (``experiments/fap/output_null/
null_maxpower.npz``) and regenerates the two E3_MEMO.md figures:

  fig_e3_primary.png   -- N=40-subsampled real max powers (clean vs screened)
                          overlaid on the D1 single-band N=40 null, with the
                          D1 empirical p99 and GEV-99 thresholds.
  fig_e3_secondary.png -- full-N real max powers per band overlaid on the
                          fresh N/T/grid-matched null, with its empirical p99.

Also prints (stdout) every exceedance count / expectation quoted in the memo.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as sps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_NPZ = os.path.join(HERE, 'e3_results.npz')
D1_NPZ = os.path.abspath(os.path.join(
    HERE, '..', '..', 'fap', 'output_null', 'null_maxpower.npz'))

# palette (dataviz reference instance, light mode; validated)
C_SURFACE = '#fcfcfb'
C_INK = '#0b0b0b'
C_INK2 = '#52514e'
C_MUTED = '#898781'
C_GRID = '#e1e0d9'
C_NULL = '#c3c2b7'      # neutral reference fill (chrome, not a series hue)
C_CLEAN = '#2a78d6'     # categorical slot 1 (blue)
C_SCREEN = '#1baf7a'    # categorical slot 2 (aqua) -- relief via legend labels

PANELS = [('gls', 'GLS  (= FTP H=1)'),
          ('ftp_H3', 'FTP medoid, H=3'),
          ('cat_K1', 'catalog max, K=1  (H=8 vocab)'),
          ('cat_K8', 'catalog max, K=8  (H=8 vocab)')]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.edgecolor': C_MUTED, 'axes.labelcolor': C_INK2,
    'xtick.color': C_MUTED, 'ytick.color': C_MUTED,
    'axes.linewidth': 0.8, 'figure.facecolor': C_SURFACE,
    'axes.facecolor': C_SURFACE, 'savefig.facecolor': C_SURFACE})


def style_ax(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)


def panel(ax, null, real, clean, title, thr_lines, rug_named=()):
    """One overlay panel: null density fill + clean step hist + screened rug."""
    fin = np.isfinite(real)
    lo = min(null.min(), real[fin].min())
    hi = max(null.max(), real[fin].max()) * 1.02
    bins = np.linspace(lo, hi, 36)
    ax.hist(null, bins=bins, density=True, color=C_NULL, alpha=0.55,
            label='null (n=%d)' % null.size, zorder=1)
    ax.hist(real[fin & clean], bins=bins, density=True, histtype='step',
            color=C_CLEAN, lw=1.8,
            label='real, clean (n=%d)' % int((fin & clean).sum()), zorder=3)
    scr = real[fin & ~clean]
    ymax = ax.get_ylim()[1]
    ax.vlines(scr, 0, 0.10 * ymax, color=C_SCREEN, lw=1.4, zorder=4,
              label='real, screened (n=%d)' % scr.size)
    for lab, x, ls, c in thr_lines:
        ax.axvline(x, color=c, ls=ls, lw=1.0, zorder=2)
        ax.text(x, ymax * 0.99, ' ' + lab, color=c, fontsize=7.5,
                ha='left', va='top', rotation=90)
    for name, x in rug_named:
        ax.annotate(name, xy=(x, 0.10 * ymax), xytext=(x, 0.30 * ymax),
                    color=C_INK2, fontsize=7.5, ha='center',
                    arrowprops=dict(arrowstyle='-', color=C_MUTED, lw=0.7))
    ax.set_title(title, color=C_INK, fontsize=9.5, loc='left')
    ax.set_xlabel('max power on [1, 5] cyc/day')
    style_ax(ax)


def main():
    res = np.load(RESULTS_NPZ, allow_pickle=True)
    summ = json.loads(str(res['summary_json']))
    d1 = np.load(D1_NPZ, allow_pickle=True)
    gev = json.loads(str(d1['gev_json']))
    clean = res['clean'].astype(bool)
    band = res['band'].astype(str)
    gid = res['gid'].astype(str)

    # ---------------- figure 1: primary (N=40, D1 grid, D1 null) ----------
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    for ax, (key, label) in zip(axes.ravel(), PANELS):
        null = np.asarray(d1[key], float)
        real = res['prim_%s' % key]
        thr99 = float(np.quantile(null, 0.99))
        g = gev[key]
        thr_gev = float(sps.genextreme.ppf(0.99, g['c'], loc=g['loc'],
                                           scale=g['scale']))
        named = []
        if key == 'cat_K8':
            for want in ('g00017', 'g00052'):
                m = (gid == want) & np.isfinite(real)
                i = np.argmax(np.where(m, real, -np.inf))
                named.append(('%s %s' % (want, band[i]), float(real[i])))
        panel(ax, null, real, clean, label,
              [('D1 p99', thr99, '--', C_INK),
               ('GEV 1%', thr_gev, ':', C_INK2)], rug_named=named)
        fin = np.isfinite(real)
        n_c = int((fin & clean).sum())
        obs = int((fin & clean & (real > thr99)).sum())
        obs_s = int((fin & ~clean & (real > thr99)).sum())
        exp = 0.01 * n_c
        pval = float(sps.binom.sf(obs - 1, n_c, 0.01))
        ax.text(0.985, 0.72, 'clean > p99: %d (exp %.1f, P>=obs %.2f)\n'
                'screened > p99: %d' % (obs, exp, pval, obs_s),
                transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
                color=C_INK2)
        print('[fig1] %-8s thr99=%.4f gev99=%.4f clean_exceed=%d/%d exp=%.2f '
              'binom_sf=%.3f screened_exceed=%d/%d'
              % (key, thr99, thr_gev, obs, n_c, exp, pval, obs_s,
                 int((fin & ~clean).sum())))
    axes[0, 0].set_ylabel('density')
    axes[1, 0].set_ylabel('density')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', frameon=False,
               bbox_to_anchor=(0.5, 0.965), fontsize=8.5)
    fig.suptitle('E3 primary: real ZTF LCs subsampled to N=40 on the D1 grid '
                 '(10k freqs, [1,5] cyc/day) vs the D1 N=40 null',
                 color=C_INK, fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out1 = os.path.join(HERE, 'fig_e3_primary.png')
    fig.savefig(out1, dpi=160)
    print('[fig1] wrote', out1)

    # ---------------- figure 2: secondary (full N, matched null) ----------
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    cols = [('gls', 'GLS  (= FTP H=1)'), ('cat_K8', 'catalog max, K=8')]
    for i_b, b in enumerate(('g', 'r')):
        meta = json.loads(str(res['null_%s_meta_json' % b]))
        m = band == b
        for i_c, (key, label) in enumerate(cols):
            ax = axes[i_b, i_c]
            null = res['null_%s_%s' % (b, key)]
            real = np.where(m, res['sec_%s' % key], np.nan)
            thr99 = float(np.quantile(null, 0.99))
            panel(ax, null, real, clean, '%s-band -- %s' % (b, label),
                  [('null p99', thr99, '--', C_INK)])
            fin = np.isfinite(real)
            n_c = int((fin & clean).sum())
            obs = int((fin & clean & (real > thr99)).sum())
            obs_s = int((fin & ~clean & (real > thr99)).sum())
            pval = float(sps.binom.sf(obs - 1, n_c, 0.01))
            ax.text(0.985, 0.72, 'clean > p99: %d (exp %.1f, P>=obs %.2f)\n'
                    'screened > p99: %d' % (obs, 0.01 * n_c, pval, obs_s),
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7.5, color=C_INK2)
            print('[fig2] %s %-8s thr99=%.4f clean_exceed=%d/%d exp=%.2f '
                  'binom_sf=%.3f screened_exceed=%d/%d  '
                  '(null n=%d, N_med=%d, T_med=%.0f, nf=%d)'
                  % (b, key, thr99, obs, n_c, 0.01 * n_c, pval, obs_s,
                     int((fin & ~clean).sum()), meta['n_real'], meta['N_med'],
                     meta['T_med'], meta['n_freq']))
        axes[i_b, 0].set_ylabel('density')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', frameon=False,
               bbox_to_anchor=(0.5, 0.965), fontsize=8.5)
    fig.suptitle('E3 secondary: full-N real max powers (df=0.2/T grids) vs '
                 'the fresh N/T/grid-matched null, per band',
                 color=C_INK, fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out2 = os.path.join(HERE, 'fig_e3_secondary.png')
    fig.savefig(out2, dpi=160)
    print('[fig2] wrote', out2)


if __name__ == '__main__':
    main()
