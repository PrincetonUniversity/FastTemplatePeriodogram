"""WP E4 Adjudication phase -- evidence generator (plots + metrics only).

For every (star, ftp_mb) row whose E4 verdict is alias or miss (score_e4.py
queue), produce:

  * ``folds/<uid>_ftp_mb.png`` -- the light curve phase-folded at every
    candidate period: the row's truth (Chen for science, Gaia for controls),
    the recovered period, and the alternate truth (Gaia) when it exists and
    differs meaningfully.  Both bands on every panel (g blue, r red; per-band
    median subtracted), two phase cycles.
  * ``e4_adjudication_evidence.json`` -- per star: candidate periods with
    weighted PDM fold statistics (theta = binned residual variance / total
    variance, 25 bins, 1/dy^2 weights; lower = cleaner fold), reduced chi2
    about the binned fold model, the star's best_freq + E4 class from ALL six
    methods (consensus check), the ftp_mb top-5 peaks with the rank of the
    truth frequency, and magnitudes / epoch counts.

The adjudication CALLS themselves (lawful vs genuine failure, per the E4
brief's categories) are made by the reviewing agent from this evidence and
recorded in ``e4_adjudication.json`` -- not by this script.
"""
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.expanduser('~/.ftperiodogram_data/phase4_rrl_sample')
RESULTS = os.path.join(HERE, 'e4_results.npz')
SCORES_CSV = os.path.join(HERE, 'e4_scores.csv')
STAR_TABLE = os.path.join(HERE, 'star_table.json')
FOLDS = os.path.join(HERE, 'folds')
EVIDENCE = os.path.join(HERE, 'e4_adjudication_evidence.json')

METHODS = ('gls_1band', 'ftp_1band', 'mbls_h1', 'mhls_h8', 'ce', 'ftp_mb')
NBINS = 25

# reference categorical palette (dataviz skill, validated): g band blue,
# r band red -- domain-conventional band identities, CVD dE 74.6, both >=3:1
# on the light surface.
INK = {'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781',
       'grid': '#e1e0d9', 'axis': '#c3c2b7', 'surface': '#fcfcfb'}
BAND_COLOR = {'g': '#2a78d6', 'r': '#e34948'}


def load_lc(lc_file):
    d = np.load(os.path.join(DATA_DIR, lc_file))
    out = {}
    for b in ('g', 'r'):
        t = np.asarray(d['%s_hjd' % b], float)
        y = np.asarray(d['%s_mag' % b], float)
        dy = np.asarray(d['%s_magerr' % b], float)
        good = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy)
        out[b] = (t[good], y[good], dy[good])
    return out


def pdm_theta(lc, period, nbins=NBINS):
    """Weighted PDM: (theta, chi2_red) pooled over bands.

    theta = sum_b sum_i w(y - <y>_bin)^2 / sum_b sum_i w(y - <y>_band)^2,
    w = 1/dy^2; lower = cleaner fold.  chi2_red = weighted residual chi2
    about the binned fold model / (N - nbins_occupied).
    """
    num = den = chi2 = 0.0
    ndof = 0
    for b, (t, y, dy) in lc.items():
        w = 1.0 / dy ** 2
        ph = np.mod(t / period, 1.0)
        idx = np.minimum((ph * nbins).astype(int), nbins - 1)
        ybin = np.zeros(nbins)
        occupied = 0
        for k in range(nbins):
            m = idx == k
            if m.any():
                ybin[k] = np.average(y[m], weights=w[m])
                occupied += 1
        resid = y - ybin[idx]
        ybar = np.average(y, weights=w)
        num += float(np.sum(w * resid ** 2))
        den += float(np.sum(w * (y - ybar) ** 2))
        chi2 += float(np.sum(resid ** 2 / dy ** 2))
        ndof += y.size - occupied
    return num / den, chi2 / max(ndof, 1)


def fold_panel(ax, lc, period, title, theta):
    for b in ('g', 'r'):
        t, y, dy = lc[b]
        ph = np.mod(t / period, 1.0)
        ym = y - np.median(y)
        for cyc in (0.0, 1.0):
            ax.scatter(ph + cyc, ym, s=4, alpha=0.45, lw=0,
                       color=BAND_COLOR[b],
                       label=('%s (median-sub)' % b) if cyc == 0 else None)
    ax.invert_yaxis()
    ax.set_xlim(0, 2)
    ax.set_title('%s\nP = %.6f d   PDM theta = %.3f' % (title, period, theta),
                 fontsize=9, color=INK['primary'])
    ax.set_xlabel('phase (2 cycles)', fontsize=8, color=INK['muted'])
    ax.tick_params(labelsize=7, colors=INK['muted'])
    for s in ax.spines.values():
        s.set_color(INK['axis'])
    ax.grid(True, color=INK['grid'], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def main():
    os.makedirs(FOLDS, exist_ok=True)
    with open(SCORES_CSV) as fh:
        rows = list(csv.DictReader(fh))
    with open(STAR_TABLE) as fh:
        stars = {s['uid']: s for s in json.load(fh)['stars']}
    npz = np.load(RESULTS, allow_pickle=True)

    by_star = {}
    for r in rows:
        by_star.setdefault(r['uid'], {})[r['method']] = r

    queue = [r for r in rows if r['method'] == 'ftp_mb'
             and r['class'] not in ('exact', '2f', 'f/2')]
    print('adjudication queue: %d stars' % len(queue))

    evidence = []
    for r in queue:
        uid = r['uid']
        star = stars[uid]
        lc = load_lc(star['lc_file'])
        P_true = float(r['P_true'])
        P_rec = float(r['P_rec'])
        T = float(r['T_grid_days'])

        # candidate periods: truth, recovered, alternate (Gaia) truth
        cands = [('truth (%s)' % r['truth_source'], P_true)]
        # recovered can be near-identical to truth; still plot both panels.
        cands.append(('recovered (ftp_mb)', P_rec))
        gp = float(r['gaia_period']) if r['gaia_period'] not in ('', 'nan') \
            else float('nan')
        if (r['truth_source'] == 'chen' and np.isfinite(gp)
                and abs(1 / gp - 1 / P_true) * T > 0.05):
            cands.append(('alt truth (gaia %s)'
                          % r['gaia_period_source'].replace('gaia_dr3_', ''),
                          gp))

        cand_out = []
        fig, axes = plt.subplots(1, len(cands),
                                 figsize=(4.1 * len(cands), 3.6), dpi=150)
        axes = np.atleast_1d(axes)
        for ax, (name, P) in zip(axes, cands):
            theta, chi2r = pdm_theta(lc, P)
            dphi = abs(1 / P - 1 / P_true) * T
            cand_out.append({'name': name, 'P': P, 'theta': round(theta, 4),
                             'chi2_red_fold': round(chi2r, 2),
                             'dphi_vs_truth_cycles': round(dphi, 3)})
            fold_panel(ax, lc, P, name, theta)
        axes[0].set_ylabel('mag - median (per band)', fontsize=8,
                           color=INK['muted'])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize=7,
                   frameon=False)
        fig.suptitle('%s   role=%s type=%s   E4 class=%s   '
                     '|df|*T=%.2f cyc  frac=%.2e'
                     % (uid, r['role'], r['type'] or r['gaia_type'],
                        r['class'], float(r['dphi_cycles']),
                        float(r['frac_err'])),
                     fontsize=10, color=INK['primary'])
        fig.patch.set_facecolor(INK['surface'])
        for ax in axes:
            ax.set_facecolor(INK['surface'])
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        png = os.path.join(FOLDS, '%s_ftp_mb.png' % uid)
        fig.savefig(png, facecolor=INK['surface'])
        plt.close(fig)

        # cross-method consensus + top-5 peaks with truth rank
        consensus = {}
        for m in METHODS:
            rm = by_star[uid][m]
            consensus[m] = {'f_rec': float(rm['f_rec']),
                            'class': rm['class'],
                            'dphi_cycles': round(float(rm['dphi_cycles']), 2)}
        sel = (npz['uid'] == uid) & (npz['method'] == 'ftp_mb')
        i = int(np.flatnonzero(sel)[0])
        top5f = npz['top5_freq'][i]
        top5s = npz['top5_stat'][i]
        f_true = 1.0 / P_true
        # rank of truth among top-5 (phase-coherence window)
        rank = next((k for k, f in enumerate(top5f)
                     if np.isfinite(f) and abs(f - f_true) * T < 0.5), None)
        evidence.append({
            'uid': uid, 'role': r['role'],
            'type': r['type'] or r['gaia_type'], 'class': r['class'],
            'truth_source': r['truth_source'], 'P_true': P_true,
            'P_rec': P_rec, 'f_true': f_true,
            'dphi_cycles': round(float(r['dphi_cycles']), 3),
            'frac_err': float(r['frac_err']),
            'chen_period': r['chen_period'], 'gaia_period': r['gaia_period'],
            'gaia_period_source': r['gaia_period_source'],
            'g_med_mag': round(float(r['g_med_mag']), 2),
            'r_med_mag': round(float(r['r_med_mag']), 2),
            'n_epochs_joint': int(r['n_epochs_joint']),
            'candidates': cand_out,
            'truth_rank_in_ftp_mb_top5': rank,
            'ftp_mb_top5_freq': [round(float(f), 6) for f in top5f],
            'ftp_mb_top5_stat': [round(float(s), 5) for s in top5s],
            'all_methods': consensus,
            'png': os.path.relpath(png, HERE),
        })
        print('%-42s class=%-5s dphi=%6.2f  thetas=%s  truth_rank=%s'
              % (uid, r['class'], float(r['dphi_cycles']),
                 ['%.3f' % c['theta'] for c in cand_out], rank))

    with open(EVIDENCE, 'w') as fh:
        json.dump(evidence, fh, indent=1, sort_keys=True)
    print('wrote %s + %d fold PNGs in %s' % (EVIDENCE, len(evidence), FOLDS))


if __name__ == '__main__':
    main()
