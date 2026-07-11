"""WP E4 finalize: post-adjudication rates + memo figures.

Inputs: e4_scores.csv (Score phase), e4_rates.json, e4_adjudication.json
(recorded adjudication calls).  Applies the post-adjudication rule stated in
e4_adjudication.json:

  * lawful star  -> adjudicated_period (the ftp_mb 7.5-yr consensus period);
    every method's best_freq is re-classified against it with the SAME AND
    criterion (frac < 1% and |df|*T < 0.5; exact-or-harmonic credited);
  * failure star -> no method credited;
  * non-adjudicated (star, method) cells keep their pre-adjudication verdict
    (only the ftp_mb queue was adjudicated, per the E4 brief).

Outputs:
  e4_post_adjudication.json          per-method pre/post rates, all groups
  fig_recovery_by_method.png         grouped bars + Wilson CIs + post markers
  fig_strata.png                     stratified table figure (science stars)
"""
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from score_e4 import classify, wilson, METHODS  # same seams, same spec

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES_CSV = os.path.join(HERE, 'e4_scores.csv')
RATES_JSON = os.path.join(HERE, 'e4_rates.json')
ADJ_JSON = os.path.join(HERE, 'e4_adjudication.json')
POST_JSON = os.path.join(HERE, 'e4_post_adjudication.json')
FIG_METHOD = os.path.join(HERE, 'fig_recovery_by_method.png')
FIG_STRATA = os.path.join(HERE, 'fig_strata.png')

INK = {'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781',
       'grid': '#e1e0d9', 'axis': '#c3c2b7', 'surface': '#fcfcfb'}
C_SCI = '#2a78d6'    # categorical slot 1 (blue)  - science
C_AJ = '#1baf7a'     # categorical slot 2 (aqua)  - anti-join controls
                     # (aqua < 3:1 on light surface -> relief rule satisfied
                     #  by the direct k/n labels on every bar)
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95',
       '#0d366b']    # sequential blue ramp, light->dark

GROUPS = {'science': ('science',), 'antijoin': ('control_antijoin',),
          'all_unique': ('science', 'control_antijoin', 'control_orig')}


def load_scores():
    with open(SCORES_CSV) as fh:
        return list(csv.DictReader(fh))


def post_verdicts(rows, adj):
    """{(uid, method): recovered_post (bool)} + per-cell post class."""
    calls = {s['uid']: s for s in adj['stars']}
    out = {}
    for r in rows:
        uid, m = r['uid'], r['method']
        pre = r['class'] in ('exact', '2f', 'f/2')
        if uid not in calls:
            out[(uid, m)] = (pre, r['class'])
            continue
        call = calls[uid]
        if call['verdict'] == 'failure':
            out[(uid, m)] = (False, 'failure(depth)')
            continue
        P_adj = float(call['adjudicated_period'])
        lab = classify(float(r['f_rec']), 1.0 / P_adj,
                       float(r['T_grid_days']))[0]
        out[(uid, m)] = (pre or lab in ('exact', '2f', 'f/2'),
                         '%s->adj:%s' % (r['class'], lab))
    return out


def rates_table(rows, post):
    out = {}
    for m in METHODS:
        out[m] = {}
        for g, roles in GROUPS.items():
            sel = [r for r in rows if r['method'] == m and r['role'] in roles]
            k_pre = sum(r['class'] in ('exact', '2f', 'f/2') for r in sel)
            k_post = sum(post[(r['uid'], r['method'])][0] for r in sel)
            n = len(sel)
            p_pre, lo_pre, hi_pre = wilson(k_pre, n)
            p_post, lo_post, hi_post = wilson(k_post, n)
            out[m][g] = {'n': n,
                         'pre': {'k': k_pre, 'rate': p_pre,
                                 'ci95': [lo_pre, hi_pre]},
                         'post': {'k': k_post, 'rate': p_post,
                                  'ci95': [lo_post, hi_post]}}
    return out


# ------------------------------------------------------------------- figures
def style_ax(ax):
    ax.set_facecolor(INK['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK['axis'])
    ax.tick_params(colors=INK['muted'], labelsize=8)
    ax.grid(axis='y', color=INK['grid'], lw=0.7)
    ax.set_axisbelow(True)


def fig_method(table):
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    fig.patch.set_facecolor(INK['surface'])
    style_ax(ax)
    x = np.arange(len(METHODS))
    w = 0.34
    for off, g, color, label in ((-w / 2, 'science', C_SCI,
                                  'science (n=58, Chen truth)'),
                                 (w / 2, 'antijoin', C_AJ,
                                  'anti-join controls (n=12, Gaia truth)')):
        pre = [table[m][g]['pre'] for m in METHODS]
        post = [table[m][g]['post'] for m in METHODS]
        y = [100 * p['rate'] for p in pre]
        yerr = np.array([[100 * (p['rate'] - p['ci95'][0]) for p in pre],
                         [100 * (p['ci95'][1] - p['rate']) for p in pre]])
        ax.bar(x + off, y, width=w - 0.04, color=color, label=label,
               zorder=2)
        ax.errorbar(x + off, y, yerr=yerr, fmt='none', ecolor=INK['secondary'],
                    elinewidth=1.1, capsize=2.5, zorder=4)
        ax.scatter(x + off, [100 * p['rate'] for p in post], marker='D', s=26,
                   facecolor='white', edgecolor=INK['primary'], lw=1.1,
                   zorder=5,
                   label=('post-adjudication' if g == 'science' else None))
        n = table[METHODS[0]][g]['n']
        for xi, p in zip(x + off, pre):
            ax.text(xi, 3.5, '%d/%d' % (p['k'], n), ha='center', va='bottom',
                    fontsize=7, color='white', fontweight='bold', zorder=6,
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=8.5, color=INK['primary'])
    ax.set_ylim(0, 104)
    ax.set_ylabel('period recovery (%), exact-or-harmonic', fontsize=9,
                  color=INK['secondary'])
    ax.set_title('WP E4: known-period recovery on 73 ZTF RRL -- bars = pre-'
                 'adjudication (Wilson 95%), diamonds = post-adjudication',
                 fontsize=9.5, color=INK['primary'], loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=7.5, frameon=False, labelcolor=INK['secondary'])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG_METHOD, facecolor=INK['surface'])
    plt.close(fig)


def seq_color(rate):
    """rate in [0,1] -> (fill hex, text color) on the blue sequential ramp,
    mapping the useful 0.70-1.00 range light->dark."""
    t = np.clip((rate - 0.70) / 0.30, 0, 1)
    idx = int(round(t * (len(SEQ) - 1)))
    return SEQ[idx], ('white' if idx >= 3 else INK['primary'])


def fig_strata(strata):
    mag_lab = strata['mag_labels']
    ep_lab = strata['epoch_labels']
    cols = [('mag', 'median r mag\n%s' % l, i) for i, l in enumerate(mag_lab)]
    cols += [('epochs', 'g+r epochs\n%s' % l, i) for i, l in enumerate(ep_lab)]
    fig, ax = plt.subplots(figsize=(10.2, 3.9), dpi=150)
    fig.patch.set_facecolor(INK['surface'])
    ax.set_facecolor(INK['surface'])
    ax.set_xlim(0, len(cols) + 1.4)
    ax.set_ylim(-0.6, len(METHODS) + 1.6)
    ax.axis('off')
    ax.text(0.02, len(METHODS) + 1.15,
            'WP E4: pre-adjudication recovery, 58 science stars, by stratum '
            '-- k/n and Wilson 95% CI; fill = rate (70%% light -> 100%% dark)',
            fontsize=9, color=INK['primary'], fontweight='bold')
    for j, (kind, lab, _) in enumerate(cols):
        ax.text(1.4 + j + 0.5, len(METHODS) + 0.08, lab, ha='center',
                va='bottom', fontsize=7.2, color=INK['secondary'])
    for i, m in enumerate(METHODS):
        yy = len(METHODS) - 1 - i
        ax.text(1.32, yy + 0.5, m, ha='right', va='center', fontsize=8,
                color=INK['primary'])
        for j, (kind, _, b) in enumerate(cols):
            cell = strata['per_method'][m][kind][b]
            rate = cell['rate']
            fill, txt = seq_color(rate)
            ax.add_patch(Rectangle((1.4 + j + 0.03, yy + 0.03), 0.94, 0.94,
                                   facecolor=fill, edgecolor=INK['surface'],
                                   lw=1.5))
            ax.text(1.4 + j + 0.5, yy + 0.60, '%d/%d' % (cell['k'], cell['n']),
                    ha='center', va='center', fontsize=7.6, color=txt,
                    fontweight='bold')
            ax.text(1.4 + j + 0.5, yy + 0.28,
                    '%.0f%% [%.0f-%.0f]' % (100 * rate, 100 * cell['ci95'][0],
                                            100 * cell['ci95'][1]),
                    ha='center', va='center', fontsize=6.4, color=txt)
    fig.tight_layout()
    fig.savefig(FIG_STRATA, facecolor=INK['surface'])
    plt.close(fig)


def main():
    rows = load_scores()
    with open(ADJ_JSON) as fh:
        adj = json.load(fh)
    with open(RATES_JSON) as fh:
        rates = json.load(fh)

    post = post_verdicts(rows, adj)
    table = rates_table(rows, post)

    print('=== pre -> post adjudication recovery (exact-or-harmonic) ===')
    for m in METHODS:
        line = ['%-10s' % m]
        for g in GROUPS:
            c = table[m][g]
            line.append('%s: %d->%d/%d (%.1f->%.1f%%)'
                        % (g, c['pre']['k'], c['post']['k'], c['n'],
                           100 * c['pre']['rate'], 100 * c['post']['rate']))
        print('  '.join(line))

    changed = {k: v for k, v in post.items() if '->' in v[1]}
    print('\nadjudicated cells (%d):' % len(changed))
    for (uid, m), (rec, note) in sorted(changed.items()):
        print('  %-42s %-10s %-24s recovered_post=%s' % (uid, m, note, rec))

    out = {'rule': adj['post_adjudication_rule'],
           'summary_counts': adj['summary'],
           'per_method': table}
    with open(POST_JSON, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    fig_method(table)
    fig_strata(rates['strata_science'])
    print('\nwrote %s, %s, %s' % (POST_JSON, FIG_METHOD, FIG_STRATA))


if __name__ == '__main__':
    main()
