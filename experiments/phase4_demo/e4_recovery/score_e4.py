"""WP E4 Score phase: alias-aware scoring of e4_results.npz (E1 protocol + B5).

Scoring spec (embedded verbatim in e4_results.npz meta_json, from the E4
brief): a peak at f_rec matches a target frequency f_t iff

    |f_rec - f_t| / f_t < 0.01           (fractional, Stringer+19 convention)
    AND |f_rec - f_t| * T < 0.5          (phase coherence, Graham+13; T = the
                                          per-star grid baseline T_grid_days)

Both tests must pass (the E4 brief ANDs them).  Each (star, method) TOP peak
(best_freq) is classified against, in order:

    exact      f_true
    harmonic   2 f_true ('2f'), f_true / 2 ('f/2')
    alias      f_true +- 1 cyc/day        ('beat+-1d')
               f_true +- 1/365.25 cyc/day ('beat+-1yr')
               f_true +- 1/354.37 cyc/day ('beat+-354d', the lunar-year /
                                           "sidereal-ish" beat of the brief)
    miss       none of the above

With T ~ 2600-2800 d the phase-coherence half-width (0.5/T ~ 1.9e-4 c/d) is
far narrower than any target separation (>= 1/365.25 - 1/354.37 ~ 8.4e-5 c/d
apart at worst... in fact the two year-beat targets are 8.4e-5 c/d apart, so
they CAN both pass the coherence test only if |f_rec - f_t| < 1.9e-4 for both;
classification takes the SMALLEST |f_rec - f_t| among passing targets to break
ties).  Everything else is unambiguous.

Truth convention (E4 brief):
    science (58, Chen+2020)        -> chen_period
    control_antijoin (12, Gaia)    -> gaia_period
    control_orig (3, Gaia)         -> gaia_period

'Recovered' (pre-adjudication headline) = exact OR harmonic.  Aliases and
misses count as failures pre-adjudication; the Adjudication phase may relabel
ftp_mb aliases/misses as lawful (post-adjudication rates reported separately).

Outputs (all in this directory):
    e4_scores.csv    one row per (star, method): classification + evidence
    e4_rates.json    rate tables (Wilson 95% CIs), alias breakdowns, strata,
                     dual-truth comparison, meta
Stdout: every table, human-readable.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)

RESULTS = os.path.join(HERE, 'e4_results.npz')
STAR_TABLE = os.path.join(HERE, 'star_table.json')
DATA_DIR = os.path.expanduser('~/.ftperiodogram_data/phase4_rrl_sample')
SCORES_CSV = os.path.join(HERE, 'e4_scores.csv')
RATES_JSON = os.path.join(HERE, 'e4_rates.json')

METHODS = ('gls_1band', 'ftp_1band', 'mbls_h1', 'mhls_h8', 'ce', 'ftp_mb')
MULTIBAND = ('mbls_h1', 'mhls_h8', 'ce', 'ftp_mb')

RTOL = 0.01
DPHI_MAX = 0.5
LUNAR_YEAR = 354.37          # the brief's "+-1/354-ish" beat
YEAR = 365.25
Z95 = 1.959963984540054

CLASS_ORDER = ['exact', '2f', 'f/2', 'beat+1d', 'beat-1d', 'beat+1yr',
               'beat-1yr', 'beat+354d', 'beat-354d', 'miss']


def targets_for(f_true):
    """Ordered (name, f_target) classification targets for one truth freq."""
    return [
        ('exact', f_true),
        ('2f', 2.0 * f_true),
        ('f/2', 0.5 * f_true),
        ('beat+1d', f_true + 1.0),
        ('beat-1d', f_true - 1.0),
        ('beat+1yr', f_true + 1.0 / YEAR),
        ('beat-1yr', f_true - 1.0 / YEAR),
        ('beat+354d', f_true + 1.0 / LUNAR_YEAR),
        ('beat-354d', f_true - 1.0 / LUNAR_YEAR),
    ]


def classify(f_rec, f_true, T):
    """(label, delta_f_to_truth, dphi_to_truth, frac_to_truth).

    label = first CLASS_ORDER target passing BOTH tests, ties broken by
    smallest |f_rec - f_target| (only year-vs-354d beats can ever tie).
    """
    passing = []
    for name, ft in targets_for(f_true):
        if ft <= 0:
            continue
        adf = abs(f_rec - ft)
        if adf / ft < RTOL and adf * T < DPHI_MAX:
            passing.append((adf, name))
    label = min(passing)[1] if passing else 'miss'
    adf_true = abs(f_rec - f_true)
    return label, adf_true, adf_true * T, adf_true / f_true


def wilson(k, n, z=Z95):
    """Wilson 95% interval on a pooled count k/n -> (rate, lo, hi)."""
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def fmt_ci(k, n):
    p, lo, hi = wilson(k, n)
    return '%2d/%2d = %5.1f%% [%5.1f, %5.1f]' % (k, n, 100 * p, 100 * lo,
                                                 100 * hi)


def star_mags():
    """{uid: (g_median_mag, r_median_mag, n_epochs_joint)} from the lc npz."""
    with open(STAR_TABLE) as fh:
        stars = json.load(fh)['stars']
    out = {}
    for s in stars:
        d = np.load(os.path.join(DATA_DIR, s['lc_file']))
        med = {}
        for b in ('g', 'r'):
            m = np.asarray(d['%s_mag' % b], dtype=float)
            med[b] = float(np.nanmedian(m[np.isfinite(m)]))
        out[s['uid']] = (med['g'], med['r'], int(s['joint']['n_epochs']))
    return out


def tercile_bins(values):
    """3 bins at the terciles; returns (edges, labels)."""
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    edges = (q1, q2)
    labels = ['<= %.2f' % q1, '(%.2f, %.2f]' % (q1, q2), '> %.2f' % q2]
    return edges, labels


def bin_index(v, edges):
    return 0 if v <= edges[0] else (1 if v <= edges[1] else 2)


def main():
    d = np.load(RESULTS, allow_pickle=True)
    n = d['uid'].size
    assert np.all(d['status'] == 'done'), 'aggregate has missing rows'
    mags = star_mags()

    # ---------------------------------------------------------------- score
    rows = []
    for i in range(n):
        uid = str(d['uid'][i])
        role = str(d['role'][i])
        truth_src = 'chen' if role == 'science' else 'gaia'
        P_true = float(d['chen_period'][i] if truth_src == 'chen'
                       else d['gaia_period'][i])
        assert np.isfinite(P_true) and P_true > 0, (uid, role)
        f_true = 1.0 / P_true
        T = float(d['T_grid_days'][i])
        f_rec = float(d['best_freq'][i])
        f_lo = float(d['df'][i]) * float(d['i_min'][i])
        f_hi = f_lo + float(d['df'][i]) * (float(d['n_freq'][i]) - 1)
        label, adf, dphi, frac = classify(f_rec, f_true, T)
        g_med, r_med, n_joint = mags[uid]
        rows.append({
            'uid': uid, 'method': str(d['method'][i]), 'role': role,
            'type': str(d['type'][i]), 'gaia_type': str(d['gaia_type'][i]),
            'dual_truth': bool(d['dual_truth'][i]),
            'truth_source': truth_src, 'P_true': P_true, 'f_true': f_true,
            'f_true_in_grid': bool(f_lo <= f_true <= f_hi),
            'f_rec': f_rec, 'P_rec': 1.0 / f_rec,
            'class': label, 'recovered_pre': label in ('exact', '2f', 'f/2'),
            'delta_f': adf, 'dphi_cycles': dphi, 'frac_err': frac,
            'T_grid_days': T, 'df': float(d['df'][i]),
            'pts_per_rayleigh': float(d['pts_per_rayleigh'][i]),
            'n_obs': int(d['n_obs'][i]), 'band_used': str(d['band_used'][i]),
            'g_med_mag': g_med, 'r_med_mag': r_med,
            'n_epochs_joint': n_joint,
            'chen_period': float(d['chen_period'][i]),
            'gaia_period': float(d['gaia_period'][i]),
            'gaia_period_source': str(d['gaia_period_source'][i]),
        })

    n_out = sum(1 for r in rows if not r['f_true_in_grid'])
    print('truth-in-grid check: %d/%d rows have f_true inside the grid'
          % (len(rows) - n_out, len(rows)))
    assert n_out == 0, 'truth frequency outside [1,5] grid for some star!'

    with open(SCORES_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------- rate tables (1)
    def group_rows(method, roles):
        return [r for r in rows if r['method'] == method and r['role'] in roles]

    groups = {
        'science': ('science',),
        'antijoin': ('control_antijoin',),
        'all_unique': ('science', 'control_antijoin', 'control_orig'),
    }
    rates = {}
    print('\n=== (1) Pre-adjudication recovery (exact-or-harmonic), '
          'Wilson 95% ===')
    hdr = '%-10s  %-28s  %-28s  %-28s' % ('method', 'science (n=58, Chen)',
                                          'antijoin (n=12, Gaia)',
                                          'all unique (n=73)')
    print(hdr)
    for m in METHODS:
        rates[m] = {}
        cells = []
        for gname, groles in groups.items():
            g = group_rows(m, groles)
            k = sum(r['recovered_pre'] for r in g)
            ke = sum(r['class'] == 'exact' for r in g)
            p, lo, hi = wilson(k, len(g))
            rates[m][gname] = {
                'n': len(g), 'k_recovered_pre': k, 'k_exact': ke,
                'rate': p, 'ci95': [lo, hi],
            }
            cells.append(fmt_ci(k, len(g)))
        print('%-10s  %s' % (m, '  '.join(cells)))

    print('\n--- exact-only rates (same groups) ---')
    for m in METHODS:
        cells = [fmt_ci(rates[m][g]['k_exact'], rates[m][g]['n'])
                 for g in groups]
        print('%-10s  %s' % (m, '  '.join(cells)))

    # ------------------------------------------------ alias breakdowns (1b)
    print('\n=== (1b) classification breakdown per method ===')
    breakdowns = {}
    for scope, groles in (('science', ('science',)),
                          ('all_unique', groups['all_unique'])):
        breakdowns[scope] = {}
        print('-- %s --' % scope)
        print('%-10s  %s' % ('method', '  '.join('%-9s' % c
                                                 for c in CLASS_ORDER)))
        for m in METHODS:
            g = group_rows(m, groles)
            bd = {c: sum(r['class'] == c for r in g) for c in CLASS_ORDER}
            bd['n'] = len(g)
            breakdowns[scope][m] = bd
            print('%-10s  %s' % (m, '  '.join('%-9d' % bd[c]
                                              for c in CLASS_ORDER)))

    # ------------------------------------------------------------ strata (2)
    sci_stars = sorted({r['uid'] for r in rows if r['role'] == 'science'})
    by_uid = {}
    for r in rows:
        if r['role'] == 'science' and r['method'] == 'gls_1band':
            by_uid[r['uid']] = r
    mag_vals = [by_uid[u]['r_med_mag'] for u in sci_stars]
    ep_vals = [by_uid[u]['n_epochs_joint'] for u in sci_stars]
    mag_edges, mag_labels = tercile_bins(mag_vals)
    ep_edges, ep_labels = tercile_bins(ep_vals)

    strata = {'mag_variable': 'median r-band mag',
              'mag_edges': list(mag_edges), 'mag_labels': mag_labels,
              'epoch_variable': 'joint g+r epochs',
              'epoch_edges': [float(e) for e in ep_edges],
              'epoch_labels': ep_labels, 'per_method': {}}
    print('\n=== (2) science-star strata (pre-adjudication recovery, '
          'Wilson 95%) ===')
    for axis, edges, labels, key in (
            ('r_med_mag', mag_edges, mag_labels, 'mag'),
            ('n_epochs_joint', ep_edges, ep_labels, 'epochs')):
        print('-- by %s (bins: %s) --' % (axis, ' | '.join(labels)))
        for m in METHODS:
            g = group_rows(m, ('science',))
            cells = []
            per_bin = []
            for b in range(3):
                gb = [r for r in g if bin_index(r[axis], edges) == b]
                k = sum(r['recovered_pre'] for r in gb)
                p, lo, hi = wilson(k, len(gb))
                per_bin.append({'bin': labels[b], 'n': len(gb), 'k': k,
                                'rate': p, 'ci95': [lo, hi]})
                cells.append(fmt_ci(k, len(gb)))
            strata['per_method'].setdefault(m, {})[key] = per_bin
            print('  %-10s  %s' % (m, '  '.join(cells)))

    # -------------------------------------------------------- dual truth (3)
    print('\n=== (3) dual-truth stars (n=12): Chen vs Gaia ===')
    dt_uids = sorted({r['uid'] for r in rows if r['dual_truth']})
    dual = {'n_stars': len(dt_uids), 'stars': [], 'n_verdict_changes': 0}
    for uid in dt_uids:
        rr = [r for r in rows if r['uid'] == uid]
        Pc, Pg = rr[0]['chen_period'], rr[0]['gaia_period']
        rel = abs(Pc - Pg) / Pc
        T = rr[0]['T_grid_days']
        dphi_truths = abs(1 / Pc - 1 / Pg) * T
        star_rec = {'uid': uid, 'chen_period': Pc, 'gaia_period': Pg,
                    'rel_diff': rel, 'truths_dphi_cycles': dphi_truths,
                    'methods': {}}
        for r in rr:
            lab_c = classify(r['f_rec'], 1 / Pc, T)[0]
            lab_g = classify(r['f_rec'], 1 / Pg, T)[0]
            star_rec['methods'][r['method']] = {'chen': lab_c, 'gaia': lab_g}
            if lab_c != lab_g:
                dual['n_verdict_changes'] += 1
        dual['stars'].append(star_rec)
        flag = (' <-- truths differ by %.2f cycles over T!' % dphi_truths
                if dphi_truths > DPHI_MAX else '')
        print('%-40s relP=%.2e dphi(truths)=%.3f cyc%s'
              % (uid, rel, dphi_truths, flag))
        for m in METHODS:
            v = star_rec['methods'][m]
            mark = '' if v['chen'] == v['gaia'] else '  *** CHANGES'
            if v['chen'] != v['gaia'] or m == 'ftp_mb':
                print('    %-10s chen=%-9s gaia=%-9s%s'
                      % (m, v['chen'], v['gaia'], mark))
    print('verdict changes (star x method cells): %d / %d'
          % (dual['n_verdict_changes'], len(dt_uids) * len(METHODS)))

    # ------------------------------------------------- adjudication queue (4)
    queue = [r for r in rows if r['method'] == 'ftp_mb'
             and r['class'] not in ('exact', '2f', 'f/2')]
    print('\n=== (4) adjudication queue: ftp_mb alias/miss rows ===')
    for r in queue:
        print('%-40s role=%-16s type=%-4s class=%-9s P_true=%.5f P_rec=%.5f '
              'dphi=%.1f' % (r['uid'], r['role'], r['type'] or r['gaia_type'],
                             r['class'], r['P_true'], r['P_rec'],
                             r['dphi_cycles']))

    # --------------------------------------------------- escalate check (6)
    best_mb = max(MULTIBAND, key=lambda m: rates[m]['science']['rate'])
    best_rate = rates[best_mb]['science']['rate']
    escalate = bool(best_rate < 0.85)
    print('\n=== (6) ESCALATE check: best multiband pre-adjudication on '
          'science = %s at %.1f%% -> escalate=%s'
          % (best_mb, 100 * best_rate, escalate))

    grid_meta = {
        'df_range': [float(np.min(d['df'])), float(np.max(d['df']))],
        'pts_per_rayleigh_range': [float(np.min(d['pts_per_rayleigh'])),
                                   float(np.max(d['pts_per_rayleigh']))],
        'T_grid_days_range': [float(np.min(d['T_grid_days'])),
                              float(np.max(d['T_grid_days']))],
        'f_min': 1.0, 'f_max': 5.0,
    }
    out = {
        'spec': {'rtol': RTOL, 'dphi_max': DPHI_MAX,
                 'criterion': 'fractional AND phase_coherence (E4 brief)',
                 'harmonics': ['2f', 'f/2'],
                 'beats_cyc_per_day': [1.0, 1.0 / YEAR, 1.0 / LUNAR_YEAR],
                 'recovered_pre': 'exact or harmonic',
                 'truth': {'science': 'chen_period',
                           'control_antijoin': 'gaia_period',
                           'control_orig': 'gaia_period'}},
        'rates_pre_adjudication': rates,
        'breakdowns': breakdowns,
        'strata_science': strata,
        'dual_truth': dual,
        'adjudication_queue_ftp_mb': [r['uid'] for r in queue],
        'escalate': {'best_multiband_method': best_mb,
                     'science_rate_pre': best_rate,
                     'threshold': 0.85, 'escalate': escalate},
        'grid_meta': grid_meta,
    }
    with open(RATES_JSON, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('\nwrote %s and %s' % (SCORES_CSV, RATES_JSON))


if __name__ == '__main__':
    main()
