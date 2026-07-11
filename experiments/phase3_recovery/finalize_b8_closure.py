#!/usr/bin/env python
"""WP B8-closure arm (b) finalize: refined-grid vs 10k-grid acceptance check.

Fresh consumer for the arm-(b) grid-closure pods (``g-refine-*``): the same
10k production grid re-run with ``--refine`` local peak refinement (plan of
record in ``SUMMARY_c-grid20k.md``), N-sweep stage only, ``--fixed-k 2``.
Pairs every refined per-source mask against its arm-(a) 10k baseline on the
IDENTICAL frozen sources (asserted via ``p_true``) and answers the WP
acceptance question: **does any headline contrast move >1 SE (~0.017)?**

Emits:

* ``rerun_202606/SUMMARY_g-refine.md`` -- per-cell paired McNemar table
  (refined vs 10k), pooled sesar (3 seeds) + bv, the arm-(c) flagged cells
  marked, and the headline-movement verdict;
* ``rerun_202606/aggregate_g_refine.json`` -- machine-readable per-cell rows.

Conventions follow ``finalize_b8.py``: exact McNemar on per-source masks for
paired significance; Wilson-derived SE of the pooled 10k rate for the ">1 SE"
movement benchmark (0.017-0.018 at mid rates for the 768-source sesar pool).

    python finalize_b8_closure.py        # writes both outputs, prints verdict
"""
import json
import os

import numpy as np

from finalize_b8 import wilson_se
from ftperiodogram.validation import mcnemar_test

HERE = os.path.dirname(os.path.abspath(__file__))
RERUN = os.path.join(HERE, 'rerun_202606')
RAW = os.path.join(RERUN, 'raw')

METHODS = ('ftp_pam', 'ftp_greedy', 'gls', 'mhls', 'mbls', 'ce')
N_VALUES = (4, 8, 12, 16, 24, 40)
# arm-(c) 20k-vs-10k significant cells (SUMMARY_c-grid20k.md) -- the cells B8
# flagged as grid-limited; these made the 10k rates "lower bounds"
FLAGGED = {('ftp_pam', 8), ('mhls', 8), ('mhls', 12), ('mhls', 16),
           ('mhls', 40), ('mbls', 12), ('ce', 12), ('ce', 16)}
# (refined tag, 10k baseline tag, seed) pairs
PAIRS = {'sesar': [('g-refine-sesar-%d' % s, 'a-sesar-%d' % s, s)
                   for s in (0, 1, 2)],
         'bv': [('g-refine-bv-0', 'a-bv-0', 0)]}


def _mask(tag, seed, method, n):
    z = np.load(os.path.join(RAW, tag, 'out', 'per_source',
                             'seed%d__%s__nsweep-N%d.npz' % (seed, method, n)))
    return z['recovered'].astype(bool), np.asarray(z['p_true'], dtype=float)


def collect(universe):
    """Per-cell rows for one universe, pooled over its (refined, base) pairs."""
    rows = []
    pairs = [p for p in PAIRS[universe]
             if os.path.isdir(os.path.join(RAW, p[0], 'out', 'per_source'))]
    if not pairs:
        return rows, 0, 0
    for method in METHODS:
        for n in N_VALUES:
            ref_all, base_all = [], []
            for rtag, btag, seed in pairs:
                r, p_r = _mask(rtag, seed, method, n)
                b, p_b = _mask(btag, seed, method, n)
                if not np.allclose(p_r, p_b):
                    raise SystemExit('p_true mismatch %s vs %s (%s N=%d): '
                                     'pairing broken' % (rtag, btag, method, n))
                ref_all.append(r)
                base_all.append(b)
            ref = np.concatenate(ref_all)
            base = np.concatenate(base_all)
            b_only, r_only, p = mcnemar_test(base, ref)   # (base-only, ref-only, p)
            n_pool = ref.size
            r_ref, r_base = float(ref.mean()), float(base.mean())
            se = wilson_se(r_base, n_pool)
            delta = r_ref - r_base
            rows.append({'universe': universe, 'method': method, 'N': n,
                         'n_pool': n_pool, 'refined': r_ref, 'base10k': r_base,
                         'delta': delta, 'se_base': se,
                         'delta_over_se': (abs(delta) / se if se else 0.0),
                         'ref_only': r_only, 'base_only': b_only, 'p': p,
                         'flagged': (method, n) in FLAGGED})
    return rows, len(pairs), len(PAIRS[universe])


def contrast_moves(rows):
    """FTP(PAM)-GLS headline-contrast movement per (universe, N)."""
    idx = {(r['universe'], r['method'], r['N']): r for r in rows}
    out = []
    for universe in ('sesar', 'bv'):
        for n in N_VALUES:
            f = idx.get((universe, 'ftp_pam', n))
            g = idx.get((universe, 'gls', n))
            if not (f and g):
                continue
            c_ref = f['refined'] - g['refined']
            c_base = f['base10k'] - g['base10k']
            se = float(np.hypot(f['se_base'], g['se_base']))
            out.append({'universe': universe, 'N': n, 'contrast_refined': c_ref,
                        'contrast_10k': c_base, 'move': c_ref - c_base,
                        'se': se,
                        'move_over_se': (abs(c_ref - c_base) / se if se else 0.0)})
    return out


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def main():
    all_rows, meta = [], {}
    for universe in ('sesar', 'bv'):
        rows, got, want = collect(universe)
        all_rows += rows
        meta[universe] = {'pairs_found': got, 'pairs_expected': want}
    if not all_rows:
        raise SystemExit('no g-refine results under %s' % RAW)
    contrasts = contrast_moves(all_rows)

    moved = [r for r in all_rows if r['delta_over_se'] > 1.0]
    flagged_rows = [r for r in all_rows if r['flagged']]
    sig = [r for r in all_rows if r['p'] < 0.05]
    c_moved = [c for c in contrasts if c['move_over_se'] > 1.0]

    body = md_table(
        ['universe', 'method', 'N', 'refined', '10k', 'delta', '|d|/SE',
         'ref-only', '10k-only', 'p', 'flagged'],
        [[r['universe'], r['method'], r['N'], '%.3f' % r['refined'],
          '%.3f' % r['base10k'], '%+.3f' % r['delta'],
          '%.1f' % r['delta_over_se'], r['ref_only'], r['base_only'],
          '%.3g' % r['p'], 'FLAG' if r['flagged'] else '']
         for r in all_rows])
    cbody = md_table(
        ['universe', 'N', 'FTP-GLS refined', 'FTP-GLS 10k', 'move', '|move|/SE'],
        [[c['universe'], c['N'], '%+.3f' % c['contrast_refined'],
          '%+.3f' % c['contrast_10k'], '%+.3f' % c['move'],
          '%.1f' % c['move_over_se']] for c in contrasts])

    verdict = (
        'VERDICT: %d/%d cells moved >1 SE (McNemar p<0.05 in %d); '
        'flagged arm-(c) cells moved >1 SE: %d/%d; FTP-GLS headline contrast '
        'moved >1 SE at %d/%d (universe, N) cells.'
        % (len(moved), len(all_rows), len(sig),
           sum(1 for r in flagged_rows if r['delta_over_se'] > 1.0),
           len(flagged_rows), len(c_moved), len(contrasts)))

    with open(os.path.join(RERUN, 'SUMMARY_g-refine.md'), 'w') as fh:
        fh.write('# Arm (b) grid closure -- N-sweep + peak refinement vs the '
                 '10k grid, PAIRED exact McNemar on identical sources\n\n'
                 'Refined = same 10k production grid + local peak refinement '
                 '(RefinedEstimator defaults), --stages n_sweep --fixed-k 2; '
                 'pairs: sesar pooled %d/%d seeds, bv %d/%d.\n\n'
                 % (meta['sesar']['pairs_found'], meta['sesar']['pairs_expected'],
                    meta['bv']['pairs_found'], meta['bv']['pairs_expected'])
                 + body
                 + '\n\n## FTP(PAM)-GLS headline-contrast movement\n\n' + cbody
                 + '\n\n' + verdict + '\n')
    with open(os.path.join(RERUN, 'aggregate_g_refine.json'), 'w') as fh:
        json.dump({'cells': all_rows, 'contrasts': contrasts, 'meta': meta},
                  fh, indent=2)
    print(verdict)
    print('wrote rerun_202606/SUMMARY_g-refine.md + aggregate_g_refine.json')


if __name__ == '__main__':
    main()
