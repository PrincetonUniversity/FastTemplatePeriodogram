"""The never-run os-doubling H8-grid convergence spot check.

Reconstructs the exact B1 population (deterministic, seed=0), rescored at H8
with os=3 (must reproduce stored P_rec -- reconstruction sanity) and os=6.
Also H4 at os=6. Serial, 1 BLAS thread. Writes incremental JSONL.
"""
import os, sys, json
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np

ROOT = '/Users/johnhoffman/Documents/fast_template_periodogram/FastTemplatePeriodogram'
sys.path.insert(0, ROOT + '/experiments/paper2_gpu/trackB')
import lib
from b1_recovery_vs_H import build_population, SEARCH_BAND

SUBTYPE = sys.argv[1] if len(sys.argv) > 1 else 'RRab'
N_SUB = int(sys.argv[2]) if len(sys.argv) > 2 else 64
OUT = sys.argv[3]

stored = json.load(open(ROOT + '/experiments/paper2_gpu/trackB/output/b1_%s_m21_n8.json'
                        % SUBTYPE.lower()))
voc_stars, tru_stars = lib.split_stars(SUBTYPE, seed=0)
vocab8 = lib.detection_vocab(SUBTYPE, 8, K=4, seed=0, stars=voc_stars)
vocab4 = lib.detection_vocab(SUBTYPE, 4, K=4, seed=0, stars=voc_stars)
srcs = build_population(SUBTYPE, 256, 8, 21.0, 1300.0, 250.0, 0, tru_stars)
f_lo, f_hi = SEARCH_BAND[SUBTYPE]

with open(OUT, 'w') as fh:
    for i in range(N_SUB):
        src = srcs[i]
        st = stored['sources'][i]
        assert abs(src['P_true'] - st['P_true']) < 1e-12, (i, src['P_true'], st['P_true'])
        assert src['star'] == st['star'], i
        row = {'i': i, 'P_true': src['P_true'], 'T': src['baseline']}
        for tag, vocab, H, osamp in (('H8_os3', vocab8, 8, 3.0),
                                     ('H8_os6', vocab8, 8, 6.0),
                                     ('H4_os3', vocab4, 4, 3.0),
                                     ('H4_os6', vocab4, 4, 6.0)):
            grid = lib.converged_grid(f_lo, f_hi, H, src['baseline'], oversample=osamp)
            Prec, _ = lib.ftp_best_period(vocab, src, grid)
            s = lib.score(Prec, src['P_true'], src['baseline'])
            row[tag] = {'P_rec': Prec, 'rec': bool(s['frac_recovered']),
                        'exact': bool(s['frac_exact']),
                        'phase': bool(s['phase_recovered'])}
        row['match_stored_H8'] = bool(abs(row['H8_os3']['P_rec'] - st['H8']['P_rec']) < 1e-10)
        row['match_stored_H4'] = bool(abs(row['H4_os3']['P_rec'] - st['H4']['P_rec']) < 1e-10)
        fh.write(json.dumps(row) + '\n')
        fh.flush()
print('done', N_SUB)
