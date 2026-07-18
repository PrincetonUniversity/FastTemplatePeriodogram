"""For deep-band misses: is the DENSE pass miss filter-caused or polish-caused?"""
import sys, os, importlib.util
import numpy as np
import numpy.polynomial as pol

REPO = '/Users/johnhoffman/Documents/fast_template_periodogram/FastTemplatePeriodogram'
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from ftperiodogram.summations import direct_summations, stack_summations
from ftperiodogram.utils import weights
from ftperiodogram import core as cur

spec = importlib.util.spec_from_file_location(
    'ftperiodogram.core_base', os.path.join(SCRATCH, 'core_base.py'))
base = importlib.util.module_from_spec(spec)
sys.modules['ftperiodogram.core_base'] = base
spec.loader.exec_module(base)

from probe_hunt import make_dataset

CASES = [(38, [282, 580, 144, 83, 429]), (76, [311, 73]), (69, [122])]
H = 2

for seed, rows in CASES:
    rng = np.random.default_rng(7000000 + 1000 * seed + H)
    t, y, dy = make_dataset(rng, 'clusters', H)
    cn = np.array([0.5 / (1 + h) for h in range(H)])
    sn = np.array([0.3 / (1 + h) ** 1.5 * (-1) ** h for h in range(H)])
    freqs = np.linspace(0.2, 3.0, 600) + 1e-4
    w = weights(dy)
    ybar = w @ y
    YY = w @ (y - ybar) ** 2
    sums = stack_summations(direct_summations(t, y, w, freqs, H))
    YM, MM, AC = cur.batched_YM_MM_from_sums(cn, sn, sums)
    M = max(cur._SCAN_MIN_ANGLES, cur._SCAN_ANGLES_PER_H * H)
    gate0 = 0.5 * (4.0 * np.pi * H / M) ** 2
    for row in rows:
        YMi, MMi, ACi = YM[row:row + 1], MM[row:row + 1], AC[row:row + 1]
        _, pw_e, _ = cur.roots_from_YM_MM(pol.Polynomial(YMi[0]),
                                          pol.Polynomial(MMi[0]), ACi[0], H,
                                          ybar, YY)
        _, pw_c, _ = cur.scan_polish_from_coefs(YMi, MMi, ACi, H, ybar, YY)
        _, _, _, mmn, mmx = cur._scan_pass(YMi, MMi, ACi, H, ybar, YY, False,
                                           M, cur._SCAN_NEWTON_STEPS)
        r = float(mmn[0] / mmx[0])
        M_need = (4.0 * np.pi * H) / np.sqrt(2.0 * r)
        Md = int(np.exp2(np.ceil(np.log2(max(M_need, 2.0 * M)))))
        gate_d = 0.5 * (4.0 * np.pi * H / Md) ** 2
        _, pw_nd, _, mmn_d, mmx_d = cur._scan_pass(YMi, MMi, ACi, H, ybar,
                                                   YY, False, Md,
                                                   cur._SCAN_NEWTON_STEPS)
        r_d = float(mmn_d[0] / mmx_d[0])
        saved = base._exact_root_fallback
        base._exact_root_fallback = lambda *a, **k: None
        _, pw_od, _ = base.scan_polish_from_coefs(YMi, MMi, ACi, H, ybar, YY,
                                                  n_angles=Md)
        base._exact_root_fallback = saved
        attr = ('FILTER@DENSE' if pw_od[0] - pw_nd[0] > 1e-9 else
                'POLISH@DENSE' if pw_e - pw_od[0] > 1e-9 else 'other')
        print('seed={0} row={1} r={2:.3e} Md={3} r_d={4:.3e} gate_d={5:.3e} '
              'pass={6}'.format(seed, row, r, Md, r_d, gate_d, r_d > gate_d))
        print('   eig={0:.12g} cur={1:.12g} newdense={2:.12g} '
              'olddense={3:.12g} -> {4}'.format(pw_e, pw_c[0], pw_nd[0],
                                                pw_od[0], attr))
