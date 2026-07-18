"""Extended targeted hunt with per-miss attribution (FILTER vs POLISH)."""
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

from probe_hunt import make_dataset, eig_reference

TOL = 1e-9
count = dict(rows=0, deep=0, gateband=0, mid=0, ok=0)
misses = []

for seed in range(25, 90):
    for kind in ('clusters', 'wconc'):
        for H in (2, 5, 8):
            rng = np.random.default_rng(7000000 + 1000 * seed + H)
            t, y, dy = make_dataset(rng, kind, H)
            cn = np.array([0.5 / (1 + h) for h in range(H)])
            sn = np.array([0.3 / (1 + h) ** 1.5 * (-1) ** h
                           for h in range(H)])
            freqs = np.linspace(0.2, 3.0, 600) + 1e-4
            w = weights(dy)
            ybar = w @ y
            YY = w @ (y - ybar) ** 2
            sums = stack_summations(direct_summations(t, y, w, freqs, H))
            YM, MM, AC = cur.batched_YM_MM_from_sums(cn, sn, sums)
            M = max(cur._SCAN_MIN_ANGLES, cur._SCAN_ANGLES_PER_H * H)
            gate0 = 0.5 * (4.0 * np.pi * H / M) ** 2
            absM = np.abs(cur._eval_polys_on_circle(MM, M))
            with np.errstate(divide='ignore', invalid='ignore'):
                r_row = np.min(absM, axis=1) / np.max(absM, axis=1)
            count['rows'] += len(r_row)
            count['deep'] += int((r_row <= gate0).sum())
            count['gateband'] += int(((r_row > gate0) & (r_row < 0.15)).sum())
            count['mid'] += int(((r_row >= 0.15) & (r_row < 0.5)).sum())
            count['ok'] += int((r_row >= 0.5).sum())
            for pos in (False, True):
                pl_c, pw_c, _ = cur.scan_polish_from_coefs(
                    YM, MM, AC, H, ybar, YY, positive_amplitude=pos)
                pl_b, pw_b, _ = base.scan_polish_from_coefs(
                    YM, MM, AC, H, ybar, YY, positive_amplitude=pos)
                sus = np.where((pw_b - pw_c > TOL))[0]
                for i in sus:
                    # eig + attribution only on suspicious rows (cheap)
                    prm_e, pw_e, _ = cur.roots_from_YM_MM(
                        pol.Polynomial(YM[i]),
                        pol.Polynomial(MM[i]), AC[i], H, ybar, YY,
                        positive_amplitude=pos)
                    saved = base._exact_root_fallback
                    base._exact_root_fallback = lambda *a, **k: None
                    _, pw_op, _ = base.scan_polish_from_coefs(
                        YM[i:i + 1], MM[i:i + 1], AC[i:i + 1], H, ybar, YY,
                        positive_amplitude=pos)
                    base._exact_root_fallback = saved
                    attr = ('FILTER' if pw_op[0] - pw_c[i] > TOL
                            else 'POLISH/OTHER')
                    band = ('deep' if r_row[i] <= gate0 else
                            'gateband' if r_row[i] < 0.15 else
                            'mid' if r_row[i] < 0.5 else 'ok')
                    misses.append((kind, seed, H, pos, int(i),
                                   float(r_row[i]), band, attr,
                                   float(pw_b[i]), float(pw_c[i]),
                                   float(pw_e), float(pw_op[0])))
                if pos:
                    a_c = np.array([p.a for p in pl_c])
                    a_b = np.array([p.a for p in pl_b])
                    flip = np.where((a_b >= 0) & (a_c < -1e-12) &
                                    (np.abs(pw_c - pw_b) > TOL))[0]
                    for i in flip:
                        misses.append((kind, seed, H, pos, int(i),
                                       float(r_row[i]), 'K18FLIP', '-',
                                       float(pw_b[i]), float(pw_c[i]),
                                       np.nan, np.nan))

print('counts:', count)
print('misses:', len(misses))
for m in misses:
    print(m)
