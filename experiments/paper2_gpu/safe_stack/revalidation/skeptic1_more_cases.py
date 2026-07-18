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

for seed, row, kind, H, pos in [(20002, 20, 'clusters', 2, False),
                                (22002, 530, 'clusters', 2, False),
                                (8002, 367, 'clusters', 2, False)]:
    rng = np.random.default_rng(seed)
    t, y, dy = make_dataset(rng, kind, H)
    cn = np.array([0.5 / (1 + h) for h in range(H)])
    sn = np.array([0.3 / (1 + h) ** 1.5 * (-1) ** h for h in range(H)])
    freqs = np.linspace(0.2, 3.0, 600) + 1e-4
    w = weights(dy); ybar = w @ y; YY = w @ (y - ybar) ** 2
    sums = stack_summations(direct_summations(t, y, w, freqs, H))
    YM, MM, AC = cur.batched_YM_MM_from_sums(cn, sn, sums)
    _, pw_c, _ = cur.scan_polish_from_coefs(YM, MM, AC, H, ybar, YY, positive_amplitude=pos)
    _, pw_b, _ = base.scan_polish_from_coefs(YM, MM, AC, H, ybar, YY, positive_amplitude=pos)
    _, pw_e, _ = cur.roots_from_YM_MM(pol.Polynomial(YM[row]), pol.Polynomial(MM[row]),
                                      AC[row], H, ybar, YY, positive_amplitude=pos)
    d = pw_b - pw_c
    reg = np.where(d > 1e-9)[0]
    print('seed={0} row={1}: cur={2:.9f} base={3:.9f} eig={4:.9f} '
          'deficit={5:.3e}; all reg rows this case: {6} (max {7:.3e})'.format(
              seed, row, pw_c[row], pw_b[row], pw_e, pw_e - pw_c[row],
              list(reg), d.max()))
