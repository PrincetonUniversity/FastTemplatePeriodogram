"""Adversarial parity hunt: current scan vs pre-change scan vs eigvals ref.

Focus (math-correctness lens):
  (a) Bernstein candidate filter (step 3c) incl. K.18 sign-flip cases
  (b) deep-dip triage gate r > (2H dtheta)^2/2 (band gate0 < r < 0.15)
  (c) densified rescan / nesting
  (d) max-merge
"""
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


def eig_reference(YM, MM, AC, H, ybar, YY, pos):
    nf = YM.shape[0]
    pw = np.zeros(nf)
    aa = np.zeros(nf)
    for i in range(nf):
        prm, p, _ = cur.roots_from_YM_MM(
            pol.Polynomial(YM[i]), pol.Polynomial(MM[i]), AC[i], H, ybar, YY,
            positive_amplitude=pos)
        pw[i] = p
        aa[i] = prm.a
    return pw, aa


def make_dataset(rng, kind, H):
    """Return t, y, dy engineered toward |MM| dips."""
    if kind == 'clump':
        # nightly-cadence-like: phases clump at integer-ish frequencies
        n_nights = rng.integers(8, 20)
        nights = np.sort(rng.choice(np.arange(1, 200), n_nights, replace=False))
        t = nights + rng.normal(0, 0.01, n_nights)
        t = np.sort(np.concatenate([t, t + rng.uniform(0.01, 0.03, n_nights)]))
    elif kind == 'sparse':
        N = 2 * H + 2 + rng.integers(0, 3)
        t = np.sort(rng.uniform(0, 100, N))
    elif kind == 'clusters':
        # k distinct phase clusters at f ~ 1: rank deficiency when k < dof
        k = rng.integers(2, 2 * H + 2)
        centers = rng.uniform(0, 50, k)
        t = np.sort(np.concatenate(
            [c + rng.normal(0, 0.002, rng.integers(2, 6)) for c in centers]))
    else:  # 'random'
        N = rng.integers(30, 80)
        t = np.sort(rng.uniform(0, 100, N))
    y = rng.normal(0, 1, len(t)) + 0.8 * np.sin(2 * np.pi * 1.3 * t + 0.3)
    dy = np.full(len(t), 0.5)
    if kind == 'wconc':
        dy = rng.uniform(0.3, 1.0, len(t))
        j = rng.choice(len(t), 2, replace=False)
        dy[j] = 1e-4  # weight concentration in 2 points
    return t, y, dy


def run_case(seed, kind, H, pos, nfreq=600):
    rng = np.random.default_rng(seed)
    t, y, dy = make_dataset(rng, kind, H)
    # template: decaying Fourier coefficients
    cn = np.array([0.5 / (1 + h) for h in range(H)])
    sn = np.array([0.3 / (1 + h) ** 1.5 * (-1) ** h for h in range(H)])
    freqs = np.linspace(0.2, 3.0, nfreq) + 1e-4
    w = weights(dy)
    ybar = w @ y
    YY = w @ (y - ybar) ** 2
    sums = stack_summations(direct_summations(t, y, w, freqs, H))
    YM, MM, AC = cur.batched_YM_MM_from_sums(cn, sn, sums)

    M = max(cur._SCAN_MIN_ANGLES, cur._SCAN_ANGLES_PER_H * H)
    absM = np.abs(cur._eval_polys_on_circle(MM, M))
    with np.errstate(divide='ignore', invalid='ignore'):
        r_row = np.min(absM, axis=1) / np.max(absM, axis=1)

    pl_c, pw_c, _ = cur.scan_polish_from_coefs(YM, MM, AC, H, ybar, YY,
                                               positive_amplitude=pos)
    pl_b, pw_b, _ = base.scan_polish_from_coefs(YM, MM, AC, H, ybar, YY,
                                                positive_amplitude=pos)
    pw_e, a_e = eig_reference(YM, MM, AC, H, ybar, YY, pos)
    a_c = np.array([p.a for p in pl_c])
    a_b = np.array([p.a for p in pl_b])
    return dict(r=r_row, pw_c=pw_c, pw_b=pw_b, pw_e=pw_e,
                a_c=a_c, a_b=a_b, a_e=a_e, H=H, pos=pos,
                kind=kind, seed=seed)


def main():
    gate = {}
    findings = []
    TOL = 1e-9
    rows_tot = 0
    band_counts = {}
    for H in (2, 5, 8):
        Mg = max(cur._SCAN_MIN_ANGLES, cur._SCAN_ANGLES_PER_H * H)
        gate[H] = 0.5 * (4.0 * np.pi * H / Mg) ** 2
    for seed in range(25):
        for kind in ('clump', 'sparse', 'clusters', 'wconc', 'random'):
            for H in (2, 5, 8):
                for pos in (False, True):
                    d = run_case(1000 * seed + H, kind, H, pos)
                    r = d['r']
                    rows_tot += len(r)
                    g = gate[H]
                    bands = [('deep', r <= g), ('gateband', (r > g) & (r < 0.15)),
                             ('mid', (r >= 0.15) & (r < 0.5)), ('ok', r >= 0.5)]
                    for name, m in bands:
                        band_counts[name] = band_counts.get(name, 0) + int(m.sum())
                    # regression vs base scan
                    dreg = d['pw_b'] - d['pw_c']
                    bad = np.where(dreg > TOL)[0]
                    for i in bad:
                        findings.append(('REG_VS_BASE', d['kind'], d['seed'], H,
                                         pos, i, r[i], d['pw_b'][i], d['pw_c'][i],
                                         d['pw_e'][i]))
                    # miss vs eig where base agreed with eig
                    de = d['pw_e'] - d['pw_c']
                    db = d['pw_e'] - d['pw_b']
                    bad2 = np.where((de > TOL) & (np.abs(db) < TOL))[0]
                    for i in bad2:
                        findings.append(('MISS_VS_EIG', d['kind'], d['seed'], H,
                                         pos, i, r[i], d['pw_b'][i], d['pw_c'][i],
                                         d['pw_e'][i]))
                    if pos:
                        # feasibility: eig found positive amp; current returns
                        # negative-amp fit with materially different power
                        feas = np.where((d['a_e'] >= 0) & (d['a_c'] < -1e-12)
                                        & (np.abs(d['pw_c'] - d['pw_e']) > TOL))[0]
                        for i in feas:
                            base_ok = (d['a_b'][i] >= 0)
                            findings.append(('K18_INFEAS', d['kind'], d['seed'],
                                             H, pos, i, r[i], d['pw_b'][i],
                                             d['pw_c'][i], d['pw_e'][i],
                                             'base_pos={0}'.format(base_ok)))
    print('rows scanned:', rows_tot)
    print('band counts:', band_counts)
    print('findings:', len(findings))
    for f in findings[:80]:
        print(f)


if __name__ == '__main__':
    main()
