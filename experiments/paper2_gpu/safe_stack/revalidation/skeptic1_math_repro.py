"""Skeptic #1 (math lens): independent repro of the step-3c filter finding.

Reconstructs seed 6002 / kind='clusters' / H=2 / row 521 from scratch and
checks, WITHOUT trusting the hunt probes' printouts:
  A. current scan power vs pre-change (1529c33) scan power vs eigvals ref
  B. brute-force truth: dense-grid (2^20) + bounded refinement max of
     P(theta) = Re(YM^2/MM)/YY  -- validates the eigvals number independently
  C. mechanism: recompute the step-3c keep mask; identify the dropped
     candidate, polish it in isolation (clamped +-1 grid step, same Newton),
     show its polished value beats the shipped power
  D. stage-1 triage gate: r vs gate0 (does the row escape the fallback?)
  E. K.18 status of this case (positive_amplitude flag used by the hunt)
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

SEED, ROW, H, KIND = 6002, 521, 2, 'clusters'

# --- rebuild dataset exactly as probe_hunt.make_dataset('clusters', H=2) ---
rng = np.random.default_rng(SEED)
k = rng.integers(2, 2 * H + 2)
centers = rng.uniform(0, 50, k)
t = np.sort(np.concatenate(
    [c + rng.normal(0, 0.002, rng.integers(2, 6)) for c in centers]))
y = rng.normal(0, 1, len(t)) + 0.8 * np.sin(2 * np.pi * 1.3 * t + 0.3)
dy = np.full(len(t), 0.5)
print('dataset: N={0}, k_clusters={1}'.format(len(t), k))

cn = np.array([0.5 / (1 + h) for h in range(H)])
sn = np.array([0.3 / (1 + h) ** 1.5 * (-1) ** h for h in range(H)])
freqs = np.linspace(0.2, 3.0, 600) + 1e-4
w = weights(dy)
ybar = w @ y
YY = w @ (y - ybar) ** 2
sums = stack_summations(direct_summations(t, y, w, freqs, H))
YM, MM, AC = cur.batched_YM_MM_from_sums(cn, sn, sums)

for POS in (False, True):
    print('\n================ positive_amplitude={0} ================'.format(POS))
    _, pw_c, _ = cur.scan_polish_from_coefs(YM, MM, AC, H, ybar, YY,
                                            positive_amplitude=POS)
    _, pw_b, _ = base.scan_polish_from_coefs(YM, MM, AC, H, ybar, YY,
                                             positive_amplitude=POS)
    prm_e, pw_e, _ = cur.roots_from_YM_MM(
        pol.Polynomial(YM[ROW]), pol.Polynomial(MM[ROW]), AC[ROW], H, ybar,
        YY, positive_amplitude=POS)
    print('A. current scan  P[{0}] = {1:.9f}'.format(ROW, pw_c[ROW]))
    print('   base    scan  P[{0}] = {1:.9f}'.format(ROW, pw_b[ROW]))
    print('   eigvals ref   P[{0}] = {1:.9f}  (a={2:+.4f})'.format(
        ROW, pw_e, prm_e.a))
    print('   current-vs-eig deficit = {0:.3e};  current-vs-base = {1:.3e}'
          .format(pw_e - pw_c[ROW], pw_b[ROW] - pw_c[ROW]))
    # count all rows where current < base - 1e-9 (regression rows this case)
    reg = np.where(pw_b - pw_c > 1e-9)[0]
    print('   rows with current < base - 1e-9 in this 600-row case:', reg,
          'max deficit {0:.3e}'.format((pw_b - pw_c).max()))

# --- B: brute-force truth for the row (independent of eigvals) -----------
YMi, MMi = YM[ROW], MM[ROW]
th_dense = np.linspace(0, 2 * np.pi, 1 << 20, endpoint=False)
ph = np.exp(1j * th_dense)
Yd = pol.polynomial.polyval(ph, YMi)
Md = pol.polynomial.polyval(ph, MMi)
with np.errstate(divide='ignore', invalid='ignore'):
    Pd = np.real(Yd * Yd / Md) / YY
Pd[~np.isfinite(Pd)] = -np.inf
g0 = int(np.argmax(Pd))
from scipy.optimize import minimize_scalar
def negP(th):
    p = np.exp(1j * th)
    return -np.real(pol.polynomial.polyval(p, YMi) ** 2
                    / pol.polynomial.polyval(p, MMi)).real / YY
dt = th_dense[1] - th_dense[0]
res = minimize_scalar(negP, bounds=(th_dense[g0] - dt, th_dense[g0] + dt),
                      method='bounded', options=dict(xatol=1e-15))
print('\nB. brute-force true max P = {0:.9f} at theta={1:.6f}'.format(
    -res.fun, res.x))

# --- C: mechanism -- recompute the 3c filter state for this row ----------
M = max(cur._SCAN_MIN_ANGLES, cur._SCAN_ANGLES_PER_H * H)
Yv = cur._eval_polys_on_circle(YMi[None], M)[0]
Mv = cur._eval_polys_on_circle(MMi[None], M)[0]
with np.errstate(divide='ignore', invalid='ignore'):
    Pg = np.real(Yv * Yv / Mv) / YY
Pg[~np.isfinite(Pg)] = -np.inf
ismax = ((Pg >= np.roll(Pg, 1)) & (Pg >= np.roll(Pg, -1)) & (Pg > -np.inf))
gcols = np.where(ismax)[0]
absM = np.abs(Mv)
r = absM.min() / absM.max()
ismin = ((absM <= np.roll(absM, 1)) & (absM <= np.roll(absM, -1)) &
         (absM < cur._SCAN_DIP_RTOL * absM.max()))
dcols = np.where(ismin)[0]

# dip candidate refinement (as in _scan_pass) -- Newton on |MM|^2, then P
kM = np.arange(MMi.shape[0])
th_dip = (2 * np.pi / M) * dcols.astype(float)
for _ in range(cur._SCAN_NEWTON_STEPS):
    p = np.exp(1j * th_dip)
    Mm = pol.polynomial.polyval(p, MMi)
    M1 = pol.polynomial.polyval(p, MMi * kM)
    M2 = pol.polynomial.polyval(p, MMi * kM * kM)
    q1 = 2.0 * np.real(1j * M1 * np.conj(Mm))
    q2 = 2.0 * (np.abs(M1) ** 2 - np.real(np.conj(Mm) * M2))
    th_dip = np.clip(th_dip - q1 / q2, (2*np.pi/M)*dcols - 2*np.pi/M,
                     (2*np.pi/M)*dcols + 2*np.pi/M)
pdip = np.exp(1j * th_dip)
P_dip = np.real(pol.polynomial.polyval(pdip, YMi) ** 2
                / pol.polynomial.polyval(pdip, MMi)).real / YY

P0_all = np.concatenate([Pg[gcols], P_dip])
rowmax = P0_all.max()
gain = (np.pi / M) ** 2 * (2.0 * H) ** 2 / 2.0
rowabs = np.where(np.isfinite(Pg), np.abs(Pg), 0.0).max()
thr = rowmax - 2.0 * gain * rowabs
print('\nC. M={0} r={1:.4e} gain={2:.4e} rowabs={3:.4f} rowmax={4:.9f} '
      'thr={5:.9f}'.format(M, r, gain, rowabs, rowmax, thr))
print('   grid-max candidates:')
kY = np.arange(YMi.shape[0])
half_window = 2 * np.pi / M
for g in gcols:
    kept = Pg[g] >= thr
    # replicate the shipped Newton polish for this single candidate
    th = 2 * np.pi * g / M
    th0 = th
    for _ in range(cur._SCAN_NEWTON_STEPS):
        p = np.exp(1j * th)
        Y = pol.polynomial.polyval(p, YMi)
        Y1 = pol.polynomial.polyval(p, YMi * kY)
        Y2 = pol.polynomial.polyval(p, YMi * kY * kY)
        Mm = pol.polynomial.polyval(p, MMi)
        M1 = pol.polynomial.polyval(p, MMi * kM)
        M2 = pol.polynomial.polyval(p, MMi * kM * kM)
        dP, d2P = cur._scan_dP_d2P(np.atleast_1d(Y), np.atleast_1d(Y1),
                                   np.atleast_1d(Y2), np.atleast_1d(Mm),
                                   np.atleast_1d(M1), np.atleast_1d(M2))
        st = (dP / d2P)[0]
        if not np.isfinite(st):
            st = 0.0
        th = np.clip(th - st, th0 - half_window, th0 + half_window)
    p = np.exp(1j * th)
    Pp = np.real(pol.polynomial.polyval(p, YMi) ** 2
                 / pol.polynomial.polyval(p, MMi)).real / YY
    dist = np.min(np.abs(((g - dcols + M / 2) % M) - M / 2)) if len(dcols) else -1
    print('   g={0:4d} P0={1:.9f} kept={2!s:5} newton_polished={3:.9f} '
          'improve={4:+.3e} dist_to_dip_steps={5:.0f}'.format(
              g, Pg[g], bool(kept), Pp, Pp - Pg[g], dist))
print('   dip candidates: cols={0} P_dip={1}'.format(
    dcols, np.array2string(P_dip, precision=9)))
print('   allowed improvement margin 2*gain*rowabs = {0:.3e}'.format(
    2 * gain * rowabs))

# --- D: stage-1 triage gate ----------------------------------------------
gate0 = 0.5 * (4.0 * np.pi * H / M) ** 2
print('\nD. gate0={0:.4e}, r={1:.4e}  -> flagged={2}, stage1_gate_pass={3} '
      '(gate pass => scan result stands, no fallback)'.format(
          gate0, r, r < cur._SCAN_EXACT_RTOL, r > gate0))
