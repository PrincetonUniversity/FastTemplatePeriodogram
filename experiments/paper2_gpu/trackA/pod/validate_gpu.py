#!/usr/bin/env python
"""Pod-side GPU validation: compile all RawKernels, run the fixtures
saved by validate_local.py, and compare kernel outputs against the numpy
bit-mirror outputs (reference.py) at near-machine tolerances.

RUN ORDER (mandatory, see runbook.md):
  1. python validate_gpu.py --stage stage2     <- Stage-2 FP64 accuracy
     spike FIRST (GPU_FEASIBILITY par.8 sequencing; the weight-
     concentration fixture). If this FAILS: STOP -- do not benchmark.
  2. python validate_gpu.py                    <- everything.

Why tolerance gates, not literal bitwise: the mirrors are operation-order
identical to the kernels (--fmad=false), but CUDA sincos vs libm sin/cos
differ by ~1 ulp and cuFFT vs pocketfft round differently, so stages with
transcendentals/FFTs are gated at ~1e-11 while stage 2 (pure +-*/ in
fixed order) is expected to match near-bitwise and is gated at 1e-13
relative. Exact-bit match fractions are reported for stage 2.

Rows flagged by the mirror as exact-routed (cond_r < 0.15) are excluded
from power comparisons: the kernel deliberately does not solve them (the
host routes them to the CPU exact path -- design requirement (v)).

Outputs validate_gpu_results.json next to this script. Exit code != 0 on
any FAIL.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                     # noqa: E402
import cupy as cp                      # noqa: E402
import kernels as K                    # noqa: E402

GATES = dict(
    # stage 2 on identical saved sums: pure FP64 +-*/ in mirrored order
    s2_spike_rel=1e-13,
    # stage 1 chained from raw inputs: CUDA-vs-libm sincos ulps only
    s1_abs=1e-11,
    # chained stage1->stage2 coefficients vs mirror coefficients (rel)
    coef_chain_rel=1e-11,
    # stage 3 on SAVED mirror coefs (isolated): sincos + cuFFT ulps
    s3_power=1e-11,
    # full chained pipeline powers vs mirror powers (non-exact rows)
    e2e_power=1e-10,
    # conditioning ratio (cuFFT vs pocketfft circle values)
    cond_rel=1e-11,
    # a theta disagreement above this with a power disagreement above
    # the power gate = a REAL winner disagreement (fails); with power
    # agreeing it is a benign candidate tie flip (reported only)
    theta_flip=1e-6,
)

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append(dict(name=name, ok=bool(ok), detail=detail))
    print('  [%s] %-52s %s' % ('PASS' if ok else 'FAIL', name, detail))


def info(msg):
    print('     (%s)' % msg)


def maxabs(x):
    x = cp.asnumpy(x) if hasattr(x, 'device') else np.asarray(x)
    return 0.0 if x.size == 0 else float(np.max(np.abs(x)))


def relscale(d, ref):
    return maxabs(d) / max(maxabs(ref), 1e-300)


def find_fixtures(arg):
    for cand in ([arg] if arg else []) + [
            os.path.join(HERE, 'fixtures'),
            os.path.join(os.path.dirname(HERE), 'fixtures')]:
        if cand and os.path.isdir(cand):
            return cand
    raise SystemExit('fixtures directory not found; scp trackA/fixtures')


def check_compile():
    print('-- compile all RawKernels (FTP_MAX_H=8, --fmad=false)')
    try:
        mod = K.get_module(max_h=8, fmad=False)
        for fn in ('stage1_direct_sums', 'stage2_assemble',
                   'stage3_scan_polish'):
            mod.get_function(fn)
        check('RawModule compiles + all kernels resolve', True)
        return mod
    except Exception as exc:                       # noqa: BLE001
        check('RawModule compiles + all kernels resolve', False,
              repr(exc)[:200])
        return None


def stage2_spike(fx, module):
    """THE accuracy spike: stage-2 FP64 assembly under adversarial weight
    concentration (moment-sum cancellation; GPU_FEASIBILITY par.7 risk #1)."""
    print('-- STAGE-2 FP64 ACCURACY SPIKE (f4_weights: 1e8-concentrated '
          'inverse-variance weights)')
    z = np.load(os.path.join(fx, 'f4_weights.npz'))
    bands = [str(b) for b in z['bands']]
    H = int(z['H'])
    worst_rel, exact_frac = 0.0, []
    for band in bands:
        sums = {f: cp.asarray(z['sums_%s_%s' % (band, f)])
                for f in ('C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS')}
        nrow = int(sums['C'].shape[0])
        ones = cp.ones(nrow)
        YM, MM, AC = K.gpu_stage2(sums, z['cn'], z['sn'], ones,
                                  accum=False, module=module)
        for tag, gpu, mir in (('YM', YM, z['mir_YM_%s' % band]),
                              ('MM', MM, z['mir_MM_%s' % band]),
                              ('AC', AC, z['mir_AC_%s' % band])):
            g = cp.asnumpy(gpu)
            worst_rel = max(worst_rel, relscale(g - mir, mir))
            exact_frac.append(float(np.mean(g == mir)))
    check('stage-2 spike: GPU vs mirror on identical sums',
          worst_rel <= GATES['s2_spike_rel'],
          'max rel=%.2e (gate %.0e)' % (worst_rel, GATES['s2_spike_rel']))
    info('exact-bit match fraction per array: min=%.4f mean=%.4f'
         % (min(exact_frac), float(np.mean(exact_frac))))

    # chained: stage-1 GPU on the raw weight-fixture inputs vs the saved
    # PACKAGE sums (recurrence-vs-direct + sincos ulps; small-angle case)
    worst1 = 0.0
    for kb, band in enumerate(bands):
        bd = K.GpuBandData(dict(t=z['in_b%d_t' % kb], w=z['in_b%d_w' % kb],
                                u=z['in_b%d_u' % kb], N=z['in_b%d_N' % kb],
                                W=z['in_b%d_W' % kb]))
        s = K.gpu_stage1(bd, cp.asarray(z['freqs']), H, module=module)
        for f in ('C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS'):
            worst1 = max(worst1, maxabs(cp.asnumpy(s[f][0]) -
                                        z['sums_%s_%s' % (band, f)]))
    check('stage-1 on weight fixture vs package sums',
          worst1 <= GATES['s1_abs'], 'max|d|=%.2e' % worst1)


def run_f1(fx, module):
    print('-- F1: chained pipeline (griz K=4, B=3, templates H8+H4, '
          'sums reuse)')
    z = np.load(os.path.join(fx, 'f1_griz.npz'))
    Kb, B, Nmax = z['t'].shape
    Hs = int(z['Hs'])
    ns = int(z['slice_n'])
    freqs = z['freqs']
    nf = len(freqs)

    band_data = [K.GpuBandData(dict(t=z['t'][kb], w=z['w'][kb],
                                    u=z['u'][kb], N=z['N'][kb],
                                    W=z['W'][kb]))
                 for kb in range(Kb)]

    # ---- stage-1 isolated on the saved slice -------------------------
    worst = 0.0
    fr_dev = cp.asarray(freqs[:ns])
    sums_slice = []
    for kb in range(Kb):
        s = K.gpu_stage1(band_data[kb], fr_dev, Hs, module=module)
        sums_slice.append(s)
        for f in ('C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS'):
            worst = max(worst, maxabs(cp.asnumpy(s[f]) -
                                      z['s1_b%d_%s' % (kb, f)]))
    check('F1 stage-1 GPU vs mirror (slice)', worst <= GATES['s1_abs'],
          'max|d|=%.2e (gate %.0e)' % (worst, GATES['s1_abs']))

    # ---- chained stage-2 coefficients vs mirror coefficients ---------
    i0, i1 = int(z['coef_i0']), int(z['coef_i1'])
    fr_dev = cp.asarray(freqs[i0:i1])
    m = i1 - i0
    W_rows = [cp.repeat(band_data[kb].W, m) for kb in range(Kb)]
    sums = [K.gpu_stage1(band_data[kb], fr_dev, Hs, module=module)
            for kb in range(Kb)]
    coefs = None
    for kb in range(Kb):
        coefs = K.gpu_stage2(sums[kb], z['cn8'], z['sn8'], W_rows[kb],
                             accum=(kb > 0), out=coefs, module=module)
    YM, MM, AC = coefs
    d_ym = relscale(cp.asnumpy(YM) - z['mir_YM'], z['mir_YM'])
    d_mm = relscale(cp.asnumpy(MM) - z['mir_MM'], z['mir_MM'])
    d_ac = relscale(cp.asnumpy(AC) - z['mir_AC'], z['mir_AC'])
    ok = max(d_ym, d_mm, d_ac) <= GATES['coef_chain_rel']
    check('F1 chained stage-2 coefs vs mirror', ok,
          'rel YM=%.1e MM=%.1e AC=%.1e' % (d_ym, d_mm, d_ac))

    # ---- full chained pipeline, K=2 templates, sums reuse ------------
    templates = [(z['cn8'], z['sn8']), (z['cn4'], z['sn4'])]
    out = K.gpu_multiband_powers(band_data, templates, z['YY'], freqs,
                                 chunk=512)
    p_gpu = cp.asnumpy(out['powers'])
    th_gpu = cp.asnumpy(out['theta'])
    cond_gpu = cp.asnumpy(out['cond_r'])
    st_gpu = cp.asnumpy(out['status'])
    p_mir = z['mir_powers']
    cond_mir = z['mir_cond']
    st_mir = z['mir_status']

    with np.errstate(invalid='ignore'):
        cmp_rows = cond_mir >= K.SCAN_EXACT_RTOL   # kernel-solved rows
    n_excl = int(np.sum(~cmp_rows))
    dP = np.abs(p_gpu - p_mir)
    worstP = maxabs(dP[cmp_rows])
    check('F1 e2e powers GPU vs mirror (non-exact rows)',
          worstP <= GATES['e2e_power'],
          'max|dP|=%.2e excl=%d rows (gate %.0e)'
          % (worstP, n_excl, GATES['e2e_power']))

    d_cond = relscale(cond_gpu - cond_mir, cond_mir)
    check('F1 cond_r GPU vs mirror', d_cond <= GATES['cond_rel'],
          'max rel=%.2e' % d_cond)

    dth = np.abs(th_gpu - z['mir_theta'])
    flips = (dth > GATES['theta_flip']) & cmp_rows
    real_bad = flips & (dP > GATES['e2e_power'])
    check('F1 no real winner disagreements',
          int(real_bad.sum()) == 0,
          '%d benign tie flips, %d real' % (int(flips.sum()),
                                            int(real_bad.sum())))
    flat_ok = np.all(((st_gpu & K.STATUS_FLAT) > 0)[cmp_rows] ==
                     ((st_mir & K.STATUS_FLAT) > 0)[cmp_rows])
    check('F1 FLAT-flag agreement (non-exact rows)', flat_ok)
    info('status full-word agreement: %.4f'
         % float(np.mean(st_gpu[cmp_rows] == st_mir[cmp_rows])))

    # vs the package (context only -- the CPU package is not on the pod;
    # saved package powers ride along in the fixture)
    for kt, key in ((0, 'pkg_powers_h8'), (1, 'pkg_powers_h4')):
        d = np.abs(p_gpu[kt] - z[key])[cmp_rows[kt]]
        info('template %d vs saved package powers: max|dP|=%.2e'
             % (kt, maxabs(d)))


def run_stage3_fixture(fx, module, fname, title):
    print('-- %s' % title)
    z = np.load(os.path.join(fx, fname))
    H = int(z['H'])
    YM = cp.asarray(z['YM'])
    MM = cp.asarray(z['MM'])
    YY = cp.asarray(z['YY'])
    exact = z['exact_mask']

    variants = [('pos=True', True, 'mir_true_power')]
    if 'mir_false_power' in z:
        variants.append(('pos=False', False, 'mir_false_power'))
    if 'mir_power' in z:
        variants = [('pos=True', True, 'mir_power')]
    outs = {}
    for tag, pos, mkey in variants:
        r = K.gpu_stage3(YM, MM, YY, H, positive_amplitude=pos,
                         module=module)
        outs[pos] = r
        d = maxabs((cp.asnumpy(r['power']) - z[mkey])[~exact])
        check('%s stage-3 GPU vs mirror %s (non-exact rows)'
              % (fname.split("_")[0], tag), d <= GATES['s3_power'],
              'max|dP|=%.2e excl=%d' % (d, int(exact.sum())))

    ckey = 'mir_true_cond' if 'mir_true_cond' in z else 'mir_cond'
    d_cond = relscale(cp.asnumpy(outs[True]['cond_r']) - z[ckey], z[ckey])
    check('%s cond_r GPU vs mirror' % fname.split("_")[0],
          d_cond <= GATES['cond_rel'], 'max rel=%.2e' % d_cond)

    skey = 'mir_true_status' if 'mir_true_status' in z else 'mir_status'
    st_gpu = cp.asnumpy(outs[True]['status'])
    dipc_gpu = (st_gpu & K.STATUS_DIP_CAND) > 0
    dipc_mir = (z[skey] & K.STATUS_DIP_CAND) > 0
    info('dip-candidate flag agreement: %.4f (gpu %d, mirror %d rows)'
         % (float(np.mean(dipc_gpu == dipc_mir)), int(dipc_gpu.sum()),
            int(dipc_mir.sum())))
    if False in outs:
        p_t = cp.asnumpy(outs[True]['power'])
        p_f = cp.asnumpy(outs[False]['power'])
        n_active = int(np.sum(np.abs(p_t - p_f) > 1e-9))
        n_pkg = int(np.sum(np.abs(z['pkg_true'] - z['pkg_false']) > 1e-9))
        check('%s K.18 filter active on GPU' % fname.split("_")[0],
              n_active > 0.5 * n_pkg,
              'gpu %d vs package %d active rows' % (n_active, n_pkg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixtures', default=None)
    ap.add_argument('--stage', default='all',
                    choices=('all', 'stage2', 'f1', 'f2', 'f3'))
    args = ap.parse_args()
    fx = find_fixtures(args.fixtures)
    print('validate_gpu: fixtures=%s device=%s' %
          (fx, cp.cuda.runtime.getDeviceProperties(0)['name'].decode()))

    module = check_compile()
    if module is not None:
        # Stage-2 spike ALWAYS first (audit par.4 / GPU_FEASIBILITY par.8)
        stage2_spike(fx, module)
        spike_ok = all(r['ok'] for r in RESULTS)
        if args.stage == 'stage2':
            pass
        elif not spike_ok:
            print('\nSTAGE-2 SPIKE FAILED -- aborting (do NOT benchmark; '
                  'see runbook ESCALATE)')
        else:
            if args.stage in ('all', 'f2'):
                run_stage3_fixture(fx, module, 'f2_k18.npz',
                                   'F2: K.18-active stage-3 fixture')
            if args.stage in ('all', 'f3'):
                run_stage3_fixture(fx, module, 'f3_deepdip.npz',
                                   'F3: deep-dip stage-3 fixture')
            if args.stage in ('all', 'f1'):
                run_f1(fx, module)

    npass = sum(1 for r in RESULTS if r['ok'])
    nfail = len(RESULTS) - npass
    print('\n================ PASS/FAIL table ================')
    for r in RESULTS:
        print('%s  %-56s %s' % ('PASS' if r['ok'] else 'FAIL',
                                r['name'], r['detail']))
    print('=================================================')
    print('%d passed, %d failed' % (npass, nfail))
    with open(os.path.join(HERE, 'validate_gpu_results.json'), 'w') as f:
        json.dump(dict(results=RESULTS, gates=GATES), f, indent=1)
    sys.exit(1 if nfail else 0)


if __name__ == '__main__':
    main()
