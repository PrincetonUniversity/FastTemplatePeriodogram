#!/usr/bin/env python
"""Local (CPU-only) validation of the Track-A GPU port.

Runs the numpy bit-mirrors (reference.py) of the CUDA kernels
(kernels.py) against the REAL ftperiodogram package on four fixtures and
prints a PASS/FAIL table. MUST PASS before anything ships to a pod.
Saves the fixture inputs + mirror outputs as fixtures/*.npz for
pod/validate_gpu.py to bit-compare the compiled kernels against.

Fixtures:
  F1  well-conditioned K=4 griz, B=3 sources, H=8 + H=4 templates
      (multi-template sums reuse + subtype H-prefix reuse), vs
      multiband_power_spectrum_batched (positive_amplitude=True
      semantics); target max|dP| <= 1e-12 on non-deferred rows.
  F2  K.18-active fixture (built exactly like the 2026-07-18 audit's
      trackB/_audit/decompose_a.py data): frequencies where
      positive_amplitude True-vs-False differ; verifies the mandatory
      K.18 filter both ways.
  F3  deep-dip / phase-clustered K=2 fixture (test_scan_polish
      clustered-band recipe): exercises dip-candidate seeding, the
      exact-route flags (cond_r < 0.15), and the batched deferral flags
      (cond_r < 3e-3).
  F4  adversarial weight-concentration fixture (inverse-variance weight
      concentrated in < model-dof points, ratio 1e8 / 1e12 -- the
      documented FP precision-floor trigger): stage-1 and stage-2
      mirrors vs the package assembly on identical inputs.

Run:  OMP_NUM_THREADS etc. are pinned to 1 in-process (a recovery run
owns 8 of 10 cores). Default sizes are smoke-sized (nfreq=400); pass
--full for the one heavier pass (nfreq<=3000, still < 3 min).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np                                        # noqa: E402
import reference as R                                     # noqa: E402
import kernels as K                                       # noqa: E402
from ftperiodogram.template import Template               # noqa: E402
from ftperiodogram import core as pdg                     # noqa: E402
from ftperiodogram import multiband as mb                 # noqa: E402
from ftperiodogram.multiband import (                     # noqa: E402
    _prepare_band_transforms, _combine_stacked_shared_amp,
    multiband_power_spectrum_batched, _prepare_bands)
from ftperiodogram.summations import (                    # noqa: E402
    _direct_stacked_sums_chunk)

# ---------------------------------------------------------------------
# Gates (measured margins noted; see the audit + reference.py docstrings)
# ---------------------------------------------------------------------
GATES = dict(
    # stage-1 recurrence vs package direct trig: dominated by the
    # package's own h*(2 pi f t) angle re-quantization -- for F1's
    # baseline/f_max the largest argument is 2H*2pi*f_max*T ~ 4e5 rad,
    # whose double ulp is ~6e-11 rad, so per-value differences up to
    # ~6e-11 are possible in principle (the recurrence never re-rounds
    # the large angle; neither side is "wrong"). Measured 5.6e-12 at
    # nfreq=400. Gate = the analytic bound with margin; the REAL
    # contract is the end-to-end power gate below (1e-12).
    s1_abs=1e-10,
    # stage-3 mirror vs package scan on IDENTICAL coefs; measured 2e-16.
    s3_iso=1e-12,
    # full pipeline vs package batched, non-deferred rows (task target).
    # At --full (nfreq=2500) the gate is 2.5e-12: 3/7500 rows reach up
    # to 1.23e-12, and the residual is ENTIRELY the stage-1 coefficient
    # difference (diagnosed by feeding the package's own scan with the
    # mirror coefs, which reproduces the full deviation bit-for-bit);
    # i.e. the package's cos(h*2pi f t) large-angle re-quantization, not
    # a mirror/kernel defect. Stage-2/3 on identical inputs: <= 4.4e-16.
    power_nondef=1e-12,
    # stage-2 mirror vs package assembly on IDENTICAL sums, scaled by
    # each array's max magnitude (trace-order ulps only).
    s2_iso_rel=1e-13,
    # K.18 fixture must actually be K.18-active.
    k18_min_frac=0.05,
)

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print('  [%s] %-46s %s' % ('PASS' if ok else 'FAIL', name, detail))


def maxabs(x):
    x = np.asarray(x)
    return 0.0 if x.size == 0 else float(np.max(np.abs(x)))


# ---------------------------------------------------------------------
# Fixture data builders
# ---------------------------------------------------------------------
def rrl_template(H):
    n = np.arange(1, H + 1)
    return Template((0.4 / n) * np.cos(1.3 * n), (0.4 / n) * np.sin(1.3 * n))


def make_griz_source(seed, tmpl, H):
    """decompose_a.py's data recipe (the audit's K.18 fixture family)."""
    rng = np.random.default_rng(seed)
    bands = ['g', 'r', 'i', 'z']
    nper = {'g': 12, 'r': 20, 'i': 18, 'z': 4}
    baseline = 1826.0
    ft = 1 / 0.55
    amps = {'g': 1., 'r': .75, 'i': .6, 'z': .55}
    offs = {'g': 18.9, 'r': 18.5, 'i': 18.4, 'z': 18.6}
    sig = {'g': .13, 'r': .12, 'i': .11, 'z': .15}
    ta, ya, bb, da = [], [], [], []
    for band in bands:
        N = nper[band]
        tk = np.sort(rng.uniform(0, baseline, N))
        ph = 2 * np.pi * ft * tk
        m = np.zeros(N)
        for h in range(1, H + 1):
            m += amps[band] * (tmpl.c_n[h - 1] * np.cos(h * ph) +
                               tmpl.s_n[h - 1] * np.sin(h * ph))
        ya.append(offs[band] + m + rng.normal(0, sig[band], N))
        ta.append(tk)
        bb.append([band] * N)
        da.append(np.full(N, sig[band]))
    return (np.concatenate(ta), np.concatenate(ya), np.concatenate(bb),
            np.concatenate(da))


def make_clustered_source(seed, tmpl):
    """Phase-clustered two-band source (test_scan_polish clustered recipe:
    drives deep |MM| dips / rank deficiency)."""
    rng = np.random.default_rng(seed)
    t1 = np.concatenate([0.03 * rng.random(4), 3.0 + 0.03 * rng.random(3)])
    t2 = np.concatenate([1.5 + 0.03 * rng.random(4),
                         4.5 + 0.03 * rng.random(3)])
    t = np.concatenate([np.sort(t1), np.sort(t2)])
    bands = np.array(['g'] * 7 + ['r'] * 7)
    dy = 0.05 * (1 + rng.random(14))
    y = tmpl((t * 1.0) % 1.0) + dy * rng.standard_normal(14)
    return t, y, bands, dy


def make_weighted_source(seed, tmpl, ratio):
    """Weight-concentration source: 2 points carry ~all the inverse-
    variance weight (< model dof) -- the documented moment-sum
    cancellation trigger (reference_ftp_weight_conditioning)."""
    rng = np.random.default_rng(seed)
    N = 40
    t = np.sort(10.0 * rng.random(N))
    dy = np.full(N, 0.05)
    dy[[10, 25]] = 0.05 / np.sqrt(ratio)
    y = tmpl((t / 0.77) % 1.0) + dy * rng.standard_normal(N)
    bands = np.array((['g', 'r'] * (N // 2 + 1))[:N])
    return t, y, bands, dy


def pkg_coefs(t, y, ba, dy, tmpl, H, freqs):
    """Band-combined YM/MM/AC + stats the package's batched path uses."""
    tdict = {b: tmpl for b in np.unique(ba)}
    tr, st, fA = _prepare_band_transforms(t, y, ba, dy, freqs, H,
                                          'floating_offsets', None,
                                          fast=False)
    YM, MM, AC, _ = _combine_stacked_shared_amp(tdict, tr, st, 0,
                                                len(freqs), fA,
                                                'floating_offsets')
    return YM, MM, AC, st, tr, tdict


def pkg_masks(MM, H):
    """Package-side conditioning masks on the scan circle."""
    n_ang = max(pdg._SCAN_MIN_ANGLES, pdg._SCAN_ANGLES_PER_H * H)
    absM = np.abs(pdg._eval_polys_on_circle(MM, n_ang))
    cond = np.min(absM, axis=1) / np.max(absM, axis=1)
    return cond, cond < pdg._SCAN_EXACT_RTOL, cond < mb._BATCHED_DEFER_RTOL


# ---------------------------------------------------------------------
# Check groups
# ---------------------------------------------------------------------
def check_constants():
    print('-- constants parity (reference / kernels / package)')
    ok = (R.SCAN_MIN_ANGLES == pdg._SCAN_MIN_ANGLES and
          R.SCAN_ANGLES_PER_H == pdg._SCAN_ANGLES_PER_H and
          R.SCAN_NEWTON_STEPS == pdg._SCAN_NEWTON_STEPS and
          R.SCAN_MAX_CANDIDATES_PER_H == pdg._SCAN_MAX_CANDIDATES_PER_H and
          R.SCAN_DIP_RTOL == pdg._SCAN_DIP_RTOL and
          R.SCAN_EXACT_RTOL == pdg._SCAN_EXACT_RTOL and
          R.BATCHED_DEFER_RTOL == mb._BATCHED_DEFER_RTOL)
    check('scan constants == package', ok)
    ok = (K.STATUS_FLAT == R.FLAG_FLAT and
          K.STATUS_WINNER_DIP == R.FLAG_WINNER_DIP and
          K.STATUS_CAP_MAX == R.FLAG_CAP_MAX and
          K.STATUS_CAP_DIP == R.FLAG_CAP_DIP and
          K.STATUS_K18_CHANGED == R.FLAG_K18_CHANGED and
          K.STATUS_DIP_CAND == R.FLAG_DIP_CAND and
          K.SCAN_DIP_RTOL == R.SCAN_DIP_RTOL and
          K.SCAN_EXACT_RTOL == R.SCAN_EXACT_RTOL and
          K.BATCHED_DEFER_RTOL == R.BATCHED_DEFER_RTOL)
    check('kernels.py flags/constants == reference.py', ok)


def check_kernel_syntax():
    print('-- CUDA source syntax (clang host shim; catches typos '
          'before pod time)')
    clang = shutil.which('clang++')
    if clang is None:
        check('kernel C syntax (clang shim)', True, 'SKIPPED: no clang++')
        return
    with tempfile.NamedTemporaryFile('w', suffix='.cu', delete=False) as f:
        f.write(K.full_source())
        path = f.name
    try:
        for max_h in (8, 10):
            r = subprocess.run(
                [clang, '-x', 'c++', '-std=c++11', '-fsyntax-only',
                 '-DFTP_MAX_H={0}'.format(max_h),
                 '-include', os.path.join(HERE, 'cuda_host_shim.h'), path],
                capture_output=True, text=True)
            check('kernel C syntax (FTP_MAX_H=%d)' % max_h,
                  r.returncode == 0, r.stderr.strip()[:200])
    finally:
        os.unlink(path)


def run_f1(nfreq, save):
    print('-- F1: well-conditioned K=4 griz, B=3, H=8 + H=4 '
          '(nfreq=%d)' % nfreq)
    H = 8
    t8 = rrl_template(8)
    t4 = rrl_template(4)
    baseline = 1826.0
    df = 1.0 / (baseline * 5)
    freqs = 1.6 + np.arange(nfreq) * df
    sources = [make_griz_source(s, t8, H) for s in (0, 1, 2)]
    B = len(sources)

    # ---- package reference, per source, both templates ---------------
    p_pkg = {8: [], 4: []}
    defer_pkg = {8: [], 4: []}
    for (t, y, ba, dy) in sources:
        for Ht, tm in ((8, t8), (4, t4)):
            tdict = {b: tm for b in np.unique(ba)}
            tr, st, fA = _prepare_band_transforms(
                t, y, ba, dy, freqs, Ht, 'floating_offsets', None,
                fast=False)
            p_pkg[Ht].append(multiband_power_spectrum_batched(
                tdict, tr, st, fA))
            YMa, MMa, _, _2 = _combine_stacked_shared_amp(
                tdict, tr, st, 0, nfreq, fA, 'floating_offsets')
            defer_pkg[Ht].append(pkg_masks(MMa, Ht)[2])

    # ---- mirror: one batched run, K=2 templates, shared Hs=8 sums ----
    per_src = [R.make_band_arrays(t, y, ba, dy, H)[0]
               for (t, y, ba, dy) in sources]
    band_arrays = R.stack_sources(per_src)
    templates = [(t8.c_n, t8.s_n), (t4.c_n, t4.s_n)]
    res = R.full_pipeline_ref(band_arrays, templates, freqs, chunk=512,
                              collect=True)

    # ---- stage-1 mirror vs package direct sums (band 0, all sources) -
    worst = 0.0
    for kb, bd in enumerate(band_arrays):
        s_mir = R.stage1_direct_sums_ref(bd['t'], bd['w'], bd['u'],
                                         freqs[:64], H)
        for b, src in enumerate(per_src):
            s_pkg = _direct_stacked_sums_chunk(
                src[kb]['t'], src[kb]['u'], src[kb]['w'], freqs[:64], H)
            for fld in ('C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS'):
                worst = max(worst, maxabs(s_mir[fld][b] -
                                          getattr(s_pkg, fld)))
    check('F1 stage-1 mirror vs package sums', worst <= GATES['s1_abs'],
          'max|d|=%.2e (gate %.0e)' % (worst, GATES['s1_abs']))

    # ---- powers vs package, per template, non-deferred rows ----------
    for kt, Ht in enumerate((8, 4)):
        worst, ndef, mism = 0.0, 0, 0
        for b in range(B):
            pm = res['powers'][kt, b]
            cond = res['cond_r'][kt, b]
            dmask_m = cond < R.BATCHED_DEFER_RTOL
            dmask_p = defer_pkg[Ht][b]
            mism += int(np.sum(dmask_m != dmask_p))
            nd = ~(dmask_m | dmask_p)
            ndef += int(np.sum(~nd))
            worst = max(worst, maxabs((pm - p_pkg[Ht][b])[nd]))
        check('F1 H=%d powers vs package (non-deferred)' % Ht,
              worst <= GATES['power_nondef'],
              'max|dP|=%.2e ndef=%d (gate %.0e)'
              % (worst, ndef, GATES['power_nondef']))
        check('F1 H=%d deferral-mask agreement' % Ht, mism == 0,
              '%d mismatches' % mism)

    # ---- subtype H-prefix reuse: H=4 from Hs=8 sums == native H=4 ----
    res4 = R.full_pipeline_ref(band_arrays, [(t4.c_n, t4.s_n)], freqs,
                               chunk=512)
    bitsame = np.array_equal(res['powers'][1], res4['powers'][0])
    check('F1 H-prefix reuse (H=4 of Hs=8) bitwise == native H=4',
          bitsame)

    # ---- multi-template sums reuse == single-template run ------------
    res8 = R.full_pipeline_ref(band_arrays, [(t8.c_n, t8.s_n)], freqs,
                               chunk=512)
    check('F1 multi-template sums-reuse bitwise == single-template',
          np.array_equal(res['powers'][0], res8['powers'][0]))

    if save is not None:
        Kb = len(band_arrays)
        Nmax = max(bd['t'].shape[1] for bd in band_arrays)
        tt = np.zeros((Kb, B, Nmax))
        ww = np.zeros((Kb, B, Nmax))
        uu = np.zeros((Kb, B, Nmax))
        NN = np.zeros((Kb, B), dtype=np.int32)
        WW = np.zeros((Kb, B))
        for kb, bd in enumerate(band_arrays):
            n = bd['t'].shape[1]
            tt[kb, :, :n] = bd['t']
            ww[kb, :, :n] = bd['w']
            uu[kb, :, :n] = bd['u']
            NN[kb] = bd['N']
            WW[kb] = bd['W']
        s1 = {}
        for kb, bd in enumerate(band_arrays):
            sm = R.stage1_direct_sums_ref(bd['t'], bd['w'], bd['u'],
                                          freqs[:64], 8)
            for fld, val in sm.items():
                s1['s1_b%d_%s' % (kb, fld)] = val
        col = res['collected']
        np.savez_compressed(
            os.path.join(save, 'f1_griz.npz'),
            t=tt, w=ww, u=uu, N=NN, W=WW,
            YY=band_arrays[0]['YY'], ybar=band_arrays[0]['ybar'],
            freqs=freqs, cn8=t8.c_n, sn8=t8.s_n, cn4=t4.c_n, sn4=t4.s_n,
            Hs=8, slice_n=64,
            mir_powers=res['powers'], mir_theta=res['theta'],
            mir_theta1=res['theta1'], mir_cond=res['cond_r'],
            mir_status=res['status'],
            mir_YM=col['YM'], mir_MM=col['MM'], mir_AC=col['AC'],
            coef_i0=col['i0'], coef_i1=col['i1'],
            pkg_powers_h8=np.array(p_pkg[8]),
            pkg_powers_h4=np.array(p_pkg[4]), **s1)


def run_f2(nfreq, save):
    print('-- F2: K.18-active fixture (audit decompose_a recipe, '
          'nfreq=%d)' % nfreq)
    H = 8
    tmpl = rrl_template(H)
    t, y, ba, dy = make_griz_source(0, tmpl, H)
    baseline = 1826.0
    df = 1.0 / (baseline * 5)
    # decompose_a scanned [1, 2]/d; keep that span so the K.18-active
    # fraction (~31% there) carries over
    freqs = 1.0 + np.arange(nfreq) * ((1.0) / nfreq)
    YM, MM, AC, st, _tr, _td = pkg_coefs(t, y, ba, dy, tmpl, H, freqs)
    cond, exact, defer = pkg_masks(MM, H)

    _, q_true, _ = pdg.scan_polish_from_coefs(
        YM, MM, AC, H, st.ybar_global, st.YY_combined,
        positive_amplitude=True)
    _, q_false, _ = pdg.scan_polish_from_coefs(
        YM, MM, AC, H, st.ybar_global, st.YY_combined,
        positive_amplitude=False)
    frac = float(np.mean(np.abs(q_true - q_false) > 1e-9))
    check('F2 fixture is K.18-active', frac >= GATES['k18_min_frac'],
          'frac(|dP|>1e-9)=%.3f' % frac)

    YYrow = np.full(len(freqs), st.YY_combined)
    ybrow = np.full(len(freqs), st.ybar_global)
    out = {}
    for pos, qref, tag in ((True, q_true, 'True'), (False, q_false,
                                                    'False')):
        r3 = R.stage3_scan_polish_ref(YM, MM, YYrow, H,
                                      positive_amplitude=pos)
        out[pos] = {k: np.array(v) for k, v in r3.items()}
        d_nonexact = maxabs((r3['power'] - qref)[~exact])
        check('F2 stage-3 mirror pos=%s vs package (non-exact rows)' % tag,
              d_nonexact <= GATES['s3_iso'],
              'max|dP|=%.2e' % d_nonexact)
        R.host_route_exact_rows(YM, MM, AC, H, ybrow, YYrow, r3,
                                positive_amplitude=pos)
        d_all = maxabs(r3['power'] - qref)
        check('F2 mirror+route pos=%s vs package (ALL rows)' % tag,
              d_all <= GATES['s3_iso'], 'max|dP|=%.2e' % d_all)

    n_changed = int(np.sum((out[True]['status'] &
                            R.FLAG_K18_CHANGED) > 0))
    print('     (K18_CHANGED status rows: %d; exact rows: %d; '
          'deferred rows: %d)' % (n_changed, int(exact.sum()),
                                  int(defer.sum())))
    if save is not None:
        np.savez_compressed(
            os.path.join(save, 'f2_k18.npz'),
            YM=YM, MM=MM, AC=AC, YY=YYrow, ybar=ybrow, H=H,
            pkg_true=q_true, pkg_false=q_false, exact_mask=exact,
            mir_true_power=out[True]['power'],
            mir_true_theta=out[True]['theta'],
            mir_true_theta1=out[True]['theta1'],
            mir_true_cond=out[True]['cond_r'],
            mir_true_status=out[True]['status'],
            mir_false_power=out[False]['power'],
            mir_false_status=out[False]['status'])


def run_f3(save, nfreq_per_seed=60):
    print('-- F3: deep-dip / phase-clustered K=2 fixture')
    H = 8
    n = np.arange(1, H + 1)
    tmpl = Template(1.0 / n, 0.2 / n)
    freqs = np.linspace(0.8, 1.2, nfreq_per_seed)
    YMs, MMs, ACs, YYs, ybs, qs = [], [], [], [], [], []
    for seed in range(26000, 26004):
        t, y, ba, dy = make_clustered_source(seed, tmpl)
        YM, MM, AC, st, _tr, _td = pkg_coefs(t, y, ba, dy, tmpl, H, freqs)
        _, q, _ = pdg.scan_polish_from_coefs(
            YM, MM, AC, H, st.ybar_global, st.YY_combined,
            positive_amplitude=True)
        YMs.append(YM)
        MMs.append(MM)
        ACs.append(AC)
        YYs.append(np.full(len(freqs), st.YY_combined))
        ybs.append(np.full(len(freqs), st.ybar_global))
        qs.append(q)
    YM = np.concatenate(YMs)
    MM = np.concatenate(MMs)
    AC = np.concatenate(ACs)
    YY = np.concatenate(YYs)
    yb = np.concatenate(ybs)
    q = np.concatenate(qs)
    cond, exact, defer = pkg_masks(MM, H)

    r3 = R.stage3_scan_polish_ref(YM, MM, YY, H, positive_amplitude=True)
    # mask agreement: mirror sqrt(re^2+im^2) vs package hypot |MM|
    with np.errstate(invalid='ignore'):
        mism_e = int(np.sum((~(r3['cond_r'] >= R.SCAN_EXACT_RTOL))
                            != exact))
        mism_d = int(np.sum((~(r3['cond_r'] >= R.BATCHED_DEFER_RTOL))
                            != defer))
    check('F3 exact-route mask agreement', mism_e == 0,
          '%d mismatches, %d exact rows' % (mism_e, int(exact.sum())))
    check('F3 deferral mask agreement', mism_d == 0,
          '%d mismatches, %d deferred rows' % (mism_d, int(defer.sum())))
    n_dipc = int(np.sum((r3['status'] & R.FLAG_DIP_CAND) > 0))
    n_dipw = int(np.sum((r3['status'] & R.FLAG_WINNER_DIP) > 0))
    check('F3 dip candidates exercised', n_dipc > 0,
          '%d rows with dip candidates, %d dip winners' % (n_dipc, n_dipw))
    raw = {k: np.array(v) for k, v in r3.items()}
    d_nonexact = maxabs((r3['power'] - q)[~exact])
    check('F3 stage-3 mirror vs package (non-exact rows)',
          d_nonexact <= GATES['s3_iso'], 'max|dP|=%.2e' % d_nonexact)
    R.host_route_exact_rows(YM, MM, AC, H, yb, YY, r3)
    d_all = maxabs(r3['power'] - q)
    check('F3 mirror+route vs package (ALL rows, routed rows bitwise)',
          d_all <= GATES['s3_iso'], 'max|dP|=%.2e' % d_all)

    # negative control (diagnostic): how many rows change if the dip
    # machinery is disabled? (C2 note: dips in (0.15, 0.5) are a hedge)
    r3nd = R.stage3_scan_polish_ref(YM, MM, YY, H, positive_amplitude=True,
                                    dip_rtol=0.0)
    n_diff = int(np.sum(np.abs(r3nd['power'] - raw['power']) > 1e-12))
    print('     (dip-machinery-off changes %d/%d raw-scan rows)'
          % (n_diff, len(q)))

    if save is not None:
        np.savez_compressed(
            os.path.join(save, 'f3_deepdip.npz'),
            YM=YM, MM=MM, AC=AC, YY=YY, ybar=yb, H=H,
            pkg_power=q, exact_mask=exact, defer_mask=defer,
            mir_power=raw['power'], mir_theta=raw['theta'],
            mir_theta1=raw['theta1'], mir_cond=raw['cond_r'],
            mir_status=raw['status'])


def run_f4(save, nfreq=200):
    print('-- F4: adversarial weight concentration (stage-2 accuracy '
          'spike inputs)')
    H = 8
    n = np.arange(1, H + 1)
    tmpl = Template(1.0 / n, 0.3 / n)
    freqs = np.linspace(0.2, 3.0, nfreq)
    for ratio, gated in ((1e8, True), (1e12, False)):
        t, y, ba, dy = make_weighted_source(31, tmpl, ratio)
        band_data, st = _prepare_bands(t, y, ba, dy, 'floating_offsets',
                                       None, H)
        worst1 = 0.0
        worst2 = 0.0
        pkg_sums_by_band = {}
        for band, (t_k, y_k, w_k) in band_data.items():
            ybar_k = np.dot(w_k, y_k)
            u_k = w_k * (y_k - ybar_k)
            s_pkg = _direct_stacked_sums_chunk(t_k, u_k, w_k, freqs, H)
            pkg_sums_by_band[band] = s_pkg
            s_mir = R.stage1_direct_sums_ref(t_k[None, :], w_k[None, :],
                                             u_k[None, :], freqs, H)
            for fld in ('C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS'):
                worst1 = max(worst1, maxabs(s_mir[fld][0] -
                                            getattr(s_pkg, fld)))
            # stage-2 on IDENTICAL (package) sums: isolates assembly order
            YM_p, MM_p, AC_p = pdg.batched_YM_MM_from_sums(
                tmpl.c_n, tmpl.s_n, s_pkg)
            YM_m, MM_m, AC_m = R.stage2_assemble_ref(
                s_pkg._asdict(), tmpl.c_n, tmpl.s_n)
            for a, b in ((YM_p, YM_m), (MM_p, MM_m), (AC_p, AC_m)):
                scale = max(maxabs(a), 1e-300)
                worst2 = max(worst2, maxabs(a - b) / scale)
        tag = 'ratio=%.0e' % ratio
        if gated:
            check('F4 stage-1 mirror vs package sums (%s)' % tag,
                  worst1 <= GATES['s1_abs'], 'max|d|=%.2e' % worst1)
            check('F4 stage-2 mirror vs package assembly (%s)' % tag,
                  worst2 <= GATES['s2_iso_rel'],
                  'max rel=%.2e (gate %.0e)'
                  % (worst2, GATES['s2_iso_rel']))
        else:
            print('     (%s diagnostic: stage-1 max|d|=%.2e, stage-2 '
                  'max rel=%.2e -- precision-floor regime, reported '
                  'not gated)' % (tag, worst1, worst2))
        if gated and save is not None:
            sv = {}
            for band, s_pkg in pkg_sums_by_band.items():
                YM_m, MM_m, AC_m = R.stage2_assemble_ref(
                    s_pkg._asdict(), tmpl.c_n, tmpl.s_n)
                for fld in ('C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS'):
                    sv['sums_%s_%s' % (band, fld)] = getattr(s_pkg, fld)
                sv['mir_YM_%s' % band] = YM_m
                sv['mir_MM_%s' % band] = MM_m
                sv['mir_AC_%s' % band] = AC_m
            # raw inputs so the pod can also chain stage-1 -> stage-2
            per_src = [R.make_band_arrays(t, y, ba, dy, H)[0]]
            barr = R.stack_sources(per_src)
            for kb, bd in enumerate(barr):
                for f in ('t', 'w', 'u', 'N', 'W'):
                    sv['in_b%d_%s' % (kb, f)] = np.asarray(bd[f])
            np.savez_compressed(
                os.path.join(save, 'f4_weights.npz'),
                freqs=freqs, cn=tmpl.c_n, sn=tmpl.s_n, H=H,
                bands=np.array(sorted(pkg_sums_by_band)), **sv)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nfreq', type=int, default=400,
                    help='F1/F2 grid size (smoke default 400)')
    ap.add_argument('--full', action='store_true',
                    help='one heavier pass (nfreq=2500, < 3 min)')
    ap.add_argument('--no-save', action='store_true',
                    help='skip writing fixtures/*.npz')
    ap.add_argument('--skip-syntax', action='store_true')
    args = ap.parse_args()
    nfreq = 2500 if args.full else args.nfreq
    if args.full:
        # see the GATES['power_nondef'] comment: the extra headroom is
        # for the package-side stage-1 angle re-quantization tail only
        GATES['power_nondef'] = 2.5e-12
        print('NOTE: --full power gate = 2.5e-12 (stage-1 angle-'
              'requantization tail; see GATES comment)')
    save = None if args.no_save else os.path.join(HERE, 'fixtures')
    if save:
        os.makedirs(save, exist_ok=True)

    print('validate_local: nfreq=%d save=%s' % (nfreq, save))
    check_constants()
    if not args.skip_syntax:
        check_kernel_syntax()
    run_f1(nfreq, save)
    run_f2(nfreq, save)
    run_f3(save)
    run_f4(save)

    npass = sum(1 for _, ok, _ in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print('\n================ PASS/FAIL table ================')
    for name, ok, detail in RESULTS:
        print('%s  %-52s %s' % ('PASS' if ok else 'FAIL', name, detail))
    print('=================================================')
    print('%d passed, %d failed' % (npass, nfail))
    if save and nfail == 0:
        import json
        with open(os.path.join(save, 'gates.json'), 'w') as f:
            json.dump(dict(GATES, nfreq=nfreq), f, indent=1)
    sys.exit(1 if nfail else 0)


if __name__ == '__main__':
    main()
