#!/usr/bin/env python
"""Pod-side throughput benchmark for the Track-A fused kernels.

Methodology (2026-07-18 audit corrections baked in):
  - MEDIAN of reps is the headline (the audit flagged min-of-3 as
    optimistic); min is reported alongside for context only.
  - H2D upload + D2H download are timed and included in the reported
    'total' column (a separate 'compute' column excludes them).
  - Report per-LC-per-freq-per-template microseconds. When quoting
    speedups, anchor to BOTH the same-code CPU floor and the best
    single-thread CPU path (~78-81 us/LC/freq @H8 on one Apple M5
    P-core) -- NEVER the inflated proto-B=256 baseline (audit par.1).

Sweep (task spec): B in {32,128,512,2048}, H in {2,4,8}, nfreq in
{8000, 45000}, K_templates in {1,4,8}, sums-reuse on/off (off only
meaningful for K>1). The full cross is hours of A100 time; use --tier
quick first, then targeted sweeps (see runbook.md staging + budget).

Each config is budget-guarded (--budget seconds; reps stop early after
>= 3 once the budget is spent) and memory-guarded (frequency chunk is
sized to a device-memory target; configs that cannot fit are SKIPPED
with a note). Results append to bench_results.json after EVERY config
(crash-safe).
"""
import argparse
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                     # noqa: E402
import cupy as cp                      # noqa: E402
import kernels as K                    # noqa: E402

K_BANDS = 4
N_PER_BAND = 15          # the FINDINGS.md throughput config (griz, N=15)


def gen_batch_host(B, seed=0, baseline=1826.0):
    """Synthetic griz-like padded batch (host arrays)."""
    rng = np.random.default_rng(seed)
    bands = []
    for kb in range(K_BANDS):
        t = np.sort(rng.uniform(0, baseline, (B, N_PER_BAND)), axis=1)
        w = rng.uniform(0.5, 1.5, (B, N_PER_BAND))
        w /= w.sum(axis=1, keepdims=True)
        y = rng.normal(0.0, 1.0, (B, N_PER_BAND))
        ybar = np.sum(w * y, axis=1, keepdims=True)
        u = w * (y - ybar)
        bands.append(dict(t=t, w=w, u=u,
                          N=np.full(B, N_PER_BAND, np.int32),
                          W=np.full(B, 1.0 / K_BANDS)))
    YY = np.ones(B)
    return bands, YY


def make_templates(K_t, H, seed=1):
    n = np.arange(1, H + 1)
    return [((0.4 / n) * np.cos(1.3 * n + 0.1 * k),
             (0.4 / n) * np.sin(1.3 * n + 0.07 * k))
            for k in range(K_t)]


def pick_chunk(B, H, mem_target=6e9):
    """Frequency-chunk size bounding the K_BANDS stage-1 sum stacks."""
    bytes_per_freq = K_BANDS * B * (4 * H + 3 * H * H) * 8
    chunk = int(mem_target // max(bytes_per_freq, 1))
    return max(32, min(2048, chunk))


def estimate_bytes(B, H, chunk, nfreq, K_t):
    s1 = K_BANDS * B * chunk * (4 * H + 3 * H * H) * 8
    M = K.scan_n_angles(H)
    rows = min(B * chunk, 1 << 18)
    s3 = rows * M * 16 * 2
    coefs = B * chunk * (6 * H + 2 + H) * 16
    outputs = K_t * B * nfreq * (4 * 8 + 4)     # 4 f64 + 1 i32 arrays
    return s1 + s3 + coefs + outputs


def run_config(B, H, nfreq, K_t, reuse, reps, budget, fmad):
    templates = make_templates(K_t, H)
    freqs = 0.5 + np.arange(nfreq) * (2.0 / nfreq)
    chunk = pick_chunk(B, H)
    est = estimate_bytes(B, H, chunk, nfreq, K_t)
    free_b = cp.cuda.runtime.memGetInfo()[0]
    if est > 0.85 * free_b:
        return dict(skipped=True, reason='est %.1f GB > free %.1f GB'
                    % (est / 1e9, free_b / 1e9))

    host_bands, YY = gen_batch_host(B)
    totals, computes, h2ds, d2hs = [], [], [], []
    t_start = time.perf_counter()
    for rep in range(reps):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        band_data = [K.GpuBandData(b) for b in host_bands]
        YY_dev = cp.asarray(YY)
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        out = K.gpu_multiband_powers(band_data, templates, YY_dev, freqs,
                                     chunk=chunk, reuse_sums=reuse,
                                     fmad=fmad)
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()
        p_host = cp.asnumpy(out['powers'])
        c_host = cp.asnumpy(out['cond_r'])
        cp.cuda.Device().synchronize()
        t3 = time.perf_counter()
        del out, band_data
        cp.get_default_memory_pool().free_all_blocks()
        totals.append(t3 - t0)
        computes.append(t2 - t1)
        h2ds.append(t1 - t0)
        d2hs.append(t3 - t2)
        assert np.all(np.isfinite(p_host)), 'non-finite powers!'
        if rep + 1 >= 3 and (time.perf_counter() - t_start) > budget:
            break
    n = len(totals)
    lcf = float(B) * nfreq * K_t
    med = float(np.median(totals))
    res = dict(B=B, H=H, nfreq=nfreq, K_templates=K_t,
               reuse_sums=bool(reuse), fmad=bool(fmad), reps_done=n,
               chunk=chunk,
               total_s_median=med, total_s_min=float(np.min(totals)),
               compute_s_median=float(np.median(computes)),
               h2d_s_median=float(np.median(h2ds)),
               d2h_s_median=float(np.median(d2hs)),
               us_per_lc_freq_tmpl=med / lcf * 1e6,
               lc_per_s=float(B) / med,
               frac_deferred=float(np.mean(c_host <
                                           K.BATCHED_DEFER_RTOL)))
    return res


def breakdown(B, H, nfreq, K_t):
    """One-off per-stage timing (events) for a single chunk."""
    templates = make_templates(K_t, H)
    chunk = pick_chunk(B, H)
    m = min(chunk, nfreq)
    host_bands, YY = gen_batch_host(B)
    band_data = [K.GpuBandData(b) for b in host_bands]
    YY_dev = cp.asarray(YY)
    fr = cp.asarray(0.5 + np.arange(m) * (2.0 / m))
    module = K.get_module(max_h=max(H, 8))
    Hs = H
    W_rows = [cp.repeat(b.W, m) for b in band_data]
    YY_rows = cp.repeat(YY_dev, m)

    def timed(fn):
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        e0.record()
        out = fn()
        e1.record()
        e1.synchronize()
        return out, cp.cuda.get_elapsed_time(e0, e1) / 1e3

    sums, t_s1 = timed(lambda: [K.gpu_stage1(b, fr, Hs, module=module)
                                for b in band_data])

    def do_s2():
        coefs = None
        for kb, s in enumerate(sums):
            coefs = K.gpu_stage2(s, templates[0][0], templates[0][1],
                                 W_rows[kb], accum=(kb > 0), out=coefs,
                                 module=module)
        return coefs
    coefs, t_s2 = timed(do_s2)
    YM, MM, AC = coefs
    _, t_s3 = timed(lambda: K.gpu_stage3(YM, MM, YY_rows, H,
                                         module=module))
    return dict(B=B, H=H, m=m, K_templates=K_t, stage1_s=t_s1,
                stage2_s=t_s2, stage3_s=t_s3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, nargs='+',
                    default=[32, 128, 512, 2048])
    ap.add_argument('--H', type=int, nargs='+', default=[2, 4, 8])
    ap.add_argument('--nfreq', type=int, nargs='+',
                    default=[8000, 45000])
    ap.add_argument('--K', type=int, nargs='+', default=[1, 4, 8],
                    help='template counts')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--budget', type=float, default=90.0,
                    help='seconds per config (>=3 reps always run)')
    ap.add_argument('--tier', choices=('quick', 'spec'), default=None,
                    help='quick = tiny smoke sweep; spec = the full '
                         'task sweep (default uses the CLI lists)')
    ap.add_argument('--fmad', action='store_true',
                    help='compile with FMA contraction (throughput knob; '
                         'NEVER for validation numbers)')
    ap.add_argument('--breakdown', action='store_true',
                    help='per-stage event timing for one config first')
    ap.add_argument('--out', default=os.path.join(HERE,
                                                  'bench_results.json'))
    args = ap.parse_args()
    if args.tier == 'quick':
        args.B, args.H, args.nfreq, args.K = [32, 512], [2, 8], [8000], \
            [1, 4]
        args.reps = 3

    dev = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print('bench_gpu: device=%s reps=%d budget=%.0fs fmad=%s'
          % (dev, args.reps, args.budget, args.fmad))
    results = dict(device=dev, fmad=bool(args.fmad), configs=[])
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)

    if args.breakdown:
        bd = breakdown(max(args.B), max(args.H), max(args.nfreq),
                       min(args.K))
        print('breakdown:', json.dumps(bd))
        results.setdefault('breakdowns', []).append(bd)

    hdr = ('B', 'H', 'nfreq', 'K', 'reuse', 'reps', 'total_med_s',
           'compute_s', 'h2d_s', 'd2h_s', 'us/LC/f/T', 'LC/s')
    print(('%6s %3s %6s %3s %6s %5s %12s %10s %8s %8s %10s %9s') % hdr)
    for B, H, nf, Kt in itertools.product(args.B, args.H, args.nfreq,
                                          args.K):
        for reuse in ((True, False) if Kt > 1 else (True,)):
            r = run_config(B, H, nf, Kt, reuse, args.reps, args.budget,
                           args.fmad)
            r_meta = dict(r)
            results['configs'].append(r_meta)
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=1)
            if r.get('skipped'):
                print('%6d %3d %6d %3d %6s SKIP  %s'
                      % (B, H, nf, Kt, reuse, r['reason']))
                continue
            print('%6d %3d %6d %3d %6s %5d %12.3f %10.3f %8.3f %8.3f '
                  '%10.3f %9.1f'
                  % (B, H, nf, Kt, reuse, r['reps_done'],
                     r['total_s_median'], r['compute_s_median'],
                     r['h2d_s_median'], r['d2h_s_median'],
                     r['us_per_lc_freq_tmpl'], r['lc_per_s']))
    print('results -> %s' % args.out)


if __name__ == '__main__':
    main()
