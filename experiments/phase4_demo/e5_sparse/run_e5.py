"""WP E5: sparse-regime link on REAL light curves -- recovery vs N_epochs.

Extends the WP E4 runner (../e4_recovery/run_e4.py -- vocabulary, estimator
seams, grids, peak extraction and block-checkpoint machinery are IMPORTED
from it, not reimplemented).  For each star in the stratified 24-star subset
(e5_star_subset.json; built by build_e5_subset.py from the E4 73-star table),
each method sees a SEEDED RANDOM per-epoch subsample of N epochs per band
drawn WITHOUT replacement from the full light curve:

    idx_band = sort(RandomState(crc32('E5|<uid>|N=<N>|rep=<rep>|band=<b>')
                                & 0xffffffff).choice(n_band, N, replace=False))

This deliberately replaces the old np.linspace even-thinning of the E1
protocol (ftperiodogram/simulate.py real-cadence loader), which strips
intra-night pairs; random subsampling keeps them at the survey rate.  The
draw depends only on (uid, N, rep, band) -- NOT on method -- so all 6 methods
score identical epochs within a cell.  Single-band methods use the primary
band's draw of the same realization.

Design matrix (random mode): 24 stars x N in {8,12,16,24,40} per band
x reps {0,1,2} x 6 methods = 2160 tasks.  Pooling stars x reps gives 72
realizations per (N, method) cell for Wilson CIs.

Thinning-comparison extra (validates the E1 protocol change): ``--mode thin``
runs ftp_mb at N=12 only with the OLD even-thinning, replicated verbatim from
ftperiodogram/simulate.py::from_ztf_cache:

    idx = np.linspace(0, n_band - 1, N); take np.unique(np.round(idx)) ints

(deterministic -> a single rep, rep=0).  24 extra tasks.

Grids: the star's E4 grids REUSED UNCHANGED (per-band grid for 1-band
methods, joint grid for multiband; df = F_MIN/ceil(F_MIN/(0.2/T_full)),
explicit [1, 5] cyc/day, never nyquist_factor).  Because the subsample is
drawn from the WHOLE light curve, its baseline T_sub <= T_full, hence
df <= 0.2/T_full <= 0.2/T_sub: the E4 oversampling rule holds a fortiori at
every N.  T_sub is stored per task for diagnostics; scoring uses the E4
criterion verbatim (frac < 1% AND |df|*T < 0.5 with T = T_grid_days).

Intra-night pairs: per band we store the count of epoch pairs on the same
night (consecutive sorted epochs chained while the gap < 0.4 d; pairs =
sum_nights C(k,2)) plus the raw subsampled times, so the thinning comparison
can recompute under any definition.

Chunked + resumable exactly as E4: unit = (star, N, rep, mode, method) ->
output/<uid>__N<NN>_r<rep>_<mode>__<method>.npz, skip-if-exists, per-block
.part.npz checkpoints, --budget-s wall deadline (foreground calls <= 8 min).

Usage::

    run_e5.py [--budget-s 400] [--workers 6] [--block 4096]
              [--stars UID,...|smoke2] [--N 8,12,16,24,40] [--reps 0,1,2]
              [--methods ...] [--mode random|thin] [--list]

``--stars smoke2`` = the subset's median-joint-epoch science RRab + RRc.
Exit 0 = all requested tasks complete; 3 = budget hit, rerun to resume.
"""
import argparse
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E4DIR = os.path.abspath(os.path.join(HERE, '..', 'e4_recovery'))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (REPO, E4DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_e4  # noqa: E402  (the E4 machinery, reused)

SUBSET = os.path.join(HERE, 'e5_star_subset.json')
OUTDIR = os.path.join(HERE, 'output')

N_GRID = (8, 12, 16, 24, 40)          # epochs per band
REPS = (0, 1, 2)
METHODS = run_e4.METHODS              # same 6, same order
NIGHT_GAP_D = 0.4                     # intra-night chaining threshold


# ------------------------------------------------------------- subsampling
def subsample_seed(uid, N, rep, band):
    return zlib.crc32(('E5|%s|N=%d|rep=%d|band=%s'
                       % (uid, N, rep, band)).encode()) & 0xffffffff


def draw_indices(n_band, uid, N, rep, band, mode):
    """Sorted epoch indices for one band of one realization."""
    if mode == 'thin':
        # OLD E1 protocol, verbatim from ftperiodogram/simulate.py
        # (from_ztf_cache): linspace over indices, round, unique.
        idx = np.linspace(0, n_band - 1, N)
        return np.unique(np.round(idx).astype(int))
    rs = np.random.RandomState(subsample_seed(uid, N, rep, band))
    return np.sort(rs.choice(n_band, size=N, replace=False))


def intranight_pairs(t):
    """Number of same-night epoch pairs: chain sorted epochs while the gap
    < NIGHT_GAP_D; per night of k epochs count C(k,2)."""
    t = np.sort(np.asarray(t, dtype=float))
    total, k = 0, 1
    for i in range(1, t.size):
        if t[i] - t[i - 1] < NIGHT_GAP_D:
            k += 1
        else:
            total += k * (k - 1) // 2
            k = 1
    return total + k * (k - 1) // 2


def task_inputs(star, method, N, rep, mode):
    """Subsampled (t, y, bands, dy, grid_entry, band_used, sub_meta)."""
    arrays = run_e4.load_star_arrays(star)
    sub, meta = {}, {}
    for band in ('g', 'r'):
        t, y, dy = arrays[band]
        idx = draw_indices(t.size, star['uid'], N, rep, band, mode)
        sub[band] = (t[idx], y[idx], dy[idx])
        meta[band] = {
            'idx': idx, 't': t[idx],
            'seed': (subsample_seed(star['uid'], N, rep, band)
                     if mode == 'random' else -1),
            'n_intranight_pairs': intranight_pairs(t[idx]),
            'T_sub_days': float(t[idx][-1] - t[idx][0]),
        }
    if method in ('gls_1band', 'ftp_1band'):
        pb = star['primary_band']
        t, y, dy = sub[pb]
        bands = np.array([pb] * t.size)
        return t, y, bands, dy, star['grid'][pb], pb, meta
    t = np.concatenate([sub[b][0] for b in ('g', 'r')])
    y = np.concatenate([sub[b][1] for b in ('g', 'r')])
    dy = np.concatenate([sub[b][2] for b in ('g', 'r')])
    bands = np.concatenate([[b] * sub[b][0].size for b in ('g', 'r')])
    order = np.argsort(t)
    return (t[order], y[order], bands[order], dy[order],
            star['grid']['joint'], 'g+r', meta)


def task_stem(uid, N, rep, mode, method):
    return '%s__N%02d_r%d_%s__%s' % (uid, N, rep, mode, method)


# ------------------------------------------------------------------ the task
MAX_ATTEMPTS = 2                      # failures recorded; skipped after 2


def fail_path(stem):
    return os.path.join(OUTDIR, stem + '.fail')


def n_failures(stem):
    """Number of recorded failed attempts for a task (lines in .fail)."""
    fp = fail_path(stem)
    if not os.path.exists(fp):
        return 0
    with open(fp) as fh:
        return sum(1 for line in fh if line.startswith('ATTEMPT'))


def run_task(args):
    """One (star, N, rep, mode, method) unit with block checkpointing.
    Same contract as run_e4.run_task; exceptions are recorded to
    output/<stem>.fail (one ATTEMPT line + traceback per failure) and
    reported as status 'failed' instead of killing the pool."""
    star, method, N, rep, mode, deadline, block = args
    stem = task_stem(star['uid'], N, rep, mode, method)
    try:
        return _run_task_inner(args)
    except Exception:
        import traceback
        with open(fail_path(stem), 'a') as fh:
            fh.write('ATTEMPT %s\n%s\n'
                     % (time.strftime('%Y-%m-%d %H:%M:%S'),
                        traceback.format_exc()))
        return stem, 'failed', 0, 0, 0.0


def _run_task_inner(args):
    star, method, N, rep, mode, deadline, block = args
    uid = star['uid']
    stem = task_stem(uid, N, rep, mode, method)
    final = os.path.join(OUTDIR, stem + '.npz')
    part = os.path.join(OUTDIR, stem + '.part.npz')
    if os.path.exists(final):
        return stem, 'exists', 0, 0, 0.0
    if time.time() > deadline:
        return stem, 'skipped', 0, 0, 0.0

    t, y, bands, dy, grid, band_used, meta = \
        task_inputs(star, method, N, rep, mode)
    df, i_min, n_freq = grid['df'], grid['i_min'], grid['n_freq']
    freqs = df * (i_min + np.arange(n_freq))

    n_done, runtime = 0, 0.0
    stat = np.full(n_freq, np.nan)
    if os.path.exists(part):
        p = np.load(part)
        if (int(p['n_freq']) == n_freq and float(p['df']) == df
                and str(p['method']) == method):
            n_done = int(p['n_done'])
            stat[:n_done] = p['stat'][:n_done]
            runtime = float(p['runtime_s'])

    est, config = run_e4.make_estimator(method)

    def stat_block(fblock):
        if method == 'ce':
            return -est.entropy_spectrum(t, y, bands, dy, fblock)
        return est.power_spectrum(t, y, bands, dy, fblock)

    while n_done < n_freq:
        hi = min(n_done + block, n_freq)
        t0 = time.time()
        stat[n_done:hi] = stat_block(freqs[n_done:hi])
        runtime += time.time() - t0
        n_done = hi
        np.savez(part + '.tmp.npz', method=method, df=df, n_freq=n_freq,
                 n_done=n_done, stat=stat, runtime_s=runtime)
        os.replace(part + '.tmp.npz', part)
        if time.time() > deadline and n_done < n_freq:
            return stem, 'partial', n_done, n_freq, runtime

    rayleigh = 1.0 / grid['T_days']
    pf, ps = run_e4.top_peaks(freqs, stat, rayleigh)
    from ftperiodogram.baselines import MHLSEstimator
    mhls_order = (MHLSEstimator(8)._capped_order(t.size, 2)
                  if method == 'mhls_h8' else -1)
    np.savez_compressed(
        final + '.tmp.npz',
        uid=uid, method=method, config=config, band_used=band_used,
        mode=mode, N_per_band=N, rep=rep,
        seed_g=meta['g']['seed'], seed_r=meta['r']['seed'],
        idx_g=meta['g']['idx'], idx_r=meta['r']['idx'],
        t_g=meta['g']['t'], t_r=meta['r']['t'],
        n_intranight_pairs_g=meta['g']['n_intranight_pairs'],
        n_intranight_pairs_r=meta['r']['n_intranight_pairs'],
        T_sub_g_days=meta['g']['T_sub_days'],
        T_sub_r_days=meta['r']['T_sub_days'],
        night_gap_days=NIGHT_GAP_D,
        f_min=run_e4.F_MIN, f_max=run_e4.F_MAX,
        df=df, i_min=i_min, n_freq=n_freq,
        T_grid_days=grid['T_days'], T_joint_days=star['joint']['T_days'],
        pts_per_rayleigh=grid['pts_per_rayleigh'], n_obs=t.size,
        stat=stat.astype(np.float32), stat_is_neg_entropy=(method == 'ce'),
        best_freq=freqs[int(np.nanargmax(stat))],
        top5_freq=pf, top5_stat=ps, mhls_capped_order=mhls_order,
        runtime_s=runtime,
        role=star['role'], var_type=star['type'],
        dual_truth=star['dual_truth'], primary_band=star['primary_band'],
        chen_period=(star['chen_period'] or np.nan),
        gaia_period=(star['gaia_period'] or np.nan))
    os.replace(final + '.tmp.npz', final)
    if os.path.exists(part):
        os.remove(part)
    return stem, 'done', n_freq, n_freq, runtime


# ---------------------------------------------------------------------- main
def pick_smoke(subset_uids, stars_by_uid):
    """Median-joint-epoch science RRab + RRc WITHIN the E5 subset
    (mirror of run_e4.pick_smoke)."""
    out = []
    subset = [stars_by_uid[u] for u in subset_uids]
    for typ in ('RRab', 'RRc'):
        sci = sorted((s for s in subset
                      if s['role'] == 'science' and s['type'] == typ),
                     key=lambda s: (s['joint']['n_epochs'], s['uid']))
        out.append(sci[len(sci) // 2]['uid'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-s', type=float, default=400.0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--block', type=int, default=4096)
    ap.add_argument('--stars', default=None,
                    help="comma-separated uids, or 'smoke2' (default: full "
                         "24-star subset)")
    ap.add_argument('--N', default=','.join(str(n) for n in N_GRID))
    ap.add_argument('--reps', default=','.join(str(r) for r in REPS))
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--mode', choices=('random', 'thin'), default='random')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    warnings.simplefilter('ignore')

    with open(os.path.join(E4DIR, 'star_table.json')) as fh:
        by_uid = {s['uid']: s for s in json.load(fh)['stars']}
    with open(SUBSET) as fh:
        subset_uids = [r['uid'] for r in json.load(fh)['stars']]

    if args.stars == 'smoke2':
        wanted = pick_smoke(subset_uids, by_uid)
    elif args.stars:
        wanted = args.stars.split(',')
    else:
        wanted = subset_uids
    missing = [u for u in wanted if u not in by_uid]
    if missing:
        raise SystemExit('unknown uids: %s' % missing)
    Ns = [int(x) for x in args.N.split(',')]
    reps = [int(x) for x in args.reps.split(',')] \
        if args.mode == 'random' else [0]    # thinning is deterministic
    methods = args.methods.split(',')
    for m in methods:
        if m not in METHODS:
            raise SystemExit('unknown method %r' % m)
    for u in wanted:
        for N in Ns:
            for b in 'gr':
                if by_uid[u]['bands'][b]['n_epochs'] < N:
                    raise SystemExit('%s band %s has < %d epochs' % (u, b, N))

    tasks, n_exists, failed_skipped = [], 0, []
    for uid in wanted:
        for N in Ns:
            for rep in reps:
                for m in methods:
                    stem = task_stem(uid, N, rep, args.mode, m)
                    if os.path.exists(os.path.join(OUTDIR, stem + '.npz')):
                        n_exists += 1
                    elif n_failures(stem) >= MAX_ATTEMPTS:
                        failed_skipped.append(stem)
                    else:
                        tasks.append((uid, N, rep, m))
    print('tasks: %d pending, %d complete, %d failed-skipped '
          '(of %d requested)'
          % (len(tasks), n_exists, len(failed_skipped),
             len(tasks) + n_exists + len(failed_skipped)), flush=True)
    for stem in failed_skipped:
        print('  FAILED_SKIPPED %s (%d attempts)' % (stem, n_failures(stem)),
              flush=True)
    if args.list:
        for uid, N, rep, m in tasks:
            stem = task_stem(uid, N, rep, args.mode, m)
            part = os.path.join(OUTDIR, stem + '.part.npz')
            frac = ''
            if os.path.exists(part):
                p = np.load(part)
                frac = ' [%.0f%%]' % (100.0 * int(p['n_done'])
                                      / int(p['n_freq']))
            print('  pending %s%s' % (stem, frac))
        return 0
    if not tasks:
        print('ALL_DONE')
        return 0

    deadline = time.time() + args.budget_s
    vocab_coeffs = run_e4.build_vocab_coeffs()
    payloads = [(by_uid[uid], m, N, rep, args.mode, deadline, args.block)
                for uid, N, rep, m in tasks]

    results = []
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers, initializer=run_e4._init_worker,
                      initargs=(vocab_coeffs,)) as pool:
            for res in pool.imap_unordered(run_task, payloads):
                results.append(res)
                print('  %s %s %d/%d %.1fs' % res, flush=True)
    else:
        run_e4._init_worker(vocab_coeffs)
        for pl in payloads:
            if time.time() > deadline:
                break
            res = run_task(pl)
            results.append(res)
            print('  %s %s %d/%d %.1fs' % res, flush=True)

    n_left = sum(1 for r in results
                 if r[1] in ('partial', 'skipped', 'failed')) \
        + len(payloads) - len(results)
    print('BUDGET_HIT rerun to resume (%d unfinished)' % n_left
          if n_left else 'ALL_DONE', flush=True)
    return 3 if n_left else 0


if __name__ == '__main__':
    sys.exit(main())
