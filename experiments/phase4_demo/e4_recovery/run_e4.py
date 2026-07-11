"""WP E4: known-period recovery on the Phase-4 ZTF RRL sample -- runner.

Computes, per unique star (73; see ``build_star_table.py`` /
``star_table.json``) and per method, the periodogram over the star's explicit
[1, 5] cyc/day grid and persists the full spectrum plus the top-5 distinct
peaks (for alias classification / adjudication in the Score phase).

Methods (the paper's sim-validation reference set; estimator seams REUSED
from ``ftperiodogram.baselines`` -- nothing reimplemented):

  ==========  ==============================================================
  gls_1band   GLSEstimator on the primary band (band with more epochs)
  ftp_1band   FTPEstimator (K=8 order-8 Sesar PAM vocab, floating_offsets,
              catalog-max) on the primary band
  mbls_h1     MultibandLSEstimator(n_harmonics=1)  [VanderPlas & Ivezic 2015]
  mhls_h8     MHLSEstimator(n_harmonics=8), capped per B2 (cap is built in)
  ce          ConditionalEntropyEstimator (Graham+13 defaults 10x5)
  ftp_mb      FTPEstimator (same vocab), g+r joint, floating_offsets
  ==========  ==============================================================

Vocabulary: identical construction to D1/E3 (``fetch_sesar_templates(
nharmonics=8)`` -> ``build_template_catalog(..., 8, method='pam',
random_state=0)``).

Grid: from ``star_table.json`` -- per-star df = F_MIN/ceil(F_MIN/(0.2/T))
(NFFT-snapped, <= 0.2/T so >= 5 points per Rayleigh), T = per-band baseline;
two-band methods use the joint grid at the shorter per-band T; freqs =
df * (i_min + arange(n_freq)).  df and points-per-Rayleigh are stored in
every output npz and printed per task.

Chunked + resumable (HARD REQUIREMENT -- foreground calls <= 8 min):
  * unit of work = (star, method); final artifact
    ``output/<uid>__<method>.npz`` (skip-if-exists),
  * within a unit, the grid is processed in blocks of ``--block`` freqs with
    a partial checkpoint ``output/<uid>__<method>.part.npz`` written after
    every block, so a deadline can interrupt ANY task mid-grid and a rerun
    resumes at the next block,
  * ``--budget-s`` sets the wall deadline; workers stop at the next block
    boundary past it (overshoot <= one block, ~11 s for the FTP methods).

Usage::

    run_e4.py [--budget-s 400] [--workers 4] [--block 2048]
              [--stars UID,UID|smoke2] [--methods m1,m2] [--list]

``--stars smoke2`` = deterministic smoke pair: the median-epoch science RRab
and RRc.  Exit code 0 = all requested tasks complete; 3 = budget hit, rerun
to resume.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)

STAR_TABLE = os.path.join(HERE, 'star_table.json')
DATA_DIR = os.path.expanduser('~/.ftperiodogram_data/phase4_rrl_sample')
OUTDIR = os.path.join(HERE, 'output')

F_MIN, F_MAX = 1.0, 5.0            # cyc/day, explicit (never nyquist_factor)
VOCAB_H = 8                        # PAM vocabulary order (as D1/E3)
VOCAB_K = 8                        # catalog size (as D1/E3)
PAM_SEED = 0                       # identical vocab to D1/E3
N_PEAKS = 5                        # distinct peaks persisted per spectrum
METHODS = ('gls_1band', 'ftp_1band', 'mbls_h1', 'mhls_h8', 'ce', 'ftp_mb')

_G = {}                            # per-process globals (vocab)


# ---------------------------------------------------------------- vocabulary
def build_vocab_coeffs():
    """Identical to D1/E3 build_templates(): K=8 PAM medoids of the order-8
    Sesar library, seed 0.  Returned as plain coefficient arrays (picklable)."""
    from ftperiodogram.catalog_builder import (fetch_sesar_templates,
                                               build_template_catalog)
    lib = fetch_sesar_templates(nharmonics=VOCAB_H)
    vocab = build_template_catalog(lib, VOCAB_K, method='pam',
                                   random_state=PAM_SEED)
    return [(np.asarray(tm.c_n), np.asarray(tm.s_n)) for tm in vocab]


def _vocab_templates():
    from ftperiodogram.template import Template
    return [Template(c_n=c, s_n=s) for c, s in _G['vocab_coeffs']]


def _init_worker(vocab_coeffs):
    warnings.simplefilter('ignore')
    _G['vocab_coeffs'] = [(np.asarray(c), np.asarray(s))
                          for c, s in vocab_coeffs]


# -------------------------------------------------------------------- data IO
def load_star_arrays(star):
    """(t, y, dy, band_labels) per band, non-finite rows dropped (same mask
    as build_star_table.lc_stats)."""
    d = np.load(os.path.join(DATA_DIR, star['lc_file']))
    out = {}
    for band in ('g', 'r'):
        t = np.asarray(d['%s_hjd' % band], dtype=float)
        y = np.asarray(d['%s_mag' % band], dtype=float)
        dy = np.asarray(d['%s_magerr' % band], dtype=float)
        good = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy)
        order = np.argsort(t[good])
        out[band] = (t[good][order], y[good][order], dy[good][order])
    return out


def task_inputs(star, method):
    """(t, y, bands, dy, grid_entry, band_used) for one (star, method)."""
    arrays = load_star_arrays(star)
    if method in ('gls_1band', 'ftp_1band'):
        pb = star['primary_band']
        t, y, dy = arrays[pb]
        bands = np.array([pb] * t.size)
        return t, y, bands, dy, star['grid'][pb], pb
    t = np.concatenate([arrays[b][0] for b in ('g', 'r')])
    y = np.concatenate([arrays[b][1] for b in ('g', 'r')])
    dy = np.concatenate([arrays[b][2] for b in ('g', 'r')])
    bands = np.concatenate([[b] * arrays[b][0].size for b in ('g', 'r')])
    order = np.argsort(t)
    return t[order], y[order], bands[order], dy[order], \
        star['grid']['joint'], 'g+r'


def make_estimator(method):
    """Estimator seams from ftperiodogram.baselines (REUSED, not rewritten).
    Returns (estimator, config_str).  For CE the maximized statistic is
    -entropy (lower entropy = better fold)."""
    from ftperiodogram.baselines import (GLSEstimator, MHLSEstimator,
                                         MultibandLSEstimator,
                                         ConditionalEntropyEstimator,
                                         FTPEstimator)
    if method == 'gls_1band':
        return GLSEstimator(), 'GLSEstimator()'
    if method == 'mhls_h8':
        return MHLSEstimator(n_harmonics=8), 'MHLSEstimator(8) [B2 cap]'
    if method == 'mbls_h1':
        return MultibandLSEstimator(n_harmonics=1), 'MultibandLSEstimator(1)'
    if method == 'ce':
        return ConditionalEntropyEstimator(), 'CondEntropy(10x5), stat=-H'
    if method in ('ftp_1band', 'ftp_mb'):
        return (FTPEstimator(_vocab_templates(), mode='floating_offsets'),
                'FTPEstimator(K=%d,H=%d,pam,seed=%d,floating_offsets,'
                'catalog-max)' % (VOCAB_K, VOCAB_H, PAM_SEED))
    raise ValueError(method)


# ------------------------------------------------------------ peak extraction
def top_peaks(freqs, stat, rayleigh, n_peaks=N_PEAKS):
    """Top-``n_peaks`` distinct local maxima of ``stat`` (higher = better),
    greedily separated by >= one Rayleigh width in frequency."""
    s = np.asarray(stat)
    is_max = np.ones(s.size, dtype=bool)
    is_max[1:] &= s[1:] > s[:-1]
    is_max[:-1] &= s[:-1] >= s[1:]
    cand = np.flatnonzero(is_max)
    cand = cand[np.argsort(s[cand])[::-1]]
    picked = []
    for i in cand:
        if all(abs(freqs[i] - freqs[j]) >= rayleigh for j in picked):
            picked.append(i)
            if len(picked) == n_peaks:
                break
    pf = np.full(n_peaks, np.nan)
    ps = np.full(n_peaks, np.nan)
    pf[:len(picked)] = freqs[picked]
    ps[:len(picked)] = s[picked]
    return pf, ps


# ------------------------------------------------------------------ the task
def run_task(args):
    """Compute one (star, method) unit with block checkpointing.

    Returns (uid, method, status, n_done, n_freq, secs) with status in
    {'done', 'partial', 'exists'}.
    """
    star, method, deadline, block = args
    uid = star['uid']
    final = os.path.join(OUTDIR, '%s__%s.npz' % (uid, method))
    part = final[:-4] + '.part.npz'
    if os.path.exists(final):
        return uid, method, 'exists', 0, 0, 0.0
    if time.time() > deadline:            # budget already hit: don't start
        return uid, method, 'skipped', 0, 0, 0.0

    t, y, bands, dy, grid, band_used = task_inputs(star, method)
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

    est, config = make_estimator(method)

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
            return uid, method, 'partial', n_done, n_freq, runtime

    rayleigh = 1.0 / grid['T_days']
    pf, ps = top_peaks(freqs, stat, rayleigh)
    from ftperiodogram.baselines import MHLSEstimator
    mhls_order = (MHLSEstimator(8)._capped_order(t.size, 2)
                  if method == 'mhls_h8' else -1)
    np.savez_compressed(
        final + '.tmp.npz',
        uid=uid, method=method, config=config, band_used=band_used,
        f_min=F_MIN, f_max=F_MAX, df=df, i_min=i_min, n_freq=n_freq,
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
    return uid, method, 'done', n_freq, n_freq, runtime


# ---------------------------------------------------------------------- main
def pick_smoke(stars):
    """Deterministic smoke pair: median-joint-epoch science RRab and RRc."""
    out = []
    for typ in ('RRab', 'RRc'):
        sci = sorted((s for s in stars
                      if s['role'] == 'science' and s['type'] == typ),
                     key=lambda s: (s['joint']['n_epochs'], s['uid']))
        out.append(sci[len(sci) // 2]['uid'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-s', type=float, default=400.0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--block', type=int, default=2048)
    ap.add_argument('--stars', default=None,
                    help="comma-separated uids, or 'smoke2'")
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    warnings.simplefilter('ignore')

    with open(STAR_TABLE) as fh:
        stars = json.load(fh)['stars']
    if args.stars == 'smoke2':
        wanted = pick_smoke(stars)
    elif args.stars:
        wanted = args.stars.split(',')
    else:
        wanted = [s['uid'] for s in stars]
    by_uid = {s['uid']: s for s in stars}
    missing = [u for u in wanted if u not in by_uid]
    if missing:
        raise SystemExit('unknown uids: %s' % missing)
    methods = args.methods.split(',')

    tasks, n_exists = [], 0
    for uid in wanted:
        for m in methods:
            if m not in METHODS:
                raise SystemExit('unknown method %r' % m)
            if os.path.exists(os.path.join(OUTDIR, '%s__%s.npz' % (uid, m))):
                n_exists += 1
            else:
                tasks.append((uid, m))
    print('tasks: %d pending, %d complete (of %d requested)'
          % (len(tasks), n_exists, len(tasks) + n_exists), flush=True)
    if args.list:
        for uid, m in tasks:
            part = os.path.join(OUTDIR, '%s__%s.part.npz' % (uid, m))
            frac = ''
            if os.path.exists(part):
                p = np.load(part)
                frac = ' [%.0f%%]' % (100.0 * int(p['n_done']) / int(p['n_freq']))
            print('  pending %s %s%s' % (uid, m, frac))
        return 0
    if not tasks:
        print('ALL_DONE')
        return 0

    deadline = time.time() + args.budget_s
    vocab_coeffs = build_vocab_coeffs()
    payloads = [(by_uid[uid], m, deadline, args.block) for uid, m in tasks]

    results = []
    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(vocab_coeffs,)) as pool:
            for res in pool.imap_unordered(run_task, payloads):
                results.append(res)
                print('  %s %s %s %d/%d %.1fs' % res, flush=True)
                # do not submit past deadline: imap has already queued all
                # payloads, but each task checks the deadline itself.
    else:
        _init_worker(vocab_coeffs)
        for pl in payloads:
            if time.time() > deadline:
                break
            res = run_task(pl)
            results.append(res)
            print('  %s %s %s %d/%d %.1fs' % res, flush=True)

    n_left = sum(1 for r in results if r[2] in ('partial', 'skipped')) \
        + len(payloads) - len(results)
    print('BUDGET_HIT rerun to resume (%d unfinished)' % n_left
          if n_left else 'ALL_DONE', flush=True)
    return 3 if n_left else 0


if __name__ == '__main__':
    sys.exit(main())
