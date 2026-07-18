"""B6 -- GLS-cascade survival fraction (MEASUREMENT arm only, no adoption).

The audit's SCIENCE-class cascade lever (AUDIT_2026-07-18 S2: ~3.5-4.5x on
detection): a cheap multiband GLS stage ranks the frequency grid; only the top
fraction q of frequencies -- PLUS their 1-day alias partners f+/-1, f+/-2 and
harmonic partners f/2, 2f, each snapped to the nearest in-band grid frequency
-- reach the FTP stage. This arm measures what that cut costs: the fraction of
injected sources whose true frequency (or an accepted alias per lib.score's
crediting) is still inside the surviving set, at q in {.01,.02,.05,.10,.20},
with and without the partner expansion. No FTP scan is run here -- survival is
a pure property of the GLS ranking + the acceptance set.

GLS stage: ftperiodogram.baselines.GLSEstimator IS the multiband GLS (one
shared sinusoid + per-band floating offsets -- the floating_offsets model at
H=1), so no per-band single-band combination is needed.

Acceptance set: a grid frequency f is accepted iff lib.score(1/f, ...) would
report frac_recovered -- i.e. the fractional rtol=0.01 test against P_true or
any member of recovery.harmonic_alias_set (P/2, 2P, P/3, 3P, 1-day and 1-yr
beats). Implemented vectorized from the same alias-set helper and
recovered_fractional, and spot-checked per source against lib.score itself.

Population/cadence/grid: identical construction to b3 (PS1 TTI-pair griz,
N=8/band, mag 21.0, drawn p2p; per-source converged grid at os=3 for the
production detection H -- RRab H=4, RRc H=2 -- over the b1 search bands).
Same (seed0, i) keying as b3, so the b3/b6 populations are paired.

Incremental per-source output (<out>.part.jsonl, --resume), finalized to
<out>.json. Explicit search bands only (no nyquist_factor anywhere).
"""
import argparse
import json
import os
import sys
import time
import warnings

# single-thread BLAS per worker (set before numpy import; workers re-import
# this module under spawn so they inherit it too)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

from ftperiodogram import recovery as rec
from ftperiodogram.baselines import GLSEstimator

warnings.filterwarnings('ignore')

SEARCH_BAND = {'RRab': (0.9, 3.4), 'RRc': (1.8, 6.5)}
DETECTION_H = {'RRab': 4, 'RRc': 2}
Q_LIST = (0.01, 0.02, 0.05, 0.10, 0.20)
RTOL = 0.01                   # lib.score's fractional-criterion default
SPLIT_SEED = 0                # b1/b3 production split (truth half only here)


def build_source(subtype, i, seed0, truth_order, truth_stars, n_per_band,
                 mean_mag, baseline_days, season_length_days):
    """Identical construction (and (seed0, i) keying) to b3.build_source."""
    bank = lib.load_bank(subtype, 8)
    star = truth_stars[truth_order[i % len(truth_stars)]]
    truth = lib.shared_shape_template(bank, star)
    cad = lib.ps1_cadence(n_per_band, baseline_days=baseline_days,
                          season_length_days=season_length_days,
                          seed=(seed0 * 100003 + i) % (2 ** 32))
    src = lib.inject(truth, subtype, cad, seed=(seed0 * 7919 + i) % (2 ** 32),
                     mean_mag=mean_mag)
    src['star'] = star
    return src


def accepted_masks(grid, P_true):
    """(accepted, accepted_exact) boolean masks over the sorted grid.

    ``accepted[j]`` is True iff ``lib.score(1/grid[j], P_true, T)`` would set
    frac_recovered: the rtol test against P_true or any alias from
    recovery.harmonic_alias_set. Each target contributes a narrow frequency
    interval; candidates are located by searchsorted (with margin) and then
    filtered by recovered_fractional itself, so the mask reproduces the
    classifier exactly rather than re-deriving its inequalities."""
    accepted = np.zeros(grid.size, dtype=bool)
    accepted_exact = np.zeros(grid.size, dtype=bool)
    targets = [('exact', float(P_true))]
    targets += [(a.name, float(a.P_alias))
                for a in rec.harmonic_alias_set(P_true)]
    for name, Pa in targets:
        f_lo = 1.0 / (Pa * (1.0 + RTOL))
        f_hi = 1.0 / (Pa * (1.0 - RTOL))
        i0 = max(0, int(np.searchsorted(grid, f_lo)) - 2)
        i1 = min(grid.size, int(np.searchsorted(grid, f_hi)) + 2)
        if i1 <= i0:
            continue
        sel = np.arange(i0, i1)
        ok = rec.recovered_fractional(1.0 / grid[sel], Pa, rtol=RTOL)
        accepted[sel[ok]] = True
        if name == 'exact':
            accepted_exact[sel[ok]] = True
    return accepted, accepted_exact


def expand_partners(grid, idx):
    """Surviving set = idx plus the 1-day alias partners f+/-1, f+/-2 and the
    harmonic partners f/2, 2f of every surviving frequency, each snapped to
    the NEAREST grid frequency, kept only if within the band."""
    f = grid[idx]
    partners = np.concatenate([f + 1.0, f - 1.0, f + 2.0, f - 2.0,
                               0.5 * f, 2.0 * f])
    partners = partners[(partners >= grid[0]) & (partners <= grid[-1])]
    if partners.size == 0:
        return np.asarray(idx)
    j = np.clip(np.searchsorted(grid, partners), 1, grid.size - 1)
    j = np.where(np.abs(grid[j] - partners) < np.abs(grid[j - 1] - partners),
                 j, j - 1)
    return np.union1d(idx, j)


_G = {}


def _init_worker(subtype, band, oversample, seed0, truth_order, truth_stars,
                 n_per_band, mean_mag, baseline_days, season_length_days):
    _G.update(subtype=subtype, band=band, oversample=oversample, seed0=seed0,
              truth_order=truth_order, truth_stars=truth_stars,
              n_per_band=n_per_band, mean_mag=mean_mag,
              baseline_days=baseline_days,
              season_length_days=season_length_days, gls=GLSEstimator())


def _score_source(i):
    src = build_source(_G['subtype'], i, _G['seed0'], _G['truth_order'],
                       _G['truth_stars'], _G['n_per_band'], _G['mean_mag'],
                       _G['baseline_days'], _G['season_length_days'])
    f_lo, f_hi = _G['band']
    H = DETECTION_H[_G['subtype']]
    grid = lib.converged_grid(f_lo, f_hi, H, src['baseline'],
                              oversample=_G['oversample'])
    n = grid.size

    accepted, accepted_exact = accepted_masks(grid, src['P_true'])
    # per-source fidelity spot-check of the vectorized mask vs lib.score
    # itself: a few random grid points plus one accepted point (if any)
    chk_rng = np.random.RandomState((_G['seed0'] * 31 + i) % (2 ** 32))
    chk = list(chk_rng.choice(n, size=8, replace=False))
    if accepted.any():
        chk.append(int(np.where(accepted)[0][0]))
    for j in chk:
        s = lib.score(1.0 / grid[j], src['P_true'], src['baseline'],
                      rtol=RTOL)
        assert bool(accepted[j]) == bool(s['frac_recovered']), \
            "accepted-mask mismatch at grid[%d] (source %d)" % (j, i)
        assert bool(accepted_exact[j]) == bool(s['frac_exact']), \
            "exact-mask mismatch at grid[%d] (source %d)" % (j, i)

    powers = _G['gls'].power_spectrum(src['t'], src['y'], src['bands'],
                                      src['dy'], grid)
    order = np.argsort(-powers, kind='stable')        # rank 0 = best GLS power
    rank_of = np.empty(n, dtype=np.int64)
    rank_of[order] = np.arange(n)

    def _min_q(mask):
        return ((int(rank_of[mask].min()) + 1) / float(n) if mask.any()
                else None)

    out = {'i': int(i), 'star': src['star'], 'P_true': src['P_true'],
           'p2p_g': src['p2p_g'], 'baseline': src['baseline'],
           'N': int(src['t'].size), 'nfreq': int(n),
           'n_accepted': int(accepted.sum()),
           'n_accepted_exact': int(accepted_exact.sum()),
           'min_q_plain': _min_q(accepted),
           'min_q_exact_plain': _min_q(accepted_exact),
           'survival': {}}
    for q in Q_LIST:
        m = max(1, int(np.ceil(q * n)))
        top = order[:m]
        surv = expand_partners(grid, top)
        out['survival']['q%.2f' % q] = {
            'plain': bool(accepted[top].any()),
            'expanded': bool(accepted[surv].any()),
            'exact_plain': bool(accepted_exact[top].any()),
            'exact_expanded': bool(accepted_exact[surv].any())}
    return _py(out)


def _py(x):
    """JSON-safe copy (numpy scalars -> Python scalars)."""
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def load_part(part_path, config, resume):
    """Return {i: record} already computed (see b3_vocab_size.load_part)."""
    if not os.path.exists(part_path):
        return {}
    if not resume:
        os.remove(part_path)
        return {}
    done = {}
    with open(part_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue                    # truncated tail of a killed run
            if '_config' in r:
                for k, v in config.items():
                    if r['_config'].get(k) != v:
                        raise SystemExit(
                            "--resume config mismatch on %r: part has %r, "
                            "args give %r" % (k, r['_config'].get(k), v))
                continue
            done[int(r['i'])] = r
    print("resuming: %d sources already in %s" % (len(done), part_path))
    return done


def run(args):
    subtype = args.subtype
    band = SEARCH_BAND[subtype]
    H = DETECTION_H[subtype]
    _voc, truth_stars = lib.split_stars(subtype, seed=SPLIT_SEED)
    rng = np.random.RandomState(args.seed0 % (2 ** 32))
    truth_order = rng.permutation(len(truth_stars))

    config = dict(subtype=subtype, H=H, n_sources=args.n_sources,
                  n_per_band=args.n_per_band, mean_mag=args.mean_mag,
                  oversample=args.oversample, baseline_days=args.baseline_days,
                  season_length_days=args.season_length_days,
                  seed0=args.seed0, split_seed=SPLIT_SEED, band=list(band),
                  q_list=list(Q_LIST), rtol=RTOL,
                  partners=['f+1', 'f-1', 'f+2', 'f-2', 'f/2', '2f'],
                  gls='baselines.GLSEstimator (multiband: shared sinusoid + '
                      'per-band floating offsets)',
                  heldout=dict(n_truth_stars=len(truth_stars)))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output',
        'b6_cascade_%s.json' % subtype.lower())
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    part_path = out_path + '.part.jsonl'
    done = load_part(part_path, config, args.resume)
    todo = [i for i in range(args.n_sources) if i not in done]

    initargs = (subtype, band, args.oversample, args.seed0, truth_order,
                truth_stars, args.n_per_band, args.mean_mag,
                args.baseline_days, args.season_length_days)
    t0 = time.time()
    with open(part_path, 'a') as fh:
        if not done:
            fh.write(json.dumps({'_config': config}) + '\n')
            fh.flush()
        if todo:
            with Pool(min(args.nproc, len(todo)), initializer=_init_worker,
                      initargs=initargs) as pool:
                for k, rec_i in enumerate(pool.imap_unordered(_score_source,
                                                              todo)):
                    fh.write(json.dumps(rec_i) + '\n')
                    fh.flush()
                    done[rec_i['i']] = rec_i
                    n_done = len(done)
                    if n_done % 16 == 0 or n_done == args.n_sources:
                        dt = time.time() - t0
                        print("  [%d/%d] %.1f s elapsed (%.2f wall-s/src)"
                              % (n_done, args.n_sources, dt, dt / (k + 1)),
                              flush=True)
    dt = time.time() - t0
    if todo:
        print("computed %d sources in %.1f s (%.2f wall-s/src at nproc=%d)"
              % (len(todo), dt, dt / len(todo), args.nproc))

    res = dict(config=config, sources=[done[i] for i in sorted(done)])
    with open(out_path, 'w') as fh:
        json.dump(res, fh)
    os.remove(part_path)
    print("wrote", out_path)
    return res


def _rate_se(p, n):
    """Binomial SE; p==0 or 1 quotes the rule-of-three 95% bound as (<=x)."""
    if n and 0.0 < p < 1.0:
        return "%.3f+/-%.3f" % (p, np.sqrt(p * (1.0 - p) / n))
    return "%.3f (<=%.3f)" % (p, 3.0 / max(n, 1))


def summarize(res):
    cfg = res['config']
    srcs = res['sources']
    n = len(srcs)
    lines = ["subtype=%s  H=%d  n=%d  N/band=%s  mag=%.1f  os=%.1f  "
             "nfreq~%d  seed0=%d"
             % (cfg['subtype'], cfg['H'], n, cfg['n_per_band'],
                cfg['mean_mag'], cfg['oversample'],
                int(np.median([s['nfreq'] for s in srcs])) if srcs else 0,
                cfg['seed0'])]
    lines.append("   q    survive         +partners       "
                 "exact-only      exact+partners")
    for q in Q_LIST:
        key = 'q%.2f' % q
        cols = []
        for field in ('plain', 'expanded', 'exact_plain', 'exact_expanded'):
            p = np.mean([s['survival'][key][field] for s in srcs])
            cols.append("%-15s" % _rate_se(p, n))
        lines.append("  %.2f  %s" % (q, " ".join(cols)))
    mq = [s['min_q_plain'] for s in srcs if s['min_q_plain'] is not None]
    if mq:
        lines.append("  min-q to survive plain (alias-credited): median %.4f"
                     "  p90 %.4f  max %.4f"
                     % (np.median(mq), np.percentile(mq, 90), np.max(mq)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subtype', required=True, choices=['RRab', 'RRc'])
    ap.add_argument('--n-sources', type=int, default=256)
    ap.add_argument('--seed0', type=int, default=9000)
    ap.add_argument('--n-per-band', type=int, default=8)
    ap.add_argument('--mean-mag', type=float, default=21.0)
    ap.add_argument('--oversample', type=float, default=3.0)
    ap.add_argument('--baseline-days', type=float, default=1300.0)
    ap.add_argument('--season-length-days', type=float, default=250.0)
    ap.add_argument('--nproc', type=int, default=os.cpu_count())
    ap.add_argument('--resume', action='store_true',
                    help='continue a killed run from <out>.part.jsonl')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    res = run(args)
    print(summarize(res))


if __name__ == '__main__':
    main()
