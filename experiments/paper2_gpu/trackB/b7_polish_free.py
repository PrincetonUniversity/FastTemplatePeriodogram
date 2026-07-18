"""B7 -- polish-free detection neutrality (MEASUREMENT arm only, no adoption).

The audit's SCIENCE-class polish-free lever (AUDIT_2026-07-18 S2: ~1.2-1.5x
incremental): run the detection scan with NO Newton polish (grid + dip-seed
argmax only) and polish only surfaced candidates. This arm measures whether
dropping the polish from the catalog power spectrum changes DETECTION at all:
for each injected source the FTP catalog spectrum (RRab production detection
arm: H=4, K=4 held-out vocab, b3's cadence/mag/grid) is computed twice --

  (a) production settings (scan + up-to-8-step Newton polish, the default),
  (b) n_newton=0: candidates are the raw circle-grid maxima plus the unrefined
      deep-|MM|-dip seeds; the true P is still evaluated at those angles and
      the argmax taken. The deep-dip triage (Bernstein gate / densified rescan
      / exact-root fallback) runs in BOTH arms -- only the polish differs.

MONKEYPATCH MECHANISM (verified 2026-07-18, this session): the batched
floating_offsets catalog path (multiband.py multiband_power_spectra_batched)
calls pdg.scan_polish_from_coefs WITHOUT passing n_newton, so the def-time
default ``n_newton=_SCAN_NEWTON_STEPS`` applies. Python binds keyword defaults
at function definition, so the task-suggested ``core._SCAN_NEWTON_STEPS = 0``
alone is a SILENT NO-OP on this path (measured: max|dP| = 0.0 exactly).
Mechanism used instead: rebind ``core.scan_polish_from_coefs`` to a wrapper
forcing ``n_newton=0`` -- multiband resolves ``pdg.scan_polish_from_coefs`` at
call time (``pdg`` IS the core module), so the wrapper takes effect there and
in the deferred-row reference path (scan_polish_YM_MM). We ALSO set
``core._SCAN_NEWTON_STEPS = 0`` for the call sites that read the module
attribute at call time (multiband shared_phase paths; inert for
floating_offsets, kept for completeness). Effectiveness is asserted twice:
a fail-fast self-test on source 0 before the campaign, and a campaign-level
assert that the a/b spectra differed somewhere (per-source max|dP| recorded).

Outputs per source: P_rec + lib.score for both arms (paired comparison), the
argmax-bin agreement, and the distribution of the per-frequency power change
|P_a - P_b| over the grid (max/mean/median/p99 + the value at arm (a)'s
argmax). Incremental per-source output (<out>.part.jsonl, --resume),
finalized to <out>.json. Explicit search bands only (no nyquist_factor).
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

import ftperiodogram.core as core

warnings.filterwarnings('ignore')

# the RRab production detection arm (B1 staging choice; b3's config)
SUBTYPE = 'RRab'
DETECTION_H = 4
VOCAB_K = 4
SEARCH_BAND = {'RRab': (0.9, 3.4), 'RRc': (1.8, 6.5)}
SPLIT_SEED = 0                # b1/b3 production held-out split + medoid seed


class no_newton_polish(object):
    """Context manager forcing n_newton=0 through the batched multiband path
    (see the module docstring for why patching _SCAN_NEWTON_STEPS alone does
    NOT work: it is a def-time keyword default, not a call-time read)."""

    def __enter__(self):
        self._orig_fn = core.scan_polish_from_coefs
        self._orig_steps = core._SCAN_NEWTON_STEPS
        orig = self._orig_fn

        def _scan_no_polish(*args, **kwargs):
            kwargs['n_newton'] = 0
            return orig(*args, **kwargs)

        core.scan_polish_from_coefs = _scan_no_polish
        core._SCAN_NEWTON_STEPS = 0
        return self

    def __exit__(self, *exc):
        core.scan_polish_from_coefs = self._orig_fn
        core._SCAN_NEWTON_STEPS = self._orig_steps
        return False


def build_source(i, seed0, truth_order, truth_stars, n_per_band, mean_mag,
                 baseline_days, season_length_days):
    """Identical construction (and (seed0, i) keying) to b3.build_source."""
    bank = lib.load_bank(SUBTYPE, 8)
    star = truth_stars[truth_order[i % len(truth_stars)]]
    truth = lib.shared_shape_template(bank, star)
    cad = lib.ps1_cadence(n_per_band, baseline_days=baseline_days,
                          season_length_days=season_length_days,
                          seed=(seed0 * 100003 + i) % (2 ** 32))
    src = lib.inject(truth, SUBTYPE, cad, seed=(seed0 * 7919 + i) % (2 ** 32),
                     mean_mag=mean_mag)
    src['star'] = star
    return src


_G = {}


def _init_worker(vocab, band, oversample, seed0, truth_order, truth_stars,
                 n_per_band, mean_mag, baseline_days, season_length_days):
    _G.update(vocab=vocab, band=band, oversample=oversample, seed0=seed0,
              truth_order=truth_order, truth_stars=truth_stars,
              n_per_band=n_per_band, mean_mag=mean_mag,
              baseline_days=baseline_days,
              season_length_days=season_length_days)


def _score_source(i):
    src = build_source(i, _G['seed0'], _G['truth_order'], _G['truth_stars'],
                       _G['n_per_band'], _G['mean_mag'], _G['baseline_days'],
                       _G['season_length_days'])
    f_lo, f_hi = _G['band']
    grid = lib.converged_grid(f_lo, f_hi, DETECTION_H, src['baseline'],
                              oversample=_G['oversample'])

    t0 = time.time()
    P_a, pw_a = lib.ftp_best_period(_G['vocab'], src, grid)   # (a) production
    t_a = time.time() - t0
    t0 = time.time()
    with no_newton_polish():                                  # (b) no polish
        P_b, pw_b = lib.ftp_best_period(_G['vocab'], src, grid)
    t_b = time.time() - t0

    i_a = int(np.argmax(pw_a))
    i_b = int(np.argmax(pw_b))
    d = np.abs(pw_a - pw_b)
    out = {'i': int(i), 'star': src['star'], 'P_true': src['P_true'],
           'p2p_g': src['p2p_g'], 'baseline': src['baseline'],
           'N': int(src['t'].size), 'nfreq': int(grid.size),
           'polish': {'P_rec': P_a, 'pow_max': float(pw_a[i_a]),
                      **lib.score(P_a, src['P_true'], src['baseline'])},
           'nopolish': {'P_rec': P_b, 'pow_max': float(pw_b[i_b]),
                        **lib.score(P_b, src['P_true'], src['baseline'])},
           'argmax_same_bin': bool(i_a == i_b),
           'dpow': {'max': float(d.max()), 'mean': float(d.mean()),
                    'p50': float(np.median(d)),
                    'p99': float(np.percentile(d, 99)),
                    'at_argmax_a': float(d[i_a])},
           'wall_s': {'polish': t_a, 'nopolish': t_b}}
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


def _self_test_patch(vocab, band, oversample, seed0, truth_order, truth_stars,
                     args):
    """Fail fast if the monkeypatch does not bite: the a/b spectra of source 0
    on a truncated grid must differ somewhere."""
    src = build_source(0, seed0, truth_order, truth_stars, args.n_per_band,
                       args.mean_mag, args.baseline_days,
                       args.season_length_days)
    f_lo, f_hi = band
    grid = lib.converged_grid(f_lo, f_hi, DETECTION_H, src['baseline'],
                              oversample=oversample)[:4096]
    _, pw_a = lib.ftp_best_period(vocab, src, grid)
    with no_newton_polish():
        _, pw_b = lib.ftp_best_period(vocab, src, grid)
    dmax = float(np.max(np.abs(pw_a - pw_b)))
    assert dmax > 0.0, ("monkeypatch self-test FAILED: n_newton=0 arm is "
                        "bit-identical to production -- the patch did not "
                        "take effect")
    print("monkeypatch self-test OK: max|dP| = %.3e over %d freqs (source 0)"
          % (dmax, grid.size))


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
    band = SEARCH_BAND[SUBTYPE]
    vocab_stars, truth_stars = lib.split_stars(SUBTYPE, seed=SPLIT_SEED)
    vocab = lib.detection_vocab(SUBTYPE, DETECTION_H, K=VOCAB_K,
                                seed=SPLIT_SEED, stars=vocab_stars)
    rng = np.random.RandomState(args.seed0 % (2 ** 32))
    truth_order = rng.permutation(len(truth_stars))

    config = dict(subtype=SUBTYPE, H=DETECTION_H, K=VOCAB_K,
                  n_sources=args.n_sources, n_per_band=args.n_per_band,
                  mean_mag=args.mean_mag, oversample=args.oversample,
                  baseline_days=args.baseline_days,
                  season_length_days=args.season_length_days,
                  seed0=args.seed0, split_seed=SPLIT_SEED, band=list(band),
                  mechanism='wrapper forcing n_newton=0 on '
                            'core.scan_polish_from_coefs (+ attr patch); '
                            'attr patch alone is a def-time-default no-op',
                  heldout=dict(n_vocab_stars=len(vocab_stars),
                               n_truth_stars=len(truth_stars)))

    _self_test_patch(vocab, band, args.oversample, args.seed0, truth_order,
                     truth_stars, args)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output',
        'b7_polish_free.json')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    part_path = out_path + '.part.jsonl'
    done = load_part(part_path, config, args.resume)
    todo = [i for i in range(args.n_sources) if i not in done]

    initargs = (vocab, band, args.oversample, args.seed0, truth_order,
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

    srcs = [done[i] for i in sorted(done)]
    # campaign-level patch-effectiveness assert (covers the pool workers, whose
    # patched core module is their own spawn-fresh copy)
    assert any(s['dpow']['max'] > 0.0 for s in srcs), \
        "no source shows any a/b spectrum difference -- patch ineffective?"

    res = dict(config=config, sources=srcs)
    with open(out_path, 'w') as fh:
        json.dump(res, fh)
    os.remove(part_path)
    print("wrote", out_path)
    return res


def summarize(res):
    cfg = res['config']
    srcs = res['sources']
    n = len(srcs)
    lines = ["subtype=%s  H=%d  K=%d  n=%d  N/band=%s  mag=%.1f  os=%.1f  "
             "seed0=%d"
             % (cfg['subtype'], cfg['H'], cfg['K'], n, cfg['n_per_band'],
                cfg['mean_mag'], cfg['oversample'], cfg['seed0'])]
    for crit in ('frac_recovered', 'frac_exact', 'phase_recovered'):
        a = np.array([s['polish'][crit] for s in srcs], bool)
        b = np.array([s['nopolish'][crit] for s in srcs], bool)
        delta = np.mean(a) - np.mean(b)   # polish minus no-polish
        disagree = int(np.sum(a != b))
        if disagree:
            se = np.sqrt(disagree) / n
            lines.append("  %-15s: polish %.3f  no-polish %.3f  paired delta "
                         "= %+.3f +/- %.3f (disagree %d/%d)"
                         % (crit, np.mean(a), np.mean(b), delta, se,
                            disagree, n))
        else:
            lines.append("  %-15s: polish %.3f  no-polish %.3f  paired delta "
                         "= 0 (0 disagreements; 95%% upper bound %.3f)"
                         % (crit, np.mean(a), np.mean(b), 3.0 / n))
    same = int(np.sum([s['argmax_same_bin'] for s in srcs]))
    lines.append("  argmax grid bin identical: %d/%d (%.1f%%)"
                 % (same, n, 100.0 * same / n))
    for key, label in (('max', 'per-source max|dP| over grid'),
                       ('p99', 'per-source p99|dP|'),
                       ('at_argmax_a', '|dP| at arm-(a) argmax')):
        v = np.array([s['dpow'][key] for s in srcs])
        lines.append("  %-28s: median %.2e  p90 %.2e  max %.2e"
                     % (label, np.median(v), np.percentile(v, 90), v.max()))
    ta = np.array([s['wall_s']['polish'] for s in srcs])
    tb = np.array([s['wall_s']['nopolish'] for s in srcs])
    lines.append("  wall/src: polish %.2f s, no-polish %.2f s (ratio %.2f)"
                 % (ta.mean(), tb.mean(), ta.mean() / tb.mean()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
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
