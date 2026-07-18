"""B3 -- vocabulary size (K) vs recall on the HELD-OUT split.

Confirms (or refutes) the rerun_202606 flat-in-K result on the audit-corrected
harness: vocab medoids come from the vocab half of the BV-2025 stars
(lib.split_stars), truths are cycled from the OTHER half, so K genuinely
measures shape-coverage generalization rather than self-recovery.

Design:
  * one config per subtype -- the production detection arms: RRab at H=4,
    RRc at H=2 (B1's staging choice);
  * PAIRED across K: each injected source (PS1 TTI-pair griz cadence, N=8/band,
    mag 21.0, drawn p2p) is scanned by every K in {1, 2, 4, 8} on the SAME
    per-source converged grid (df = 1/(os*H*T), grid depends on H only), so the
    K-deltas carry the tight McNemar CI;
  * absolute rates carry the usual binomial SE.

Emits per-source booleans for every K (alias-credited frac_recovered plus the
strict frac_exact and phase_* columns, exactly lib.score / b1). Incremental
per-source output: records append to <out>.part.jsonl as they finish (killed
runs resume with --resume), finalized to <out>.json at the end.

Explicit search bands only (no nyquist_factor anywhere).
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

warnings.filterwarnings('ignore')

# per-subtype search band (covers the Bailey box + main P/2, 2P aliases; same
# as b1) and the production detection order H (B1's staging measurement)
SEARCH_BAND = {'RRab': (0.9, 3.4), 'RRc': (1.8, 6.5)}
DETECTION_H = {'RRab': 4, 'RRc': 2}
K_GRID = (1, 2, 4, 8)
K_REF = 8                     # paired deltas are quoted against the largest K
# held-out split + medoid seed fixed at 0 = the b1 production run's split, so
# b3's K=4 arm uses bit-identical vocab/truth halves to B1_RESULT.md
SPLIT_SEED = 0


def build_source(subtype, i, seed0, truth_order, truth_stars, n_per_band,
                 mean_mag, baseline_days, season_length_days):
    """Deterministic source #i: truth shapes cycled over the HELD-OUT truth
    stars, each on its own cadence realisation + noise; p2p drawn per-source
    (mirrors b1.build_population, keyed by (seed0, i) so a resumed run
    rebuilds the identical source)."""
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


_G = {}


def _init_worker(vocabs, subtype, band, oversample, seed0, truth_order,
                 truth_stars, n_per_band, mean_mag, baseline_days,
                 season_length_days):
    _G.update(vocabs=vocabs, subtype=subtype, band=band, oversample=oversample,
              seed0=seed0, truth_order=truth_order, truth_stars=truth_stars,
              n_per_band=n_per_band, mean_mag=mean_mag,
              baseline_days=baseline_days,
              season_length_days=season_length_days)


def _score_source(i):
    src = build_source(_G['subtype'], i, _G['seed0'], _G['truth_order'],
                       _G['truth_stars'], _G['n_per_band'], _G['mean_mag'],
                       _G['baseline_days'], _G['season_length_days'])
    f_lo, f_hi = _G['band']
    H = DETECTION_H[_G['subtype']]
    # grid depends on H only -> one grid per source, shared by every K (PAIRED)
    grid = lib.converged_grid(f_lo, f_hi, H, src['baseline'],
                              oversample=_G['oversample'])
    out = {'i': int(i), 'star': src['star'], 'P_true': src['P_true'],
           'p2p_g': src['p2p_g'], 'baseline': src['baseline'],
           'N': int(src['t'].size), 'nfreq': int(grid.size)}
    for K in K_GRID:
        Prec, _ = lib.ftp_best_period(_G['vocabs'][K], src, grid)
        s = lib.score(Prec, src['P_true'], src['baseline'])
        out['K%d' % K] = {'P_rec': Prec, **s}
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


# ----------------------------------------------------------------------
# Incremental .part.jsonl store (killed runs resume with --resume)
# ----------------------------------------------------------------------
def load_part(part_path, config, resume):
    """Return {i: record} already computed. Without --resume any stale part
    file is discarded; with it, the stored config must match."""
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
    vocab_stars, truth_stars = lib.split_stars(subtype, seed=SPLIT_SEED)
    vocabs = {K: lib.detection_vocab(subtype, H, K=K, seed=SPLIT_SEED,
                                     stars=vocab_stars) for K in K_GRID}
    rng = np.random.RandomState(args.seed0 % (2 ** 32))
    truth_order = rng.permutation(len(truth_stars))

    config = dict(subtype=subtype, H=H, n_sources=args.n_sources,
                  n_per_band=args.n_per_band, mean_mag=args.mean_mag,
                  oversample=args.oversample, baseline_days=args.baseline_days,
                  season_length_days=args.season_length_days,
                  seed0=args.seed0, split_seed=SPLIT_SEED, band=list(band),
                  K_grid=list(K_GRID), K_ref=K_REF,
                  vocab_sizes={('K%d' % K): len(vocabs[K]) for K in K_GRID},
                  heldout=dict(n_vocab_stars=len(vocab_stars),
                               n_truth_stars=len(truth_stars)))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'output',
        'b3_vocab_%s.json' % subtype.lower())
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    part_path = out_path + '.part.jsonl'
    done = load_part(part_path, config, args.resume)
    todo = [i for i in range(args.n_sources) if i not in done]

    initargs = (vocabs, subtype, band, args.oversample, args.seed0,
                truth_order, truth_stars, args.n_per_band, args.mean_mag,
                args.baseline_days, args.season_length_days)
    t0 = time.time()
    with open(part_path, 'a') as fh:
        if not done:
            fh.write(json.dumps({'_config': config}) + '\n')
            fh.flush()
        if todo:
            with Pool(min(args.nproc, len(todo)), initializer=_init_worker,
                      initargs=initargs) as pool:
                for k, rec in enumerate(pool.imap_unordered(_score_source,
                                                            todo)):
                    fh.write(json.dumps(rec) + '\n')
                    fh.flush()
                    done[rec['i']] = rec
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

    res = dict(config=config,
               sources=[done[i] for i in sorted(done)])
    with open(out_path, 'w') as fh:
        json.dump(res, fh)
    os.remove(part_path)
    print("wrote", out_path)
    return res


def _rate_se(p, n):
    """Binomial SE string; p==0 or 1 quotes the rule-of-three 95% bound as
    (<=x) instead of a misleading +/-0 (b1 convention)."""
    if n and 0.0 < p < 1.0:
        return "%.3f+/-%.3f" % (p, np.sqrt(p * (1.0 - p) / n))
    return "%.3f (<=%.3f)" % (p, 3.0 / max(n, 1))


def summarize(res):
    cfg = res['config']
    srcs = res['sources']
    n = len(srcs)
    lines = ["subtype=%s  H=%d  n=%d  N/band=%s  mag=%.1f  os=%.1f  "
             "vocab_pool=%d  seed0=%d"
             % (cfg['subtype'], cfg['H'], n, cfg['n_per_band'],
                cfg['mean_mag'], cfg['oversample'],
                cfg['heldout']['n_vocab_stars'], cfg['seed0'])]
    lines.append("  K : frac_rec         frac_exact  phase_rec   (n_tmpl)")
    for K in K_GRID:
        fr = np.mean([s['K%d' % K]['frac_recovered'] for s in srcs])
        fe = np.mean([s['K%d' % K]['frac_exact'] for s in srcs])
        pr = np.mean([s['K%d' % K]['phase_recovered'] for s in srcs])
        lines.append("  %d : %-16s %.3f       %.3f      (%d)"
                     % (K, _rate_se(fr, n), fe, pr,
                        cfg['vocab_sizes']['K%d' % K]))
    # paired deltas vs the largest K (McNemar SE; rule of three at 0 disagree)
    b = np.array([s['K%d' % K_REF]['frac_recovered'] for s in srcs], bool)
    for K in K_GRID:
        if K == K_REF:
            continue
        a = np.array([s['K%d' % K]['frac_recovered'] for s in srcs], bool)
        loss = np.mean(b) - np.mean(a)   # K_REF minus K (positive => K worse)
        disagree = int(np.sum(a != b))
        if disagree:
            se = np.sqrt(disagree) / n
            lines.append("  paired K%d vs K%d: rate loss = %+.3f +/- %.3f "
                         "(disagree %d/%d)" % (K, K_REF, loss, se, disagree, n))
        else:
            lines.append("  paired K%d vs K%d: rate loss = 0 (0 disagreements;"
                         " 95%% upper bound %.3f)" % (K, K_REF, 3.0 / n))
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
