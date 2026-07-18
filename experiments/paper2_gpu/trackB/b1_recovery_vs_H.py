"""B1 -- recovery vs H at each H's converged grid, RRab & RRc separately.

The biggest cost lever (HANDOFF S4.1): the detection scan runs at low H (H~4
RRab / H~2 RRc); refine at H=8 only at candidates. This confirms the low-H
detection loses <= a few % recovery vs H=8, ON REAL TEMPLATE SHAPES.

Design:
  * inject from the full BV-2025 typed bank (shared griz shape + band amps +
    per-band offsets + noise) onto a synthetic PS1-3pi-like griz cadence;
  * PAIRED over H: each source is scored at every H on that H's own converged
    grid (df = 1/(os*H*T)), so H=8 is never grid-starved -- the flaw the b8
    headline had. One grid built for H=8 is converged for all smaller H, but we
    still build per-H grids (cheaper for low H) and check H=8 vs os*2 separately;
  * report the full recovery-vs-H curve and the PAIRED H_low-vs-H8 loss (tight CI
    because it is the same population).

Emits per-source booleans for every H so the paired loss + CI are computed
downstream (b1_report.py). Multiprocessing over sources.
"""
import argparse
import json
import os
import sys
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

# per-subtype search band (covers the Bailey box + main P/2, 2P aliases)
SEARCH_BAND = {'RRab': (0.9, 3.4), 'RRc': (1.8, 6.5)}
H_GRID = (1, 2, 3, 4, 5, 6, 8)


def build_population(subtype, n_sources, n_per_band, mean_mag,
                     baseline_days, season_length_days, seed, truth_stars):
    """Deterministic fixed population: truth shapes cycled over the HELD-OUT
    truth stars (audit fix), each on its own cadence realisation + noise;
    p2p amplitude drawn per-source from the subtype range."""
    bank = lib.load_bank(subtype, 8)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(truth_stars))
    srcs = []
    for i in range(n_sources):
        star = truth_stars[order[i % len(truth_stars)]]
        truth = lib.shared_shape_template(bank, star)
        cad = lib.ps1_cadence(n_per_band, baseline_days=baseline_days,
                              season_length_days=season_length_days,
                              seed=seed * 100003 + i)
        src = lib.inject(truth, subtype, cad, seed=seed * 7919 + i,
                         mean_mag=mean_mag)
        src['star'] = star
        srcs.append(src)
    return srcs


_VOCABS = None
_BAND = None
_OS = None


def _init_worker(vocabs, band, oversample):
    global _VOCABS, _BAND, _OS
    _VOCABS, _BAND, _OS = vocabs, band, oversample


def _score_source(src):
    out = {'P_true': src['P_true'], 'baseline': src['baseline'],
           'N': int(src['t'].size), 'star': src['star']}
    f_lo, f_hi = _BAND
    for H in H_GRID:
        grid = lib.converged_grid(f_lo, f_hi, H, src['baseline'], oversample=_OS)
        Prec, _ = lib.ftp_best_period(_VOCABS[H], src, grid)
        s = lib.score(Prec, src['P_true'], src['baseline'])
        out['H%d' % H] = {'P_rec': Prec, 'nfreq': int(grid.size), **s}
    return out


def run(subtype, n_sources, n_per_band, mean_mag, K, oversample,
        baseline_days, season_length_days, seed, nproc):
    band = SEARCH_BAND[subtype]
    vocab_stars, truth_stars = lib.split_stars(subtype, seed=seed)
    vocabs = {H: lib.detection_vocab(subtype, H, K=K, seed=seed,
                                     stars=vocab_stars) for H in H_GRID}
    srcs = build_population(subtype, n_sources, n_per_band, mean_mag,
                            baseline_days, season_length_days, seed,
                            truth_stars)
    with Pool(nproc, initializer=_init_worker,
              initargs=(vocabs, band, oversample)) as pool:
        results = pool.map(_score_source, srcs)
    return dict(
        config=dict(subtype=subtype, n_sources=n_sources, n_per_band=n_per_band,
                    mean_mag=mean_mag, K=K,
                    oversample=oversample, baseline_days=baseline_days,
                    season_length_days=season_length_days, seed=seed,
                    band=band, H_grid=list(H_GRID),
                    heldout=dict(n_vocab_stars=len(vocab_stars),
                                 n_truth_stars=len(truth_stars))),
        sources=results)


def summarize(res):
    srcs = res['sources']
    n = len(srcs)
    lines = ["subtype=%s  n=%d  N/band=%s  mag=%.1f  p2p_g=drawn  K=%d  os=%.1f"
             % (res['config']['subtype'], n, res['config']['n_per_band'],
                res['config']['mean_mag'],
                res['config']['K'], res['config']['oversample'])]
    lines.append("  H : frac_rec  frac_exact  phase_rec   (nfreq)")
    rec = {}
    for H in H_GRID:
        fr = np.mean([s['H%d' % H]['frac_recovered'] for s in srcs])
        fe = np.mean([s['H%d' % H]['frac_exact'] for s in srcs])
        pr = np.mean([s['H%d' % H]['phase_recovered'] for s in srcs])
        nf = srcs[0]['H%d' % H]['nfreq']
        rec[H] = fr
        lines.append("  %d : %.3f      %.3f       %.3f      (%d)" % (H, fr, fe, pr, nf))
    # paired loss vs H=8
    for Hl in (2, 4):
        a = np.array([s['H%d' % Hl]['frac_recovered'] for s in srcs], bool)
        b = np.array([s['H8']['frac_recovered'] for s in srcs], bool)
        loss = np.mean(b) - np.mean(a)  # H8 minus Hl (positive => Hl worse)
        disagree = int(np.sum(a != b))
        # McNemar paired SE; disagree=0 reports a one-sided 95% bound (rule of
        # three) instead of a misleading +/-0 (audit minor fix)
        if disagree:
            se = np.sqrt(disagree) / n
            lines.append("  paired H%d vs H8: rate loss = %+.3f +/- %.3f "
                         "(disagree %d/%d)" % (Hl, loss, se, disagree, n))
        else:
            lines.append("  paired H%d vs H8: rate loss = 0 (0 disagreements; "
                         "95%% upper bound %.3f)" % (Hl, 3.0 / n))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subtype', required=True, choices=['RRab', 'RRc'])
    ap.add_argument('--n-sources', type=int, default=256)
    ap.add_argument('--n-per-band', type=int, default=8)
    ap.add_argument('--mean-mag', type=float, default=21.0)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--oversample', type=float, default=3.0)
    ap.add_argument('--baseline-days', type=float, default=1300.0)
    ap.add_argument('--season-length-days', type=float, default=250.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--nproc', type=int, default=os.cpu_count())
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    res = run(args.subtype, args.n_sources, args.n_per_band, args.mean_mag,
              args.K, args.oversample, args.baseline_days,
              args.season_length_days, args.seed, args.nproc)
    print(summarize(res))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as fh:
            json.dump(res, fh)
        print("wrote", args.out)


if __name__ == '__main__':
    main()
