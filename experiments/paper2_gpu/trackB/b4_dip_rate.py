"""B4 -- deep-|MM'|-dip / deferral rate on REAL cadences (HANDOFF S4.4).

Measures, per cadence x H, the fraction of frequencies whose band-combined
|MM'| circle conditioning r = min|MM|/max|MM| falls below (a) the batched-path
deferral threshold 3e-3 (multiband._BATCHED_DEFER_RTOL -> per-freq reference
on CPU; host round-trip / Aberth on GPU) and (b) the scan exact-root threshold
0.15 (core._SCAN_EXACT_RTOL). This is GPU_FEASIBILITY risk #2: only synthetic
was measured (0%); real phase-clustered cadences could differ. Sets whether
the Aberth rooter is load-bearing for the GPU port.

|MM| depends only on (t, bands, weights, template, grid) -- no injection.
Arms: real ZTF g/r cadences (99 cached oids), thinned to N/band in {8, 20};
synthetic PS1 TTI-pair griz cadences (64 seeds, N=8/band). H in {2, 4, 8},
RRab search band, os=3 H-matched grids, r-band RRab medoid template per H.
"""
import argparse
import json
import os
import sys
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

from ftperiodogram import core as pdg
from ftperiodogram import multiband as mb

warnings.filterwarnings('ignore')

BAND = (0.9, 3.4)
H_SET = (2, 4, 8)
CHUNK = 4096

_TMPL = None


def _init(tmpls):
    global _TMPL
    _TMPL = tmpls


def dip_fracs(t, y, bands, dy, H, template):
    """(frac r<3e-3, frac r<0.15, min r, nfreq) over the H-matched grid."""
    T = float(np.max(t) - np.min(t))
    freqs = lib.converged_grid(BAND[0], BAND[1], H, T, oversample=3.0)
    tdict = {b: template for b in np.unique(bands)}
    transforms, stats, freqs = mb._prepare_band_transforms(
        t, y, bands, dy, freqs, H, 'floating_offsets', None, fast=True)
    n_ang = max(pdg._SCAN_MIN_ANGLES, pdg._SCAN_ANGLES_PER_H * H)
    n_def = n_ex = 0
    r_min = np.inf
    nf = len(freqs)
    for i0 in range(0, nf, CHUNK):
        i1 = min(i0 + CHUNK, nf)
        _YM, MM, _AC, _ACb = mb._combine_stacked_shared_amp(
            tdict, transforms, stats, i0, i1, freqs, 'floating_offsets')
        absM = np.abs(pdg._eval_polys_on_circle(MM, n_ang))
        r = absM.min(axis=1) / absM.max(axis=1)
        n_def += int((r < mb._BATCHED_DEFER_RTOL).sum())
        n_ex += int((r < pdg._SCAN_EXACT_RTOL).sum())
        r_min = min(r_min, float(r.min()))
    return n_def / nf, n_ex / nf, r_min, nf


def _work(job):
    kind, ident, t, bandarr, dy = job
    rng = np.random.RandomState(abs(hash(ident)) % (2**31))
    y = 21.0 + 0.05 * rng.randn(t.size)     # y is irrelevant to MM; non-flat
    out = {'arm': kind, 'id': str(ident), 'N': int(t.size)}
    for H in H_SET:
        fd, fx, rmin, nf = dip_fracs(t, y, bandarr, dy, H, _TMPL[H])
        out['H%d' % H] = {'frac_defer': fd, 'frac_exact': fx,
                          'r_min': rmin, 'nfreq': nf}
    return out


def build_jobs(n_ps1=64):
    jobs = []
    for nthin in (8, 20):
        for oid in lib.ztf_cadence_ids():
            try:
                cad = lib.load_ztf_cadence(oid, max_epochs_per_band=nthin)
            except Exception:
                continue
            s = cad.sample()
            dy = np.asarray(s.mag_err_model(np.full(s.t.size, 21.0)), float)
            jobs.append(('ztf_n%d' % nthin, '%s_n%d' % (oid, nthin),
                         np.asarray(s.t, float), np.asarray(s.bands), dy))
    for seed in range(n_ps1):
        cad = lib.ps1_cadence(8, baseline_days=1600.0, seed=90000 + seed)
        s = cad.sample()
        dy = np.asarray(s.mag_err_model(np.full(s.t.size, 21.0)), float)
        jobs.append(('ps1_pair_n8', 'ps1_%d' % seed,
                     np.asarray(s.t, float), np.asarray(s.bands), dy))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nproc', type=int, default=8)
    ap.add_argument('--out', default='output/b4_dip_rate.json')
    args = ap.parse_args()

    vocab_stars, _ = lib.split_stars('RRab', seed=0)
    tmpls = {H: lib.detection_vocab('RRab', H, K=4, seed=0,
                                    stars=vocab_stars)[0] for H in H_SET}
    jobs = build_jobs()
    print('jobs:', len(jobs))
    with Pool(args.nproc, initializer=_init, initargs=(tmpls,)) as pool:
        rows = pool.map(_work, jobs)

    print('%-12s %s' % ('arm', ' | '.join(
        'H%d: defer%% / exact%% / worst-r' % H for H in H_SET)))
    arms = sorted({r['arm'] for r in rows})
    for arm in arms:
        sub = [r for r in rows if r['arm'] == arm]
        cells = []
        for H in H_SET:
            fd = [r['H%d' % H]['frac_defer'] for r in sub]
            fx = [r['H%d' % H]['frac_exact'] for r in sub]
            rm = min(r['H%d' % H]['r_min'] for r in sub)
            cells.append('%.4f/%.4f/%.1e (max %.4f/%.4f)'
                         % (np.mean(fd), np.mean(fx), rm, max(fd), max(fx)))
        print('%-12s %s  [n=%d]' % (arm, ' | '.join(cells), len(sub)))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'rows': rows}, fh)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
