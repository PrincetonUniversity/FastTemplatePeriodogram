"""Before/after benchmark for the SAFE CPU stack.

Usage: .venv/bin/python bench.py <tag> [--reps 5] [--nfreq 8000]
Writes output/bench_<tag>.json. Pin threads:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

Rows (all wall-clock, median of reps):
  sb_scan_h{2,4,8}      single-band template_periodogram method='scan', NFFT
  mb_batched_h{2,4,8}   multiband batched floating_offsets, K=4 griz, 1 template
  mb_catalog_h{4,8}     same, K_vocab=4 catalog (hoist/cache win target)
  mb_direct_h4          multiband batched fast=False (direct sums; trig win)
  mb_sparse2_h4         sparse 2-band ZTF-like (deep-dip fallback win target)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import fixtures  # noqa: E402
from ftperiodogram.core import template_periodogram  # noqa: E402
from ftperiodogram.multiband import FastMultibandTemplatePeriodogram  # noqa: E402


def _med(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--nfreq', type=int, default=8000)
    args = ap.parse_args()

    rows = {}

    for H in (2, 4, 8):
        fx = fixtures.sb_fixture(H=H, nfreq=args.nfreq)
        tm = fx['template']

        def run(fx=fx, tm=tm):
            template_periodogram(fx['t'], fx['y'], fx['dy'], tm.c_n, tm.s_n,
                                 fx['freqs'], fast=True, method='scan')
        rows['sb_scan_h%d' % H] = _med(run, args.reps)
        print('sb_scan_h%d: %.3fs' % (H, rows['sb_scan_h%d' % H]))

    for H, sub in ((2, 'RRc'), (4, 'RRab'), (8, 'RRab')):
        fx = fixtures.mb_fixture(sub, H=H, nfreq=args.nfreq)
        ftp = FastMultibandTemplatePeriodogram(templates=fx['vocab'][0],
                                               mode='floating_offsets')
        ftp.fit(fx['t'], fx['y'], fx['bands'], fx['dy'])

        def run(ftp=ftp, fx=fx):
            ftp.power(fx['freqs'], save_best_model=False, fast=True)
        rows['mb_batched_h%d' % H] = _med(run, args.reps)
        print('mb_batched_h%d: %.3fs' % (H, rows['mb_batched_h%d' % H]))

        cat = FastMultibandTemplatePeriodogram(templates=fx['vocab'],
                                               mode='floating_offsets')
        cat.fit(fx['t'], fx['y'], fx['bands'], fx['dy'])

        def runc(cat=cat, fx=fx):
            cat.power(fx['freqs'], save_best_model=False, fast=True)
        rows['mb_catalog_h%d' % H] = _med(runc, args.reps)
        print('mb_catalog_h%d: %.3fs' % (H, rows['mb_catalog_h%d' % H]))

        if H == 4:
            def rund(ftp=ftp, fx=fx):
                ftp.power(fx['freqs'], save_best_model=False, fast=False)
            rows['mb_direct_h4'] = _med(rund, args.reps)
            print('mb_direct_h4: %.3fs' % rows['mb_direct_h4'])

    fx = fixtures.mb_sparse2_fixture(H=4, nfreq=args.nfreq)
    ftp = FastMultibandTemplatePeriodogram(templates=fx['vocab'][0],
                                           mode='floating_offsets')
    ftp.fit(fx['t'], fx['y'], fx['bands'], fx['dy'])

    def runs(ftp=ftp, fx=fx):
        ftp.power(fx['freqs'], save_best_model=False, fast=True)
    rows['mb_sparse2_h4'] = _med(runs, args.reps)
    print('mb_sparse2_h4: %.3fs' % rows['mb_sparse2_h4'])

    rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                         capture_output=True, text=True,
                         cwd=_HERE).stdout.strip()
    # provenance: HEAD alone cannot attest a files-only checkout (the
    # 'before' arm restores an old ftperiodogram/ without moving HEAD),
    # so also record a content hash of the measured package source
    import hashlib
    import ftperiodogram
    h = hashlib.sha256()
    for p in sorted(Path(ftperiodogram.__file__).parent.glob('*.py')):
        h.update(p.read_bytes())
    out = dict(tag=args.tag, rev=rev, pkg_sha=h.hexdigest()[:12],
               nfreq=args.nfreq, reps=args.reps, rows=rows)
    path = _HERE / 'output' / ('bench_%s.json' % args.tag)
    path.write_text(json.dumps(out, indent=1))
    print('saved', path.name)


if __name__ == '__main__':
    main()
