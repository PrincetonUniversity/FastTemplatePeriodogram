"""Golden parity capture for the SAFE CPU stack.

Usage:
  .venv/bin/python golden.py capture <tag>    # save powers under output/golden_<tag>.npz
  .venv/bin/python golden.py compare <tagA> <tagB>   # max abs/rel diffs per fixture

Captures, per fixture:
  - multiband fixtures: batched scan powers (fast=True NFFT and fast=False
    direct), catalog max-over-vocab powers, and the per-frequency 'eigvals'
    reference on a 200-frequency subsample.
  - single-band fixtures: method='scan' powers (fast NFFT), method='eigvals'
    reference on a 200-frequency subsample, and fast=False scan powers.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import fixtures  # noqa: E402
from ftperiodogram.core import template_periodogram  # noqa: E402
from ftperiodogram.multiband import (  # noqa: E402
    FastMultibandTemplatePeriodogram, build_template_set,
    multiband_template_periodogram)


def _mb_powers(fx, fast, catalog):
    ftp = FastMultibandTemplatePeriodogram(
        templates=fx['vocab'] if catalog else fx['vocab'][0],
        mode='floating_offsets')
    ftp.fit(fx['t'], fx['y'], fx['bands'], fx['dy'])
    return ftp.power(fx['freqs'], save_best_model=False, fast=fast)


def _mb_ref(fx, sub):
    tdict = build_template_set(fx['vocab'][0], fx['bands'])
    pw, _ = multiband_template_periodogram(
        fx['t'], fx['y'], fx['bands'], tdict, fx['freqs'][sub],
        dy=fx['dy'], mode='floating_offsets', fast=False, method='eigvals')
    return np.asarray(pw)


def _sb_powers(fx, method, fast, sub=None):
    tm = fx['template']
    fr = fx['freqs'] if sub is None else fx['freqs'][sub]
    pw, _ = template_periodogram(fx['t'], fx['y'], fx['dy'], tm.c_n, tm.s_n,
                                 fr, fast=fast, method=method)
    return np.asarray(pw)


def capture(tag):
    out = {}
    t0 = time.time()
    for fx in fixtures.all_fixtures():
        lab = fx['label']
        sub = np.arange(0, len(fx['freqs']), len(fx['freqs']) // 200)
        if 'vocab' in fx:
            out[lab + ':scan_nfft'] = _mb_powers(fx, fast=True, catalog=False)
            out[lab + ':scan_direct'] = _mb_powers(fx, fast=False,
                                                   catalog=False)
            out[lab + ':scan_catalog'] = _mb_powers(fx, fast=True,
                                                    catalog=True)
            out[lab + ':eig_ref'] = _mb_ref(fx, sub)
        else:
            out[lab + ':scan_nfft'] = _sb_powers(fx, 'scan', True)
            out[lab + ':scan_direct'] = _sb_powers(fx, 'scan', False)
            out[lab + ':eig_ref'] = _sb_powers(fx, 'eigvals', False, sub=sub)
        out[lab + ':sub'] = sub
        print('%-16s captured  (%.1fs elapsed)' % (lab, time.time() - t0))
    np.savez(_HERE / 'output' / ('golden_%s.npz' % tag), **out)
    print('saved golden_%s.npz' % tag)


def compare(tag_a, tag_b):
    A = np.load(_HERE / 'output' / ('golden_%s.npz' % tag_a))
    B = np.load(_HERE / 'output' / ('golden_%s.npz' % tag_b))
    report = {}
    for key in sorted(A.files):
        if key.endswith(':sub'):
            continue
        a, b = A[key], B[key]
        if a.shape != b.shape:
            report[key] = 'SHAPE %s vs %s' % (a.shape, b.shape)
            continue
        d = np.abs(a - b)
        scale = np.maximum(np.abs(a), 1e-300)
        report[key] = dict(max_abs=float(d.max()),
                           max_rel=float((d / scale).max()),
                           argmax_moved=bool(int(np.argmax(a))
                                             != int(np.argmax(b))),
                           n_diff=int(np.sum(d > 0)))
    print(json.dumps(report, indent=1))
    return report


if __name__ == '__main__':
    if sys.argv[1] == 'capture':
        capture(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
