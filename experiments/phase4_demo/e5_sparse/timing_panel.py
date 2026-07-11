"""WP E5 extra (b): REAL-DATA TIMING PANEL -- ftp_mb (scan default, C4) vs
SlowTemplatePeriodogram (nguesses=10), on a reduced 2000-freq grid.

HONESTY LABEL: this is a TIMING-ONLY grid -- 2000 evenly spaced frequencies
on the explicit [1, 5] cyc/day science range, NOT the production
df <= 0.2/T grid (~53k freqs).  Full-grid Slow is infeasible (that is the
point of the panel).  Per-frequency scan cost is grid-stable (WP C5), so
s/freq measured here extrapolates linearly to the science grid.

Comparability (all recorded in the output json, normalization left to the
memo):
  * ftp_mb : FTPEstimator seam (K=8 PAM vocab, catalog-max over 8
    templates), g + r joint data (2N epochs), floating_offsets, scan
    default per WP C4.
  * slow   : SlowTemplatePeriodogram(template=vocab medoid 0, nguesses=10)
    -- single template, primary band only (N epochs); single-band
    single-template by construction (the historical gold standard).
  So slow does 1/8 the templates on 1/2 the data; divide the ftp_mb wall by
  8 for a per-template view.

Protocol: 3 stars (RRab, RRc, faint science) x N in {15, 40} per band
x 2 methods, min-of-3 reps.  Each rep = one full 2000-freq evaluation timed
with time.perf_counter (model construction + fit included; that is what a
user pays).  BLAS pinned to 1 thread (env set before numpy import).
os.getloadavg() recorded before/after every rep.  Data = the SAME seeded
random per-epoch subsample machinery as the E5 production runs
(run_e5.task_inputs, mode='random', rep=0).  np.random seeded per (cell,
rep) because Slow draws its phase guesses from np.random.

Stars (deterministic rule, from e5_star_subset.json):
  RRab  = E5 smoke RRab  (median-joint-epoch science RRab)
  RRc   = E5 smoke RRc   (median-joint-epoch science RRc)
  faint = faintest science star by r_med_mag

Resumable at (star, N, method, rep) granularity ->
e5_timing_panel.part.json; finalized (min-of-3 per cell + metadata) to
e5_timing_panel.json.  Exit 0 = panel complete; 3 = budget hit, rerun.

Usage::

    timing_panel.py [--budget-s 400] [--probe NFREQ]  (probe: RRab star,
    both methods, both N, 1 rep on NFREQ freqs; prints cost, stores nothing)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse   # noqa: E402
import json       # noqa: E402
import sys        # noqa: E402
import time       # noqa: E402
import zlib       # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E4DIR = os.path.abspath(os.path.join(HERE, '..', 'e4_recovery'))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (REPO, E4DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_e4  # noqa: E402
import run_e5  # noqa: E402

STARS = {                                   # deterministic rule in docstring
    'RRab':  'chen_f686_ZTFJ194248.05+333430.0',
    'RRc':   'chen_f686_ZTFJ193825.40+352053.3',
    'faint': 'chen_f786_ZTFJ092825.79+535045.1',
}
N_LIST = (15, 40)
METHODS = ('ftp_mb', 'slow')
N_REPS = 3
N_FREQ = 2000


def timing_grid(n_freq):
    """Evenly spaced timing-only grid on [F_MIN, F_MAX): df chosen so F_MIN
    is an integer multiple of df (the FTP NFFT path requires it).  n_freq
    must be divisible by (F_MAX - F_MIN) / F_MIN = 4."""
    df = (run_e4.F_MAX - run_e4.F_MIN) / n_freq
    i_min = run_e4.F_MIN / df
    assert abs(i_min - round(i_min)) < 1e-12, 'n_freq must be divisible by 4'
    return df * (round(i_min) + np.arange(n_freq))


FREQS = timing_grid(N_FREQ)                               # timing-only grid
PART = os.path.join(HERE, 'e5_timing_panel.part.json')
FINAL = os.path.join(HERE, 'e5_timing_panel.json')


def rep_key(label, N, method, rep):
    return '%s|N%d|%s|rep%d' % (label, N, method, rep)


def get_inputs(star, method, N):
    """(t, y, bands, dy, n_obs, band_used) via the E5 production subsampler
    (mode='random', rep=0).  slow -> primary band (ftp_1band branch)."""
    m = 'ftp_mb' if method == 'ftp_mb' else 'ftp_1band'
    t, y, bands, dy, _grid, band_used, _meta = \
        run_e5.task_inputs(star, m, N, 0, 'random')
    return t, y, bands, dy, int(t.size), band_used


def time_rep(star, method, N, label, rep, freqs):
    """One timed full-grid evaluation.  Returns a result dict."""
    t, y, bands, dy, n_obs, band_used = get_inputs(star, method, N)
    np.random.seed(zlib.crc32(('E5timing|%s|N%d|%s|rep%d'
                               % (label, N, method, rep)).encode())
                   & 0xffffffff)
    la0 = os.getloadavg()
    if method == 'ftp_mb':
        t0 = time.perf_counter()
        est, config = run_e4.make_estimator('ftp_mb')
        stat = est.power_spectrum(t, y, bands, dy, freqs)
        wall = time.perf_counter() - t0
    else:
        from ftperiodogram.modeler import SlowTemplatePeriodogram
        from ftperiodogram.template import Template
        c, s = run_e4._G['vocab_coeffs'][0]
        config = ('SlowTemplatePeriodogram(vocab medoid 0, nguesses=10), '
                  'primary band')
        t0 = time.perf_counter()
        model = SlowTemplatePeriodogram(template=Template(c_n=c, s_n=s),
                                        nguesses=10).fit(t, y, dy)
        stat = model.power(freqs)
        wall = time.perf_counter() - t0
    la1 = os.getloadavg()
    return {
        'wall_s': wall, 's_per_freq': wall / freqs.size,
        'n_freq': int(freqs.size), 'n_obs': n_obs, 'band_used': band_used,
        'loadavg_before': list(la0), 'loadavg_after': list(la1),
        'config': config, 'best_freq': float(freqs[int(np.nanargmax(stat))]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-s', type=float, default=400.0)
    ap.add_argument('--probe', type=int, default=0,
                    help='probe mode: NFREQ freqs, RRab star, 1 rep each '
                         'cell, nothing stored')
    args = ap.parse_args()
    deadline = time.time() + args.budget_s

    with open(os.path.join(E4DIR, 'star_table.json')) as fh:
        by_uid = {s['uid']: s for s in json.load(fh)['stars']}

    run_e4._init_worker(run_e4.build_vocab_coeffs())

    if args.probe:
        freqs = timing_grid(args.probe)
        for method in METHODS:
            for N in N_LIST:
                r = time_rep(by_uid[STARS['RRab']], method, N,
                             'RRab', 0, freqs)
                print('PROBE %s N=%d: %.2fs for %d freqs = %.3f ms/freq '
                      '(n_obs=%d) -> est %.0fs per %d-freq rep'
                      % (method, N, r['wall_s'], args.probe,
                         1e3 * r['s_per_freq'], r['n_obs'],
                         r['s_per_freq'] * N_FREQ, N_FREQ), flush=True)
        return 0

    done = {}
    if os.path.exists(PART):
        with open(PART) as fh:
            done = json.load(fh)

    pending = [(lab, N, m, rep) for lab in STARS for N in N_LIST
               for m in METHODS for rep in range(N_REPS)
               if rep_key(lab, N, m, rep) not in done]
    print('timing panel: %d/%d reps done, %d pending'
          % (len(done), len(done) + len(pending), len(pending)), flush=True)

    for lab, N, m, rep in pending:
        if time.time() > deadline:
            print('BUDGET_HIT rerun to resume', flush=True)
            return 3
        r = time_rep(by_uid[STARS[lab]], m, N, lab, rep, FREQS)
        done[rep_key(lab, N, m, rep)] = r
        with open(PART + '.tmp', 'w') as fh:
            json.dump(done, fh, indent=1, sort_keys=True)
        os.replace(PART + '.tmp', PART)
        print('  %s wall=%.2fs (%.3f ms/freq) load %.2f->%.2f'
              % (rep_key(lab, N, m, rep), r['wall_s'],
                 1e3 * r['s_per_freq'], r['loadavg_before'][0],
                 r['loadavg_after'][0]), flush=True)

    cells = {}
    for lab in STARS:
        for N in N_LIST:
            for m in METHODS:
                reps = [done[rep_key(lab, N, m, i)] for i in range(N_REPS)]
                walls = [r['wall_s'] for r in reps]
                cells['%s|N%d|%s' % (lab, N, m)] = {
                    'uid': STARS[lab], 'star_type': by_uid[STARS[lab]]['type'],
                    'N_per_band': N, 'method': m,
                    'wall_s_min_of_3': min(walls), 'wall_s_all': walls,
                    's_per_freq_min': min(walls) / N_FREQ,
                    'n_obs': reps[0]['n_obs'],
                    'band_used': reps[0]['band_used'],
                    'config': reps[0]['config'],
                    'loadavg': [r['loadavg_before'] for r in reps]
                    + [reps[-1]['loadavg_after']],
                }
    out = {
        'label': 'TIMING-ONLY reduced grid: 2000 evenly spaced freqs, '
                 'df=0.002 on [1, 4.998] cyc/day (F_MIN integer multiple '
                 'of df for the NFFT path); NOT the production df<=0.2/T '
                 'science grid (~53k freqs; full-grid Slow infeasible). '
                 'Per-freq scan cost is grid-stable (WP C5) so s/freq '
                 'extrapolates.',
        'comparability': 'ftp_mb = K=8 templates, g+r joint (2N epochs), '
                         'scan default (C4); slow = 1 template (vocab '
                         'medoid 0), primary band (N epochs), nguesses=10. '
                         'Divide ftp_mb by 8 for per-template view.',
        'protocol': 'min-of-3 reps, perf_counter over construction+fit+'
                    'power, BLAS pinned to 1 thread, np.random seeded per '
                    '(cell,rep), E5 seeded random subsample rep=0',
        'n_freq': N_FREQ, 'f_min': run_e4.F_MIN, 'f_max': run_e4.F_MAX,
        'stars': STARS, 'cells': cells,
    }
    with open(FINAL + '.tmp', 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    os.replace(FINAL + '.tmp', FINAL)
    print('ALL_DONE -> %s' % FINAL, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
