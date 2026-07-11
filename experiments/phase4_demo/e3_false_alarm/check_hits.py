#!/usr/bin/env python
"""WP E3 phase 3 -- peak-frequency identification for the screened hits.

For the screened objects that exceed the null thresholds, recompute the
catalog-max and GLS spectra (identical machinery/seeds to run_e3.py) and
report the peak frequency, compared against the Chen+2020 orbital frequency
f_orb = 1/P and its double 2/P (EW light curves are dominated by the second
harmonic of the orbit, so a half-period alias at 2 f_orb is the expected
detection). Uses the primary N=40 subsample (exact D1 10k grid) and, for the
two Chen EWs, the full-N secondary df=0.2/T grid as well.
"""
import os
import warnings

import numpy as np

warnings.simplefilter('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import run_e3  # noqa: E402  (same directory; reuse machinery + seeds)
from ftperiodogram.validation import frequency_grid  # noqa: E402
from ftperiodogram.core import template_periodogram  # noqa: E402
from ftperiodogram.baselines import GLSEstimator  # noqa: E402

# (gid, band, Chen+2020 period in days or None)
HITS = [('g00015', 'g', 1.2068), ('g00017', 'g', 0.46366),
        ('g00052', 'r', None)]


def spectra(t, y, dy, freqs, ftp_coeffs, vocab_coeffs):
    cat = None
    for c_n, s_n in vocab_coeffs:
        p, _ = template_periodogram(t, y, dy, c_n, s_n, freqs)
        cat = p if cat is None else np.maximum(cat, p)
    gls = GLSEstimator().power_spectrum(t, y, None, dy, freqs)
    return cat, gls


def report(tag, freqs, cat, gls, period):
    fc, fg = freqs[np.argmax(cat)], freqs[np.argmax(gls)]
    line = '%-24s catK8: f_peak=%.5f (P=%.5f) max=%.4f | GLS: f_peak=%.5f max=%.4f' \
        % (tag, fc, 1.0 / fc, cat.max(), fg, gls.max())
    if period is not None:
        line += ' | Chen f_orb=%.5f 2f_orb=%.5f (df_cat=%.4f from 2f_orb)' \
            % (1.0 / period, 2.0 / period, fc - 2.0 / period)
    print(line, flush=True)


def main():
    ftp_coeffs, vocab_coeffs = run_e3.build_templates()
    vocab = [(np.asarray(c), np.asarray(s)) for c, s in vocab_coeffs]
    gids = run_e3.list_gids()
    for gid, bnd, period in HITS:
        i_obj = gids.index(gid)
        per_band, _ = run_e3.load_object(gid)
        t, y, dy = per_band[bnd]
        # primary: identical seeded N=40 subsample + D1 grid
        seed = run_e3.SUB_SEED_BASE + i_obj * 10 + run_e3.BAND_IDX[bnd]
        idx = np.sort(np.random.default_rng(seed).choice(
            t.size, run_e3.N_SUB, replace=False))
        freqs = frequency_grid(run_e3.F_MIN, run_e3.F_MAX, run_e3.N_FREQ_D1)
        cat, gls = spectra(t[idx], y[idx], dy[idx], freqs, ftp_coeffs, vocab)
        report('%s %s prim(N=40)' % (gid, bnd), freqs, cat, gls, period)
        if period is not None:  # full-N secondary grid for the EWs
            T = float(t.max() - t.min())
            freqs, _, _, _ = run_e3.secondary_grid(T)
            cat, gls = spectra(t, y, dy, freqs, ftp_coeffs, vocab)
            report('%s %s sec(N=%d)' % (gid, bnd, t.size), freqs, cat, gls,
                   period)


if __name__ == '__main__':
    main()
