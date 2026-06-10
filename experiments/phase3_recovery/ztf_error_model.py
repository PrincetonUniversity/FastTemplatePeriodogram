#!/usr/bin/env python
"""Empirical ZTF error-vs-magnitude model built from the cached real cadence.

Track B step 3: from the measured ``magerr`` vs ``mag`` scatter of the cached real
ZTF light curves, build a smooth ``mag -> sigma`` callable that matches the
:func:`ftperiodogram.simulate.exp_mag_error` interface exactly (so it drops into a
:class:`~ftperiodogram.simulate.RealZTFCadence` / :class:`SyntheticCadence` with no
API change).

The model is a *binned median* of ``magerr`` vs ``mag`` (robust to outliers /
catflagged epochs), linearly interpolated, with:

* a constant extrapolation below the bright end and a clip to the faint-end
  median above it (so weights ``dy**-2`` never blow up, like ``exp_mag_error``'s
  ``sigma_max``);
* an optional global floor ``sigma_floor``.

This is *experiments-only* (numpy + the cached real data); the core package is
untouched.  Build it once and pass the returned callable to a cadence.

CLI: print the binned curve and how it compares to ``exp_mag_error`` defaults::

    python ztf_error_model.py
"""
import glob
import json
import os

import numpy as np


DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ftperiodogram_data",
                                 "ztf_cadence_sample")


def load_mag_magerr(cache_dir=DEFAULT_CACHE_DIR, catflags_max=0):
    """Pool (mag, magerr) over every cached grouped object, keeping clean epochs.

    Returns ``(mag, magerr, band)`` flat arrays.  ``catflags_max`` keeps only
    epochs with ``catflags <= catflags_max`` (0 = unflagged)."""
    mags, errs, bands = [], [], []
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        with np.load(path, allow_pickle=True) as rec:
            mag = np.asarray(rec['mag'], dtype=float)
            err = np.asarray(rec['magerr'], dtype=float)
            cf = (np.asarray(rec['catflags'], dtype=int) if 'catflags' in rec
                  else np.zeros(mag.size, dtype=int))
            bnd = (np.asarray(rec['band']).astype(str) if 'band' in rec
                   else np.full(mag.size, '?'))
        keep = (cf <= int(catflags_max)) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
        mags.append(mag[keep])
        errs.append(err[keep])
        bands.append(bnd[keep])
    if not mags:
        raise FileNotFoundError("no cached ZTF cadence .npz in %s" % cache_dir)
    return (np.concatenate(mags), np.concatenate(errs), np.concatenate(bands))


def binned_median_curve(mag, magerr, n_bins=24, mag_range=None, min_per_bin=10):
    """Median ``magerr`` in equal-width ``mag`` bins; drop sparse bins.

    Returns ``(bin_centers, median_sigma)`` ascending in magnitude."""
    mag = np.asarray(mag, dtype=float)
    magerr = np.asarray(magerr, dtype=float)
    if mag_range is None:
        mag_range = (float(np.percentile(mag, 0.5)), float(np.percentile(mag, 99.5)))
    edges = np.linspace(mag_range[0], mag_range[1], n_bins + 1)
    centers, meds = [], []
    for i in range(n_bins):
        m = (mag >= edges[i]) & (mag < edges[i + 1])
        if int(m.sum()) >= min_per_bin:
            centers.append(0.5 * (edges[i] + edges[i + 1]))
            meds.append(float(np.median(magerr[m])))
    if len(centers) < 2:
        raise ValueError("too few populated magnitude bins to build a curve")
    return np.asarray(centers), np.asarray(meds)


def make_empirical_error_model(cache_dir=DEFAULT_CACHE_DIR, *, n_bins=24,
                               catflags_max=0, sigma_floor=0.0,
                               mag_range=None, min_per_bin=10):
    """Build a ``mag -> sigma`` callable from the cached real magerr-vs-mag scatter.

    Matches the :func:`ftperiodogram.simulate.exp_mag_error` interface (vectorized,
    returns ``sigma`` clipped positive).  Linear interpolation of the binned-median
    curve, constant-extrapolated past both ends (bright -> first bin's sigma;
    faint -> last bin's sigma, the empirical ``sigma_max``).

    With the shipped cache the faint clip is ``sigma ~= 0.0393`` past mag ~18.5,
    so pushing ``mean_mag`` toward 20 does NOT keep lowering per-epoch SNR; for
    low-SNR studies shrink the source amplitude or raise sigma directly.

    Also returns the ``(centers, medians)`` curve for inspection/plotting as the
    callable's ``.curve`` attribute.
    """
    mag, magerr, _ = load_mag_magerr(cache_dir, catflags_max=catflags_max)
    centers, meds = binned_median_curve(mag, magerr, n_bins=n_bins,
                                        mag_range=mag_range, min_per_bin=min_per_bin)
    lo_sigma, hi_sigma = float(meds[0]), float(meds[-1])
    floor = float(sigma_floor)

    def _sigma(m):
        m = np.asarray(m, dtype=float)
        s = np.interp(m, centers, meds, left=lo_sigma, right=hi_sigma)
        return np.maximum(s, floor)

    _sigma.curve = (centers, meds)
    _sigma.faint_sigma = hi_sigma
    _sigma.bright_sigma = lo_sigma
    return _sigma


def compare_to_exp_mag_error(emp, exp, mags=None):
    """Tabulate the empirical model vs an ``exp_mag_error`` callable at sample mags.

    Returns a list of ``(mag, sigma_empirical, sigma_exp, ratio)`` rows."""
    if mags is None:
        mags = np.array([14.0, 15.0, 16.0, 17.0, 18.0, 18.5])
    rows = []
    for mg in np.atleast_1d(mags):
        se = float(emp(mg))
        sx = float(exp(mg))
        rows.append((float(mg), se, sx, se / sx if sx else float('nan')))
    return rows


def main(argv=None):
    import argparse
    from ftperiodogram.simulate import exp_mag_error
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    p.add_argument('--n-bins', type=int, default=24)
    p.add_argument('--catflags-max', type=int, default=0)
    args = p.parse_args(argv)

    mag, magerr, band = load_mag_magerr(args.cache_dir, catflags_max=args.catflags_max)
    print("pooled %d clean epochs over %d unique band labels"
          % (mag.size, len(set(band.tolist()))))
    emp = make_empirical_error_model(args.cache_dir, n_bins=args.n_bins,
                                     catflags_max=args.catflags_max)
    centers, meds = emp.curve
    print("\nbinned-median magerr vs mag (empirical ZTF):")
    for c, m in zip(centers, meds):
        print("  mag %5.2f -> sigma %.4f" % (c, m))

    exp = exp_mag_error()
    print("\nempirical vs exp_mag_error() defaults:")
    print("  %5s  %10s  %10s  %8s" % ("mag", "ZTF_emp", "exp_model", "emp/exp"))
    for mg, se, sx, ratio in compare_to_exp_mag_error(emp, exp):
        print("  %5.1f  %10.4f  %10.4f  %8.2f" % (mg, se, sx, ratio))
    print("\nempirical bright-end sigma=%.4f, faint-end sigma=%.4f"
          % (emp.bright_sigma, emp.faint_sigma))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
