#!/usr/bin/env python
"""Adversarial sanity-check of the cached real ZTF cadence (Track B step 7).

Be skeptical: confirm the pulled cadence is actually a realistic full-survey ZTF
sample, not a handful of epochs or a single season.  Reports, across the cached
objects:

* per-band epoch-count distribution (median / range / total);
* total baseline span and per-object baselines (should span the released ZTF
  survey, ~2018..2024+, i.e. ~2000+ days);
* seasonal-gap structure -- the fraction of the year actually observed (ZTF has
  a clear yearly visibility window with a multi-month gap), via the largest and
  median gaps and an observed-season duty estimate;
* the error-vs-mag shape vs ``exp_mag_error`` defaults;
* loud warnings if anything looks wrong (too few epochs, < ~1.5 yr baseline,
  no seasonal gap, single field only).

Numbers-only (numpy + cached data + ftperiodogram core for exp_mag_error); writes
nothing by default.  ``--json PATH`` dumps the findings.

    python check_ztf_cadence.py
"""
import argparse
import glob
import json
import os

import numpy as np


DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ftperiodogram_data",
                                 "ztf_cadence_sample")


def load_records(cache_dir):
    recs = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "g*.npz"))):
        with np.load(path, allow_pickle=True) as r:
            recs.append(dict(
                gid=os.path.basename(path)[:-4],
                mjd=np.asarray(r['mjd'], float),
                mag=np.asarray(r['mag'], float),
                magerr=np.asarray(r['magerr'], float),
                catflags=(np.asarray(r['catflags'], int) if 'catflags' in r
                          else np.zeros(r['mjd'].shape, int)),
                band=np.asarray(r['band']).astype(str),
                metadata=(r['metadata'].item() if 'metadata' in r.files else {})))
    return recs


def seasonal_structure(mjd, year=365.25, season_gap_days=60.0):
    """Phase epochs into the year and estimate the observed duty cycle + the
    largest within-baseline gap (a real ZTF LC has a multi-month yearly gap).

    ``season_gap_days`` (default 60 d) is the minimum gap counted as a between-
    season visibility gap.  At ZTF's mid-northern fields the yearly gap is only
    ~2-3 months (shorter than half a year), so a 0.5-year threshold would
    under-count seasons -- 60 d catches the real yearly structure."""
    mjd = np.sort(np.asarray(mjd, float))
    if mjd.size < 3:
        return dict(duty=None, max_gap_days=None, median_gap_days=None,
                    n_seasons=None, n_season_gaps=None)
    gaps = np.diff(mjd)
    # yearly phase coverage: fraction of 24 year-bins that contain an epoch
    phase = np.mod(mjd, year)
    bins = np.linspace(0, year, 25)
    occ = np.histogram(phase, bins=bins)[0] > 0
    duty = float(occ.mean())
    n_season_gaps = int(np.sum(gaps > season_gap_days))
    return dict(duty=duty, max_gap_days=float(gaps.max()),
                median_gap_days=float(np.median(gaps)),
                n_seasons=n_season_gaps + 1, n_season_gaps=n_season_gaps)


def analyze(recs, catflags_max=0):
    per_band = {}
    baselines = []
    mjd_lo, mjd_hi = np.inf, -np.inf
    duties, max_gaps, seasons = [], [], []
    fields, modes = set(), {}
    all_mag, all_err = [], []
    for rec in recs:
        keep = rec['catflags'] <= catflags_max
        mjd = rec['mjd'][keep]
        band = rec['band'][keep]
        if mjd.size == 0:
            continue
        baselines.append(float(mjd.max() - mjd.min()))
        mjd_lo = min(mjd_lo, float(mjd.min()))
        mjd_hi = max(mjd_hi, float(mjd.max()))
        for b in set(band.tolist()):
            per_band.setdefault(b, []).append(int(np.sum(band == b)))
        ss = seasonal_structure(mjd)
        if ss['duty'] is not None:
            duties.append(ss['duty'])
            max_gaps.append(ss['max_gap_days'])
            seasons.append(ss['n_seasons'])
        meta = rec['metadata'] or {}
        for f in meta.get('fields', []):
            fields.add(int(f))
        modes[meta.get('group_mode', '?')] = modes.get(meta.get('group_mode', '?'), 0) + 1
        all_mag.append(rec['mag'][keep])
        all_err.append(rec['magerr'][keep])
    mag = np.concatenate(all_mag) if all_mag else np.array([])
    err = np.concatenate(all_err) if all_err else np.array([])
    return dict(
        n_objects=len(recs),
        per_band={b: dict(n_objects=len(c), median=float(np.median(c)),
                          min=int(np.min(c)), max=int(np.max(c)),
                          total=int(np.sum(c)))
                  for b, c in per_band.items()},
        baseline_days=dict(median=float(np.median(baselines)) if baselines else 0.0,
                           min=float(np.min(baselines)) if baselines else 0.0,
                           max=float(np.max(baselines)) if baselines else 0.0),
        mjd_span=[mjd_lo, mjd_hi], years=(mjd_hi - mjd_lo) / 365.25 if baselines else 0,
        seasonal=dict(duty_median=float(np.median(duties)) if duties else None,
                      max_gap_days_median=float(np.median(max_gaps)) if max_gaps else None,
                      n_seasons_median=float(np.median(seasons)) if seasons else None),
        fields=sorted(fields), group_modes=modes,
        mag_range=[float(mag.min()), float(mag.max())] if mag.size else None,
        magerr_median=float(np.median(err)) if err.size else None)


def mjd_to_year(mjd):
    # MJD 51544.5 = 2000-01-01
    return 2000.0 + (mjd - 51544.5) / 365.25


def adversarial_warnings(stats):
    warns = []
    bl = stats['baseline_days']
    if bl['median'] < 1.5 * 365.25:
        warns.append("median per-object baseline %.0f d < 1.5 yr -- NOT a full "
                     "ZTF survey span" % bl['median'])
    for b, st in stats['per_band'].items():
        if st['median'] < 50:
            warns.append("band %s median epochs %.0f is low for a multi-year ZTF "
                         "DR LC (expect hundreds)" % (b, st['median']))
    sea = stats['seasonal']
    # Expect roughly one season per year of baseline; <~half that is suspicious.
    expected_seasons = max(1.0, bl['median'] / 365.25)
    if (sea['n_seasons_median'] is not None
            and sea['n_seasons_median'] < 0.5 * expected_seasons):
        warns.append("median seasons %.1f << expected ~%.0f over a %.1f-yr "
                     "baseline -- weak yearly visibility-gap structure"
                     % (sea['n_seasons_median'], expected_seasons,
                        bl['median'] / 365.25))
    if sea['max_gap_days_median'] is not None and sea['max_gap_days_median'] < 30:
        warns.append("median largest gap %.0f d < 30 d -- no multi-month seasonal "
                     "gap (unlike real ZTF)" % sea['max_gap_days_median'])
    if len(stats['fields']) <= 1:
        warns.append("only %d ZTF field(s) in the sample -- low cadence diversity"
                     % len(stats['fields']))
    return warns


def main(argv=None):
    from ftperiodogram.simulate import exp_mag_error
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    p.add_argument('--catflags-max', type=int, default=0)
    p.add_argument('--json', default=None)
    args = p.parse_args(argv)

    recs = load_records(args.cache_dir)
    if not recs:
        print("no cached records in %s" % args.cache_dir)
        return 1
    stats = analyze(recs, catflags_max=args.catflags_max)

    print("=== ZTF cadence adversarial check (%d objects) ===" % stats['n_objects'])
    lo, hi = stats['mjd_span']
    print("MJD span %.1f..%.1f  =>  ~%.2f..%.2f  (%.1f yr)"
          % (lo, hi, mjd_to_year(lo), mjd_to_year(hi), stats['years']))
    bl = stats['baseline_days']
    print("per-object baseline: median %.0f d (%.1f yr), range %.0f..%.0f d"
          % (bl['median'], bl['median'] / 365.25, bl['min'], bl['max']))
    print("per-band epochs:")
    for b, st in sorted(stats['per_band'].items()):
        print("  %s: %d objects, median %.0f [%d..%d], total %d"
              % (b, st['n_objects'], st['median'], st['min'], st['max'], st['total']))
    sea = stats['seasonal']
    print("seasonal: median %.1f seasons, yearly duty %.2f, median largest gap %.0f d"
          % (sea['n_seasons_median'] or 0, sea['duty_median'] or 0,
             sea['max_gap_days_median'] or 0))
    print("fields: %s ; group modes: %s" % (stats['fields'], stats['group_modes']))
    if stats['mag_range']:
        print("mag range %.2f..%.2f, median magerr %.4f"
              % (stats['mag_range'][0], stats['mag_range'][1], stats['magerr_median']))

    exp = exp_mag_error()
    print("\nerror-vs-mag shape (real median magerr vs exp_mag_error()):")
    for mg in (14, 15, 16, 17, 18, 18.5):
        print("  mag %.1f: exp_model sigma=%.4f" % (mg, float(exp(mg))))

    warns = adversarial_warnings(stats)
    print("\n=== adversarial warnings ===")
    if warns:
        for w in warns:
            print("  !! " + w)
    else:
        print("  none -- cadence looks like a realistic full-survey ZTF sample")

    stats['warnings'] = warns
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(stats, fh, indent=2)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
