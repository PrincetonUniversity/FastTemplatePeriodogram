#!/usr/bin/env python
"""Fetch a real ZTF DR cadence sample from IRSA and cache it locally.

Goal (Track B): pull ~150 *real* ZTF light curves in the RR-Lyrae magnitude
range (~14-18.5 mag), bands g and r (i when available), to recover the realistic
DR *sampling pattern* (seasonal gaps, clumping, per-band epoch counts) and the
empirical *error-vs-magnitude* relation.  We do NOT need confirmed RR Lyrae -- a
magnitude-matched sample in a well-covered field gives realistic cadence + noise.

Data source / endpoints (probed 2026-06-04)
-------------------------------------------
* IRSA TAP ``ztf_objects_dr22`` (per-object summary; select oids by magnitude):
  PREFERRED, but its Oracle backend was *down* at fetch time
  (``ORA-12541: TNS:no listener``) -- both DR22 and sibling DRs.  We retry it and
  fall back if it stays down.
* IRSA ZTF Light Curve API ``nph_light_curves`` by ``ID=<oid>`` (per-epoch
  photometry; columns oid,hjd,mjd,mag,magerr,catflags,filtercode,ra,dec,...):
  WORKS by oid even while the cone/POS spatial path is down.  This is the
  fallback we rely on -- enumerate oids in a dense field block, pull each LC,
  keep magnitude-matched ones, and group g/r/i by sky position.

The released MJD span observed at fetch time was ~58200..60950
(~2018-04 .. ~2025-09), i.e. the full ZTF survey baseline the API currently
serves (a few releases past nominal DR22).  We record the actual span in each
cached record rather than asserting a release label.

astroquery is imported lazily and ONLY here (experiments-only); the core
``ftperiodogram`` package stays pure numpy/scipy/nfft.  Because astroquery's
astropy pin conflicts with the core's numpy 2.x on Python 3.9, the TAP path is
imported only inside ``_tap_pool_oids``; the lightcurve-API path uses the stdlib
(urllib + csv) and works under any numpy.

Cache layout (atomic, skip-if-cached)
-------------------------------------
``~/.ftperiodogram_data/ztf_cadence_sample/``
    ``<gid>.npz``    one *grouped object* (g+r[+i] matched by position):
                     arrays mjd, hjd, mag, magerr, catflags, band(g/r/i),
                     plus a ``metadata`` dict (ra, dec, oids, release span...).
    ``manifest.json` summary of the whole sample.

Run::

    python fetch_ztf_cadence.py --n-objects 150        # full pull
    python fetch_ztf_cadence.py --smoke                # tiny wiring check
"""
import argparse
import csv
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np


# ---------------------------------------------------------------------------
# Cache location (mirrors ftperiodogram's ~/.ftperiodogram_data convention)
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ftperiodogram_data",
                                 "ztf_cadence_sample")

LC_API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
#: fid -> single-letter band used throughout ftperiodogram (zg/zr/zi -> g/r/i)
FID_BAND = {1: 'g', 2: 'r', 3: 'i'}
FILTERCODE_BAND = {'zg': 'g', 'zr': 'r', 'zi': 'i'}


# ---------------------------------------------------------------------------
# Low-level: IRSA ZTF Light Curve API by oid (stdlib only)
# ---------------------------------------------------------------------------
def _http_get(url, timeout, retries, pause):
    """GET with retry/backoff; return decoded text or raise the last error."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'ftperiodogram-ztf-fetch'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as exc:                       # 4xx/5xx
            body = ''
            try:
                body = exc.read().decode('utf-8', 'replace')
            except Exception:
                pass
            last = '%s: %s' % (exc.code, body[:200])
        except Exception as exc:                                    # timeout/conn
            last = '%s: %s' % (type(exc).__name__, str(exc)[:200])
        time.sleep(pause * (attempt + 1))
    raise RuntimeError("IRSA request failed after %d tries: %s" % (retries, last))


def fetch_lightcurve_by_oid(oid, *, timeout=120, retries=4, pause=4.0,
                            bad_catflags=False):
    """Pull one ZTF object's full epochal photometry via the lightcurve API.

    Returns a dict of equal-length arrays (``mjd, hjd, mag, magerr, catflags,
    band, ra, dec``) plus scalars (``filtercode, n``); ``None`` if the oid has no
    rows.  ``bad_catflags=False`` keeps every catflag (filtering is deferred to
    the cadence builder so the cache stays raw)."""
    url = LC_API + '?' + urllib.parse.urlencode({'ID': str(oid), 'FORMAT': 'CSV'})
    text = _http_get(url, timeout, retries, pause)
    if not text.strip() or text.lstrip().startswith('<'):           # empty / VOTable error
        return None
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return None

    def col(name, cast=float):
        return np.array([cast(r[name]) for r in rows])

    fc = [r['filtercode'] for r in rows]
    band = np.array([FILTERCODE_BAND.get(f, f[-1]) for f in fc])
    field = int(rows[0]['field']) if rows[0].get('field') else -1
    return dict(
        oid=str(oid), mjd=col('mjd'), hjd=col('hjd'), mag=col('mag'),
        magerr=col('magerr'), catflags=col('catflags', int), band=band,
        ra=col('ra'), dec=col('dec'), filtercode=fc[0], field=field, n=len(rows))


# ---------------------------------------------------------------------------
# oid pool: PRIMARY = TAP objects table; FALLBACK = oid enumeration
# ---------------------------------------------------------------------------
def _tap_pool_oids(n_pool, mag_lo, mag_hi, min_epochs, retries=2, pause=4.0):
    """Try the IRSA TAP ``ztf_objects_dr22`` summary table for magnitude-matched
    oids.  Returns a list of oid strings, or [] if the backend is unavailable
    (its Oracle listener has been down -- see module docstring)."""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from astroquery.ipac.irsa import Irsa
    except Exception as exc:                                        # noqa: BLE001
        print("  [tap] astroquery unavailable (%s); skipping TAP path"
              % type(exc).__name__)
        return []
    q = ("SELECT TOP %d oid, ra, dec, filtercode, ngoodobs, medianmag "
         "FROM ztf_objects_dr22 "
         "WHERE filtercode IN ('zg','zr') AND ngoodobs > %d "
         "AND medianmag BETWEEN %g AND %g" % (n_pool, min_epochs, mag_lo, mag_hi))
    for attempt in range(retries):
        try:
            tab = Irsa.query_tap(query=q).to_table()
            print("  [tap] objects table returned %d rows" % len(tab))
            return [str(o) for o in tab['oid']]
        except Exception as exc:                                    # noqa: BLE001
            print("  [tap] attempt %d failed: %s" % (attempt, str(exc)[:90]))
            time.sleep(pause * (attempt + 1))
    return []


#: 0-indexed digit of a 15-digit ZTF DR oid that encodes the filter (1=g,2=r,3=i).
#: Verified empirically: 686[1]03400067717 (zg) <-> 686[2]03400067717 (zr).
_OID_FILTER_DIGIT = 3


def _oid_for_filter(oid, fid):
    """Return the sibling oid in filter ``fid`` (1=g,2=r,3=i) -- a *different sky
    position's* running number, used only to reach that filter's oid block; g/r
    are aligned afterwards by sky position, not by oid arithmetic."""
    s = list(str(int(oid)))
    if len(s) <= _OID_FILTER_DIGIT:
        return int(oid)
    s[_OID_FILTER_DIGIT] = str(int(fid))
    return int(''.join(s))


def _enumerate_oid_pool(seed_oids, span, *, mag_lo, mag_hi, min_epochs,
                        max_probe, timeout, retries, pause, fids=(1, 2),
                        verbose=True):
    """FALLBACK: walk contiguous oid ranges around ``seed_oids`` in each requested
    filter block (``fids``: 1=g, 2=r, 3=i), pull each LC, keep magnitude-matched,
    well-sampled ones.  Contiguous oids within a field/ccd/quad block are distinct
    real sources at varied RA/mag, so a short walk harvests a magnitude-matched
    pool in every band without the (down) spatial index.  g/r/i are matched into
    one object later by sky position (:func:`_group_by_position`).

    Returns a list of full lightcurve dicts (from :func:`fetch_lightcurve_by_oid`).
    """
    starts = []
    for seed in seed_oids:
        for fid in fids:
            starts.append(_oid_for_filter(seed, fid))
    kept, probed = [], 0
    for seed in starts:
        for off in range(span):
            if probed >= max_probe:
                break
            oid = int(seed) + off
            probed += 1
            try:
                lc = fetch_lightcurve_by_oid(oid, timeout=timeout, retries=retries,
                                             pause=pause)
            except RuntimeError as exc:
                if verbose:
                    print("    oid %d: fetch error (%s)" % (oid, str(exc)[:60]))
                continue
            if lc is None:
                continue
            med = float(np.median(lc['mag']))
            if not (mag_lo <= med <= mag_hi) or lc['n'] < min_epochs:
                continue
            kept.append(lc)
            if verbose:
                print("    + oid %d  %s  nep=%d  medmag=%.2f  ra=%.4f dec=%.4f"
                      % (oid, lc['filtercode'], lc['n'], med,
                         float(np.median(lc['ra'])), float(np.median(lc['dec']))))
        if probed >= max_probe:
            break
    return kept


# ---------------------------------------------------------------------------
# Group single-filter oid LCs into multiband objects
# ---------------------------------------------------------------------------
def _oid_band(m):
    return FILTERCODE_BAND.get(m['filtercode'], m['filtercode'][-1])


def _make_record(gid, members, ra, dec, group_mode):
    bands_present = sorted({_oid_band(m) for m in members})
    return dict(
        gid=gid,
        mjd=np.concatenate([m['mjd'] for m in members]),
        hjd=np.concatenate([m['hjd'] for m in members]),
        mag=np.concatenate([m['mag'] for m in members]),
        magerr=np.concatenate([m['magerr'] for m in members]),
        catflags=np.concatenate([m['catflags'] for m in members]),
        band=np.concatenate([m['band'] for m in members]),
        metadata=dict(
            ra=ra, dec=dec, bands=bands_present, group_mode=group_mode,
            oids=[m['oid'] for m in members],
            fields=sorted({int(m['field']) for m in members}),
            per_band_oid={_oid_band(m): m['oid'] for m in members},
            source='IRSA nph_light_curves (by oid)'))


def group_objects(lcs, match_arcsec=1.5, pair_same_field=True):
    """Group single-filter oid light curves into (multiband) objects.

    Two grouping modes, in priority order:

    1. ``same_star`` -- oids at the *same sky position* (within ``match_arcsec``)
       are the genuine multiband photometry of one physical star.  This is the
       ideal, but ZTF's per-filter reference catalogs use *independent* running-
       number orderings, so without the (currently-down) IRSA spatial index a
       walk through the g-block and the r-block lands on disjoint positions and
       this almost never fires.

    2. ``same_field_cadence`` -- a fallback that pairs an unmatched g oid with an
       unmatched r oid *from the same ZTF field*.  These are different physical
       stars, but ZTF observes a field in g and r off the *same* survey schedule,
       so the pair's epoch *times* are an authentic real two-band cadence (real
       seasonal gaps, clumping, per-band counts) -- exactly what Track B needs
       (sampling + error-vs-mag, not a confirmed RR Lyrae).  The grouping mode is
       recorded in each record's metadata so it is never silently conflated with
       a true same-star object.

    Returns a list of records (dicts) ready for :func:`cache_group`.
    """
    pts = [(float(np.median(lc['ra'])), float(np.median(lc['dec'])), lc)
           for lc in lcs]
    used = [False] * len(pts)
    tol_deg = match_arcsec / 3600.0
    records, gi = [], 0

    # Mode 1: genuine same-star multiband by sky position ---------------------
    for i in range(len(pts)):
        if used[i]:
            continue
        ra_i, dec_i, lc_i = pts[i]
        members = [lc_i]
        cosd = np.cos(np.radians(dec_i))
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            ra_j, dec_j, lc_j = pts[j]
            dra = (ra_i - ra_j) * cosd
            ddec = dec_i - dec_j
            if dra * dra + ddec * ddec <= tol_deg * tol_deg:
                members.append(lc_j)
                used[j] = True
        if len(members) >= 2:                       # only keep true multiband here
            used[i] = True
            records.append(_make_record("g%05d" % gi, members, ra_i, dec_i,
                                        'same_star'))
            gi += 1

    # Mode 2: pair leftover g & r oids per field for a real two-band cadence ---
    leftover = [pts[k][2] for k in range(len(pts)) if not used[k]]
    if pair_same_field:
        by_field = {}
        for lc in leftover:
            by_field.setdefault(int(lc['field']), {}).setdefault(
                _oid_band(lc), []).append(lc)
        consumed = set()
        for field, byband in by_field.items():
            gs = list(byband.get('g', []))
            rs = list(byband.get('r', []))
            for g_lc, r_lc in zip(gs, rs):
                ra = float(np.median(g_lc['ra']))
                dec = float(np.median(g_lc['dec']))
                records.append(_make_record("g%05d" % gi, [g_lc, r_lc], ra, dec,
                                            'same_field_cadence'))
                consumed.add(id(g_lc))
                consumed.add(id(r_lc))
                gi += 1
        leftover = [lc for lc in leftover if id(lc) not in consumed]

    # Anything still single-filter: keep as a single-band object --------------
    for lc in leftover:
        records.append(_make_record("g%05d" % gi, [lc],
                                    float(np.median(lc['ra'])),
                                    float(np.median(lc['dec'])), 'single_band'))
        gi += 1
    return records


# ---------------------------------------------------------------------------
# Atomic caching (mirror the Sesar/BV loader pattern)
# ---------------------------------------------------------------------------
def _atomic_savez(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def cache_group(rec, cache_dir):
    """Atomically write one grouped object to ``<gid>.npz`` (skip if present)."""
    path = os.path.join(cache_dir, "%s.npz" % rec['gid'])
    if os.path.exists(path):
        return path, False
    _atomic_savez(
        path, mjd=rec['mjd'], hjd=rec['hjd'], mag=rec['mag'],
        magerr=rec['magerr'], catflags=rec['catflags'], band=rec['band'],
        metadata=np.array(rec['metadata'], dtype=object))
    return path, True


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def summarize(records):
    """Per-band epoch counts, mag/baseline/error spans across the sample."""
    per_band_counts = {}
    all_baselines, all_mag, all_err = [], [], []
    mjd_min, mjd_max = np.inf, -np.inf
    multiband = 0
    group_modes = {}
    for rec in records:
        mode = rec.get('metadata', {}).get('group_mode', 'unknown')
        group_modes[mode] = group_modes.get(mode, 0) + 1
        b = rec['band']
        bands_here = set(b.tolist())
        if len(bands_here) >= 2:
            multiband += 1
        for band in bands_here:
            m = b == band
            per_band_counts.setdefault(band, []).append(int(m.sum()))
        all_baselines.append(float(rec['mjd'].max() - rec['mjd'].min()))
        all_mag.append(rec['mag'])
        all_err.append(rec['magerr'])
        mjd_min = min(mjd_min, float(rec['mjd'].min()))
        mjd_max = max(mjd_max, float(rec['mjd'].max()))
    mag = np.concatenate(all_mag) if all_mag else np.array([])
    err = np.concatenate(all_err) if all_err else np.array([])
    summary = dict(
        n_objects=len(records), n_multiband=multiband, group_modes=group_modes,
        mjd_min=mjd_min, mjd_max=mjd_max,
        baseline_days_median=float(np.median(all_baselines)) if all_baselines else 0.0,
        baseline_days_max=float(np.max(all_baselines)) if all_baselines else 0.0,
        per_band_epochs={
            band: dict(n_objects=len(c), median=float(np.median(c)),
                       min=int(np.min(c)), max=int(np.max(c)),
                       total=int(np.sum(c)))
            for band, c in per_band_counts.items()},
        mag_range=[float(mag.min()), float(mag.max())] if mag.size else None,
        magerr_range=[float(err.min()), float(err.max())] if err.size else None)
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-objects', type=int, default=150,
                   help="target number of grouped (g+r) objects to cache")
    p.add_argument('--mag-lo', type=float, default=14.0)
    p.add_argument('--mag-hi', type=float, default=18.5)
    p.add_argument('--min-epochs', type=int, default=50,
                   help="minimum per-oid epoch count to keep")
    p.add_argument('--seed-oids', default='686103400067717',
                   help="comma-separated oid seeds for the enumeration fallback "
                        "(dense ZTF field blocks)")
    p.add_argument('--walk-span', type=int, default=400,
                   help="contiguous oids to walk from each seed")
    p.add_argument('--max-probe', type=int, default=900,
                   help="hard cap on total LC API requests")
    p.add_argument('--match-arcsec', type=float, default=1.5,
                   help="sky-match radius to group g/r/i oids into one object")
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--retries', type=int, default=4)
    p.add_argument('--pause', type=float, default=3.0)
    p.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    p.add_argument('--no-tap', action='store_true',
                   help="skip the TAP objects path, go straight to enumeration")
    p.add_argument('--bands', default='g,r',
                   help="bands to harvest (filter blocks): g,r[,i]")
    p.add_argument('--no-pair-field', action='store_true',
                   help="disable the same-field g/r cadence-pairing fallback "
                        "(keep only genuine same-star multiband + single-band)")
    p.add_argument('--smoke', action='store_true',
                   help="tiny wiring check (a handful of objects)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args.n_objects = 4
        args.walk_span = 20
        args.max_probe = 60          # reach both the g- and r-filter blocks
        args.min_epochs = 30
    os.makedirs(args.cache_dir, exist_ok=True)
    seed_oids = [int(x) for x in str(args.seed_oids).split(',') if x.strip()]
    band_to_fid = {'g': 1, 'r': 2, 'i': 3}
    fids = tuple(band_to_fid[b] for b in str(args.bands).split(',')
                 if b.strip() in band_to_fid)
    if not fids:
        fids = (1, 2)
    t0 = time.time()

    print("ZTF cadence fetch: target %d objects, mag [%g, %g], >=%d epochs/oid"
          % (args.n_objects, args.mag_lo, args.mag_hi, args.min_epochs))
    print("cache dir: %s" % args.cache_dir)

    # 1. Build a pool of single-filter oid light curves -----------------------
    lcs = []
    if not args.no_tap:
        print("[1] trying IRSA TAP ztf_objects_dr22 for magnitude-matched oids ...")
        oids = _tap_pool_oids(args.n_objects * 3, args.mag_lo, args.mag_hi,
                              args.min_epochs)
        for oid in oids:
            try:
                lc = fetch_lightcurve_by_oid(oid, timeout=args.timeout,
                                             retries=args.retries, pause=args.pause)
            except RuntimeError:
                continue
            if lc is not None and lc['n'] >= args.min_epochs:
                lcs.append(lc)
    if not lcs:
        print("[1] TAP path empty/unavailable; FALLBACK = oid enumeration via "
              "lightcurve API")
        # Need ~2x oids (g and r separately) for n_objects grouped pairs.
        lcs = _enumerate_oid_pool(
            seed_oids, args.walk_span, mag_lo=args.mag_lo, mag_hi=args.mag_hi,
            min_epochs=args.min_epochs, max_probe=args.max_probe,
            timeout=args.timeout, retries=args.retries, pause=args.pause,
            fids=fids)
    print("[1] pooled %d single-filter light curves" % len(lcs))
    if not lcs:
        print("FETCH FAILED: no light curves retrieved (IRSA backend down?). "
              "Nothing cached. Reporting shortfall honestly.")
        return 1

    # 2. Group g/r/i oids into multiband objects ------------------------------
    groups = group_objects(lcs, match_arcsec=args.match_arcsec,
                           pair_same_field=not args.no_pair_field)
    by_mode = {}
    for g in groups:
        by_mode[g['metadata']['group_mode']] = \
            by_mode.get(g['metadata']['group_mode'], 0) + 1
    print("[2] grouped into %d objects (%d multiband); modes=%s"
          % (len(groups), sum(1 for g in groups if len(set(g['band'])) >= 2),
             by_mode))
    groups = groups[:args.n_objects]

    # 3. Cache atomically -----------------------------------------------------
    n_written, n_skipped = 0, 0
    for rec in groups:
        _, wrote = cache_group(rec, args.cache_dir)
        n_written += int(wrote)
        n_skipped += int(not wrote)
    print("[3] cached %d new, %d already present" % (n_written, n_skipped))

    # 4. Summary + manifest ---------------------------------------------------
    summary = summarize(groups)
    summary['fetch_seconds'] = round(time.time() - t0, 1)
    summary['gids'] = [g['gid'] for g in groups]
    manifest_path = os.path.join(args.cache_dir, 'manifest.json')
    tmp = manifest_path + ".tmp"
    with open(tmp, 'w') as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, manifest_path)

    print("\n=== ZTF cadence sample summary ===")
    print("objects: %d (%d multiband); group modes: %s"
          % (summary['n_objects'], summary['n_multiband'],
             summary['group_modes']))
    print("MJD span: %.1f .. %.1f  (%.0f d, ~%.1f yr)"
          % (summary['mjd_min'], summary['mjd_max'],
             summary['mjd_max'] - summary['mjd_min'],
             (summary['mjd_max'] - summary['mjd_min']) / 365.25))
    print("per-object baseline: median %.0f d, max %.0f d"
          % (summary['baseline_days_median'], summary['baseline_days_max']))
    for band, st in sorted(summary['per_band_epochs'].items()):
        print("  band %s: %d objects, epochs median=%.0f [%d..%d], total=%d"
              % (band, st['n_objects'], st['median'], st['min'], st['max'],
                 st['total']))
    if summary['mag_range']:
        print("mag range: %.2f .. %.2f" % tuple(summary['mag_range']))
    if summary['magerr_range']:
        print("magerr range: %.4f .. %.4f" % tuple(summary['magerr_range']))
    print("wrote manifest %s in %.1fs" % (manifest_path, summary['fetch_seconds']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
