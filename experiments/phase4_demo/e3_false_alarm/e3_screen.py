#!/usr/bin/env python
"""E3 Phase 1: variability screen of the 99-object ZTF cadence sample.

The cached objects in ~/.ftperiodogram_data/ztf_cadence_sample/ are
magnitude-selected field stars (14-18.5), NOT confirmed variables. Before
using them as a near-null real sample (PLAN.md Phase 4, element 2) we screen
each underlying ZTF star for known variability and for photometric-amplitude
excess:

  (1) resolve per-oid positions via IRSA TAP (same_field_cadence pairs are
      two DIFFERENT stars, up to ~1 deg apart -- metadata ra/dec covers only
      one member, so every oid is resolved and screened individually);
  (2) crossmatch every unique star (<=2 arcsec) against
        - Chen+2020 ZTF variable catalog (VizieR J/ApJS/249/18/table2)
        - Gaia DR3 variability (gaiadr3.vari_summary joined to gaia_source
          and vari_classifier_result, ESA Gaia TAP);
  (3) per-band robust amplitude statistic from the cached photometry
      (catflags==0): sigma_MAD = 1.4826*MAD(mag) and sigma_IQR = IQR/1.349
      vs the median reported magerr; outlier flag is sample-relative:
      ratio = sigma_MAD/median(magerr) >= median(ratio) + 5*MAD(ratio),
      computed over all 155 band-lightcurves with >=20 good epochs.

Network: sync TAP only, batched OR-CONTAINS cones (<=10 circles/query),
<=1 request/s, 3 retries with backoff, every batch cached to --cache-dir
(skip-if-exists => fully resumable).

Output: screen_table.json (per object: per-band stats + match records +
flags) and screen_memo_section.md next to this script.
"""
import argparse
import csv
import glob
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

VIZ = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
GAIA = "https://gea.esac.esa.int/tap-server/tap/sync"
IRSA = "https://irsa.ipac.caltech.edu/TAP/sync"

MATCH_RADIUS_ARCSEC = 2.0
MATCH_RADIUS_DEG = MATCH_RADIUS_ARCSEC / 3600.0
CIRCLES_PER_QUERY = 10
MIN_EPOCHS = 20
LAST_REQUEST = [0.0]


def tap(url, adql, timeout=60, retries=3):
    """Sync TAP query -> list of dict rows; <=1 req/s; retry with backoff."""
    for attempt in range(retries):
        wait = 1.0 - (time.time() - LAST_REQUEST[0])
        if wait > 0:
            time.sleep(wait)
        LAST_REQUEST[0] = time.time()
        try:
            data = urllib.parse.urlencode({
                "REQUEST": "doQuery", "LANG": "ADQL",
                "FORMAT": "csv", "QUERY": adql}).encode()
            req = urllib.request.Request(
                url, data=data, headers={"User-Agent": "ftp-e3-screen"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read().decode("utf-8", "replace")
            return list(csv.DictReader(io.StringIO(text)))
        except Exception as e:
            if attempt == retries - 1:
                raise
            back = 3.0 * (attempt + 1)
            print("  [tap] retry in %.0fs after: %s" % (back, str(e)[:120]))
            time.sleep(back)


def cached(path, fn):
    """Resumable step: load JSON cache if present, else compute and save."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    out = fn()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    return out


def sep_arcsec(ra1, dec1, ra2, dec2):
    cd = math.cos(math.radians(0.5 * (dec1 + dec2)))
    return math.hypot((ra1 - ra2) * cd, dec1 - dec2) * 3600.0


def load_objects(sample_dir):
    objs = []
    for f in sorted(glob.glob(os.path.join(sample_dir, "g*.npz"))):
        d = np.load(f, allow_pickle=True)
        md = d["metadata"].item()
        ok = ((d["catflags"] == 0) & np.isfinite(d["mag"])
              & np.isfinite(d["magerr"]) & (d["magerr"] > 0))
        band_stats = {}
        for b in sorted(set(d["band"].tolist())):
            m = ok & (d["band"] == b)
            n = int(m.sum())
            st = {"oid": str(md["per_band_oid"][b]), "n_good": n}
            if n >= MIN_EPOCHS:
                mag = d["mag"][m]
                dy = d["magerr"][m]
                smad = 1.4826 * float(np.median(np.abs(mag - np.median(mag))))
                siqr = float(np.percentile(mag, 75)
                             - np.percentile(mag, 25)) / 1.349
                meddy = float(np.median(dy))
                st.update({
                    "median_mag": round(float(np.median(mag)), 4),
                    "median_magerr": round(meddy, 5),
                    "sigma_mad": round(smad, 5),
                    "sigma_iqr": round(siqr, 5),
                    "ratio_mad": round(smad / meddy, 3),
                })
            band_stats[b] = st
        objs.append({
            "gid": os.path.basename(f)[:-4],
            "group_mode": md["group_mode"],
            "meta_ra": md["ra"], "meta_dec": md["dec"],
            "bands": band_stats,
        })
    return objs


def fetch_oid_positions(oids, cache_dir):
    """IRSA ztf_objects_dr23 positions for every oid, in batches of 50."""
    def run():
        pos = {}
        batch = 50
        for i in range(0, len(oids), batch):
            chunk = oids[i:i + batch]
            q = ("SELECT oid, ra, dec, medianmag, ngoodobsrel "
                 "FROM ztf_objects_dr23 WHERE oid IN (%s)"
                 % ",".join(chunk))
            for row in tap(IRSA, q):
                pos[row["oid"]] = {
                    "ra": float(row["ra"]), "dec": float(row["dec"]),
                    "medianmag": float(row["medianmag"])}
            print("  [irsa] oids %d-%d: %d resolved"
                  % (i, i + len(chunk) - 1, len(pos)))
        return pos
    return cached(os.path.join(cache_dir, "oid_positions.json"), run)


def unique_stars(objs, oid_pos):
    """Dedupe oids into physical stars (positions agreeing within 1'')."""
    stars = []  # each: {ra, dec, oids: []}
    for o in objs:
        for b, st in o["bands"].items():
            oid = st["oid"]
            if oid not in oid_pos:
                print("  WARNING: no IRSA position for oid %s (%s %s)"
                      % (oid, o["gid"], b))
                continue
            p = oid_pos[oid]
            hit = None
            for s in stars:
                if sep_arcsec(p["ra"], p["dec"], s["ra"], s["dec"]) <= 1.0:
                    hit = s
                    break
            if hit is None:
                hit = {"ra": p["ra"], "dec": p["dec"], "oids": []}
                stars.append(hit)
            hit["oids"].append(oid)
    return stars


def cone_batches(stars, cache_dir, tag, url, select_from, point_expr):
    """Batched OR-CONTAINS cones; returns all rows across batches."""
    rows = []
    for i in range(0, len(stars), CIRCLES_PER_QUERY):
        chunk = stars[i:i + CIRCLES_PER_QUERY]
        path = os.path.join(cache_dir, "%s_batch_%03d.json"
                            % (tag, i // CIRCLES_PER_QUERY))

        def run(chunk=chunk):
            cond = " OR ".join(
                "1=CONTAINS(%s,CIRCLE('ICRS',%.7f,%.7f,%.6f))"
                % (point_expr, s["ra"], s["dec"], MATCH_RADIUS_DEG)
                for s in chunk)
            return tap(url, select_from + " WHERE " + cond)
        got = cached(path, run)
        print("  [%s] batch %d (%d circles): %d rows"
              % (tag, i // CIRCLES_PER_QUERY, len(chunk), len(got)))
        rows.extend(got)
    return rows


def match_rows_to_stars(rows, stars, ra_key, dec_key):
    """Attach catalog rows to stars by separation <= 2 arcsec."""
    out = {id(s): [] for s in stars}
    for row in rows:
        ra, dec = float(row[ra_key]), float(row[dec_key])
        best, best_sep = None, MATCH_RADIUS_ARCSEC
        for s in stars:
            d = sep_arcsec(ra, dec, s["ra"], s["dec"])
            if d <= best_sep:
                best, best_sep = s, d
        if best is not None:
            row = dict(row)
            row["_sep_arcsec"] = round(best_sep, 3)
            out[id(best)].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", default=os.path.expanduser(
        "~/.ftperiodogram_data/ztf_cadence_sample"))
    ap.add_argument("--cache-dir", default=os.path.expanduser(
        "~/.ftperiodogram_data/e3_screen_cache"))
    ap.add_argument("--out-dir", default=os.path.dirname(
        os.path.abspath(__file__)))
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    print("== load cached sample ==")
    objs = load_objects(args.sample_dir)
    print("%d objects" % len(objs))

    print("== resolve per-oid positions (IRSA) ==")
    all_oids = sorted({st["oid"] for o in objs for st in o["bands"].values()})
    oid_pos = fetch_oid_positions(all_oids, args.cache_dir)
    print("%d/%d oids resolved" % (len(oid_pos), len(all_oids)))

    stars = unique_stars(objs, oid_pos)
    print("%d unique stars (1 arcsec dedupe)" % len(stars))

    print("== Chen+2020 (VizieR J/ApJS/249/18/table2) ==")
    chen_rows = cone_batches(
        stars, args.cache_dir, "chen", VIZ,
        'SELECT RAJ2000,DEJ2000,Per,Type,gmag,rmag '
        'FROM "J/ApJS/249/18/table2"',
        "POINT('ICRS',RAJ2000,DEJ2000)")
    chen_by_star = match_rows_to_stars(chen_rows, stars, "RAJ2000", "DEJ2000")

    print("== Gaia DR3 vari_summary (ESA Gaia TAP) ==")
    gaia_rows = cone_batches(
        stars, args.cache_dir, "gaia", GAIA,
        "SELECT s.source_id, s.ra, s.dec, s.phot_g_mean_mag,"
        " c.best_class_name, c.best_class_score"
        " FROM gaiadr3.vari_summary v"
        " JOIN gaiadr3.gaia_source s ON s.source_id=v.source_id"
        " LEFT JOIN gaiadr3.vari_classifier_result c"
        " ON c.source_id=v.source_id",
        "POINT('ICRS',s.ra,s.dec)")
    gaia_by_star = match_rows_to_stars(gaia_rows, stars, "ra", "dec")

    # oid -> star and star-level match records
    star_of_oid = {}
    for s in stars:
        for oid in s["oids"]:
            star_of_oid[oid] = s

    # sample-relative amplitude outlier threshold
    ratios = np.array([st["ratio_mad"] for o in objs
                       for st in o["bands"].values() if "ratio_mad" in st])
    med = float(np.median(ratios))
    sig = 1.4826 * float(np.median(np.abs(ratios - med)))  # robust sigma
    thresh = med + 5.0 * sig       # outlier flag (5 sigma_MAD)
    thresh_marg = med + 3.0 * sig  # marginal tier (3 sigma_MAD)

    table = {}
    n_matched = n_amp = n_clean = 0
    for o in objs:
        rec = {
            "group_mode": o["group_mode"],
            "meta_ra": o["meta_ra"], "meta_dec": o["meta_dec"],
            "bands": {},
        }
        matched = []
        amp_bands = []
        marg_bands = []
        for b, st in o["bands"].items():
            bs = dict(st)
            s = star_of_oid.get(st["oid"])
            if s is not None:
                bs["star_ra"] = round(s["ra"], 6)
                bs["star_dec"] = round(s["dec"], 6)
                chen = chen_by_star.get(id(s), [])
                gaia = gaia_by_star.get(id(s), [])
                if chen:
                    c = min(chen, key=lambda r: r["_sep_arcsec"])
                    bs["chen2020"] = {
                        "sep_arcsec": c["_sep_arcsec"], "type": c["Type"],
                        "period_days": c["Per"], "rmag": c["rmag"]}
                    matched.append("chen2020:%s:%s" % (b, c["Type"]))
                if gaia:
                    g = min(gaia, key=lambda r: r["_sep_arcsec"])
                    bs["gaia_dr3_vari"] = {
                        "sep_arcsec": g["_sep_arcsec"],
                        "source_id": g["source_id"],
                        "best_class_name": g["best_class_name"],
                        "best_class_score": g["best_class_score"]}
                    matched.append("gaia:%s:%s" % (b, g["best_class_name"]))
            r = bs.get("ratio_mad", 0.0)
            bs["amplitude_flag"] = bool(r >= thresh)
            bs["amplitude_marginal"] = bool(thresh_marg <= r < thresh)
            if bs["amplitude_flag"]:
                amp_bands.append(b)
            elif bs["amplitude_marginal"]:
                marg_bands.append(b)
            rec["bands"][b] = bs
        rec["matched_variable"] = sorted(set(matched))
        rec["amplitude_flag_bands"] = amp_bands
        rec["amplitude_marginal_bands"] = marg_bands
        # strict clean: no catalog match, no 5-sigma flag, no 3-sigma marginal
        rec["clean"] = not matched and not amp_bands and not marg_bands
        n_matched += bool(matched)
        n_amp += bool(amp_bands)
        n_clean += rec["clean"]
        table[o["gid"]] = rec

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "experiments/phase4_demo/e3_false_alarm/e3_screen.py",
        "n_objects": len(objs),
        "n_unique_stars": len(stars),
        "match_radius_arcsec": MATCH_RADIUS_ARCSEC,
        "catalogs": ["Chen+2020 J/ApJS/249/18/table2 (VizieR TAP)",
                     "Gaia DR3 vari_summary+vari_classifier_result (ESA TAP)"],
        "amplitude_stat": "ratio_mad = 1.4826*MAD(mag)/median(magerr), "
                          "catflags==0, per band, n_good>=%d" % MIN_EPOCHS,
        "amplitude_threshold_flag": round(thresh, 3),
        "amplitude_threshold_marginal": round(thresh_marg, 3),
        "amplitude_threshold_rule": "flag: ratio >= median + 5*sigma_MAD; "
                                    "marginal: >= median + 3*sigma_MAD; "
                                    "sigma_MAD = 1.4826*MAD(ratio_mad) over "
                                    "all %d band-lightcurves" % len(ratios),
        "ratio_mad_median": round(med, 3),
        "ratio_mad_sigma": round(sig, 3),
        "summary": {"n_matched_variable": n_matched,
                    "n_amplitude_flagged": n_amp,
                    "n_amplitude_marginal_only": sum(
                        1 for r in table.values()
                        if r["amplitude_marginal_bands"]
                        and not r["amplitude_flag_bands"]),
                    "n_clean": n_clean},
        "objects": table,
    }
    out_json = os.path.join(args.out_dir, "screen_table.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote %s" % out_json)
    print(json.dumps(out["summary"]))
    return out


if __name__ == "__main__":
    main()
