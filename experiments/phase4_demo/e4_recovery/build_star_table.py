"""WP E4 Phase 1 prep: dedup'd star table + independent Gaia periods.

Builds ``star_table.json`` for the E4 known-period recovery run from the
Phase-4 sample manifest (``../manifest.json``):

1. **Dedupe by ZTF oid pair.**  The 85 manifest entries collapse to 73 unique
   stars: 12 Gaia controls are byte-identical duplicates of Chen science stars
   ("dual-truth" stars -- one light curve, two independent truth periods), and
   are folded into their science twin rather than counted separately.

2. **Per-star light-curve stats** from the fetched npz files: per-band good
   epochs, per-band baseline T (days, from HJD span), joint (union) baseline,
   and the E4 frequency-grid parameters (df = 0.2/T per band; two-band methods
   use the joint grid at the *shorter* per-band T; explicit [1, 5] cyc/day).

3. **Independent Gaia DR3 periods for the science stars** (bounded network
   step, <= 3 requests): re-runs the exact per-field ``vari_rrlyrae x
   gaia_source_lite`` box query from ``fetch_rrl_sample.py`` against ESA sync
   TAP (fresh pull, cached under ``gaia_xmatch/`` for resume), then
   crossmatches the 58 Chen science stars within 1 arcsec.  Science stars get
   a second truth column ``gaia_period`` (null where Gaia SOS missed them).

Roles carried per unique star:
  * ``science``          -- 58 Chen-selected stars (Chen ``Per`` truth).
  * ``control_orig``     -- 3 original non-Chen-overlap Gaia controls.
  * ``control_antijoin`` -- 12 Chen-missed anti-join Gaia controls
                            (the selection-function measurement sample).

Usage::

    .venv/bin/python experiments/phase4_demo/e4_recovery/build_star_table.py
        [--offline]   # skip TAP, use cached gaia_xmatch/*.csv only

stdlib + numpy only; <= 1 request/s; every number in star_table.json is
computed here from the manifest, the npz files, and the TAP pulls.
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, '..', 'manifest.json')
DATA_DIR = os.path.expanduser('~/.ftperiodogram_data/phase4_rrl_sample')
XMATCH_DIR = os.path.join(HERE, 'gaia_xmatch')
OUT_JSON = os.path.join(HERE, 'star_table.json')

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
FIELD_CENTERS = {486: (274.351, 4.550), 686: (298.269, 33.350),
                 786: (147.601, 54.950)}
HALF_DEG = 3.43                        # identical to fetch_rrl_sample.py
XMATCH_RADIUS_ARCSEC = 1.0             # identical to fetch_rrl_sample.py

F_MIN, F_MAX = 1.0, 5.0                # cyc/day, explicit (never nyquist_factor)
DF_FACTOR = 0.2                        # df = DF_FACTOR / T

_LAST_REQ = [0.0]
STATS = {"requests": 0, "bytes": 0, "cache_hits": 0}


# ------------------------------------------------------------- TAP plumbing
def _http(url, post_data, timeout=120):
    last_err = None
    for attempt in range(3):
        wait = 1.0 - (time.time() - _LAST_REQ[0])   # sustain <= 1 req/s
        if wait > 0:
            time.sleep(wait)
        _LAST_REQ[0] = time.time()
        STATS["requests"] += 1
        try:
            req = urllib.request.Request(
                url, data=post_data,
                headers={"User-Agent": "ftp-phase4-e4-gaia-xmatch (johnh2o2)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            STATS["bytes"] += len(body)
            return body.decode("utf-8", "replace")
        except Exception as exc:                    # noqa: BLE001
            last_err = exc
            if attempt < 2:
                time.sleep(2.0 + 4.0 * attempt)
    raise RuntimeError("TAP failed after 3 attempts: %s" % str(last_err)[:200])


def tap_query(adql, timeout=120):
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": adql}).encode()
    return _http(GAIA_TAP, data, timeout=timeout)


def _cached_text(cache_path, fetch_fn):
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        STATS["cache_hits"] += 1
        with open(cache_path) as fh:
            return fh.read()
    text = fetch_fn()
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, cache_path)
    return text


def field_box(field):
    rc, dc = FIELD_CENTERS[field]
    dra = HALF_DEG / math.cos(math.radians(dc))
    return rc - dra, rc + dra, dc - HALF_DEG, dc + HALF_DEG


def sep_arcsec(ra1, dec1, ra2, dec2):
    cd = math.cos(math.radians(dec1))
    return math.hypot((ra1 - ra2) * cd, dec1 - dec2) * 3600.0


def ffloat(v, default=None):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def fetch_gaia_field(field, offline):
    """Fresh ESA TAP pull of the per-field Gaia SOS RRL table (query pattern
    identical to fetch_rrl_sample.fetch_gaia_field; cached for resume)."""
    r0, r1, d0, d1 = field_box(field)
    adql = ("SELECT v.source_id, g.ra, g.dec, v.pf, v.p1_o, "
            "v.best_classification, g.phot_g_mean_mag "
            "FROM gaiadr3.vari_rrlyrae AS v "
            "JOIN gaiadr3.gaia_source_lite AS g ON g.source_id = v.source_id "
            "WHERE g.ra BETWEEN %.6f AND %.6f "
            "AND g.dec BETWEEN %.6f AND %.6f" % (r0, r1, d0, d1))
    cache = os.path.join(XMATCH_DIR, "gaia_field%d.csv" % field)
    if offline:
        fallback = os.path.join(DATA_DIR, "raw", "gaia_field%d.csv" % field)
        path = cache if os.path.exists(cache) else fallback
        with open(path) as fh:
            text = fh.read()
    else:
        text = _cached_text(cache, lambda: tap_query(adql))
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        cls = (r.get("best_classification") or "").strip()
        pf, p1o = ffloat(r.get("pf")), ffloat(r.get("p1_o"))
        if cls == "RRab":
            per, psrc = (pf, "gaia_dr3_pf") if pf else (p1o, "gaia_dr3_p1_o")
        else:
            per, psrc = (p1o, "gaia_dr3_p1_o") if p1o else (pf, "gaia_dr3_pf")
        ra, dec = ffloat(r.get("ra")), ffloat(r.get("dec"))
        if ra is None or dec is None or per is None:
            continue
        rows.append({"ra": ra, "dec": dec, "per": per, "per_source": psrc,
                     "type": cls, "gmag": ffloat(r.get("phot_g_mean_mag")),
                     "source_id": r.get("source_id", "").strip()})
    return rows


# --------------------------------------------------------------- star table
def oid_key(entry):
    return tuple(sorted(b["oid"] for b in entry["bands"].values()))


def lc_stats(lc_file):
    """Per-band epochs / baselines from the fetched npz (HJD spans).

    Rows with a non-finite hjd/mag/magerr are dropped (the ZTF LC API very
    occasionally returns a null HJD; 1 epoch in this sample) -- the identical
    mask is applied by run_e4.py when loading data.
    """
    d = np.load(os.path.join(DATA_DIR, lc_file))
    out, all_t = {}, []
    for band in ("g", "r"):
        key = "%s_hjd" % band
        if key not in d.files:
            continue
        t = np.asarray(d[key], dtype=float)
        good = (np.isfinite(t) & np.isfinite(d["%s_mag" % band])
                & np.isfinite(d["%s_magerr" % band]))
        t = t[good]
        out[band] = {"n_epochs": int(t.size),
                     "n_dropped_nonfinite": int((~good).sum()),
                     "T_days": float(t.max() - t.min())}
        all_t.append(t)
    tt = np.concatenate(all_t)
    out["joint"] = {"n_epochs": int(tt.size),
                    "T_days": float(tt.max() - tt.min())}
    return out


def _snap_grid(T):
    """NFFT-compliant grid at target df = 0.2/T.

    The FTP fast path requires F_MIN to be an integer multiple of df, so df is
    snapped DOWN to F_MIN / ceil(F_MIN / (0.2/T)) (slightly finer than the
    0.2/T target -> oversampling >= 5 per Rayleigh is preserved; actual df and
    points-per-Rayleigh are recorded in every artifact)."""
    df0 = DF_FACTOR / T
    n0 = int(np.ceil(F_MIN / df0))
    df = F_MIN / n0
    n_freq = int(np.floor((F_MAX - F_MIN) / df)) + 1
    return df, n0, n_freq


def grid_params(stats):
    """E4 grid: df ~= 0.2/T per band (NFFT-snapped, see _snap_grid); two-band
    methods use the joint grid at the SHORTER per-band baseline.  Explicit
    [F_MIN, F_MAX]."""
    g = {}
    for band in ("g", "r"):
        T = stats[band]["T_days"]
        df, n0, n_freq = _snap_grid(T)
        g[band] = {"df": df, "i_min": n0, "n_freq": n_freq, "T_days": T,
                   "pts_per_rayleigh": (1.0 / T) / df}
    T_short = min(stats["g"]["T_days"], stats["r"]["T_days"])
    df, n0, n_freq = _snap_grid(T_short)
    T_joint = stats["joint"]["T_days"]
    g["joint"] = {"df": df, "i_min": n0, "n_freq": n_freq, "T_days": T_short,
                  "pts_per_rayleigh": (1.0 / T_joint) / df}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="skip TAP; use cached gaia_xmatch/ or fetch raw/ CSVs")
    args = ap.parse_args()
    os.makedirs(XMATCH_DIR, exist_ok=True)

    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    entries = manifest["stars"]

    # -- dedupe by oid pair --------------------------------------------------
    by_oid = {}
    for e in entries:
        by_oid.setdefault(oid_key(e), []).append(e)
    assert len(by_oid) == 73, "expected 73 unique oid pairs, got %d" % len(by_oid)

    stars = []
    for key, group in by_oid.items():
        sci = [e for e in group if not e["control"]]
        ctl = [e for e in group if e["control"]]
        assert len(group) <= 2 and len(sci) <= 1 and len(ctl) <= 1
        base = sci[0] if sci else ctl[0]
        rec = {
            "uid": base["star_id"],
            "field": base["field"],
            "ra": base["ra"], "dec": base["dec"],
            "type": base["type"],
            "oids": {b: v["oid"] for b, v in base["bands"].items()},
            "lc_file": base["lc_file"],
            "dual_truth": bool(sci and ctl),
            "chen_period": None, "gaia_period": None,
            "gaia_period_source": None, "gaia_source_id": None,
            "gaia_xmatch_sep_arcsec": None,
        }
        if sci:
            rec["role"] = "science"
            assert sci[0]["period_source"] == "chen2020_Per"
            rec["chen_period"] = float(sci[0]["period_truth_days"])
        else:
            rec["role"] = ("control_antijoin" if ctl[0].get("control_antijoin")
                           else "control_orig")
        if ctl:
            rec["gaia_period"] = float(ctl[0]["period_truth_days"])
            rec["gaia_period_source"] = ctl[0]["period_source"]
            rec["gaia_source_id"] = ctl[0]["catalog_id"]
            rec["gaia_type"] = ctl[0]["type"]
            if ctl[0].get("chen_per") is not None:      # dual-truth cross-ref
                rec["chen_per_from_control_entry"] = float(ctl[0]["chen_per"])
            rec["also_manifest_id"] = ctl[0]["star_id"] if sci else None
        stats = lc_stats(base["lc_file"])
        rec["bands"] = {b: stats[b] for b in ("g", "r") if b in stats}
        rec["joint"] = stats["joint"]
        rec["grid"] = grid_params(stats)
        # primary band for single-band methods: more epochs, ties -> r
        ng, nr = stats["g"]["n_epochs"], stats["r"]["n_epochs"]
        rec["primary_band"] = "g" if ng > nr else "r"
        stars.append(rec)

    # -- independent Gaia periods for the science stars (<= 3 TAP requests) --
    science = [s for s in stars if s["role"] == "science"]
    for field in sorted({s["field"] for s in science}):
        gaia = fetch_gaia_field(field, args.offline)
        print("field %d: %d Gaia SOS RRab/RRc with periods" % (field, len(gaia)),
              flush=True)
        for s in (x for x in science if x["field"] == field):
            best, best_sep = None, XMATCH_RADIUS_ARCSEC
            for g in gaia:
                d = sep_arcsec(s["ra"], s["dec"], g["ra"], g["dec"])
                if d < best_sep:
                    best, best_sep = g, d
            if best is not None:
                # dual-truth stars already carry the control entry's Gaia
                # period; the fresh xmatch must agree (same catalog).
                if s["gaia_period"] is not None:
                    assert abs(best["per"] - s["gaia_period"]) < 1e-9, \
                        "xmatch/control Gaia period mismatch for %s" % s["uid"]
                s["gaia_period"] = best["per"]
                s["gaia_period_source"] = best["per_source"]
                s["gaia_source_id"] = best["source_id"]
                s["gaia_type"] = best["type"]
                s["gaia_xmatch_sep_arcsec"] = round(best_sep, 3)

    # -- summary -------------------------------------------------------------
    n_sci = len(science)
    n_aj = sum(1 for s in stars if s["role"] == "control_antijoin")
    n_orig = sum(1 for s in stars if s["role"] == "control_orig")
    n_dual = sum(1 for s in stars if s["dual_truth"])
    n_gaia_sci = sum(1 for s in science if s["gaia_period"] is not None)
    dual_rel = [abs(s["gaia_period"] - s["chen_period"]) / s["chen_period"]
                for s in stars
                if s["dual_truth"] and s["gaia_period"] and s["chen_period"]]
    n_freq_joint = sorted(s["grid"]["joint"]["n_freq"] for s in stars)
    df_joint = sorted(s["grid"]["joint"]["df"] for s in stars)

    table = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": {
            "f_min_cpd": F_MIN, "f_max_cpd": F_MAX, "df_factor": DF_FACTOR,
            "grid_rule": ("df = F_MIN/ceil(F_MIN/(0.2/T)) per band (NFFT-"
                          "snapped, <= 0.2/T); two-band methods use the "
                          "joint grid at the shorter per-band T; explicit "
                          "[f_min, f_max], never nyquist_factor; freqs = "
                          "df * (i_min + arange(n_freq))"),
            "xmatch_radius_arcsec": XMATCH_RADIUS_ARCSEC,
            "gaia_query": "vari_rrlyrae JOIN gaia_source_lite, ESA sync TAP, "
                          "per-field box (pattern of fetch_rrl_sample.py)",
            "time_column": "hjd",
        },
        "counts": {
            "n_unique_stars": len(stars), "n_science": n_sci,
            "n_control_antijoin": n_aj, "n_control_orig": n_orig,
            "n_dual_truth": n_dual, "n_gaia_periods_science": n_gaia_sci,
        },
        "dual_truth_consistency": {
            "n": len(dual_rel),
            "max_rel_period_diff": max(dual_rel) if dual_rel else None,
            "median_rel_period_diff": (float(np.median(dual_rel))
                                       if dual_rel else None),
        },
        "grid_summary": {
            "median_nfreq_joint": int(np.median(n_freq_joint)),
            "nfreq_joint_range": [n_freq_joint[0], n_freq_joint[-1]],
            "df_joint_range": [df_joint[0], df_joint[-1]],
        },
        "http": dict(STATS),
        "stars": sorted(stars, key=lambda s: (s["field"], s["uid"])),
    }
    with open(OUT_JSON + ".tmp", "w") as fh:
        json.dump(table, fh, indent=1)
    os.replace(OUT_JSON + ".tmp", OUT_JSON)

    print(json.dumps({k: table[k] for k in
                      ("counts", "dual_truth_consistency", "grid_summary",
                       "http")}, indent=1))
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
