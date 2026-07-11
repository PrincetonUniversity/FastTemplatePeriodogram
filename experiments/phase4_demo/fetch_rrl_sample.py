#!/usr/bin/env python
"""Phase-4 RRL science-sample acquisition (approved spec, John 2026-07-10).

Builds a labeled same-star g+r ZTF RRL sample for the Phase-4 demo:

  * SCIENCE: ~50-60 RRL from Chen+2020 (VizieR J/ApJS/249/18/table2; ZTF-native
    positions, 'Per' = period truth, Type 'RR'=RRab / 'RRc'), balanced RRab/RRc
    per field as far as each field allows.  Field targets: 486->22, 686->18,
    786->18 (786 has only ~27 Chen RRL; take what passes cuts).
  * 686 SANITY FIRST: 2-star live g+r crossmatch in field 686 (b=+2.8,
    E(B-V)=2.12).  If both sanity stars fail same-star g+r within 1", the 686
    slots shift to 486 and the reason is recorded.
  * CONTROL: 5 per field (10-15 total) Gaia DR3 SOS RRL (gaiadr3.vari_rrlyrae
    JOIN gaiadr3.gaia_source_lite, ESA TAP sync; the full gaia_source join
    hits the server-side 60 s sync abort) inside the same footprints, fetched
    IDENTICALLY (same IRSA TAP crossmatch + LC API + cuts), flagged
    "control": true, with Gaia pf/p1_o as independent period truth.
    Overlap with Chen allowed but flagged per star.

Same-star requirement: both zg and zr objects in ztf_objects_dr23 within
1.0 arcsec of the catalog position (prefer in-field oid, then max ngoodobsrel
-- recipe from e2_xmatch_probe.py).  Light curves via the IRSA
nph_light_curves API with BAD_CATFLAGS_MASK=65535 (server keeps only
catflags==0 epochs; re-verified client-side).  Quality gate: >=40 catflags==0
epochs in BOTH bands; failures promote the next deterministic alternate from
the same field (same type first, then the other type).

Deterministic selection rule (no hand-picking, no RNG): within each field and
type, candidates are ordered brightest-first by catalog r/G magnitude
(ties -> ascending RA).  Controls: brightest-first by phot_g_mean_mag.

Politeness / budget: <=1 HTTP request/s (global throttle), explicit timeouts,
2 retries max with backoff.  Hard stop at 480 requests or 90 MB downloaded
(spec ceiling: ~500 requests / ~100 MB).  RESUMABLE: every raw HTTP payload
(catalog CSV, crossmatch JSON, LC CSV) is cached under the data dir and reused
if present and parseable, so a rerun refetches nothing already on disk.

Outputs:
  ~/.ftperiodogram_data/phase4_rrl_sample/lc/<star_id>.npz   (LC data; NOT git)
  ~/.ftperiodogram_data/phase4_rrl_sample/raw/               (resume cache)
  experiments/phase4_demo/manifest.json                      (committed)
  experiments/phase4_demo/SUMMARY.md                         (committed)

Run:  FastTemplatePeriodogram/.venv/bin/python fetch_rrl_sample.py
(stdlib urllib + numpy only; no astropy/astroquery.)
"""
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

# ---------------------------------------------------------------- endpoints
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
LC_API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

# ---------------------------------------------------------------- spec knobs
FIELD_CENTERS = {486: (274.351, 4.550), 686: (298.269, 33.350),
                 786: (147.601, 54.950)}
HALF_DEG = 3.43                      # field-footprint half-width (E2 recipe)
SCIENCE_TARGET = {486: 22, 686: 18, 786: 18}   # 58 total (50-60 approved)
CONTROL_TARGET = {486: 5, 686: 5, 786: 5}      # 15 total (10-15 approved)
XMATCH_RADIUS_ARCSEC = 1.0
MIN_EPOCHS = 40                      # per band, catflags==0
CHEN_OVERLAP_ARCSEC = 1.5            # control-vs-Chen overlap flag radius
REQUEST_BUDGET = 480
BYTE_BUDGET = 90 * 1024 * 1024

# --topup-controls knobs (2026-07-10; see main_topup docstring for WHY)
TOPUP_ANTIJOIN_ARCSEC = 2.0          # "Chen-missed" = no Chen RRL within this
TOPUP_TARGET = {486: 5, 686: 6, 786: 4}   # ~10-12 new; 786 pool is thin
# (686 gets the largest share: its anti-join pool is by far the deepest --
#  824 Chen-missed Gaia RRL, i.e. the strongest Chen-selection signal.)
TOPUP_REQUEST_BUDGET = 150

DATA_DIR = os.path.expanduser("~/.ftperiodogram_data/phase4_rrl_sample")
RAW_DIR = os.path.join(DATA_DIR, "raw")
LC_DIR = os.path.join(DATA_DIR, "lc")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

STATS = {"requests": 0, "bytes": 0, "cache_hits": 0}
_LAST_REQ = [0.0]


# ---------------------------------------------------------------- HTTP layer
def _http(url, post_data=None, timeout=60):
    """Throttled (<=1 req/s), budgeted, retry-x2 HTTP fetch -> text."""
    if STATS["requests"] >= REQUEST_BUDGET:
        raise RuntimeError("HARD STOP: request budget (%d) exhausted"
                           % REQUEST_BUDGET)
    if STATS["bytes"] >= BYTE_BUDGET:
        raise RuntimeError("HARD STOP: byte budget (%d MB) exhausted"
                           % (BYTE_BUDGET // 2**20))
    last_err = None
    for attempt in range(3):                       # 1 try + 2 retries
        wait = 1.0 - (time.time() - _LAST_REQ[0])  # sustain <=1 req/s
        if wait > 0:
            time.sleep(wait)
        _LAST_REQ[0] = time.time()
        STATS["requests"] += 1
        try:
            req = urllib.request.Request(
                url, data=post_data,
                headers={"User-Agent": "ftp-phase4-rrl-fetch (johnh2o2)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            STATS["bytes"] += len(body)
            return body.decode("utf-8", "replace")
        except Exception as exc:                   # noqa: BLE001
            last_err = exc
            if attempt < 2:
                time.sleep(2.0 + 4.0 * attempt)    # backoff 2s, 6s
    raise RuntimeError("HTTP failed after 3 attempts: %s (%s)"
                       % (url.split("?")[0], str(last_err)[:200]))


def tap_query(url, adql, timeout=90):
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": adql}).encode()
    return _http(url, post_data=data, timeout=timeout)


def _cached_text(cache_path, fetch_fn):
    """Resume support: reuse cache_path if present/non-empty, else fetch
    atomically (tmp+rename) so a killed run never leaves a truncated cache."""
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


# ---------------------------------------------------------------- geometry
def field_box(field):
    rc, dc = FIELD_CENTERS[field]
    dra = HALF_DEG / math.cos(math.radians(dc))
    return rc - dra, rc + dra, dc - HALF_DEG, dc + HALF_DEG


def sep_arcsec(ra1, dec1, ra2, dec2):
    cd = math.cos(math.radians(dec1))
    return math.hypot((ra1 - ra2) * cd, dec1 - dec2) * 3600.0


def ffloat(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- catalogs
def fetch_chen_field(field):
    """All Chen+2020 RRL (Type RR/RRc) inside the field box, via VizieR TAP."""
    r0, r1, d0, d1 = field_box(field)
    adql = ('SELECT * FROM "J/ApJS/249/18/table2" '
            "WHERE Type LIKE 'RR%%' "
            "AND RAJ2000 BETWEEN %.6f AND %.6f "
            "AND DEJ2000 BETWEEN %.6f AND %.6f" % (r0, r1, d0, d1))
    text = _cached_text(os.path.join(RAW_DIR, "chen_field%d.csv" % field),
                        lambda: tap_query(VIZIER_TAP, adql))
    rows = list(csv.DictReader(io.StringIO(text)))
    for r in rows:
        r["Type"] = (r.get("Type") or "").strip()
    rows = [r for r in rows if r["Type"] in ("RR", "RRc")]
    id_col = next((c for c in ("ID", "Star", "SourceID", "Source", "ZTF",
                               "recno") if rows and c in rows[0]), None)
    for r in rows:
        r["_field"] = field
        r["_ra"] = ffloat(r.get("RAJ2000"))
        r["_dec"] = ffloat(r.get("DEJ2000"))
        r["_rmag"] = ffloat(r.get("rmag"), 99.0)
        r["_per"] = ffloat(r.get("Per"))
        r["_type"] = "RRab" if r["Type"] == "RR" else "RRc"
        r["_id"] = (r.get(id_col) or "").strip() if id_col else ""
    rows = [r for r in rows
            if r["_ra"] is not None and r["_dec"] is not None
            and r["_per"] is not None]
    return rows, id_col


def fetch_gaia_field(field):
    """Gaia DR3 SOS RRL (vari_rrlyrae x gaia_source_lite) inside the field box.

    NOTE: joining against full gaiadr3.gaia_source deterministically hits the
    ESA sync-TAP 60 s server-side abort (HTTP 408 "Job timeout/aborted") for
    these ~47 deg^2 boxes; gaia_source_lite carries the same needed columns
    (ra, dec, phot_g_mean_mag) and the join completes in ~30 s (verified
    2026-07-10)."""
    r0, r1, d0, d1 = field_box(field)
    adql = ("SELECT v.source_id, g.ra, g.dec, v.pf, v.p1_o, "
            "v.best_classification, g.phot_g_mean_mag "
            "FROM gaiadr3.vari_rrlyrae AS v "
            "JOIN gaiadr3.gaia_source_lite AS g ON g.source_id = v.source_id "
            "WHERE g.ra BETWEEN %.6f AND %.6f "
            "AND g.dec BETWEEN %.6f AND %.6f" % (r0, r1, d0, d1))
    text = _cached_text(os.path.join(RAW_DIR, "gaia_field%d.csv" % field),
                        lambda: tap_query(GAIA_TAP, adql))
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        cls = (r.get("best_classification") or "").strip()
        if cls not in ("RRab", "RRc"):
            continue
        pf, p1o = ffloat(r.get("pf")), ffloat(r.get("p1_o"))
        if cls == "RRab":
            per, psrc = (pf, "gaia_dr3_pf") if pf else (p1o, "gaia_dr3_p1_o")
        else:
            per, psrc = (p1o, "gaia_dr3_p1_o") if p1o else (pf, "gaia_dr3_pf")
        if per is None:
            continue
        out.append({"_field": field, "_ra": ffloat(r.get("ra")),
                    "_dec": ffloat(r.get("dec")), "_per": per,
                    "_per_source": psrc, "_type": cls,
                    "_gmag": ffloat(r.get("phot_g_mean_mag"), 99.0),
                    "_id": r.get("source_id", "").strip()})
    return [r for r in out if r["_ra"] is not None and r["_dec"] is not None]


# ---------------------------------------------------------------- crossmatch
def xmatch_gr(ra, dec, field):
    """Same-star g+r match to ztf_objects_dr23 within 1" (single TAP query;
    both bands).  Prefer in-field oid, then max ngoodobsrel (E2 recipe)."""
    cache = os.path.join(RAW_DIR, "xm_%.6f_%+.6f.json" % (ra, dec))
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        STATS["cache_hits"] += 1
        with open(cache) as fh:
            return json.load(fh)
    adql = ("SELECT oid,filtercode,ngoodobsrel,medianmag,ra,dec "
            "FROM ztf_objects_dr23 "
            "WHERE CONTAINS(POINT('ICRS',ra,dec),"
            "CIRCLE('ICRS',%.6f,%.6f,%.7f))=1"
            % (ra, dec, XMATCH_RADIUS_ARCSEC / 3600.0))
    rows = list(csv.DictReader(io.StringIO(
        tap_query(IRSA_TAP, adql, timeout=90))))
    best = {}
    for band, fc in (("g", "zg"), ("r", "zr")):
        cand = [o for o in rows if (o.get("filtercode") or "").strip() == fc]
        for o in cand:
            o["sep_arcsec"] = round(sep_arcsec(ra, dec, float(o["ra"]),
                                               float(o["dec"])), 3)
            o["in_field"] = o["oid"].startswith(str(field))
        cand.sort(key=lambda o: (not o["in_field"],
                                 -int(float(o["ngoodobsrel"]))))
        best[band] = ({"oid": cand[0]["oid"],
                       "sep_arcsec": cand[0]["sep_arcsec"],
                       "ngoodobsrel": int(float(cand[0]["ngoodobsrel"])),
                       "medianmag": ffloat(cand[0]["medianmag"]),
                       "in_field": cand[0]["in_field"]}
                      if cand else None)
    tmp = cache + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(best, fh)
    os.replace(tmp, cache)
    return best


# ---------------------------------------------------------------- LC fetch
LC_COLS = ("oid", "hjd", "mjd", "mag", "magerr", "catflags", "filtercode")


def _lc_url(oids):
    return LC_API + "?" + urllib.parse.urlencode(
        {"ID": ",".join(str(o) for o in oids), "FORMAT": "csv",
         "BAD_CATFLAGS_MASK": "65535"})


def fetch_lc_pair(g_oid, r_oid):
    """catflags==0 epochs for both oids via nph_light_curves (one request for
    the pair; per-oid fallback if the joint reply misses a band).
    Returns {band: list[(mjd, hjd, mag, magerr)]}."""
    text = _cached_text(os.path.join(RAW_DIR, "lc_%s_%s.csv" % (g_oid, r_oid)),
                        lambda: _http(_lc_url([g_oid, r_oid]), timeout=120))

    def parse(txt, keep_oids):
        eps = {str(o): [] for o in keep_oids}
        for row in csv.DictReader(io.StringIO(txt)):
            oid = (row.get("oid") or "").strip()
            if oid not in eps:
                continue
            if int(float(row.get("catflags") or -1)) != 0:
                continue        # belt-and-braces; mask already applied
            vals = [ffloat(row.get(k)) for k in ("mjd", "hjd", "mag",
                                                 "magerr")]
            if any(v is None for v in vals):
                continue
            eps[oid].append(tuple(vals))
        return eps

    eps = parse(text, [g_oid, r_oid])
    for oid in (g_oid, r_oid):                       # per-oid fallback
        if not eps[str(oid)]:
            solo = _cached_text(os.path.join(RAW_DIR, "lc_%s.csv" % oid),
                                lambda o=oid: _http(_lc_url([o]), timeout=120))
            eps[str(oid)] = parse(solo, [oid])[str(oid)]
    return {"g": eps[str(g_oid)], "r": eps[str(r_oid)]}


# ---------------------------------------------------------------- per star
def make_star_id(prefix, field, ident, ra, dec):
    if ident:
        clean = "".join(ch for ch in str(ident) if ch.isalnum() or ch in "+-._")
        return "%s_f%d_%s" % (prefix, field, clean)
    return "%s_f%d_%.5f%+.5f" % (prefix, field, ra, dec)


def process_star(cand, star_id):
    """Crossmatch -> LC fetch -> cuts -> npz.

    Returns (ok, record|reason, stage) with stage in {"ok", "xmatch",
    "prescreen", "epochs"} naming the pipeline stage that failed (or "ok").
    """
    ra, dec, field = cand["_ra"], cand["_dec"], cand["_field"]
    best = xmatch_gr(ra, dec, field)
    if not (best.get("g") and best.get("r")):
        missing = [b for b in ("g", "r") if not best.get(b)]
        return False, "no %s match within %.1f arcsec" % (
            "+".join(missing), XMATCH_RADIUS_ARCSEC), "xmatch"
    if any(best[b]["ngoodobsrel"] < MIN_EPOCHS for b in ("g", "r")):
        return False, ("prescreen ngoodobsrel g=%d r=%d < %d"
                       % (best["g"]["ngoodobsrel"], best["r"]["ngoodobsrel"],
                          MIN_EPOCHS)), "prescreen"
    lc = fetch_lc_pair(best["g"]["oid"], best["r"]["oid"])
    ngood = {b: len(lc[b]) for b in ("g", "r")}
    if any(ngood[b] < MIN_EPOCHS for b in ("g", "r")):
        return False, ("good epochs g=%d r=%d < %d after catflags==0 cut"
                       % (ngood["g"], ngood["r"], MIN_EPOCHS)), "epochs"
    arrays = {}
    for b in ("g", "r"):
        arr = np.asarray(sorted(lc[b]), dtype=np.float64)
        arrays["%s_mjd" % b] = arr[:, 0]
        arrays["%s_hjd" % b] = arr[:, 1]
        arrays["%s_mag" % b] = arr[:, 2]
        arrays["%s_magerr" % b] = arr[:, 3]
    lc_path = os.path.join(LC_DIR, star_id + ".npz")
    np.savez_compressed(lc_path, **arrays)
    record = {
        "star_id": star_id, "field": field, "ra": ra, "dec": dec,
        "type": cand["_type"], "period_truth_days": cand["_per"],
        "bands": {b: {"oid": best[b]["oid"],
                      "sep_arcsec": best[b]["sep_arcsec"],
                      "ngoodobs": ngood[b],
                      "ngoodobsrel_dr23": best[b]["ngoodobsrel"],
                      "medianmag": best[b]["medianmag"],
                      "oid_in_field": best[b]["in_field"]}
                  for b in ("g", "r")},
        "lc_file": os.path.relpath(lc_path, DATA_DIR),
    }
    return True, record, "ok"


# ---------------------------------------------------------------- selection
def run_selection(cands_by_type, quota_by_type, field, id_prefix,
                  decorate, rejects, promoted_counter):
    """Deterministic quota fill with promotion: per type, brightest-first;
    leftover quota spills to the other type's remaining candidates."""
    picked = []
    queues = {t: list(v) for t, v in cands_by_type.items()}
    quota = dict(quota_by_type)

    def attempt(cand, t):
        sid = make_star_id(id_prefix, field, cand["_id"],
                           cand["_ra"], cand["_dec"])
        print("  [%s f%d %s] %s ra=%.5f dec=%.5f Per=%.5f ..."
              % (id_prefix, field, t, sid, cand["_ra"], cand["_dec"],
                 cand["_per"]), flush=True)
        ok, payload, _stage = process_star(cand, sid)
        if ok:
            decorate(payload, cand)
            picked.append(payload)
            print("      OK  ng=%d nr=%d sep g=%.2f\" r=%.2f\""
                  % (payload["bands"]["g"]["ngoodobs"],
                     payload["bands"]["r"]["ngoodobs"],
                     payload["bands"]["g"]["sep_arcsec"],
                     payload["bands"]["r"]["sep_arcsec"]), flush=True)
            return True
        rejects.append({"star_id": sid, "field": field, "type": t,
                        "ra": cand["_ra"], "dec": cand["_dec"],
                        "reason": payload})
        print("      REJECT: %s" % payload, flush=True)
        return False

    for t in sorted(quota):                          # pass 1: per type
        while quota[t] > 0 and queues[t]:
            if attempt(queues[t].pop(0), t):
                quota[t] -= 1
            else:
                promoted_counter[0] += 1
    spill = sum(quota.values())                      # pass 2: cross-type spill
    if spill > 0:
        rest = sorted([(c, t) for t, q in queues.items() for c in q],
                      key=lambda ct: (ct[0].get("_rmag",
                                                ct[0].get("_gmag", 99.0)),
                                      ct[0]["_ra"]))
        while spill > 0 and rest:
            cand, t = rest.pop(0)
            if attempt(cand, t):
                spill -= 1
            else:
                promoted_counter[0] += 1
    return picked, spill


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    for d in (DATA_DIR, RAW_DIR, LC_DIR):
        os.makedirs(d, exist_ok=True)
    problems = []

    # ---- (a) Chen+2020 RRL per field ------------------------------------
    print("=== Chen+2020 RRL per field (VizieR TAP) ===", flush=True)
    chen, chen_id_col = {}, None
    for field in FIELD_CENTERS:
        chen[field], idc = fetch_chen_field(field)
        chen_id_col = chen_id_col or idc
        nab = sum(1 for r in chen[field] if r["_type"] == "RRab")
        print("field %d: %d RRL (RRab %d / RRc %d)"
              % (field, len(chen[field]), nab, len(chen[field]) - nab),
              flush=True)

    # ---- (b) 686 sanity crossmatch FIRST ---------------------------------
    print("\n=== 686 sanity crossmatch (2 brightest RRL, 1\") ===", flush=True)
    sanity686 = {"stars": [], "passed": None}
    for cand in sorted(chen[686], key=lambda r: (r["_rmag"], r["_ra"]))[:2]:
        best = xmatch_gr(cand["_ra"], cand["_dec"], 686)
        ok = bool(best.get("g") and best.get("r"))
        sanity686["stars"].append(
            {"ra": cand["_ra"], "dec": cand["_dec"], "rmag": cand["_rmag"],
             "same_star_gr": ok,
             "g": best.get("g"), "r": best.get("r")})
        print("  %.5f %+.5f rmag=%.2f -> same-star g+r: %s"
              % (cand["_ra"], cand["_dec"], cand["_rmag"], ok), flush=True)
    sanity686["passed"] = any(s["same_star_gr"] for s in sanity686["stars"])
    science_target = dict(SCIENCE_TARGET)
    if not sanity686["passed"]:
        science_target[486] += science_target[686]
        science_target[686] = 0
        msg = ("686 sanity crossmatch FAILED (0/2 same-star g+r within 1\"); "
               "686 slots shifted to 486 per spec")
        problems.append(msg)
        print("  ! " + msg, flush=True)
    else:
        print("  686 sanity PASSED (%d/2)"
              % sum(s["same_star_gr"] for s in sanity686["stars"]), flush=True)

    # ---- (c-f) science selection + fetch ---------------------------------
    rejects, science, unfilled = [], [], {}
    promoted = [0]
    for field in FIELD_CENTERS:
        n = science_target[field]
        if n == 0:
            continue
        by_type = {}
        for t in ("RRab", "RRc"):
            by_type[t] = sorted((r for r in chen[field] if r["_type"] == t),
                                key=lambda r: (r["_rmag"], r["_ra"]))
        # balanced split, clamped to availability (spill handles shortfall)
        quota = {"RRab": min((n + 1) // 2, len(by_type["RRab"])),
                 "RRc": min(n // 2, len(by_type["RRc"]))}
        short = n - sum(quota.values())
        for t in ("RRab", "RRc"):
            if short <= 0:
                break
            extra = min(short, len(by_type[t]) - quota[t])
            quota[t] += extra
            short -= extra
        print("\n=== SCIENCE field %d: target %d (RRab %d / RRc %d) ==="
              % (field, n, quota["RRab"], quota["RRc"]), flush=True)

        def deco(rec, cand):
            rec.update({"catalog": "chen2020", "control": False,
                        "period_source": "chen2020_Per",
                        "catalog_id": cand["_id"],
                        "gmag": ffloat(cand.get("gmag")),
                        "rmag": ffloat(cand.get("rmag"))})
        got, left = run_selection(by_type, quota, field, "chen",
                                  deco, rejects, promoted)
        science.extend(got)
        if left:
            unfilled[field] = left
            problems.append("field %d science quota unfilled by %d "
                            "(candidates exhausted)" % (field, left))

    # ---- (g) Gaia DR3 SOS controls, identical path -----------------------
    print("\n=== CONTROL: Gaia DR3 SOS RRL (ESA TAP) ===", flush=True)
    controls = []
    control_target = dict(CONTROL_TARGET)
    if not sanity686["passed"]:
        control_target[486] += control_target[686]
        control_target[686] = 0
    for field in FIELD_CENTERS:
        n = control_target[field]
        if n == 0:
            continue
        try:
            gaia = fetch_gaia_field(field)
        except RuntimeError as exc:
            problems.append("Gaia TAP failed for field %d: %s"
                            % (field, str(exc)[:150]))
            print("  ! Gaia TAP failed for field %d" % field, flush=True)
            continue
        gaia.sort(key=lambda r: (r["_gmag"], r["_ra"]))
        print("field %d: %d Gaia SOS RRab/RRc candidates -> target %d"
              % (field, len(gaia), n), flush=True)

        def deco_ctl(rec, cand, fld=field):
            near = min(((sep_arcsec(cand["_ra"], cand["_dec"],
                                    c["_ra"], c["_dec"]), c)
                        for c in chen[fld]), default=(1e9, None),
                       key=lambda x: x[0])
            overlap = near[0] <= CHEN_OVERLAP_ARCSEC
            rec.update({"catalog": "gaia_dr3_sos", "control": True,
                        "period_source": cand["_per_source"],
                        "catalog_id": cand["_id"],
                        "phot_g_mean_mag": cand["_gmag"],
                        "chen_overlap": overlap,
                        "chen_sep_arcsec": (round(near[0], 3)
                                            if overlap else None),
                        "chen_per": near[1]["_per"] if overlap else None})
        got, left = run_selection({"any": gaia}, {"any": n}, field, "gaia",
                                  deco_ctl, rejects, promoted)
        controls.extend(got)
        if left:
            problems.append("field %d control quota unfilled by %d"
                            % (field, left))

    # ---- (h) manifest + summary ------------------------------------------
    wall_s = time.time() - t_start
    lc_bytes = sum(os.path.getsize(os.path.join(LC_DIR, f))
                   for f in os.listdir(LC_DIR))
    per_field = {}
    for field in FIELD_CENTERS:
        sf = [s for s in science if s["field"] == field]
        cf = [c for c in controls if c["field"] == field]
        per_field[field] = {
            "science": len(sf),
            "RRab": sum(1 for s in sf if s["type"] == "RRab"),
            "RRc": sum(1 for s in sf if s["type"] == "RRc"),
            "control": len(cf),
        }
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": {
            "science_catalog": "Chen+2020 VizieR J/ApJS/249/18/table2",
            "control_catalog": "Gaia DR3 SOS gaiadr3.vari_rrlyrae",
            "field_half_width_deg": HALF_DEG,
            "field_centers": {str(k): v for k, v in FIELD_CENTERS.items()},
            "xmatch_radius_arcsec": XMATCH_RADIUS_ARCSEC,
            "min_good_epochs_per_band": MIN_EPOCHS,
            "catflags_cut": "catflags==0 (BAD_CATFLAGS_MASK=65535)",
            "ztf_release": "ztf_objects_dr23 (LC API default collection)",
            "selection_rule": ("deterministic brightest-first by catalog "
                               "rmag (Chen) / phot_g_mean_mag (Gaia), ties "
                               "by RA; balanced RRab/RRc per field where "
                               "available; failures promote next alternate"),
            "chen_id_column": chen_id_col,
        },
        "sanity_686": sanity686,
        "targets": {"science": {str(k): v for k, v in science_target.items()},
                    "control": {str(k): v for k, v in control_target.items()}},
        "counts": {"science": len(science), "control": len(controls),
                   "rejected": len(rejects),
                   "n_failed_promoted": promoted[0],
                   "per_field": {str(k): v for k, v in per_field.items()}},
        "http": dict(STATS),
        "lc_bytes_on_disk": lc_bytes,
        "wall_time_s": round(wall_s, 1),
        "data_dir": DATA_DIR,
        "stars": science + controls,
        "rejects": rejects,
        "problems": problems,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    lines = [
        "# Phase-4 RRL sample — acquisition summary",
        "",
        "Built %s by `fetch_rrl_sample.py` (approved spec, John 2026-07-10)."
        % manifest["created_utc"],
        "",
        "## Counts",
        "",
        "| Field | Science | RRab | RRc | Control (Gaia SOS) |",
        "|---|---|---|---|---|",
    ]
    for field in FIELD_CENTERS:
        pf = per_field[field]
        lines.append("| %d | %d | %d | %d | %d |"
                     % (field, pf["science"], pf["RRab"], pf["RRc"],
                        pf["control"]))
    lines += [
        "| **total** | **%d** | %d | %d | **%d** |"
        % (len(science), sum(p["RRab"] for p in per_field.values()),
           sum(p["RRc"] for p in per_field.values()), len(controls)),
        "",
        "- 686 sanity crossmatch: **%s**"
        % ("PASSED" if sanity686["passed"] else
           "FAILED -> slots shifted to 486"),
        "- Rejected candidates: %d (see `manifest.json` rejects[]); "
        "failures promoted to alternates: %d"
        % (len(rejects), promoted[0]),
        "- Cuts: same-star g+r within %.1f\", catflags==0 only, >=%d good "
        "epochs/band." % (XMATCH_RADIUS_ARCSEC, MIN_EPOCHS),
        "- Selection: deterministic brightest-first (rmag / G), balanced "
        "RRab/RRc per field where available. No hand-picking, no RNG.",
        "- Controls fetched identically (same TAP xmatch + LC API + cuts); "
        "`\"control\": true` + Gaia pf/p1_o truth; Chen overlap flagged "
        "per star (%d overlap)."
        % sum(1 for c in controls if c.get("chen_overlap")),
        "",
        "## Cost",
        "",
        "- HTTP requests: %d (budget %d); downloaded %.1f MB (budget %d MB); "
        "cache hits %d."
        % (STATS["requests"], REQUEST_BUDGET, STATS["bytes"] / 2**20,
           BYTE_BUDGET // 2**20, STATS["cache_hits"]),
        "- LC data on disk: %.1f MB in `%s/lc/` (NOT committed)."
        % (lc_bytes / 2**20, DATA_DIR),
        "- Wall time: %.0f s." % wall_s,
        "",
    ]
    if problems:
        lines += ["## Problems", ""]
        lines += ["- %s" % p for p in problems]
        lines += [""]
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "FastTemplatePeriodogram/.venv/bin/python \\",
        "    FastTemplatePeriodogram/experiments/phase4_demo/"
        "fetch_rrl_sample.py",
        "```",
        "",
        "Resumable: raw HTTP payloads cached under "
        "`~/.ftperiodogram_data/phase4_rrl_sample/raw/`; a rerun refetches "
        "nothing already on disk. Per-star LCs: "
        "`~/.ftperiodogram_data/phase4_rrl_sample/lc/<star_id>.npz` "
        "(keys `{g,r}_{mjd,hjd,mag,magerr}`).",
        "",
    ]
    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines))

    print("\n=== DONE ===", flush=True)
    print("science=%d control=%d rejects=%d promoted=%d "
          "requests=%d bytes=%.1fMB lc_disk=%.1fMB wall=%.0fs"
          % (len(science), len(controls), len(rejects), promoted[0],
             STATS["requests"], STATS["bytes"] / 2**20, lc_bytes / 2**20,
             wall_s), flush=True)
    for p in problems:
        print("PROBLEM: %s" % p, flush=True)


# ------------------------------------------------------------- control top-up
def _nearest_chen_sep(cand, chen_rows):
    return min((sep_arcsec(cand["_ra"], cand["_dec"], c["_ra"], c["_dec"])
                for c in chen_rows), default=1e9)


def main_topup():
    """--topup-controls: append "Chen-missed" (anti-join) Gaia SOS controls.

    WHY (2026-07-10): 12/15 of the original controls are the SAME physical
    stars as Chen science members -- brightest-first Gaia selection in the
    same fields simply re-found Chen's stars.  The control's purpose is to
    measure the Chen-selection effect, which needs stars Gaia found but Chen
    did NOT.  This mode selects additional controls from the cached per-field
    Gaia SOS tables keeping only candidates with NO Chen+2020 RRL within
    TOPUP_ANTIJOIN_ARCSEC (anti-join against the FULL per-field Chen tables,
    not just sampled stars), brightest-first by G, then runs the identical
    xmatch/LC/cuts pipeline.  New entries carry "control": true,
    "chen_overlap": false, "control_antijoin": true.

    SELECTION-FUNCTION RECORD: the per-field funnel (anti-join pool size /
    attempted / pass g+r xmatch / pass epoch cut / fetched) is itself a
    measurement -- if Chen-missed stars mostly fail ZTF quality cuts, that
    quantifies the Chen selection function -- and is written to
    manifest.json ("control_topup") and SUMMARY.md.  Few passes = a finding,
    not a failure.

    Resumable like the main mode (raw payload cache; killed runs refetch
    nothing).  NOTE: a rerun AFTER a completed top-up would select the
    next-brightest candidates (i.e. top up again); the previous funnel is
    preserved under "control_topup_history".
    """
    global REQUEST_BUDGET
    REQUEST_BUDGET = TOPUP_REQUEST_BUDGET
    t_start = time.time()
    for d in (DATA_DIR, RAW_DIR, LC_DIR):
        os.makedirs(d, exist_ok=True)
    man_path = os.path.join(OUT_DIR, "manifest.json")
    with open(man_path) as fh:
        man = json.load(fh)
    have_ids = {s["catalog_id"] for s in man["stars"]}

    funnel, new_stars, rejects = {}, [], []
    for field in FIELD_CENTERS:
        chen_rows, _ = fetch_chen_field(field)     # cached: 0 new requests
        gaia = fetch_gaia_field(field)             # cached: 0 new requests
        pool = [c for c in gaia
                if c["_id"] not in have_ids
                and _nearest_chen_sep(c, chen_rows) > TOPUP_ANTIJOIN_ARCSEC]
        pool.sort(key=lambda c: (c["_gmag"], c["_ra"]))
        fun = {"antijoin_candidates": len(pool), "attempted": 0,
               "pass_xmatch": 0, "pass_epochs": 0, "fetched": 0}
        funnel[str(field)] = fun
        print("\n=== TOPUP field %d: %d anti-join candidates, target %d ==="
              % (field, len(pool), TOPUP_TARGET[field]), flush=True)
        for cand in pool:
            if fun["fetched"] >= TOPUP_TARGET[field]:
                break
            fun["attempted"] += 1
            sid = make_star_id("gaia", field, cand["_id"],
                               cand["_ra"], cand["_dec"])
            print("  [topup f%d %s] %s G=%.2f ra=%.5f dec=%.5f ..."
                  % (field, cand["_type"], sid, cand["_gmag"],
                     cand["_ra"], cand["_dec"]), flush=True)
            ok, payload, stage = process_star(cand, sid)
            if stage != "xmatch":
                fun["pass_xmatch"] += 1
            if ok:
                fun["pass_epochs"] += 1
                fun["fetched"] += 1
                near = _nearest_chen_sep(cand, chen_rows)
                payload.update({
                    "catalog": "gaia_dr3_sos", "control": True,
                    "period_source": cand["_per_source"],
                    "catalog_id": cand["_id"],
                    "phot_g_mean_mag": cand["_gmag"],
                    "chen_overlap": False, "chen_sep_arcsec": None,
                    "chen_per": None, "control_antijoin": True,
                    "chen_nearest_sep_arcsec": (round(near, 3)
                                                if near < 1e8 else None)})
                new_stars.append(payload)
                print("      OK  ng=%d nr=%d sep g=%.2f\" r=%.2f\""
                      % (payload["bands"]["g"]["ngoodobs"],
                         payload["bands"]["r"]["ngoodobs"],
                         payload["bands"]["g"]["sep_arcsec"],
                         payload["bands"]["r"]["sep_arcsec"]), flush=True)
            else:
                rejects.append({"star_id": sid, "field": field,
                                "type": cand["_type"], "ra": cand["_ra"],
                                "dec": cand["_dec"],
                                "phot_g_mean_mag": cand["_gmag"],
                                "stage": stage, "reason": payload})
                print("      REJECT (%s): %s" % (stage, payload), flush=True)

    # ---- manifest update --------------------------------------------------
    man["stars"].extend(new_stars)
    man["counts"]["control"] += len(new_stars)
    man["counts"]["control_antijoin"] = sum(
        1 for s in man["stars"] if s.get("control_antijoin"))
    for f, fun in funnel.items():
        man["counts"]["per_field"][f]["control"] += fun["fetched"]
    topup = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": ("12/15 original controls are chen_overlap (same physical "
                   "stars as science); measuring the Chen-selection effect "
                   "needs Chen-missed stars"),
        "antijoin_radius_arcsec": TOPUP_ANTIJOIN_ARCSEC,
        "selection_rule": ("Gaia SOS RRab/RRc in field box with NO Chen+2020 "
                           "RRL within %.1f arcsec (full per-field Chen "
                           "tables), not already in manifest; brightest-first "
                           "by phot_g_mean_mag, ties by RA; identical "
                           "xmatch/LC/cuts pipeline" % TOPUP_ANTIJOIN_ARCSEC),
        "targets": {str(k): v for k, v in TOPUP_TARGET.items()},
        "funnel": funnel,
        "rejects": rejects,
        "http": dict(STATS),
        "wall_time_s": round(time.time() - t_start, 1),
    }
    if "control_topup" in man:
        man.setdefault("control_topup_history", []).append(
            man["control_topup"])
    man["control_topup"] = topup
    with open(man_path, "w") as fh:
        json.dump(man, fh, indent=1)

    # ---- SUMMARY.md: replace-or-append the top-up section -----------------
    n_entries = len(man["stars"])
    n_unique = len({s["bands"]["g"]["oid"] for s in man["stars"]})
    sum_path = os.path.join(OUT_DIR, "SUMMARY.md")
    with open(sum_path) as fh:
        txt = fh.read()
    marker = "## Control top-up (Chen-missed anti-join)"
    if marker in txt:
        txt = txt[:txt.index(marker)]
    lines = [
        marker,
        "",
        "Added %s by `fetch_rrl_sample.py --topup-controls`."
        % topup["created_utc"],
        "",
        "**Why:** 12 of the 15 original controls are `chen_overlap` — the "
        "same physical stars as science members (brightest-first Gaia "
        "selection in the same fields re-found Chen's stars), so they cannot "
        "measure the Chen-selection effect. This top-up keeps only Gaia SOS "
        "RRL with NO Chen+2020 RRL within %.1f\" (anti-join against the full "
        "per-field Chen tables), brightest-first, through the identical g+r "
        "xmatch / catflags==0 / >=%d-epochs pipeline. New entries carry "
        "`\"control_antijoin\": true`, `\"chen_overlap\": false`."
        % (TOPUP_ANTIJOIN_ARCSEC, MIN_EPOCHS),
        "",
        "**Dedupe note for downstream:** manifest entries are per catalog "
        "row, not per physical star — the original 73 entries span 61 unique "
        "stars (12 controls duplicate science members). Dedupe by ZTF `oid` "
        "before per-star statistics. After this top-up: %d entries, %d "
        "unique stars (+%d new, all non-Chen)."
        % (n_entries, n_unique, len(new_stars)),
        "",
        "### Chen-missed selection funnel (a measurement, not bookkeeping)",
        "",
        "The pass rate of Chen-missed stars through the ZTF quality cuts "
        "quantifies the Chen selection function:",
        "",
        "| Field | Anti-join pool | Attempted | Pass g+r xmatch (1\") "
        "| Pass epoch cut | Fetched |",
        "|---|---|---|---|---|---|",
    ]
    for f in sorted(funnel):
        fun = funnel[f]
        lines.append("| %s | %d | %d | %d | %d | %d |"
                     % (f, fun["antijoin_candidates"], fun["attempted"],
                        fun["pass_xmatch"], fun["pass_epochs"],
                        fun["fetched"]))
    tot = {k: sum(f[k] for f in funnel.values())
           for k in ("antijoin_candidates", "attempted", "pass_xmatch",
                     "pass_epochs", "fetched")}
    lines += [
        "| **total** | **%d** | %d | %d | %d | **%d** |"
        % (tot["antijoin_candidates"], tot["attempted"], tot["pass_xmatch"],
           tot["pass_epochs"], tot["fetched"]),
        "",
        "- \"Attempted\" = brightest-first prefix of the anti-join pool "
        "(until per-field target %s or pool exhausted); pass rates are "
        "measured over that prefix, pool sizes over the whole field."
        % json.dumps({str(k): v for k, v in TOPUP_TARGET.items()}),
        "- \"Pass epoch cut\" = >=%d catflags==0 epochs in BOTH bands "
        "(includes the DR23 `ngoodobsrel` prescreen). Per-candidate failure "
        "stages/reasons: `manifest.json` `control_topup.rejects[]`."
        % MIN_EPOCHS,
        "- Cost this run: %d HTTP requests (budget %d), %.1f MB, "
        "%d cache hits, %.0f s."
        % (STATS["requests"], TOPUP_REQUEST_BUDGET, STATS["bytes"] / 2**20,
           STATS["cache_hits"], topup["wall_time_s"]),
        "",
    ]
    with open(sum_path, "w") as fh:
        fh.write(txt.rstrip() + "\n\n" + "\n".join(lines))

    print("\n=== TOPUP DONE ===", flush=True)
    print("new_controls=%d attempted=%d requests=%d bytes=%.1fMB wall=%.0fs"
          % (len(new_stars), tot["attempted"], STATS["requests"],
             STATS["bytes"] / 2**20, topup["wall_time_s"]), flush=True)
    print("funnel=%s" % json.dumps(funnel), flush=True)


if __name__ == "__main__":
    if "--topup-controls" in sys.argv[1:]:
        sys.exit(main_topup())
    sys.exit(main())
