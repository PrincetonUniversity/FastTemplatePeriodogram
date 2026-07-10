# WP E2 — acquisition probe (bounded) — findings

**Date:** 2026-07-10 (unattended Opus queue). All numbers below were re-measured by
bounded live queries **this session** via `e2_xmatch_probe.py` + one-off `curl`/urllib
probes (venv has no astropy/astroquery/pyvo; raw HTTP TAP + the LC API only). No large
download was performed. **This WP ends in an ESCALATE** per its brief ("ESCALATE with
findings before any large download") — see root `ESCALATE.md`.

## Goal
Characterise how to acquire a **labeled same-star g+r RRL** sample (period ground truth)
in the three cached ZTF fields (486 / 686 / 786), target *tens*, and compare the two
acquisition paths (IRSA TAP retry vs bulk-DR parquet) so John can green-light the build.

## Target fields (ZTF_Fields.txt, fetched this session)
| Field | Center RA,Dec | Gal b | E(B−V) | Character |
|------|---------------|-------|--------|-----------|
| 486 | 274.351, +4.550 | +9.10° | 0.20 | low-lat, moderate density/crowding |
| 686 | 298.269, +33.350 | **+2.77°** | **2.12** | near-plane, heavy extinction + crowding |
| 786 | 147.601, +54.950 | +47.69° | 0.02 | clean high-lat halo, low density |

Cached-object anchors (99-obj cadence sample) sit inside each field but sample only a
single CCD/quadrant patch, not the full ~47 deg² footprint — counts below use a
±3.43°(Dec)/±3.43°/cos(Dec)(RA) box around the true field center (≈ the field footprint,
slightly generous).

## Path A — IRSA TAP retry + LC API (per-object)  → **RECOMMENDED**
- **TAP is back UP.** `ztf_objects_dr23` returns real rows (it was Oracle-`ORA-12541`-down
  at the June cadence fetch; the fetch script fell back to oid-block enumeration then).
  So the "TAP retry" A/B arm is now the live path.
- **Labeled truth = Chen+2020 ZTF periodic-variable catalog** (VizieR `J/ApJS/249/18/table2`;
  reachable this session). ZTF-native ⇒ positions align with ZTF DR objects to sub-arcsec;
  carries `Type` (`RR`=RRab, `RRc`), `Per`, per-band `Per-g`/`Per-r`, `gmag`/`rmag`,
  `Ng`/`Nr` — i.e. period ground truth for recovery scoring, for free.

  RRL counts in the field-footprint boxes (this session):

  | Field | RRab (`RR`) | RRc | Total |
  |------|------|-----|-------|
  | 486 | 275 | 133 | **408** |
  | 686 | 99  | 51  | **150** |
  | 786 | 19  | 8   | **27**  |
  | **all 3** | 393 | 192 | **≈585** |

  (No RRd in these boxes.) Even the sparsest field (786, halo) has 27 — the "tens" target
  is easily met; combined ≈585 is a hundreds-scale ceiling if wanted later.

- **Same-star g+r crossmatch WORKS end-to-end.** 4/4 test RRL (2 in 786, 2 in 486)
  crossmatch to **both** a `zg` and a `zr` object in `ztf_objects_dr23` within 1.5″
  (measured sep 0.05–0.17″), all in the target field, with hundreds of DR23 epochs/band:

  | RRL (fld) | Per (d) | g oid / nobs | r oid / nobs |
  |-----------|---------|--------------|--------------|
  | 786 RRc 152.264 +51.629 | 0.2601 | 786101400001918 / 673 | 786201400003106 / 975 |
  | 786 RRab 147.479 +51.740 | 0.5721 | 786103400001690 / 644 | 786203400002274 / 972 |
  | 486 RRab 273.799 +6.784 | 0.6146 | 486115400029983 / 407 | 486215400035016 / 915 |
  | 486 RRab 272.624 +3.844 | 0.7010 | 486107200033761 / 399 | 486207200036231 / 914 |

  DR23 epoch counts are ~7–20× larger than Chen's DR2-era `Ng`/`Nr` (54–96), and r-band
  carries ~50% more epochs than g (consistent with the README g/r LC totals 1.56B/2.43B).

- **Per-object fetch cost** (`nph_light_curves` by oid, CSV, this session):
  g 902 epochs / 226.5 KB / 0.9 s; r 1206 epochs / 301.1 KB / 1.0 s (raw, all catflags;
  drops to ≈`ngoodobsrel` after catflag=0). ⇒ per RRL (g+r) ≈ 2 fetches, ~0.5 MB, ~2 s.
  A 30–60-RRL labeled sample ≈ 60–120 fetches ≈ **~15–30 MB, a few minutes** — **not a
  large download.**

## Path B — bulk-DR parquet (whole-field)
From `lc_dr23/README.txt` (this session): **7.3 TB total**, 177,218 parquet files across
1,185 field subdirs (one file per field/chip/quadrant/filter) ⇒ **~6.2 GB/field average**;
low-lat crowded fields (e.g. 686 at b=+2.8) are well above average. Whole-field = every
object, not just the ~tens of labeled RRL. **Large download → sign-off required.** Only
worth it if E3's false-alarm run wants *all* objects in a field, or a large (thousands)
RRL sample is later desired. For the bounded labeled-recovery demo it is strictly wasteful.

## Recommendation (for John's sign-off — see ESCALATE.md)
1. **Use Path A.** Build a labeled sample with: Chen+2020 RRL (`Type` ∈ {RR, RRc}, in the
   three field boxes) → crossmatch to `ztf_objects_dr23` for the **target-field, highest-
   `ngoodobsrel`** g and r oids (a position also matches oids in overlapping ZTF fields —
   the probe prefers `oid startswith field` + max nobs; dedup accordingly) → fetch g+r LCs.
2. **Suggested composition:** ~50–60 RRL, balanced RRab/RRc, spread across all three fields
   to span the b/extinction range (786 clean-halo control ≈15–20; 486 ≈20–25; 686
   ≈15–20 with a crowding/extinction caveat). ~15–30 MB, ~few min. No large download.
3. **Do NOT pull bulk parquet** unless E3 needs whole-field FAP over all objects — that is
   the "large download" the E2 brief gates.

## Open design choices for John (these gate E4's <85% recovery ESCALATE)
- Sample size/composition and RRab:RRc balance; per-field balance vs. pooling.
- Epoch/quality cuts (min good epochs/band; catflag=0; `nobs` floor).
- Whether to seed the demo from the already-cached same-star pairs (only 8 same-star g+r
  in the cadence cache) or build fresh from Chen×TAP (recommended — cache is unlabeled).
- Alias/Blazhko handling already in E1's protocol; period truth = Chen `Per` (ZTF-derived).

## Reproduce
`experiments/phase4_demo/e2_xmatch_probe.py` (stdlib urllib; VizieR + IRSA TAP). Bounded:
4 RRL × 2 bands. Fresh-write, does not import the paper scripts.
