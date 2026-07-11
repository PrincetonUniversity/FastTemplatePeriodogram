# Phase-4 RRL sample — acquisition summary

Built 2026-07-11T04:40:12Z by `fetch_rrl_sample.py` (approved spec, John 2026-07-10).

## Counts

| Field | Science | RRab | RRc | Control (Gaia SOS) |
|---|---|---|---|---|
| 486 | 22 | 11 | 11 | 5 |
| 686 | 18 | 9 | 9 | 5 |
| 786 | 18 | 10 | 8 | 5 |
| **total** | **58** | 30 | 28 | **15** |

- 686 sanity crossmatch: **PASSED**
- Rejected candidates: 5 (see `manifest.json` rejects[]); failures promoted to alternates: 5
- Cuts: same-star g+r within 1.0", catflags==0 only, >=40 good epochs/band.
- Selection: deterministic brightest-first (rmag / G), balanced RRab/RRc per field where available. No hand-picking, no RNG.
- Controls fetched identically (same TAP xmatch + LC API + cuts); `"control": true` + Gaia pf/p1_o truth; Chen overlap flagged per star (12 overlap).

## Cost

- HTTP requests: 28 (budget 480); downloaded 0.6 MB (budget 90 MB); cache hits 204.
- LC data on disk: 2.9 MB in `/Users/johnhoffman/.ftperiodogram_data/phase4_rrl_sample/lc/` (NOT committed).
- Wall time: 281 s.

### Acquisition history (cumulative across resumed invocations)

The Cost numbers above are from the final (fully cache-resumed) invocation
only. The acquisition actually spanned four invocations on 2026-07-10/11
(all resumed from the raw cache, refetching nothing):

1. Killed by a 120 s tool timeout after 8 field-486 science LCs
   (29 raw payloads) — not a script failure.
2. Killed manually mid-science while being relaunched (resumable, no loss).
3. Completed all 58/58 science LCs (55 new requests, 6.3 MB, 765 s), but all
   3 Gaia control queries failed: the `gaiadr3.gaia_source` join
   deterministically hits the ESA sync-TAP 60 s server-side abort
   (HTTP 408 "Job timeout/aborted") for these ~47 deg^2 boxes.
4. After switching the join to `gaiadr3.gaia_source_lite` (same columns,
   ~30 s/query; selection region unchanged): all 15/15 controls
   (28 new requests, 0.6 MB, 281 s).

Cumulative measured totals: 206 cached payloads, 23.8 MB downloaded and
kept, 38.9 min wall-clock from first to last payload (including idle gaps
between invocations). Note: the IRSA LC API returns only the first oid of a
comma-separated `ID` list, so the script's per-oid fallback fires for the
r-band oid of every star (works, cached, ~1 extra request/star).

## Reproduce

```bash
FastTemplatePeriodogram/.venv/bin/python \
    FastTemplatePeriodogram/experiments/phase4_demo/fetch_rrl_sample.py
```

Resumable: raw HTTP payloads cached under `~/.ftperiodogram_data/phase4_rrl_sample/raw/`; a rerun refetches nothing already on disk. Per-star LCs: `~/.ftperiodogram_data/phase4_rrl_sample/lc/<star_id>.npz` (keys `{g,r}_{mjd,hjd,mag,magerr}`).

## Control top-up (Chen-missed anti-join)

Added 2026-07-11T04:59:43Z by `fetch_rrl_sample.py --topup-controls`.

**Why:** 12 of the 15 original controls are `chen_overlap` — the same physical stars as science members (brightest-first Gaia selection in the same fields re-found Chen's stars), so they cannot measure the Chen-selection effect. This top-up keeps only Gaia SOS RRL with NO Chen+2020 RRL within 2.0" (anti-join against the full per-field Chen tables), brightest-first, through the identical g+r xmatch / catflags==0 / >=40-epochs pipeline. New entries carry `"control_antijoin": true`, `"chen_overlap": false`.

**Dedupe note for downstream:** manifest entries are per catalog row, not per physical star — the original 73 entries span 61 unique stars (12 controls duplicate science members). Dedupe by ZTF `oid` before per-star statistics. After this top-up: 85 entries, 73 unique stars (+12 new, all non-Chen).

### Chen-missed selection funnel (a measurement, not bookkeeping)

The pass rate of Chen-missed stars through the ZTF quality cuts quantifies the Chen selection function:

| Field | Anti-join pool | Attempted | Pass g+r xmatch (1") | Pass epoch cut | Fetched |
|---|---|---|---|---|---|
| 486 | 274 | 18 | 17 | 5 | 5 |
| 686 | 824 | 7 | 6 | 6 | 6 |
| 786 | 4 | 4 | 3 | 1 | 1 |
| **total** | **1102** | 29 | 26 | 12 | **12** |

- "Attempted" = brightest-first prefix of the anti-join pool (until per-field target {"486": 5, "686": 6, "786": 4} or pool exhausted); pass rates are measured over that prefix, pool sizes over the whole field.
- "Pass epoch cut" = >=40 catflags==0 epochs in BOTH bands (includes the DR23 `ngoodobsrel` prescreen). Per-candidate failure stages/reasons: `manifest.json` `control_topup.rejects[]`.
- Cost this run: 10 HTTP requests (budget 150), 0.1 MB, 49 cache hits, 42 s.
