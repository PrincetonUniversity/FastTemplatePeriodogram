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
