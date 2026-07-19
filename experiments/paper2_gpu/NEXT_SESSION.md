# Paper-2 engine — next-session task queue (set 2026-07-19, closing out speed/cost)

Speed/cost optimization on FTP is essentially DONE and verified (SAFE
stack landed; efficiency-hunt-2 wins recorded; GPU validated). The
remaining items close the last cost-calibration gaps and PIVOT to the
whole-pipeline / pipeline-development questions. Ordered by
priority-to-unblock.

---

## T1 (BLOCKING, John-action + research). Resolve PS1 data access → N_cand
The single unknown gating any absolute cost number. Three paths, in
increasing autonomy:

**T1a — CasJobs (fullest, needs John once).** MAST CasJobs
(mastweb.stsci.edu/mcasjobs) requires a free account + interactive
login the agent cannot do. SPECIFIC STEPS for John:
  1. Register/login at mastweb.stsci.edu/mcasjobs (top-right "Create
     Account" / "Login"; if you already have a MAST GALEX/Kepler
     CasJobs login it works here — do not re-register).
  2. Leave the Chrome session logged in — the agent's browser tools can
     then drive the query (session persists ~minutes of inactivity;
     re-login if it times out).
  3. Context = PanSTARRS_DR2. The funnel: spatial/cone or
     fGetNearestObjEq → ObjectThin + MeanObject (positions, mean mags,
     colors) → Detection (per-epoch light curves) → batch to MyDB →
     extract. Reference: ps1images.stsci.edu/ps1_dr2_query.html.

**T1b — account-free MAST DR2 API (agent CAN do this; recommended for
the N_cand pilot).** catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/…
(verified working, PAPER2_BRIEF WP P0.2). Task for the agent: cone-search
a few representative high-|b| pilot fields, apply the shortlist cuts
(point source, variability pre-cut, dereddened color box for the
instability strip, magnitude floor r≲21.5 single-epoch), get a per-deg²
density, extrapolate to the 3π footprint with the |b| cut → **N_cand
estimate with error bars**. This unblocks the whole cost table WITHOUT a
login. DO THIS FIRST.

**T1c — PS1 on AWS Open Data (S3 Parquet, no account).** For the eventual
BULK pull: PS1 tables are on the AWS Open Data registry (free, us-east-1;
compute in-region to avoid egress). Best path for the production
shortlist extraction; research the exact bucket/schema next session
(registry.opendata.aws + STScI docs).

**Agent research sub-task:** before T1b, spend ~15 min confirming (a) the
current MAST DR2 API endpoints/rate limits, (b) the exact S3 bucket path
+ schema for PS1 DR2 Detection/MeanObject, (c) whether a free
Detection-table bulk path exists that beats CasJobs. Give John a
one-page "here's how we get the data, here's the recommended path" with
concrete commands.

## T2 (RESEARCH, high strategic value). Whole-pipeline bottleneck profile
HYPOTHESIS (2026-07-19, unvalidated): FTP is no longer the bottleneck
(~40× optimized → ~2-5% of wall-clock). The realistic bottleneck ranks
(1) data acquisition/IO, (2) data reduction, (3) FTP detection, with FTP
COMPUTE returning to dominance ONLY in the injection/completeness
campaign (10-100× multiplier). VALIDATE by building a back-of-envelope
(then measured, on the pilot field) end-to-end budget:
  - Stage 1 acquire: rows pulled, bytes, CasJobs-queue vs S3-scan time,
    egress $.
  - Stage 2 reduce: per-object LC assembly + cuts cost × N_cand.
  - Stage 3 FTP detection: 0.09 s (GPU) / 3.5 s (CPU) per candidate
    (measured).
  - Stage 4 vet: generative model-comparison at the peak (single freq —
    estimate, likely cheap).
  - Stage 5 injections: FTP × (injections/object); the windowing lever
    (13-18×, B6-adjacent) is the load-bearing optimization HERE, not in
    detection.
Deliverable: an Amdahl table that says where to spend the NEXT
optimization hour — and, per the hypothesis, likely tells us to STOP
optimizing detection-FTP and NOT implement hunt-2 yet.

## T3 (RESEARCH, pairs with T2). Data-reduction design
Research + prototype the Stage-2 reduction: from raw PS1 Detection rows
to clean per-object multiband light curves ready for FTP. Cover:
forced-warp vs stack photometry at the faint end (r>21.5 epochs live/die
on forced photometry — PAPER2_BRIEF §), catflags/quality cuts, the
censoring-fraction-vs-mag measurement (a selection-function input, WP
P0.2), dereddening, the variability + color pre-selection that shrinks
N_cand. This is where the light curves the FTP engine consumes actually
come from — currently unbuilt.

## T4 (measurement, ~30 min compute). B1 fixed-vocab + os≥6 rerun
Closes the one open completeness number: the RRab H4-vs-H8 staging cost.
Re-run B1 with (a) a FIXED vocabulary across H (build at H8, truncate —
removes the per-H medoid-swap confound the re-audit found), (b) os≥6
detection grids for H≤4 (the os-doubling probe showed H4 os=3 is NOT
converged — 4.7% coarse false-recoveries), (c) score with the STRICT
phase-coherence column (not the day-beat-inflated alias-credited rate).
Then re-quote the staged-H tradeoff honestly. Spec/script: adapt
b1_recovery_vs_H.py; ~256 sources.

## Deferred / optional (do NOT gate pipeline work on these)
- **Implement efficiency-hunt-2 wins** (safe_stack/EFFICIENCY_HUNT_2.md):
  ~1.3-1.55× more, ulp-safe. Per T2 this is likely PREMATURE — bank it.
- **GPU session B** (~$3-5): H4 rows, H100 column. Only if the injection
  campaign (T2 stage 5) shows FTP compute matters.
- **Production CUDA kernel** (fused, K.18, candidate compaction): only if
  GPU throughput is on the critical path per T2.
- **Bernstein r≥0.15 filter extension**: SOLID+bit-identical but re-touches
  the FILTER-DIP-1 code — needs the full revalidation ensemble first.
- **Cascade faint-edge arm**: B6 marginal-signal survival, if the
  Aggressive cost tier is wanted.
- **John's adoption calls**: K=1 vocab, polish-free detection, GLS cascade
  (completeness/FAP judgment, not engineering).

## Portfolio summary (for the cost decision, per-million candidates)
| Tier | config | SNR/completeness cost | CPU $/M | GPU $/M |
|---|---|---|---|---|
| Maximal | H8 both, K=4, full grid | ~zero | ~$400 (scaled) | ~$170 |
| **Balanced** (measured) | staged H4/H2, K=4, refine H8 | final SNR full; RRab detection completeness = T4 | **~$75** | **~$38** |
| Lean | staged H, K=1 | none measured (B3) | ~$30 | ~$16 |
| Aggressive | +GLS cascade q≈2% | faint-edge unmeasured (B6) | ~$8-12 | ~$5 |
All tiers hold FINAL measurement SNR at full H=8 (staged-refine). Only
the balanced row is directly measured; others scaled from anchors. All
compute wins are numerically lossless — SNR is traded only by the
detection-approximation levers, and only in completeness, not in
reported precision. N_cand (T1) turns per-million into absolutes.
