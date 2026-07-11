# WP E3 memo — false-alarm behaviour of the FTP catalog statistic on real ZTF photometry

**Date:** 2026-07-11 (branch `dev`).
**Sample:** the 99 cached magnitude-selected (14–18.5) ZTF field objects in
`~/.ftperiodogram_data/ztf_cadence_sample/` (155 band-light-curves; 0 skipped for
N<40). Phase-1 screen (`e3_screen.py`, `screen_table.json`): **81 objects clean**,
**18 flagged** (8 amplitude-flagged, 7 amplitude-marginal-only, 7 catalog-matched
variables; overlapping categories). At the band-LC level: 126 clean, 29 screened.
**Statistics (identical machinery/seeds to D1):** FTP catalog-max over the order-8
Sesar PAM vocabulary with nested prefixes K ∈ {1,2,4,8} (seed 0), FTP single-medoid
H ∈ {1,3}, and GLS. All FTP runs use the production `method='scan'` default (C4).
Frequency bounds are always the explicit `[f_min, f_max] = [1, 5]` cyc/day;
`nyquist_factor` is never used.

**Artifacts:** `e3_results.npz` (per-band-LC powers, matched-null draws, full
`summary_json` with grid metadata), per-unit npz under `output/` (155 real +
100 null chunks), `fig_e3_primary.png`, `fig_e3_secondary.png`, `check_hits.py`
(peak-frequency identification), `make_e3_figures.py` (also prints every count
quoted below). All numbers below are from this run of the pipeline
(`run_e3.py --stage real|null|aggregate`, seeded and deterministic).

## Grid metadata

| run | grid | df (cyc/day) | pts/Rayleigh |
|---|---|---|---|
| Primary (real, N=40 subsample) | 10,000 freqs on [1, 5] (exact D1 grid) | 4.0004e-4 | 0.95 at the real median subsampled baseline 2631 d |
| D1 null (reference) | same 10k grid | 4.0004e-4 | 2.28 at D1's baseline 1095.75 d |
| Secondary (real, full N) | per-object df ≤ 0.2/T on [1, 5] | median 7.324e-5 (min 7.281e-5, max 1.892e-4) | 5.00 by construction; n_freq median 54,615 (max 54,935) |
| Matched null, g stratum | df ≤ 0.2/T, [1, 5]; n_freq 54,877 | 7.289e-5 | 5.00; N=626, T=2743.8 d, 500 realizations |
| Matched null, r stratum | df ≤ 0.2/T, [1, 5]; n_freq 54,615 | 7.324e-5 | 5.00; N=947, T=2730.7 d, 500 realizations |

Matched-null seeds 3,000,000+ (g) / 3,500,000+ (r), disjoint from D1's 1,000,000+;
subsample seeds 500,000+. Matched null uses D1's synthetic seasonal cadence +
`exp_mag_error` model with amplitude=0, N and T matched to the per-band sample
medians. CPU: real stage 6,833 s, null stage 36,571 s, total ≈ 43,404 CPU-s.

## (a) PRIMARY — N=40 seeded subsamples on the D1 grid vs the D1 null

Each band-LC was subsampled (seeded, catflags==0 epochs only) to N=40 and scanned
on the identical 10k-frequency grid as the D1 null
(`experiments/fap/output_null/null_maxpower.npz`, n=3000 realizations). Thresholds
are D1's empirical p99 and the D1 GEV fit at the 1% tail (`gev_json`).

**Screened-clean band-LCs (n=126, from the 81 clean objects) — observed exceedances
vs expected 1.26 per statistic:**

| statistic | D1 p99 thr | GEV-99 thr | clean > p99 | clean > GEV | binom P(≥obs) at p99 |
|---|---|---|---|---|---|
| GLS (= FTP H=1) | 0.5347 | 0.5293 | 3 | 3 | 0.13 |
| FTP H=3 | 0.5509 | 0.5508 | 2 | 2 | 0.36 |
| cat K=1 | 0.5403 | 0.5410 | 4 | 4 | 0.04 |
| cat K=2 | 0.5606 | 0.5563 | 3 | 3 | 0.13 |
| cat K=4 | 0.5695 | 0.5689 | 3 | 3 | 0.13 |
| cat K=8 | 0.5760 | 0.5789 | 3 | 2 | 0.13 |

(ftp_H1 duplicates GLS to numerical precision, as expected.) Five distinct clean
band-LCs account for all exceedances (any statistic): g00014 r, g00029 g, g00037 g,
g00044 g, g00082 r. Union across the 7 statistics: the D1 null's union rate is
2.20%, so expected 2.77 clean band-LCs, observed 5 (binomial P(≥5) = 0.15). The
clean-vs-D1 KS distances are small but nonzero (e.g. cat_K8: KS=0.14, p=0.016),
with the clean medians slightly *below* the D1 medians and the tail slightly
heavier.

**Reading:** the clean sample's tail is consistent with, or at most mildly above,
the D1 false-alarm calibration — and the mild excess runs in the direction
predicted by the known trials mismatch: the real subsampled baselines (median
2631 d) exceed D1's 1096 d, so the shared 10k grid holds ~0.95 pts/Rayleigh for
the real side vs 2.28 for D1, i.e. ~2.4x more independent frequencies for the real
LCs even under pure noise. No clean object stands out individually (largest clean
d1-p-value tail: g00037 g at p≈7e-4 for GLS, still a ~1-in-1400 event given ~880
clean trials across 7 correlated statistics).

**Screened/flagged objects (n=29 band-LCs from the 18 flagged objects), reported
separately.** Three band-LCs exceed the D1 p99 (and GEV-99) thresholds in at least
one statistic — all three are flagged objects whose flags correspond to plausible
or confirmed signal, i.e. the screen and the periodogram agree:

| object/band | screen flags | peak identification (`check_hits.py`) |
|---|---|---|
| g00017 g (Chen+2020 EW, P=0.46366 d; amplitude-flagged g) | cat_K8 = 0.688, GLS = 0.672 (d1-p ≈ 3e-4–7e-4) | full-N peak f=4.31343 = 2·f_orb to 1e-4 cyc/day (EW half-period harmonic); N=40 peak at 3.31073 = 2·f_orb − 1.0028, a 1-day alias of the same signal |
| g00015 g (Chen+2020 EW, P=1.2068 d; Gaia ECL; amplitude-flagged g+r) | GLS = 0.577, cat = 0.566 (d1-p ≈ 2e-3) | full-N peak f=1.65726 = 2·f_orb to 2e-5 cyc/day; N=40 peak at 3.65997 = 2·f_orb + 2.0027, an alias family member |
| g00052 r (Gaia DR3 AGN candidate, score 0.92; amplitude-marginal r) | cat_K8 = 0.688 (d1-p ≈ 7e-4) | peak f=2.00220 cyc/day (P≈0.4995 d), i.e. sitting on the 2 cyc/day diurnal-alias comb — consistent with stochastic AGN variability aliased through the ZTF window function, not a coherent period |

The remaining 26 screened band-LCs do not exceed the p99 thresholds; the screened
stratum's medians sit only slightly above the clean stratum's (e.g. cat_K8: 0.459
vs 0.459). The two catalogued EWs are recovered at the correct (half-period
harmonic) frequency, and the one non-EW exceedance is an amplitude-flagged AGN
candidate showing alias-comb power — the amplitude screen and the periodogram tail
select the same objects, which is signal behaving like signal, not a false-alarm
pathology.

## (b) SECONDARY — full-N max powers vs the FRESH matched null (only)

Full-N runs (per-object df ≤ 0.2/T grids) are compared exclusively to the fresh
N/T/grid-matched synthetic null per band stratum (never to the N=40 D1 null).

| band, statistic | null p99 | null median | real clean median | clean > p99 | screened > p99 | KS (clean vs null) |
|---|---|---|---|---|---|---|
| g, GLS | 0.0470 | 0.0337 | 0.1106 | 52/53 | 11/11 | 0.97 |
| g, FTP H=3 | 0.0505 | 0.0384 | 0.1159 | 51/53 | 10/11 | 0.96 |
| g, cat K=1 | 0.0481 | 0.0352 | 0.1169 | 50/53 | 11/11 | 0.96 |
| g, cat K=8 | 0.0521 | 0.0409 | 0.1341 | 51/53 | 11/11 | 0.96 |
| r, GLS | 0.0300 | 0.0226 | 0.0521 | 61/73 | 15/18 | 0.85 |
| r, FTP H=3 | 0.0348 | 0.0252 | 0.0603 | 60/73 | 14/18 | 0.86 |
| r, cat K=1 | 0.0314 | 0.0236 | 0.0571 | 61/73 | 15/18 | 0.84 |
| r, cat K=8 | 0.0366 | 0.0270 | 0.0664 | 61/73 | 14/18 | 0.86 |

**Reading:** at full N (median 626/947 epochs), essentially the whole real sample —
clean included — sits far above a homoscedastic-Gaussian synthetic null with the
pipeline error bars (96–98% of clean g LCs and 82–84% of clean r LCs exceed the
null p99; expected 1%). This is the known excess-variance property of real ZTF
photometry (empirical errors 1.6–3.5x the synthetic model at these magnitudes;
low-level correlated systematics), not a defect of the statistic: the absolute
power levels remain small (clean medians 0.05–0.13), the g stratum — with larger
per-point errors relative to systematics at these magnitudes — is farther from the
null than r, and GLS is displaced from its null by the same amount as the FTP
catalog statistic (KS 0.97 vs 0.96 in g; 0.85 vs 0.86 in r), so the effect is a
property of the photometry, not of template fitting or the K-axis. **Operational
conclusion:** synthetic Gaussian nulls (even N-, T- and grid-matched) must not be
used to set detection thresholds for full-depth real light curves; thresholds
there need an empirical null (e.g. shuffled/bootstrap real photometry), while the
N=40-matched comparison of (a) is the clean like-for-like test of the D1
calibration.

## (c) Caveats (mandatory)

1. **n=99 ⇒ ~1% FAP floor.** With 99 objects (126 clean band-LCs), the smallest
   per-object false-alarm rate this sample can probe empirically is ~1%; nothing
   here constrains the deep (per-candidate 1e-3 and beyond) tail.
2. **Only 8 same_star g+r objects are physical two-band stars.** The 48
   same_field_cadence pairs are a cadence null only; the g and r members of each
   pair are DISTINCT stars, separated by up to ~0.85 deg (median 0.25 deg,
   measured from `screen_table.json` coordinates; the task-level figure "~0.7 deg"
   is superseded by this measurement). No multiband/joint statistic is computed
   here; band-LCs are treated as independent trials, which is exactly right for
   the 48 cadence pairs and 43 single-band objects and conservative bookkeeping
   for the 8 same_star pairs (their g and r draws share a star and are therefore
   correlated trials, not independent ones).
3. **The D1 null is single-band, N_obs=40, on one specific 10k grid — it is NOT
   transferable across N, band structure, or frequency grid.** That is why the
   primary comparison subsamples the real data to N=40 on the identical grid, and
   why the secondary (full-N) runs are compared only against the fresh
   N/T/grid-matched null. Even in the primary comparison a residual mismatch
   remains: the real subsampled baseline (median 2631 d) exceeds D1's 1096 d, so
   the shared grid gives the real side ~0.95 pts/Rayleigh vs D1's 2.28 (~2.4x more
   independent trials under pure noise); the observed mild clean-tail excess is in
   the direction and rough magnitude of this mismatch.
4. **The D1 K-axis nests one K=8 PAM vocabulary** (nested cumulative maxima over
   prefixes of a single seed-0, H=8 vocabulary) **rather than re-selecting a
   per-K vocabulary as production does; this underestimates the production
   K-dependence by a bounded amount (~0.025 in max-power units, per the D1
   analysis).** The same nested construction is used identically for the real
   runs and both nulls here, so all comparisons in this memo are internally
   consistent; only the absolute K-axis interpretation carries the bound.

## Reproduction

```
.venv/bin/python experiments/phase4_demo/e3_false_alarm/run_e3.py --stage real  --workers 8   # resumable
.venv/bin/python experiments/phase4_demo/e3_false_alarm/run_e3.py --stage null  --limit 8 --workers 8   # repeat until COMPLETE
.venv/bin/python experiments/phase4_demo/e3_false_alarm/run_e3.py --stage aggregate
.venv/bin/python experiments/phase4_demo/e3_false_alarm/make_e3_figures.py
.venv/bin/python experiments/phase4_demo/e3_false_alarm/check_hits.py
```
