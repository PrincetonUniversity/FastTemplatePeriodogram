# Phase 3.2 — Sesar recovery experiment

`run_sesar_recovery.py` is the first paper-quality recovery driver (PLAN §3.2
deliverables i + ii): it fetches the Sesar 2010 ugriz templates, builds a
PAM-learned vocabulary, and sweeps period recovery over one frozen, seeded
synthetic multiband population — both vs vocabulary size **K** and, at the K-sweep
knee, vs per-band epoch count **N_epochs** (the sparse-regime story). The FTP@H=1
single-cosine (== GLS) baseline is included as the reference. Headline criterion is
the hard 1% fractional rule; the search grid is an explicit `[f_min, f_max]` band.

**Year-beat caveat (WP B5).** Under the headline fractional criterion a ±1/yr
window beat (|Δf| = 1/365.25 c/d) falls within the 1% tolerance for
f_true ≳ 0.28 c/d and is counted as *exact* — the exact test runs first, so the
in-run alias scan never surfaces it — while 1-day beats and P/2 fail the same
tolerance (asymmetric). `rescore_alias_breakdown.py` re-scores the persisted
per-source periods post hoc under phase coherence (|Δf|·T < 0.5), which does
separate year beats, and emits an alias-breakdown table per (method, cell). The
recovery-driven greedy selection optimized the fractional criterion and therefore
inherited this convention — caveat on the selection objective only, no
re-selection.

## Running

```bash
# tiny end-to-end wiring check (~1 min)
python run_sesar_recovery.py --smoke

# bounded-local draft on 10 cores (~19 min; what produced output_draft/)
python run_sesar_recovery.py \
    --n-sources 96 --n-master-epochs 20 --baseline-days 365 --n-freq 3000 \
    --k-values 1,2,4,8 --n-epochs-values 4,8,12,16,20 --n-jobs 10 \
    --outdir output_draft
```

The committed `output_draft/` (full 98-template ugriz universe, 96 sources, 1-year
baseline, `--n-jobs 10`) makes the **recovery-vs-N_epochs** panel the headline: a
single learned RRab template (the knee is K=1) recovers far more than the GLS sinusoid
in the sparse regime — N=4: **0.375 vs 0.104**, N=8: **0.958 vs 0.604** — converging by
N=16. Remember N is **epochs per band**, so with g+r the x-axis spans 8→40 total
points per light curve.

The **recovery-vs-K** panel is flat at 1.0 here, and that is a real result rather than
saturation alone: at ≥~12 epochs/band over a one-year baseline the period is
recoverable by any method, and where it *is* hard (the sparse N-sweep) K=1 already
captures the whole shape-prior gain — RRab shapes are homogeneous enough that
vocabulary *size* adds little; the win is template *shape* vs. sinusoid. A
discriminating recovery-vs-K panel needs a **sparse** master cadence (e.g.
`--n-master-epochs 6`), where it tests whether K>1 helps when data is scarce.

`output_sparse_k/` is exactly that run (6 epochs/band, 96 sources): the recovery-vs-K
panel is now discriminating — FTP holds **~0.88–0.92** across K=1–8, far above the GLS
line at **0.25** — and shows K=1–2 already suffices (larger K marginally hurts via
catalog-max alias peaks). Together the two dirs give deliverable (ii) (`output_draft/`,
the N_epochs headline) and deliverable (i) (`output_sparse_k/`, the K panel).

Outputs: `results.json` + `results.npz` (raw numbers) and, when matplotlib is
installed, `figures/` (recovery-vs-K, recovery-vs-N_epochs, and the two-panel
figure).

## Track B — real ZTF cadence

`fetch_ztf_cadence.py`, `ztf_error_model.py`, and `run_real_cadence_recovery.py`
drop a **real ZTF DR observing cadence** into this same harness and re-run the
recovery sweeps on real vs. synthetic sampling.

```bash
# 1. fetch ~150 real ZTF light curves (magnitude-matched, g+r) and cache them
#    (uses the isolated fetch venv -- see the astroquery note below)
python fetch_ztf_cadence.py --no-tap --n-objects 150 --bands g,r \
    --seed-oids 686103400067717,486103400000001,786103400000001 \
    --walk-span 160 --max-probe 900 --min-epochs 80

# 2. inspect the empirical error-vs-mag model vs exp_mag_error()
python ztf_error_model.py

# 3. real-vs-synthetic recovery draft (LOCAL scale; writes real_cadence_draft/)
python run_real_cadence_recovery.py            # bounded draft
python run_real_cadence_recovery.py --smoke    # tiny wiring check
```

`RealZTFCadence` (in `ftperiodogram.simulate`) replays the cached per-band epochs
through the same `CadenceSample` contract as `SyntheticCadence`, so every simulator
and driver consumes it unchanged. The driver matches the synthetic cadence to the
real one's baseline + bands and uses the *same* empirical error model, so a
real-vs-synthetic recovery gap is attributable to the **sampling structure**
(seasonal gaps + clumping), the regime where FTP's shape prior should help most.

### Data source (probed 2026-06-04)

IRSA's TAP `ztf_objects_dr22` summary table (the natural magnitude-selection path)
was **down at fetch time** (`ORA-12541: TNS:no listener`, all DRs), as was the
spatial/cone path of the lightcurve API. The **`nph_light_curves` API by `ID=<oid>`
worked**, returning full epochal photometry, so the fetcher falls back to walking
oid ranges and pulling LCs by id. Because ZTF's per-filter reference catalogs use
independent running-number orderings, a g-block walk and an r-block walk land on
disjoint sky positions, so genuine same-star g+r matching needs the (down) spatial
index; the fetcher instead pairs a g oid with an r oid **from the same ZTF field**
(`group_mode='same_field_cadence'`): different physical stars, but an authentic
two-band real cadence off one survey schedule — exactly what Track B needs
(realistic *sampling + error-vs-mag*, not a confirmed RR Lyrae). The grouping mode
is recorded per object so it is never silently conflated with a true multiband star.

### astroquery / venv note (important)

`astroquery` pulls in `astropy`, whose only Python-3.9 release (6.0.1) is
**binary- and runtime-incompatible with the core package's numpy 2.0.2**
(`np._core.umath._ljust` is a numpy≥2.1 symbol; the precompiled wheel is numpy-1.x
ABI). Installing it into the shared `.venv` breaks the core test suite
(`test_slow_template_modeler.py` collection). So astroquery lives in an **isolated
`.fetch_venv/`** (numpy<2 + astroquery) used *only* by `fetch_ztf_cadence.py`; the
shared `.venv` stays pure numpy/scipy/nfft and the suite stays green. The fetch's
TAP path lazy-imports astroquery (and its stdlib `nph_light_curves` fallback needs
no astroquery at all), so the cache it writes is consumed downstream under numpy 2.

## Scale / compute note

The script **defaults** are the full bounded-ugriz config (98 templates,
`n_sources=128`, `n_freq=10000`, 3-year baseline). The per-frequency multiband solve
is ~1 ms, so cost scales as `sum(K) x n_freq x n_sources`. The single biggest lever is
`--n-jobs`: the per-source loop is embarrassingly parallel, so `--n-jobs -1` gives
roughly an Nx speedup on an N-core box (this draft saw ~8x on 10 cores). Even so the
full default config is ~1.5-2.5 hr on 10 cores (the 10k-freq grid dominates) — for the
true publication-resolution run, a many-core cloud box (`--n-jobs -1`, torn down after)
brings it to ~10-15 min. The committed `output_draft/` is the bounded-but-parallel
config above; treat its absolute numbers as a draft, the full-resolution run as the
publication figure.
