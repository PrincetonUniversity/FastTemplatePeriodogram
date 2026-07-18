# B6 — GLS-cascade survival fraction (2026-07-18)

Measurement arm for the SCIENCE-class "GLS harmonic-closure frequency
cascade" win (audit §2): if a cheap multiband GLS stage ranks the grid
and only the top fraction q (plus 1-day alias partners f±1, f±2 and
harmonic partners f/2, 2f) reaches the FTP stage, what fraction of
injected RRL keep their true (alias-credited) frequency in the surviving
set? 256 sources/subtype, PS1-pair cadence N=8/band griz, mag 21.0,
baseline 1300 d, held-out truths, os=3 grids (RRab H4 ~35.7k freqs, RRc
H2 ~33.6k). GLS = baselines.GLSEstimator (multiband shared sinusoid +
per-band floating offsets). Raw: output/b6_cascade_{rrab,rrc}.json.

| q | RRab survive (+partners) | RRc survive (+partners) |
|---|---|---|
| 0.01 | **1.000** (loss 95% UB 1.2%) | **1.000** (UB 1.2%) |
| 0.02 | 1.000 | 1.000 |
| 0.05–0.20 | 1.000 | 1.000 |

Exact-only (no partner expansion): RRab 0.906 at q=0.01 → 1.000 by
q=0.05; RRc 0.953 → 1.000 by q=0.10. Min-q to survive (alias-credited,
plain): median 5e-4 (RRab) / 3e-4 (RRc), p90 2.5e-3/1.3e-3, max
8.4e-3/6.1e-3.

Reading: at q=0.02 with partner expansion, the FTP detection stage
would scan ~2-4% of the grid (partners add ~6 extra windows/source)
with measured survival 256/256 on BOTH subtypes — the hunt's projected
~3.5-4.5x cascade speedup is conservative; the lever is real and
larger, IF adopted.

Caveats (measurement arm — adoption is John's science call):
1. Injections carry drawn detectable amplitudes at mag 21; survival of
   marginal signals near the detection floor is not measured here (the
   completeness-map injections would inherit the cascade if adopted, so
   the cascade must be part of any adopted pipeline's injection arm too).
2. Synthetic PS1-like cadence; real-PS1 arm pending WP P0.2.
3. GLS stage cost: baselines.GLSEstimator is a per-frequency Python
   lstsq loop (~0.5 s/grid) — fine for this arm, but an adopted cascade
   needs the O(N log N) GLS (astropy fast Lomb-Scargle per band) for the
   stage-1 cost to stay negligible.
4. Survival here upper-bounds cascade harm: sources FTP would not have
   recovered anyway cannot be lost to the cascade.
