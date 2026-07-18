# B3 — vocab size (K) vs recall, held-out split (2026-07-18)

256 sources/subtype, PS1-pair cadence N=8/band griz, mag 21.0, baseline
1300 d, os=3 per-subtype grids (RRab H=4, RRc H=2 — the staged detection
configs), vocab from the held-out vocab half via K-medoid clustering,
truths from the truth half. WITHIN each subtype the K arms share the
same clustering pool and scoring, and every K vocab is built at the SAME
H — so the paired K-comparisons are free of the per-H vocabulary-swap
confound the 2026-07-18 re-audit found in B1's H-lever. Raw:
output/b3_vocab_{rrab,rrc}.json.

| K | RRab frac_rec (H4) | RRc frac_rec (H2) |
|---|---|---|
| 1 | **0.344 +/- 0.030** | 0.137 +/- 0.021 |
| 2 | 0.309 | 0.148 |
| 4 | 0.305 | 0.148 |
| 8 | 0.301 | **0.152 +/- 0.022** |

Paired vs K=8: RRab K=1 **-0.043 +/- 0.024** (K=1 BETTER, 37/256
disagree); K=2 -0.008 +/- 0.017; K=4 -0.004 +/- 0.014. RRc K=1
+0.016 +/- 0.016; K=2 +0.004 +/- 0.012; K=4 +0.004 +/- 0.010.
Exact and phase-coherent columns show the same ordering.

Reading: the K-lever is FLAT for RRc and flat-to-INVERTED for RRab —
the single best medoid template detects as well as (RRab: measurably
better than) an 8-template vocabulary at this cell. Mechanism for the
inversion: detection takes max power over templates, so each extra
template adds a chance for a wrong-frequency spurious win; RRab's
high-amplitude asymmetric shapes appear to make the extra templates
pure noise here. IF adopted (science gate, John's call): K=1 per
subtype cuts detection cost ~4x on both arms on top of the SAFE stack,
and the purity stage (generative model comparison at candidates) is
unaffected — it was never the detection vocab's job.

Caveats:
1. Single (mag, N) cell; synthetic PS1-like cadence (real-PS1 pending
   WP P0.2).
2. frac_rec is lib.score's alias-credited rate; per the re-audit this
   crediting includes day/year window beats (42% of RRc H4/H8 credits
   in B1 were day-beats). The PAIRED K-deltas use identical crediting
   on both sides, so the flat/inverted K-shape is robust to this, but
   the absolute rates in the table inherit the crediting semantics
   (exact-only columns in the JSONs are lower; see the B1 rescore item).
3. K=1's winner is the PAM medoid of the held-out pool — vocab
   construction still needs the pool; this is a scan-cost lever, not a
   data-requirements change.
