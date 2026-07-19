# B1 — recovery vs H at each H's converged grid (2026-07-18)

Audit-corrected harness (PS1 5σ errors, TTI pair cadence, true-p2p amplitude
draws, held-out vocab). 256 sources/subtype, mag 21.0, N=8/band griz, K=4,
os=3 per-H grids. Raw: output/b1_{rrab,rrc}_m21_n8.json.

| subtype | H1 | H2 | H3 | H4 | H5 | H6 | H8 |
|---|---|---|---|---|---|---|---|
| RRab | .133 | .238 | .266 | .328 | .316 | .336 | .336 |
| RRc  | .109 | .102 | .102 | .121 | .121 | .121 | .121 |

Paired losses vs H8 (the staging lever, HANDOFF §2.3):
- **RRab H4: +0.008 ± 0.020** (26/256 disagree) — lossless within noise, 95% UB ~3%.
- RRab H2: +0.098 ± 0.031 — H2 is NOT enough for RRab (plan's H4 choice right).
- **RRc H4: 0 (0/256 disagreements, 95% UB 0.012)** — exactly lossless.
- RRc H2: +0.020 ± 0.015 — small but likely real loss.

**Surfaced tradeoff (John):** locked plan says H≈2 for RRc detection. Measured:
H2 costs ~2% (CI 0.5–3.5%) of RRc recovery vs H4's exact 0; RRc H2→H4 raises
the RRc detection arm cost ~5–8× (cost ∝ ~H^2.1 per-freq × H grid). Options:
keep H2 + quote the measured loss in the completeness map (default), or
H3/H4 if the ~2% matters. RRab arm dominates total cost either way.

Caveats: synthetic PS1-like cadence (real-PS1 arm pending WP P0.2); absolute
rates are alias-credited (exact-or-harmonic; strict-exact and phase-coherence
columns in the JSONs); single (mag, N) cell — the full completeness surface
is the injection campaign's job. H8-grid convergence: os=3 with df ∝ 1/(H·T),
per-H matched (audit reviewed the formula as sound; os-doubling spot-check
queued with B5).

---

## CORRECTIONS (2026-07-18 both-sessions re-audit, panel-confirmed)

1. **Alias crediting:** the "alias-credited (exact-or-harmonic)" label
   was wrong — lib.score's default crediting ALSO accepts ±1,±2 c/d and
   ±1/yr window beats. Recount from the raw JSONs: at RRc H4/H8, 13/31
   credited recoveries (42%) are day-beats and 6 more are P/2; only
   12/256 exact, 9/256 phase-coherent (vs the headline .121). RRab is
   mildly affected (~5% beats). The "strict-exact" JSON column is also
   not strict (rtol=0.01 admits year-beats; 3-7 "exact" sources per arm
   fail phase coherence). RRc ABSOLUTE rates in the table are therefore
   substantially alias-inflated; RRab conclusions stand.
2. **The RRc "H2 costs ~2%" figure is statistically null and must not
   be quoted** (final panel verdict): the +0.020 +/- 0.015 paired loss
   is a 10-vs-5 disagreement split (McNemar exact p = 0.30 — noise);
   under strict frac_exact the sign REVERSES (H2 better by 0.004) and
   under phase coherence it is exactly 0. The "(CI 0.5-3.5%)" was a
   mislabeled +/-1-sigma interval; the true 95% CI is ~[-0.010, +0.049]
   and includes 0. Net: there is NO measured RRc H2 penalty — "RRc
   detection at H2" stands with no loss caveat. (A per-H vocabulary-
   swap confound was also raised — each H re-clusters its own medoids,
   RRc H2 sharing 0/4 stars with H8, undisclosed in the original doc —
   but the panels REFUTED it as material: a de-confounded rerun with
   the swapped vocabulary reproduced the arm's results, consistent with
   B3's flat K-lever.)
3. **Grid convergence — RESOLVED for H8, FAILS for H4** (64-source
   probe, OS_DOUBLING_RESULT.md): H8 os=3 is converged (0/64 flips);
   **H4 os=3 is NOT** — 3/64 (4.7%) os=3 recoveries are coarse-grid
   artifacts that vanish at os=6, all spurious-high. So the RRab H4
   absolute rate (.328) is ~4.7% grid-inflated (converged ~.31), and
   the H4-vs-H8 "+0.008 lossless" headline is biased LOW (H8 converged,
   H4 not) — the true staging loss is larger by up to the coarse
   inflation. The fixed-vocab rerun must use os>=6 for H<=4 grids.
4. **Stats/config notes:** the RRab H4 one-sided 95% UB is ~4.1-4.2%
   (not "~3%"); the run baseline was `--baseline-days 1300` (realized
   median ~1200 d), not the 1600 d survey span — paired comparisons
   unaffected, absolute rates correspond to the shorter baseline.
