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
