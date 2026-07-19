# os-doubling grid-convergence probe (2026-07-19)

The convergence check the 2026-07-18 re-audit found had never been run
(and whose absence lib.py's converged_grid docstring had falsely
claimed complete). Reconstructs the exact B1 RRab population
(deterministic seed=0, held-out split, 1300 d baseline), rescores the
first 64 sources at H8 and H4 on os=3 (the B1 production grid) AND os=6
(a strict superset), and counts recovery/argmax changes. Script:
b1_os_doubling.py; raw: output/os_doubling_rrab_64.jsonl.

Sanity: all 64 sources reproduce the stored B1 H8 and H4 P_rec bitwise
at os=3 (confirms B1 is uncontaminated by the later SAFE stack and the
reconstruction is exact).

| grid | os3->os6 recovery flips | argmax moved >0.1% |
|---|---|---|
| **H8** | **0 / 64** | 1 / 64 (eval noise) |
| **H4** | **3 / 64 (4.7%)** | 7 / 64 |

All 3 H4 flips are os3-recovered -> os6-NOT-recovered (sources 4, 15,
47): the os=3 "recovery" was a coarse-grid argmax landing on an accepted
alias that the finer grid resolves away. None are os6-only finds.

**Conclusions:**
1. **H8 os=3 is converged** — the staged-refine tier and all H8 columns
   in B1 are grid-safe.
2. **H4 os=3 is NOT converged** — the RRab **H4 absolute recovery rate
   (0.328) is inflated by ~4.7% coarse-grid false recoveries**. The
   converged H4 rate is ~0.31 (subtract the spurious ~3/64).
3. **Impact on the H4-vs-H8 paired loss (B1's staging headline,
   +0.008 +/- 0.020):** H8 is converged, H4 is coarse and biased HIGH,
   so the true H4-vs-H8 loss is LARGER than the recorded +0.008 by up to
   the ~4.7% coarse inflation — i.e. the "H4 is lossless vs H8" claim is
   WEAKENED, though the two effects (grid bias, alias crediting) are
   entangled and need a joint converged+strict-exact rerun to pin.
   Direction is clear; the magnitude is the open item.

**Action for the B1 fixed-vocab rerun:** use os>=6 for H<=4 detection
grids (or verify per-H convergence by doubling), and score with the
strict phase-coherence column, before re-quoting any staged-H tradeoff.
RRc was not probed here (H2 grids are even coarser in absolute df but
the RRc H-tradeoff is already deferred pending the fixed-vocab rerun).
