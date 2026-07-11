# Arm (b) grid closure -- NO DATA (attempt 2026-07-11, canary killed at deadline)

Status: **ATTEMPTED, ZERO REFINED CELLS LANDED.** The 32-vCPU-pinned canary
(g-refine-sesar-0, pod `ehrzmk00e1cq0d`, cpu5c/32 SECURE) booted and ran
cleanly but burned >=91.2 core-hr against the 45 core-hr estimate (>=2.03x,
gate FAILED) and was killed at the hard 2h45m deadline with no RESULT beacon;
per-source npz only ship in the RESULT, so no refined masks exist. The batch
(g-refine-sesar-1/2, g-refine-bv-0) was NOT released: extrapolation >=$17.8 vs
the $12 task budget, and both sesar (>=2.85 h) and bv (>=4.3 h) walls exceed
the per-job deadline at the measured floor. Full measurement + de-risk options
in `../B8_COST_ESTIMATE.md` ("arm (b) 32-vCPU canary", 2026-07-11).

Consequences for this directory's numbers:

- The arm-(c) disposition in `SUMMARY_c-grid20k.md` is UNCHANGED: absolute
  sparse-N rates for H=8/binned methods remain **lower bounds "at the 10k
  production grid"**; the 8 flagged cells (ftp_pam N=8; mhls N=8,12,16,40;
  mbls N=12; ce N=12,16) remain open.
- Key new fact: the `--refine` overhead measured >=1.9x on the N-sweep (floor;
  job incomplete) vs the ~1.4x RefinedEstimator cost-formula prediction -- any
  re-attempt must re-estimate from a completed run, or shard the sweep by N.

When real g-refine results land under `raw/g-refine-*/`, run
`python finalize_b8_closure.py` -- it overwrites this file with the per-cell
paired McNemar table (refined vs 10k on identical sources) and writes
`aggregate_g_refine.json`.
