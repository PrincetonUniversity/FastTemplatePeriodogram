# B7 — polish-free detection neutrality (2026-07-18)

Measurement arm for the SCIENCE-class "polish-free detection" win (audit
§2): production RRab detection arm (H=4, K=4 held-out vocab, PS1-pair
cadence N=8/band griz, mag 21.0, baseline 1300 d, os=3 grid), 256
sources, paired: (a) production scan settings vs (b) n_newton=0 (grid +
dip argmax, no Newton polish; applied via a self-testing wrapper around
core.scan_polish_from_coefs — effectiveness asserted per-run). Raw:
output/b7_polish_free.json.

- **frac_recovered: paired delta = 0** (0/256 disagreements; 95% upper
  bound 0.012). frac_exact / phase_recovered: -0.004 +/- 0.004 in favor
  of NO-polish (1/256, noise).
- Argmax grid bin identical 247/256 (96.5%); the 9 movers land on
  equivalent-credit frequencies (recovery unchanged).
- Spectrum-shape cost: per-source max |dP| over the grid median 4.1e-3,
  max 9.7e-3; p99 |dP| ~1e-3; |dP| at the production argmax median
  2.5e-4.
- Cost: wall/src 8.06 s -> 4.85 s (**1.66x cheaper**) in this harness.

Reading: polish-free detection is RECOVERY-NEUTRAL at the production
arm (the audit's projected 1.2-1.5x is measured at 1.66x here). BUT the
off-peak spectrum differs at up to ~1e-2 — same caveat class as K.18:
detection argmax is robust, FAP/null calibration and any peak-height
thresholding would see a slightly different distribution, so adoption
(John's call) should pair it with the empirical-null arm recalibrated
under the same settings. Candidate-refinement at the peak would still
run the full polish (it is per-candidate, cost-negligible).

Caveats: single (mag, N) cell; synthetic PS1-like cadence; RRab H4 arm
only (RRc H2 shares the mechanism; not separately measured).
