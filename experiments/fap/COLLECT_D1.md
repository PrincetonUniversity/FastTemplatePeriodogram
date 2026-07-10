# WP D1 collection guide (for the next unattended queue session)

WP **D1** (measured null / FAP calibration) launched a **detached** Monte-Carlo run
on 2026-07-10 ~14:22 CDT. Per the queue's long-compute rule this session ended with
a `WAITING` marker; **collecting/finalizing this IN PROGRESS row counts as your one WP.**

## 1. Check the marker
- Completion marker: `experiments/fap/output_null/DONE.marker`
- Run log: `experiments/fap/run_full.log`
- If the marker is **absent** and `run_full.log` still shows no `[fap]   done`,
  the run is still going (or died). Check `pgrep -fl measure_null.py`.
  - Still running → re-write `WAITING` at the project root and END (another cycle).
  - Died (no process, no marker, traceback in the log) → **ESCALATE** (write
    `ESCALATE.md`); do not silently relaunch.
- If the marker is **present**, proceed.

## 2. Verify the deliverables + acceptance gate (re-measure, don't quote)
Run this (venv python), from `FastTemplatePeriodogram/`:
```
.venv/bin/python - <<'PY'
import numpy as np, json
d = np.load('experiments/fap/output_null/null_maxpower.npz')
cfg = json.loads(str(d['config_json']))
print('n_real', cfg['n_real'], 'n_obs', cfg['n_obs'], 'n_freq', cfg['n_freq'])
# ACCEPTANCE GATE (audit): pure-noise max power ~0.5 at N=40
for k in ['ftp_H1','ftp_H3','ftp_H6','ftp_H12','gls']:
    a=d[k]; print('%-8s mean=%.4f median=%.4f p99=%.4f'%(k,a.mean(),np.median(a),np.quantile(a,.99)))
# invariants
print('FTP H=1 == GLS (indep cross-check) max|diff|=%.2e'%np.max(np.abs(d['ftp_H1']-d['gls'])))
vpt=d['vocab_per_template']
print('cat_K8 == max over 8 vocab templates? %.2e'%np.max(np.abs(d['cat_K8']-vpt.max(1))))
cats=np.stack([d['cat_K1'],d['cat_K2'],d['cat_K4'],d['cat_K8']],1)
print('cat monotone in K per realization:', bool(np.all(np.diff(cats,1)>=-1e-15)))
print('all finite:', all(np.all(np.isfinite(d[k])) for k in ['ftp_H1','ftp_H3','ftp_H6','ftp_H12','gls','cat_K1','cat_K8']))
PY
```
Gate PASSES if the FTP/GLS mean/median/p99 max powers at N=40 are ~0.5 (canary gave
mean ~0.42-0.44, p99 ~0.51 — in the audit's "~0.5" ballpark), FTP H=1 == GLS to <~1e-8,
catalog-max nesting exact, monotone, all finite. If the anchor is wildly off (e.g. ~0.05
or ~0.99) or an invariant fails → **ESCALATE**, do not commit.

Also open `experiments/fap/output_null/SUMMARY.md` (auto-generated from the data) and the
two figures `fap_vs_threshold.png`, `nullmax_vs_K.png`.

## 3. Adversarial verification (headline-bearing result — required)
This null is a headline-number-bearing result feeding D2. Spawn ONE bounded verifier
agent (xhigh, <30 s script, <=~50 cells) with a SINGLE claim: *"Independently confirm
the pure-noise FTP single-template max-power null at N=40 is ~0.4-0.5 and that FTP H=1
equals a floating-mean Lomb-Scargle to <1e-8 — write your own tiny GLS/χ² oracle on a
handful of fresh pure-noise light curves at the same [1,5] cyc/day grid; do NOT read
measure_null.py's outputs."* Record its verdict in `VERIFICATION.md` (WP D1). If it
refutes → ESCALATE.

## 4. Commit + finalize
- `cd FastTemplatePeriodogram`
- `git add experiments/fap/measure_null.py experiments/fap/COLLECT_D1.md experiments/fap/output_null/null_maxpower.npz experiments/fap/output_null/SUMMARY.md experiments/fap/output_null/fap_vs_threshold.png experiments/fap/output_null/nullmax_vs_K.png`
  (do NOT commit `run_full.log` or `DONE.marker` — runtime artifacts.)
- Commit prefix: `experiments: fap ` (e.g. `experiments: fap measured null / FAP calibration (WP D1)`).
  End the commit body with the `Co-Authored-By` trailer per Git hygiene.
- Push `dev`.
- Update the **D1 row** in `EXECUTION_PLAN.md` Status table to `DONE` with the
  re-measured anchor numbers + the adversarial-verify verdict.
- Supersede the PLAN.md "Status & next session" handoff.
- Remove `experiments/fap/run_full.log` and `DONE.marker` if you like (optional).
- END (one WP per session). Next eligible WP after D1 is **C5** (requires B8-closure —
  still held for the supervised RunPod session — so C5 stays blocked; the next Opus-
  approved *unblocked* WP is **E1** docs-only, then E2/E3).

## Config recap (what was run)
Pure-noise single-band LCs, `SyntheticCadence` (3-yr baseline, 270/365.25-d seasons),
`exp_mag_error` synthetic model, mean_mag=15, amplitude=0 (=> homoscedastic sigma≈0.010).
FTP **scan** path (C4 default) at H∈{1,3,6,12}; GLS baseline; grid [1,5] cyc/day,
n_freq=10000 (>= B7 guard floor 8767; = rerun_202606 10k grid); N_obs=40 (audit anchor);
n_real=3000; PAM vocab = order-8 Sesar medoids, catalog-max nested over K∈{1,2,4,8};
GEV tail via scipy.stats.genextreme. Per-realization seed = seed_base(1e6)+idx.
