# WP B8 consolidated rerun — RunPod cost estimate (2026-06-11)

Escalation artifact required by `EXECUTION_PLAN.md` WP B8 ("ESCALATE first with a cost
estimate"; budget cap ~$60). Prepared before any pod is launched.
**Approved by John 2026-06-11** (full batch incl. arm f, canary-first).

**Post-approval amendment (same day): grid guard forces 10k freqs.** The headline 8k
grid fails WP B7's own hard guard at T=3yr (df=5.0e-4 > 0.5/T=4.56e-4; 1.83 grid
points per Rayleigh width — the rerun would SystemExit at startup). Production arms
therefore run `--n-freq 10000` (2.28 pts/Rayleigh; hard tier passes, warn tier fires
honestly) and the convergence arm (c) runs 20000 to keep the intended 2× ratio
(spec said 16000 against an 8k production grid). FTP/baseline cost is ~grid-linear:
arms (a)/(b)/(d) ×1.25, (c) ×2.5 vs the table below. Amended bottom line, per the
`fleet_b8.py` job table: **(a)–(e) 910 core-hr ≈ $37; with (f) 1098 ≈ $45**; +10%
overhead ≈ **$41–50 all-in** — still under the $60 cap, (f) still gated at $45
projected.

## Measured anchors (2026-06-02 production fleet — not guesses)

| anchor | value | source |
|---|---|---|
| Full production job (one universe×seed, 256 src / 8k freq / K≤8, ALL stages) | **~81 core-hr** (~3 h on cpu5c/32) | `fleet.py:35-38` comment, measured |
| Fleet of 6 such jobs | **~$20** | same |
| Effective rate | **~$0.041/core-hr** (≈$1.31/h per 32-vCPU cpu5c SECURE) | derived from the two above |
| BV job penalty vs sesar | ~1.5× core-hr (560-template griz library; the 7–8 h walls were partly cpu3c silicon) | PLAN.md compute saga |
| Blended per-(freq × template × source) eval at H=8 | ~3.2 ms (81 core-hr ÷ 44 fit-units × 256 src × 8k freq); cost-panel timing brackets it at 3–7 ms | derived |

Stage shares of a full job, from FTP template-fit units (sparse K-sweep ΣK=15, dense
K-sweep 15, N-sweep@K=2 12, fixed-K 2, cost panel + greedy ≈ small):
**sparse-K ≈ 34 %, N-sweep ≈ 27 %, dense-K ≈ 34 %, rest ≈ 5 %.**
So "N-sweep + sparse-K only" ≈ 60 % of a full job ≈ **50 core-hr** (sesar) / **75** (BV).

## Per-arm estimate

| arm | scope assumed | core-hr | est. cost |
|---|---|---|---|
| (a) MHLS-capped N-sweep + sparse K-sweep | sesar seeds 0–2 + BV seed 0; CE estimator folded in (+0.02 s/src — negligible) | 3×50 + 75 = 225 | $9.2 |
| (b) robustness arms ×3 (holdout 0.5, cross-universe, band-amp 1.4) | each 1 seed × (sparse-K + N-sweep@K=2) | 3×50 = 150 | $6.2 |
| (c) grid convergence | sesar N-sweep @ n_freq=16 000, 1 seed (2× grid ⇒ 2× FTP cost) | 44 | $1.8 |
| (d) empirical-error revalidation | full headline config, 1 seed sesar, `--err-model empirical` | 81 | $3.3 |
| (e) joint/EM | (3 scenarios + low-SNR amplitude arm) × 3 seeds, 256 src, H=8, 2k freq, max_iter 8 w/ early stop; E-step refits every source over the full grid per iter | 220–340 | $9–14 |
| (f) OPTIONAL | BV seeds 1–2 at arm-(a) scope + CE baseline curve | 150 | $6.2 |

**Nominal (a)–(e): ~720–840 core-hr ≈ $30–34.**
**With (f): ~870–990 core-hr ≈ $36–40.**
**All-in with ~10 % pod/bootstrap/idle overhead: ~$40–45** → ~1.4× headroom under the
$60 cap. Wall-clock with sharded jobs across ~12–14 pods: ~3–4 h compute; overnight easily.

## Protocol (the "sizing was 15× off once" lesson, PLAN.md)

1. **Canary first**: launch ONE arm-(a) sesar shard + ONE joint/EM cell pod; compare
   measured core-hr (START→RESULT beacon timestamps) against this table. Release the
   rest only if within ~2× — otherwise stop and re-estimate.
2. Arm (f) launches **only** after (a)–(e) results are beaconed and projected spend ≤ $45.
3. Hard kill: any pod past 12 h wall gets `cleanup`-terminated; `collect` runs every
   few hours so finished pods never idle-bill.

## Implementation deltas needed before launch (no cost impact)

- `run_production_matrix.py`: stage-selection / `--shard` flag (skip dense-K + cost
  panel in arms a/b; split K-values / N-sweep across pods — flattens the slow-job tail).
- `fleet.py`: bootstrap currently posts only `seed_<s>.json`; B1's per-source npz
  (needed for Wilson/McNemar + B5 alias re-scoring) must come back too — post a tar.gz
  of the npz dir (KB-scale, fine for webhook.site).

## Mid-run decision log (2026-06-11/12, during the live run)

1. **Canary gate, applied per family.** Production canary a-sesar-0: 100.8 core-hr vs 63
   est = **1.6×** → gate passed → arms (a)–(d) released (8 pods, 19:03 UTC). Joint/EM
   canary e-canary: 134.6 core-hr vs 25 est = **5.4×** → gate breached → the four
   remaining e-pods were **never launched**. This is a deliberate per-family reading of
   the "within ~2× or stop and re-estimate" protocol: the production estimate was
   validated, the joint estimate was not.
2. **Arm (e) DEFERRED to post-C2.** Remainder = 11 (scenario, seed) runs ≈ 1,481 core-hr
   ≈ **$61 at current code** — alone over the cap. Mechanical decomposition of the 5.4×
   (audit finding B8E-2): 2.62× uncounted full-grid passes (B6 val-margin pass per EM
   iteration + 2× assignment_accuracy) × ~2× from assignment_accuracy being a SERIAL
   per-source loop billing 32 idle vCPUs (~87 % of wall). Post-C2 projection $4–6 at the
   ≥15× C2 gate, **IF** it transfers to this eval shape (NFFT sums are not accelerated —
   Amdahl risk). Pre-relaunch de-risks: (i) parallelize or eliminate the
   assignment_accuracy recompute (alone → ~$9.7 even pre-C2); (ii) re-grid the
   non-lowsnr scenarios (e.g. N ∈ {4,5,6,8,12}) — the canary shows 4/5 cells saturated
   (gap ≡ 0 by construction) and EM-accepted iterations only on saturated cells, so the
   current grid cannot falsify the MRA-gap hypothesis outside e-lowsnr (finding B8E-1).
3. **Arm (f) CUT** (BV seeds 1–2): the $45 gate is unreachable after the e-canary spend.
4. **--fixed-k 2 pinned** in all N-sweep arms (vs per-seed knee derivation): keeps
   shards/arms K-comparable. Side effect: seed-0's sparse PAM curve now peaks at K=1
   (knee_k → 1 vs headline 2; paired K1−K2 contrast +0.023 ± 0.019, NOT significant —
   curve is flat in K within noise). Formal knee re-derivation happens at the pooled
   3-seed finalize with an SE-aware rule. Exception: d-empirical was launched WITHOUT
   the pin (oversight, finding B8F-6) — its N-sweep K is its own derived knee; check
   comparability when it lands.
5. **Acceptance-gate escalation (>1 SE) WILL FIRE at finalize** (finding B8V-1): every
   unsaturated FTP cell moved UP 1.3–5.1 SE vs headline, MHLS up 10–17 SE (the B2 cap
   fix), ftp_greedy N=4 down ~2 SE (B4 disjoint-selection fix), GLS/MBLS static (<1.6 SE).
   Mechanism: the headline 8k grid was below B7's own hard guard — H=8 methods were
   grid-limited, H=1 baselines were not. Disposition: supersede headline numbers with
   the rerun; arm (c) 20k-vs-10k is the convergence closure check.
6. CE baseline folded into ALL production arms (spec had it under optional (f)) — cost
   ≈ 0; delivers the (f) CE curve early.

## Ordering note (decision surfaced, not made here)

`EXECUTION_PLAN.md:41` nominal chain runs C1–C4 (scan+polish, measured 24–43×) BEFORE
B8; rerunning after C-track would cut FTP-dominated arms ~5–10× (→ roughly $10–15
total). Running B8 now costs ~$25–30 more but (i) decouples the consolidated science
rerun from a brand-new numerical path (C2's gate is ≤1e-12 identical powers, so B8
results remain valid regardless), and (ii) runs overnight in parallel with local
C-track work. Either order is defensible under the cap.

## B8-closure canary (2026-07-10/11)

**Outcome: NO timing data — the canary pod never booted.** Measured facts only:

- Pod `kc626ub1i8b9zk` (`b8c-e2-jitter-0`, cpu3c/32 SECURE, $0.96/h ⇒ $0.030/core-hr
  list), rented 04:35:26 UTC 2026-07-11 on machine `aehmymvwnx01`. RunPod telemetry
  read `uptimeInSeconds: 0`, CPU 0 %, mem 0 % for the pod's entire 50-minute life;
  zero beacons (no START/PROG/RESULT) on the fresh webhook token, while a supervisor
  test POST to the same token returned 200 (ingest path proven good). Conclusion: the
  container never started — stuck host/image-pull, billing anyway.
- Supervisor DELETEd the pod at 05:26:17 UTC (before the 2h40m deadline; flat
  telemetry made waiting pure burn). **Wall 50.9 min ⇒ $0.81 spent, 0 core-hr of
  science.** GET /pods after teardown: account has zero pods.
- The second canary job **g-refine-sesar-0 was never launched** — `state.json` still
  shows it `pending` (no pod, no create attempt recorded). Only 1 of the 2 canary
  pods ever existed. So BOTH canary gates (e2 family and g-refine family) remain
  **unvalidated**.
- Consequence: the B8E-2 headline check — assignment_accuracy share of wall vs the
  old ~87 % — is **unmeasured**. The internal self-bound (`sleep 9000`) never armed
  either; an external watchdog was the only kill path. (New failure mode for the
  protocol list: pods that bill while never booting emit NO beacon at all, which
  `killstale <hours>` catches only after `<hours>`; a boot watchdog should kill any
  pod with no START beacon within ~20 min.)

### Extrapolation (falls back to the job-table estimates — NOT canary-validated)

16 jobs pending per `fleet_b8_closure.py`/state: arm (a) = 12 e2 jobs
(4 scenarios × 3 seeds, incl. `--amplitude 0.1`) × est 10 core-hr; arm (b) = all 4
g-refine jobs (3 sesar × 45 + 1 bv × 68). At the June effective $0.041/core-hr:

| | core-hr | est. cost |
|---|---|---|
| arm (a) e2 relaunch | 120 | **$4.92** |
| arm (b) grid-refine | 203 | **$8.32** |
| total | 323 | **$13.24** |
| total ×1.5 safety | 484 | **$19.86** |

$19.9 ≤ $25 cap **nominally**, but the ×1.5 does not cover the e2 estimate risk the
canary was supposed to retire: the 10 core-hr/job figure assumes the B8E-1/B8E-2
de-risks (re-grid + parallel assignment_accuracy + C4 scan transfer) deliver ~13×
vs the June-measured 134.6 core-hr e-canary. Sensitivity:
- **cap-breach threshold**: e2 > ~34 core-hr/job (3.4× est) breaches $25 (given
  arm (b) at table value);
- **mid** (only assignment-parallelization delivers, June decision-log arithmetic
  ≈ 21.5 core-hr/job): total ≈ $18.9, ×1.5 ≈ $28.4 — over cap with safety margin;
- **worst** (June repeat, 134.6): arm (a) alone ≈ $66 — 2.6× over cap.
Arm (b)'s ×1.4 `--refine` overhead is likewise unmeasured (June a-sesar-0 anchor
covers only the unrefined N-sweep), and cpu3c silicon is the June fleet's known-slow
flavor.

**Recommendation: do NOT release the full batch.** Re-run the canary pair (both
jobs this time), with a boot watchdog (no START beacon within ~20 min → kill + retry
next flavor, skip machine `aehmymvwnx01`), and hold the release behind the
protocol's within-~2× gate per family. Canary retry cost ≈ $2–3.

Ledger to date for B8-closure: $0.81 burned, nothing measured.

## B8-closure boot-failure post-mortem: root cause found + fixed (2026-07-11)

**Overnight orphan spend.** The canary retry (launched 2026-07-10 evening) created
pods `u4jeu6pyzbkiuy` (`b8c-g-refine-sesar-0`) and `z7khjh3glpht1e`
(`b8c-e2-jitter-0`), both cpu5c/16 SECURE at $0.56/h, on two *different* machines
(`3n7hlaf04ogy`, `qblpxryxs2k6`). Same signature as `kc626ub1i8b9zk`:
`uptimeInSeconds: 0`, CPU 0 %, **zero beacons**, billed as RUNNING. The launching
session was killed by a session limit before its foreground `bootwatch` loop could
run, so both pods sat orphaned ~7.4 h overnight: **~$8.25 spent, zero science.**
Both torn down next morning; GET /pods confirmed the account empty.

**Root cause (diagnosed by diffing against the June fleet.py, which booted 10/10).**
The pod-create payloads are field-for-field identical (same `imageName: python:3.11`,
`computeType`/`cpuFlavorIds`/`vcpuCount`, `containerDiskInGb: 20`, `ports`,
`cloudType`, `dockerStartCmd: ["bash", "-c", script]`). The difference is *inside
the script*: `fleet_b8_closure.py`'s new `_prog_beacon()` returned a snippet ending
in a bare `&`, and `_bootstrap()` joins its lines with `" ; "`, producing

    ... ) >/dev/null 2>&1 & ; cd "$REPO/experiments/phase3_recovery" ...

`cmd & ; next` is a **bash syntax error**, and `bash -c` aborts the *entire* command
string at parse time — nothing executes, the container dies in milliseconds, RunPod
restart-loops it at uptime ~0 while billing. Verified locally: `bash -n` on the
generated script exits 2 with `syntax error near unexpected token ';'` (June
fleet.py's script parses clean). This is host/flavor-independent, exactly matching
3/3 failures on 3 different machines and 2 flavors. Machine `aehmymvwnx01` was
innocent; the `BAD_MACHINES` blacklist has been cleared.

**Fix (commit `620df98` on `dev`).**
- `_prog_beacon` now terminates the backgrounded group with `& true` (join-safe:
  `cmd & true ; next` is valid bash);
- `plan` and `_create` both run `bash -n` on every generated bootstrap script and
  refuse to proceed on a parse error — no pod can ever again be created with an
  unparseable start command;
- `_fleet_b8c/state.json` (gitignored): the two dead pods moved to job `history`,
  `boot_attempts` reset to 0 (the failures were our bug, not boot flakiness, and
  must not burn the jobs' retry budget or blacklist cpu5c).

**Micro-test (2026-07-11, $0.04): PASS.** One minimal pod (`apl2fflwxobqhr`,
cpu5c/32 SECURE via the fixed `_create` path, including the exact fixed
`( ... ) >/dev/null 2>&1 & true` background pattern) posted its START beacon to the
B8-closure webhook token **24 s** after create and its DONE beacon 60 s later — both
well inside the 10-min gate. Pod deleted at t+1.7 min; GET /pods after: **account
empty**. Containers boot and execute the fixed script structure end-to-end.

**Ledger to date for B8-closure:** $0.81 (first canary) + ~$8.25 (overnight
orphans) + $0.04 (micro-test) ≈ **$9.10 of the $25 cap; $15.90 remains.** Nothing
scientific measured yet; the canary pair is still the next gate (est. $2–3), now
unblocked.

## B8-closure canary pair — measured (2026-07-11)

Both canary pods launched 13:14 UTC via the fixed launcher (commit `620df98`) and
**booted first attempt** — START beacons ~2.5 min after create (boot attempts 1/2
used per job; the `& ;` fix is production-verified). Neither job posted a RESULT;
both were killed in-protocol. Supervised end to end in foreground; account holds
zero b8c pods after teardown (GET /pods survivor = John's external `cuvarbase-dev`,
untouched).

| job | pod / flavor | wall | billed | outcome |
|---|---|---|---|---|
| e2-jitter-0 | `3b4hxssly4oemn` cpu5c/32 SECURE ($1.12/h) | 8 394 s (2h20m) | **$2.61** / 74.6 core-hr | INCOMPLETE — killed at the 2h15m deadline (+5 min poll latency); driver burned ≥73.3 core-hr, no RESULT |
| g-refine-sesar-0 | `813dcqknmnhg3z` cpu5c/16 SECURE ($0.56/h) | 5 251 s (1h28m) | **$0.82** / 23.3 core-hr | killed early by supervisor: on 16 vCPU the est-45-core-hr job needs 2.8 h wall — mathematically outside the 2h15m deadline; PROG quota (4) already exhausted |
| **canary pair total** | | | **$3.43** | (≈$4.0 at the June effective $0.041/core-hr) |

### What was measured (PROG-beacon salvage + timestamps)

**e2 (arm a): the B8E-2 assignment fix DELIVERED; the job estimate still fails
its gate by ≥7.3×.** The only cell to complete inside the PROG window, N=4
(3 EM iterations), timed: `em=992.6 s (62.2%) | eval=142.5+142.2=284.7 s (17.8%) |
gls=35.3 s (2.2%) | assign=141.9+141.5=283.4 s (17.8%)` — accounted cell wall
1 596 s × 32 vCPU = **14.2 core-hr for ONE of five N-cells**.

- **assignment_accuracy share = 17.8%** of accounted wall, vs the June ~87%
  serial baseline (local 2-worker smoke: 23%). Parallelization confirmed.
- **Job total ≥73 core-hr** (floor: driver killed at 2h17m with no RESULT) vs the
  10 core-hr estimate — **≥7.3×, gate FAILED**. 5 equal cells project 71 core-hr
  + startup, yet 73.3 were burned without finishing: later cells run *slower*
  (larger N and/or more EM iterations). Central if-completed estimate
  **~90 core-hr/job** (floor 73).
- Why the ~13× assumed speedup did not appear: the E-step (`em`, full-grid NFFT
  refits per iteration) now dominates at 62% — the C4 scan speedup does not touch
  this code path, and removing the serial assign bottleneck removed a smaller
  absolute share than the June decomposition implied. vs the June-measured
  134.6 core-hr e-canary: measured floor = 0.54×, central ≈ 0.67× — a real ~1.5×
  improvement, nowhere near estimate.

**g-refine (arm b): refine overhead UNMEASURED.** The pod landed on 16 vCPU
(the 32-vCPU cpu5c pool was exhausted by the e2 create 2 s earlier), which made
the est-45-core-hr job deadline-infeasible from the start. Salvage: greedy
ordering (dense, 64 sel-src × 2000 sel-freq) = 1 627 s × 16 = **7.2 core-hr**;
no N-sweep cell line before the PROG quota. The 45 core-hr table estimate and
the ×1.4 `--refine` factor remain unvalidated (23.3 core-hr burned is consistent
with, but does not confirm, 45). **Placement lesson: g-refine jobs must pin
32 vCPU** (or the deadline must be ≥3.5 h).

### Extrapolation (measured canary + launcher job table, ×1.5 safety)

Remaining cap after this canary: $15.90 − $3.43 = **$12.47**.

| | jobs | core-hr | $ @0.041 | ×1.5 |
|---|---|---|---|---|
| arm (a) e2, measured floor 73/job | 12 (incl. jitter-0 rerun) | 876 | $35.9 | $53.9 |
| arm (a) e2, central 90/job | 12 | 1 080 | $44.3 | $66.4 |
| arm (b) g-refine, table est (unvalidated) | 4 (incl. sesar-0 rerun) | 203 | $8.3 | $12.5 |
| **batch (a central + b)** | 16 | 1 283 | **$52.6** | **$78.9** |

**The batch does NOT fit: $52.6 (no safety) / $78.9 (×1.5) vs $12.47 remaining —
and arm (a) alone at its measured floor is ~3× over the remainder. Even arm (b)
alone ×1.5 consumes the entire remainder.**

**Recommendation: do NOT release the batch.** Decision options for the supervisor:
(i) re-defer arm (e2) — it fails the within-~2× gate at ≥7.3× and needs either an
E-step algorithmic cut (the 62% `em` share) or a scope cut (sources/grid/seeds)
before pods make sense; (ii) if arm (b) is wanted alone, first validate with ONE
g-refine canary pinned to 32 vCPU (~1.4 h, ~$1.6–1.9 at table est, deadline-feasible)
— arm (b) at table value then costs ~$8.3 more, which fits the remainder only
without the ×1.5 margin and only if arm (a) is dropped.

**Ledger:** $9.10 + $3.43 = **$12.53 of the $25 cap; $12.47 remains.**

## B8-closure arm (b) 32-vCPU canary -- measured, gate FAILED (2026-07-11)

Arm (a) e2-* was DROPPED for this run (measured >=7.3x over estimate; escalated
to John separately); this session ran arm (b) only, task budget $12, canary-first
with g-refine-sesar-0 PINNED to 32 vCPU (new ``B8C_PIN_VCPU`` env in the
launcher + per-job cpu3c avoid; new ``launch <tag>`` subcommand so the e2 canary
tag cannot be relaunched by accident).

| job | pod / flavor | wall | billed | outcome |
|---|---|---|---|---|
| g-refine-sesar-0 | `ehrzmk00e1cq0d` cpu5c/32 SECURE ($1.12/h) | 2.87 h | **$3.21** / >=91.2 core-hr | INCOMPLETE -- killed 18:37Z at the hard 2h45m deadline (+grace poll); no RESULT beacon, so NO refined per-source data landed |

Timeline: create 15:45:25Z -> fresh START 15:46:00Z (35 s boot; image cached;
the 620df98 fix holds) -> greedy ordering done in 836 s -> N-sweep silent by
design (single log line only at sweep end; all 4 PROG posts show only the
greedy line) -> deadline breach 18:30Z -> killed 18:37Z. Account after: only
`cuvarbase-dev` (untouched; launcher guard now also matches the pod NAME --
the old id constant had gone stale).

### What was measured

- **Greedy phase is core-hr-stable across vCPU**: 836 s x 32 = 7.43 core-hr vs
  7.2 core-hr on the killed 16-vCPU attempt. Good sanity anchor.
- **The refined N-sweep alone burned >=83.7 core-hr without finishing**, vs the
  June UNREFINED N-sweep measurement of ~44 core-hr (a-sesar-0 share). So the
  ``--refine`` overhead is **>=1.9x** (floor -- job never completed), not the
  ~1.4x the RefinedEstimator cost formula predicted, and any C4-speedup offset
  assumed in the est-45 arithmetic did NOT materialize on this path.
- **Job total >=91.2 core-hr vs est 45 -- >=2.03x, canary gate FAILED** (the
  protocol's within-~2x threshold, and a floor: completion cost unknown).

### Extrapolation and release decision (task budget $12)

At the measured FLOOR (91.2 core-hr/job; bv x1.51): sesar jobs >=2.85 h wall
(**already past the 2h45m deadline**) >= $3.20 each; bv >=4.3 h (**deadline-
infeasible**) >= $4.83. Batch = $3.21 (canary) + (2 x $3.20 + $4.83) x 1.3 =
**>=$17.8 vs the $12 task budget -- batch NOT released** (gate failed on both
budget and per-job deadline feasibility; floors, so the true overrun is larger).

### Acceptance analysis status

**Cannot run**: the per-cell paired McNemar (refined vs 10k) requires the
refined per-source npz, which only ship in the RESULT beacon. Zero arm-(b)
cells landed. The 10k-grid rates therefore REMAIN "lower bounds at the 10k
production grid" (SUMMARY_c-grid20k.md disposition unchanged). The analysis
is implemented and smoke-tested in ``finalize_b8_closure.py`` (pairs refined
vs arm-(a) masks per cell, asserts identical ``p_true``, pools sesar seeds,
marks the 8 arm-(c) flagged cells, writes ``SUMMARY_g-refine.md`` +
``aggregate_g_refine.json``) -- ready the moment real g-refine results land.

### De-risks before any re-attempt (decision John's)

1. **Shard the N-sweep by N-value** (6 pods x ~15-16 core-hr at the measured
   floor => ~30 min wall each on 32 vCPU; same total cost, deadline-trivial,
   launcher job-table change only).
2. Or raise the per-job deadline to >=3.5 h (sesar) / >=5 h (bv) and accept
   ~$3.4-5 per job -- but first measure refine overhead to completion once,
   locally at reduced scale, so the next canary gate is against a validated
   number, not another 1.4x assumption.
3. The June effective rate ($0.041/core-hr) now looks optimistic for pinned
   32-vCPU SECURE ($1.12/h = $0.035/core-hr list, but only when jobs FINISH).

**Ledger:** $12.53 + $3.21 = **$15.74 of the $25 cap; $9.26 remains.** Arm (b)
still has zero refined cells; arms (a) and (b) both await re-scoping.

### 2026-07-12 sharded-batch supervision run (per-N shards, ceiling $28)

24-shard plan (4 jobs x N4 + 5 injected siblings each). All 4 N4 baseline
shards (sesar-0, sesar-1, sesar-2, bv-0) were killed at their FIRST deadline
(2.0 h sesar / 3.0 h bv) with no RESULT beacon, auto-retried once on a fresh
pod per the fleet driver's one-retry policy, and ALL FOUR retries also hit
their full deadline with no RESULT -- each job marked `FAILED: 2x deadline
kills` by the driver. Zero g-refine `results.json`/per-source npz landed
under `rerun_202606/raw/`; the 20 injected N8/N12/N16/N24/N40 shards never
became launchable (gated on an N4 sibling result that never appeared).
`finalize_b8_closure.py` was run per protocol and raised its
`SystemExit('no g-refine results under ...')` guard (not the p_true-mismatch
variant) -- `SUMMARY_g-refine.md` is therefore UNCHANGED (still the prior
NO-DATA breadcrumb). Final `state.json['spend_usd']` = **$20.67** (within
the $28 ceiling). Pod-by-pod: sesar-0-N4 attempt 1 $2.28 + attempt 2 $2.37;
sesar-1-N4 $2.24 + $2.27; sesar-2-N4 $2.30 + $2.38; bv-0-N4 $3.46 + $3.37.
Teardown confirmed zero `b8c-*` pods remain; `cuvarbase-dev` untouched.
Supervisor disposition (re-estimate deadlines vs re-shard further) is
John's/the supervisor's call, not this run's.

### 2026-07-13 CORRECTION — unrecorded E5 RunPod spend (~$21.90)

**Money-accounting fix.** A prior (Fable) session launched the E5 sparse-FTP
fleet on RunPod on 2026-07-11 and it FAILED exactly like the arm-b sharded
batch above — 8 pods (`e5-0..7`) on `cpu3c/32/SECURE`, booted 16:31–16:32
(START beacons received), ran to the 2.0 h STALE limit, killed with **zero
result beacons received/merged**. `_fleet_e5/state.json`: all 8 jobs
`started=True received=False killed=True`, `core_hr` total **534.1** ⇒
**~$21.90** at $0.041/core-hr. Both `_fleet_e5/HANDOFF.md` and
`RECOVERY_2026-07-11.md` wrongly recorded E5 as "$0.00 / never launched"
(the HANDOFF predates the launch; the RECOVERY doc did not reconcile the
later state.json). **Corrected total wrap-up RunPod spend = arm-b $20.67 +
E5 ~$21.90 = ~$42.6.** No live pods remain (verified 2026-07-13).

**Shared root cause (arm-b + E5).** The shard driver posts results ONLY after
ALL tasks in a shard finish (per-task `deadline = now + 1e9`, i.e. infinite),
so on a slow flavor (cpu3c after cpu5c exhaustion) a shard that does not
complete inside the stale window loses EVERYTHING. Not thread
oversubscription (locally disconfirmed: FTP is NFFT + a Python scan loop, not
BLAS-bound — 4-way pinned 24.1 s vs unpinned 22.9 s). Any future RunPod use of
this driver needs: per-task timeout (so a hung/slow task can't sink the shard)
+ streaming/partial result posting + right-sized workers/flavor + canary-first.

**E5 disposition: DROPPED (John, 2026-07-13).** Not a submission gate (E4 met
the HOLD-for-ZTF-demo gate); F4 ships from E4 + a sparse-N-panel deferral note.
No further E5 compute, local or paid.
