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
