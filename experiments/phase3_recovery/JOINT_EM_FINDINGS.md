# Phase 3.4 — joint (EM/MRA) vs pipeline vocabulary: findings

**Question (design memo §A.4, falsifiable):** does jointly re-estimating the template
vocabulary from the observed population (alternating-minimization / multireference
alignment) beat the one-shot pipeline vocabulary (`build_template_catalog`, k-medoids
of a clean library), and does the advantage grow as data get sparse (MRA ~1/SNR³)?

**Answer (bounded local draft, H=4, K=2, g+r, T=60 d, 64 sources/arm, 1 seed):**
**No.** The joint-minus-pipeline recovery gap is ≈ 0 in every regime tested.

| scenario | how joint gets headroom | gap @ N=4 | gap @ N≥6 |
|---|---|---|---|
| same-library (`joint_draft`, jitter 0) | none (population drawn from the library both arms cluster) | +0.000 | +0.000 |
| zero-mean jitter (`joint_jitter`, jitter 0.15, mean_mag 19) | per-source Fourier perturbation | +0.000 | +0.000 |
| library holdout (`joint_holdout`, 50% disjoint) | population shapes absent from the clustered library | −0.031 | +0.000 |

`−0.031` on 64 sources at recovery ≈0.8 is within one binomial SE (≈0.05) of zero. In the
holdout/N=4 cell the EM *did* iterate and improve held-out **val** recovery
(`best_iter=4/7`) — the machinery works — but the gain does not generalize to a positive
**eval** gap. Everywhere else pipeline already saturates (≥0.98 by N=6), leaving no room.

## Why joint doesn't help here (mechanism)
1. **Tight shape manifold.** RR Lyrae light-curve shapes are low-dimensional and densely
   sampled by the Sesar library; even a *disjoint* 23-template library has a near medoid
   for any population shape, so the pipeline vocabulary is already near-optimal.
2. **Period recovery is shape-robust.** The 1% fractional metric saturates to 1.0 by N≥6
   even with an imperfect template, so a better-matched template cannot raise it (the
   metric clips shape gains — design memo §A.6 "condition on correct-period subset").
3. **Wrong SNR regime.** MRA's pooling advantage is a low-per-epoch-SNR effect (~1/SNR³).
   RR Lyrae have ~0.5 mag amplitude vs ~0.01–0.02 mag errors ⇒ per-epoch SNR ≈ 40–50, so
   single-source shape information is already sufficient and pooling adds nothing. The
   "sparse" axis here is epoch *count*, not SNR.

**Zero-mean jitter is not a valid headroom lever** (learned the hard way): the Bayes-optimal
template for a zero-mean-jittered population is the clean centre, which the pipeline medoid
already approximates — so joint can at best tie and at low N (noisier estimate) ties or loses.
Library holdout is the only realistic lever, and even it yields ≈0.

## Caveats before a publication claim
The negative result is consistent across three scenarios with a clear mechanism, but these
are H=4, single-seed, local-draft runs. To make it airtight:
- H=8, multi-seed error bars (RunPod), both universes (Sesar + BV).
- A **mechanism metric** (correct-template-assignment / power on the correct-period subset)
  to confirm joint isn't learning better shapes that the period metric merely clips.
- A genuinely **low-per-epoch-SNR** run (small amplitude, or mag>20, or the empirical ZTF
  error model — which Track B found is 1.6–3.5× larger than the synthetic default) — the one
  regime MRA theory predicts joint *should* help. For RR Lyrae's large amplitudes the
  high-SNR regime is the realistic one, so the negative result already covers the science case.

## Reproduce
```
python experiments/phase3_recovery/run_joint_vs_pipeline.py --universe sesar \
  --library-holdout-frac 0.5 --n-epochs-values 4 6 8 12 20 --n-sources 64 \
  --n-harmonics 4 --f-min 1.4 --f-max 3.6 --n-freq 2000 --baseline-days 60 \
  --max-iter 8 --n-jobs 8 --out experiments/phase3_recovery/joint_holdout
```
The implementation (`ftperiodogram/joint_em.py`) is correct and adversarially verified
(pooled MRA M-step = exact amplitude-weighted joint minimizer; median aggregator sign bug
found+fixed; no train/val/eval leakage). The finding is about the *science*, not the code.
