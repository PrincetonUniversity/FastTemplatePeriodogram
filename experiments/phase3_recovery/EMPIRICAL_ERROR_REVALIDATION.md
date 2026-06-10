# Empirical-error re-validation of the recovery headline

**Question.** The synthetic `exp_mag_error` noise model is 1.6-3.5x smaller than the
empirical ZTF photometric error-vs-mag relation measured from real DR cadence
(`ztf_error_model.make_empirical_error_model`, from the cached IRSA fetch). Do the
Paper-1 recovery deliverables (recovery-vs-N_epochs, K-sweep) change under the
realistic, larger errors?

**Answer: no.** At matched settings, swapping the synthetic error model for the
empirical one leaves recovery essentially unchanged.

Recovery-vs-N_epochs at the knee K=2 (Sesar, g+r, bounded local: 48 src, 1 seed,
H=8, T=120 d, n_freq=1500; the *delta* is the result -- absolute numbers differ
from the 256-src/8000-freq/3-yr `headline_full` run):

| epochs/band | 4 | 8 | 12 | 20 |
|---|---|---|---|---|
| FTP(PAM) synthetic | 0.458 | 0.979 | 1.000 | 1.000 |
| FTP(PAM) **empirical** | 0.458 | 0.979 | 1.000 | 1.000 |
| GLS synthetic | 0.146 | 0.708 | 0.958 | 1.000 |
| GLS **empirical** | 0.146 | 0.708 | 0.958 | 1.000 |
| MHLS synthetic / empirical | 0/0/0.583/0.750 | vs | 0/0/0.583/0.771 | |
| mb-LS synthetic / empirical | 0/0.667/0.896/1.0 | vs | 0.042/0.625/0.896/1.0 | |

FTP and GLS are **bit-identical**; MHLS/mb-LS differ only at the ~0.02 level
(sampling noise on 48 sources). K-sweep (sparse and dense) is identical.

**Why.** RR Lyrae have large pulsation amplitudes (~0.5 mag). Even with 1.6-3.5x
larger errors the per-epoch shape-SNR stays high (~10-25 vs ~50 synthetic), so
period recovery is limited by **epoch count and the window function (aliasing)**,
not by photometric noise. The realistic errors do not move the recovery curves.

**Implication for Paper 1.** The published synthetic-error headline numbers are
robust to the realistic ZTF error model -- they do **not** need re-deriving under
empirical errors. (Note the empirical error model is NOT a "low per-epoch SNR"
lever for the §3.4 joint study: it constant-clips at sigma = 0.0393 past mag
~18.5, the cached sample's faint limit, so mean_mag -> 20 does not lower SNR any
further. To reach the low-SNR regime, shrink `amplitude` (e.g. 0.1-0.15) or
raise sigma directly.)

**Caveat.** This holds for the bright headline population (mean_mag ~ 15). The
empirical model is built from a bright cached sample and constant-extrapolates
(sigma = 0.0393) past mag ~18.5, so it UNDERSTATES true faint-end ZTF errors;
if Paper 1's ZTF demo includes faint sources, rebuild the error curve from a
fainter sample rather than trusting the clipped extrapolation.

## Reproduce
```
python experiments/phase3_recovery/run_production_matrix.py --universe sesar \
  --seeds 0 --n-sources 48 --max-templates 30 --n-freq 1500 --baseline-days 120 \
  --dense-master-epochs 40 --sparse-master-epochs 6 --k-values 1,2,4 \
  --n-epochs-values 4,8,12,20 --greedy-select-sources 24 --greedy-select-nfreq 1200 \
  --no-cost --no-figures --n-jobs 8 --err-model {synthetic,empirical} \
  --outdir experiments/phase3_recovery/reval_{synthetic,empirical}
```
