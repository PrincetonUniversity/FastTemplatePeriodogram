# B4 — deep-dip / deferral rate on real cadences (2026-07-18)

262 cadences (99 real ZTF g/r thinned to N/band 8 and 20; 64 synthetic PS1
TTI-pair griz N=8/band), H in {2,4,8}, RRab band, os=3 H-matched grids,
r-band RRab medoid. Raw: output/b4_dip_rate.json. |MM'| conditioning only —
no injections (MM is y-independent).

| arm | defer (r<3e-3) mean/max | scan-exact (r<0.15) mean/max |
|---|---|---|
| ps1_pair_n8 (griz) | **0 / 0 at every H** | H2 8.1%/9.7%; H4 1.9%/2.5%; H8 1.9%/2.5% |
| ztf_n20 (g/r) | H2 0.00%/0.05%; H4,H8 0/0 | H2 0.7%/12.6%; H4,H8 0.1%/2.7% |
| ztf_n8 (g/r) | H2 0.05%/0.62%; H4,H8 0/0 | H2 18.1%/49.2%; H4,H8 5.5%/22.4% |

Conclusions:
1. **GPU_FEASIBILITY risk #2 RETIRED: deferral ~0 on real cadences** (max
   0.62%, only at H2 on ultra-sparse 2-band ZTF; exactly 0 at H4/H8 and on
   all 4-band PS1-like cadences). The GPU host-round-trip for deferred rows
   is negligible; **Aberth stays deprioritized** (matches the vetted hunt
   verdict). K>=3 bands conditioning MM' well is confirmed (griz ~10x fewer
   scan-exact rows than 2-band at H2).
2. **CPU-side**: the scan exact-fallback zone (r<0.15 -> per-row eigvals) is
   NON-trivial on sparse real cadences (mean 5.5%, max 22% at H4/H8 on
   ztf_n8) — the vetted Bernstein-gated fallback win (audit §2 item 3) is
   MORE valuable on real data than on the clean-synthetic 1.8% it was
   measured at; implement it in the SAFE stack.
Caveat: RRab template/band only; RRc (H2, wider band) shares the H2 columns'
qualitative picture; real-PS1 cadences pending WP P0.2.

---

## CORRECTIONS (2026-07-18 both-sessions re-audit, panel-confirmed)

1. **Cadence composition:** 43/99 cached "ZTF g/r" cadences retain only
   ONE band after quality cuts, and that single-band (K=1,
   rank-deficient at H>=4) subset drives every headline ztf_n8 number:
   genuine 2-band rates are ~5-15x smaller (H2 scan-exact 3.5%/4.9%
   mean/max, H4 0.38%/0.60%; H2 defer max 0.0025%). All 17 cadences
   with nonzero H4/H8 deferral are single-band. The "mean 5.5% / max
   22%" scan-exact zone quoted in conclusion 2 is a single-band-driven
   mixture.
2. **"defer exactly 0 at H4/H8" was wrong** — the raw JSON has 16 (H4)
   / 15 (H8) ztf_n8 rows with nonzero deferral (all single-band; tiny
   but nonzero).
3. **Risk-retirement conflation:** conclusion 1 retired GPU risk #2
   from the r<3e-3 deferral rate, but the code AT THE TIME sent every
   r<0.15 row to host-side eigvals — host exposure was the scan-exact
   column (1.9-8.1% PS1-like, up to 49% worst single-band), not the
   deferral column. The retirement became true only AFTER the SAFE
   stack's Bernstein gate: a probe of the post-fix triage on these same
   cadences finds ZERO eigvals rows (all flagged rows resolve via
   gate-pass or densified rescan). Aberth stays deprioritized — but on
   the strength of the gate, not of this memo's original argument.

4. **The "griz ~10x fewer scan-exact rows than 2-band" claim is false**
   (final panel verdict): the raw JSON gives 2.2x on means (8.14% vs
   18.06% at H2), and against the GENUINE 2-band subset the direction
   REVERSES — griz PS1-pair cadences have ~2.3x MORE scan-exact rows
   than true 2-band ZTF at H2 (~5x at H4), because TTI same-night pairs
   concentrate phase coverage into few distinct nights. The "K>=3 bands
   condition MM' well" conclusion is unsupported by this data.
5. **Thinning artifact:** linspace thinning to N/band destroys
   intra-night clumping (27,549 same-night pairs across the 99 full
   cadences -> 9 at n8), contrary to the loader docstring's
   "preserves clumping" claim (docstring corrected). The ztf_n8 arm is
   therefore a sparse-random stressor more than a "real intra-night
   structure" one.
6. **Uniform synthetic weights:** load_ztf_cadence exposes no
   err_model, so the ZTF arms ran with a constant 0.0236-mag error at
   flat mag 21 — uniform weights, not real ZTF photometric errors.
   |MM| conditioning depends on weights, so real heteroscedastic
   errors would shift the dip rates (direction unmeasured).
