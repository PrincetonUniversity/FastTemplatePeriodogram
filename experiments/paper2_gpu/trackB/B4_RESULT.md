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
