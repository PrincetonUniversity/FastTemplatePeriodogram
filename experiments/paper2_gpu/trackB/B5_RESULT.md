# B5 — real PS1 inputs + bottom-up cost rebuild (2026-07-18, PARTIAL)

Audit §1 voided the "$30-70" headline for two gaps: cost anchors at
nfreq=8000 vs the real grid, and no explicit K_vocab factor. This note
closes both and rebuilds the cost bottom-up. The remaining open input is
the faint-candidate count N_cand (CasJobs-blocked, see below).

## 1. Pinned inputs

**Baseline T (literature, Sesar et al. 2017, arXiv:1611.08596):** PS1 3pi
light curves span a **4.5-year period** (~1640 d) with **<~12 epochs in
each of five bands** (grizy). The Track-B harness values (T=1600 d,
N/band=8 griz at the faint end after censoring) are consistent;
CasJobs-side verification of the faint-end N/band distribution remains a
WP P0.2 item.

**Frequency grids** (converged-grid formula df = 1/(os*H*T), os=3,
T=1600 d, Bailey-box bands from the locked plan):
- RRab detection, staged **H=4**, band 0.9-3.4 c/d: **nfreq = 48,000**
- RRc detection, **H=2**, band 1.8-6.5 c/d: **nfreq = 45,120**
(The A100 session-A throughput was measured AT 45k nfreq and is invariant
8k->45k, so the GPU per-unit anchors already hold at the real grid.)

**K_vocab:** detection scans a K=4 vocabulary per subtype (B1 setup; B3
will measure the K-lever on held-out recall). Cost scales ~linearly in K
minus the sums-reuse discount (measured ~6% @H8 on GPU; the CPU catalog
path amortizes much more, see below).

## 2. Per-candidate detection cost (staged H, K=4 both arms)

CPU (post-SAFE-stack, single M5 P-core, measured safe_stack/BENCH.md):
- RRab H4: 0.301 s per 8k freqs (K=4) x 48/8      = **1.81 s/LC**
- RRc  H2: 0.162 s per 8k freqs (K=4) x 45.12/8   = **0.91 s/LC**
- **Total ~2.7 s per candidate per M5 core** (~0.76 core-hours per 1000)

GPU (A100, session-A measured 0.55-0.59 us/LC/f/T @H8; H4 bracketed
0.28-0.33, H2 0.16-0.21 — H4 row is a session-B item):
- RRab H4: 48,000 x 4 x ~0.30 us = **0.058 s/LC**
- RRc  H2: 45,120 x 4 x ~0.185 us = **0.033 s/LC**
- **Total ~0.091 s per candidate** (~25 GPU-h per million)

## 3. Bottom-up cost table (detection scan; injections separate)

| N_cand | A100 (25.3 h/M, $1.49/h) | cloud 64-vCPU CPU (a) |
|---|---|---|
| 0.3 M | ~8 h, **~$11** | ~14 h, ~$35 |
| 1 M | ~25 h, **~$38** | ~47 h, ~$118 |
| 3 M | ~76 h, **~$113** | ~142 h, ~$355 |

(a) assumes cloud vCPU ~ 0.5x an M5 P-core and ~$2.5/h per 64-vCPU node;
the playbook's CPU-only decision stands on budget-dial grounds, but note
the SAFE stack narrowed GPU-vs-CPU to ~39x/core (was ~135x) — at these
prices the A100 is now the cheaper scan engine and session-B H4 rows
would firm its column. Injection campaign: ~$4-12 per 100k injections
(same per-LC cost; alias-partner windowing, if validated, cuts 13-18x).

**The binding unknown is N_cand.** Everything above is exact per-unit;
total cost is linear in the faint shortlist size after cuts (point
source, |b|, variability pre-cut, color box).

## 4. CasJobs status (blocked this session)

MAST CasJobs (mastweb.stsci.edu/mcasjobs) reachable but **Not Logged
in**; logging in requires John's password, which the agent does not
handle. Options to unblock WP P0.2 / N_cand:
1. John logs into MAST CasJobs in Chrome once (session persists), or
2. runs a pilot-field count via the account-free MAST DR2 API
   (verified working per PAPER2_BRIEF WP P0.2), or
3. the playbook's free PS1 Parquet on S3 path (no account at all).

Until then the §3 table is quoted per-million with scenario rows, NOT a
single headline.
