# Post-stack efficiency hunt #2 (2026-07-19) — NOT YET IMPLEMENTED

Fresh 5-lens hunt on the POST-SAFE-stack code, every proposal
skeptic-vetted with its own A/B probe (run wf_8afd047f-74b; the hunt:sums
and hunt:python lenses and 12 vet agents died on a usage-credit limit, so
coverage is partial — a resume would complete them). One SOLID win
survived independent verification; recorded here for a future
implementation session (per the "recover and persist, don't go further"
instruction this was NOT applied to the code).

## SOLID (1) — fused diagonal-segment-sum coefficient assembly

**Seam:** `multiband.py::_combine_stacked_shared_amp` (floating_offsets
batched path), subsuming `core.py::batched_YM_MM_from_sums` (the function
carrying the old "TODO: use numpy to speed this up").

**Mechanism** (each flattened (r,c) covariance entry maps to exactly one
MM diagonal):
1. Hoist template-INDEPENDENT kernels once per chunk per band and reuse
   across all K templates: `Pcc = conj(UU)-VV`, `Pcs = UU+conj(VV)`
   flattened to (nf, H^2), band-concatenated, columns permuted to group
   by diagonal (dsum = r+c for CC, ddif = r-c+H-1 for CS); plus
   yc = YC-iYS, cs = C-iS.
2. Per template, the whole {outer(alpha,alpha) weighting + 3*(2H-1)
   np.trace calls + 3 np.stack + 3 ascontiguousarray + per-band
   W_k-weighted accumulation} collapses to ONE (nf, nb*H^2) complex
   multiply by the W_k-folded permuted alpha-outer weights + ONE
   `np.add.reduceat` segment sum per matrix. reduceat's fixed
   within-segment order keeps rounding independent of row count, so the
   pinned chunk-size bitwise invariance
   (test_batched_multiband_chunks_are_recomputed_per_chunk) is PRESERVED.
3. SS is never formed: `SS_diags = conj(CC_diags)[:, ::-1]` exactly
   (SS = conj(CC), conj is exact).

**Measured gain** (independent skeptic ABBA-interleaved medians, pinned
threads): H4 K=4 +11%, H2 K=4 +7-8%, H8 K=4 +13-19%, H4 K=1 +8%,
production nfreq=32000 +8-11%. Assembly slice itself 2.4-3.7x.

**Safety class: ulp-equivalent, NOT bit-identical.** It reassociates the
diagonal/band sums, so it REQUIRES a safe_stack re-golden + parity-harness
rerun before adoption. Measured parity: e2e power maxRel <= 1.4e-15, ZERO
rows past the repo's 1e-13 gate on all production shapes incl. an
adversarial sparse 2-band deferral-zone fixture; combine-output YM/MM/AC
worst rel dev 3.8e-16; argmax identical everywhere; patched chunk-size
64-vs-4096 np.array_equal = True. No deferral-decision flips observed
(a flip would show ~1e-8-scale diffs), though knife-edge flips are
possible in principle and the re-golden must check for them.
Prototype: scratchpad/hunt2/probe_e2e.py::fast_combine.
Bit-identical-only fallback (UU/VV hoist, no reassociation) gives ~2%
e2e — the option if the ulp change is refused.

## Vetted-out (do not re-probe)
- BLAS matmul reducer: fastest raw but NOT row-count-invariant (breaks
  the bitwise chunk pin) + spurious Accelerate FP-flag warnings.
- Non-optimized einsum: invariant but slow.
- chunk_size retune to 8192: REFUTED at nfreq=32k (slower than 4096; the
  8k-grid "win" was a single-chunk boundary artifact).
- Rejected by vet: fast=False direct-sum switch, _scan_pass inner-loop
  rework, NFFT 5-smooth padding, scan-grid floor 128->64, restricting dip
  candidates to r<0.15, hoisting diagonal tensors across the catalog
  (negative result). (Several of these had their vet cut short by the
  credit limit — verdicts are "vet-failed", not necessarily "bad";
  the resume would re-adjudicate.)

## To adopt (future session)
1. Port fast_combine into _combine_stacked_shared_amp behind the
   floating_offsets batched path.
2. Re-golden safe_stack (capture + compare vs golden_fix) and rerun the
   full parity harness + the deferral-contract regression test.
3. Full suite; bench delta; commit as an ulp-equivalent (not
   bit-identical) SAFE-adjacent item with its own parity report entry.
