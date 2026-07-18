# Track-A pod runbook (A100, staged, $50 cap)

Exact workflow for the two pod sessions in `AUDIT_2026-07-18.md` §4
(session A ≈ 3–4 h ≈ $6 compile/validate; session B ≈ 1–2 h ≈ $3 bench).
**The main session creates/deletes the pod — never a subagent.**

## 0. Preconditions (local, before any pod exists)

- `validate_local.py` prints **0 failed** (23 checks) — rerun if any
  trackA file changed: `.venv/bin/python experiments/paper2_gpu/trackA/validate_local.py`
- `fixtures/*.npz` + `fixtures/gates.json` exist (written by that run).
- Branch pushed or files available to scp (the pod pulls nothing from git).

## 1. Pod creation (REST — GraphQL is Cloudflare-blocked)

API: `https://rest.runpod.io/v1` with `Authorization: Bearer $RUNPOD_API_KEY`.
GPU: **A100-SXM4-80GB** (~$1.49/hr secure; FP64 is the binding constraint —
datacenter cards only, GPU_FEASIBILITY §4). H100 column **only if total
spend ≤ ~$15 by then** (audit §4 — may be dropped this round).

```bash
curl -s -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  -d '{
    "name": "ftp-trackA",
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "gpuTypeIds": ["NVIDIA A100-SXM4-80GB"],
    "gpuCount": 1, "containerDiskInGb": 40,
    "env": {"PUBLIC_KEY": "<contents of ~/.ssh/id_ed25519.pub>"},
    "ports": ["22/tcp"]
  }'
```
(Any CUDA-12.x python image works; pod schema details are in this dir's
git history per HANDOFF §4b.) Poll `GET /pods/{id}` for the ssh host/port.
Record the pod id and START TIME in the session notes immediately.

## 2. Ship files

```bash
scp -P $PORT experiments/paper2_gpu/trackA/{kernels.py,reference.py} \
    experiments/paper2_gpu/trackA/pod/{validate_gpu.py,bench_gpu.py} \
    root@$HOST:/workspace/
scp -P $PORT -r experiments/paper2_gpu/trackA/fixtures root@$HOST:/workspace/
```

## 3. On-pod setup

```bash
nvidia-smi                      # confirm A100-SXM4-80GB, idle
python -m pip install -U pip
pip install cupy-cuda12x numpy  # cupy 13/14.x, CUDA 12
python -c "import cupy; cupy.show_config()"
```

## 4. RUN ORDER (mandatory)

1. **Stage-2 FP64 accuracy spike FIRST** (GPU_FEASIBILITY §8 sequencing;
   audit §4 re-ordering). The weight-concentration fixture is the
   moment-sum-cancellation risk (#1) probe:

   ```bash
   cd /workspace && python validate_gpu.py --stage stage2
   ```

   **ESCALATE + STOP if it fails** (gate: rel ≤ 1e-13 vs the mirror on
   identical sums; expected ~bitwise). Do NOT benchmark a numerically
   wrong kernel. Delete the pod, bring the JSON + stderr home, debug
   locally against `reference.py`.

2. Full validation (compile + all fixtures, ~minutes):

   ```bash
   python validate_gpu.py            # writes validate_gpu_results.json
   ```

   All gates in the file header. Benign candidate-tie flips are reported
   separately from real winner disagreements (which fail).

3. Benchmark — staged, budget-guarded (median-of-reps, H2D/D2H included;
   NEVER quote min-of-3 — audit §1):

   ```bash
   python bench_gpu.py --tier quick                 # ~2 min sanity
   python bench_gpu.py --breakdown --nfreq 8000     # per-stage split +
                                                    # the 8k sweep
   python bench_gpu.py --nfreq 45000 --K 1 4        # PS1-scale grid
   python bench_gpu.py --nfreq 45000 --K 8 --budget 240   # only if time
   # optional throughput knob (validation numbers stay --fmad=false):
   python bench_gpu.py --tier quick --fmad
   ```

   Configs that don't fit memory are SKIPPED with a note; results append
   crash-safe to `bench_results.json` after every config.

4. Bring results home:

   ```bash
   scp -P $PORT root@$HOST:/workspace/{validate_gpu_results.json,bench_results.json} \
       experiments/paper2_gpu/trackA/pod/
   ```

5. **ALWAYS DELETE THE POD** (both success and failure paths), then
   VERIFY it is gone and record the spend:

   ```bash
   curl -s -X DELETE https://rest.runpod.io/v1/pods/$POD_ID \
     -H "Authorization: Bearer $RUNPOD_API_KEY"
   curl -s https://rest.runpod.io/v1/pods \
     -H "Authorization: Bearer $RUNPOD_API_KEY"   # must not list it
   ```

## 5. Reporting rules (audit §1/§4)

- Headline = median, transfers included.
- Speedups vs BOTH baselines: the same-code CPU floor AND the best
  single-thread CPU path (~78–81 µs/LC/freq @H8, one M5 P-core). Never
  the proto's inflated ~38× framing.
- Record: device, cupy version, driver, chunk sizes, reps completed,
  `frac_deferred` (dip/deferral rate on the synthetic batch), spend.

## 6. ESCALATE conditions

- Stage-2 spike FAIL (step 4.1) — stop, no bench.
- Any real winner disagreement in F1/F2/F3 (not benign tie flips).
- Compile failure on the pod that the local clang shim did not catch —
  fix locally, re-run validate_local, re-ship; do not hand-edit on pod.
- Projected spend > $25 for the session — stop and report.
