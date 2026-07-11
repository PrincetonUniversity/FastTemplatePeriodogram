#!/usr/bin/env python
"""RunPod CPU fleet for WP B8-closure -- the final-numbers batch (EXECUTION_PLAN.md).

Forked from fleet_b8.py (same proven webhook.site beacon plumbing; no SSH/IP/logs
on RunPod CPU pods) with the B8-closure deltas:

* own state at ``_fleet_b8c/state.json`` (the June ``_fleet_b8`` state and its
  stale token are never touched); ``mint`` creates a FRESH webhook token up
  front (B8 deferral precondition 4);
* two arms: ``e2-*`` = arm-(e) relaunch (joint/EM, N re-gridded to
  {4,5,6,8,12} per finding B8E-1, assignment_accuracy parallelized per B8E-2,
  per-phase timers in results.json), sharded one (scenario, seed) per pod;
  ``g-refine-*`` = grid-closure N-sweep with ``--refine`` peak refinement at
  the B8-flagged cells (SUMMARY_c-grid20k.md plan of record);
* results extract into the CANONICAL ``rerun_202606/raw/<tag>/`` -- extends the
  B8 artifact dir, never forks a sibling;
* **NO ``cleanup`` subcommand.** fleet.py-style cleanup terminates EVERY pod on
  the account and would kill the protected ``cuvarbase-dev`` pod
  (%(cuvarbase)s -- John's separate GPU workstream).  ``teardown`` deletes only
  pods recorded in THIS state file, then lists live pods via GET /pods and
  verifies the only survivor is cuvarbase-dev;
* watchdog v2: ``killstale <hours>`` terminates only OWN pods older than the
  wall limit (prevents the B8 ~$16 idle tail); a backgrounded PROG beacon posts
  a gzipped run.log tail every 15 min (max 4) so a wall-killed canary still
  yields the per-phase timer lines (fixes the B8 blind-canary failure mode);
* watchdog v3 (BOOT): the 2026-07-11 canary pod billed 50 min with
  ``uptimeInSeconds: 0`` -- the container NEVER started, and boot failures emit
  NO beacon, so ``killstale`` only catches them after ``<hours>``.  ``bootwatch``
  (run in a FOREGROUND poll loop, never a daemon) deletes any OWN pod with no
  START beacon after ``BOOT_LIMIT_MIN`` minutes and relaunches the job ONCE on
  a different CPU flavor (``MAX_BOOT_ATTEMPTS`` pod creates per job, then the
  job is marked boot-dead).  Pods that land on a known-bad machine
  (``BAD_MACHINES``) are deleted at create time and don't burn a boot attempt.
  POST-MORTEM 2026-07-11: all three never-booted pods (50 min + 2 x 7.4 h,
  ~$9.1 total) were ONE bug -- ``_prog_beacon`` ended in a bare ``&`` and the
  '' ; '' line join made the whole ``bash -c`` string a syntax error, so the
  container died at parse time on every host/flavor.  Fixed; ``plan`` and
  ``_create`` now ``bash -n`` every generated script before any pod exists;
* canary-first: ``canary`` launches ONLY e2-jitter-0 + g-refine-sesar-0
  (~$2-3); ``report`` compares measured core-hr against the estimates; the
  FULL batch (cap $25) is released only after a human-approved estimate.
* SHARDED arm (b) (2026-07-11, post-canary): the g-refine jobs are split ONE
  POD PER N-VALUE (see the job-table comment); the N4 shard of each
  (universe, seed) computes the dense greedy order once, siblings inject it
  via ``--greedy-order`` (placeholder resolved at launch from the collected
  N4 results.json).  ``sync`` merges the shard jobs into an existing state
  and retires the superseded monolith entries; ``retoken`` rotates to a
  fresh webhook token (request-count budget); ``deadline`` kills own pods
  past their per-job ``deadline_h`` (one fresh-pod retry, then FAILED);
  ``step`` = collect + deadline + bootwatch + dependency-aware slot fill
  (max MAX_LIVE concurrent pods), the one-call foreground supervision pass.
  Launch with ``B8C_PIN_VCPU=32``: hold-and-retry beats accepting 16 vCPU.

    .venv/bin/python fleet_b8_closure.py mint       # fresh token + pending state (no pods)
    .venv/bin/python fleet_b8_closure.py plan       # dry-run: job table + drivers, no network
    .venv/bin/python fleet_b8_closure.py sync       # merge shard jobs into existing state
    .venv/bin/python fleet_b8_closure.py retoken    # rotate to a fresh webhook token
    .venv/bin/python fleet_b8_closure.py step       # one supervision pass (poll foreground)
    .venv/bin/python fleet_b8_closure.py report     # timing vs estimate + $ projection
    .venv/bin/python fleet_b8_closure.py collect    # fetch results, extract, kill done pods
    .venv/bin/python fleet_b8_closure.py prog       # latest PROG log tail per running job
    .venv/bin/python fleet_b8_closure.py deadline   # one deadline-watchdog pass
    .venv/bin/python fleet_b8_closure.py bootwatch  # one boot-watchdog pass (poll from foreground)
    .venv/bin/python fleet_b8_closure.py killstale 12   # kill OWN pods past 12 h wall
    .venv/bin/python fleet_b8_closure.py teardown   # delete OWN pods; verify zero b8c pods live
"""
import base64
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".env"))
FLEET = os.path.join(HERE, "_fleet_b8c")
STATE = os.path.join(FLEET, "state.json")
RAW = os.path.join(HERE, "rerun_202606", "raw")     # canonical B8 dir -- EXTEND
REST = "https://rest.runpod.io/v1"
TGZ = ("https://github.com/PrincetonUniversity/FastTemplatePeriodogram/"
       "archive/refs/heads/dev.tar.gz")
IMAGE = "python:3.11"
RATE = 0.041                     # $/core-hr, measured on the 2026-06-02 fleet
# cpu3c deprioritized to the tail: the June fleet measured it slow.
FALLBACKS = [("cpu5c", 32, "SECURE"), ("cpu5c", 16, "SECURE"), ("cpu5g", 16, "SECURE"),
             ("cpu3g", 16, "SECURE"), ("cpu5c", 32, "COMMUNITY"), ("cpu5c", 8, "SECURE"),
             ("cpu3c", 32, "SECURE"), ("cpu3c", 16, "SECURE")]

# Boot watchdog (v3): boot failures bill but emit NO beacon (the 2026-07-10/11
# pods: 50 min - 7.4 h at uptime 0 s, ~$9, zero science).  No START beacon
# within BOOT_LIMIT_MIN of pod create => delete + relaunch on a different
# flavor; at most MAX_BOOT_ATTEMPTS pod creates per job, then boot-dead.
# ROOT CAUSE of those three failures (found 2026-07-11, machine-independent):
# _prog_beacon ended in a bare '&', and the ' ; ' line join produced '& ;' --
# a bash syntax error, so `bash -c` aborted the whole script before the first
# statement.  Fixed in _prog_beacon; `plan` now runs `bash -n` as a guard.
BOOT_LIMIT_MIN = 20
MAX_BOOT_ATTEMPTS = 2
BAD_MACHINES = set()  # 2026-07-11: cleared -- the never-booted pods were the
                      # '& ;' script bug on 3 different hosts, not bad machines

# The ONE pre-existing pod on the account (John's separate cuvarbase GPU
# workstream).  NEVER stopped, deleted, or modified by this launcher.
# Its POD ID CHANGES when John recreates it (2026-07-11: 6nzaafwb96j38r ->
# bp2pmad0mz6gjl), so every guard also matches the NAME below.
CUVARBASE_POD = "bp2pmad0mz6gjl"
CUVARBASE_NAME = "cuvarbase-dev"


def _is_protected(pid, name=None):
    """True for John's cuvarbase-dev pod, by id OR name (ids rotate)."""
    if pid == CUVARBASE_POD:
        return True
    if name is None and pid:
        m = re.search(r'"name"\s*:\s*"([^"]*)"',
                      _runpod("GET", "/pods/%s" % pid) or "")
        name = m.group(1) if m else ""
    return (name or "").strip().lower() == CUVARBASE_NAME

# ----------------------------------------------------------------------
# Job table.
# est = core-hr GUESSES pending the canary (the whole point of canary-first):
#   e2: B8 e-canary measured 134.6 core-hr for ONE (jitter, seed 0) run at the
#       old grid with the serial assignment_accuracy (~87%% of the bill, B8E-2);
#       parallelizing it + the {4,5,6,8,12} grid (n_master 40->12) + the C4
#       scan default project ~10 core-hr/run.  Gate: within ~2x or stop.
#   g-refine: a-sesar-0 measured 100.8 core-hr for k_sparse+n_sweep at 10k;
#       N-sweep share ~44%% => ~45 core-hr, x~1.4 refinement overhead, minus
#       the C4 FTP speedup (MHLS/CE dominate and are NOT accelerated) => ~45;
#       BV x1.5.
# ----------------------------------------------------------------------
PROD = ("python run_production_matrix.py --universe %(u)s --seeds %(s)d --n-jobs -1 "
        "--nharmonics 8 --n-sources 256 --baseline-days 1095.75 --obs-bands g,r "
        "--dense-master-epochs 60 --sparse-master-epochs 6 --k-values 1,2,4,8 "
        "--n-epochs-values %(nvals)s --mhls-h 8 --mbls-h 1 "
        "--greedy-select-sources 64 --greedy-select-nfreq 2000 --bv-bands g,r "
        "--with-ce --no-figures --outdir /workspace/out")
# grid-closure: same 10k grid + peak refinement, N-sweep only (the flagged
# cells all live in the N-sweep; --fixed-k 2 matches arm a for McNemar pairing)
ARM_REFINE = " --n-freq 10000 --stages n_sweep --fixed-k 2 --refine"

# ----------------------------------------------------------------------
# Arm-(b) SHARDING (2026-07-11, post-canary).  The monolithic g-refine jobs
# measured >=91.2 (sesar) / ~138 (bv, x1.51) core-hr -- deadline-infeasible on
# one pod.  Split each job into ONE POD PER N-VALUE of its own
# --n-epochs-values grid (job table: 4,8,12,16,24,40).  downsample(N,
# random_state=seed) draws ONE frozen permutation per (source, band) and keeps
# the first N, so a single-N shard reproduces the monolithic cell EXACTLY and
# p_true pairing vs arm (a) is untouched.
#
# The greedy-order precompute is template-count linear (measured: sesar
# 98 templates = 7.4 core-hr; bv 560 => ~42, which is also exactly the June
# a-bv-vs-a-sesar wall gap) and per-N sharding would replay it 6x per job, so
# only the N4 shard computes it (cheapest cell) and records it in
# results.json; sibling shards inject it via --greedy-order and skip the
# precompute.  Sibling drivers carry a placeholder resolved at launch time
# from the collected N4 result.
# ----------------------------------------------------------------------
N_SHARDS = (4, 8, 12, 16, 24, 40)
GREEDY_PLACEHOLDER = "{GREEDY_ORDER}"
MAX_LIVE = 20        # one pod per shard authorized 2026-07-11 (wall-clock min)
DEADLINE_DEFAULT_H = 1.5


def _shard_est(u, n, with_greedy):
    """Per-shard core-hr estimate from MEASURED anchors (2026-07-11 canary +
    June a-jobs): unrefined N-sweep ~44 core-hr across sum(N)=104 (~prop. N),
    refine adds ~8 core-hr/cell flat (>=1.9x overall floor), greedy dense
    7.4 core-hr (sesar) / ~42 (bv, 560/98 templates)."""
    cell = 44.0 * n / 104.0 + 8.0 + (1.0 if u == "bv" else 0.0)
    greedy = (7.4 if u == "sesar" else 42.0) if with_greedy else 0.0
    return int(round(cell + greedy))


def _shard_driver(u, s, n, inject):
    d = PROD % {"u": u, "s": s, "nvals": str(n)} + ARM_REFINE
    if inject:
        d += " --greedy-order " + GREEDY_PLACEHOLDER
    return d


_G_JOBS = []
for _u, _seeds in (("sesar", (0, 1, 2)), ("bv", (0,))):
    for _s in _seeds:
        for _n in N_SHARDS:
            _inject = (_n != N_SHARDS[0])
            _G_JOBS.append({
                "tag": "g-refine-%s-%d-N%d" % (_u, _s, _n),
                "est": _shard_est(_u, _n, not _inject),
                "driver": _shard_driver(_u, _s, _n, _inject),
                "avoid_flavors": ["cpu3c"],       # June-measured slow silicon
                # bv-N4 carries the one-time 560-template greedy (~42 core-hr
                # = ~1.6 h on 32 vCPU): the only shard allowed past 1.5 h
                "deadline_h": (2.5 if (_u == "bv" and not _inject)
                               else DEADLINE_DEFAULT_H),
                # N4 shards get 2 PROG posts (greedy timing salvage), siblings
                # 1 -- keeps the batch under the ~100-request token cap
                "prog_posts": (2 if not _inject else 1),
            })

# arm-(e) relaunch: ONE (scenario, seed) per pod (B8 lesson: shard; one job
# must not set the tail).  --n-epochs-values 4 5 6 8 12 re-grids N so the
# 4->8 transition where the MRA-gap hypothesis is falsifiable is populated
# (B8E-1: the old {4,8,12,20,40} saturated 4/5 canary cells).
JOINT = ("python run_joint_vs_pipeline.py --universe sesar --k 2 --n-sources 256 "
         "--n-harmonics 8 --f-min 1.4 --f-max 3.6 --n-freq 2000 --baseline-days 60 "
         "--mean-mag 19 --max-iter 8 --n-jobs -1 --n-epochs-values 4 5 6 8 12 "
         "--seed %(seed)d --no-figure%(extra)s --out /workspace/out/%(scen)s_seed%(seed)d")


def _joint(scen, seed, extra):
    return JOINT % {"scen": scen, "seed": seed,
                    "extra": (" " + extra.strip()) if extra else ""}


_E2_SCENARIOS = [("base", ""),
                 ("holdout", "--library-holdout-frac 0.5"),
                 ("jitter", "--intrinsic-jitter 0.15"),
                 ("lowsnr", "--amplitude 0.1")]

JOBS = ([{"tag": "e2-%s-%d" % (scen, seed), "est": 10,
          "driver": _joint(scen, seed, extra)}
         for scen, extra in _E2_SCENARIOS for seed in (0, 1, 2)]
        + _G_JOBS)

# canary phase CLOSED 2026-07-11 (measured; both gates failed at monolithic
# scope -- see B8_COST_ESTIMATE.md).  Tag kept only for plan's sanity assert.
CANARY_TAGS = ("e2-jitter-0", "g-refine-sesar-0-N4")


# ----------------------------------------------------------------------
# RunPod / webhook plumbing (proven in fleet.py / fleet_b8.py)
# ----------------------------------------------------------------------
def _key():
    return [l.split("=", 1)[1].strip() for l in open(ENV)
            if l.startswith("RUNPOD_API_KEY=")][0]


def _runpod(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(REST + path, data=data, method=method,
                               headers={"Authorization": "Bearer " + _key(),
                                        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=45) as f:
            return f.read().decode()
    except urllib.error.HTTPError as e:
        return e.read().decode()


def _delete_pod(pid):
    """DELETE one pod -- with the cuvarbase-dev guard on EVERY delete path."""
    if not pid:
        return
    if _is_protected(pid):
        print("REFUSING to delete protected pod %s (cuvarbase-dev)" % pid)
        return
    _runpod("DELETE", "/pods/%s" % pid)


def _live_pods():
    """[(id, name), ...] of every live pod on the account (GET /pods)."""
    text = _runpod("GET", "/pods")
    try:
        data = json.loads(text)
        pods = data.get("pods", data) if isinstance(data, dict) else data
        out = [(p.get("id"), p.get("name", "?")) for p in pods
               if isinstance(p, dict) and p.get("id")]
        if out or pods == []:
            return out
    except Exception:
        pass
    return [(pid, "?") for pid in re.findall(r'"id":"([a-z0-9]{10,})"', text)]


def _mint_token():
    r = urllib.request.Request("https://webhook.site/token", data=b"{}",
                               method="POST",
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read().decode())["uuid"]


def _post(token, job, suffix):
    # payload comes from a FILE: a tar.gz beacon base64s to ~0.5 MB, far past
    # Linux's 128 KiB per-argument cap, so "$PAYLOAD" in argv would fail
    return ("for i in $(seq 1 20); do curl -sf --max-time 90 -X POST "
            "https://webhook.site/%s -H 'X-Job: %s%s' -H \"X-Status: $ST\" "
            "--data-binary @/workspace/payload.b64 && break; sleep 20; done"
            % (token, job, suffix))


def _prog_beacon(token, tag, n_posts=4):
    """Backgrounded 15-min progress beacon (``n_posts`` max -- webhook.site
    free tokens store ~100 requests; the 24-shard batch runs 1-2 posts/shard
    so START+RESULT+PROG stays under the cap with retry margin).  Uses its
    own payload file so it never clobbers /workspace/payload.b64.

    MUST NOT end in a bare ``&``: _bootstrap joins lines with '' ; '' and
    ``cmd & ; next`` is a bash SYNTAX ERROR that aborts the ENTIRE -c string
    before anything runs -- the root cause of the three 2026-07-10/11 pods
    that billed at uptime 0 with zero beacons.  The trailing ``true`` makes
    the join valid (``cmd & true ; next``)."""
    return ("( for i in $(seq 1 %d); do sleep 900; "
            "tail -40 /workspace/run.log 2>/dev/null | gzip -c | base64 -w0 "
            ">/workspace/prog.b64; "
            "curl -sf --max-time 60 -X POST https://webhook.site/%s "
            "-H 'X-Job: %s-PROG' --data-binary @/workspace/prog.b64; "
            "done ) >/dev/null 2>&1 & true" % (int(n_posts), token, tag))


def _bootstrap(tag, driver, token, n_prog=4):
    diag = ("( echo \"REPO=[$REPO]\"; echo --ls--; ls /workspace; echo --dl--; "
            "cat /workspace/dl.log; echo --tar--; cat /workspace/tar.log; echo --pip--; "
            "tail -6 /workspace/pip.log; echo --numpy--; "
            "python -c 'import numpy;print(numpy.__version__)' 2>&1; echo --ftp--; "
            "python -c 'import ftperiodogram;print(\"ftp_ok\")' 2>&1 ) "
            ">/workspace/diag.txt 2>&1")
    # success -> tar.gz of the whole out dir (results json + per_source npz);
    # failure -> run.log tail.  [ -n "$(ls -A ...)" ] guards tar-of-empty.
    result = ('if [ "$ST" = "0" ] && [ -n "$(ls -A /workspace/out 2>/dev/null)" ]; '
              "then tar czf - -C /workspace out | base64 -w0 >/workspace/payload.b64; "
              "else tail -80 /workspace/run.log 2>/dev/null | gzip -c | base64 -w0 "
              ">/workspace/payload.b64; fi")
    lines = [
        "set +e",
        "mkdir -p /workspace /workspace/out",            # volume mount point may not exist
        "cd /workspace",
        "curl -fsSL -o repo.tgz %s >dl.log 2>&1" % TGZ,
        "tar xzf repo.tgz 2>tar.log",
        "REPO=$(ls -d /workspace/FastTemplatePeriodogram-*/ 2>/dev/null | head -1)",
        "pip install --break-system-packages -q numpy scipy nfft >/workspace/pip.log 2>&1",
        'export PYTHONPATH="$REPO"',
        diag,
        'ST=diag; gzip -c /workspace/diag.txt | base64 -w0 >/workspace/payload.b64',
        _post(token, tag, "-START"),
        _prog_beacon(token, tag, n_prog),
        'cd "$REPO/experiments/phase3_recovery"',
        "ST=run; { %s ; } >/workspace/run.log 2>&1; ST=$?" % driver,
        result,
        _post(token, tag, ""),
        "sleep 9000",
    ]
    return " ; ".join(lines)


def _machine_id(pid):
    """machineId of a live pod (GET /pods/<id>); '' if unparseable."""
    m = re.search(r'"machineId"\s*:\s*"([^"]+)"', _runpod("GET", "/pods/%s" % pid) or "")
    return m.group(1) if m else ""


def _create(tag, driver, token, avoid=(), n_prog=4):
    """Create a pod, skipping flavors in ``avoid`` (boot-watchdog retries) and
    deleting-and-continuing if the pod lands on a BAD_MACHINES host (a bad
    host is RunPod's placement, not a boot attempt of ours)."""
    script = _bootstrap(tag, driver, token, n_prog)
    _check_script_syntax(script, tag)     # never POST a script bash can't parse
    # B8C_PIN_VCPU=32: canary lesson 2026-07-11 -- g-refine-sesar-0 landed on
    # 16 vCPU (32-pool exhausted) and was deadline-infeasible from the start.
    # With the pin set, ONLY flavors at exactly that vcpu count are tried; the
    # caller wait-retries the create when the pinned pool is empty.
    pin = int(os.environ.get("B8C_PIN_VCPU") or 0)
    last = ""
    for flav, vcpu, cloud in FALLBACKS:
        if flav in avoid:
            continue
        if pin and vcpu != pin:
            continue
        body = {"name": "b8c-%s" % tag, "computeType": "CPU",
                "cpuFlavorIds": [flav], "vcpuCount": vcpu, "imageName": IMAGE,
                "containerDiskInGb": 20, "ports": ["22/tcp"], "cloudType": cloud,
                "dockerStartCmd": ["bash", "-c", script]}
        t = _runpod("POST", "/pods", body)
        m = re.search(r'"id":"([a-z0-9]+)"', t)
        if not m:
            last = t
            continue
        pid = m.group(1)
        mach = _machine_id(pid) or (re.search(
            r'"machineId"\s*:\s*"([^"]+)"', t).group(1)
            if re.search(r'"machineId"\s*:\s*"([^"]+)"', t) else "")
        if mach in BAD_MACHINES:
            print("  %s: pod %s landed on bad machine %s -- deleted, next flavor"
                  % (tag, pid, mach))
            _delete_pod(pid)
            last = "bad machine %s" % mach
            continue
        mc = re.search(r'"costPerHr"\s*:\s*([0-9.]+)', t)
        cost = float(mc.group(1)) if mc else None
        return pid, "%s/%d/%s@%s" % (flav, vcpu, cloud, mach or "?"), cost
    return None, (last[:140] or "no fallback flavor available"), None


def _resolve_driver(j):
    """Substitute the greedy-order placeholder from the collected sibling N4
    shard's results.json (SystemExit if it has not landed yet)."""
    d = j["driver"]
    if GREEDY_PLACEHOLDER not in d:
        return d
    prefix = j["tag"].rsplit("-N", 1)[0]
    src = os.path.join(RAW, "%s-N%d" % (prefix, N_SHARDS[0]),
                       "out", "results.json")
    if not os.path.exists(src):
        raise SystemExit("%s: sibling N%d shard result not collected yet (%s)"
                         % (j["tag"], N_SHARDS[0], src))
    order = json.load(open(src))["per_seed"][0]["n_epochs_sweep"][
        "greedy_order_dense"]
    if not order:
        raise SystemExit("%s: empty greedy_order_dense in %s" % (j["tag"], src))
    return d.replace(GREEDY_PLACEHOLDER,
                     ",".join(str(int(i)) for i in order))


def _record_spend(st, j, note):
    """Accumulate billed wall for job ``j``'s live pod into st['spend_usd']
    (called at EVERY pod-delete site so the ledger never loses a billed pod)."""
    if not j.get("pod") or not j.get("created"):
        return
    wall_h = (time.time() - j["created"]) / 3600.0
    cost = wall_h * float(j.get("cost_hr") or 1.12)
    st["spend_usd"] = round(st.get("spend_usd", 0.0) + cost, 4)
    print("  spend +$%.2f (%s %s, %.2f h) -> total $%.2f"
          % (cost, j["tag"], note, wall_h, st["spend_usd"]))


def _launch(j, token):
    """Create a pod for job ``j`` honoring the boot-attempt cap and the job's
    avoid-flavor list; updates the job record in place.  Failed CREATEs (no
    pod id returned -- nothing billed) do not consume a boot attempt."""
    if j.get("boot_dead") or j.get("boot_attempts", 0) >= MAX_BOOT_ATTEMPTS:
        j["boot_dead"] = True
        print("launch %s: SKIP -- boot-attempt cap (%d) reached"
              % (j["tag"], MAX_BOOT_ATTEMPTS))
        return None
    driver = _resolve_driver(j)
    pid, info, cost = _create(j["tag"], driver, token,
                              avoid=tuple(j.get("avoid_flavors") or ()),
                              n_prog=int(j.get("prog_posts") or 4))
    j["driver"] = driver               # record what actually runs (resolved)
    j["pod"], j["info"] = pid, info
    j["created"] = time.time() if pid else None
    if cost is not None:
        j["cost_hr"] = cost
    if pid:
        j["boot_attempts"] = j.get("boot_attempts", 0) + 1
        print("launch %s -> %s (%s, $%s/h) [boot attempt %d/%d]"
              % (j["tag"], pid, info, cost, j["boot_attempts"],
                 MAX_BOOT_ATTEMPTS))
    else:
        print("launch %s -> FAIL (%s)" % (j["tag"], info))
    return pid


def _vcpu(info):
    m = re.match(r"[a-z0-9]+/(\d+)/", info or "")
    return int(m.group(1)) if m else 32


def _requests(token):
    """All beacon requests, paged (this batch posts <100 beacons by design)."""
    out, page = [], 1
    while page <= 5:
        url = ("https://webhook.site/token/%s/requests?sorting=oldest"
               "&per_page=100&page=%d" % (token, page))
        d = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())["data"]
        out.extend(d)
        if len(d) < 100:
            break
        page += 1
    return out


def _headers(r):
    return {k.lower(): (v[0] if isinstance(v, list) else v)
            for k, v in (r.get("headers") or {}).items()}


def _start_times(token):
    """{tag: latest START-beacon epoch} -- timestamped so a deadline-retry pod
    is never credited with its killed predecessor's START beacon."""
    from datetime import datetime, timezone
    out = {}
    for r in _requests(token):
        job = _headers(r).get("x-job", "")
        if not job.endswith("-START"):
            continue
        try:
            ts = datetime.strptime(
                r["created_at"][:19], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc).timestamp()
        except Exception:
            continue
        tag = job[:-6]
        out[tag] = max(out.get(tag, 0.0), ts)
    return out


def _started(j, starts):
    """True if the job's CURRENT pod (created at j['created']) has beaconed."""
    ts = starts.get(j["tag"])
    return ts is not None and ts >= (j.get("created") or 0) - 180.0


def _load_state():
    return json.load(open(STATE))


def _save_state(st):
    json.dump(st, open(STATE, "w"), indent=2)


def _new_job(j, pod=None, info="pending"):
    nj = {"tag": j["tag"], "est": j["est"], "driver": j["driver"], "pod": pod,
          "info": info, "received": False, "status": None, "started": False,
          "created": None}
    for k in ("avoid_flavors", "deadline_h", "prog_posts"):
        if j.get(k) is not None:
            nj[k] = j[k]
    return nj


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------
def _check_script_syntax(script, tag):
    """`bash -n` the generated bootstrap script.  Guard added 2026-07-11: a
    '& ;' join bug made every generated script a bash syntax error, so three
    pods billed at uptime 0 with zero beacons.  Cheap, local, no network."""
    import subprocess
    p = subprocess.run(["bash", "-n", "-c", script],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("BOOTSTRAP SYNTAX ERROR for %s -- refusing to plan/"
                         "launch:\n%s" % (tag, p.stderr.strip()[:500]))


def plan():
    """Dry-run: the job table + full driver strings + a `bash -n` syntax check
    of every generated bootstrap script.  Zero network, zero paid."""
    tags = [j["tag"] for j in JOBS]
    assert len(tags) == len(set(tags)), "duplicate job tags: %s" % tags
    assert all(t in tags for t in CANARY_TAGS), "canary tag not in JOBS"
    for j in JOBS:
        # syntax-check what launch will actually post: placeholder resolved
        # (dummy order here; the real one comes from the sibling N4 result)
        drv = j["driver"].replace(GREEDY_PLACEHOLDER, "0,1")
        _check_script_syntax(
            _bootstrap(j["tag"], drv, "TOKEN", int(j.get("prog_posts") or 4)),
            j["tag"])
    print("bash -n: all %d bootstrap scripts parse clean\n" % len(JOBS))
    total = canary_total = 0
    for j in JOBS:
        mark = "  <-- CANARY" if j["tag"] in CANARY_TAGS else ""
        print("%-20s est %4d core-hr (~$%4.1f)%s"
              % (j["tag"], j["est"], j["est"] * RATE, mark))
        print("    %s" % j["driver"])
        total += j["est"]
        if j["tag"] in CANARY_TAGS:
            canary_total += j["est"]
    print("\n%d jobs | canary %d core-hr (~$%.1f) | full batch est %d core-hr "
          "(~$%.1f) at $%.3f/core-hr -- cap $25, canary gate first"
          % (len(JOBS), canary_total, canary_total * RATE, total,
             total * RATE, RATE))


def mint():
    """Mint a FRESH webhook token and write the pending state (no pods)."""
    if os.path.exists(STATE):
        st = _load_state()
        raise SystemExit("state already exists with token %s -- refusing to "
                         "re-mint (delete %s deliberately to start over)"
                         % (st.get("token"), STATE))
    os.makedirs(FLEET, exist_ok=True)
    token = _mint_token()
    _save_state({"token": token, "jobs": [_new_job(j) for j in JOBS]})
    print("minted token %s | webhook https://webhook.site/#!/%s" % (token, token))
    print("state -> %s (%d jobs pending; next: canary)" % (STATE, len(JOBS)))


def canary():
    """Launch ONLY the canary pair (uses the minted state, or mints one)."""
    if os.path.exists(STATE):
        st = _load_state()
    else:
        os.makedirs(FLEET, exist_ok=True)
        st = {"token": _mint_token(), "jobs": [_new_job(j) for j in JOBS]}
    for j in st["jobs"]:
        if j["tag"] not in CANARY_TAGS or j.get("pod") or j["received"]:
            continue
        _launch(j, st["token"])
        time.sleep(2)
    _save_state(st)
    print("token %s | webhook https://webhook.site/#!/%s"
          % (st["token"], st["token"]))


def launch_one():
    """Launch exactly ONE job by tag (``launch <tag>``) -- the arm-(b)-only
    path: ``canary``/``release`` would also (re)launch e2-* jobs, which the
    2026-07-11 canary showed fail their estimate gate by >=7.3x and are
    escalated separately.  Never touches any other job record."""
    tag = sys.argv[2]
    st = _load_state()
    jobs = [j for j in st["jobs"] if j["tag"] == tag]
    if len(jobs) != 1:
        raise SystemExit("no unique job with tag %r" % tag)
    j = jobs[0]
    if j.get("pod") or j["received"]:
        raise SystemExit("%s already has pod=%s received=%s -- refusing"
                         % (tag, j.get("pod"), j["received"]))
    _launch(j, st["token"])
    _save_state(st)


def release():
    """Launch every job that has no pod yet (post-canary, or capacity retry).
    NOT used for the sharded arm-(b) batch -- ``step`` is (dependency-aware,
    concurrency-capped); release would also launch the escalated e2-* arm."""
    st = _load_state()
    for j in st["jobs"]:
        if (j.get("pod") or j["received"] or j.get("boot_dead")
                or j.get("retired") or j.get("failed")):
            continue
        _launch(j, st["token"])
        time.sleep(2)
    _save_state(st)
    pending = sum(1 for j in st["jobs"] if not j.get("pod") and not j["received"])
    print("pending (no pod): %d/%d" % (pending, len(st["jobs"])))


def start_beacon():
    st = _load_state()
    for r in _requests(st["token"]):
        job = _headers(r).get("x-job", "")
        if job.endswith("-START"):
            try:
                txt = gzip.decompress(base64.b64decode(r["content"])).decode(
                    "utf-8", "replace")
            except Exception as e:
                txt = "<decode err %s>" % e
            print("%s: %s" % (job, txt.strip()[:200]))


def prog():
    """Latest PROG run.log tail per job (the per-phase timer lines land here)."""
    st = _load_state()
    latest = {}
    for r in _requests(st["token"]):
        job = _headers(r).get("x-job", "")
        if job.endswith("-PROG"):
            latest[job[:-5]] = r
    for tag, r in sorted(latest.items()):
        try:
            txt = gzip.decompress(base64.b64decode(r["content"])).decode(
                "utf-8", "replace")
        except Exception as e:
            txt = "<decode err %s>" % e
        print("=== %s (posted %s) ===\n%s" % (tag, r.get("created_at"),
                                              txt.strip()[-1500:]))
    if not latest:
        print("no PROG beacons yet")


def _extract(tag, raw):
    """Extract a result tar.gz under RAW/<tag>/ (refuses path escapes)."""
    dest = os.path.join(RAW, tag)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not (m.isfile() or m.isdir()) or m.name.startswith(("/", "..")):
                continue
            tf.extract(m, dest)
    return dest


def collect():
    st = _load_state()
    posts = {}
    # first SUCCESS wins: a crashed pod restart-loops the driver, so a later
    # beacon can be a re-run's failure -- never let it shadow a good result
    for r in _requests(st["token"]):
        job = _headers(r).get("x-job", "")
        if not job or job.endswith("-PROG"):
            continue
        prev = posts.get(job)
        if prev is None or (_headers(prev).get("x-status") != "0"
                            and _headers(r).get("x-status") == "0"):
            posts[job] = r
    os.makedirs(RAW, exist_ok=True)
    starts = _start_times(st["token"])
    for j in st["jobs"]:
        if not j["started"] and _started(j, starts):
            j["started"] = True
        if j["received"]:
            continue
        r = posts.get(j["tag"])
        if not r:
            continue
        status = _headers(r).get("x-status", "?")
        try:
            raw = base64.b64decode(r["content"])
        except Exception as e:
            print("%s: decode fail %s" % (j["tag"], e))
            continue
        if status == "0":
            try:
                dest = _extract(j["tag"], raw)
                print("%s: RESULT extracted -> %s" % (j["tag"], dest))
            except Exception as e:
                open(os.path.join(RAW, "%s.payload.bin" % j["tag"]), "wb").write(raw)
                print("%s: extract fail (%s); payload saved" % (j["tag"], e))
        else:
            log = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
            open(os.path.join(RAW, "%s.FAILED.log" % j["tag"]), "wb").write(log)
            print("%s: FAILED status=%s (log saved)" % (j["tag"], status))
        j["received"], j["status"] = True, status
        if j["pod"]:
            j["done_wall_h"] = round(
                (time.time() - (j.get("created") or time.time())) / 3600.0, 3)
            _record_spend(st, j, "RESULT")
            _delete_pod(j["pod"])
            print("%s: pod %s terminated" % (j["tag"], j["pod"]))
    _save_state(st)
    n = len(st["jobs"])
    print("started %d/%d | received %d/%d | ok %d/%d"
          % (sum(1 for j in st["jobs"] if j["started"]), n,
             sum(1 for j in st["jobs"] if j["received"]), n,
             sum(1 for j in st["jobs"] if j["status"] == "0"), n))
    return all(j["received"] for j in st["jobs"])


def report():
    """Measured core-hr per finished job vs estimate + projected total spend."""
    st = _load_state()
    seen = {}
    for r in _requests(st["token"]):
        job = _headers(r).get("x-job", "")
        if job and not job.endswith("-PROG"):
            seen[job] = r.get("created_at")
    total_meas = total_proj = 0.0
    print("%-20s %5s %9s %9s %7s" % ("job", "vcpu", "est c-hr", "meas c-hr", "ratio"))
    for j in st["jobs"]:
        t0, t1 = seen.get(j["tag"] + "-START"), seen.get(j["tag"])
        vcpu = _vcpu(j.get("info"))
        meas = None
        if t0 and t1:
            try:
                from datetime import datetime
                f = "%Y-%m-%d %H:%M:%S"
                hrs = (datetime.strptime(t1[:19], f)
                       - datetime.strptime(t0[:19], f)).total_seconds() / 3600.0
                meas = hrs * vcpu
            except Exception:
                pass
        proj = meas if meas is not None else j["est"]
        total_proj += proj
        if meas is not None:
            total_meas += meas
            print("%-20s %5d %9.0f %9.1f %6.1fx"
                  % (j["tag"], vcpu, j["est"], meas, meas / j["est"]))
        else:
            print("%-20s %5d %9.0f %9s %7s"
                  % (j["tag"], vcpu, j["est"], "-", "-"))
    print("\nmeasured so far: %.0f core-hr (~$%.1f)" % (total_meas, total_meas * RATE))
    print("projected total: %.0f core-hr (~$%.1f) at $%.3f/core-hr [cap $25]"
          % (total_proj, total_proj * RATE, RATE))


def count():
    """RESULT beacons received on THIS batch's token (NOT a live-pod check --
    use ``teardown``/GET /pods for that; fleet.py count reads the stale June
    token and must not be used for B8-closure)."""
    st = _load_state()
    try:
        res = {_headers(r).get("x-job", "") for r in _requests(st["token"])}
    except Exception:
        print(0)
        return
    print(len({j["tag"] for j in st["jobs"]} & res))


def bootwatch():
    """ONE boot-watchdog pass (v3): any OWN pod with no START beacon after
    BOOT_LIMIT_MIN minutes is DELETEd (boot failures bill but never beacon --
    the 2026-07-11 lesson) and the job relaunched once on a different flavor,
    up to MAX_BOOT_ATTEMPTS pod creates per job.  Call this repeatedly from a
    FOREGROUND poll loop; it is deliberately not a daemon."""
    st = _load_state()
    try:
        starts = _start_times(st["token"])
    except Exception as e:
        print("bootwatch: beacon fetch failed (%s) -- no action taken" % e)
        return
    now = time.time()
    for j in st["jobs"]:
        if not j["started"] and _started(j, starts):
            j["started"] = True
            print("bootwatch %s: START beacon seen -- boot OK" % j["tag"])
        if (not j.get("pod") or j["received"] or j["started"]
                or j.get("boot_dead") or j.get("killed")):
            continue
        age_min = (now - (j.get("created") or now)) / 60.0
        if age_min <= BOOT_LIMIT_MIN:
            print("bootwatch %s: pod %s booting, %.1f/%d min"
                  % (j["tag"], j["pod"], age_min, BOOT_LIMIT_MIN))
            continue
        print("bootwatch %s: NO START after %.1f min -- deleting pod %s (%s)"
              % (j["tag"], age_min, j["pod"], j.get("info")))
        _record_spend(st, j, "BOOT-TIMEOUT")
        _delete_pod(j["pod"])
        flav = (j.get("info") or "").split("/", 1)[0]
        j.setdefault("history", []).append(
            {"pod": j["pod"], "created": j.get("created"),
             "info": (j.get("info") or "") + " BOOT-TIMEOUT@%.0fmin" % age_min})
        if flav and flav not in j.setdefault("avoid_flavors", []):
            j["avoid_flavors"].append(flav)
        j["pod"], j["info"], j["created"] = None, "boot-timeout", None
        if j.get("boot_attempts", 0) >= MAX_BOOT_ATTEMPTS:
            j["boot_dead"] = True
            print("bootwatch %s: %d boot attempts exhausted -- BOOT-DEAD, "
                  "no more pods for this job" % (j["tag"], MAX_BOOT_ATTEMPTS))
        else:
            _launch(j, st["token"])
    _save_state(st)


def killstale():
    """Watchdog v2: terminate OWN pods past ``sys.argv[2]`` hours wall."""
    hours = float(sys.argv[2])
    st = _load_state()
    now = time.time()
    n = 0
    for j in st["jobs"]:
        if not j.get("pod") or j["received"] or j.get("killed"):
            continue
        age = (now - (j.get("created") or now)) / 3600.0
        if age > hours:
            _record_spend(st, j, "KILLED-STALE")
            _delete_pod(j["pod"])
            j["killed"] = True
            j["info"] = (j.get("info") or "") + " KILLED-STALE@%.1fh" % age
            print("killstale %s: pod %s (%.1f h) terminated" %
                  (j["tag"], j["pod"], age))
            n += 1
    _save_state(st)
    print("killed %d stale pods" % n)


def teardown():
    """Delete every pod THIS state launched, then verify via GET /pods that no
    b8c-* pod survives.  Pods that are neither b8c-* nor cuvarbase-dev (e.g.
    the concurrent E5-offload fleet's) are EXTERNAL workstreams: listed,
    never touched, and not a failure."""
    st = _load_state()
    for j in st["jobs"]:
        if j.get("pod") and not j["received"]:
            _record_spend(st, j, "TEARDOWN")
        if j.get("pod"):
            _delete_pod(j["pod"])
            print("teardown %s: DELETE pod %s" % (j["tag"], j["pod"]))
    _save_state(st)
    time.sleep(5)
    survivors = _live_pods()
    print("live pods after teardown:")
    for pid, name in survivors:
        note = ("  [protected cuvarbase-dev]" if _is_protected(pid, name)
                else ("  [OURS -- SHOULD BE DEAD]"
                      if (name or "").startswith("b8c-")
                      else "  [external workstream -- untouched]"))
        print("  %s  %s%s" % (pid, name, note))
    stray = [p for p in survivors if (p[1] or "").startswith("b8c-")]
    if stray:
        print("WARNING: %d b8c pod(s) still live: %s"
              % (len(stray), [p[0] for p in stray]))
        sys.exit(1)
    print("OK: zero b8c pods remain (cuvarbase-dev and any external-fleet "
          "pods are not ours to manage)")


def sync():
    """Merge the current JOBS table into an existing state: append job entries
    that state does not know yet (the per-N shards) and RETIRE monolithic
    g-refine entries a shard set supersedes (never touching pods or history).
    Idempotent."""
    st = _load_state()
    have = {j["tag"] for j in st["jobs"]}
    added = 0
    for j in JOBS:
        if j["tag"] in have:
            continue
        st["jobs"].append(_new_job(j))
        added += 1
    shard_prefixes = {j["tag"].rsplit("-N", 1)[0] for j in JOBS
                      if re.search(r"-N\d+$", j["tag"])}
    retired = 0
    for j in st["jobs"]:
        if j["tag"] not in shard_prefixes or j.get("retired"):
            continue
        if j.get("pod"):
            print("sync: NOT retiring %s -- it has live pod %s"
                  % (j["tag"], j["pod"]))
            continue
        j["retired"] = True
        j["info"] = ((j.get("info") or "")
                     + " | RETIRED: superseded by per-N shards 2026-07-11")
        retired += 1
    _save_state(st)
    print("sync: +%d shard jobs, %d monolith(s) retired, %d jobs total"
          % (added, retired, len(st["jobs"])))


def retoken():
    """Rotate to a FRESH webhook token (the canary token already carries dozens
    of requests; the 24-shard batch needs the full ~100-request budget).  Old
    token kept in state for provenance; all canary outcomes are already
    recorded in state history + B8_COST_ESTIMATE.md."""
    st = _load_state()
    old = st.get("token")
    st.setdefault("old_tokens", []).append(old)
    st["token"] = _mint_token()
    _save_state(st)
    print("token rotated %s -> %s | webhook https://webhook.site/#!/%s"
          % (old, st["token"], st["token"]))


def deadline():
    """ONE deadline-watchdog pass: kill any OWN pod past its job's
    ``deadline_h`` (default %.1f h) without a RESULT, then allow ONE retry on
    a fresh pod (a shard at 2x its expected wall is wrong, not slow -- the
    2026-07-11 lesson).  Second breach marks the job FAILED (the batch
    proceeds without it; the gap is reported)."""
    st = _load_state()
    now = time.time()
    for j in st["jobs"]:
        if not j.get("pod") or j["received"] or j.get("killed"):
            continue
        dl = float(j.get("deadline_h") or DEADLINE_DEFAULT_H)
        age_h = (now - (j.get("created") or now)) / 3600.0
        if age_h <= dl:
            continue
        print("deadline %s: %.2f h > %.2f h -- killing pod %s"
              % (j["tag"], age_h, dl, j["pod"]))
        _record_spend(st, j, "KILLED-DEADLINE")
        _delete_pod(j["pod"])
        j.setdefault("history", []).append(
            {"pod": j["pod"], "created": j.get("created"),
             "info": "%s KILLED-DEADLINE@%.2fh" % (j.get("info") or "", age_h)})
        j["pod"], j["created"], j["started"] = None, None, False
        att = j.get("deadline_attempts", 0) + 1
        j["deadline_attempts"] = att
        if att > 1:
            j["failed"] = True
            j["info"] = "FAILED: 2x deadline kills"
            print("deadline %s: second breach -- job FAILED, batch proceeds "
                  "without it" % j["tag"])
        else:
            j["boot_attempts"] = 0      # fresh cycle for the one retry
            j["info"] = "killed-deadline; retry pending"
            print("deadline %s: ONE retry on a fresh pod queued" % j["tag"])
    _save_state(st)


deadline.__doc__ = deadline.__doc__ % DEADLINE_DEFAULT_H


def _launchable(j):
    """Shard jobs ready for a pod: never e2-* (escalated separately), never
    retired/failed/boot-dead/done, and greedy-injected shards only after their
    sibling N4 result landed (the placeholder must be resolvable)."""
    if (j.get("pod") or j["received"] or j.get("retired") or j.get("failed")
            or j.get("boot_dead")):
        return False
    if not j["tag"].startswith("g-refine-"):
        return False
    if GREEDY_PLACEHOLDER in j["driver"]:
        prefix = j["tag"].rsplit("-N", 1)[0]
        return os.path.exists(os.path.join(
            RAW, "%s-N%d" % (prefix, N_SHARDS[0]), "out", "results.json"))
    return True


def step():
    """ONE supervision pass (call from a FOREGROUND poll loop): collect
    finished shards (+ kill their pods), enforce deadlines, run the boot
    watchdog, then fill free slots up to MAX_LIVE with launchable shards --
    N4 gate shards first (bv first: longest pole), then injected shards by
    descending N.  Launch creates are staggered 30 s, max 6 per pass (keeps a
    pass well under a foreground call budget)."""
    collect()
    deadline()
    bootwatch()
    st = _load_state()
    live = [j for j in st["jobs"]
            if j.get("pod") and not j["received"] and not j.get("killed")]
    cand = [j for j in st["jobs"] if _launchable(j)]

    def prio(j):
        gate = GREEDY_PLACEHOLDER in j["driver"]      # False = N4 gate shard
        bv = "-bv-" in j["tag"]
        n = int(j["tag"].rsplit("-N", 1)[1])
        return (gate, not bv, -n)
    cand.sort(key=prio)
    slots = max(0, MAX_LIVE - len(live))
    launched = 0
    for j in cand:
        if launched >= min(slots, 6):
            break
        if launched:
            time.sleep(30)
        _launch(j, st["token"])
        _save_state(st)                  # persist after EVERY create
        if j.get("pod"):
            launched += 1
    st = _load_state()
    done = sum(1 for j in st["jobs"] if j["received"] and j["status"] == "0")
    fail = sum(1 for j in st["jobs"]
               if j.get("failed") or j.get("boot_dead")
               or (j["received"] and j["status"] != "0"))
    live_n = sum(1 for j in st["jobs"]
                 if j.get("pod") and not j["received"] and not j.get("killed"))
    wait = sum(1 for j in st["jobs"] if _launchable(j))
    print("step: ok %d | live %d | launchable %d | failed %d | spend $%.2f"
          % (done, live_n, wait, fail, st.get("spend_usd", 0.0)))


if __name__ == "__main__":
    # NO 'cleanup' on purpose: it would terminate ALL account pods, including
    # the protected cuvarbase-dev pod.  Use teardown/killstale (own pods only).
    {"mint": mint, "plan": plan, "canary": canary, "release": release,
     "launch": launch_one, "collect": collect, "report": report,
     "count": count, "prog": prog, "bootwatch": bootwatch,
     "killstale": killstale, "teardown": teardown, "sync": sync,
     "retoken": retoken, "deadline": deadline, "step": step,
     "start_beacon": start_beacon}[sys.argv[1]]()
