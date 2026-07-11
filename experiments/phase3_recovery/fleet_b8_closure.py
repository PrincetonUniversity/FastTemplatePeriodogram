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
  (``BAD_MACHINES``) are deleted at create time and don't burn a boot attempt;
* canary-first: ``canary`` launches ONLY e2-jitter-0 + g-refine-sesar-0
  (~$2-3); ``report`` compares measured core-hr against the estimates; the
  FULL batch (cap $25) is released only after a human-approved estimate.

    .venv/bin/python fleet_b8_closure.py mint       # fresh token + pending state (no pods)
    .venv/bin/python fleet_b8_closure.py plan       # dry-run: job table + drivers, no network
    .venv/bin/python fleet_b8_closure.py canary     # launch the 2 canary pods
    .venv/bin/python fleet_b8_closure.py report     # canary timing vs estimate + $ projection
    .venv/bin/python fleet_b8_closure.py release    # launch every job without a pod yet
    .venv/bin/python fleet_b8_closure.py collect    # fetch results, extract, kill done pods
    .venv/bin/python fleet_b8_closure.py prog       # latest PROG log tail per running job
    .venv/bin/python fleet_b8_closure.py bootwatch  # one boot-watchdog pass (poll from foreground)
    .venv/bin/python fleet_b8_closure.py killstale 12   # kill OWN pods past 12 h wall
    .venv/bin/python fleet_b8_closure.py teardown   # delete OWN pods; verify only cuvarbase-dev lives
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
# cpu3c deprioritized to the tail: the June fleet measured it slow AND the
# 2026-07-11 canary boot failure was a cpu3c host (machine aehmymvwnx01).
FALLBACKS = [("cpu5c", 32, "SECURE"), ("cpu5c", 16, "SECURE"), ("cpu5g", 16, "SECURE"),
             ("cpu3g", 16, "SECURE"), ("cpu5c", 32, "COMMUNITY"), ("cpu5c", 8, "SECURE"),
             ("cpu3c", 32, "SECURE"), ("cpu3c", 16, "SECURE")]

# Boot watchdog (v3): boot failures bill but emit NO beacon (2026-07-11 canary:
# 50 min at uptime 0 s, $0.81, zero science).  No START beacon within
# BOOT_LIMIT_MIN of pod create => delete + relaunch on a different flavor;
# at most MAX_BOOT_ATTEMPTS pod creates per job, then the job is boot-dead.
BOOT_LIMIT_MIN = 20
MAX_BOOT_ATTEMPTS = 2
BAD_MACHINES = {"aehmymvwnx01"}  # host of the 2026-07-11 never-booted pod

# The ONE pre-existing pod on the account (John's separate cuvarbase GPU
# workstream).  NEVER stopped, deleted, or modified by this launcher.
CUVARBASE_POD = "6nzaafwb96j38r"

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
        "--n-epochs-values 4,8,12,16,24,40 --mhls-h 8 --mbls-h 1 "
        "--greedy-select-sources 64 --greedy-select-nfreq 2000 --bv-bands g,r "
        "--with-ce --no-figures --outdir /workspace/out")
# grid-closure: same 10k grid + peak refinement, N-sweep only (the flagged
# cells all live in the N-sweep; --fixed-k 2 matches arm a for McNemar pairing)
ARM_REFINE = " --n-freq 10000 --stages n_sweep --fixed-k 2 --refine"

# arm-(e) relaunch: ONE (scenario, seed) per pod (B8 lesson: shard; one job
# must not set the tail).  --n-epochs-values 4 5 6 8 12 re-grids N so the
# 4->8 transition where the MRA-gap hypothesis is falsifiable is populated
# (B8E-1: the old {4,8,12,20,40} saturated 4/5 canary cells).
JOINT = ("python run_joint_vs_pipeline.py --universe sesar --k 2 --n-sources 256 "
         "--n-harmonics 8 --f-min 1.4 --f-max 3.6 --n-freq 2000 --baseline-days 60 "
         "--mean-mag 19 --max-iter 8 --n-jobs -1 --n-epochs-values 4 5 6 8 12 "
         "--seed %(seed)d --no-figure%(extra)s --out /workspace/out/%(scen)s_seed%(seed)d")


def _prod(u, s, extra):
    return PROD % {"u": u, "s": s} + extra


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
        + [{"tag": "g-refine-sesar-%d" % s, "est": 45,
            "driver": _prod("sesar", s, ARM_REFINE)} for s in (0, 1, 2)]
        + [{"tag": "g-refine-bv-0", "est": 68,
            "driver": _prod("bv", 0, ARM_REFINE)}])

CANARY_TAGS = ("e2-jitter-0", "g-refine-sesar-0")


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
    if pid == CUVARBASE_POD:
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


def _prog_beacon(token, tag):
    """Backgrounded 15-min progress beacon (max 4 posts -- webhook.site free
    tokens store ~100 requests; 16 jobs x (START+RESULT+4 PROG) = 96).  Uses
    its own payload file so it never clobbers /workspace/payload.b64."""
    return ("( for i in 1 2 3 4; do sleep 900; "
            "tail -40 /workspace/run.log 2>/dev/null | gzip -c | base64 -w0 "
            ">/workspace/prog.b64; "
            "curl -sf --max-time 60 -X POST https://webhook.site/%s "
            "-H 'X-Job: %s-PROG' --data-binary @/workspace/prog.b64; "
            "done ) >/dev/null 2>&1 &" % (token, tag))


def _bootstrap(tag, driver, token):
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
        _prog_beacon(token, tag),
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


def _create(tag, driver, token, avoid=()):
    """Create a pod, skipping flavors in ``avoid`` (boot-watchdog retries) and
    deleting-and-continuing if the pod lands on a BAD_MACHINES host (a bad
    host is RunPod's placement, not a boot attempt of ours)."""
    script = _bootstrap(tag, driver, token)
    last = ""
    for flav, vcpu, cloud in FALLBACKS:
        if flav in avoid:
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
        return pid, "%s/%d/%s@%s" % (flav, vcpu, cloud, mach or "?")
    return None, (last[:140] or "no fallback flavor available")


def _launch(j, token):
    """Create a pod for job ``j`` honoring the boot-attempt cap and the job's
    avoid-flavor list; updates the job record in place.  Failed CREATEs (no
    pod id returned -- nothing billed) do not consume a boot attempt."""
    if j.get("boot_dead") or j.get("boot_attempts", 0) >= MAX_BOOT_ATTEMPTS:
        j["boot_dead"] = True
        print("launch %s: SKIP -- boot-attempt cap (%d) reached"
              % (j["tag"], MAX_BOOT_ATTEMPTS))
        return None
    pid, info = _create(j["tag"], j["driver"], token,
                        avoid=tuple(j.get("avoid_flavors") or ()))
    j["pod"], j["info"] = pid, info
    j["created"] = time.time() if pid else None
    if pid:
        j["boot_attempts"] = j.get("boot_attempts", 0) + 1
        print("launch %s -> %s (%s) [boot attempt %d/%d]"
              % (j["tag"], pid, info, j["boot_attempts"], MAX_BOOT_ATTEMPTS))
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


def _load_state():
    return json.load(open(STATE))


def _save_state(st):
    json.dump(st, open(STATE, "w"), indent=2)


def _new_job(j, pod=None, info="pending"):
    return {"tag": j["tag"], "est": j["est"], "driver": j["driver"], "pod": pod,
            "info": info, "received": False, "status": None, "started": False,
            "created": None}


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------
def plan():
    """Dry-run: the job table + full driver strings.  Zero network, zero paid."""
    tags = [j["tag"] for j in JOBS]
    assert len(tags) == len(set(tags)), "duplicate job tags: %s" % tags
    assert all(t in tags for t in CANARY_TAGS), "canary tag not in JOBS"
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


def release():
    """Launch every job that has no pod yet (post-canary, or capacity retry)."""
    st = _load_state()
    for j in st["jobs"]:
        if j.get("pod") or j["received"] or j.get("boot_dead"):
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
    for j in st["jobs"]:
        if posts.get(j["tag"] + "-START"):
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
        starts = {_headers(r).get("x-job", "") for r in _requests(st["token"])}
    except Exception as e:
        print("bootwatch: beacon fetch failed (%s) -- no action taken" % e)
        return
    now = time.time()
    for j in st["jobs"]:
        if not j["started"] and (j["tag"] + "-START") in starts:
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
            _delete_pod(j["pod"])
            j["killed"] = True
            j["info"] = (j.get("info") or "") + " KILLED-STALE@%.1fh" % age
            print("killstale %s: pod %s (%.1f h) terminated" %
                  (j["tag"], j["pod"], age))
            n += 1
    _save_state(st)
    print("killed %d stale pods" % n)


def teardown():
    """Delete every pod THIS state launched, then verify via GET /pods that the
    only survivor on the account is the protected cuvarbase-dev pod."""
    st = _load_state()
    for j in st["jobs"]:
        if j.get("pod"):
            _delete_pod(j["pod"])
            print("teardown %s: DELETE pod %s" % (j["tag"], j["pod"]))
    time.sleep(5)
    survivors = _live_pods()
    print("live pods after teardown:")
    for pid, name in survivors:
        print("  %s  %s%s" % (pid, name,
                              "  [protected cuvarbase-dev]"
                              if pid == CUVARBASE_POD else ""))
    others = [p for p in survivors if p[0] != CUVARBASE_POD]
    if others:
        print("WARNING: %d non-cuvarbase pod(s) still live (NOT touched -- "
              "delete by id only if they are yours): %s"
              % (len(others), [p[0] for p in others]))
        sys.exit(1)
    print("OK: no foreign pods remain (cuvarbase-dev %s may or may not exist "
          "-- external workstream, never ours to manage)" % CUVARBASE_POD)


if __name__ == "__main__":
    # NO 'cleanup' on purpose: it would terminate ALL account pods, including
    # the protected cuvarbase-dev pod.  Use teardown/killstale (own pods only).
    {"mint": mint, "plan": plan, "canary": canary, "release": release,
     "collect": collect, "report": report, "count": count, "prog": prog,
     "bootwatch": bootwatch, "killstale": killstale, "teardown": teardown,
     "start_beacon": start_beacon}[sys.argv[1]]()
