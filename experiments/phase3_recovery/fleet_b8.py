#!/usr/bin/env python
"""RunPod CPU fleet for WP B8 -- the consolidated rerun (EXECUTION_PLAN.md).

Same proven plumbing as fleet.py (webhook.site beacons; no SSH/IP/logs on RunPod
CPU pods) with three B8 deltas:

* heterogeneous job table (arms a-e; arm f added by `addf` only after the budget
  check) instead of a single (universe, seed) matrix;
* the RESULT beacon is a tar.gz of the WHOLE /workspace/out dir, so WP B1's
  per-source npz (Wilson/McNemar/alias re-scoring inputs) come back, not just
  the seed json;
* canary-first protocol: `canary` launches a-sesar-0 + e-canary only; `report`
  compares measured core-hr (START->RESULT beacon timestamps x vCPU) against the
  estimates in B8_COST_ESTIMATE.md; `release` fills the rest only after that
  gate (within ~2x) passes.

Grid note: the headline 8k grid fails WP B7's hard guard at T=3yr (1.83 grid
points per Rayleigh width < 2.0), so production arms run --n-freq 10000 (2.28,
passes hard tier, warn fires honestly) and the convergence arm runs 20000 to
keep the intended 2x ratio.

    .venv/bin/python fleet_b8.py canary     # mint token + the 2 canary pods
    .venv/bin/python fleet_b8.py report     # canary timing vs estimate + $ projection
    .venv/bin/python fleet_b8.py release    # launch every job without a pod yet
    .venv/bin/python fleet_b8.py collect    # fetch results, extract, kill done pods
    .venv/bin/python fleet_b8.py addf       # append arm-f jobs (budget permitting)
    .venv/bin/python fleet_b8.py cleanup    # terminate ALL pods (safety)
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
FLEET = os.path.join(HERE, "_fleet_b8")
STATE = os.path.join(FLEET, "state.json")
RAW = os.path.join(HERE, "rerun_202606", "raw")
REST = "https://rest.runpod.io/v1"
TGZ = ("https://github.com/PrincetonUniversity/FastTemplatePeriodogram/"
       "archive/refs/heads/dev.tar.gz")
IMAGE = "python:3.11"
RATE = 0.041                     # $/core-hr, measured on the 2026-06-02 fleet
FALLBACKS = [("cpu5c", 32, "SECURE"), ("cpu3c", 32, "SECURE"), ("cpu5c", 16, "SECURE"),
             ("cpu3c", 16, "SECURE"), ("cpu5g", 16, "SECURE"), ("cpu3g", 16, "SECURE"),
             ("cpu5c", 32, "COMMUNITY"), ("cpu5c", 8, "SECURE")]

# ----------------------------------------------------------------------
# Job table.  est = core-hr from B8_COST_ESTIMATE.md (x1.25 for the 8k->10k grid).
# ----------------------------------------------------------------------
PROD = ("python run_production_matrix.py --universe %(u)s --seeds %(s)d --n-jobs -1 "
        "--nharmonics 8 --n-sources 256 --baseline-days 1095.75 --obs-bands g,r "
        "--dense-master-epochs 60 --sparse-master-epochs 6 --k-values 1,2,4,8 "
        "--n-epochs-values 4,8,12,16,24,40 --mhls-h 8 --mbls-h 1 "
        "--greedy-select-sources 64 --greedy-select-nfreq 2000 --bv-bands g,r "
        "--with-ce --no-figures --outdir /workspace/out")
ARM_A = " --n-freq 10000 --stages k_sparse,n_sweep --fixed-k 2"
COST = (" --cost-subsample 32 --cost-n-freq 1500 --cost-k 2 --oracle-n-tau 128")

# fail-tracking loop: ST picks up the worst per-seed exit code
JOINT = ("RC=0; for s in %(seeds)s; do "
         "python run_joint_vs_pipeline.py --universe sesar --k 2 --n-sources 256 "
         "--n-harmonics 8 --f-min 1.4 --f-max 3.6 --n-freq 2000 --baseline-days 60 "
         "--mean-mag 19 --max-iter 8 --n-jobs -1 --seed $s --no-figure%(extra)s "
         "--out /workspace/out/%(scen)s_seed$s || RC=$?; done; (exit $RC)")


def _prod(u, s, extra):
    return PROD % {"u": u, "s": s} + extra


def _joint(scen, seeds, extra):
    return JOINT % {"scen": scen, "seeds": " ".join(str(s) for s in seeds),
                    "extra": (" " + extra.strip()) if extra else ""}


JOBS = [
    # arm (a): capped-MHLS N-sweep + sparse K-sweep, both universes
    {"tag": "a-sesar-0", "est": 63, "driver": _prod("sesar", 0, ARM_A)},
    {"tag": "a-sesar-1", "est": 63, "driver": _prod("sesar", 1, ARM_A)},
    {"tag": "a-sesar-2", "est": 63, "driver": _prod("sesar", 2, ARM_A)},
    {"tag": "a-bv-0",    "est": 94, "driver": _prod("bv", 0, ARM_A)},
    # arm (b): robustness arms, 1 seed each
    {"tag": "b-holdout-0", "est": 63,
     "driver": _prod("sesar", 0, ARM_A + " --library-holdout-frac 0.5")},
    {"tag": "b-xuniv-0", "est": 63,
     "driver": _prod("sesar", 0, ARM_A + " --truth-universe bv")},
    {"tag": "b-bandamp-0", "est": 63,
     "driver": _prod("sesar", 0, ARM_A + " --band-amp-ratio 1.4")},
    # arm (c): grid-convergence N-sweep at 2x the production grid
    {"tag": "c-grid20k-0", "est": 55,
     "driver": _prod("sesar", 0, " --n-freq 20000 --stages n_sweep --fixed-k 2")},
    # arm (d): empirical-error revalidation, full matrix (all stages)
    {"tag": "d-empirical-0", "est": 101,
     "driver": _prod("sesar", 0, " --n-freq 10000" + COST + " --err-model empirical")},
    # arm (e): joint/EM, 4 scenario pods (seeds sequential in-pod);
    # e-canary covers jitter seed 0 so the jitter pod only runs 1,2
    {"tag": "e-canary", "est": 25,
     "driver": _joint("jitter", [0], "--intrinsic-jitter 0.15")},
    {"tag": "e-jitter", "est": 47,
     "driver": _joint("jitter", [1, 2], "--intrinsic-jitter 0.15")},
    {"tag": "e-base", "est": 70, "driver": _joint("base", [0, 1, 2], "")},
    {"tag": "e-holdout", "est": 70,
     "driver": _joint("holdout", [0, 1, 2], "--library-holdout-frac 0.5")},
    {"tag": "e-lowsnr", "est": 70,
     "driver": _joint("lowsnr", [0, 1, 2], "--amplitude 0.1")},
]
CANARY_TAGS = ("a-sesar-0", "e-canary")
# arm (f), appended by `addf` ONLY if projected spend stays <= ~$45
JOBS_F = [
    {"tag": "f-bv-1", "est": 94, "driver": _prod("bv", 1, ARM_A)},
    {"tag": "f-bv-2", "est": 94, "driver": _prod("bv", 2, ARM_A)},
]


# ----------------------------------------------------------------------
# RunPod / webhook plumbing (proven in fleet.py)
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
        'cd "$REPO/experiments/phase3_recovery"',
        "ST=run; { %s ; } >/workspace/run.log 2>&1; ST=$?" % driver,
        result,
        _post(token, tag, ""),
        "sleep 9000",
    ]
    return " ; ".join(lines)


def _create(tag, driver, token):
    script = _bootstrap(tag, driver, token)
    last = ""
    for flav, vcpu, cloud in FALLBACKS:
        body = {"name": "b8-%s" % tag, "computeType": "CPU",
                "cpuFlavorIds": [flav], "vcpuCount": vcpu, "imageName": IMAGE,
                "containerDiskInGb": 20, "ports": ["22/tcp"], "cloudType": cloud,
                "dockerStartCmd": ["bash", "-c", script]}
        t = _runpod("POST", "/pods", body)
        m = re.search(r'"id":"([a-z0-9]+)"', t)
        if m:
            return m.group(1), "%s/%d/%s" % (flav, vcpu, cloud)
        last = t
    return None, last[:140]


def _vcpu(info):
    m = re.match(r"[a-z0-9]+/(\d+)/", info or "")
    return int(m.group(1)) if m else 32


def _requests(token):
    """All beacon requests, paged (B8 posts ~30 beacons + retries)."""
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
            "info": info, "received": False, "status": None, "started": False}


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------
def canary():
    """Mint token; launch ONLY the canary pair; everything else pending."""
    os.makedirs(FLEET, exist_ok=True)
    token = _mint_token()
    jobs = []
    for j in JOBS:
        if j["tag"] in CANARY_TAGS:
            pid, info = _create(j["tag"], j["driver"], token)
            time.sleep(2)
        else:
            pid, info = None, "pending"
        jobs.append(_new_job(j, pid, info))
        print("canary %s -> %s (%s)" % (j["tag"], pid or "pending", info))
    _save_state({"token": token, "jobs": jobs})
    print("token", token, "| webhook https://webhook.site/#!/%s" % token)


def release():
    """Launch every job that has no pod yet (post-canary, or capacity retry).

    Jobs with ``held: true`` in state.json are NEVER launched (arm-e deferral);
    clear the flag deliberately to re-enable them."""
    st = _load_state()
    for j in st["jobs"]:
        if j.get("pod") or j["received"] or j.get("held"):
            continue
        pid, info = _create(j["tag"], j["driver"], st["token"])
        j["pod"], j["info"] = pid, info
        print("launch %s -> %s (%s)" % (j["tag"], pid or "FAIL", info))
        time.sleep(2)
    _save_state(st)
    pending = sum(1 for j in st["jobs"] if not j.get("pod") and not j["received"])
    print("pending (no pod): %d/%d" % (pending, len(st["jobs"])))


def addf():
    """Append arm-f jobs (call only after `report` projects spend <= ~$45)."""
    st = _load_state()
    have = {j["tag"] for j in st["jobs"]}
    for j in JOBS_F:
        if j["tag"] not in have:
            st["jobs"].append(_new_job(j))
            print("queued %s (launch with `release`)" % j["tag"])
    _save_state(st)


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
    # beacon can be a re-run's failure — never let it shadow a good result
    for r in _requests(st["token"]):
        job = _headers(r).get("x-job", "")
        if not job:
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
            _runpod("DELETE", "/pods/%s" % j["pod"])
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
        if job:
            seen[job] = r.get("created_at")
    total_meas = total_proj = 0.0
    print("%-14s %5s %9s %9s %7s" % ("job", "vcpu", "est c-hr", "meas c-hr", "ratio"))
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
            print("%-14s %5d %9.0f %9.1f %6.1fx"
                  % (j["tag"], vcpu, j["est"], meas, meas / j["est"]))
        else:
            print("%-14s %5d %9.0f %9s %7s"
                  % (j["tag"], vcpu, j["est"], "-", "-"))
    print("\nmeasured so far: %.0f core-hr (~$%.1f)" % (total_meas, total_meas * RATE))
    print("projected total: %.0f core-hr (~$%.1f) at $%.3f/core-hr"
          % (total_proj, total_proj * RATE, RATE))


def count():
    st = _load_state()
    try:
        res = {_headers(r).get("x-job", "") for r in _requests(st["token"])}
    except Exception:
        print(0)
        return
    print(len({j["tag"] for j in st["jobs"]} & res))


def cleanup():
    text = _runpod("GET", "/pods")
    ids = re.findall(r'"id":"([a-z0-9]{10,})"', text)
    for pid in ids:
        _runpod("DELETE", "/pods/%s" % pid)
    print("terminated %d pods" % len(ids))


if __name__ == "__main__":
    {"canary": canary, "release": release, "addf": addf, "collect": collect,
     "report": report, "count": count, "cleanup": cleanup,
     "start_beacon": start_beacon}[sys.argv[1]]()
