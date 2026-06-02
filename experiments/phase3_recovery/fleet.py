#!/usr/bin/env python
"""RunPod CPU fleet for the Phase 3.2 production matrix, with webhook.site result
retrieval (RunPod pods run fine but have no reachable SSH / public IP / logs; a pod
POSTs its small gzipped result JSON to a pre-minted webhook.site token and we fetch it
over HTTP -- the only retrieval method that works headlessly, proven by the probe).

Each (universe, seed) job runs on its own cpu5c/32-vCPU pod (image python:3.11, no apt:
the repo is fetched as a branch tarball via curl). Bootstrap posts a START beacon after
install, then the RESULT beacon (gzip|base64 of seed_<s>.json, or the run.log tail on
failure) with X-Job/X-Status headers.

    .venv/bin/python fleet.py launch     # mint token + create 6 pods, write _fleet/state.json
    .venv/bin/python fleet.py collect    # fetch any new results, tear down finished pods
    .venv/bin/python fleet.py cleanup    # terminate ALL pods (safety)
"""
import base64
import gzip
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".env"))
FLEET = os.path.join(HERE, "_fleet")
STATE = os.path.join(FLEET, "state.json")
OUTDIR = os.path.join(HERE, "output_prod")
REST = "https://rest.runpod.io/v1"
TGZ = ("https://github.com/PrincetonUniversity/FastTemplatePeriodogram/"
       "archive/refs/heads/phase3.2-figures.tar.gz")

# Right-sized to the PROVEN preview config (256 src / 8k freq / K<=8 ~ 81 core-hr/job);
# the original 1024/15k/K<=16 was ~15x that (~40-78h/job, ~$260) -- infeasible. This
# finishes overnight (~3h/32-vCPU, ~6h/16-vCPU, ~$20) across 3 seeds x both universes
# with the same conclusions (768 realizations/point, SE~0.017 << the FTP-vs-baseline gaps).
CONFIG = ("--nharmonics 8 --n-sources 256 --n-freq 8000 --baseline-days 1095.75 "
          "--obs-bands g,r --dense-master-epochs 60 --sparse-master-epochs 6 "
          "--k-values 1,2,4,8 --n-epochs-values 4,8,12,16,24,40 --mhls-h 8 --mbls-h 1 "
          "--greedy-select-sources 64 --greedy-select-nfreq 2000 --cost-subsample 32 "
          "--cost-n-freq 1500 --cost-k 2 --oracle-n-tau 128 --bv-bands g,r")
JOBS = [(u, s) for u in ("sesar", "bv") for s in (0, 1, 2)]
IMAGE = "python:3.11"
# capacity-fallback chain (flavor, vcpu, cloudType): bigger/secure first, then degrade
FALLBACKS = [("cpu5c", 32, "SECURE"), ("cpu3c", 32, "SECURE"), ("cpu5c", 16, "SECURE"),
             ("cpu3c", 16, "SECURE"), ("cpu5g", 16, "SECURE"), ("cpu3g", 16, "SECURE"),
             ("cpu5c", 32, "COMMUNITY"), ("cpu5c", 8, "SECURE")]


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
                               method="POST", headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read().decode())["uuid"]


def _post(token, job, suffix):
    """Shell snippet (one statement): POST $PAYLOAD with X-Job/X-Status headers."""
    return ("for i in $(seq 1 20); do curl -sf --max-time 40 -X POST "
            "https://webhook.site/%s -H 'X-Job: %s%s' -H \"X-Status: $ST\" "
            "--data-binary \"$PAYLOAD\" && break; sleep 20; done"
            % (token, job, suffix))


def _bootstrap(u, s, token):
    job = "%s-%d" % (u, s)
    driver = ("python run_production_matrix.py --universe %s --seeds %d --n-jobs -1 %s "
              "--no-figures --outdir /workspace/out" % (u, s, CONFIG))
    diag = ("( echo \"REPO=[$REPO]\"; echo --ls--; ls /workspace; echo --dl--; "
            "cat /workspace/dl.log; echo --tar--; cat /workspace/tar.log; echo --pip--; "
            "tail -6 /workspace/pip.log; echo --numpy--; "
            "python -c 'import numpy;print(numpy.__version__)' 2>&1; echo --ftp--; "
            "python -c 'import ftperiodogram;print(\"ftp_ok\")' 2>&1 ) "
            ">/workspace/diag.txt 2>&1")
    result = ("if [ -s /workspace/out/seed_%d.json ]; then "
              "PAYLOAD=$(gzip -c /workspace/out/seed_%d.json | base64 -w0); else "
              "PAYLOAD=$(tail -60 /workspace/run.log 2>/dev/null | gzip -c | base64 -w0); fi"
              % (s, s))
    lines = [
        "set +e",
        "mkdir -p /workspace",                           # volume mount point may not exist
        "cd /workspace",
        "curl -fsSL -o repo.tgz %s >dl.log 2>&1" % TGZ,
        "tar xzf repo.tgz 2>tar.log",
        "REPO=$(ls -d /workspace/FastTemplatePeriodogram-*/ 2>/dev/null | head -1)",
        # python:3.11 is Debian bookworm (PEP 668) -> --break-system-packages; PYTHONPATH
        # instead of `pip install -e .` (pure-python pkg)
        "pip install --break-system-packages -q numpy scipy nfft >/workspace/pip.log 2>&1",
        'export PYTHONPATH="$REPO"',
        diag,                                            # full setup dump
        'ST=diag; PAYLOAD=$(gzip -c /workspace/diag.txt | base64 -w0)',
        _post(token, job, "-START"),
        'cd "$REPO/experiments/phase3_recovery"',
        "ST=run; %s >/workspace/run.log 2>&1; ST=$?" % driver,
        result,
        _post(token, job, ""),
        "sleep 9000",
    ]
    return " ; ".join(lines)


def _create(u, s, token):
    script = _bootstrap(u, s, token)
    last = ""
    for flav, vcpu, cloud in FALLBACKS:
        body = {"name": "ftp-%s-%d" % (u, s), "computeType": "CPU",
                "cpuFlavorIds": [flav], "vcpuCount": vcpu, "imageName": IMAGE,
                "containerDiskInGb": 20, "ports": ["22/tcp"], "cloudType": cloud,
                "dockerStartCmd": ["bash", "-c", script]}
        t = _runpod("POST", "/pods", body)
        m = re.search(r'"id":"([a-z0-9]+)"', t)
        if m:
            return m.group(1), "%s/%d/%s" % (flav, vcpu, cloud)
        last = t
    return None, last[:140]


def launch():
    os.makedirs(FLEET, exist_ok=True)
    token = _mint_token()
    jobs = []
    for u, s in JOBS:
        pid, info = _create(u, s, token)
        jobs.append({"tag": "%s-%d" % (u, s), "u": u, "s": s, "pod": pid, "info": info,
                     "received": False, "status": None, "started": False})
        print("launch %s-%d -> %s (%s)" % (u, s, pid or "FAIL", info))
        time.sleep(2)
    json.dump({"token": token, "jobs": jobs}, open(STATE, "w"), indent=2)
    print("token", token, "| webhook https://webhook.site/#!/%s" % token)


def canary():
    """Launch ONLY sesar-0 first (rest pending) so its START beacon validates setup
    before committing the full fleet; then `relaunch` fills the other 5."""
    os.makedirs(FLEET, exist_ok=True)
    token = _mint_token()
    jobs = []
    for u, s in JOBS:
        if (u, s) == ("sesar", 0):
            pid, info = _create(u, s, token)
        else:
            pid, info = None, "pending"
        jobs.append({"tag": "%s-%d" % (u, s), "u": u, "s": s, "pod": pid, "info": info,
                     "received": False, "status": None, "started": False})
        print("canary %s-%d -> %s (%s)" % (u, s, pid or "pending", info))
    json.dump({"token": token, "jobs": jobs}, open(STATE, "w"), indent=2)
    print("token", token)


def start_beacon():
    """Decode + print each job's START beacon content (setup diagnostic)."""
    st = json.load(open(STATE))
    url = ("https://webhook.site/token/%s/requests?sorting=oldest&per_page=100"
           % st["token"])
    data = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())["data"]
    for r in data:
        job = _headers(r).get("x-job", "")
        if job.endswith("-START"):
            try:
                txt = gzip.decompress(base64.b64decode(r["content"])).decode("utf-8", "replace")
            except Exception as e:
                txt = "<decode err %s>" % e
            print("%s: %s" % (job, txt.strip()[:200]))


def relaunch():
    """Retry any job that has no pod yet (capacity ran out), reusing the token."""
    st = json.load(open(STATE))
    token = st["token"]
    for j in st["jobs"]:
        if j.get("pod") or j["received"]:
            continue
        pid, info = _create(j["u"], j["s"], token)
        j["pod"], j["info"] = pid, info
        print("relaunch %s -> %s (%s)" % (j["tag"], pid or "FAIL", info))
        time.sleep(2)
    json.dump(st, open(STATE, "w"), indent=2)
    pending = sum(1 for j in st["jobs"] if not j.get("pod"))
    print("pending (no pod): %d/6" % pending)


def _headers(r):
    return {k.lower(): (v[0] if isinstance(v, list) else v)
            for k, v in (r.get("headers") or {}).items()}


def collect():
    st = json.load(open(STATE))
    token = st["token"]
    url = "https://webhook.site/token/%s/requests?sorting=oldest&per_page=100" % token
    data = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())["data"]
    posts = {}                                       # X-Job -> latest request
    for r in data:
        job = _headers(r).get("x-job", "")
        if job:
            posts[job] = r
    os.makedirs(OUTDIR, exist_ok=True)
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
            raw = gzip.decompress(base64.b64decode(r["content"]))
        except Exception as e:
            print("%s: decode fail %s" % (j["tag"], e)); continue
        if status == "0":
            open(os.path.join(OUTDIR, "%s.json" % j["tag"]), "wb").write(raw)
            print("%s: RESULT received (status 0)" % j["tag"])
        else:
            open(os.path.join(OUTDIR, "%s.FAILED.log" % j["tag"]), "wb").write(raw)
            print("%s: FAILED status=%s (log saved)" % (j["tag"], status))
        j["received"], j["status"] = True, status
        if j["pod"]:
            _runpod("DELETE", "/pods/%s" % j["pod"])
            print("%s: pod %s terminated" % (j["tag"], j["pod"]))
    json.dump(st, open(STATE, "w"), indent=2)
    started = sum(1 for j in st["jobs"] if j["started"])
    done = sum(1 for j in st["jobs"] if j["received"])
    ok = sum(1 for j in st["jobs"] if j["status"] == "0")
    print("started %d/6 | received %d/6 | ok %d/6" % (started, done, ok))
    return done == len(st["jobs"])


def count():
    """Print the number of distinct RESULT beacons posted (for the monitor)."""
    st = json.load(open(STATE))
    try:
        url = ("https://webhook.site/token/%s/requests?per_page=100" % st["token"])
        d = json.loads(urllib.request.urlopen(url, timeout=25).read().decode())["data"]
    except Exception:
        print(0); return
    tags = {j["tag"] for j in st["jobs"]}
    res = {_headers(r).get("x-job", "") for r in d}
    print(len(tags & res))


def cleanup():
    text = _runpod("GET", "/pods")
    ids = re.findall(r'"id":"([a-z0-9]{10,})"', text)
    for pid in ids:
        _runpod("DELETE", "/pods/%s" % pid)
    print("terminated %d pods" % len(ids))


if __name__ == "__main__":
    {"launch": launch, "canary": canary, "start_beacon": start_beacon,
     "relaunch": relaunch, "collect": collect, "count": count,
     "cleanup": cleanup}[sys.argv[1]]()
