#!/usr/bin/env python
"""RunPod CPU fleet for the WP E5 sparse-regime FTP backlog (2026-07-11).

A LOCAL workflow is concurrently running the full E5 task matrix
(experiments/phase4_demo/e5_sparse/run_e5.py) at ~40 tasks/hr; this fleet
computes ONLY the ftp_1band/ftp_mb backlog remotely and merges results into
the local runner's output/ so it finishes early on its own.  The local
runner is never disturbed: merges are atomic no-clobber (hardlink-then-
unlink; an existing full .npz ALWAYS wins), .part/.fail files are never
touched or shipped back.

Plumbing forked from fleet_b8_closure.py (same webhook.site beacon pattern,
the bash -n bootstrap guard and the '& true' beacon-join fix -- the '& ;'
bug history is in B8_COST_ESTIMATE.md), with the E5 deltas:

* OWN namespace ``_fleet_e5/state.json`` + FRESH webhook tokens, ONE PER POD
  (chunked result beacons would blow the ~100-request/token cap on a shared
  token); the concurrent arm-b ``_fleet_b8c`` state/token/pods are never read
  or written, and the protected ``cuvarbase-dev`` pod is name-guarded on
  every delete path;
* payload is a tarball STAGED FROM THE LOCAL WORKING TREE (not the GitHub
  branch tarball): the numerical-parity gate compares against local outputs,
  so pods must run byte-identical code; deps are PINNED to the local venv
  (numpy==2.0.2 scipy==1.13.1 nfft==0.1);
* each pod runs an explicit remaining-task shard (backlog enumerated at
  ``build`` time, minus every FULL local .npz; .part files are ignored --
  harmless duplication) via shard_driver.py -> run_e5.run_task, then posts
  results as <=~1.8 MB base64 tar.gz chunk beacons + a sha256 MANIFEST;
* ``gate``: before any pod exists, one already-completed local task per FTP
  method is recomputed in a python-3.11 venv with the pod's exact pins and
  compared to the local .npz (identical subsample indices; |stat diff| <=
  1e-10).  A failed gate BLOCKS launch/merge;
* pool exhaustion: 32-vCPU flavors ONLY (cpu5c SECURE -> COMMUNITY ->
  cpu3c SECURE); never downgrade vCPU -- wait and retry instead.

    .venv/bin/python fleet_e5.py build      # enumerate backlog, shard, build payload
    .venv/bin/python fleet_e5.py gate       # numerical-parity gate (venv, no pods)
    .venv/bin/python fleet_e5.py stage      # upload payload tarball -> URL
    .venv/bin/python fleet_e5.py launch     # mint per-pod tokens + create pods
    .venv/bin/python fleet_e5.py poll [sec] # foreground loop: bootwatch+collect+merge
    .venv/bin/python fleet_e5.py status     # one-line fleet status
    .venv/bin/python fleet_e5.py teardown   # delete OWN pods; verify none remain
    .venv/bin/python fleet_e5.py report     # final JSON (tasks merged/skipped/cost)
"""
import base64
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ENV = os.path.abspath(os.path.join(REPO, "..", ".env"))
FLEET = os.path.join(HERE, "_fleet_e5")
STATE = os.path.join(FLEET, "state.json")
E5 = os.path.join(REPO, "experiments", "phase4_demo", "e5_sparse")
E4 = os.path.join(REPO, "experiments", "phase4_demo", "e4_recovery")
OUTDIR = os.path.join(E5, "output")
DATA_HOME = os.path.expanduser("~/.ftperiodogram_data")
REST = "https://rest.runpod.io/v1"
IMAGE = "python:3.11"
RATE = 0.041                    # $/core-hr (2026-06-02 fleet, measured)
PINS = "numpy==2.0.2 scipy==1.13.1 nfft==0.1"   # local .venv versions
N_PODS = 8                      # supervisor-authorized 2026-07-11 (cap $8)
CHUNK_BYTES = 1350 * 1024       # binary per beacon; ~1.8 MB after base64
BOOT_LIMIT_MIN = 20
MAX_BOOT_ATTEMPTS = 2
STALE_HOURS = 2.0               # hard kill for own pods (jobs are ~30 min)
# 32-vCPU ONLY -- supervisor: retry pool exhaustion, never downgrade vCPU
FLAVORS = [("cpu5c", 32, "SECURE"), ("cpu5c", 32, "COMMUNITY"),
           ("cpu3c", 32, "SECURE")]

N_GRID = (8, 12, 16, 24, 40)
REPS = (0, 1, 2)
FTP_METHODS = ("ftp_1band", "ftp_mb")

CUVARBASE_NAME = "cuvarbase-dev"


def _is_protected(pid, name=None):
    """True for John's cuvarbase-dev pod, by NAME (ids rotate)."""
    if name is None and pid:
        m = re.search(r'"name"\s*:\s*"([^"]*)"',
                      _runpod("GET", "/pods/%s" % pid) or "")
        name = m.group(1) if m else ""
    return (name or "").strip().lower() == CUVARBASE_NAME


# ----------------------------------------------------------------------
# RunPod / webhook plumbing (fleet_b8_closure.py, verbatim where possible)
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
    except Exception as e:
        return '{"error": "%s"}' % e


def _delete_pod(pid):
    """DELETE one pod -- cuvarbase-dev name-guard on EVERY delete path."""
    if not pid:
        return
    if _is_protected(pid):
        print("REFUSING to delete protected pod %s (cuvarbase-dev)" % pid)
        return
    _runpod("DELETE", "/pods/%s" % pid)


def _live_pods():
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
    return json.loads(
        urllib.request.urlopen(r, timeout=30).read().decode())["uuid"]


def _requests(token):
    out, page = [], 1
    while page <= 5:
        url = ("https://webhook.site/token/%s/requests?sorting=oldest"
               "&per_page=100&page=%d" % (token, page))
        d = json.loads(
            urllib.request.urlopen(url, timeout=30).read().decode())["data"]
        out.extend(d)
        if len(d) < 100:
            break
        page += 1
    return out


def _headers(r):
    return {k.lower(): (v[0] if isinstance(v, list) else v)
            for k, v in (r.get("headers") or {}).items()}


def _post(token, job, suffix):
    # payload from a FILE (base64 far past the 128 KiB argv cap)
    return ("for i in $(seq 1 20); do curl -sf --max-time 90 -X POST "
            "https://webhook.site/%s -H 'X-Job: %s%s' -H \"X-Status: $ST\" "
            "--data-binary @/workspace/payload.b64 && break; sleep 20; done"
            % (token, job, suffix))


def _prog_beacon(token, tag):
    """15-min progress beacon, max 4 posts.  MUST NOT end in a bare '&':
    the ' ; ' line join would make '& ;' -- a bash syntax error that killed
    three B8-closure pods at uptime 0 (see B8_COST_ESTIMATE.md).  The
    trailing 'true' keeps the join valid."""
    return ("( for i in 1 2 3 4; do sleep 900; "
            "tail -40 /workspace/run.log 2>/dev/null | gzip -c | base64 -w0 "
            ">/workspace/prog.b64; "
            "curl -sf --max-time 60 -X POST https://webhook.site/%s "
            "-H 'X-Job: %s-PROG' --data-binary @/workspace/prog.b64; "
            "done ) >/dev/null 2>&1 & true" % (token, tag))


def _check_script_syntax(script, tag):
    """`bash -n` every generated bootstrap before any pod exists."""
    p = subprocess.run(["bash", "-n", "-c", script],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("BOOTSTRAP SYNTAX ERROR for %s -- refusing:\n%s"
                         % (tag, p.stderr.strip()[:500]))


def _load_state():
    return json.load(open(STATE))


def _save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE)


# ----------------------------------------------------------------------
# Pod-side scripts (shipped inside the payload tarball)
# ----------------------------------------------------------------------
SHARD_DRIVER = '''#!/usr/bin/env python
"""E5 fleet pod driver: run an EXPLICIT [uid, N, rep, mode, method] task
list through run_e5.run_task (same code path, atomic .part -> rename,
skip-if-exists).  Longest grids first for pool tail packing."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_e5, run_e4    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasklist")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--block", type=int, default=4096)
    a = ap.parse_args()
    tasks = json.load(open(a.tasklist))
    with open(os.path.join(run_e5.E4DIR, "star_table.json")) as fh:
        by_uid = {s["uid"]: s for s in json.load(fh)["stars"]}
    os.makedirs(run_e5.OUTDIR, exist_ok=True)
    deadline = time.time() + 10 ** 9

    def cost(t):
        s = by_uid[t[0]]
        g = s["grid"][s["primary_band"]] if t[4] == "ftp_1band" \\
            else s["grid"]["joint"]
        return g["n_freq"]

    tasks = sorted(tasks, key=cost, reverse=True)
    payloads = [(by_uid[u], m, int(N), int(rep), mode, deadline, a.block)
                for u, N, rep, mode, m in tasks]
    vocab = run_e4.build_vocab_coeffs()
    results = []
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=run_e4._init_worker,
                  initargs=(vocab,)) as pool:
        for res in pool.imap_unordered(run_e5.run_task, payloads):
            results.append(res)
            print("  %s %s %d/%d %.1fs" % res, flush=True)
    bad = [r for r in results if r[1] not in ("done", "exists")]
    print("DRIVER_DONE %d ok %d bad" % (len(results) - len(bad), len(bad)),
          flush=True)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
'''

POST_RESULTS = '''#!/usr/bin/env python
"""E5 fleet pod result poster: tar.gz the completed npz in chunks
(<=CHUNK binary -> <= ~1.8 MB base64, webhook.site-safe), POST each with
retries, then a gzip MANIFEST beacon (sha256 + sizes + driver exit +
.fail list).  Never ships .part.npz or zero-byte files."""
import base64, gzip, hashlib, io, json, os, sys, tarfile, time
import urllib.request

token, tag, driver_status = sys.argv[1], sys.argv[2], sys.argv[3]
CHUNK = %(chunk)d
OUT = "output"


def post(suffix, body, extra=None, tries=8):
    hdr = {"X-Job": tag + suffix, "X-Status": str(driver_status),
           "Content-Type": "application/octet-stream"}
    hdr.update(extra or {})
    for _ in range(tries):
        try:
            r = urllib.request.Request("https://webhook.site/" + token,
                                       data=body, headers=hdr, method="POST")
            urllib.request.urlopen(r, timeout=90).read()
            return True
        except Exception as e:
            print("post %%s retry: %%s" %% (suffix, e), flush=True)
            time.sleep(15)
    return False


files = sorted(f for f in os.listdir(OUT)
               if f.endswith(".npz") and not f.endswith(".part.npz")
               and os.path.getsize(os.path.join(OUT, f)) > 0)
man = {"files": {}, "driver_exit": driver_status,
       "fails": sorted(f for f in os.listdir(OUT) if f.endswith(".fail"))}
for f in files:
    b = open(os.path.join(OUT, f), "rb").read()
    man["files"][f] = {"sha256": hashlib.sha256(b).hexdigest(),
                       "size": len(b)}
chunks, cur, sz = [], [], 0
for f in files:
    s = man["files"][f]["size"]
    if cur and sz + s > CHUNK:
        chunks.append(cur)
        cur, sz = [], 0
    cur.append(f)
    sz += s
if cur:
    chunks.append(cur)
man["n_chunks"] = len(chunks)
ok = True
for i, ch in enumerate(chunks):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in ch:
            tf.add(os.path.join(OUT, f), arcname=f)
    ok &= post("-C%%03dof%%03d" %% (i, len(chunks)),
               base64.b64encode(buf.getvalue()),
               {"X-Chunk": "%%d/%%d" %% (i, len(chunks))})
ok &= post("-MANIFEST", base64.b64encode(gzip.compress(
    json.dumps(man).encode())))
print("POSTED %%d files in %%d chunks ok=%%s" %% (len(files), len(chunks), ok),
      flush=True)
sys.exit(0 if ok else 4)
'''


# ----------------------------------------------------------------------
# Backlog enumeration + payload build
# ----------------------------------------------------------------------
def _task_stem(uid, N, rep, mode, method):
    return "%s__N%02d_r%d_%s__%s" % (uid, N, rep, mode, method)


def _full_task_set():
    """The full E5 FTP matrix: subset stars x N x reps x random, plus the
    N=12 rep=0 linspace-thinning cells, x {ftp_1band, ftp_mb}."""
    uids = [r["uid"] for r in
            json.load(open(os.path.join(E5, "e5_star_subset.json")))["stars"]]
    full = []
    for uid in uids:
        for m in FTP_METHODS:
            for N in N_GRID:
                for rep in REPS:
                    full.append([uid, N, rep, "random", m])
            full.append([uid, 12, 0, "thin", m])
    return full


def _remaining():
    """Full set minus tasks with a FULL local .npz (.part ignored: the local
    runner owns those; recompute duplication is harmless)."""
    return [t for t in _full_task_set()
            if not os.path.exists(os.path.join(OUTDIR,
                                               _task_stem(*t) + ".npz"))]


def _star_table():
    return {s["uid"]: s for s in
            json.load(open(os.path.join(E4, "star_table.json")))["stars"]}


def _shard(tasks, n_shards):
    """Greedy balanced partition by grid size (task cost ~ n_freq)."""
    by_uid = _star_table()

    def cost(t):
        s = by_uid[t[0]]
        g = s["grid"][s["primary_band"]] if t[4] == "ftp_1band" \
            else s["grid"]["joint"]
        return g["n_freq"]

    bins = [{"tasks": [], "cost": 0} for _ in range(n_shards)]
    for t in sorted(tasks, key=cost, reverse=True):
        b = min(bins, key=lambda b: b["cost"])
        b["tasks"].append(t)
        b["cost"] += cost(t)
    return [b["tasks"] for b in bins if b["tasks"]]


def build():
    """Enumerate the backlog NOW, shard, and build _fleet_e5/payload.tgz
    from the LOCAL working tree (code parity with the local runner)."""
    os.makedirs(FLEET, exist_ok=True)
    remaining = _remaining()
    shards = _shard(remaining, N_PODS)
    print("backlog: %d tasks -> %d shards (%s)"
          % (len(remaining), len(shards),
             ", ".join(str(len(s)) for s in shards)))
    by_uid = _star_table()
    uids = sorted({t[0] for t in _full_task_set()})

    payload = os.path.join(FLEET, "payload.tgz")
    with tarfile.open(payload, "w:gz") as tf:
        def add_file(src, arc):
            tf.add(src, arcname="e5payload/" + arc, recursive=False)

        for root, dirs, files in os.walk(os.path.join(REPO, "ftperiodogram")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    src = os.path.join(root, f)
                    add_file(src, "repo/" + os.path.relpath(src, REPO))
        for src, arc in [
                (os.path.join(E4, "run_e4.py"),
                 "repo/experiments/phase4_demo/e4_recovery/run_e4.py"),
                (os.path.join(E4, "star_table.json"),
                 "repo/experiments/phase4_demo/e4_recovery/star_table.json"),
                (os.path.join(E5, "run_e5.py"),
                 "repo/experiments/phase4_demo/e5_sparse/run_e5.py"),
                (os.path.join(E5, "e5_star_subset.json"),
                 "repo/experiments/phase4_demo/e5_sparse/e5_star_subset.json"),
                (os.path.join(DATA_HOME, "RRLyr_ugriz_templates.tar.gz"),
                 "data/RRLyr_ugriz_templates.tar.gz")]:
            add_file(src, arc)
        for uid in uids:
            lc = by_uid[uid]["lc_file"]
            add_file(os.path.join(DATA_HOME, "phase4_rrl_sample", lc),
                     "data/phase4_rrl_sample/" + lc)

        def add_text(text, arc):
            data = text.encode()
            ti = tarfile.TarInfo("e5payload/" + arc)
            ti.size = len(data)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(data))

        add_text(SHARD_DRIVER,
                 "repo/experiments/phase4_demo/e5_sparse/shard_driver.py")
        add_text(POST_RESULTS % {"chunk": CHUNK_BYTES},
                 "repo/experiments/phase4_demo/e5_sparse/post_results.py")
        for k, s in enumerate(shards):
            add_text(json.dumps(s), "shards/shard_%d.json" % k)

    sha = hashlib.sha256(open(payload, "rb").read()).hexdigest()
    st = _load_state() if os.path.exists(STATE) else {}
    st.update({"payload_sha256": sha, "n_tasks_shipped": len(remaining),
               "shard_sizes": [len(s) for s in shards],
               "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "jobs": st.get("jobs") or [
                   {"tag": "e5-%d" % k, "shard": k, "n_tasks": len(s),
                    "token": None, "pod": None, "info": None, "created": None,
                    "started": False, "received": False, "status": None,
                    "boot_attempts": 0, "avoid_flavors": [],
                    "next_create_after": 0, "deleted_at": None,
                    "core_hr": 0.0}
                   for k, s in enumerate(shards)]})
    _save_state(st)
    print("payload %s (%.2f MB) sha256 %s"
          % (payload, os.path.getsize(payload) / 1e6, sha[:16]))


def stage():
    """Upload payload.tgz somewhere the pods can curl (0x0.st, fallback
    litterbox 24h).  Stores the URL in state."""
    payload = os.path.join(FLEET, "payload.tgz")
    st = _load_state()
    url = None
    for cmd in (
            ["curl", "-sf", "-F", "file=@%s" % payload,
             "-A", "ftp-e5-fleet/1.0", "https://0x0.st"],
            ["curl", "-sf", "-F", "reqtype=fileupload", "-F", "time=24h",
             "-F", "fileToUpload=@%s" % payload,
             "https://litterbox.catbox.moe/resources/internals/api.php"]):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        cand = p.stdout.strip()
        if p.returncode == 0 and cand.startswith("http"):
            url = cand
            break
        print("stage attempt failed: rc=%d %s" % (p.returncode,
                                                  (p.stderr or cand)[:200]))
    if not url:
        raise SystemExit("staging failed on all hosts")
    # verify the staged bytes round-trip
    body = urllib.request.urlopen(url, timeout=120).read()
    got = hashlib.sha256(body).hexdigest()
    if got != st["payload_sha256"]:
        raise SystemExit("staged sha mismatch: %s != %s"
                         % (got[:16], st["payload_sha256"][:16]))
    st["payload_url"] = url
    _save_state(st)
    print("staged %s (sha verified)" % url)


# ----------------------------------------------------------------------
# Numerical-parity gate (venv, BEFORE any pod exists)
# ----------------------------------------------------------------------
def gate():
    """Recompute one already-completed local task per FTP method inside a
    python-3.11 venv with the pod's exact pins (payload tree, fake HOME with
    the shipped data -- the pod bootstrap environment as exactly as this Mac
    can host it), then require identical subsample indices and
    max|stat - stat_local| <= 1e-10.  Writes gate verdict into state;
    launch refuses without it."""
    st = _load_state()
    sim = os.path.join(FLEET, "podsim")
    subprocess.run(["rm", "-rf", sim], check=True)
    os.makedirs(sim)
    with tarfile.open(os.path.join(FLEET, "payload.tgz")) as tf:
        tf.extractall(sim)
    root = os.path.join(sim, "e5payload")
    home = os.path.join(sim, "home")
    os.makedirs(os.path.join(home, ".ftperiodogram_data"))
    subprocess.run(["cp", "-r", os.path.join(root, "data",
                                             "phase4_rrl_sample"),
                    os.path.join(home, ".ftperiodogram_data/")], check=True)
    subprocess.run(["cp", os.path.join(root, "data",
                                       "RRLyr_ugriz_templates.tar.gz"),
                    os.path.join(home, ".ftperiodogram_data/")], check=True)
    venv = os.path.join(FLEET, "venv311")
    if not os.path.exists(os.path.join(venv, "bin", "python")):
        subprocess.run(["uv", "venv", "--python", "3.11", venv], check=True)
        subprocess.run(["uv", "pip", "install", "--python",
                        os.path.join(venv, "bin", "python")] + PINS.split(),
                       check=True)

    # pick one DONE local task per method (prefer the smallest grid)
    by_uid = _star_table()
    done = [t for t in _full_task_set()
            if os.path.exists(os.path.join(OUTDIR, _task_stem(*t) + ".npz"))]
    picks = []
    for m in FTP_METHODS:
        cand = [t for t in done if t[4] == m]
        if not cand:
            raise SystemExit("gate: no completed local %s task to check" % m)
        picks.append(min(cand, key=lambda t: by_uid[t[0]]["grid"]
                         ["joint"]["n_freq"]))
    e5dir = os.path.join(root, "repo", "experiments", "phase4_demo",
                         "e5_sparse")
    json.dump(picks, open(os.path.join(e5dir, "gate_tasks.json"), "w"))
    env = dict(os.environ, HOME=home,
               PYTHONPATH=os.path.join(root, "repo"))
    t0 = time.time()
    p = subprocess.run([os.path.join(venv, "bin", "python"),
                        "shard_driver.py", "gate_tasks.json", "--workers",
                        "2"], cwd=e5dir, env=env, capture_output=True,
                       text=True, timeout=3000)
    print(p.stdout[-2000:])
    if p.returncode != 0:
        raise SystemExit("gate driver failed rc=%d\n%s"
                         % (p.returncode, p.stderr[-2000:]))
    import numpy as np
    verdict = {"picks": picks, "wall_s": time.time() - t0, "cells": []}
    worst = 0.0
    for t in picks:
        stem = _task_stem(*t) + ".npz"
        a = np.load(os.path.join(e5dir, "output", stem))
        b = np.load(os.path.join(OUTDIR, stem))
        idx_ok = (np.array_equal(a["idx_g"], b["idx_g"])
                  and np.array_equal(a["idx_r"], b["idx_r"]))
        d = float(np.nanmax(np.abs(a["stat"].astype(np.float64)
                                   - b["stat"].astype(np.float64))))
        bf = (float(a["best_freq"]), float(b["best_freq"]))
        verdict["cells"].append({"stem": stem, "idx_identical": bool(idx_ok),
                                 "max_stat_diff": d, "best_freq": bf})
        print("gate %s: idx_identical=%s max|dstat|=%.3g best_freq %s"
              % (stem, idx_ok, d, bf))
        if not idx_ok:
            raise SystemExit("GATE FAIL: subsample indices differ -- "
                             "environment NOT reproducing local draws")
        worst = max(worst, d)
    verdict["max_stat_diff"] = worst
    verdict["passed"] = bool(worst <= 1e-10)
    st["gate"] = verdict
    _save_state(st)
    if not verdict["passed"]:
        raise SystemExit("GATE FAIL: max|stat diff| %.3g > 1e-10 -- STOP, "
                         "do not launch/merge (numpy/env divergence)" % worst)
    print("GATE PASS: max|stat diff| = %.3g (<= 1e-10)" % worst)


# ----------------------------------------------------------------------
# Launch / poll / merge / teardown
# ----------------------------------------------------------------------
def _bootstrap(tag, shard_k, token, url):
    diag = ("( echo boot-ok; ls /workspace; echo --pip--; "
            "tail -4 /workspace/pip.log; echo --numpy--; "
            "python -c 'import numpy;print(numpy.__version__)' 2>&1; "
            "echo --ftp--; "
            "python -c 'import ftperiodogram;print(\"ftp_ok\")' 2>&1 ) "
            ">/workspace/diag.txt 2>&1")
    fallback = ("if [ \"$PR\" != \"0\" ]; then "
                "tail -100 /workspace/run.log 2>/dev/null | gzip -c | "
                "base64 -w0 >/workspace/payload.b64; ST=postfail; %s; fi"
                % _post(token, tag, "-POSTFAIL"))
    lines = [
        "set +e",
        "mkdir -p /workspace",
        "cd /workspace",
        "for i in 1 2 3 4 5; do curl -fsSL -o payload.tgz '%s' "
        ">dl.log 2>&1 && break; sleep 20; done" % url,
        "tar xzf payload.tgz 2>tar.log",
        "pip install --break-system-packages -q %s >/workspace/pip.log 2>&1"
        % PINS,
        "mkdir -p /root/.ftperiodogram_data",
        "cp -r /workspace/e5payload/data/phase4_rrl_sample "
        "/root/.ftperiodogram_data/",
        "cp /workspace/e5payload/data/RRLyr_ugriz_templates.tar.gz "
        "/root/.ftperiodogram_data/",
        "export PYTHONPATH=/workspace/e5payload/repo",
        diag,
        "ST=diag; gzip -c /workspace/diag.txt | base64 -w0 "
        ">/workspace/payload.b64",
        _post(token, tag, "-START"),
        _prog_beacon(token, tag),
        "cd /workspace/e5payload/repo/experiments/phase4_demo/e5_sparse",
        "ST=run; { python shard_driver.py /workspace/e5payload/shards/"
        "shard_%d.json --workers 32 ; } >/workspace/run.log 2>&1; ST=$?"
        % shard_k,
        "{ python post_results.py %s %s \"$ST\" ; } >>/workspace/run.log "
        "2>&1; PR=$?" % (token, tag),
        fallback,
        "sleep 9000",
    ]
    return " ; ".join(lines)


def _create(j, st):
    """Create a pod for job j (32-vCPU flavors only; on pool exhaustion set
    a 10-min retry backoff instead of downgrading)."""
    script = _bootstrap(j["tag"], j["shard"], j["token"], st["payload_url"])
    _check_script_syntax(script, j["tag"])
    last = ""
    for flav, vcpu, cloud in FLAVORS:
        if flav + "/" + cloud in (j.get("avoid_flavors") or []):
            continue
        body = {"name": j["tag"], "computeType": "CPU",
                "cpuFlavorIds": [flav], "vcpuCount": vcpu,
                "imageName": IMAGE, "containerDiskInGb": 20,
                "ports": ["22/tcp"], "cloudType": cloud,
                "dockerStartCmd": ["bash", "-c", script]}
        t = _runpod("POST", "/pods", body)
        m = re.search(r'"id":"([a-z0-9]+)"', t)
        if m:
            j["pod"], j["info"] = m.group(1), "%s/%d/%s" % (flav, vcpu, cloud)
            j["created"] = time.time()
            j["boot_attempts"] = j.get("boot_attempts", 0) + 1
            print("launch %s -> %s (%s) [attempt %d/%d]"
                  % (j["tag"], j["pod"], j["info"], j["boot_attempts"],
                     MAX_BOOT_ATTEMPTS))
            return True
        last = t[:140]
    j["next_create_after"] = time.time() + 600
    print("launch %s: no 32-vCPU capacity (%s) -- retry in 10 min"
          % (j["tag"], last))
    return False


def launch():
    st = _load_state()
    if not (st.get("gate") or {}).get("passed"):
        raise SystemExit("REFUSING to launch: parity gate not passed")
    if not st.get("payload_url"):
        raise SystemExit("REFUSING to launch: payload not staged")
    for j in st["jobs"]:
        if j.get("pod") or j["received"]:
            continue
        if not j.get("token"):
            j["token"] = _mint_token()
            print("%s token %s" % (j["tag"], j["token"]))
        _create(j, st)
        _save_state(st)
        time.sleep(2)


def _merge_ready(st, j):
    """Extract + verify + merge one finished shard into the LOCAL output/.
    No-clobber: hardlink tmp -> final fails if final exists (local won).
    Cross-checks any pod/local duplicate at float32 tolerance."""
    import numpy as np
    inc = os.path.join(FLEET, "incoming", j["tag"])
    man = json.load(open(os.path.join(inc, "manifest.json")))
    merged = skipped = 0
    problems = st.setdefault("problems", [])
    for name, meta in sorted(man["files"].items()):
        src = os.path.join(inc, name)
        if not os.path.exists(src):
            problems.append("%s: %s in manifest but chunk missing"
                            % (j["tag"], name))
            continue
        blob = open(src, "rb").read()
        if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
            problems.append("%s: %s sha256 mismatch -- NOT merged"
                            % (j["tag"], name))
            continue
        if not re.match(r"^[\w+.-]+__N\d{2}_r\d_(random|thin)__"
                        r"(ftp_1band|ftp_mb)\.npz$", name):
            problems.append("%s: unexpected filename %s -- NOT merged"
                            % (j["tag"], name))
            continue
        final = os.path.join(OUTDIR, name)
        if os.path.exists(final):
            skipped += 1
            if st.get("xchecked", 0) < 24:   # pod-vs-local duplicate parity
                a, b = np.load(src), np.load(final)
                same_idx = (np.array_equal(a["idx_g"], b["idx_g"]) and
                            np.array_equal(a["idx_r"], b["idx_r"]))
                d = float(np.nanmax(np.abs(
                    a["stat"].astype(np.float64)
                    - b["stat"].astype(np.float64))))
                st["xchecked"] = st.get("xchecked", 0) + 1
                st.setdefault("xcheck_worst", 0.0)
                st["xcheck_worst"] = max(st["xcheck_worst"], d)
                if not same_idx or d > 1e-6:
                    problems.append("XCHECK FAIL %s: idx=%s max|dstat|=%.3g"
                                    % (name, same_idx, d))
                    raise SystemExit("POD/LOCAL PARITY FAILURE on %s "
                                     "(max|dstat|=%.3g) -- merging STOPPED"
                                     % (name, d))
            continue
        tmp = os.path.join(OUTDIR, ".e5m_%s.tmp" % name)
        with open(tmp, "wb") as fh:
            fh.write(blob)
        try:
            os.link(tmp, final)          # atomic no-clobber
            merged += 1
        except FileExistsError:
            skipped += 1                 # local won the race
        finally:
            os.unlink(tmp)
    j["merged"], j["skipped_local_won"] = merged, skipped
    j["fails_reported"] = man.get("fails", [])
    if man.get("fails"):
        problems.append("%s reported .fail files: %s"
                        % (j["tag"], man["fails"]))
    print("%s: merged %d, skipped (local won) %d, manifest %d files"
          % (j["tag"], merged, skipped, len(man["files"])))


def _poll_once(st):
    now = time.time()
    all_done = True
    for j in st["jobs"]:
        if j["received"]:
            continue
        all_done = False
        if not j.get("token"):
            continue
        try:
            reqs = _requests(j["token"])
        except Exception as e:
            print("%s: beacon fetch failed (%s)" % (j["tag"], e))
            continue
        inc = os.path.join(FLEET, "incoming", j["tag"])
        os.makedirs(inc, exist_ok=True)
        manifest = None
        n_chunks = got_chunks = 0
        for r in reqs:
            job = _headers(r).get("x-job", "")
            if job == j["tag"] + "-START" and not j["started"]:
                j["started"] = True
                print("%s: START (boot OK)" % j["tag"])
            m = re.match(re.escape(j["tag"]) + r"-C(\d+)of(\d+)$", job)
            if m:
                n_chunks = int(m.group(2))
                p = os.path.join(inc, "chunk_%03d.tar" % int(m.group(1)))
                if not os.path.exists(p):
                    with open(p, "wb") as fh:
                        fh.write(base64.b64decode(r["content"]))
            elif job == j["tag"] + "-MANIFEST":
                manifest = r
            elif job == j["tag"] + "-POSTFAIL":
                st.setdefault("problems", []).append(
                    "%s POSTFAIL beacon (post_results crashed)" % j["tag"])
        got_chunks = len([f for f in os.listdir(inc)
                          if f.startswith("chunk_")])
        if manifest is not None and j.get("manifest_seen_at") is None:
            j["manifest_seen_at"] = now
        # normally every chunk is posted before the manifest; the 5-min
        # grace merges a shard with a permanently lost chunk (recorded as a
        # problem; the local runner recomputes those tasks anyway)
        if manifest is not None and (
                got_chunks >= n_chunks
                or now - j["manifest_seen_at"] > 300):
            with open(os.path.join(inc, "manifest.json"), "wb") as fh:
                fh.write(gzip.decompress(base64.b64decode(
                    manifest["content"])))
            for f in sorted(os.listdir(inc)):
                if f.startswith("chunk_"):
                    with tarfile.open(os.path.join(inc, f)) as tf:
                        for mem in tf.getmembers():
                            if mem.isfile() and "/" not in mem.name and \
                                    not mem.name.startswith("."):
                                tf.extract(mem, inc)
            _merge_ready(st, j)
            j["received"] = True
            j["status"] = _headers(manifest).get("x-status", "?")
            if j.get("pod"):
                dur = (now - (j["created"] or now)) / 3600.0
                j["core_hr"] += dur * 32
                _delete_pod(j["pod"])
                j["deleted_at"] = now
                print("%s: pod %s deleted after %.1f min"
                      % (j["tag"], j["pod"], dur * 60))
                j["pod"] = None
            continue
        # boot watchdog + stale kill + capacity retry
        if j.get("pod"):
            age_min = (now - (j["created"] or now)) / 60.0
            if not j["started"] and age_min > BOOT_LIMIT_MIN:
                print("%s: NO START after %.0f min -- deleting pod %s"
                      % (j["tag"], age_min, j["pod"]))
                j["core_hr"] += age_min / 60.0 * 32
                _delete_pod(j["pod"])
                flav = (j.get("info") or "").split("/")
                if len(flav) == 3:
                    j.setdefault("avoid_flavors", []).append(
                        flav[0] + "/" + flav[2])
                j["pod"], j["created"] = None, None
                if j["boot_attempts"] >= MAX_BOOT_ATTEMPTS:
                    j["boot_dead"] = True
                    st.setdefault("problems", []).append(
                        "%s boot-dead after %d attempts"
                        % (j["tag"], MAX_BOOT_ATTEMPTS))
                else:
                    _create(j, st)
            elif age_min > STALE_HOURS * 60:
                print("%s: STALE %.1f h -- killing pod %s"
                      % (j["tag"], age_min / 60, j["pod"]))
                j["core_hr"] += age_min / 60.0 * 32
                _delete_pod(j["pod"])
                j["pod"], j["killed"] = None, True
                st.setdefault("problems", []).append(
                    "%s killed stale at %.1f h" % (j["tag"], age_min / 60))
        elif (not j.get("boot_dead") and not j.get("killed")
              and now > j.get("next_create_after", 0)
              and st.get("payload_url")):
            _create(j, st)
    return all_done


def poll():
    """Foreground poll loop (default one pass; `poll 420` loops ~7 min)."""
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    t0 = time.time()
    while True:
        st = _load_state()
        done = _poll_once(st)
        _save_state(st)
        n = len(st["jobs"])
        print("[%s] received %d/%d | started %d/%d"
              % (time.strftime("%H:%M:%S"),
                 sum(1 for j in st["jobs"] if j["received"]), n,
                 sum(1 for j in st["jobs"] if j["started"]), n), flush=True)
        if done:
            print("ALL_SHARDS_RECEIVED")
            return
        if time.time() - t0 + 45 > budget:
            return
        time.sleep(45)


def status():
    st = _load_state()
    for j in st["jobs"]:
        print("%-6s pod=%-14s started=%-5s received=%-5s merged=%s "
              "skipped=%s status=%s" %
              (j["tag"], j.get("pod"), j["started"], j["received"],
               j.get("merged"), j.get("skipped_local_won"), j.get("status")))


def teardown():
    """Delete every pod THIS state launched; verify none of ours remain
    (cuvarbase-dev and any foreign [e.g. b8c] pods are never touched)."""
    st = _load_state()
    now = time.time()
    for j in st["jobs"]:
        if j.get("pod"):
            j["core_hr"] += (now - (j["created"] or now)) / 3600.0 * 32
            _delete_pod(j["pod"])
            print("teardown %s: DELETE pod %s" % (j["tag"], j["pod"]))
            j["pod"] = None
    _save_state(st)
    time.sleep(5)
    mine = {"e5-%d" % k for k in range(len(st["jobs"]))}
    survivors = _live_pods()
    print("live pods after teardown:")
    leftovers = []
    for pid, name in survivors:
        tagged = (" [protected cuvarbase-dev]" if _is_protected(pid, name)
                  else (" [FOREIGN -- not ours, untouched]"
                        if name not in mine else " [OURS -- LEAK]"))
        print("  %s  %s%s" % (pid, name, tagged))
        if name in mine:
            leftovers.append(pid)
    if leftovers:
        for pid in leftovers:
            _delete_pod(pid)
        print("deleted %d leftover own pods by name" % len(leftovers))
        sys.exit(1)
    print("OK: zero e5 pods remain")


def report():
    st = _load_state()
    out = {
        "tasks_shipped": st.get("n_tasks_shipped"),
        "tasks_merged": sum(j.get("merged") or 0 for j in st["jobs"]),
        "tasks_skipped_local_won": sum(j.get("skipped_local_won") or 0
                                       for j in st["jobs"]),
        "pods_used": sum(j.get("boot_attempts", 0) for j in st["jobs"]),
        "total_cost_usd": round(sum(j.get("core_hr", 0.0)
                                    for j in st["jobs"]) * RATE, 2),
        "gate": {"passed": (st.get("gate") or {}).get("passed"),
                 "max_stat_diff": (st.get("gate") or {}).get(
                     "max_stat_diff")},
        "xcheck": {"n": st.get("xchecked", 0),
                   "worst": st.get("xcheck_worst")},
        "problems": st.get("problems", []),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    # NO account-wide 'cleanup' on purpose (cuvarbase-dev + b8c protection).
    {"build": build, "stage": stage, "gate": gate, "launch": launch,
     "poll": poll, "status": status, "teardown": teardown,
     "report": report}[sys.argv[1]]()
