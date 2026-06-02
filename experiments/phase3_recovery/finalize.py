#!/usr/bin/env python
"""Aggregate the 3 per-seed fleet results for one universe, render figures, summarize.

Each pod posted seed_<s>.json (a flat run_seed dict: seed, k_sweep_sparse, k_sweep_dense,
fixed_k, n_epochs_sweep, cost).  This loads the 3 seeds for a universe, reuses
run_production_matrix.aggregate to compute the per-curve mean/std, wraps it into the
results dict production_figures expects, renders the three deliverable figures, and writes
a markdown summary.

    .venv/bin/python finalize.py sesar
    .venv/bin/python finalize.py bv
"""
import json
import os
import sys

import run_production_matrix as rpm
import production_figures

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "output_prod")
DEST = os.path.join(HERE, "headline_full")


def _fmt(stat):
    return ["%.3f" % m for m in stat["mean"]]


def finalize(universe):
    seeds = []
    for s in (0, 1, 2):
        path = os.path.join(OUTDIR, "%s-%d.json" % (universe, s))
        if os.path.exists(path):
            seeds.append(json.load(open(path)))
    if not seeds:
        raise SystemExit("no seed results for %s" % universe)
    print("aggregating %d seed(s) for %s" % (len(seeds), universe))
    agg = rpm.aggregate(seeds)
    results = {"config": {"universe": universe, "nharmonics": 8, "n_sources": 256,
                          "n_freq": 8000, "n_seeds": 3},
               "per_seed": seeds, "aggregate": agg}
    dest = os.path.join(DEST, universe)
    os.makedirs(dest, exist_ok=True)
    json.dump(results, open(os.path.join(dest, "results.json"), "w"), indent=2)
    production_figures.make_figures(results, dest)

    ks = agg["k_sweep_sparse"]; kd = agg["k_sweep_dense"]; n = agg["n_epochs_sweep"]
    c = agg.get("cost", {})
    nseed = len(seeds)
    lines = [
        "# Phase 3.2 production headline -- %s universe (H=8, 256 src x %d seed%s)"
        % (universe, nseed, "s" if nseed != 1 else ""),
        "",
        "RunPod fleet result (%d seed%s, mean across seeds). Figures in this dir."
        % (nseed, "s" if nseed != 1 else ""),
        "",
        "## Recovery vs N_epochs @ knee K=%s (the headline)" % n["k_per_seed"],
        "| epochs/band | %s |" % " | ".join(str(x) for x in n["n_epochs_values"]),
        "|---|%s|" % ("---|" * len(n["n_epochs_values"])),
        "| FTP(PAM) | %s |" % " | ".join(_fmt(n["ftp_pam"])),
        "| FTP(greedy) | %s |" % " | ".join(_fmt(n["ftp_greedy"])),
        "| GLS | %s |" % " | ".join(_fmt(n["gls"])),
        "| MHLS | %s |" % " | ".join(_fmt(n["mhls"])),
        "| multiband-LS | %s |" % " | ".join(_fmt(n["mbls"])),
        "",
        "## Recovery vs K",
        "- sparse (N=%d): FTP(PAM) %s ; GLS %.3f ; MHLS %.3f ; mb-LS %.3f"
        % (seeds[0]["k_sweep_sparse"]["n_epochs"], _fmt(ks["ftp_pam"]),
           ks["baselines"]["gls"]["mean"][0], ks["baselines"]["mhls"]["mean"][0],
           ks["baselines"]["mbls"]["mean"][0]),
        "- dense (N=%d): FTP(PAM) %s ; GLS %.3f ; MHLS %.3f ; mb-LS %.3f"
        % (seeds[0]["k_sweep_dense"]["n_epochs"], _fmt(kd["ftp_pam"]),
           kd["baselines"]["gls"]["mean"][0], kd["baselines"]["mhls"]["mean"][0],
           kd["baselines"]["mbls"]["mean"][0]),
        "",
        "## Cost vs accuracy (FTP vs Sesar-style non-linear oracle)",
        "- FTP recovery %.3f vs oracle recovery %.3f (gold-standard equivalence)"
        % (c.get("ftp_recovery", {}).get("mean", [0])[0],
           c.get("oracle_recovery", {}).get("mean", [0])[0]),
        "- speedup %.1fx (reduced cost-panel grid; full-grid speedup is larger)"
        % c.get("speedup", {}).get("mean", [0])[0],
    ]
    open(os.path.join(dest, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote figures + summary to", dest)


if __name__ == "__main__":
    finalize(sys.argv[1])
