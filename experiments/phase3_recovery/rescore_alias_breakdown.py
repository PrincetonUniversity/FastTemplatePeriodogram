#!/usr/bin/env python
"""Post-hoc alias-aware re-scoring of persisted per-source recovery results (WP B5).

Reads the ``per_source/*.npz`` files written by run_production_matrix.py (WP B1:
one ``(P_true, P_rec, recovered)`` array set per (seed, method, cell)), pools
seeds, re-classifies every source under ``--criterion``, and emits an
alias-breakdown table per (method, cell) -- in particular per (method, N) for the
N-sweep, the sparse-regime story.

Why: the headline fractional criterion (rtol=0.01) counts a +/-1/yr window beat
(|df| = 1/365.25 c/d) as EXACT for f_true >~ 0.28 c/d -- the exact test runs
first, so the in-run alias scan cannot surface year beats -- while 1-day beats
and P/2 fail the same tolerance (asymmetric).  Phase coherence (|df|*T <
delta_phi_max, the default here) separates them.  Per-source baselines come from
the npz (``baseline`` field, runs after WP B5) or ``--baseline-days`` /
results.json config for legacy runs.

    python rescore_alias_breakdown.py --outdir output_prod
    python rescore_alias_breakdown.py --outdir output_prod --criterion fractional
"""
import argparse
import glob
import json
import os

import numpy as np

from ftperiodogram.recovery import rescore_aliases


def _config_baseline_days(outdir):
    path = os.path.join(outdir, 'results.json')
    try:
        with open(path) as fh:
            return float(json.load(fh)['config']['baseline_days'])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def load_groups(outdir, baseline_days, log):
    """Pool per-source npz files across seeds -> {(method, cell): arrays}."""
    files = sorted(glob.glob(os.path.join(outdir, 'per_source', '*.npz')))
    if not files:
        raise SystemExit("no per_source/*.npz under %r (WP B1 persistence "
                         "required)" % outdir)
    groups = {}
    n_legacy = 0
    for path in files:
        with np.load(path) as arr:
            key = (str(arr['method']), str(arr['cell']))
            if 'baseline' in arr.files:
                baseline = np.asarray(arr['baseline'], dtype=float)
            else:
                n_legacy += 1
                if baseline_days is None:
                    raise SystemExit(
                        "%s lacks per-source baselines (pre-B5 run) and no "
                        "--baseline-days / results.json fallback is available"
                        % os.path.basename(path))
                baseline = np.full(arr['p_rec'].shape, baseline_days)
            g = groups.setdefault(key, {'p_rec': [], 'p_true': [],
                                        'baseline': []})
            g['p_rec'].append(np.asarray(arr['p_rec'], dtype=float))
            g['p_true'].append(np.asarray(arr['p_true'], dtype=float))
            g['baseline'].append(baseline)
    if n_legacy:
        log("NOTE %d legacy npz lack per-source baselines; using "
            "baseline_days=%.1f for those (downsampling freezes the baseline, "
            "so this is the nominal span, not the realized t.max()-t.min())"
            % (n_legacy, baseline_days))
    return {k: {f: np.concatenate(v[f]) for f in v} for k, v in groups.items()}


def print_table(rows, log):
    """Aligned text table; one row per (method, cell), pooled over seeds."""
    names = ['exact']
    for r in rows:
        for k in r['breakdown']:
            if k not in names and k not in ('miss', 'n'):
                names.append(k)
    names += ['miss', 'n']
    head = ['method', 'cell'] + names
    table = [head] + [
        [r['method'], r['cell']] + [str(r['breakdown'].get(k, 0))
                                    for k in names] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    for row in table:
        log('  '.join(c.ljust(w) for c, w in zip(row, widths)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outdir', required=True,
                   help="run directory containing per_source/ and results.json")
    p.add_argument('--criterion', choices=['phase_coherence', 'fractional'],
                   default='phase_coherence',
                   help="re-scoring criterion (default phase_coherence -- the "
                        "one that can surface year beats)")
    p.add_argument('--rtol', type=float, default=0.01)
    p.add_argument('--delta-phi-max', type=float, default=0.5)
    p.add_argument('--baseline-days', type=float, default=None,
                   help="fallback per-source baseline for legacy npz without a "
                        "'baseline' field (default: results.json config)")
    args = p.parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    baseline_days = (args.baseline_days
                     if args.baseline_days is not None
                     else _config_baseline_days(args.outdir))
    groups = load_groups(args.outdir, baseline_days, log)

    rows = []
    for (method, cell) in sorted(groups):
        g = groups[(method, cell)]
        _, breakdown = rescore_aliases(
            g['p_rec'], g['p_true'], baseline=g['baseline'],
            criterion=args.criterion, rtol=args.rtol,
            delta_phi_max=args.delta_phi_max)
        rows.append({'method': method, 'cell': cell, 'breakdown': breakdown})

    out = {'criterion': args.criterion, 'rtol': args.rtol,
           'delta_phi_max': args.delta_phi_max, 'rows': rows}
    out_path = os.path.join(args.outdir,
                            'alias_breakdown_%s.json' % args.criterion)
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    log("alias breakdown (criterion=%s, pooled over seeds):" % args.criterion)
    print_table(rows, log)
    log("wrote %s" % out_path)
    return out


if __name__ == '__main__':
    main()
