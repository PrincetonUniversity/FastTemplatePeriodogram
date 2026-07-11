#!/usr/bin/env python
"""Phase 3.4 headline: the joint-minus-pipeline recovery gap vs N_epochs.

Both arms produce a size-``K`` template vocabulary and are scored *identically*
(FTP over the vocabulary on a held-out eval population), so the only variable is
how the vocabulary was built:

* **pipeline** -- :func:`build_template_catalog` (k-medoids over the clean library
  shapes), N-independent.
* **joint** -- :func:`build_joint_em_catalog` (the same k-medoids init, then EM
  refinement against a *training* population observed at the same N_epochs).

Three populations are drawn from the same truth shapes + cadence with disjoint
seeds (train / val / eval) so the joint arm never sees its eval data: it trains on
``train``, early-stops on ``val``, and is scored on ``eval`` -- the honest test.
MRA theory predicts the gap is largest when data are sparse and vanishes when
dense; this driver measures that curve.

Local-draft scale by default; scale up (``--n-sources``, ``--n-freq``, more seeds)
on a many-core box for the publication figure.  Search band is an explicit
``[f_min, f_max]`` -- never a Nyquist heuristic.
"""
import argparse
import json
import os
import time

import numpy as np

from ftperiodogram.template import Template
from ftperiodogram.simulate import SyntheticCadence, exp_mag_error
from ftperiodogram.validation import make_recovery_scorer, frequency_grid
from ftperiodogram.catalog_builder import (build_template_catalog,
                                           fetch_sesar_templates,
                                           fetch_baeza_villagra_templates)
from ftperiodogram.joint_em import build_joint_em_catalog
import ztf_error_model as zerr             # sibling experiment module


def _load_truth(universe, n_harmonics, bands):
    if universe == 'sesar':
        return fetch_sesar_templates(nharmonics=n_harmonics, bands=bands)
    if universe == 'bv':
        return fetch_baeza_villagra_templates(nharmonics=n_harmonics, bands=bands)
    raise ValueError("universe must be 'sesar' or 'bv'; got %r" % (universe,))


def run(args):
    bands = list(args.bands)
    truth = _load_truth(args.universe, args.n_harmonics, bands)
    # Library holdout: reserve a disjoint fraction of shapes as the POPULATION
    # (injected truth) so the library both arms cluster does NOT contain the
    # generating shapes -- the realistic mismatch that gives joint legitimate
    # headroom to adapt. With holdout=0 population and library are the same set.
    if args.library_holdout_frac > 0:
        idx = np.random.RandomState(args.seed).permutation(len(truth))
        n_pop = max(1, int(round(args.library_holdout_frac * len(truth))))
        population = [truth[i] for i in idx[:n_pop]]
        library = [truth[i] for i in idx[n_pop:]]
    else:
        population = library = truth
    print("universe=%s: %d shapes (H=%d), bands=%s | population=%d, library=%d"
          % (args.universe, len(truth), args.n_harmonics, bands,
             len(population), len(library)))
    freqs = frequency_grid(args.f_min, args.f_max, args.n_freq)
    n_master = max(args.n_epochs_values)
    err_model = None
    if args.err_model == 'empirical':
        try:
            err_model = zerr.make_empirical_error_model()
            print("  empirical ZTF error model (faint sigma=%.3f)"
                  % err_model.faint_sigma)
        except (FileNotFoundError, ValueError) as exc:
            print("  WARNING empirical error model unavailable (%s); synthetic"
                  % str(exc)[:70])
    cad = SyntheticCadence(n_epochs={b: n_master for b in bands}, bands=bands,
                           baseline_days=args.baseline_days, err_model=err_model,
                           random_state=args.seed)
    # The grid must resolve periodogram peaks (width ~ 1/T): require df << 1/T.
    df = (args.f_max - args.f_min) / (args.n_freq - 1)
    if df > 0.5 / args.baseline_days:
        print("WARNING: df=%.2g exceeds 0.5/T=%.2g; the grid undersamples peaks "
              "(raise --n-freq or shorten --baseline-days)"
              % (df, 0.5 / args.baseline_days))

    def master(seed):
        return make_recovery_scorer(cad, population, freqs=freqs,
                                    n_sources=args.n_sources, mean_mag=args.mean_mag,
                                    amplitude=args.amplitude,
                                    intrinsic_jitter=args.intrinsic_jitter,
                                    random_state=seed, n_jobs=args.n_jobs)

    t_wall0 = time.perf_counter()
    t0 = time.perf_counter()
    train_m = master(args.seed + 11)
    val_m = master(args.seed + 22)
    eval_m = master(args.seed + 33)
    t_sim = time.perf_counter() - t0

    # pipeline vocabulary is N-independent -- build once (from the library).
    pipeline_vocab = build_template_catalog(library, args.k, method='pam',
                                            random_state=args.seed,
                                            n_harmonics=args.n_harmonics)
    gls = [Template([1.0], [0.0])]

    # per-phase wall-second totals (B8-closure P3): the canary acceptance check
    # is that the assignment share collapses from ~87% (serial loop, B8E-2) to
    # <~10% once the parallel E-step lands (validation.assignment_accuracy).
    phase_totals = dict(sim=t_sim, em=0.0, eval_pipe=0.0, eval_joint=0.0,
                        gls=0.0, assign_pipe=0.0, assign_joint=0.0)
    rows = []
    for N in sorted(args.n_epochs_values):
        train_N = train_m.downsample(N, random_state=args.seed)
        val_N = val_m.downsample(N, random_state=args.seed)
        eval_N = eval_m.downsample(N, random_state=args.seed)

        t0 = time.perf_counter()
        joint_vocab, diag = build_joint_em_catalog(
            library, args.k, train_N, val_scorer=val_N, max_iter=args.max_iter,
            n_harmonics=args.n_harmonics, random_state=args.seed,
            n_jobs=args.n_jobs, return_diagnostics=True)
        t_em = time.perf_counter() - t0

        t0 = time.perf_counter()
        pipe_rec = float(eval_N(pipeline_vocab))
        t_eval_pipe = time.perf_counter() - t0
        t0 = time.perf_counter()
        joint_rec = float(eval_N(joint_vocab))
        t_eval_joint = time.perf_counter() - t0
        t0 = time.perf_counter()
        gls_rec = float(eval_N(gls))
        t_gls = time.perf_counter() - t0
        # mechanism metric: correct-template assignment on the correct-period subset
        # (period recovery clips shape gains; this exposes them if joint learns
        # better-matched templates).  Quoted with n_subset and the MAJORITY-TARGET
        # null (always predicting the subset's most common correct target) -- the
        # honest chance bar; 1/K understates it because targets are not uniform.
        t0 = time.perf_counter()
        pipe_assign, _, pipe_n_sub, pipe_null = eval_N.assignment_accuracy(
            pipeline_vocab, return_counts=True)
        t_assign_pipe = time.perf_counter() - t0
        t0 = time.perf_counter()
        joint_assign, _, joint_n_sub, joint_null = eval_N.assignment_accuracy(
            joint_vocab, return_counts=True)
        t_assign_joint = time.perf_counter() - t0
        timers = dict(t_em=round(t_em, 3), t_eval_pipe=round(t_eval_pipe, 3),
                      t_eval_joint=round(t_eval_joint, 3), t_gls=round(t_gls, 3),
                      t_assign_pipe=round(t_assign_pipe, 3),
                      t_assign_joint=round(t_assign_joint, 3))
        for key, tv in (('em', t_em), ('eval_pipe', t_eval_pipe),
                        ('eval_joint', t_eval_joint), ('gls', t_gls),
                        ('assign_pipe', t_assign_pipe),
                        ('assign_joint', t_assign_joint)):
            phase_totals[key] += tv
        rows.append(dict(n_epochs=int(N), pipeline=pipe_rec, joint=joint_rec,
                         gls=gls_rec, gap=joint_rec - pipe_rec,
                         pipeline_assign=pipe_assign, joint_assign=joint_assign,
                         mechanism_gap=joint_assign - pipe_assign,
                         pipeline_assign_n=int(pipe_n_sub),
                         pipeline_assign_null=float(pipe_null),
                         joint_assign_n=int(joint_n_sub),
                         joint_assign_null=float(joint_null),
                         em_best_iter=int(diag.best_iter),
                         em_n_iter=int(diag.n_iter),
                         em_stop=diag.stop_reason,
                         em_val_signal=diag.val_signal_name,
                         em_val_hist=[float(v) for v in diag.val_signal],
                         em_n_gated=[int(v) for v in diag.n_gated],
                         em_n_reverted=[int(v) for v in diag.n_reverted],
                         **timers))
        print("N=%2d  rec pipe=%.3f joint=%.3f gap=%+.3f | assign pipe=%.3f "
              "joint=%.3f mech_gap=%+.3f (n=%d/%d null=%.2f/%.2f) | gls=%.3f "
              "(EM %d/%d %s)"
              % (N, pipe_rec, joint_rec, joint_rec - pipe_rec, pipe_assign,
                 joint_assign, joint_assign - pipe_assign, pipe_n_sub,
                 joint_n_sub, pipe_null, joint_null, gls_rec,
                 diag.best_iter, diag.n_iter, diag.stop_reason))
        # flushed so a wall-killed pod's run.log tail still carries the timers
        print("N=%2d  timers[s]: em=%.1f eval=%.1f/%.1f gls=%.1f "
              "assign=%.1f/%.1f (em iters=%d)"
              % (N, t_em, t_eval_pipe, t_eval_joint, t_gls, t_assign_pipe,
                 t_assign_joint, diag.n_iter), flush=True)

    os.makedirs(args.out, exist_ok=True)
    result = dict(universe=args.universe, bands=bands, k=args.k,
                  n_sources=args.n_sources, n_harmonics=args.n_harmonics,
                  freq_grid=[args.f_min, args.f_max, args.n_freq],
                  baseline_days=args.baseline_days,
                  intrinsic_jitter=args.intrinsic_jitter, mean_mag=args.mean_mag,
                  amplitude=args.amplitude,
                  library_holdout_frac=args.library_holdout_frac,
                  err_model=args.err_model,
                  grid_points_per_rayleigh=round(
                      (1.0 / args.baseline_days) /
                      ((args.f_max - args.f_min) / (args.n_freq - 1)), 3),
                  max_iter=args.max_iter, seed=args.seed,
                  phase_seconds={**{k: round(v, 3)
                                    for k, v in phase_totals.items()},
                                 'wall': round(time.perf_counter() - t_wall0, 3)},
                  rows=rows)
    with open(os.path.join(args.out, 'results.json'), 'w') as fh:
        json.dump(result, fh, indent=2)
    _write_summary(args, rows)
    if not args.no_figure:
        _write_figure(args, rows)
    print("wrote", args.out)


def _write_summary(args, rows):
    lines = ["# Phase 3.4 joint-vs-pipeline gap -- %s universe" % args.universe,
             "",
             "K=%d, %d sources, H=%d, max_iter=%d, seed=%d, bands=%s, "
             "jitter=%.3g, mean_mag=%.3g, amplitude=%.3g, T=%.0fd"
             % (args.k, args.n_sources, args.n_harmonics, args.max_iter,
                args.seed, ''.join(args.bands), args.intrinsic_jitter,
                args.mean_mag, args.amplitude, args.baseline_days),
             "",
             "| N_epochs | rec pipe | rec joint | rec gap | assign pipe | "
             "assign joint | mech gap | n_sub (p/j) | null (p/j) | GLS | "
             "EM best/n |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            "| %d | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f | %d/%d | "
            "%.2f/%.2f | %.3f | %d/%d |"
            % (r['n_epochs'], r['pipeline'], r['joint'], r['gap'],
               r.get('pipeline_assign', float('nan')),
               r.get('joint_assign', float('nan')),
               r.get('mechanism_gap', float('nan')),
               r.get('pipeline_assign_n', -1), r.get('joint_assign_n', -1),
               r.get('pipeline_assign_null', float('nan')),
               r.get('joint_assign_null', float('nan')), r['gls'],
               r.get('em_best_iter', -1), r.get('em_n_iter', -1)))
    lines += ["",
              "rec gap = period-recovery (1%) joint-minus-pipeline; mech gap = "
              "correct-template-assignment on the correct-period subset (the shape "
              "mechanism, unclipped by period recovery). n_sub = subset size; "
              "null = majority-target null on that subset (the honest chance bar, "
              "not 1/K). EM best/n = accepted best iteration / iterations run "
              "(early stop on the continuous power-margin val signal).",
              "Hypothesis: gap > 0 in the sparse regime, -> 0 as N_epochs grows."]
    with open(os.path.join(args.out, 'SUMMARY.md'), 'w') as fh:
        fh.write("\n".join(lines) + "\n")


def _write_figure(args, rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print("skipping figure (matplotlib unavailable):", exc)
        return
    N = [r['n_epochs'] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(N, [r['pipeline'] for r in rows], 'o-', label='pipeline (PAM)')
    ax1.plot(N, [r['joint'] for r in rows], 's-', label='joint (EM)')
    ax1.plot(N, [r['gls'] for r in rows], '^--', color='grey', label='GLS')
    ax1.set_xlabel('epochs / band'); ax1.set_ylabel('recovery rate')
    ax1.set_title('%s: recovery vs N_epochs' % args.universe); ax1.legend()
    ax2.axhline(0.0, color='grey', lw=0.8)
    ax2.plot(N, [r['gap'] for r in rows], 'd-', color='C3')
    ax2.set_xlabel('epochs / band'); ax2.set_ylabel('joint - pipeline')
    ax2.set_title('joint-minus-pipeline gap')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_joint_gap.pdf'))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--universe', default='sesar', choices=('sesar', 'bv'))
    p.add_argument('--bands', nargs='+', default=['g', 'r'])
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--n-sources', type=int, default=48)
    p.add_argument('--n-harmonics', type=int, default=8)
    p.add_argument('--n-epochs-values', type=int, nargs='+',
                   default=[4, 8, 12, 20, 40])
    p.add_argument('--f-min', type=float, default=1.0)
    p.add_argument('--f-max', type=float, default=5.0)
    p.add_argument('--n-freq', type=int, default=4000)
    p.add_argument('--baseline-days', type=float, default=365.0,
                   help='observing span T; the grid must satisfy df << 1/T')
    p.add_argument('--intrinsic-jitter', type=float, default=0.0,
                   help='per-source relative Fourier shape jitter; gives joint '
                        'MRA headroom (0 = degenerate same-library control)')
    p.add_argument('--mean-mag', type=float, default=15.0,
                   help='mean magnitude; NOT a low-SNR lever under the empirical '
                        'error model (sigma clips at 0.0393 past mag ~18.5) -- '
                        'lower per-epoch SNR via --amplitude (e.g. 0.1-0.15) or '
                        'sigma instead')
    p.add_argument('--amplitude', type=float, default=0.5,
                   help='signal amplitude (mag); THE low-SNR lever -- e.g. 0.1 '
                        'gives the MRA sparse/low-SNR regime (per WP A5, not '
                        'mean_mag->20)')
    p.add_argument('--library-holdout-frac', type=float, default=0.0,
                   help='fraction of shapes reserved as the population, disjoint '
                        'from the library both arms cluster (joint headroom)')
    p.add_argument('--err-model', choices=['synthetic', 'empirical'],
                   default='synthetic',
                   help='cadence noise: synthetic exp_mag_error or empirical ZTF '
                        '(1.6-3.5x larger -> lower per-epoch SNR, MRA regime)')
    p.add_argument('--max-iter', type=int, default=10)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='experiments/phase3_recovery/joint_draft')
    p.add_argument('--no-figure', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
