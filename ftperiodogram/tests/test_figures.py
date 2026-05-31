"""Tests for the optional matplotlib figure helpers (ftperiodogram.figures).

matplotlib is an optional dependency, so the whole module is skipped when it is not
installed; the ``Agg`` backend is forced before pyplot is imported so the tests run
headless.
"""
import subprocess
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from ftperiodogram import figures
from ftperiodogram.validation import KSweepResult, NEpochsSweepResult


def _k_result(with_baseline=True):
    return KSweepResult(
        k_values=np.array([1, 2, 4]), recovery=np.array([0.3, 0.6, 0.8]),
        recovery_by_k=None, baseline_recovery=(0.25 if with_baseline else None),
        baseline_mask=None, n_sources=10, freq_grid=(1.0, 5.0, 100),
        criterion='fractional', seed=0)


def _n_result():
    return NEpochsSweepResult(
        n_epochs_values=np.array([5, 10, 20]), recovery=np.array([0.4, 0.6, 0.7]),
        recovery_by_n=None, baseline_recovery=np.array([0.2, 0.3, 0.35]),
        baseline_mask=None, k=4, n_sources=10, freq_grid=(1.0, 5.0, 100),
        criterion='fractional', seed=0)


def test_core_import_does_not_pull_matplotlib():
    # The real invariant: `import ftperiodogram` must not import matplotlib (figures
    # is opt-in). Checked in a fresh interpreter so this test's own imports don't taint it.
    code = ("import sys, ftperiodogram; "
            "sys.exit(0 if 'matplotlib' not in sys.modules else 1)")
    assert subprocess.call([sys.executable, '-c', code]) == 0


def test_plot_recovery_vs_k_returns_axes_with_baseline():
    ax = figures.plot_recovery_vs_k(_k_result())
    assert len(ax.lines) >= 2                       # FTP curve + dashed baseline
    ys = [np.asarray(ln.get_ydata()).tolist() for ln in ax.lines]
    assert [0.3, 0.6, 0.8] in ys
    assert ax.get_ylabel()


def test_plot_recovery_vs_k_without_baseline():
    ax = figures.plot_recovery_vs_k(_k_result(with_baseline=False))
    assert len(ax.lines) == 1                       # no baseline drawn


def test_plot_recovery_vs_nepochs_draws_baseline_curve():
    ax = figures.plot_recovery_vs_nepochs(_n_result())
    assert len(ax.lines) >= 2                       # FTP curve + GLS curve
    xs = [np.asarray(ln.get_xdata()).tolist() for ln in ax.lines]
    assert [5, 10, 20] in xs


def test_plot_recovery_panels():
    fig, (ax_k, ax_n) = figures.plot_recovery_panels(_k_result(), _n_result())
    assert ax_k.has_data() and ax_n.has_data()
