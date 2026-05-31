"""Optional matplotlib helpers for the Phase 3.2 recovery figures.

matplotlib is an OPTIONAL dependency: it is imported lazily inside each function and
this module is deliberately NOT imported from ``ftperiodogram/__init__.py``, so the
core package stays numpy/scipy/nfft-only.  Import what you need explicitly::

    from ftperiodogram.figures import plot_recovery_vs_k, plot_recovery_vs_nepochs

Each helper draws the FTP@H=1 (GLS-equivalent) baseline alongside the FTP curve -- as
a horizontal reference line for the K-sweep (the baseline is K-independent) and as a
curve for the N_epochs sweep (where it varies with epoch count).
"""


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:           # pragma: no cover - via importorskip in tests
        raise ImportError(
            "ftperiodogram.figures requires matplotlib (an optional dependency); "
            "install it with `pip install matplotlib`.") from exc
    return plt


def _resolve_ax(ax):
    if ax is not None:
        return ax
    _, ax = _import_pyplot().subplots()
    return ax


def plot_recovery_vs_k(result, ax=None, label='FTP (learned vocab)',
                       show_baseline=True, baseline_label='FTP@H=1 (GLS)'):
    """Plot period recovery vs vocabulary size K from a ``KSweepResult``.

    Draws the FTP recovery curve and, when ``show_baseline`` and the result carries
    one, a horizontal reference line at the FTP@H=1 (GLS-equivalent) baseline -- the
    baseline is K-independent, so a single line is the natural reference.  Returns the
    matplotlib ``Axes``."""
    ax = _resolve_ax(ax)
    ax.plot(result.k_values, result.recovery, marker='o', label=label)
    if show_baseline and result.baseline_recovery is not None:
        ax.axhline(result.baseline_recovery, ls='--', color='0.4',
                   label=baseline_label)
    ax.set_xlabel('vocabulary size K')
    ax.set_ylabel('period recovery rate')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Recovery vs K (%s)' % result.criterion)
    ax.legend(loc='best')
    return ax


def plot_recovery_vs_nepochs(result, ax=None, label='FTP (K=%d)',
                             show_baseline=True, baseline_label='FTP@H=1 (GLS)'):
    """Plot period recovery vs per-band epoch count from a ``NEpochsSweepResult``.

    The sparse-regime panel (PLAN 3.2 deliverable ii) at fixed K.  Unlike the K-sweep
    the GLS baseline varies with N_epochs, so it is drawn as a curve rather than a
    horizontal line.  Returns the matplotlib ``Axes``."""
    ax = _resolve_ax(ax)
    if '%d' in label:
        label = label % result.k
    ax.plot(result.n_epochs_values, result.recovery, marker='o', label=label)
    if show_baseline and result.baseline_recovery is not None:
        ax.plot(result.n_epochs_values, result.baseline_recovery, marker='s',
                ls='--', color='0.4', label=baseline_label)
    ax.set_xlabel('epochs per band')
    ax.set_ylabel('period recovery rate')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Recovery vs N_epochs at K=%d (%s)' % (result.k, result.criterion))
    ax.legend(loc='best')
    return ax


def plot_recovery_panels(k_result, n_result, figsize=(11, 4.5)):
    """Two-panel figure: recovery-vs-K and recovery-vs-N_epochs side by side.

    Returns ``(fig, (ax_k, ax_n))``."""
    plt = _import_pyplot()
    fig, (ax_k, ax_n) = plt.subplots(1, 2, figsize=figsize)
    plot_recovery_vs_k(k_result, ax=ax_k)
    plot_recovery_vs_nepochs(n_result, ax=ax_n)
    fig.tight_layout()
    return fig, (ax_k, ax_n)
