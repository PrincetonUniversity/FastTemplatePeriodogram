"""Paper-quality figures for the Phase 3.2 production recovery matrix.

Consumes the ``results`` dict written by ``run_production_matrix.py`` (its
``aggregate`` block: per-curve pooled rate + Wilson 95% CI across the pooled
sources of all seeds) and renders the three deliverable figures:

  (i)   recovery-vs-K, sparse + dense -- FTP(PAM) and FTP(greedy) curves with
        Wilson 95% CI bands, the GLS / MHLS / multiband-LS baselines as reference
        lines (with CI spans);
  (ii)  recovery-vs-N_epochs at the knee K -- all five methods as curves;
  (iii) cost-vs-accuracy -- FTP vs the Sesar oracle: equal recovery, log wall-time.

matplotlib only; imported lazily so the experiment stays runnable without it.
"""
import os


_BASE_STYLE = {
    'gls':  dict(color='0.45', ls='--', marker='s', label='GLS (sinusoid LB)'),
    'mbls': dict(color='tab:green', ls='-.', marker='^', label='multiband LS (VdP&I)'),
    'mhls': dict(color='tab:red', ls=':', marker='v',
                 label='MHLS (free shape, capped $H$)'),
}


def _band(ax, x, stat, **kw):
    """Plot the pooled-rate line with its Wilson 95% CI band."""
    import numpy as np
    m = np.asarray(stat['mean'], float)
    line, = ax.plot(x, m, **kw)
    ax.fill_between(x, np.asarray(stat['lo'], float), np.asarray(stat['hi'], float),
                    color=line.get_color(), alpha=0.15)
    return line


def _k_panel(ax, k, title):
    import numpy as np
    kv = k['k_values']
    _band(ax, kv, k['ftp_pam'], color='tab:blue', marker='o',
          label='FTP (PAM vocab)')
    _band(ax, kv, k['ftp_greedy'], color='tab:purple', marker='D',
          label='FTP (greedy vocab)')
    for name, st in _BASE_STYLE.items():
        b = k['baselines'][name]
        ax.axhline(float(np.asarray(b['mean'])[0]), color=st['color'], ls=st['ls'],
                   label=st['label'], lw=1.4)
        ax.axhspan(float(np.asarray(b['lo'])[0]), float(np.asarray(b['hi'])[0]),
                   color=st['color'], alpha=0.08, lw=0)
    ax.set_xlabel('vocabulary size $K$')
    ax.set_ylabel('period recovery rate')
    ax.set_ylim(0, 1.02)
    ax.set_xscale('log', base=2)
    ax.set_xticks(kv)
    ax.get_xaxis().set_major_formatter(__import__('matplotlib').ticker.ScalarFormatter())
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)


def make_figures(results, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    agg = results['aggregate']

    # (i) recovery-vs-K, sparse + dense -------------------------------------------
    fig, (axs, axd) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    ks = agg['k_sweep_sparse']; kd = agg['k_sweep_dense']
    _k_panel(axs, ks, 'Recovery vs $K$ (sparse: %d epochs/band)'
             % results['per_seed'][0]['k_sweep_sparse']['n_epochs'])
    _k_panel(axd, kd, 'Recovery vs $K$ (dense: %d epochs/band)'
             % results['per_seed'][0]['k_sweep_dense']['n_epochs'])
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_recovery_vs_k.pdf'), bbox_inches='tight')
    plt.close(fig)

    # (ii) recovery-vs-N_epochs at knee K -----------------------------------------
    n = agg['n_epochs_sweep']
    nv = n['n_epochs_values']
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    _band(ax, nv, n['ftp_pam'], color='tab:blue', marker='o', label='FTP (PAM vocab)')
    _band(ax, nv, n['ftp_greedy'], color='tab:purple', marker='D',
          label='FTP (greedy vocab)')
    for name, st in _BASE_STYLE.items():
        _band(ax, nv, n[name], color=st['color'], ls=st['ls'], marker=st['marker'],
              label=st['label'])
    ax.set_xlabel('epochs per band $N$')
    ax.set_ylabel('period recovery rate')
    ax.set_ylim(0, 1.02)
    kset = sorted(set(n['k_per_seed']))
    ax.set_title('Recovery vs $N_{\\rm epochs}$ at $K=%s$'
                 % (kset[0] if len(kset) == 1 else '{%s}' % ','.join(map(str, kset))))
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_recovery_vs_nepochs.pdf'), bbox_inches='tight')
    plt.close(fig)

    # (iii) cost-vs-accuracy ------------------------------------------------------
    if 'cost' in agg:
        c = agg['cost']
        speedup = float(np.asarray(c['speedup']['mean'])[0])
        ftp_rec = float(np.asarray(c['ftp_recovery']['mean'])[0])
        orc_rec = float(np.asarray(c['oracle_recovery']['mean'])[0])
        # reconstruct absolute times from the first seed for the bar heights
        cs = results['per_seed'][0]['cost']
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        bars = ax.bar(['FTP', 'Sesar oracle'], [cs['ftp_seconds'], cs['oracle_seconds']],
                      color=['tab:blue', 'tab:orange'])
        ax.set_yscale('log')
        ax.set_ylabel('wall time, %d sources x %d freq (s)'
                      % (cs['n_sources'], cs['n_freq']))
        ax.set_title('Cost vs accuracy: %.0f$\\times$ speedup\n'
                     'recovery FTP=%.3f vs oracle=%.3f' % (speedup, ftp_rec, orc_rec))
        for b, rec in zip(bars, [ftp_rec, orc_rec]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    'rec=%.3f' % rec, ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'fig_cost_vs_accuracy.pdf'), bbox_inches='tight')
        plt.close(fig)
