"""WP E4: aggregate the 438 per-(star, method) recovery outputs into one npz.

Reads every ``output/<uid>__<method>.npz`` produced by ``run_e4.py`` (full
spectra stay on disk only -- 78 MB, not committed) plus ``star_table.json``,
and writes ``e4_results.npz`` holding, per (star, method) row:

  * identification: uid, method, config, role (science / control_antijoin /
    control_orig), field, band_used, primary_band, var_type (Chen),
    gaia_type, dual_truth flag,
  * grid provenance: f_min, f_max, df, i_min, n_freq, T_grid_days,
    pts_per_rayleigh, n_obs,
  * spectrum summary for alias-aware scoring: best_freq, top5_freq (5),
    top5_stat (5), stat_is_neg_entropy, mhls_capped_order,
  * matched-truth inputs: chen_period, gaia_period (NaN if absent),
    gaia_period_source, gaia_xmatch_sep_arcsec,
  * timing: runtime_s (accumulated compute time incl. resumed blocks).

Scoring itself (E1/B5 alias-aware protocol) is the next phase; this file is
the single input it needs.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAR_TABLE = os.path.join(HERE, 'star_table.json')
OUTDIR = os.path.join(HERE, 'output')
RESULTS = os.path.join(HERE, 'e4_results.npz')

METHODS = ('gls_1band', 'ftp_1band', 'mbls_h1', 'mhls_h8', 'ce', 'ftp_mb')

SCALAR_FIELDS = [
    # (npz key, dtype, fill for missing)
    ('config', 'U80', ''),
    ('band_used', 'U3', ''),
    ('f_min', 'f8', np.nan),
    ('f_max', 'f8', np.nan),
    ('df', 'f8', np.nan),
    ('i_min', 'i8', -1),
    ('n_freq', 'i8', -1),
    ('T_grid_days', 'f8', np.nan),
    ('T_joint_days', 'f8', np.nan),
    ('pts_per_rayleigh', 'f8', np.nan),
    ('n_obs', 'i8', -1),
    ('best_freq', 'f8', np.nan),
    ('stat_is_neg_entropy', '?', False),
    ('mhls_capped_order', 'i8', -1),
    ('runtime_s', 'f8', np.nan),
]

STAR_FIELDS = [
    # (star_table key, dtype, fill)
    ('role', 'U20', ''),
    ('field', 'i8', -1),
    ('type', 'U8', ''),           # Chen var type (RRab/RRc); '' for controls
    ('gaia_type', 'U8', ''),
    ('dual_truth', '?', False),
    ('primary_band', 'U1', ''),
    ('chen_period', 'f8', np.nan),
    ('gaia_period', 'f8', np.nan),
    ('gaia_period_source', 'U16', ''),
    ('gaia_xmatch_sep_arcsec', 'f8', np.nan),
]


def main():
    with open(STAR_TABLE) as fh:
        table = json.load(fh)
    stars = table['stars']

    rows = [(s, m) for s in stars for m in METHODS]
    n = len(rows)

    out = {
        'uid': np.empty(n, dtype='U40'),
        'method': np.empty(n, dtype='U12'),
        'status': np.empty(n, dtype='U8'),          # done | missing
        'top5_freq': np.full((n, 5), np.nan),
        'top5_stat': np.full((n, 5), np.nan),
    }
    for key, dt, fill in SCALAR_FIELDS + STAR_FIELDS:
        out[key] = np.full(n, fill, dtype=dt)

    n_missing = 0
    for i, (star, method) in enumerate(rows):
        uid = star['uid']
        out['uid'][i] = uid
        out['method'][i] = method
        for key, _, fill in STAR_FIELDS:
            v = star.get(key)
            out[key][i] = fill if v is None else v

        path = os.path.join(OUTDIR, '%s__%s.npz' % (uid, method))
        if not os.path.exists(path):
            out['status'][i] = 'missing'
            n_missing += 1
            continue
        d = np.load(path)
        out['status'][i] = 'done'
        for key, _, fill in SCALAR_FIELDS:
            if key in d.files:
                out[key][i] = d[key]
        out['top5_freq'][i] = d['top5_freq']
        out['top5_stat'][i] = d['top5_stat']
        # consistency guards: aggregate must mirror the artifact it summarizes
        assert str(d['uid']) == uid and str(d['method']) == method, path
        assert np.isclose(float(d['best_freq']), out['top5_freq'][i][0]), path

    meta = {
        'created_from': 'aggregate_e4.py over output/*.npz',
        'n_rows': n,
        'n_done': n - n_missing,
        'n_missing': n_missing,
        'methods': list(METHODS),
        'counts': table['counts'],
        'grid_rule': table['grid_summary'],
        'dual_truth_consistency': table['dual_truth_consistency'],
        'scoring_note': ('recovered iff |f-f_true|/f_true < 0.01 AND '
                         '|f-f_true|*T < 0.5; classify exact/harmonic(2f,f/2)'
                         '/alias(+-1, +-1/365.25, +-1/354 cyc/day)/miss '
                         'per E1+B5 -- done in the Score phase, not here'),
        'runtime_note': ('runtime_s = accumulated per-block compute time '
                         '(resumed tasks include all resumes)'),
    }
    out['meta_json'] = np.array(json.dumps(meta, sort_keys=True))

    np.savez_compressed(RESULTS, **out)

    done = out['status'] == 'done'
    cpu_h = float(np.nansum(out['runtime_s'][done])) / 3600.0
    print('rows=%d done=%d missing=%d -> %s (%.1f KB)'
          % (n, int(done.sum()), n_missing, RESULTS,
             os.path.getsize(RESULTS) / 1024.0))
    print('total accumulated compute: %.2f cpu-hours' % cpu_h)
    for m in METHODS:
        sel = done & (out['method'] == m)
        print('  %-10s n=%2d  median runtime %7.1f s  total %7.0f s'
              % (m, int(sel.sum()), float(np.median(out['runtime_s'][sel])),
                 float(np.sum(out['runtime_s'][sel]))))


if __name__ == '__main__':
    main()
