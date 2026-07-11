#!/usr/bin/env python
"""WP E3 phase 2 -- false-alarm behaviour of the FTP catalog on REAL ZTF photometry.

The 99 cached objects in ~/.ftperiodogram_data/ztf_cadence_sample/ are
magnitude-selected field stars (14-18.5), i.e. a near-null real sample (the
phase-1 screen, dev 4525c35, confirms: only 2/99 are catalogued periodic
variables).  This script runs the production FTP catalog statistic (scan path,
the C4 default) on the real photometry and compares the max-power distribution
to pure-noise nulls, honouring the AUDIT_2026-07-10 caveat that a null does
NOT transfer across N / band structure / grid:

  (a) PRIMARY   -- per object x band: seeded random subsample to N=40 good
                   (catflags==0) epochs, scanned on the IDENTICAL 10k-freq
                   [1,5] cyc/day grid as the D1 null (experiments/fap/
                   output_null/null_maxpower.npz) -> directly comparable to D1.
                   (Residual caveat, reported in the aggregate: the real
                   subsampled baseline ~2730 d exceeds D1's 1096 d, so the
                   10k grid holds ~0.9 pts/Rayleigh for the real data vs 2.28
                   for D1 -- slightly more independent trials for the real
                   side even under pure noise.)
  (b) SECONDARY -- per object x band: full-N run on that object's own grid,
                   df <= 0.2/T over explicit [1,5] cyc/day (df and
                   points-per-Rayleigh recorded per protocol element 5).
  (c) MATCHED NULL -- fresh bounded null with D1's cadence/error model
                   (SyntheticCadence seasonal windows + exp_mag_error,
                   amplitude=0, mean_mag=15; max power is scale-invariant for
                   homoscedastic noise so mean_mag is second order), but with
                   n_epochs AND baseline matched to the sample's per-band
                   MEDIANS, on the matching df=0.2/T grid -> N-, T- and
                   grid-matched to (b).  Two strata (g, r), N_REAL_NULL
                   realizations each.

Statistics everywhere (same nested K-prefix vocabulary as D1, seed=0):
  FTP catalog-max over the order-8 Sesar PAM vocab, K in {1,2,4,8} (nested
  cumulative maxima -> pure extra-trials axis), FTP single-medoid H in {1,3},
  and a GLS baseline.  All FTP runs use the default method='scan' (C4).

Frequency bounds are ALWAYS explicit [1,5] cyc/day; nyquist_factor is never
used.

Chunked + resumable: one npz per (object, band) under output/, one npz per
10-realization null chunk; every stage skips work whose output exists.

Usage (drive in bounded batches until each stage reports complete):
  run_e3.py --stage real  [--limit 60] [--workers 8]
  run_e3.py --stage null  [--limit 24] [--workers 8]
  run_e3.py --stage aggregate
"""
import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool

# Keep one process = one core (clean CPU accounting, no BLAS oversubscription).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

DATA_DIR = os.path.expanduser('~/.ftperiodogram_data/ztf_cadence_sample')
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'output')
D1_NPZ = os.path.abspath(os.path.join(
    HERE, '..', '..', 'fap', 'output_null', 'null_maxpower.npz'))
SCREEN_JSON = os.path.join(HERE, 'screen_table.json')
RESULTS_NPZ = os.path.join(HERE, 'e3_results.npz')

# --- configuration (mirrors D1 / EXECUTION_PLAN element (5)) ---------------
F_MIN, F_MAX = 1.0, 5.0          # cyc/day, explicit
N_FREQ_D1 = 10000                # the D1 grid
N_SUB = 40                       # primary subsample size (matches D1 n_obs)
HARMONICS = (1, 3)               # single-medoid FTP orders
VOCAB_H = 8                      # PAM vocabulary order (as D1)
K_VALUES = (1, 2, 4, 8)          # nested catalog prefixes (as D1)
PAM_SEED = 0                     # identical vocab to D1
DF_FACTOR = 0.2                  # df <= DF_FACTOR / T (protocol element 5)
SUB_SEED_BASE = 500000           # primary subsample seeds
NULL_SEED_BASE = 3000000         # matched-null seeds (disjoint from D1's 1e6+)
N_REAL_NULL = 500                # realizations per stratum
NULL_CHUNK = 10                  # realizations per resumable chunk
SEASON_LENGTH_DAYS = 270.0       # D1 cadence model
SEASON_PERIOD_DAYS = 365.25
MEAN_MAG = 15.0
BAND_IDX = {'g': 0, 'r': 1, 'i': 2}

_G = {}  # worker globals


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------
def build_templates():
    """Identical construction to measure_null.py (same seeds -> same vocab)."""
    from ftperiodogram.catalog_builder import (fetch_sesar_templates,
                                               build_template_catalog)
    ftp_coeffs = {}
    for H in HARMONICS:
        lib_H = fetch_sesar_templates(nharmonics=H)
        med = build_template_catalog(lib_H, 1, method='pam',
                                     random_state=PAM_SEED)[0]
        ftp_coeffs[int(H)] = (np.asarray(med.c_n), np.asarray(med.s_n))
    lib_V = fetch_sesar_templates(nharmonics=VOCAB_H)
    vocab = build_template_catalog(lib_V, max(K_VALUES), method='pam',
                                   random_state=PAM_SEED)
    vocab_coeffs = [(np.asarray(tm.c_n), np.asarray(tm.s_n)) for tm in vocab]
    return ftp_coeffs, vocab_coeffs


def _init_worker(ftp_coeffs, vocab_coeffs):
    warnings.simplefilter('ignore')
    from ftperiodogram.baselines import GLSEstimator
    _G['ftp_coeffs'] = {int(H): (np.asarray(c), np.asarray(s))
                        for H, (c, s) in ftp_coeffs.items()}
    _G['vocab_coeffs'] = [(np.asarray(c), np.asarray(s))
                          for c, s in vocab_coeffs]
    _G['gls'] = GLSEstimator()


def all_stats(t, y, dy, freqs):
    """FTP H in HARMONICS + nested catalog-max + GLS; max power over `freqs`."""
    from ftperiodogram.core import template_periodogram

    def maxp(c_n, s_n):
        p, _ = template_periodogram(t, y, dy, c_n, s_n, freqs)  # method='scan'
        return float(np.max(p))

    out = {}
    for H, (c_n, s_n) in _G['ftp_coeffs'].items():
        out['ftp_H%d' % H] = maxp(c_n, s_n)
    out['gls'] = float(np.max(_G['gls'].power_spectrum(t, y, None, dy, freqs)))
    vt = np.array([maxp(c_n, s_n) for c_n, s_n in _G['vocab_coeffs']])
    out['vocab_per_template'] = vt
    for K in K_VALUES:
        out['cat_K%d' % K] = float(np.max(vt[:K]))
    return out


def secondary_grid(T):
    """df <= DF_FACTOR/T grid over explicit [F_MIN, F_MAX]."""
    from ftperiodogram.validation import frequency_grid
    df_target = DF_FACTOR / T
    nf = int(np.ceil((F_MAX - F_MIN) / df_target)) + 1
    freqs = frequency_grid(F_MIN, F_MAX, nf)
    df = float(freqs[1] - freqs[0])
    return freqs, nf, df, (1.0 / T) / df   # pts per Rayleigh


def load_object(gid):
    d = np.load(os.path.join(DATA_DIR, '%s.npz' % gid), allow_pickle=True)
    meta = d['metadata'].item()
    ok = (d['catflags'] == 0) & np.isfinite(d['mag']) & np.isfinite(d['magerr'])
    per_band = {}
    for b in sorted(set(d['band'][ok])):
        m = ok & (d['band'] == b)
        order = np.argsort(d['mjd'][m])
        per_band[str(b)] = (d['mjd'][m][order], d['mag'][m][order],
                            d['magerr'][m][order])
    return per_band, meta


def list_gids():
    with open(os.path.join(DATA_DIR, 'manifest.json')) as fh:
        return json.load(fh)['gids']


def band_strata():
    """Per-band median n_good and median baseline over all band-LCs (>=N_SUB)."""
    strata = {}
    for i, gid in enumerate(list_gids()):
        per_band, _ = load_object(gid)
        for b, (t, y, dy) in per_band.items():
            if t.size < N_SUB:
                continue
            strata.setdefault(b, {'N': [], 'T': []})
            strata[b]['N'].append(t.size)
            strata[b]['T'].append(float(t.max() - t.min()))
    out = {}
    for b in sorted(strata):
        out[b] = dict(N_med=int(round(np.median(strata[b]['N']))),
                      T_med=float(np.median(strata[b]['T'])),
                      n_lcs=len(strata[b]['N']))
    return out


# ---------------------------------------------------------------------------
# stage: real  (primary N=40 subsample + secondary full-N, one npz per unit)
# ---------------------------------------------------------------------------
def run_real_unit(args):
    i_obj, gid, band = args
    from ftperiodogram.validation import frequency_grid
    t0 = time.time()
    per_band, meta = load_object(gid)
    t, y, dy = per_band[band]
    n_good = int(t.size)
    rec = dict(gid=gid, band=band, group_mode=str(meta['group_mode']),
               n_good=n_good)

    # (a) primary: seeded N=40 subsample on the exact D1 grid
    skipped_lt40 = n_good < N_SUB
    rec['skipped_lt40'] = skipped_lt40
    if not skipped_lt40:
        seed = SUB_SEED_BASE + i_obj * 10 + BAND_IDX.get(band, 9)
        idx = np.sort(np.random.default_rng(seed).choice(
            n_good, N_SUB, replace=False))
        freqs = frequency_grid(F_MIN, F_MAX, N_FREQ_D1)
        st = all_stats(t[idx], y[idx], dy[idx], freqs)
        rec['prim_seed'] = seed
        rec['prim_idx'] = idx
        rec['prim_T'] = float(t[idx].max() - t[idx].min())
        for k, v in st.items():
            rec['prim_%s' % k] = v

    # (b) secondary: full N on this object's df<=0.2/T grid
    T = float(t.max() - t.min())
    freqs, nf, df, ppr = secondary_grid(T)
    st = all_stats(t, y, dy, freqs)
    rec.update(sec_T=T, sec_n_freq=nf, sec_df=df, sec_pts_per_rayleigh=ppr)
    for k, v in st.items():
        rec['sec_%s' % k] = v
    rec['elapsed_s'] = time.time() - t0

    np.savez_compressed(os.path.join(OUTDIR, 'real_%s_%s.npz' % (gid, band)),
                        **rec)
    print('[e3]   real %s %s: N=%d nf=%d df=%.3g ppr=%.2f  '
          'prim_catK8=%s sec_catK8=%.4f  (%.1fs)'
          % (gid, band, n_good, nf, df, ppr,
             ('%.4f' % rec['prim_cat_K8']) if not skipped_lt40 else 'SKIP<40',
             rec['sec_cat_K8'], rec['elapsed_s']), flush=True)
    return rec['elapsed_s']


def stage_real(limit, workers, ftp_coeffs, vocab_coeffs):
    gids = list_gids()
    units = []
    for i, gid in enumerate(gids):
        per_band, _ = load_object(gid)
        for b in sorted(per_band):
            if not os.path.exists(os.path.join(OUTDIR,
                                               'real_%s_%s.npz' % (gid, b))):
                units.append((i, gid, b))
    print('[e3] real stage: %d units pending' % len(units), flush=True)
    units = units[:limit]
    if not units:
        print('[e3] real stage COMPLETE', flush=True)
        return
    with Pool(workers, initializer=_init_worker,
              initargs=(ftp_coeffs, vocab_coeffs)) as pool:
        el = pool.map(run_real_unit, units, chunksize=1)
    print('[e3] real batch done: %d units, %.1f CPU-s' % (len(el), sum(el)),
          flush=True)


# ---------------------------------------------------------------------------
# stage: null  (matched strata, chunked realizations)
# ---------------------------------------------------------------------------
def run_null_chunk(args):
    b_idx, band, N_med, T_med, chunk_idx = args
    from ftperiodogram.simulate import SyntheticCadence, simulate_lightcurve
    from ftperiodogram.template import Template
    t0 = time.time()
    freqs, nf, df, ppr = secondary_grid(T_med)
    dummy = Template(np.array([1.0]), np.array([0.0]))
    rows, seeds, nobs = [], [], []
    for j in range(NULL_CHUNK):
        ridx = chunk_idx * NULL_CHUNK + j
        if ridx >= N_REAL_NULL:
            break
        seed = NULL_SEED_BASE + b_idx * 500000 + ridx
        # D1 cadence/error model, N and baseline matched to the stratum
        cad = SyntheticCadence(n_epochs=N_med, bands=None,
                               baseline_days=T_med,
                               season_length_days=SEASON_LENGTH_DAYS,
                               season_period_days=SEASON_PERIOD_DAYS,
                               err_model=None, random_state=seed)
        lc = simulate_lightcurve(dummy, 1.0, cad, amplitude=0.0,
                                 mean_mag=MEAN_MAG, random_state=seed)
        st = all_stats(lc.t, lc.y, lc.dy, freqs)
        rows.append(st)
        seeds.append(seed)
        nobs.append(int(lc.t.size))
    rec = dict(band=band, N_med=N_med, T_med=T_med, n_freq=nf, df=df,
               pts_per_rayleigh=ppr, chunk_idx=chunk_idx,
               seeds=np.array(seeds), n_obs=np.array(nobs),
               elapsed_s=time.time() - t0)
    for key in (['ftp_H%d' % H for H in HARMONICS] + ['gls'] +
                ['cat_K%d' % K for K in K_VALUES]):
        rec[key] = np.array([r[key] for r in rows])
    rec['vocab_per_template'] = np.array([r['vocab_per_template']
                                          for r in rows])
    np.savez_compressed(
        os.path.join(OUTDIR, 'null_%s_c%03d.npz' % (band, chunk_idx)), **rec)
    print('[e3]   null %s chunk %d: %d reals, nf=%d, %.1fs'
          % (band, chunk_idx, len(rows), nf, rec['elapsed_s']), flush=True)
    return rec['elapsed_s']


def stage_null(limit, workers, ftp_coeffs, vocab_coeffs):
    strata = band_strata()
    print('[e3] null strata (from sample medians): %s'
          % json.dumps(strata), flush=True)
    n_chunks = int(np.ceil(N_REAL_NULL / NULL_CHUNK))
    units = []
    for b_idx, band in enumerate(sorted(strata)):
        s = strata[band]
        for c in range(n_chunks):
            if not os.path.exists(os.path.join(
                    OUTDIR, 'null_%s_c%03d.npz' % (band, c))):
                units.append((b_idx, band, s['N_med'], s['T_med'], c))
    print('[e3] null stage: %d chunks pending (%d per stratum, %d reals each)'
          % (len(units), n_chunks, NULL_CHUNK), flush=True)
    units = units[:limit]
    if not units:
        print('[e3] null stage COMPLETE', flush=True)
        return
    with Pool(workers, initializer=_init_worker,
              initargs=(ftp_coeffs, vocab_coeffs)) as pool:
        el = pool.map(run_null_chunk, units, chunksize=1)
    print('[e3] null batch done: %d chunks, %.1f CPU-s' % (len(el), sum(el)),
          flush=True)


# ---------------------------------------------------------------------------
# stage: aggregate
# ---------------------------------------------------------------------------
STAT_KEYS = (['ftp_H%d' % H for H in HARMONICS] + ['gls'] +
             ['cat_K%d' % K for K in K_VALUES])


def emp_pvalue(x, null):
    """(1 + #{null >= x}) / (n + 1)."""
    return (1.0 + np.sum(null >= x)) / (null.size + 1.0)


def stage_aggregate():
    from scipy import stats as sps
    with open(SCREEN_JSON) as fh:
        screen = json.load(fh)['objects']
    gids = list_gids()

    # ---- collect real units ----
    rows = []
    for gid in gids:
        per_band, _ = load_object(gid)
        for b in sorted(per_band):
            p = os.path.join(OUTDIR, 'real_%s_%s.npz' % (gid, b))
            if not os.path.exists(p):
                raise SystemExit('missing real unit %s -- run --stage real' % p)
            rows.append(dict(np.load(p, allow_pickle=True)))
    n_units = len(rows)
    gid_a = np.array([str(r['gid']) for r in rows])
    band_a = np.array([str(r['band']) for r in rows])
    clean_a = np.array([bool(screen[str(r['gid'])]['clean']) for r in rows])
    group_a = np.array([str(r['group_mode']) for r in rows])
    ngood_a = np.array([int(r['n_good']) for r in rows])
    skip_a = np.array([bool(r['skipped_lt40']) for r in rows])
    elapsed_real = float(np.sum([float(r['elapsed_s']) for r in rows]))

    def col(prefix, key, default=np.nan):
        return np.array([float(r['%s_%s' % (prefix, key)])
                         if ('%s_%s' % (prefix, key)) in r else default
                         for r in rows])

    # ---- D1 null ----
    d1 = np.load(D1_NPZ, allow_pickle=True)
    d1_cfg = json.loads(str(d1['config_json']))

    # ---- matched null ----
    strata = band_strata()
    n_chunks = int(np.ceil(N_REAL_NULL / NULL_CHUNK))
    null_cols, null_meta, elapsed_null = {}, {}, 0.0
    for band in sorted(strata):
        acc = {k: [] for k in STAT_KEYS}
        nf = df = ppr = None
        for c in range(n_chunks):
            p = os.path.join(OUTDIR, 'null_%s_c%03d.npz' % (band, c))
            if not os.path.exists(p):
                raise SystemExit('missing null chunk %s -- run --stage null' % p)
            d = np.load(p, allow_pickle=True)
            for k in STAT_KEYS:
                acc[k].append(d[k])
            nf, df, ppr = int(d['n_freq']), float(d['df']), \
                float(d['pts_per_rayleigh'])
            elapsed_null += float(d['elapsed_s'])
        null_cols[band] = {k: np.concatenate(acc[k]) for k in STAT_KEYS}
        null_meta[band] = dict(strata[band], n_freq=nf, df=df,
                               pts_per_rayleigh=ppr,
                               n_real=int(null_cols[band][STAT_KEYS[0]].size))

    # ---- summaries ----
    def dist_summary(x):
        x = x[np.isfinite(x)]
        return dict(n=int(x.size), median=float(np.median(x)),
                    mean=float(np.mean(x)), p90=float(np.quantile(x, 0.90)),
                    p99=float(np.quantile(x, 0.99)) if x.size >= 100 else None,
                    max=float(np.max(x)))

    primary, secondary = {}, {}
    top_prim, top_sec = [], []
    for key in STAT_KEYS:
        d1_null = np.asarray(d1[key], float)
        thr01, thr05 = np.quantile(d1_null, 0.99), np.quantile(d1_null, 0.95)
        pv = col('prim', key)
        fin = np.isfinite(pv)
        pvals = np.array([emp_pvalue(v, d1_null) if np.isfinite(v) else np.nan
                          for v in pv])
        for i in np.where(fin & (pv > thr01))[0]:
            top_prim.append(dict(gid=str(gid_a[i]), band=str(band_a[i]),
                                 stat=key, power=float(pv[i]),
                                 d1_pvalue=float(pvals[i]),
                                 clean=bool(clean_a[i])))
        ks_clean = sps.ks_2samp(pv[fin & clean_a], d1_null)
        primary[key] = dict(
            d1_thr_fap01=float(thr01), d1_thr_fap05=float(thr05),
            all=dist_summary(pv),
            clean=dist_summary(pv[clean_a]),
            screened=dist_summary(pv[~clean_a]),
            n_exceed_fap01=dict(
                clean=int(np.sum(fin & clean_a & (pv > thr01))),
                screened=int(np.sum(fin & ~clean_a & (pv > thr01)))),
            n_exceed_fap05=dict(
                clean=int(np.sum(fin & clean_a & (pv > thr05))),
                screened=int(np.sum(fin & ~clean_a & (pv > thr05)))),
            d1_null_median=float(np.median(d1_null)),
            ks_clean_vs_d1=dict(stat=float(ks_clean.statistic),
                                p=float(ks_clean.pvalue)))
        sec_by_band = {}
        for band in sorted(strata):
            nb = null_cols[band][key]
            thr01b, thr05b = np.quantile(nb, 0.99), np.quantile(nb, 0.95)
            m = band_a == band
            sv = col('sec', key)[m]
            cl = clean_a[m]
            for i_loc, i in enumerate(np.where(m)[0]):
                if sv[i_loc] > thr01b:
                    top_sec.append(dict(gid=str(gid_a[i]), band=band, stat=key,
                                        power=float(sv[i_loc]),
                                        null_pvalue=float(
                                            emp_pvalue(sv[i_loc], nb)),
                                        clean=bool(clean_a[i])))
            ks_c = sps.ks_2samp(sv[cl], nb)
            sec_by_band[band] = dict(
                null_thr_fap01=float(thr01b), null_thr_fap05=float(thr05b),
                null_median=float(np.median(nb)),
                all=dist_summary(sv), clean=dist_summary(sv[cl]),
                screened=dist_summary(sv[~cl]),
                n_exceed_fap01=dict(clean=int(np.sum(cl & (sv > thr01b))),
                                    screened=int(np.sum(~cl & (sv > thr01b)))),
                n_exceed_fap05=dict(clean=int(np.sum(cl & (sv > thr05b))),
                                    screened=int(np.sum(~cl & (sv > thr05b)))),
                ks_clean_vs_null=dict(stat=float(ks_c.statistic),
                                      p=float(ks_c.pvalue)))
        secondary[key] = sec_by_band

    summary = dict(
        n_objects=len(gids), n_band_lcs=n_units,
        n_skipped_lt40=int(np.sum(skip_a)),
        n_clean_band_lcs=int(np.sum(clean_a)),
        n_screened_band_lcs=int(np.sum(~clean_a)),
        primary_grid=dict(f_min=F_MIN, f_max=F_MAX, n_freq=N_FREQ_D1,
                          n_sub=N_SUB),
        secondary_grid_rule='df <= %.3g/T, [%g,%g] cyc/day' % (DF_FACTOR,
                                                               F_MIN, F_MAX),
        secondary_df=dict(median=float(np.median(col('sec', 'df'))),
                          min=float(np.min(col('sec', 'df'))),
                          max=float(np.max(col('sec', 'df')))),
        secondary_pts_per_rayleigh=dict(
            median=float(np.median(col('sec', 'pts_per_rayleigh'))),
            min=float(np.min(col('sec', 'pts_per_rayleigh')))),
        secondary_n_freq=dict(median=float(np.median(col('sec', 'n_freq'))),
                              max=int(np.max(col('sec', 'n_freq')))),
        d1_config=d1_cfg, matched_null=null_meta,
        primary=primary, secondary=secondary,
        top_primary_exceedances=sorted(top_prim, key=lambda r: -r['power']),
        top_secondary_exceedances=sorted(top_sec, key=lambda r: -r['power']),
        cpu_seconds=dict(real=elapsed_real, null=elapsed_null,
                         total=elapsed_real + elapsed_null),
        caveats=[
            'n=99 objects (155 band-LCs) -> per-object FAP floor ~1%;'
            ' D1-null empirical thresholds from n=%d realizations.'
            % int(np.asarray(d1[STAT_KEYS[0]]).size),
            'Primary comparison: real subsampled baseline (~2730 d) > D1 null'
            ' baseline (1096 d), so the shared 10k grid gives the real side'
            ' ~2.5x more independent frequencies; real primary maxima run'
            ' slightly hot vs D1 even under pure noise.',
            'Only the 8 same_star g+r objects are physical two-band stars;'
            ' 48 same_field_cadence pairs are cadence nulls only.',
            'Matched null keeps the D1 synthetic seasonal cadence + error'
            ' model but matches per-band median N and baseline T and the'
            ' df=0.2/T grid of the secondary runs (audit: nulls do not'
            ' transfer across N/grid).'])

    # persist aggregate
    save = dict(summary_json=json.dumps(summary),
                gid=gid_a, band=band_a, clean=clean_a, group_mode=group_a,
                n_good=ngood_a, skipped_lt40=skip_a)
    for key in STAT_KEYS:
        save['prim_%s' % key] = col('prim', key)
        save['sec_%s' % key] = col('sec', key)
    for extra in ('T', 'n_freq', 'df', 'pts_per_rayleigh'):
        save['sec_%s' % extra] = col('sec', extra)
    save['prim_T'] = col('prim', 'T')
    for band in null_cols:
        for key in STAT_KEYS:
            save['null_%s_%s' % (band, key)] = null_cols[band][key]
        save['null_%s_meta_json' % band] = json.dumps(null_meta[band])
    np.savez_compressed(RESULTS_NPZ, **save)
    print('[e3] wrote %s' % RESULTS_NPZ, flush=True)
    print(json.dumps(summary, indent=1), flush=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['real', 'null', 'aggregate'],
                    required=True)
    ap.add_argument('--limit', type=int, default=10 ** 9)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    if args.stage == 'aggregate':
        stage_aggregate()
        return
    warnings.simplefilter('ignore')
    ftp_coeffs, vocab_coeffs = build_templates()
    if args.stage == 'real':
        stage_real(args.limit, args.workers, ftp_coeffs, vocab_coeffs)
    else:
        stage_null(args.limit, args.workers, ftp_coeffs, vocab_coeffs)


if __name__ == '__main__':
    main()
