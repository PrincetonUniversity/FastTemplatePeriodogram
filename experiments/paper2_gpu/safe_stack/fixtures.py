"""Deterministic fixtures for the SAFE CPU stack (audit 2026-07-18 s2 items 1-6).

Every fixture is a dict with the flat multiband arrays (t, y, bands, dy), the
template vocabulary (list of {band: Template} sets or single-band Template
list), H, and the frequency grid. Seeds are fixed so golden captures are
reproducible across sessions.
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'trackB'))

import lib  # noqa: E402  (trackB harness: templates, cadences, injection)
from ftperiodogram.template import Template  # noqa: E402


def _grid(n, f_lo=1.0, f_hi=4.0):
    df = (f_hi - f_lo) / n
    n0 = int(round(f_lo / df))
    return df * (n0 + np.arange(n))


def mb_fixture(subtype='RRab', H=8, K_vocab=4, n_per_band=8, seed=7,
               nfreq=2000, mean_mag=21.0, cadence='ps1'):
    """Multiband griz fixture: PS1 pair cadence + BV template vocab."""
    vocab_stars, truth_stars = lib.split_stars(subtype, seed=0)
    vocab = lib.detection_vocab(subtype, H, K_vocab, stars=vocab_stars)
    cad = lib.ps1_cadence(n_per_band, seed=seed)
    truth_bank = lib.load_bank(subtype, H)
    truth = lib.shared_shape_template(truth_bank, truth_stars[0])
    src = lib.inject(truth, subtype, cad, seed, mean_mag=mean_mag)
    freqs = _grid(nfreq)
    return dict(t=src['t'], y=src['y'], bands=src['bands'], dy=src['dy'],
                vocab=vocab, H=H, freqs=freqs, label='mb_%s_h%d' % (subtype, H))


def mb_sparse2_fixture(H=4, seed=3, nfreq=2000):
    """Sparse 2-band fixture (ZTF-like g/r, N=8/band): drives the scan-exact
    fallback zone (B4: mean 5.5%, max 22% of rows at H4)."""
    oids = lib.ztf_cadence_ids()
    cad = lib.load_ztf_cadence(oids[seed % len(oids)], bands=('g', 'r'),
                               max_epochs_per_band=8)
    vocab_stars, truth_stars = lib.split_stars('RRab', seed=0)
    vocab = lib.detection_vocab('RRab', H, 2, bands=('g', 'r'),
                                stars=vocab_stars)
    truth_bank = lib.load_bank('RRab', H, bands=('g', 'r'))
    truth = lib.shared_shape_template(truth_bank, truth_stars[0])
    src = lib.inject(truth, 'RRab', cad, seed, mean_mag=20.5)
    freqs = _grid(nfreq)
    return dict(t=src['t'], y=src['y'], bands=src['bands'], dy=src['dy'],
                vocab=vocab, H=H, freqs=freqs, label='mb_sparse2_h%d' % H)


def sb_fixture(H=8, N=40, seed=11, nfreq=2000):
    """Single-band fixture: irregular sampling, RRab-like Fourier template."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 400.0, N))
    cn = 0.5 ** np.arange(1, H + 1)
    sn = 0.3 * 0.6 ** np.arange(1, H + 1)
    tmpl = Template(cn, sn)
    f0 = 2.123
    y = 0.8 * np.asarray(tmpl((f0 * t) % 1.0), dtype=float)
    dy = np.full(N, 0.05)
    y = y + rng.normal(0, 1, N) * dy
    freqs = _grid(nfreq)
    return dict(t=t, y=y, dy=dy, template=tmpl, H=H, freqs=freqs,
                label='sb_h%d' % H)


def sb_dip_fixture(H=8, seed=5, nfreq=2000):
    """Adversarial phase-clumped single-band fixture: observations clustered
    at a few phases so |MM| carries deep circle dips (exact-fallback zone)."""
    rng = np.random.default_rng(seed)
    N = 24
    f_clump = 2.0
    base = rng.choice([0.0, 0.31, 0.62], size=N)          # 3 phase clumps
    t = np.sort((np.arange(N) + base) / f_clump
                + rng.normal(0, 1e-3, N))
    cn = 0.5 ** np.arange(1, H + 1)
    sn = 0.3 * 0.6 ** np.arange(1, H + 1)
    tmpl = Template(cn, sn)
    y = 0.8 * np.asarray(tmpl((2.123 * t) % 1.0), dtype=float)
    dy = np.full(N, 0.05)
    y = y + rng.normal(0, 1, N) * dy
    freqs = _grid(nfreq)
    return dict(t=t, y=y, dy=dy, template=tmpl, H=H, freqs=freqs,
                label='sb_dip_h%d' % H)


def all_fixtures(nfreq=2000):
    return [
        mb_fixture('RRab', H=8, nfreq=nfreq),
        mb_fixture('RRab', H=4, nfreq=nfreq),
        mb_fixture('RRc', H=2, nfreq=nfreq),
        mb_sparse2_fixture(H=4, nfreq=nfreq),
        sb_fixture(H=8, nfreq=nfreq),
        sb_fixture(H=2, nfreq=nfreq),
        sb_dip_fixture(H=8, nfreq=nfreq),
    ]
