"""Tests for the recovery scorer + K-sweep driver (ftperiodogram.validation).

The driver bookkeeping (monotone assembly, the single-cosine baseline, masks,
the explicit-grid guard, K validation) is tested with a fast content-blind fake
scorer.  A separate bounded test exercises the *real* scorer + FTP on a tiny
seeded synthetic population (no network), asserting it runs and is reproducible.
"""
import numpy as np
import numpy.testing as npt
import pytest

from ftperiodogram.template import Template
from ftperiodogram.simulate import SyntheticCadence
from ftperiodogram import validation as val


# ----------------------------------------------------------------------
# tiny synthetic vocabulary (common H=4) so build_template_catalog runs
# ----------------------------------------------------------------------
_ARCH = [
    (np.array([1.0, 0.5, 0.33, 0.25]), np.array([0.0, 0.10, 0.05, 0.02])),
    (np.array([1.0, 0.05, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])),
    (np.array([0.2, 1.0, 0.10, 0.05]), np.array([0.0, 0.0, 0.0, 0.0])),
]


def _population(seed=0, n_per=5):
    rng = np.random.RandomState(seed)
    out = []
    for c, s in _ARCH:
        for _ in range(n_per):
            out.append(Template(c + 0.02 * rng.randn(4), s + 0.02 * rng.randn(4)))
    return out


class _FakeScorer:
    """Content-blind deterministic scorer: a size-k set covers the first 3k
    sources.  Records the template sets it is called with."""

    def __init__(self, n_sources=12):
        self.n_sources = n_sources
        self.criterion = 'fractional'
        self.freqs = np.linspace(1.0, 3.0, 64)
        self.calls = []

    def __call__(self, templates, *, return_mask=False):
        templates = list(templates)
        self.calls.append(templates)
        k = len(templates)
        mask = np.arange(self.n_sources) < min(self.n_sources, 3 * k)
        rate = float(mask.mean())
        return (rate, mask) if return_mask else rate


# ----------------------------------------------------------------------
# driver logic (fake scorer, instant)
# ----------------------------------------------------------------------
def test_k_sweep_monotone_and_above_baseline():
    res = val.k_sweep_recovery(_population(seed=1), cadence=None,
                               k_values=(1, 2, 3), scorer=_FakeScorer(12),
                               random_state=0)
    assert np.all(np.diff(res.recovery) >= 0)         # coverage rises with K
    assert res.recovery[-1] >= res.baseline_recovery
    assert res.n_sources == 12
    npt.assert_array_equal(res.k_values, [1, 2, 3])


def test_k_sweep_baseline_is_single_cosine():
    scorer = _FakeScorer()
    val.k_sweep_recovery(_population(seed=1), cadence=None, k_values=(2,),
                         scorer=scorer, random_state=0)
    first = scorer.calls[0]                            # baseline scored first
    assert len(first) == 1
    npt.assert_allclose(first[0].c_n, [1.0])           # H=1 single cosine == GLS
    npt.assert_allclose(first[0].s_n, [0.0])


def test_k_sweep_requires_explicit_grid():
    with pytest.raises(ValueError):                    # no scorer and no grid
        val.k_sweep_recovery(_population(seed=1), cadence=None, k_values=(1, 2))


def test_k_sweep_return_masks_shapes():
    res = val.k_sweep_recovery(_population(seed=1), cadence=None,
                               k_values=(1, 2), scorer=_FakeScorer(10),
                               return_masks=True, random_state=0)
    assert len(res.recovery_by_k) == 2
    assert all(m.shape == (10,) for m in res.recovery_by_k)
    assert res.baseline_mask.shape == (10,)
    npt.assert_allclose(res.recovery[0], res.recovery_by_k[0].mean())


def test_k_sweep_rejects_bad_k():
    scorer = _FakeScorer()
    with pytest.raises(ValueError):
        val.k_sweep_recovery(_population(seed=1), cadence=None, k_values=(0,),
                             scorer=scorer)
    with pytest.raises(ValueError):
        val.k_sweep_recovery(_population(seed=1), cadence=None, k_values=(999,),
                             scorer=scorer)


# ----------------------------------------------------------------------
# real scorer + FTP (bounded, seeded, no network)
# ----------------------------------------------------------------------
def test_recovery_scorer_deterministic_and_masked():
    templates = _population(seed=1)
    cad = SyntheticCadence(n_epochs={'g': 20, 'r': 20}, bands=['g', 'r'],
                           baseline_days=365.25, random_state=2)
    scorer = val.make_recovery_scorer(cad, templates,
                                      freqs=val.frequency_grid(1.0, 3.0, 150),
                                      n_sources=6, random_state=0)
    rate1, mask1 = scorer(templates[:3], return_mask=True)
    rate2, mask2 = scorer(templates[:3], return_mask=True)
    npt.assert_array_equal(mask1, mask2)               # frozen population -> deterministic
    assert rate1 == pytest.approx(mask1.mean())
    assert 0.0 <= rate1 <= 1.0
    masks = scorer.source_masks(templates[:4])
    assert masks.shape == (4, 6)


def test_k_sweep_real_scorer_runs_and_is_reproducible():
    templates = _population(seed=1)
    cad = SyntheticCadence(n_epochs={'g': 20, 'r': 20}, bands=['g', 'r'],
                           baseline_days=365.25, random_state=2)
    kw = dict(k_values=(1, 3), f_min=1.0, f_max=3.0, n_freq=150, n_sources=6,
              random_state=0)
    r1 = val.k_sweep_recovery(templates, cad, **kw)
    r2 = val.k_sweep_recovery(templates, cad, **kw)
    assert r1.recovery.shape == (2,)
    assert np.all(np.isfinite(r1.recovery))
    assert 0.0 <= r1.baseline_recovery <= 1.0
    assert r1.freq_grid[2] == 150                       # n_freq
    npt.assert_allclose(r1.freq_grid[:2], (1.0, 3.0), atol=0.02)  # NFFT-snapped band
    assert r1.criterion == 'fractional'
    npt.assert_array_equal(r1.recovery, r2.recovery)   # fully reproducible


# ----------------------------------------------------------------------
# N_epochs down-sampling + sweep (Feature 1)
# ----------------------------------------------------------------------
def _master_scorer(n_sources=6, n_master=24, seed=0):
    templates = _population(seed=1)
    cad = SyntheticCadence(n_epochs={'g': n_master, 'r': n_master},
                           bands=['g', 'r'], baseline_days=365.25,
                           random_state=2)
    scorer = val.make_recovery_scorer(
        cad, templates, freqs=val.frequency_grid(1.0, 3.0, 150),
        n_sources=n_sources, random_state=seed)
    return templates, scorer


def test_downsample_is_nested_and_per_band():
    _, master = _master_scorer()
    small = master.downsample(8, random_state=0)
    big = master.downsample(16, random_state=0)
    assert small.n_sources == master.n_sources == len(small._sources)
    for (ts, _, bs, _, _, _), (tb, _, bb, _, _, _) in zip(small._sources,
                                                          big._sources):
        for band in np.unique(bb):
            ts_b = set(np.round(ts[bs == band], 9))
            tb_b = set(np.round(tb[bb == band], 9))
            assert len(ts_b) == 8 and len(tb_b) == 16   # exactly n_epochs/band
            assert ts_b <= tb_b                          # nested (small ⊂ big)


def test_downsample_floor_and_ceiling():
    _, master = _master_scorer(n_master=20)
    with pytest.raises(ValueError):
        master.downsample(2)                # below the >=3 per-band floor
    with pytest.raises(ValueError):
        master.downsample(21)               # exceeds master per-band count (20)


def test_downsample_carries_params_and_is_deterministic():
    templates, master = _master_scorer()
    a = master.downsample(10, random_state=0)
    b = master.downsample(10, random_state=0)
    assert a.criterion == master.criterion and a.mode == master.mode
    assert a.delta_phi_max == master.delta_phi_max and a.rtol == master.rtol
    npt.assert_array_equal(a.p_true, master.p_true)
    _, m_a = a(templates[:3], return_mask=True)
    _, m_b = b(templates[:3], return_mask=True)
    npt.assert_array_equal(m_a, m_b)        # same seed -> identical thinned masks


def test_freeze_baseline_keeps_master_T():
    _, master = _master_scorer()
    frozen = master.downsample(8, random_state=0, freeze_baseline=True)
    recomputed = master.downsample(8, random_state=0, freeze_baseline=False)
    for m, fz, rc in zip(master._sources, frozen._sources, recomputed._sources):
        assert fz[5] == m[5]                # baseline preserved
        assert rc[5] <= m[5]                # recomputed T can only shrink


def test_n_epochs_sweep_runs_and_has_baseline_curve():
    templates, master = _master_scorer()
    res = val.n_epochs_sweep_recovery(master, templates, [18, 6, 12], k=2,
                                      random_state=0)
    npt.assert_array_equal(res.n_epochs_values, [6, 12, 18])   # sorted
    assert res.recovery.shape == (3,)
    assert res.baseline_recovery.shape == (3,)   # GLS varies with N -> a curve
    assert np.all(np.isfinite(res.recovery))
    assert res.k == 2 and res.n_sources == master.n_sources
    assert res.criterion == 'fractional'


def test_n_epochs_sweep_rejects_bad_k():
    templates, master = _master_scorer()
    with pytest.raises(ValueError):
        val.n_epochs_sweep_recovery(master, templates, [6], k=0)
    with pytest.raises(ValueError):
        val.n_epochs_sweep_recovery(master, templates, [6], k=999)


# ----------------------------------------------------------------------
# Batched source_masks precompute (Feature 2)
# ----------------------------------------------------------------------
def test_source_masks_batched_matches_per_template():
    templates, scorer = _master_scorer()
    masks = scorer.source_masks(templates[:5])           # batched fast path
    assert masks.shape == (5, scorer.n_sources)
    for i, tmpl in enumerate(templates[:5]):
        _, ref = scorer([tmpl], return_mask=True)        # per-template reference
        npt.assert_array_equal(masks[i], ref)            # mask-identity (argmax-stable)


def test_source_masks_batched_is_deterministic():
    templates, scorer = _master_scorer()
    npt.assert_array_equal(scorer.source_masks(templates[:4]),
                           scorer.source_masks(templates[:4]))


def test_source_masks_mixed_H_falls_back():
    templates, scorer = _master_scorer()
    truncated = Template(templates[1].c_n[:2], templates[1].s_n[:2])
    mixed = [templates[0], truncated]                    # differing H -> fallback
    masks = scorer.source_masks(mixed)
    assert masks.shape == (2, scorer.n_sources)
    for i, tmpl in enumerate(mixed):
        _, ref = scorer([tmpl], return_mask=True)
        npt.assert_array_equal(masks[i], ref)


def test_source_masks_empty():
    _, scorer = _master_scorer()
    assert scorer.source_masks([]).shape == (0, scorer.n_sources)


def test_harness_public_api_exports():
    import ftperiodogram as ftp
    for name in ('Cadence', 'SyntheticCadence', 'exp_mag_error',
                 'simulate_lightcurve', 'simulate_multiband_lightcurve',
                 'recovered_fractional', 'recovered_phase_coherence',
                 'harmonic_alias_set', 'classify_recovery', 'recovery_rate',
                 'RecoveryScorer', 'make_recovery_scorer', 'k_sweep_recovery',
                 'KSweepResult', 'frequency_grid', 'n_epochs_sweep_recovery',
                 'NEpochsSweepResult'):
        assert hasattr(ftp, name)
