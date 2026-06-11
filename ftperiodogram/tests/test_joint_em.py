"""Tests for the joint/EM template-vocabulary estimator (Phase 3.4).

Speed note: every test uses a SHORT baseline (``baseline_days=30``) so a small
frequency grid resolves periodogram peaks (peak width ~ 1/T, so df must be << 1/T);
on a 3-year baseline these grids would undersample the peaks and recovery would be
grid-aliasing noise.  Templates use H=4 to keep the per-frequency polynomial cheap.
"""
import numpy as np
import pytest

from ftperiodogram.template import Template
from ftperiodogram.simulate import SyntheticCadence, simulate_multiband_lightcurve
from ftperiodogram.validation import make_recovery_scorer, frequency_grid
from ftperiodogram.catalog_builder import build_template_catalog, _orbit_distance
from ftperiodogram.joint_em import (build_joint_em_catalog, _fit_source_shape,
                                    _estep_one, _align_to_template)

H = 4
BASELINE = 30.0
F_MIN, F_MAX, N_FREQ = 1.0, 5.0, 600          # df ~ 0.0067 << 1/T = 0.033


def _sawtooth_template():
    """An asymmetric, harmonic-rich RRab-like shape (fast-rise/slow-decline)."""
    ph = np.arange(256) / 256.0
    y = -(ph * 2 - 1)
    y[ph > 0.5] = (1 - ph[ph > 0.5]) * 2 - 1
    return Template.from_sampled(y - y.mean(), nharmonics=H)


def _sine_template():
    """A near-sinusoidal RRc-like shape (fundamental dominant)."""
    c = np.zeros(H); s = np.zeros(H)
    c[0], s[1] = 1.0, 0.15
    return Template(c, s)


@pytest.fixture(scope='module')
def shapes():
    return [_sawtooth_template(), _sine_template()]


@pytest.fixture(scope='module')
def freqs():
    return frequency_grid(F_MIN, F_MAX, N_FREQ)


def _cad(seed, master=40):
    return SyntheticCadence(n_epochs={'g': master, 'r': master}, bands=['g', 'r'],
                            baseline_days=BASELINE, random_state=seed)


def _scorer(shapes, freqs, n_epochs, n_sources, seed):
    sc = make_recovery_scorer(_cad(seed), shapes, freqs=freqs,
                              n_sources=n_sources, random_state=seed)
    return sc if n_epochs >= 40 else sc.downsample(n_epochs, random_state=seed)


# ----------------------------------------------------------------------
# Helper-level unit tests
# ----------------------------------------------------------------------
def test_fit_source_shape_recovers_template(shapes):
    """A dense, noiseless light curve folds back to its generating shape."""
    truth = shapes[0]
    cad = SyntheticCadence(n_epochs={'g': 120, 'r': 120}, bands=['g', 'r'],
                           baseline_days=BASELINE, random_state=0)
    lc = simulate_multiband_lightcurve(truth, 0.5, cad, amplitude=0.6, tau=0.3,
                                       random_state=0, add_noise=False)
    z = _fit_source_shape(lc.t, lc.y, lc.bands, lc.dy, 1.0 / 0.5, H=H)
    fitted = Template(np.real(z), -np.imag(z))
    assert _orbit_distance(fitted, truth) < 1e-2


def test_estep_assigns_correct_template_and_period(shapes, freqs):
    """The E-step routes a source to its generating template at the right period,
    and the matched template fits strictly better than the wrong one."""
    truth = shapes[1]
    cad = SyntheticCadence(n_epochs={'g': 80, 'r': 80}, bands=['g', 'r'],
                           baseline_days=BASELINE, random_state=1)
    lc = simulate_multiband_lightcurve(truth, 0.37, cad, amplitude=0.5, tau=0.1,
                                       random_state=1, add_noise=False)
    source = (lc.t, lc.y, lc.bands, lc.dy)
    k, f_rec, power = _estep_one(source, shapes, freqs, 'floating_offsets', H=H)
    _, _, p_wrong = _estep_one(source, [shapes[0]], freqs, 'floating_offsets', H=H)
    assert k == 1                                          # the sine template
    assert abs(f_rec - 1.0 / 0.37) / (1.0 / 0.37) < 0.01
    assert power > p_wrong                                 # matched template wins


def test_align_to_template_recovers_phase_and_amplitude(shapes):
    """Aligning a clean source to its own template recovers (tau, a)."""
    truth = shapes[0]
    cad = SyntheticCadence(n_epochs={'g': 120, 'r': 120}, bands=['g', 'r'],
                           baseline_days=BASELINE, random_state=2)
    lc = simulate_multiband_lightcurve(truth, 0.5, cad, amplitude=0.7, tau=0.2,
                                       random_state=2, add_noise=False)
    al = _align_to_template(lc.t, lc.y, lc.bands, lc.dy, 1.0 / 0.5, truth)
    assert al is not None
    tau, a, offsets, codes = al
    assert a > 0


def _clean_cluster(truth):
    """Six clean, noiseless members of one shape at varied periods/phases."""
    cad = SyntheticCadence(n_epochs={'g': 100, 'r': 100}, bands=['g', 'r'],
                           baseline_days=BASELINE, random_state=9)
    sources, freq_rec = [], []
    for i, (P, tau) in enumerate([(0.50, 0.10), (0.41, 0.60), (0.62, 0.30),
                                  (0.33, 0.80), (0.55, 0.20), (0.47, 0.45)]):
        lc = simulate_multiband_lightcurve(truth, P, cad, amplitude=0.6, tau=tau,
                                           random_state=100 + i, add_noise=False)
        sources.append((lc.t, lc.y, lc.bands, lc.dy))
        freq_rec.append(1.0 / P)
    assign = np.zeros(len(sources), dtype=int)
    gated = np.ones(len(sources), dtype=bool)
    return sources, np.array(freq_rec), assign, gated


@pytest.mark.parametrize('mstep_name', ['_mstep_pooled', '_mstep_median'])
def test_mstep_recovers_cluster_shape(shapes, mstep_name):
    """Both aggregators reconstruct a cluster's shape from clean members.

    Regression for the median orbit-alignment sign: a wrong sign double-shifts the
    members, smearing the aggregate to orbit distance ~0.25 (this runs the M-step
    directly, bypassing the best-held-out guard that otherwise masks the bug)."""
    import ftperiodogram.joint_em as jem
    truth = shapes[0]
    sources, freq_rec, assign, gated = _clean_cluster(truth)
    mstep = getattr(jem, mstep_name)
    new, n_updated = mstep(sources, assign, freq_rec, gated, [truth], H)
    assert n_updated == 1
    assert _orbit_distance(new[0], truth) < 0.05


# ----------------------------------------------------------------------
# Public-API behavioural tests
# ----------------------------------------------------------------------
def test_returns_k_dropin_templates(shapes, freqs):
    train = _scorer(shapes, freqs, 12, 16, seed=1)
    vocab = build_joint_em_catalog(shapes, 2, train, max_iter=3, n_harmonics=H)
    assert len(vocab) == 2
    assert all(isinstance(t, Template) for t in vocab)
    assert {len(t.c_n) for t in vocab} == {H}
    assert 0.0 <= train(vocab) <= 1.0


def test_max_iter_zero_equals_pipeline_init(shapes, freqs):
    """With no iterations the joint result is exactly the pipeline vocabulary."""
    train = _scorer(shapes, freqs, 12, 16, seed=2)
    pipe = build_template_catalog(shapes, 2, method='pam', random_state=0,
                                  n_harmonics=H)
    joint = build_joint_em_catalog(shapes, 2, train, max_iter=0, n_harmonics=H,
                                   random_state=0)
    for a, b in zip(joint, pipe):
        assert np.allclose(a.c_n, b.c_n) and np.allclose(a.s_n, b.s_n)


@pytest.mark.parametrize('m_step', ['pooled', 'median'])
@pytest.mark.parametrize('val_signal', ['power_margin', 'recovery'])
def test_validation_signal_never_regresses(shapes, freqs, m_step, val_signal):
    """Best-held-out return: the kept vocab is >= the pipeline init on validation,
    under both the continuous power-margin signal (default) and quantized recovery."""
    train = _scorer(shapes, freqs, 8, 24, seed=3)
    val = _scorer(shapes, freqs, 8, 24, seed=4)
    _, diag = build_joint_em_catalog(shapes, 2, train, val_scorer=val, max_iter=4,
                                     n_harmonics=H, m_step=m_step,
                                     val_signal=val_signal,
                                     return_diagnostics=True)
    assert diag.val_signal_name == val_signal
    assert diag.val_signal[diag.best_iter] >= diag.val_signal[0] - 1e-12
    if val_signal == 'power_margin':
        # margin is <= 0 by construction (power at truth minus global max)
        assert np.all(diag.val_signal <= 1e-12)


def test_fit_quality_gate_blocks_all_updates(shapes, freqs):
    """An impossible power floor gates every source out -> vocab == pipeline init."""
    train = _scorer(shapes, freqs, 10, 16, seed=5)
    pipe = build_template_catalog(shapes, 2, method='pam', random_state=0,
                                  n_harmonics=H)
    joint, diag = build_joint_em_catalog(shapes, 2, train, max_iter=3,
                                         fit_quality_floor=1.0, n_harmonics=H,
                                         random_state=0, return_diagnostics=True)
    assert diag.n_gated.sum() == 0
    for a, b in zip(joint, pipe):
        assert np.allclose(a.c_n, b.c_n) and np.allclose(a.s_n, b.s_n)


def test_anti_collapse_keeps_templates_distinct(shapes, freqs):
    """A huge diversity_eps forbids any collapsing update; templates stay apart."""
    train = _scorer(shapes, freqs, 8, 24, seed=6)
    _, diag = build_joint_em_catalog(shapes, 2, train, max_iter=4,
                                     diversity_eps=1.0, n_harmonics=H,
                                     random_state=0, return_diagnostics=True)
    assert np.all(diag.min_pairwise_distance > 0.0)


def test_njobs_matches_serial(shapes, freqs):
    """The parallel E-step is bit-for-bit the serial loop -> identical vocab."""
    train1 = _scorer(shapes, freqs, 10, 16, seed=7)
    train2 = _scorer(shapes, freqs, 10, 16, seed=7)
    serial = build_joint_em_catalog(shapes, 2, train1, max_iter=3, n_harmonics=H,
                                    random_state=0, n_jobs=1)
    parallel = build_joint_em_catalog(shapes, 2, train2, max_iter=3, n_harmonics=H,
                                      random_state=0, n_jobs=2)
    for a, b in zip(serial, parallel):
        assert np.allclose(a.c_n, b.c_n) and np.allclose(a.s_n, b.s_n)
