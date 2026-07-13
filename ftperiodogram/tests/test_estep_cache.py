"""E-step sums-cache (exact) + warm-start refine (guarded) tests.

The sums cache must be BITWISE invisible (it only skips recomputing
iteration-invariant NFFT summations), and the ``estep_refine`` windowed
re-solve must leave the FINAL state (assignments, recovered frequencies,
powers, vocabulary) unchanged, with its guards (edge-escape expansion,
full-grid fallback, mandatory full-grid final iteration) exercised.

Speed note: tiny cells (short baseline, small grid, H=4) -- these are
determinism gates, not recovery-science runs.
"""
import numpy as np
import pytest

from ftperiodogram.template import Template
from ftperiodogram.validation import make_recovery_scorer, frequency_grid
from ftperiodogram.simulate import SyntheticCadence
from ftperiodogram.joint_em import (build_joint_em_catalog, _estep, _estep_one,
                                    _refine_pair_solve, _top_peak_indices)

H = 4
BASELINE = 30.0


def _shapes():
    ph = np.arange(256) / 256.0
    y = -(ph * 2 - 1)
    y[ph > 0.5] = (1 - ph[ph > 0.5]) * 2 - 1
    saw = Template.from_sampled(y - y.mean(), nharmonics=H)
    c = np.zeros(H)
    s = np.zeros(H)
    c[0], s[1] = 1.0, 0.15
    return [saw, Template(c, s)]


@pytest.fixture(scope='module')
def shapes():
    return _shapes()


@pytest.fixture(scope='module')
def freqs():
    return frequency_grid(1.0, 5.0, 128)


def _scorer(shapes, freqs, n_sources, n_epochs, seed):
    cad = SyntheticCadence(n_epochs={'g': n_epochs, 'r': n_epochs},
                           bands=['g', 'r'], baseline_days=BASELINE,
                           random_state=seed)
    return make_recovery_scorer(cad, shapes, freqs=freqs, n_sources=n_sources,
                                random_state=seed)


def _run(shapes, scorer, val, **kwargs):
    return build_joint_em_catalog(shapes, 2, scorer, val_scorer=val,
                                  n_harmonics=H, random_state=0,
                                  return_diagnostics=True, **kwargs)


# ----------------------------------------------------------------------
# FIX 1: the sums cache is bitwise invisible
# ----------------------------------------------------------------------
def test_sums_cache_bitwise_parity(shapes, freqs):
    """cache_sums on/off: every per-iteration E-step trace, the val curve and
    the returned vocabulary are BITWISE identical (zero tolerance)."""
    train = _scorer(shapes, freqs, 5, 8, seed=1)
    val = _scorer(shapes, freqs, 5, 8, seed=2)
    v_on, d_on = _run(shapes, train, val, max_iter=2, cache_sums=True)
    v_off, d_off = _run(shapes, train, val, max_iter=2, cache_sums=False)

    assert d_on.n_iter == d_off.n_iter
    assert d_on.best_iter == d_off.best_iter
    assert np.array_equal(d_on.val_signal, d_off.val_signal)
    e_on, e_off = d_on.estep, d_off.estep
    assert len(e_on['assign_hist']) == len(e_off['assign_hist'])
    for j in range(len(e_on['assign_hist'])):
        assert np.array_equal(e_on['assign_hist'][j], e_off['assign_hist'][j])
        assert np.array_equal(e_on['freq_hist'][j], e_off['freq_hist'][j])
        assert np.array_equal(e_on['power_hist'][j], e_off['power_hist'][j])
    for a, b in zip(v_on, v_off):
        assert np.array_equal(a.c_n, b.c_n) and np.array_equal(a.s_n, b.s_n)
    # the cache actually cached: every task after the first pass per source hit
    assert e_on['cache_hits'] > 0
    assert e_off['cache_hits'] == 0


def test_estep_helper_matches_per_source_loop(shapes, freqs):
    """The `_estep` seam (used by assignment_accuracy) equals the plain
    per-source `_estep_one` loop."""
    train = _scorer(shapes, freqs, 3, 8, seed=3)
    assign, freq_rec, power = _estep(train._sources, shapes, freqs,
                                     train.mode, H, 1)
    for i, s in enumerate(train._sources):
        k, f, p = _estep_one(s, shapes, freqs, train.mode, H)
        assert k == assign[i]
        assert f == freq_rec[i]
        assert p == power[i]


# ----------------------------------------------------------------------
# FIX 2: estep_refine final state + mandatory full-grid final iteration
# ----------------------------------------------------------------------
def test_refine_final_state_matches_full(shapes, freqs):
    """refine on/off: final assignments and recovered frequencies identical,
    final powers within 1e-12, vocabulary identical; at least one iteration
    actually ran windowed and the last recorded E-step is always full-grid."""
    train = _scorer(shapes, freqs, 5, 8, seed=4)
    val = _scorer(shapes, freqs, 5, 8, seed=5)
    v_off, d_off = _run(shapes, train, val, max_iter=3, estep_refine=False)
    v_on, d_on = _run(shapes, train, val, max_iter=3, estep_refine=True)

    e_on, e_off = d_on.estep, d_off.estep
    assert e_on['refine'] and not e_off['refine']
    # a middle iteration ran windowed (iteration 1 and max_iter never do)
    assert sum(e_on['windowed_pairs']) > 0
    assert any(e_on['refine_used'])
    # the mandatory guard: the last recorded E-step is never windowed
    assert not e_on['refine_used'][-1]
    assert e_on['solve_fraction'][-1] == 1.0

    a_on, a_off = e_on['assign_hist'][-1], e_off['assign_hist'][-1]
    f_on, f_off = e_on['freq_hist'][-1], e_off['freq_hist'][-1]
    p_on, p_off = e_on['power_hist'][-1], e_off['power_hist'][-1]
    assert np.array_equal(a_on, a_off)
    assert np.array_equal(f_on, f_off)
    assert np.max(np.abs(p_on - p_off)) <= 1e-12
    for a, b in zip(v_on, v_off):
        assert np.allclose(a.c_n, b.c_n, rtol=0, atol=1e-12)
        assert np.allclose(a.s_n, b.s_n, rtol=0, atol=1e-12)


def test_refine_refused_outside_floating_offsets(shapes, freqs):
    """RF-4 (WP B8e-prep): estep_refine is only verified equivalent to the exact
    E-step in mode='floating_offsets'; build_joint_em_catalog must REFUSE it in
    the other modes rather than silently converge to a divergent vocabulary."""
    for mode in ('shared_phase', 'independent'):
        cad = SyntheticCadence(n_epochs={'g': 8, 'r': 8}, bands=['g', 'r'],
                               baseline_days=BASELINE, random_state=11)
        sc = make_recovery_scorer(cad, shapes, freqs=freqs, n_sources=4,
                                  mode=mode, random_state=11)
        with pytest.raises(ValueError, match='floating_offsets'):
            build_joint_em_catalog(shapes, 2, sc, n_harmonics=H,
                                   random_state=0, estep_refine=True)
        # the exact path (refine off) must still run in these modes
        build_joint_em_catalog(shapes, 2, sc, n_harmonics=H, random_state=0,
                               max_iter=2, estep_refine=False)


def test_refine_early_stop_reruns_full_grid(shapes, freqs):
    """An early stop on a windowed iteration re-runs that E-step full-grid
    (same input bank), so the last recorded E-step is never windowed and
    matches the refine-off run's final E-step exactly."""
    train = _scorer(shapes, freqs, 5, 8, seed=6)
    val = _scorer(shapes, freqs, 5, 8, seed=7)
    kwargs = dict(max_iter=6, patience=2)
    v_off, d_off = _run(shapes, train, val, estep_refine=False, **kwargs)
    v_on, d_on = _run(shapes, train, val, estep_refine=True, **kwargs)
    e_on = d_on.estep
    assert not e_on['refine_used'][-1]
    assert e_on['solve_fraction'][-1] == 1.0
    if d_on.stop_reason == 'early_stop' and len(e_on['refine_used']) >= 2 \
            and e_on['refine_used'][-2]:
        assert e_on['final_full_grid_rerun']
    assert d_on.n_iter == d_off.n_iter
    assert np.array_equal(e_on['assign_hist'][-1], d_off.estep['assign_hist'][-1])
    assert np.array_equal(e_on['freq_hist'][-1], d_off.estep['freq_hist'][-1])
    assert np.max(np.abs(e_on['power_hist'][-1]
                         - d_off.estep['power_hist'][-1])) <= 1e-12
    for a, b in zip(v_on, v_off):
        assert np.allclose(a.c_n, b.c_n, rtol=0, atol=1e-12)
        assert np.allclose(a.s_n, b.s_n, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------
# Refine guard mechanics (unit level, injectable solve_fn)
# ----------------------------------------------------------------------
def _quadratic_solver(true_g, scale=1000.0):
    def solve_fn(idx):
        return -((np.asarray(idx) - true_g) / scale) ** 2
    return solve_fn


def test_refine_pair_solve_window_hit(freqs):
    """Peak inside the warm-start window: no escape, tiny solve fraction."""
    g, p, peaks, diag = _refine_pair_solve(freqs, BASELINE, np.array([100]),
                                           100, _quadratic_solver(100))
    assert g == 100
    assert diag['edge_escape'] == 0 and diag['fallback'] == 0
    assert diag['n_solved'] < freqs.size
    assert peaks[0] == 100


def test_refine_pair_solve_edge_escape(freqs):
    """Argmax on the window edge: the window expands once (x3) and the true
    peak is found without a full-grid solve."""
    # half-width 3/T = 0.1 -> 5 grid points at df ~ 0.0201; put the previous
    # peak ~2 half-widths off the true one so the first argmax hits the edge
    g, p, peaks, diag = _refine_pair_solve(freqs, BASELINE, np.array([110]),
                                           110, _quadratic_solver(100))
    assert diag['edge_escape'] == 1
    assert diag['fallback'] == 0
    assert g == 100
    assert diag['n_solved'] < freqs.size


def test_refine_pair_solve_double_edge_falls_back_to_full_grid(freqs):
    """A second edge landing falls back to the exact full grid for the pair."""
    # true peak far from every warm-start window (including the f/2 window and
    # its one-shot expansion), so the argmax walks onto an edge twice
    g, p, peaks, diag = _refine_pair_solve(freqs, BASELINE, np.array([110]),
                                           110, _quadratic_solver(5))
    assert diag['edge_escape'] == 1
    assert diag['fallback'] == 1
    assert g == 5                                    # full grid found the truth
    assert diag['n_solved'] >= freqs.size


def test_refine_pair_solve_degenerate_inputs_full_grid(freqs):
    """No usable warm start (T <= 0 or no peaks) solves the full grid."""
    g, p, peaks, diag = _refine_pair_solve(freqs, 0.0, np.array([50]), 50,
                                           _quadratic_solver(30))
    assert diag['fallback'] == 1 and g == 30
    g, p, peaks, diag = _refine_pair_solve(freqs, BASELINE,
                                           np.array([], dtype=int), 0,
                                           _quadratic_solver(30))
    assert diag['fallback'] == 1 and g == 30


def test_top_peak_indices_segmented():
    """Local maxima across non-contiguous solved segments, power-ordered."""
    idx = np.array([0, 1, 2, 10, 11, 12])
    p = np.array([0.0, 1.0, 0.0, 5.0, 1.0, 4.0])
    peaks = _top_peak_indices(idx, p, 4)
    assert list(peaks) == [10, 12, 1]                # segment ends qualify
    assert list(_top_peak_indices(idx, p, 1)) == [10]
