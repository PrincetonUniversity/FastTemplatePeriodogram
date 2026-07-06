"""Equivalence tests for the vectorized coefficient-assembly path (WP C1).

``method='batched'`` assembles the YM/MM/stationarity polynomial coefficients
as stacked arrays over frequency chunks and must reproduce the per-frequency
reference path (``method='eigvals'``) with no behavior change. The acceptance
gate: powers agree to <= 1e-13 over H in {1, 2, 3, 5, 8, 10} x 5 seeds x
N in {30, 300}.
"""
import numpy as np
import pytest

from ..core import (YM_MM_from_sums, batched_YM_MM_from_sums,
                    batched_stationarity_coefs, template_periodogram)
from ..modeler import (FastTemplatePeriodogram, FastMultiTemplatePeriodogram,
                       TemplateModel)
from ..summations import (direct_summations, fast_summations,
                          fast_summations_batched, stack_summations)
from ..template import Template
from ..utils import Summations, weights

GATE_TOL = 1e-13
GATE_HARMONICS = [1, 2, 3, 5, 8, 10]
GATE_SEEDS = [0, 1, 2, 3, 4]
GATE_NDATA = [30, 300]


def _simulate(H, N, seed, nf=64):
    """Random template + noisy template signal + a regular frequency grid."""
    rng = np.random.RandomState(seed)
    template = Template(rng.randn(H), rng.randn(H))

    T = 10.0
    period = 0.77
    t = np.sort(T * rng.rand(N))
    dy = 0.05 * (1 + rng.rand(N))
    y = 1.5 * template((t / period) % 1.0) + dy * rng.randn(N)

    df = 0.05
    freqs = df * (10 + np.arange(nf))   # dnf = 10; spans ~0.5 - 3.65 c/d
    return t, y, dy, template, freqs


@pytest.mark.parametrize('N', GATE_NDATA)
@pytest.mark.parametrize('H', GATE_HARMONICS)
def test_gate_batched_power_matches_eigvals(H, N):
    """WP C1 acceptance gate (NFFT path): max |P_batched - P_eigvals| <= 1e-13."""
    worst = 0.0
    for seed in GATE_SEEDS:
        t, y, dy, template, freqs = _simulate(H, N, seed)
        p_ref, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                        freqs, fast=True, method='eigvals')
        p_bat, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                        freqs, fast=True, method='batched')
        worst = max(worst, float(np.max(np.abs(p_ref - p_bat))))
    assert worst <= GATE_TOL


@pytest.mark.parametrize('H', [1, 3, 8])
def test_batched_power_matches_eigvals_direct_sums(H):
    """Same equivalence on the direct (fast=False) summation path."""
    for seed in (0, 1):
        t, y, dy, template, freqs = _simulate(H, 30, seed)
        p_ref, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                        freqs, fast=False, method='eigvals')
        p_bat, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                        freqs, fast=False, method='batched')
        assert float(np.max(np.abs(p_ref - p_bat))) <= GATE_TOL


def test_chunk_size_does_not_change_powers():
    """Chunking is pure bookkeeping: results are identical for any chunk_size."""
    t, y, dy, template, freqs = _simulate(5, 30, 3, nf=100)
    p_one, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                    freqs, method='batched', chunk_size=10000)
    p_many, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                     freqs, method='batched', chunk_size=16)
    assert np.array_equal(p_one, p_many)


def test_batched_with_user_provided_summations():
    """Precomputed per-frequency summations flow through the batched path."""
    t, y, dy, template, freqs = _simulate(3, 30, 4)
    w = weights(dy)
    sums = direct_summations(t, y, w, freqs, len(template.c_n))
    p_ref, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                    freqs, summations=sums, method='eigvals')
    p_bat, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                    freqs, summations=sums, method='batched')
    assert float(np.max(np.abs(p_ref - p_bat))) <= GATE_TOL


def test_unknown_method_raises():
    t, y, dy, template, freqs = _simulate(2, 30, 0)
    with pytest.raises(ValueError):
        template_periodogram(t, y, dy, template.c_n, template.s_n, freqs,
                             method='nope')


def test_batched_coefficients_match_per_frequency_assembly():
    """The stacked YM/MM/stationarity coefficients equal the per-frequency
    polynomial-arithmetic results (up to the dropped analytically-zero top
    stationarity coefficient)."""
    H = 5
    t, y, dy, template, freqs = _simulate(H, 30, 7, nf=8)
    w = weights(dy)
    sums_list = direct_summations(t, y, w, freqs, H)
    stacked = stack_summations(sums_list)

    YM_c, MM_c, AC = batched_YM_MM_from_sums(template.c_n, template.s_n,
                                             stacked)
    p_c = batched_stationarity_coefs(YM_c, MM_c)
    assert YM_c.shape == (len(freqs), 2 * H + 1)
    assert MM_c.shape == (len(freqs), 4 * H + 1)
    assert AC.shape == (len(freqs), H)
    assert p_c.shape == (len(freqs), 6 * H - 1)

    for i, sums in enumerate(sums_list):
        YM, MM, AC_i = YM_MM_from_sums(template.c_n, template.s_n, sums)
        np.testing.assert_allclose(YM_c[i, :len(YM.coef)], YM.coef,
                                   rtol=1e-13, atol=0)
        np.testing.assert_allclose(MM_c[i, :len(MM.coef)], MM.coef,
                                   rtol=1e-13, atol=0)
        np.testing.assert_allclose(AC[i], AC_i, rtol=1e-13, atol=0)

        # pad both stationarity forms to the nominal length 6H and compare;
        # the per-frequency form may retain the analytically-zero (residue)
        # top coefficient that the batched form builds and slices off
        p_ref = 2 * MM * YM.deriv() - MM.deriv() * YM
        ref = np.zeros(6 * H, dtype=np.complex128)
        ref[:len(p_ref.coef)] = p_ref.coef
        bat = np.zeros(6 * H, dtype=np.complex128)
        bat[:p_c.shape[1]] = p_c[i]
        scale = np.max(np.abs(ref))
        assert np.max(np.abs(ref - bat)) <= 1e-12 * scale


def test_modeler_autopower_batched_matches():
    """method='batched' is reachable from FastTemplatePeriodogram.autopower."""
    t, y, dy, template, _ = _simulate(3, 30, 11)
    ftp = FastTemplatePeriodogram(template=template).fit(t, y, dy)
    kw = dict(minimum_frequency=0.5, maximum_frequency=3.0,
              samples_per_peak=3)
    f_ref, p_ref = ftp.autopower(**kw, method='eigvals')
    f_bat, p_bat = ftp.autopower(method='batched', **kw)
    assert np.array_equal(f_ref, f_bat)
    assert float(np.max(np.abs(p_ref - p_bat))) <= GATE_TOL


def test_multi_template_autopower_batched_matches():
    """method='batched' is reachable from FastMultiTemplatePeriodogram."""
    t, y, dy, _, _ = _simulate(3, 30, 13)
    rng = np.random.RandomState(99)
    templates = [Template(rng.randn(3), rng.randn(3)) for _ in range(2)]
    ftp = FastMultiTemplatePeriodogram(templates=templates).fit(t, y, dy)
    kw = dict(minimum_frequency=0.5, maximum_frequency=3.0,
              samples_per_peak=3)
    f_ref, p_ref = ftp.autopower(**kw, method='eigvals')
    f_bat, p_bat = ftp.autopower(method='batched', **kw)
    assert np.array_equal(f_ref, f_bat)
    assert float(np.max(np.abs(p_ref - p_bat))) <= GATE_TOL


@pytest.mark.parametrize('H', [2, 3, 5, 8])
def test_batched_params_match_eigvals(H):
    """best_fit_params agree between methods (the power gate alone is blind
    to the C/S -> AC -> theta_3 reconstruction). H = 1 is excluded: its
    exactly-tied +/-phi root pair is a parametrization gauge, covered by the
    fitted-curve test below."""
    t, y, dy, template, freqs = _simulate(H, 30, 5)
    _, prm_ref = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                      freqs, method='eigvals')
    _, prm_bat = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                      freqs, method='batched')
    ref = np.array([[p.a, p.b, p.c, p.sgn] for p in prm_ref])
    bat = np.array([[p.a, p.b, p.c, p.sgn] for p in prm_bat])
    assert float(np.max(np.abs(ref - bat))) <= 1e-9


def test_batched_h1_fitted_curves_match():
    """H = 1: tied phi / -phi power maxima mean (a, b, sgn) may flip between
    methods (same curve, same power); assert gauge-invariant equivalence of
    the fitted models themselves."""
    t, y, dy, template, freqs = _simulate(1, 30, 5)
    _, prm_ref = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                      freqs, method='eigvals')
    _, prm_bat = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                      freqs, method='batched')
    t_dense = np.linspace(t.min(), t.max(), 500)
    for freq, p_r, p_b in zip(freqs, prm_ref, prm_bat):
        y_r = TemplateModel(template, frequency=freq, parameters=p_r)(t_dense)
        y_b = TemplateModel(template, frequency=freq, parameters=p_b)(t_dense)
        assert float(np.max(np.abs(y_r - y_b))) <= 1e-9


def test_nfft_batched_summations_bitwise_match_per_frequency():
    """Every field of fast_summations_batched (including C/S, which feed only
    the offset reconstruction and are invisible to the power gates) is
    bitwise equal to the stacked per-frequency fast_summations."""
    H = 5
    t, y, dy, template, freqs = _simulate(H, 100, 9, nf=300)
    w = weights(dy)
    ref = stack_summations(fast_summations(t, y, w, freqs, H))
    chunks = list(fast_summations_batched(t, y, w, freqs, H, chunk_size=128))
    for field in Summations._fields:
        bat = np.concatenate([getattr(c, field) for c in chunks], axis=0)
        assert np.array_equal(getattr(ref, field), bat), field


@pytest.mark.parametrize('bad', [-5, 0, 2.5, True])
def test_invalid_chunk_size_raises(bad):
    """A negative chunk_size once emptied the chunk loop and silently
    returned a zero-length periodogram; all invalid values now raise."""
    t, y, dy, template, freqs = _simulate(2, 30, 0)
    with pytest.raises(ValueError):
        template_periodogram(t, y, dy, template.c_n, template.s_n, freqs,
                             method='batched', chunk_size=bad)
    w = weights(dy)
    with pytest.raises(ValueError):
        fast_summations_batched(t, y, w, freqs, len(template.c_n),
                                chunk_size=bad)


def test_batched_summations_length_mismatch_raises():
    """len(summations) != len(freqs) once truncated the output at a chunk
    boundary, silently and chunk_size-dependently; it now raises."""
    t, y, dy, template, freqs = _simulate(3, 30, 4)
    w = weights(dy)
    sums = direct_summations(t, y, w, freqs, len(template.c_n))
    for bad in (list(sums)[:-3], list(sums) + [sums[-1]]):
        with pytest.raises(ValueError):
            template_periodogram(t, y, dy, template.c_n, template.s_n, freqs,
                                 summations=bad, method='batched',
                                 chunk_size=16)


def test_chunk_size_bitwise_invariant_large_chunks():
    """Chunks past the ufunc-layout crossover (~1024 rows) once drifted by
    1 ulp via np.trace's layout-dependent reduction order; the stacks are now
    materialized C-contiguous, so any chunk size is bitwise identical."""
    t, y, dy, template, freqs = _simulate(5, 100, 9, nf=1999)
    ps = [template_periodogram(t, y, dy, template.c_n, template.s_n, freqs,
                               method='batched', chunk_size=cs)[0]
          for cs in (1, 256, 1365, 1999, 4096)]
    for p in ps[1:]:
        assert np.array_equal(ps[0], p)


def test_batched_empty_freqs_returns_empty():
    """Pinned: the batched path returns an empty periodogram for an empty
    grid (the per-frequency path raises on this out-of-contract input)."""
    t, y, dy, template, _ = _simulate(2, 30, 0)
    p, prm = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                  np.array([]), fast=False, method='batched')
    assert len(p) == 0 and prm == []
