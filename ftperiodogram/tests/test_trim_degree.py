"""Regression tests for the analytically-zero leading coefficient trim
(WP A1): the stationarity polynomial ``2 MM YM' - MM' YM`` has true degree
at most ``6H - 2`` (the nominal ``6H - 1`` coefficient cancels exactly), and
the multiband shared-phase ``G`` at most ``8HK - 2``. Leaving the FP residue
in place injects a spurious huge-modulus companion root.
"""
import numpy as np
import pytest

from ftperiodogram import core, summations
from ftperiodogram.modeler import FastTemplatePeriodogram
from ftperiodogram.multiband import FastMultibandTemplatePeriodogram
from ftperiodogram.template import Template
from ftperiodogram.utils import weights

HARMONICS = [1, 2, 3, 5, 8]
SEEDS = [0, 1, 2]
FREQS = np.linspace(0.05, 0.5, 80)


def _template(H, seed=0):
    rng = np.random.RandomState(seed)
    c_n, s_n = rng.randn(2, H) / np.arange(1, H + 1)
    return Template(c_n, s_n)


def _data(template, seed, N=50, freq=0.123):
    rng = np.random.RandomState(seed + 100)
    t = np.sort(rng.rand(N) * 100)
    y = 10 + 1.5 * template((freq * t) % 1.0) + 0.1 * rng.randn(N)
    dy = np.full_like(t, 0.1)
    return t, y, dy


def _untrimmed(monkeypatch):
    # both the single-band path and the multiband G look the helper up in
    # core's namespace at call time, so one setattr disables both trims
    monkeypatch.setattr(core, "trim_zero_leading_coef", lambda p: p)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("H", HARMONICS)
def test_trim_matches_untrimmed_power(H, seed, monkeypatch):
    template = _template(H, seed)
    t, y, dy = _data(template, seed)

    p_trim = FastTemplatePeriodogram(template=template).fit(t, y, dy).power(FREQS)
    _untrimmed(monkeypatch)
    p_ref = FastTemplatePeriodogram(template=template).fit(t, y, dy).power(FREQS)

    assert np.max(np.abs(p_trim - p_ref)) <= 1e-13


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("H", HARMONICS)
def test_no_huge_candidate_roots_after_trim(H, seed):
    # When the top-term cancellation is exact in FP, numpy's polynomial
    # arithmetic already dropped the zero coefficient; when a residue
    # survives (len == 6H), the conditional trim must remove exactly that
    # coefficient. Either way no candidate root may be huge afterwards.
    template = _template(H, seed)
    t, y, dy = _data(template, seed)
    w = weights(dy)
    sums = summations.direct_summations(t, y, w, FREQS[::8], H)

    n_residue = 0
    for s in sums:
        YM, MM, _ = core.YM_MM_from_sums(template.c_n, template.s_n, s)
        p = 2 * MM * YM.deriv() - MM.deriv() * YM
        assert len(p.coef) <= 6 * H            # true degree <= 6H - 2 + residue
        trimmed = core.trim_zero_leading_coef(p)
        if len(p.coef) == 6 * H:
            n_residue += 1
            assert len(trimmed.coef) == 6 * H - 1
            # the untrimmed residue injects a spurious huge companion root
            untrimmed_roots = p.roots()
            assert np.max(np.absolute(untrimmed_roots)) > 1e3
        roots = trimmed.roots()
        roots = roots[np.absolute(roots) > 0]
        assert np.max(np.absolute(roots)) < 1e3

    if H in (3, 5):
        # pin non-vacuity: these H are known to leave the FP residue on
        # this fixture set (AUDIT_2026-06-10; H in {1,2,8} cancel exactly)
        assert n_residue > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_multiband_shared_phase_trim_matches_untrimmed(seed, monkeypatch):
    H = 3
    template = _template(H, seed)
    rng = np.random.RandomState(seed + 7)
    t, y, bands, dy = [], [], [], []
    for k in range(2):
        tk = np.sort(rng.rand(40) * 80)
        yk = (10 + k + 1.5 * template((0.123 * tk) % 1.0)
              + 0.1 * rng.randn(40))
        t.append(tk)
        y.append(yk)
        bands.append(np.full(40, k))
        dy.append(np.full(40, 0.1))
    t, y = np.concatenate(t), np.concatenate(y)
    bands, dy = np.concatenate(bands), np.concatenate(dy)

    def _power():
        model = FastMultibandTemplatePeriodogram(template, mode="shared_phase")
        return model.fit(t, y, bands, dy).power(FREQS[::4])

    p_trim = _power()
    _untrimmed(monkeypatch)
    p_ref = _power()

    assert np.max(np.abs(p_trim - p_ref)) <= 1e-13
