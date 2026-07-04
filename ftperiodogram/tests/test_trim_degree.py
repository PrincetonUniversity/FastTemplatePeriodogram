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
    monkeypatch.setattr(core, "trim_zero_leading_coef",
                        lambda p, nominal_degree=None: p)


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
        trimmed = core.trim_zero_leading_coef(p, nominal_degree=6 * H - 1)
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


# ----------------------------------------------------------------------
# C3 finding FO-TRIM-1 (VERIFICATION.md WP C3 / C3.5): when the analytic
# degree-(6H-1) zero cancels EXACTLY in floating point, numpy's arithmetic
# removes it before the conditional trim runs -- the length-blind trim then
# ate the GENUINE degree-(6H-2) coefficient whenever deep-|MM|-dip
# conditioning pushed it under the 1e-9 residue threshold, displacing the
# true stationary root (adjudicated in 50-digit arithmetic: the full exact
# polynomial has an on-circle root at the true optimum; the truncated one
# does not). The length gate must refuse that second trim.
# ----------------------------------------------------------------------
def test_trim_length_gate_semantics():
    """Unit semantics of the nominal_degree gate."""
    rng = np.random.RandomState(3)
    genuine = rng.randn(47) + 1j * rng.randn(47)
    genuine[-1] = 1e-12 * np.max(np.abs(genuine))   # tiny but GENUINE
    import numpy.polynomial as pol
    # len 47 = degree 46 = 6H-2 at H=8: the exactly-cancelled analytic zero
    # (nominal degree 6H-1 = 47) is already gone
    p_pretrimmed = pol.Polynomial(genuine)

    # numpy already removed the analytic zero (len != nominal_degree+1=48):
    # the gate must refuse to trim, keeping the genuine coefficient
    out = core.trim_zero_leading_coef(p_pretrimmed, nominal_degree=47)
    assert len(out.coef) == 47

    # at full nominal length the top coefficient is the analytic-zero
    # residue: trimmed exactly as before
    p_full = pol.Polynomial(np.concatenate([genuine,
                                            [1e-19 * np.max(np.abs(genuine))]]))
    out = core.trim_zero_leading_coef(p_full, nominal_degree=47)
    assert len(out.coef) == 47

    # legacy length-blind behavior preserved when no nominal degree given
    out = core.trim_zero_leading_coef(p_pretrimmed)
    assert len(out.coef) == 46


# F2-adjudicated fixture (C3 verification, M1 seed 70001 @ f=0.11288...):
# K=3 floating_offsets, H=8 eclipse template, clumped sparse sampling. On
# this data the assembled stationarity polynomial arrives with the analytic
# zero already exactly cancelled (len == 6H-1 == 47) and its genuine leading
# coefficient at 0.947x the residue threshold -- the length-blind trim ate
# it and cost 3.93e-8 of power (real in exact arithmetic).
_FOTRIM_T = np.array([
    29.30197587230406, 2.4463261933017746, 29.290051014089332,
    2.4604726101961836, 29.266554064623055, 29.291034792723103,
    24.79979285303346, 9.829617486142451, 2.5565214129945124,
    24.802188025933187, 2.556510872171678, 24.80327929963633,
    2.498577106270781, 19.931124077932996, 19.8975606798697,
    19.91528073104593, 2.428870165224642, 2.475052911312861])
_FOTRIM_Y = np.array([
    12.74805894888122, 12.84813335007487, 12.417580711358431,
    13.103482012101002, 12.373072696988372, 12.578217091620015,
    20.272783246631395, 20.03177189239782, 20.203319512677172,
    20.213014675908106, 20.245263415412747, 20.162737363879735,
    15.661798017875602, 15.903007240062152, 15.680506317507504,
    15.030811957574787, 15.714708357541832, 15.707097846521485])
_FOTRIM_DY = np.array([
    0.06126301144469792, 0.27493945866977615, 0.261013888877076,
    0.10248847827913099, 0.1799079796257244, 0.05527251821349683,
    0.0829208991199589, 0.27511403872530465, 0.079803287516807,
    0.08818648537168085, 0.21966194986018647, 0.0702212086493375,
    0.0737858325558821, 0.14554304179502048, 0.14172469436041082,
    0.42969145401166325, 0.07891985902872689, 0.2623836824485987])
_FOTRIM_BANDS = np.array(['g'] * 6 + ['r'] * 6 + ['i'] * 6)
_FOTRIM_CN = np.array([
    -0.6684927784987966, -0.3285168637518286, -0.05742884074192249,
    0.04579459697081215, 0.04360195922050869, 0.01931809520967799,
    0.00538008096558094, 0.00097151594952111])
_FOTRIM_SN = np.array([
    0.33546874279810057, 0.4407007580716917, 0.3239231623015995,
    0.15366181754779662, 0.04635994920116317, 0.00707216459471417,
    -0.00061694215763588, -0.00063551036976203])
_FOTRIM_FREQ = 0.11287971528174655


def test_trim_gate_keeps_genuine_coef_on_adjudicated_fixture():
    """End-to-end pin on the F2-adjudicated floating_offsets case: the
    assembled polynomial arrives pre-cancelled at len 6H-1 with a genuine
    sub-threshold leading coefficient; the production path must keep it,
    and keeping it is worth +3.93e-8 of power over the old trim."""
    import numpy.polynomial as pol
    from ftperiodogram.multiband import (compute_band_summations,
                                         combine_band_summations,
                                         _per_band_YM_MM)
    H = 8
    tmpl = Template(_FOTRIM_CN, _FOTRIM_SN)
    freqs = np.array([_FOTRIM_FREQ])
    sl, stats = compute_band_summations(
        _FOTRIM_T, _FOTRIM_Y, _FOTRIM_BANDS, freqs, H, dy=_FOTRIM_DY,
        mode='floating_offsets', fast=False)
    pb = _per_band_YM_MM({b: tmpl for b in stats.bands},
                         {b: sl[b][0] for b in stats.bands}, stats.bands)
    YM, MM, AC = combine_band_summations(pb, stats.W, stats.ybar,
                                         stats.ybar_global,
                                         'floating_offsets')
    p_un = 2 * MM * YM.deriv() - MM.deriv() * YM
    scale = np.max(np.abs(p_un.coef))
    # the trap: pre-cancelled length AND a genuine sub-threshold top coef
    assert len(p_un.coef) == 6 * H - 1
    assert np.abs(p_un.coef[-1]) <= 1e-9 * scale
    # the gate refuses the second trim
    kept = core.trim_zero_leading_coef(p_un, nominal_degree=6 * H - 1)
    assert len(kept.coef) == 6 * H - 1

    # and the production solve now attains the higher power
    _, pw_fixed, _ = core.roots_from_YM_MM(
        YM, MM, AC, H, stats.ybar_global, stats.YY_combined,
        positive_amplitude=True)
    _, pw_oldtrim, _ = core.roots_from_YM_MM(
        YM, MM, AC, H, stats.ybar_global, stats.YY_combined,
        positive_amplitude=True,
        stationarity=pol.Polynomial(p_un.coef[:-1]))
    assert pw_fixed - pw_oldtrim >= 1e-8      # measured gain 3.93e-8
    assert pw_fixed >= pw_oldtrim             # one-sided by construction
