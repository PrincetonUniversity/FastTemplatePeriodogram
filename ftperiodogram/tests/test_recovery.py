"""Tests for the period-recovery metrics (ftperiodogram.recovery).

Pure-function tests: boundary behaviour of the two recovery conventions, the
configurable alias set, the exact-vs-harmonic split, and population aggregation.
All instant and seed-free (the functions are deterministic).
"""
import numpy as np
import numpy.testing as npt
import pytest

from ftperiodogram import recovery as rec


# ----------------------------------------------------------------------
# (a) hard fractional threshold
# ----------------------------------------------------------------------
def test_fractional_threshold_boundary():
    P_true = 0.5
    # rtol=0.01 -> half-width 0.005 in period
    assert rec.recovered_fractional(0.5 + 0.0049, P_true) is True
    assert rec.recovered_fractional(0.5 - 0.0049, P_true) is True
    assert rec.recovered_fractional(0.5 + 0.0051, P_true) is False
    # strictly-less-than: exactly on the boundary is NOT recovered
    assert rec.recovered_fractional(P_true * 1.01, P_true) is False
    # non-positive recovered period is never a match
    assert rec.recovered_fractional(-0.5, P_true) is False
    assert rec.recovered_fractional(0.0, P_true) is False


def test_fractional_vectorized():
    P_true = 0.5
    out = rec.recovered_fractional(np.array([0.5, 0.55, 0.5004, -1.0]), P_true)
    npt.assert_array_equal(out, [True, False, True, False])


# ----------------------------------------------------------------------
# (b) baseline-aware phase coherence
# ----------------------------------------------------------------------
def test_phase_coherence_boundary():
    P_true, T, dphi = 0.5, 1000.0, 0.5
    f_true = 1.0 / P_true                      # 2.0 cycles/day
    df_thresh = dphi / T                        # 5e-4
    # build P_rec with a chosen frequency error just inside / outside threshold
    inside = 1.0 / (f_true + 0.8 * df_thresh)
    outside = 1.0 / (f_true + 1.2 * df_thresh)
    assert rec.recovered_phase_coherence(inside, P_true, T, dphi) is True
    assert rec.recovered_phase_coherence(outside, P_true, T, dphi) is False


def test_phase_coherence_is_stricter_than_fractional_for_long_baseline():
    # a 1% period error passes the fractional test but, over a long baseline,
    # accumulates ~20 cycles of drift -> fails phase coherence.
    P_true, T = 0.5, 1000.0
    P_rec = P_true * 1.005                       # 0.5% error, inside 1%
    assert rec.recovered_fractional(P_rec, P_true) is True
    assert rec.recovered_phase_coherence(P_rec, P_true, T, 0.5) is False


def test_phase_coherence_requires_baseline():
    with pytest.raises(ValueError):
        rec.classify_recovery(0.5, 0.5, baseline=None,
                              criterion='phase_coherence')


# ----------------------------------------------------------------------
# alias set
# ----------------------------------------------------------------------
def test_harmonic_alias_set_contents():
    P_true = 0.5
    aliases = rec.harmonic_alias_set(P_true)
    by_name = {a.name: a for a in aliases}
    # the four harmonic multiples, with correct alias periods
    npt.assert_allclose(by_name['P/2'].P_alias, 0.25)
    npt.assert_allclose(by_name['2P'].P_alias, 1.0)
    npt.assert_allclose(by_name['P/3'].P_alias, 0.5 / 3.0)
    npt.assert_allclose(by_name['3P'].P_alias, 1.5)
    # 1-day beats: f_true = 2/day -> +1/day => P=1/3, -1/day => P=1
    npt.assert_allclose(by_name['beat_+1/day'].P_alias, 1.0 / 3.0)
    npt.assert_allclose(by_name['beat_-1/day'].P_alias, 1.0)
    # -2/day drives f_true to exactly 0 -> dropped (non-positive frequency)
    assert 'beat_-2/day' not in by_name
    assert 'beat_+2/day' in by_name
    # 1-year beat present and close to P_true (tiny frequency offset)
    npt.assert_allclose(by_name['beat_+1/year'].P_alias,
                        1.0 / (2.0 + 1.0 / 365.25), rtol=1e-9)


# ----------------------------------------------------------------------
# classify_recovery: exact vs exact-or-harmonic
# ----------------------------------------------------------------------
def test_classify_exact():
    r = rec.classify_recovery(0.5004, 0.5)
    assert r.exact and r.exact_or_harmonic and r.recovered
    assert r.alias_name is None
    assert r.criterion == 'fractional'


def test_classify_harmonic_double_period():
    r = rec.classify_recovery(1.0, 0.5)          # P_rec = 2 P_true
    assert r.exact is False
    assert r.exact_or_harmonic is True
    assert r.alias_name == '2P'
    assert r.recovered is True                   # count_harmonics default True
    # turning harmonics off demotes it to a miss
    r2 = rec.classify_recovery(1.0, 0.5, count_harmonics=False)
    assert r2.recovered is False
    assert r2.exact_or_harmonic is True          # still reported on the second curve


def test_classify_miss():
    r = rec.classify_recovery(0.37, 0.5)         # not exact, not a listed alias
    assert not r.exact and not r.exact_or_harmonic and not r.recovered
    assert r.alias_name is None


def test_classify_phase_coherence_criterion():
    r = rec.classify_recovery(0.5 * 1.005, 0.5, baseline=1000.0,
                              criterion='phase_coherence', delta_phi_max=0.5)
    assert r.criterion == 'phase_coherence'
    assert r.exact is False                       # 0.5% error fails over T=1000


# ----------------------------------------------------------------------
# population aggregation
# ----------------------------------------------------------------------
def test_recovery_rate_aggregation():
    results = [
        rec.classify_recovery(0.5004, 0.5),   # exact
        rec.classify_recovery(0.5002, 0.5),   # exact
        rec.classify_recovery(1.0, 0.5),      # harmonic 2P
        rec.classify_recovery(0.37, 0.5),     # miss
    ]
    summary = rec.recovery_rate(results)
    assert summary['n'] == 4
    npt.assert_allclose(summary['exact'], 0.5)
    npt.assert_allclose(summary['exact_or_harmonic'], 0.75)
    npt.assert_allclose(summary['recovered'], 0.75)
    assert summary['alias_breakdown'] == {'2P': 1}


def test_recovery_rate_empty():
    summary = rec.recovery_rate([])
    assert summary['n'] == 0
    assert summary['recovered'] == 0.0
    assert summary['alias_breakdown'] == {}
