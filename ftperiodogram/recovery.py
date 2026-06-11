"""Period-recovery metrics for the controlled-simulation harness (Phase 3.2).

Two recovery conventions, reported BOTH (design memo ``phase3_vocab_design.md`` S2):

  (a) hard fractional threshold      ``|P_rec - P_true| / P_true < rtol``  (rtol=0.01;
      DES/ZTF RRL convention, Stringer et al. 2019 "89% within 1%");
  (b) baseline-aware phase coherence ``|df| * T < delta_phi_max``          (Graham et al.
      2013) with ``df = |1/P_rec - 1/P_true|`` and ``T`` the observing baseline.

Each is scored EXACT (against ``P_true``) and EXACT-OR-HARMONIC (also against a
configurable set of period aliases -- ``P/2, 2P, P/3, 3P`` and the 1-day / 1-year
window beats, VanderPlas 2018), with the matching alias *named* for diagnostics.

The fractional criterion is the headline that drives the recovery-vs-K curve and
the greedy selector's objective; phase coherence is the strict secondary curve
(``delta_phi_max=0.5`` = a half-cycle of accumulated drift over the baseline, i.e.
the Rayleigh spectral-window main-lobe half-width).

Pure numpy -- no FTP coupling -- so the metrics are unit-testable in isolation.
"""
from collections import namedtuple

import numpy as np


# ----------------------------------------------------------------------
# The two recovery conventions (each a pure boolean test)
# ----------------------------------------------------------------------
def recovered_fractional(P_rec, P_true, rtol=0.01):
    """Hard fractional threshold: ``|P_rec - P_true| / P_true < rtol``.

    Non-positive ``P_rec`` is never recovered.  Accepts scalar or ``array_like``
    ``P_rec`` and returns a matching Python ``bool`` (scalar input) or numpy
    ``bool`` array.
    """
    P_rec = np.asarray(P_rec, dtype=float)
    P_true = np.asarray(P_true, dtype=float)
    ok = (np.abs(P_rec - P_true) < rtol * np.abs(P_true)) & (P_rec > 0)
    return ok if ok.ndim else bool(ok)


def recovered_phase_coherence(P_rec, P_true, baseline, delta_phi_max=0.5):
    """Baseline-aware phase coherence: ``|df| * T < delta_phi_max``.

    ``df = |1/P_rec - 1/P_true|`` is the frequency error and ``T = baseline`` the
    observing time span (``t.max() - t.min()``).  The product is the total phase
    drift, in cycles, accumulated across the baseline.  Non-positive ``P_rec`` is
    never recovered.  Scalar/array semantics match :func:`recovered_fractional`.
    """
    P_rec = np.asarray(P_rec, dtype=float)
    P_true = np.asarray(P_true, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        delta_f = np.abs(1.0 / P_rec - 1.0 / P_true)
    ok = (delta_f * baseline < delta_phi_max) & (P_rec > 0)
    return ok if ok.ndim else bool(ok)


# ----------------------------------------------------------------------
# Configurable harmonic / window-beat alias set
# ----------------------------------------------------------------------
#: One candidate alias period whose recovery counts as exact-or-harmonic success.
AliasMatch = namedtuple('AliasMatch', ['name', 'P_alias', 'multiplier'])


def harmonic_alias_set(P_true, baseline=None, harmonics=(0.5, 2.0, 1.0 / 3.0, 3.0),
                       day_beats=(1, 2), year_beats=(1,), day=1.0, year=365.25):
    """Configurable period-alias candidate set for ``P_true``.

    Returns a list of :class:`AliasMatch`:

      * harmonic multiples ``P_true * m`` for ``m`` in ``harmonics`` -> P/2, 2P, P/3, 3P;
      * 1-day window beats ``1 / (f_true +/- n/day)``  for ``n`` in ``day_beats``;
      * 1-year window beats ``1 / (f_true +/- n/year)`` for ``n`` in ``year_beats``;

    where ``f_true = 1/P_true``.  Beats yielding a non-positive frequency are
    dropped.  ``baseline`` is accepted (and ignored) so the signature matches the
    other helpers; aliases do not depend on it.
    """
    P_true = float(P_true)
    f_true = 1.0 / P_true
    aliases = []
    for m in harmonics:
        m = float(m)
        if m <= 0:
            continue
        if m >= 1:
            name = ("%dP" % int(round(m)) if abs(m - round(m)) < 1e-9
                    else "%gP" % m)
        else:
            name = "P/%d" % int(round(1.0 / m))
        aliases.append(AliasMatch(name=name, P_alias=P_true * m, multiplier=m))
    for unit_name, unit, beats in (("day", day, day_beats),
                                   ("year", year, year_beats)):
        for n in beats:
            for sign in (+1, -1):
                f = f_true + sign * n / unit
                if f <= 0:
                    continue
                name = "beat_%s%d/%s" % ("+" if sign > 0 else "-", n, unit_name)
                aliases.append(AliasMatch(name=name, P_alias=1.0 / f,
                                          multiplier=float('nan')))
    return aliases


# ----------------------------------------------------------------------
# Unified per-source classifier and population aggregator
# ----------------------------------------------------------------------
#: Per-source recovery verdict under one convention, with exact vs
#: exact-or-harmonic broken out and the matching alias named.
RecoveryResult = namedtuple(
    'RecoveryResult',
    ['recovered', 'exact', 'exact_or_harmonic', 'criterion', 'alias_name',
     'P_rec', 'P_true', 'delta_f', 'baseline'])


def classify_recovery(P_rec, P_true, baseline=None, criterion='fractional',
                      rtol=0.01, delta_phi_max=0.5, count_harmonics=True,
                      aliases=None, **alias_kwargs):
    """Score a single recovered period ``P_rec`` against truth ``P_true``.

    ``criterion`` selects the convention: ``'fractional'`` (default, the headline)
    or ``'phase_coherence'`` (requires ``baseline``).  ``exact`` tests against
    ``P_true``; ``exact_or_harmonic`` additionally tests the alias set (built from
    :func:`harmonic_alias_set` unless ``aliases`` is passed), recording the first
    matching alias in ``alias_name``.  ``recovered`` follows ``exact_or_harmonic``
    when ``count_harmonics`` (default), else ``exact``.  Extra keyword arguments
    are forwarded to :func:`harmonic_alias_set`.
    """
    P_rec = float(P_rec)
    P_true = float(P_true)

    if criterion == 'fractional':
        def _test(target):
            return bool(recovered_fractional(P_rec, target, rtol=rtol))
    elif criterion == 'phase_coherence':
        if baseline is None:
            raise ValueError("criterion='phase_coherence' requires `baseline` "
                             "(the observing time span T)")

        def _test(target):
            return bool(recovered_phase_coherence(P_rec, target, baseline,
                                                  delta_phi_max=delta_phi_max))
    else:
        raise ValueError("unknown criterion %r; expected 'fractional' or "
                         "'phase_coherence'" % (criterion,))

    exact = _test(P_true)

    alias_name = None
    if not exact:
        if aliases is None:
            aliases = harmonic_alias_set(P_true, baseline=baseline, **alias_kwargs)
        for alias in aliases:
            if _test(alias.P_alias):
                alias_name = alias.name
                break

    exact_or_harmonic = exact or (alias_name is not None)
    recovered = exact_or_harmonic if count_harmonics else exact

    delta_f = abs(1.0 / P_rec - 1.0 / P_true) if P_rec > 0 else float('inf')

    return RecoveryResult(
        recovered=recovered, exact=exact, exact_or_harmonic=exact_or_harmonic,
        criterion=criterion, alias_name=alias_name, P_rec=P_rec, P_true=P_true,
        delta_f=delta_f, baseline=baseline)


def rescore_aliases(P_rec, P_true, baseline=None, criterion='phase_coherence',
                    rtol=0.01, delta_phi_max=0.5, **alias_kwargs):
    """Post-hoc alias re-scoring of persisted per-source recovered periods.

    Classifies every ``(P_rec, P_true)`` pair under ``criterion`` and names WHAT
    was recovered: ``'exact'``, the first matching alias from
    :func:`harmonic_alias_set` (e.g. ``'P/2'``, ``'beat_+1/year'``), or
    ``'miss'``.  Returns ``(labels, breakdown)`` -- ``labels`` is the per-source
    list of those strings and ``breakdown`` an ordered count dict (``'exact'``
    first, aliases sorted, ``'miss'`` and ``'n'`` last).

    Why this exists: under the headline fractional criterion (``rtol=0.01``) a
    +/-1/yr window beat (``|df| = 1/365.25`` c/d) falls WITHIN the 1% tolerance
    for ``f_true >~ 0.28`` c/d and is counted as exact -- the exact test runs
    first, so the alias scan never sees it -- while 1-day beats and P/2 fail the
    same tolerance (asymmetric).  Phase coherence (``|df| * T``) separates year
    beats whenever ``T >~ 0.5 / delta_phi_max`` years; it is the intended
    re-scoring criterion (requires per-source ``baseline``).
    """
    P_rec = np.atleast_1d(np.asarray(P_rec, dtype=float))
    P_true = np.atleast_1d(np.asarray(P_true, dtype=float))
    if P_rec.shape != P_true.shape:
        raise ValueError("P_rec and P_true must have matching shapes; got %r vs "
                         "%r" % (P_rec.shape, P_true.shape))
    if baseline is None:
        baselines = [None] * P_rec.size
    else:
        baselines = np.broadcast_to(np.asarray(baseline, dtype=float),
                                    P_rec.shape)
    labels = []
    for p_rec, p_true, T in zip(P_rec, P_true, baselines):
        r = classify_recovery(p_rec, p_true, baseline=T, criterion=criterion,
                              rtol=rtol, delta_phi_max=delta_phi_max,
                              count_harmonics=True, **alias_kwargs)
        labels.append('exact' if r.exact else (r.alias_name or 'miss'))
    breakdown = {'exact': labels.count('exact')}
    for name in sorted(set(labels) - {'exact', 'miss'}):
        breakdown[name] = labels.count(name)
    breakdown['miss'] = labels.count('miss')
    breakdown['n'] = len(labels)
    return labels, breakdown


def recovery_rate(results):
    """Aggregate a sequence of :class:`RecoveryResult` into summary statistics.

    Returns a ``dict`` with the headline ``recovered`` rate, the ``exact`` and
    ``exact_or_harmonic`` fractions, the count ``n``, and an ``alias_breakdown``
    counting which aliases accounted for the non-exact recoveries.  This scalar
    summary is what the K-sweep driver and the greedy selector maximize.
    """
    results = list(results)
    n = len(results)
    if n == 0:
        return {'recovered': 0.0, 'exact': 0.0, 'exact_or_harmonic': 0.0,
                'n': 0, 'alias_breakdown': {}}
    breakdown = {}
    for r in results:
        if r.alias_name is not None:
            breakdown[r.alias_name] = breakdown.get(r.alias_name, 0) + 1
    return {
        'recovered': sum(1 for r in results if r.recovered) / n,
        'exact': sum(1 for r in results if r.exact) / n,
        'exact_or_harmonic': sum(1 for r in results if r.exact_or_harmonic) / n,
        'n': n,
        'alias_breakdown': breakdown,
    }
