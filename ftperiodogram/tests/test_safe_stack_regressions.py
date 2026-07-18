"""Mechanism pins for the 2026-07-18 "SAFE CPU stack" (paper2-fast-multiband).

An adversarial coverage review (2026-07-18) found five mechanisms from the
SAFE stack that no committed test pinned -- mutants of each survived the full
suite. Each test here was verified to PASS on the working tree and to FAIL
under the corresponding runtime-simulated mutant (monkeypatched in a
throwaway harness; no mutant code is committed):

1. Fused trig recurrence in ``summations._direct_stacked_sums_chunk``:
   harmonic phases come from one complex exponential per (frequency, sample)
   advanced per HARMONIC; a frequency-axis cumulative-product recurrence was
   explicitly rejected (drift grows with grid length: measured 7.2e-10 on the
   4096-frequency fixture below vs 2.1e-12 for the committed code -- but only
   ~1e-12-1e-13 at the suite's small grids, which is why no other test sees it).
2. Batched multiband deep-dip deferral (``multiband._BATCHED_DEFER_RTOL``):
   extreme-|MM'|-dip rows of ``multiband_power_spectra_batched`` must be
   recomputed by the per-frequency reference so they BIT-match
   ``solve_over_frequencies`` (disabling the deferral leaves ~1e-12-level
   batched-assembly divergence exactly at the worst-conditioned rows).
3. Stage-2 triage of ``core.scan_polish_from_coefs``: the densified rescan
   must RECOMPUTE the Bernstein gate from the densified measurement (rows
   whose dense-grid conditioning still fails it escalate to the exact root
   path), and rows whose prescribed density exceeds ``_SCAN_DEEP_MAX_ANGLES``
   must take the exact root path directly. Natural corner fixtures do not
   discriminate the recheck (the dense rescan already recovers the peak
   there, deficits <= ~1e-11), so these pins use synthetic YM/MM rows with a
   near-zero of |MM| on the circle and a YM zero displacing the power spike
   one dip-width from the |MM| minimum -- geometry that defeats the
   dip-Newton seeds, so skipping the escalation loses ~0.5 of power.
4. Mixed-H summation hoist in ``FastMultiTemplatePeriodogram.autopower``:
   the template-independent sums are hoisted per distinct harmonic count;
   keying them by the FIRST template's H corrupts every other-H template.
5. Multi-chunk hoist freshness in ``multiband_power_spectra_batched``: the
   per-band stacked ``Summations`` must be recomputed for every frequency
   chunk; reusing the first chunk's sums is shape-preserving (silent) when
   nfreq is an exact multiple of chunk_size.
"""
import numpy as np
import numpy.polynomial as pol

from .. import core
from .. import multiband as mb
from .. import summations as summ
from ..core import roots_from_YM_MM, template_periodogram
from ..modeler import FastMultiTemplatePeriodogram
from ..template import Template
from ..utils import Summations, weights


# ----------------------------------------------------------------------
# 1. Fused trig recurrence in _direct_stacked_sums_chunk
# ----------------------------------------------------------------------
def _explicit_trig_reference_sums(t, y, w, freqs, nh):
    """Reference stacked sums from EXPLICIT ``np.cos``/``np.sin`` of
    ``h * 2 pi f t`` (no recurrence of any kind), with the same uncentered
    product-to-sum covariance assembly as ``_direct_stacked_sums_chunk`` --
    so the comparison isolates the harmonic-phase generation."""
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    fr = np.asarray(freqs, dtype=float)
    ybar = np.dot(w, y)
    u = w * (y - ybar)

    h = np.arange(1, 2 * nh + 1)
    ang = 2 * np.pi * fr[:, None, None] * h[None, :, None] * t[None, None, :]
    cosA = np.cos(ang)                                      # (m, 2H, N)
    sinA = np.sin(ang)
    C = cosA @ w                                            # (m, 2H)
    S = sinA @ w
    YC = cosA[:, :nh, :] @ u                                # (m, H)
    YS = sinA[:, :nh, :] @ u

    k = np.arange(nh)
    j = k[:, np.newaxis]
    diag = np.arange(nh)
    Sn = np.sign(k - j) * S[:, np.abs(k - j) - 1]
    Sn[:, diag, diag] = 0
    Cn = C[:, np.abs(k - j) - 1]
    Cn[:, diag, diag] = 1
    Sp = S[:, j + k + 1]
    Cp = C[:, j + k + 1]
    Cj, Ck = C[:, :nh, np.newaxis], C[:, np.newaxis, :nh]
    Sj, Sk = S[:, :nh, np.newaxis], S[:, np.newaxis, :nh]
    CC = 0.5 * (Cn + Cp) - Cj * Ck
    CS = 0.5 * (Sn + Sp) - Cj * Sk
    SS = 0.5 * (Cn - Cp) - Sj * Sk
    return Summations(C=C[:, :nh], S=S[:, :nh], YC=YC, YS=YS,
                      CC=CC, CS=CS, SS=SS)


def test_direct_stacked_sums_accurate_over_long_frequency_grid():
    """Long-grid accuracy of the fused trig recurrence (mechanism 1): the
    batched direct sums over a 4096-frequency uniform grid (one chunk, so
    any frequency-axis recurrence would run its full length) must match the
    explicit-cos/sin reference to 1e-11 in every field. The committed
    harmonic-only recurrence measures ~2e-12 here; the rejected freq-axis
    cumprod variant drifts to ~7e-10 and fails. The suite's other fixtures
    use grids too small to expose the drift (~1e-12 at nf <~ 100)."""
    nh, N = 8, 20
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 1826.0, N))
    dy = 0.05 * (1 + rng.random(N))
    w = weights(dy)
    y = rng.normal(0, 1, N)
    freqs = 5e-5 * (4000 + np.arange(4096))     # uniform long-baseline grid

    chunks = list(summ.direct_summations_batched(t, y, w, freqs, nh,
                                                 chunk_size=4096))
    assert len(chunks) == 1                      # single chunk: full-length
    got = chunks[0]
    ref = _explicit_trig_reference_sums(t, y, w, freqs, nh)
    for field in Summations._fields:
        diff = float(np.max(np.abs(np.asarray(getattr(got, field))
                                   - np.asarray(getattr(ref, field)))))
        assert diff <= 1e-11, (field, diff)


# ----------------------------------------------------------------------
# 2. Batched multiband deep-dip deferral
# ----------------------------------------------------------------------
def _clumped_two_band_fixture(seed=26002):
    """Two phase-clumped bands at H=8 (the test_scan_polish
    test_narrow_peak_multiband_clustered recipe): the combined |MM'| dips
    below _BATCHED_DEFER_RTOL of its circle max on a few frequencies of the
    0.8-1.2 grid, so the batched deferral genuinely fires."""
    rng = np.random.default_rng(seed)
    H = 8
    t1 = np.concatenate([0.03 * rng.random(4), 3.0 + 0.03 * rng.random(3)])
    t2 = np.concatenate([1.5 + 0.03 * rng.random(4),
                         4.5 + 0.03 * rng.random(3)])
    t = np.concatenate([np.sort(t1), np.sort(t2)])
    bands = np.array(['g'] * 7 + ['r'] * 7)
    n = np.arange(1, H + 1)
    tmpl = Template(1.0 / n, 0.2 / n)
    dy = 0.05 * (1 + rng.random(14))
    y = tmpl((t * 1.0) % 1.0) + dy * rng.standard_normal(14)
    return t, y, bands, dy, tmpl, H


def test_batched_multiband_defers_deep_dip_rows_to_reference():
    """Deferral contract of multiband_power_spectra_batched (mechanism 2):
    rows whose combined-|MM'| circle conditioning falls below
    _BATCHED_DEFER_RTOL are recomputed by the per-frequency reference and
    must BIT-match solve_over_frequencies there (np.array_equal); the
    fully-batched remainder agrees to well below 1e-9. With the deferral
    disabled (deep = empty mutant) the deferred rows keep the batched
    np.trace-assembly values, which differ from the reference at the
    ~1e-12 level exactly at these worst-conditioned rows."""
    t, y, bands, dy, tmpl, H = _clumped_two_band_fixture()
    freqs = np.linspace(0.8, 1.2, 25)
    template_dict = mb.build_template_set(tmpl, np.unique(bands))

    transforms, stats, fr = mb._prepare_band_transforms(
        t, y, bands, dy, freqs, H, 'floating_offsets', None, fast=False)
    p_batched = mb.multiband_power_spectra_batched(
        [template_dict], transforms, stats, fr)[0]

    sumlists, stats_ref = mb.compute_band_summations(
        t, y, bands, freqs, H, dy=dy, mode='floating_offsets', fast=False)
    p_ref, _ = mb.solve_over_frequencies(
        template_dict, sumlists, stats_ref, len(freqs), 'floating_offsets',
        None, method='scan')
    p_ref = np.asarray(p_ref)

    # independent deferral mask: circle extrema of the combined MM' on the
    # scan's own base grid (the values the batched path gates on)
    _YM, MM, _AC, _ = mb._combine_stacked_shared_amp(
        template_dict, transforms, stats, 0, len(fr), fr, 'floating_offsets')
    M_ang = max(core._SCAN_MIN_ANGLES, core._SCAN_ANGLES_PER_H * H)
    absM = np.abs(core._eval_polys_on_circle(MM, M_ang))
    deep = absM.min(axis=1) < mb._BATCHED_DEFER_RTOL * absM.max(axis=1)

    # non-vacuity: the deferral fires on some rows but not all
    assert np.any(deep) and np.any(~deep)

    # deferred rows: exact (bitwise) match to the per-frequency reference
    assert np.array_equal(p_batched[deep], p_ref[deep])
    # everywhere else: ordinary batched-assembly agreement
    assert float(np.max(np.abs(p_batched[~deep] - p_ref[~deep]))) <= 1e-9


# ----------------------------------------------------------------------
# 3. Stage-2 densified-rescan recheck + over-cap triage
# ----------------------------------------------------------------------
def _synthetic_dip_row(H, theta0, eps, B=0.5):
    """One synthetic coefficient row with |MM| = eps + B (1 - cos(theta -
    theta0)) on the circle (a single near-zero of depth eps at theta0) and
    ym(theta) = sin(theta - thetaz), thetaz = theta0 + d, d = sqrt(2 eps/B):
    the ym zero sits one dip-width inside the dip, displacing the power
    spike to theta0 - d. The spike peaks at 2 d^2 / eps = 4 / B (power 1.0
    after the YY = 4/B normalization) while every circle-grid value and the
    |MM|-minimum dip seed sit at ~2/B (power ~0.5): the seeded dip-Newton
    geometry is defeated (d2P ~ 0 at the seed), so ONLY the exact root path
    attains the peak -- the C3/C3.5 escalation geometry in isolated form.

    Returns ``(YM_coefs, MM_coefs, AC, YY)`` shaped for
    ``core.scan_polish_from_coefs`` (nf = 1).
    """
    d = np.sqrt(2.0 * eps / B)
    thetaz = theta0 + d
    MM = np.zeros((1, 4 * H + 1), dtype=np.complex128)
    MM[0, 2 * H] = B + eps
    MM[0, 2 * H - 1] = -(B / 2.0) * np.exp(1j * theta0)
    MM[0, 2 * H + 1] = -(B / 2.0) * np.exp(-1j * theta0)
    YM = np.zeros((1, 2 * H + 1), dtype=np.complex128)
    a1 = (-0.5j) * np.exp(-1j * thetaz)
    YM[0, H + 1] = a1
    YM[0, H - 1] = np.conj(a1)
    return YM, MM, np.zeros((1, H)), 4.0 / B


def _run_scan_counting_escalations(monkeypatch, YM, MM, AC, H, YY):
    """Run core.scan_polish_from_coefs with core._exact_root_fallback wrapped
    by a counting proxy (that calls the original), returning
    ``(powers, escalated_row_indices)``."""
    calls = []
    orig = core._exact_root_fallback

    def counting(rows, *args, **kwargs):
        calls.append(np.asarray(rows, dtype=np.int64).copy())
        return orig(rows, *args, **kwargs)

    monkeypatch.setattr(core, '_exact_root_fallback', counting)
    _pl, powers, _phis = core.scan_polish_from_coefs(YM, MM, AC, H, 0.0, YY)
    monkeypatch.setattr(core, '_exact_root_fallback', orig)
    escalated = (np.concatenate(calls) if calls
                 else np.zeros(0, dtype=np.int64))
    return powers, escalated


def test_stage2_densified_recheck_escalates_to_exact_root_path(monkeypatch):
    """Densified-rescan RECHECK (mechanism 3a): theta0 halfway between base
    gridpoints, eps = 1e-12. The base grid measures conditioning r ~ 3.8e-5
    (below gate0, within the angle cap -> stage-2 rescan at M = 32768); the
    dense grid lands exactly on theta0 and measures r_d ~ 1e-12, failing the
    RECOMPUTED gate (~4.7e-6), so the row must escalate to the exact root
    path -- which is the only maximizer that attains the displaced spike
    (power ~1.0; every scan pass gets ~0.5). A recheck-always-pass mutant
    (gate_d = -inf) never escalates: the counting wrapper sees no call AND
    the returned power is ~0.49 below the root-path reference (measured),
    so both assertions below fail under it."""
    H = 8
    M = max(core._SCAN_MIN_ANGLES, core._SCAN_ANGLES_PER_H * H)
    theta0 = 2.0 * np.pi * (81 + 0.5) / M    # halfway between base points,
                                             # exact on every denser 2^k grid
    YM, MM, AC, YY = _synthetic_dip_row(H, theta0, eps=1e-12)

    # regime guard: flagged for stage 2 AND within the angle cap, so the
    # ONLY route to the exact root path is a failed densified recheck
    absM = np.abs(core._eval_polys_on_circle(MM, M))
    r = float(absM.min() / absM.max())
    assert r < core._SCAN_EXACT_RTOL
    assert r <= 0.5 * (4.0 * np.pi * H / M) ** 2         # enters stage 2
    M_need = 2.0 * (4.0 * np.pi * H) / np.sqrt(2.0 * r)
    assert M_need <= core._SCAN_DEEP_MAX_ANGLES          # NOT over-cap

    powers, escalated = _run_scan_counting_escalations(
        monkeypatch, YM, MM, AC, H, YY)
    assert 0 in escalated                                # recheck escalated

    _pr, p_ref, _phi = roots_from_YM_MM(
        pol.Polynomial(YM[0]), pol.Polynomial(MM[0]), AC[0], H, 0.0, YY)
    assert p_ref >= 0.9        # fixture guard: the hidden spike is the winner
    assert abs(float(powers[0]) - p_ref) <= 5e-10        # corner-regime bar


def test_stage2_over_cap_rows_take_exact_root_path(monkeypatch):
    """_SCAN_DEEP_MAX_ANGLES over-cap branch (mechanism 3b): theta0 exactly
    on the base grid, eps = 1e-11, so the base grid itself measures
    r ~ 2e-11 and the prescribed rescan density (~4.5e7 angles) exceeds the
    2^20 cap: the row must take the exact root path DIRECTLY (no rescan).
    Same displaced-spike geometry as the recheck test, so a mutant that
    never takes the over-cap branch keeps the stage-1 scan result: no
    fallback call and a power ~0.5 below the reference (measured 0.50),
    failing both assertions."""
    H = 8
    M = max(core._SCAN_MIN_ANGLES, core._SCAN_ANGLES_PER_H * H)
    theta0 = 2.0 * np.pi * 81 / M            # ON the base grid: the dip
                                             # bottom is measured directly
    YM, MM, AC, YY = _synthetic_dip_row(H, theta0, eps=1e-11)

    # regime guard: the prescribed density exceeds the cap
    absM = np.abs(core._eval_polys_on_circle(MM, M))
    r = float(absM.min() / absM.max())
    assert r < core._SCAN_EXACT_RTOL
    assert r <= 0.5 * (4.0 * np.pi * H / M) ** 2
    M_need = 2.0 * (4.0 * np.pi * H) / np.sqrt(2.0 * r)
    assert M_need > core._SCAN_DEEP_MAX_ANGLES           # over-cap

    powers, escalated = _run_scan_counting_escalations(
        monkeypatch, YM, MM, AC, H, YY)
    assert 0 in escalated                                # over-cap escalated

    _pr, p_ref, _phi = roots_from_YM_MM(
        pol.Polynomial(YM[0]), pol.Polynomial(MM[0]), AC[0], H, 0.0, YY)
    assert p_ref >= 0.9
    assert abs(float(powers[0]) - p_ref) <= 5e-10


# ----------------------------------------------------------------------
# 4. Mixed-H summation hoist in FastMultiTemplatePeriodogram.autopower
# ----------------------------------------------------------------------
def test_multi_template_autopower_hoist_keyed_per_harmonic_count():
    """Mixed-H hoist (mechanism 4): autopower over TWO templates each at
    H = 3 and H = 6 (the hoist only engages where >= 2 templates share an
    H -- lone templates use the internal streamed path) must equal the
    per-template template_periodogram max computed WITHOUT the hoist
    (summations=None) on the same grid -- bitwise, since the hoisted
    direct sums are the identical per-frequency computation. A mutant
    keying the hoisted sums by the FIRST template's H feeds H=3 sums to
    the H=6 templates and fails (broadcast error in the assembly)."""
    rng = np.random.RandomState(7)
    t3a = Template(rng.randn(3), rng.randn(3))
    t3b = Template(rng.randn(3), rng.randn(3))
    t6a = Template(rng.randn(6), rng.randn(6))
    t6b = Template(rng.randn(6), rng.randn(6))
    N = 30
    t = np.sort(10.0 * rng.rand(N))
    dy = 0.05 * (1 + rng.rand(N))
    y = 1.5 * t6a((t / 0.77) % 1.0) + dy * rng.randn(N)

    tmpls = [t3a, t3b, t6a, t6b]
    model = FastMultiTemplatePeriodogram(templates=tmpls).fit(t, y, dy)
    kw = dict(minimum_frequency=0.5, maximum_frequency=3.0,
              samples_per_peak=3)
    freqs = model.autofrequency(**kw)
    f_auto, p_auto = model.autopower(fast=False, save_best_model=False, **kw)
    assert np.array_equal(f_auto, freqs)

    ps = [template_periodogram(t, y, dy, tm.c_n, tm.s_n, freqs,
                               fast=False, method='scan')[0]
          for tm in tmpls]
    # non-vacuity: an H=3 and an H=6 template each win somewhere, so both
    # hoisted sums sets are load-bearing in the reported max
    p3 = np.maximum(ps[0], ps[1])
    p6 = np.maximum(ps[2], ps[3])
    assert np.any(p3 > p6) and np.any(p6 > p3)
    assert np.array_equal(p_auto, np.max(ps, axis=0))


# ----------------------------------------------------------------------
# 5. Multi-chunk hoist freshness in multiband_power_spectra_batched
# ----------------------------------------------------------------------
def test_batched_multiband_chunks_are_recomputed_per_chunk():
    """Chunk freshness (mechanism 5): with nfreq = 128 an EXACT multiple of
    chunk_size = 64, reusing the first chunk's hoisted per-band sums for the
    second chunk is shape-preserving and silent -- so pin bitwise equality
    of the chunked run against the single-chunk run (chunk_size = 4096).
    The stale-chunk mutant reproduces the first 64 rows but corrupts rows
    64-127 (measured max deviation ~0.75)."""
    rng = np.random.RandomState(3)
    H = 4
    tmplA = Template(rng.randn(H), rng.randn(H))
    tmplB = Template(rng.randn(H), rng.randn(H))
    labels = ['g', 'r', 'i']
    npb = 40
    t = np.concatenate([np.sort(10.0 * rng.rand(npb)) for _ in labels])
    bands = np.array(sum(([b] * npb for b in labels), []))
    dy = 0.05 * (1 + rng.rand(3 * npb))
    amp = {b: 0.8 + 0.4 * j for j, b in enumerate(labels)}
    off = {b: 15.0 - 0.5 * j for j, b in enumerate(labels)}
    y = (np.array([amp[b] for b in bands]) * tmplA((t / 0.77) % 1.0)
         + np.array([off[b] for b in bands]) + dy * rng.randn(3 * npb))
    freqs = np.linspace(0.5, 3.0, 128)

    tdA = mb.build_template_set(tmplA, np.unique(bands))
    tdB = mb.build_template_set(tmplB, np.unique(bands))
    transforms, stats, fr = mb._prepare_band_transforms(
        t, y, bands, dy, freqs, H, 'floating_offsets', None, fast=False)

    p_chunked = mb.multiband_power_spectra_batched(
        [tdA, tdB], transforms, stats, fr, chunk_size=64)
    p_single = mb.multiband_power_spectra_batched(
        [tdA, tdB], transforms, stats, fr, chunk_size=4096)
    assert p_chunked.shape == (2, 128)
    assert np.array_equal(p_chunked, p_single)
