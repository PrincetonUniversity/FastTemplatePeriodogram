"""Template-vocabulary builder (Phase 3).

Compress a set of phase-folded variable-star templates into a small set of ``K``
representative templates -- a *vocabulary* -- via k-medoids/PAM clustering under
a phase-shift-invariant distance.  The medoids are *real input templates* (hence
guaranteed-physical) and are returned as a plain ``list`` of :class:`Template`
that slots directly into :class:`FastMultibandTemplatePeriodogram` "catalog mode".

This module is the *producer* of a template list; ``build_template_set`` and
``FastMultibandTemplatePeriodogram(catalog)`` (in :mod:`ftperiodogram.multiband`)
are the *consumers* of one -- a separate, pre-existing notion of "catalog".

Distance
--------
Templates are compared with the orbit-minimized L2 distance (the U(1)
circular-shift Procrustes / shape distance), which quotients out the arbitrary
phase origin by construction::

    d(f, g) = min_Delta || f - g(. - Delta) ||

Every :class:`Template` is unit-Fourier-energy normalized in its constructor
(``sum_k (c_k^2 + s_k^2) = 1``), so by Parseval ``||f||^2 = ||g||^2 = 1/2`` and

    d(f, g)^2 = 1 - max_Delta  sum_k  A^f_k A^g_k cos(psi_k - 2 pi k Delta)

with the complex coefficient ``z_k = c_k - i s_k`` (matching ``Template``'s
``c_k cos + s_k sin`` model and ``np.fft.rfft`` synthesis), amplitude
``A_k = |z_k|``, and ``psi_k = arg(z^f_k) - arg(z^g_k)``.  The phase shift is
measured in *turns* (``Delta in [0, 1)``), so the shift theorem reads
``z_k -> z_k exp(-2 pi i k Delta)``.  The maximum over ``Delta`` is localized on a
coarse FFT grid and refined to machine precision with a Newton polish on the
closed-form derivatives (a bare FFT grid is alias-free but not maximizer-exact).

A cheap, *fixed-chart* surrogate distance on the Simon & Lee (1981) invariants
``R_k1 = A_k / A_1`` and ``phi_k1 = phi_k - k phi_1`` is also provided for
validation; it carries an ``A_1 -> 0`` singularity (eclipsing binaries,
near-sinusoidal RRc, pure noise) and returns ``NaN`` there rather than dividing.
"""
import numpy as np

from .template import Template


# ----------------------------------------------------------------------
# Fourier-coefficient features
# ----------------------------------------------------------------------
def _complex_coeffs(template):
    """Complex Fourier coefficients ``z_k = c_k - i s_k`` (harmonic k at index k-1)."""
    return np.asarray(template.c_n, dtype=float) - 1j * np.asarray(template.s_n,
                                                                   dtype=float)


def _amp_phase(template):
    """``(A_k, phi_k)`` with ``A_k = |z_k|`` and ``phi_k = arg(z_k)``."""
    z = _complex_coeffs(template)
    return np.abs(z), np.angle(z)


def _invariants(template, a1_floor=1e-8):
    """Phase-shift invariants ``(R_k1, phi_k1)`` for k = 2..H.

    Returns ``None`` when the fundamental amplitude ``A_1`` is below
    ``a1_floor`` (the chart is singular there).
    """
    A, phi = _amp_phase(template)
    if A[0] < a1_floor:
        return None
    k = np.arange(2, len(A) + 1)
    R_k1 = A[1:] / A[0]
    phi_k1 = phi[1:] - k * phi[0]
    return R_k1, phi_k1


def _pad(z, H):
    """Zero-pad a length-``len(z)`` coefficient vector up to ``H`` harmonics."""
    if len(z) >= H:
        return z
    out = np.zeros(H, dtype=z.dtype)
    out[:len(z)] = z
    return out


# ----------------------------------------------------------------------
# Orbit-minimized (circular-shift Procrustes) distance
# ----------------------------------------------------------------------
def _max_cross_correlation(zf, zg, oversample=8, polish=True):
    """``(max_cc, delta)`` where ``max_cc = max_Delta sum_k Re(w_k e^{-2 pi i k Delta})``.

    ``w_k = z^f_k conj(z^g_k)``; ``delta`` is the maximizing shift in turns.
    A coarse FFT (>= ``oversample`` x Nyquist) localizes the global peak; an
    optional Newton polish on the closed-form derivatives refines it.
    """
    H = max(len(zf), len(zg))
    w = _pad(zf, H) * np.conj(_pad(zg, H))            # length H, harmonic j at index j-1
    k = np.arange(1, H + 1)

    # Coarse FFT peak. CC is a degree-H trig polynomial, so M > 2H is alias-free;
    # oversample beyond Nyquist so the argmax lands in the global peak's basin.
    M = 1 << max(3, int(np.ceil(np.log2(oversample * H + 1))))
    spectrum = np.zeros(M, dtype=complex)
    spectrum[1:H + 1] = w
    cc_grid = np.fft.fft(spectrum).real               # cc_grid[m] = CC(m / M)
    m = int(np.argmax(cc_grid))
    delta = m / M

    if not polish:
        return float(cc_grid[m]), delta % 1.0

    two_pi_k = 2.0 * np.pi * k
    for _ in range(16):
        uk = w * np.exp(-1j * two_pi_k * delta)
        d1 = np.sum(two_pi_k * uk.imag)               # CC'(delta)
        d2 = -np.sum(two_pi_k ** 2 * uk.real)         # CC''(delta)
        if d2 >= 0:                                    # not near a maximum; keep grid point
            break
        step = d1 / d2
        delta -= step
        if abs(step) < 1e-15:
            break
    max_cc = float(np.sum((w * np.exp(-1j * two_pi_k * delta)).real))
    return max_cc, delta % 1.0


def _orbit_distance(template_f, template_g, oversample=8, polish=True,
                    return_shift=False):
    """Orbit-minimized L2 distance between two unit-energy templates.

    ``d = sqrt(max(0, 1 - max_cc))``; ``d == 0`` iff one template is a pure
    phase-shift of the other.  Templates of differing harmonic order are
    zero-padded (absent harmonics have zero amplitude, contributing nothing).
    """
    max_cc, delta = _max_cross_correlation(_complex_coeffs(template_f),
                                           _complex_coeffs(template_g),
                                           oversample=oversample, polish=polish)
    d = np.sqrt(max(0.0, 1.0 - max_cc))
    return (d, delta) if return_shift else d


def _orbit_distance_matrix(templates, oversample=8, polish=True):
    """Symmetric ``(N, N)`` orbit-distance matrix with an exact-zero diagonal."""
    coeffs = [_complex_coeffs(t) for t in templates]
    n = len(coeffs)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            max_cc, _ = _max_cross_correlation(coeffs[i], coeffs[j],
                                               oversample=oversample,
                                               polish=polish)
            D[i, j] = D[j, i] = np.sqrt(max(0.0, 1.0 - max_cc))
    return D


# ----------------------------------------------------------------------
# Fast fixed-chart surrogate distance (validation cross-check only)
# ----------------------------------------------------------------------
def _chart_distance(template_f, template_g, a1_floor=1e-8, harmonic_weights=None):
    """Euclidean distance on the ``(R_k1, phi_k1)`` chart; ``NaN`` if singular.

    ``d^2 = sum_{k>=2} w_k [ (R_k1^f - R_k1^g)^2 + 2 (1 - cos(phi_k1^f - phi_k1^g)) ]``.
    Returns ``NaN`` when either fundamental amplitude is below ``a1_floor``.
    """
    inv_f = _invariants(template_f, a1_floor)
    inv_g = _invariants(template_g, a1_floor)
    if inv_f is None or inv_g is None:
        return np.nan
    Rf, pf = inv_f
    Rg, pg = inv_g
    H = max(len(Rf), len(Rg))
    Rf, Rg = _pad(Rf.astype(complex), H).real, _pad(Rg.astype(complex), H).real
    pf, pg = _pad(pf.astype(complex), H).real, _pad(pg.astype(complex), H).real
    terms = (Rf - Rg) ** 2 + 2.0 * (1.0 - np.cos(pf - pg))
    if harmonic_weights is not None:
        terms = terms * np.asarray(harmonic_weights, dtype=float)[:H]
    return float(np.sqrt(np.sum(terms)))


def _chart_distance_matrix(templates, a1_floor=1e-8, harmonic_weights=None):
    """Symmetric chart-distance matrix (``NaN`` for singular pairs, zero diagonal)."""
    n = len(templates)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _chart_distance(templates[i], templates[j],
                                                a1_floor=a1_floor,
                                                harmonic_weights=harmonic_weights)
    return D
