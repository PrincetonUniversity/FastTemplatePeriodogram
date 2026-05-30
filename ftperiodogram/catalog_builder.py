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
import os
import tarfile
from collections import namedtuple
from urllib.request import urlopen

import numpy as np

from .template import Template


_SESAR_TEMPLATE_FILE = "RRLyr_ugriz_templates.tar.gz"
#: Live astroML-data mirror of the original Sesar et al. (2010) 98-template ugriz
#: archive (Sesar's own www.mpia.de/~bsesar URL, used by gatspy, 404s).  The
#: loader reads it with the standard library only -- no gatspy/astroML/astropy.
_SESAR_TEMPLATE_MIRROR = (
    "https://raw.githubusercontent.com/astroML/astroML-data/main/datasets/"
    + _SESAR_TEMPLATE_FILE)
#: Default on-disk cache for the downloaded Sesar archive.
_SESAR_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ftperiodogram_data")


#: Optional diagnostics returned by :func:`build_template_catalog` when
#: ``return_diagnostics=True``.
CatalogDiagnostics = namedtuple(
    'CatalogDiagnostics',
    ['medoid_indices', 'labels', 'cluster_sizes', 'total_cost', 'metric',
     'orbit_chart_agreement'])


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
def _max_cross_correlation(zf, zg, oversample=16, polish=True):
    """``(max_cc, delta)`` where ``max_cc = max_Delta sum_k Re(w_k e^{-2 pi i k Delta})``.

    ``w_k = z^f_k conj(z^g_k)``; ``delta`` is the maximizing shift in turns.
    ``CC(Delta)`` is a degree-``H`` trigonometric polynomial with up to ``H``
    local maxima, so a single coarse-grid argmax can land in a sub-optimal
    basin.  We therefore Newton-polish *every* coarse-grid local maximum (each
    true peak spans ``>= oversample/2`` grid points, so all are captured) and
    keep the global best.
    """
    H = max(len(zf), len(zg))
    w = _pad(zf, H) * np.conj(_pad(zg, H))            # length H, harmonic j at index j-1
    k = np.arange(1, H + 1)
    two_pi_k = 2.0 * np.pi * k

    # Coarse FFT grid. CC is band-limited to H, so M > 2H is alias-free;
    # oversample beyond Nyquist so every peak shows up as a grid local maximum.
    M = 1 << max(4, int(np.ceil(np.log2(oversample * H + 1))))
    spectrum = np.zeros(M, dtype=complex)
    spectrum[1:H + 1] = w
    cc_grid = np.fft.fft(spectrum).real               # cc_grid[m] = CC(m / M)

    if not polish:
        m = int(np.argmax(cc_grid))
        return float(cc_grid[m]), (m / M) % 1.0

    # Candidate shifts: all cyclic local maxima of the coarse grid.
    is_peak = ((cc_grid >= np.roll(cc_grid, 1)) &
               (cc_grid >= np.roll(cc_grid, -1)))
    candidates = np.flatnonzero(is_peak)
    if candidates.size == 0:                           # flat grid (e.g. H == 0)
        candidates = np.array([int(np.argmax(cc_grid))])

    best_cc, best_delta = -np.inf, 0.0
    for m in candidates:
        delta = m / M
        for _ in range(16):                            # Newton on closed-form CC', CC''
            uk = w * np.exp(-1j * two_pi_k * delta)
            d1 = np.sum(two_pi_k * uk.imag)            # CC'(delta)
            d2 = -np.sum(two_pi_k ** 2 * uk.real)      # CC''(delta)
            if d2 >= 0:                                 # not a maximum here; keep grid point
                break
            step = d1 / d2
            delta -= step
            if abs(step) < 1e-15:
                break
        value = float(np.sum((w * np.exp(-1j * two_pi_k * delta)).real))
        if value > best_cc:
            best_cc, best_delta = value, delta % 1.0
    return best_cc, best_delta


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
    """Euclidean distance on the polar-embedded ``(R_k1, phi_k1)`` chart.

    Each invariant is embedded as ``R_k1 * exp(i phi_k1)`` and compared with the
    squared chordal distance::

        d^2 = sum_{k>=2} w_k [ R^f^2 + R^g^2 - 2 R^f R^g cos(phi^f - phi^g) ]

    This is the amplitude-weighted form: a harmonic absent in one template
    (``R = 0``) contributes only its radial term, with no spurious phase penalty
    from the (meaningless) phase of a zero coefficient.  Returns ``NaN`` when
    either fundamental amplitude is below ``a1_floor``.
    """
    inv_f = _invariants(template_f, a1_floor)
    inv_g = _invariants(template_g, a1_floor)
    if inv_f is None or inv_g is None:
        return np.nan
    Rf, pf = inv_f
    Rg, pg = inv_g
    H = max(len(Rf), len(Rg))
    Rf, Rg = _pad(Rf, H), _pad(Rg, H)
    pf, pg = _pad(pf, H), _pad(pg, H)
    terms = Rf ** 2 + Rg ** 2 - 2.0 * Rf * Rg * np.cos(pf - pg)
    if harmonic_weights is not None:
        terms = terms * np.asarray(harmonic_weights, dtype=float)[:H]
    return float(np.sqrt(np.maximum(0.0, np.sum(terms))))


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


# ----------------------------------------------------------------------
# PAM (Partitioning Around Medoids) on a precomputed dissimilarity matrix
# ----------------------------------------------------------------------
def _nearest_stats(D, medoids):
    """Per-point nearest-medoid position and the two smallest medoid distances."""
    medoids = np.asarray(medoids)
    sub = D[:, medoids]                                   # (N, K)
    rows = np.arange(sub.shape[0])
    if sub.shape[1] == 1:
        return (np.zeros(sub.shape[0], dtype=int), sub[:, 0].copy(),
                np.full(sub.shape[0], np.inf))
    order = np.argsort(sub, axis=1)
    nearest = order[:, 0]
    E1 = sub[rows, order[:, 0]]                           # nearest-medoid distance
    E2 = sub[rows, order[:, 1]]                           # second-nearest distance
    return nearest, E1, E2


def _build(D, n_clusters):
    """Greedy PAM BUILD: seed ``n_clusters`` medoids minimizing total cost."""
    n = len(D)
    medoids = [int(np.argmin(D.sum(axis=1)))]
    nearest_d = D[medoids[0]].copy()
    while len(medoids) < n_clusters:
        # gain of adding h = sum_j max(0, current_nearest[j] - D[h, j])
        gains = np.maximum(0.0, nearest_d[None, :] - D).sum(axis=1)
        gains[medoids] = -np.inf
        h = int(np.argmax(gains))
        medoids.append(h)
        nearest_d = np.minimum(nearest_d, D[h])
    return medoids


def _swap(D, medoids, max_iter):
    """PAM SWAP: greedily apply the best cost-reducing medoid<->point swap."""
    medoids = list(medoids)
    n = len(D)
    for _ in range(max_iter):
        nearest, E1, E2 = _nearest_stats(D, medoids)
        non_medoids = np.setdiff1d(np.arange(n), medoids)
        best_delta = -1e-12                              # require strict improvement
        best_swap = None
        for i in range(len(medoids)):
            assigned_to_i = (nearest == i)
            for h in non_medoids:
                dh = D[:, h]
                contrib = np.where(assigned_to_i,
                                   np.minimum(dh, E2) - E1,
                                   np.minimum(dh, E1) - E1)
                delta = float(contrib.sum())
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (i, int(h))
        if best_swap is None:
            break
        i, h = best_swap
        medoids[i] = h
    cost = float(_nearest_stats(D, medoids)[1].sum())
    return medoids, cost


def _pam(D, n_clusters, n_init=10, max_iter=300, random_state=None):
    """Partitioning Around Medoids on a precomputed symmetric dissimilarity.

    Classical BUILD + SWAP with ``n_init`` restarts (one deterministic BUILD,
    the rest random medoid subsets); keeps the lowest-cost solution.

    Returns ``(medoid_indices, labels, inertia)`` where ``labels[j]`` is the
    position in ``medoid_indices`` of point ``j``'s nearest medoid and
    ``inertia`` is the total within-cluster distance to the medoids.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix")
    n = D.shape[0]
    if not 1 <= n_clusters <= n:
        raise ValueError("n_clusters must be in [1, n_samples]; "
                         "got %r for n_samples=%d" % (n_clusters, n))
    rng = (random_state if isinstance(random_state, np.random.RandomState)
           else np.random.RandomState(random_state))

    best_medoids, best_cost = None, np.inf
    for init in range(max(1, n_init)):
        if init == 0:
            medoids = _build(D, n_clusters)
        else:
            medoids = list(rng.choice(n, size=n_clusters, replace=False))
        medoids, cost = _swap(D, medoids, max_iter)
        if cost < best_cost:
            best_medoids, best_cost = medoids, cost

    medoids = sorted(best_medoids)
    labels, _, _ = _nearest_stats(D, medoids)
    return np.asarray(medoids, dtype=int), labels.astype(int), best_cost


# ----------------------------------------------------------------------
# Public API: build_template_catalog
# ----------------------------------------------------------------------
def templates_from_sampled(Y, nharmonics=8, template_ids=None):
    """Convert a 2D ``(n_templates, n_phase)`` magnitude matrix to ``Template``s.

    A thin adapter for the common phase-aligned-grid format (e.g. the Sesar
    2010 templates).  Pass an *integer* ``nharmonics`` so every template shares
    a common harmonic order (a float fraction would let H vary per row).
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if template_ids is None:
        template_ids = range(len(Y))
    return [Template.from_sampled(row, nharmonics=nharmonics, template_id=tid)
            for row, tid in zip(Y, template_ids)]


def _normalize_harmonics(templates, n_harmonics):
    """Validate a common harmonic order, or truncate/zero-pad to ``n_harmonics``."""
    counts = [len(t.c_n) for t in templates]
    if n_harmonics is None:
        if len(set(counts)) != 1:
            raise ValueError(
                "templates have differing harmonic counts %r; pass "
                "n_harmonics to truncate/pad to a common order"
                % sorted(set(counts)))
        return list(templates)
    H = int(n_harmonics)
    if H < 1:
        raise ValueError("n_harmonics must be >= 1")
    out = []
    for t in templates:
        c = _pad(np.asarray(t.c_n, float).astype(complex), H).real[:H]
        s = _pad(np.asarray(t.s_n, float).astype(complex), H).real[:H]
        out.append(Template(c.copy(), s.copy(), template_id=t.template_id))
    return out


def _medoid_assignment(templates, medoid_idx, distance):
    """Argmin-medoid assignment of every template under ``distance(a, b)``."""
    n, K = len(templates), len(medoid_idx)
    D = np.full((n, K), np.nan)
    for p, m in enumerate(medoid_idx):
        for i in range(n):
            D[i, p] = distance(templates[i], templates[m])
    valid = np.all(np.isfinite(D), axis=1)
    assign = np.argmin(np.where(np.isfinite(D), D, np.inf), axis=1)
    return assign, valid


def _orbit_chart_agreement(templates, medoid_idx, a1_floor, harmonic_weights):
    """Fraction of templates whose nearest medoid agrees between orbit and chart."""
    orbit_assign, _ = _medoid_assignment(
        templates, medoid_idx, lambda a, b: _orbit_distance(a, b))
    chart_assign, valid = _medoid_assignment(
        templates, medoid_idx,
        lambda a, b: _chart_distance(a, b, a1_floor, harmonic_weights))
    if not np.any(valid):
        return None
    return float(np.mean(orbit_assign[valid] == chart_assign[valid]))


def build_template_catalog(templates, n_clusters, metric='orbit',
                           n_harmonics=None, harmonic_weights=None,
                           a1_floor=1e-8, n_init=10, max_iter=300,
                           random_state=None, method='pam',
                           return_diagnostics=False):
    """Compress ``templates`` into ``n_clusters`` representative templates.

    Clusters the input templates with k-medoids/PAM under the orbit-minimized
    (phase-shift-invariant) distance and returns the medoids -- real input
    templates, hence guaranteed-physical -- as a ``list`` of :class:`Template`
    suitable for :class:`FastMultibandTemplatePeriodogram` catalog mode.

    Parameters
    ----------
    templates : sequence of Template
        Training templates (already unit-Fourier-energy normalized).
    n_clusters : int
        Vocabulary size ``K`` (required; ``1 <= K <= len(templates)``).
    metric : {'orbit', 'chart'}
        Distance used for clustering.  ``'orbit'`` is the reference U(1)
        circular-shift Procrustes distance; ``'chart'`` is the fast surrogate on
        the ``(R_k1, phi_k1)`` invariants (raises if any pair is singular).
    n_harmonics : int or None
        If ``None`` (default), require a common harmonic order across inputs.
        If an int, truncate/zero-pad every template to that order.
    harmonic_weights : array_like or None
        Optional per-harmonic weights for the chart metric (length ``H-1``).
    n_init, max_iter, random_state :
        PAM multi-start, swap-iteration cap, and seeding.
    method : {'pam'}
        Selection method.  ``'greedy'`` (recovery-driven) is reserved and
        raises ``NotImplementedError`` until the Phase-3 simulation harness lands.
    return_diagnostics : bool
        If ``True``, also return a :class:`CatalogDiagnostics`.

    Returns
    -------
    list of Template, or (list of Template, CatalogDiagnostics)
        When ``n_harmonics is None`` the returned medoids are the *same*
        ``Template`` objects passed in (templates are effectively immutable);
        with an integer ``n_harmonics`` they are fresh re-truncated templates.
    """
    templates = list(templates)
    n = len(templates)

    if method != 'pam':
        if method == 'greedy':
            raise NotImplementedError(
                "recovery-driven greedy selection (method='greedy') requires the "
                "Phase-3 simulation harness and is not implemented yet; "
                "use method='pam'")
        raise ValueError("unknown method %r; expected 'pam'" % (method,))
    if metric not in ('orbit', 'chart'):
        raise ValueError("metric must be 'orbit' or 'chart'; got %r" % (metric,))
    if not 1 <= n_clusters <= n:
        raise ValueError("n_clusters must be in [1, len(templates)]; "
                         "got %r for %d templates" % (n_clusters, n))

    work = _normalize_harmonics(templates, n_harmonics)

    if metric == 'orbit':
        D = _orbit_distance_matrix(work)
    else:
        D = _chart_distance_matrix(work, a1_floor=a1_floor,
                                   harmonic_weights=harmonic_weights)
        if not np.all(np.isfinite(D)):
            raise ValueError(
                "chart metric is singular (A_1 -> 0) for at least one template; "
                "use metric='orbit', which has no such singularity")

    medoid_idx, labels, cost = _pam(D, n_clusters, n_init=n_init,
                                    max_iter=max_iter, random_state=random_state)
    vocabulary = [work[idx] for idx in medoid_idx]

    if not return_diagnostics:
        return vocabulary

    diagnostics = CatalogDiagnostics(
        medoid_indices=medoid_idx,
        labels=labels,
        cluster_sizes=np.bincount(labels, minlength=n_clusters),
        total_cost=cost,
        metric=metric,
        orbit_chart_agreement=_orbit_chart_agreement(
            work, medoid_idx, a1_floor, harmonic_weights))
    return vocabulary, diagnostics


# ----------------------------------------------------------------------
# Data loader: Sesar et al. (2010) RR Lyrae templates (standard library only)
# ----------------------------------------------------------------------
def _download_sesar_archive(data_home=None, mirror_url=_SESAR_TEMPLATE_MIRROR,
                            force_download=False):
    """Download (and cache) the Sesar template archive; return its local path.

    Writes atomically (temp file + rename) so an interrupted download cannot
    leave a corrupt archive cached.
    """
    cache_dir = _SESAR_CACHE_DIR if data_home is None else data_home
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    path = os.path.join(cache_dir, _SESAR_TEMPLATE_FILE)
    if force_download or not os.path.exists(path):
        with urlopen(mirror_url) as response:
            payload = response.read()
        tmp = path + ".tmp"
        with open(tmp, 'wb') as cache:
            cache.write(payload)
        os.replace(tmp, path)
    return path


def fetch_sesar_templates(nharmonics=8, bands=None, template_ids=None,
                          data_home=None, mirror_url=_SESAR_TEMPLATE_MIRROR,
                          force_download=False):
    """Load the Sesar et al. (2010) RR Lyrae templates as Fourier ``Template``s.

    Downloads the ~170 kB template archive (the original 98 ugriz templates)
    from a live astroML-data mirror on first use, then reads the local cache.
    Implemented with the standard library only (``urllib`` + ``tarfile``) plus
    numpy -- no gatspy/astroML/astropy dependency.

    Parameters
    ----------
    nharmonics : int
        Harmonics per template.  Use an integer so every template shares a
        common order (required to cluster them together).
    bands : sequence of str or None
        Keep only these ugriz band letters (e.g. ``['r']``); ``None`` keeps all.
    template_ids : sequence of str or None
        Keep only these explicit Sesar ids (e.g. ``['0r', '1r']``).
    data_home : str or None
        Cache directory (defaults to ``~/.ftperiodogram_data``).
    mirror_url : str
        Override the template-archive download URL.
    force_download : bool
        Re-download even if the archive is already cached.

    Returns
    -------
    list of Template
        Each tagged with its Sesar id (e.g. ``'0r'``).
    """
    path = _download_sesar_archive(data_home=data_home, mirror_url=mirror_url,
                                   force_download=force_download)
    wanted = None if template_ids is None else set(template_ids)
    band_set = None if bands is None else set(bands)

    templates = []
    with tarfile.open(path) as archive:
        # member names are flat '<star><band>.dat' (e.g. '107r.dat'); sorted()
        # gives a stable (lexical-by-id) template order across machines.
        names = sorted(n for n in archive.getnames() if n.endswith('.dat'))
        for name in names:
            tid = name[:-len('.dat')]
            if wanted is not None and tid not in wanted:
                continue
            if band_set is not None and tid[-1] not in band_set:
                continue
            with archive.extractfile(name) as member:
                data = np.atleast_2d(np.loadtxt(member))  # columns: phase, mag
            templates.append(Template.from_sampled(data[:, 1],
                                                   nharmonics=nharmonics,
                                                   template_id=tid))
    if not templates:
        raise ValueError("no Sesar templates matched the requested filters")
    return templates
