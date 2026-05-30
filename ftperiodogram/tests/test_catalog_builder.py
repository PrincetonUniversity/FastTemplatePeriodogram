"""Tests for the template-vocabulary builder (ftperiodogram.catalog_builder).

The decisive correctness check for the orbit distance is the brute-force oracle:
a dense uniform-Delta scan of the cross-correlation (no FFT), against which the
coarse-FFT + Newton-polish maximizer must agree. The clustering tests assert
phase-shift-invariant recovery of known archetypes -- which raw-coefficient
k-means could not do -- proving the gauge quotient.
"""
import numpy as np
import numpy.testing as npt
import pytest

from ftperiodogram.template import Template
from ftperiodogram import catalog_builder as cb


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
def _random_template(H=4, seed=0):
    rng = np.random.RandomState(seed)
    c = rng.randn(H)
    s = rng.randn(H)
    return Template(c, s, template_id=seed)


def _shift(template, delta):
    """Return a copy of ``template`` phase-shifted by ``delta`` turns.

    Applies the shift theorem z_k -> z_k exp(-2 pi i k delta), i.e. y(phi-delta).
    """
    z = np.asarray(template.c_n, float) - 1j * np.asarray(template.s_n, float)
    k = np.arange(1, len(z) + 1)
    zs = z * np.exp(-2j * np.pi * k * delta)
    return Template(zs.real, -zs.imag, template_id=template.template_id)


def _brute_max_cc(template_f, template_g, M=200000):
    """Independent oracle: dense uniform-Delta scan of CC(Delta), no FFT."""
    zf = cb._complex_coeffs(template_f)
    zg = cb._complex_coeffs(template_g)
    H = max(len(zf), len(zg))
    w = cb._pad(zf, H) * np.conj(cb._pad(zg, H))
    k = np.arange(1, H + 1)
    deltas = np.arange(M) / M
    cc = (w[None, :] * np.exp(-2j * np.pi * np.outer(deltas, k))).real.sum(axis=1)
    return cc.max()


# ----------------------------------------------------------------------
# Distance properties
# ----------------------------------------------------------------------
def test_self_distance_is_zero():
    for seed in range(5):
        t = _random_template(H=5, seed=seed)
        assert cb._orbit_distance(t, t) == pytest.approx(0.0, abs=1e-9)


def test_distance_symmetry():
    a = _random_template(H=4, seed=1)
    b = _random_template(H=4, seed=2)
    npt.assert_allclose(cb._orbit_distance(a, b), cb._orbit_distance(b, a),
                        atol=1e-12, rtol=0)


def test_distance_matrix_symmetric_zero_diagonal():
    templates = [_random_template(H=4, seed=s) for s in range(6)]
    D = cb._orbit_distance_matrix(templates)
    npt.assert_array_equal(np.diag(D), np.zeros(len(templates)))
    npt.assert_allclose(D, D.T, atol=0, rtol=0)
    assert np.all(D >= 0.0)


def test_orbit_distance_is_shift_invariant():
    """The load-bearing test: a pure phase shift must cost ~zero distance."""
    t = _random_template(H=6, seed=7)
    for delta in [0.05, 0.25, 0.5, 0.83]:
        d, recovered = cb._orbit_distance(t, _shift(t, delta), return_shift=True)
        assert d < 1e-6
        # recovered shift aligns the shifted copy back onto t (mod 1, either sign)
        err = min((recovered - delta) % 1.0, (delta - recovered) % 1.0,
                  (recovered + delta) % 1.0, (-recovered - delta) % 1.0)
        assert err < 1e-4


def test_fft_maximizer_matches_brute_oracle():
    for seed in range(6):
        a = _random_template(H=5, seed=seed)
        b = _random_template(H=5, seed=seed + 100)
        max_cc, _ = cb._max_cross_correlation(cb._complex_coeffs(a),
                                              cb._complex_coeffs(b))
        oracle = _brute_max_cc(a, b)
        # the polished maximizer is exact, so it is >= the grid oracle
        assert max_cc >= oracle - 1e-9
        npt.assert_allclose(max_cc, oracle, atol=1e-6, rtol=0)


def test_orbit_maximizer_is_global_for_high_harmonics():
    """High-H pairs have many local maxima; the maximizer must find the global
    one, not a local basin (regression guard for the multi-peak Newton polish)."""
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(40):
        H = rng.randint(8, 14)
        a = Template(rng.randn(H), rng.randn(H))
        b = Template(rng.randn(H), rng.randn(H))
        ours, _ = cb._max_cross_correlation(cb._complex_coeffs(a),
                                            cb._complex_coeffs(b))
        oracle = _brute_max_cc(a, b, M=400000)
        # the polished maximizer is exact, hence never below the dense grid
        assert ours >= oracle - 1e-7
        worst = max(worst, oracle - ours)
    assert worst < 1e-6


def test_orbit_distance_handles_mismatched_harmonics():
    a = _random_template(H=3, seed=4)
    b = _random_template(H=6, seed=5)
    d = cb._orbit_distance(a, b)
    assert 0.0 <= d <= np.sqrt(2.0) + 1e-9
    # zero-padding a to H=6 explicitly gives the same answer
    a6 = Template(cb._pad(np.asarray(a.c_n, float).astype(complex), 6).real,
                  cb._pad(np.asarray(a.s_n, float).astype(complex), 6).real)
    npt.assert_allclose(d, cb._orbit_distance(a6, b), atol=1e-9, rtol=0)


# ----------------------------------------------------------------------
# Chart surrogate distance
# ----------------------------------------------------------------------
def test_chart_distance_invariant_to_shift():
    t = _random_template(H=5, seed=11)
    assert cb._chart_distance(t, _shift(t, 0.3)) == pytest.approx(0.0, abs=1e-9)


def test_chart_distance_absent_harmonic_has_no_phantom_phase():
    """A harmonic absent in one template (R=0) must contribute only its radial
    term -- no spurious phase penalty from a zero coefficient's meaningless phase."""
    # f has harmonics {1,2}; g has {1,2,3}. The k=3 term should add exactly R_3^g^2.
    f = Template([1.0, 0.5], [0.0, 0.2])
    g = Template([1.0, 0.5, 0.3], [0.0, 0.2, 0.1])
    Rf, pf = cb._invariants(f)
    Rg, pg = cb._invariants(g)
    H = max(len(Rf), len(Rg))
    Rf, Rg = cb._pad(Rf, H), cb._pad(Rg, H)
    pf, pg = cb._pad(pf, H), cb._pad(pg, H)
    expected = np.sqrt(np.sum(Rf ** 2 + Rg ** 2 - 2 * Rf * Rg * np.cos(pf - pg)))
    npt.assert_allclose(cb._chart_distance(f, g), expected, atol=1e-12)
    # the absent k=3 harmonic of f contributes purely radially
    only3_f = Template([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    only3_g = Template([1.0, 0.0, 0.7], [0.0, 0.0, 0.0])
    R3 = cb._invariants(only3_g)[0][-1]
    npt.assert_allclose(cb._chart_distance(only3_f, only3_g), abs(R3), atol=1e-9)


def test_chart_distance_singular_when_fundamental_vanishes():
    # a pure 2nd-harmonic template has A_1 = 0 (eclipsing-binary-like)
    eb = Template([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    other = _random_template(H=3, seed=3)
    assert np.isnan(cb._chart_distance(eb, other))
    # the orbit distance has no such singularity
    assert np.isfinite(cb._orbit_distance(eb, other))


# ----------------------------------------------------------------------
# PAM k-medoids
# ----------------------------------------------------------------------
def _euclidean_matrix(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _label_accuracy(true_labels, pred_labels, n_clusters):
    """Best-permutation fraction of points whose cluster matches the truth."""
    from itertools import permutations
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    best = 0.0
    for perm in permutations(range(n_clusters)):
        mapped = np.array([perm[p] for p in pred_labels])
        best = max(best, np.mean(mapped == true_labels))
    return best


def test_pam_recovers_well_separated_blobs():
    rng = np.random.RandomState(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    true = np.repeat(np.arange(3), 20)
    X = np.vstack([c + 0.3 * rng.randn(20, 2) for c in centers])
    D = _euclidean_matrix(X)
    medoids, labels, inertia = cb._pam(D, 3, random_state=0)
    assert len(medoids) == 3
    assert _label_accuracy(true, labels, 3) == 1.0
    assert inertia > 0.0


def test_pam_is_deterministic_given_seed():
    rng = np.random.RandomState(1)
    X = rng.randn(30, 3)
    D = _euclidean_matrix(X)
    m1, l1, c1 = cb._pam(D, 4, random_state=7)
    m2, l2, c2 = cb._pam(D, 4, random_state=7)
    npt.assert_array_equal(m1, m2)
    npt.assert_array_equal(l1, l2)
    assert c1 == c2


def test_pam_matches_brute_force_optimum_small_n():
    from itertools import combinations
    rng = np.random.RandomState(2)
    X = rng.randn(8, 2)
    D = _euclidean_matrix(X)
    K = 3
    best = min(D[:, list(sub)].min(axis=1).sum() for sub in combinations(range(8), K))
    _, _, inertia = cb._pam(D, K, n_init=20, random_state=0)
    npt.assert_allclose(inertia, best, atol=1e-9, rtol=0)


def test_pam_edge_cases():
    rng = np.random.RandomState(3)
    X = rng.randn(6, 2)
    D = _euclidean_matrix(X)
    # K = 1 -> the global medoid, inertia = min total distance
    medoids, labels, inertia = cb._pam(D, 1, random_state=0)
    assert len(medoids) == 1
    npt.assert_allclose(inertia, D.sum(axis=1).min(), atol=1e-9, rtol=0)
    assert set(labels) == {0}
    # K = N -> every point its own medoid, zero inertia
    medoids, labels, inertia = cb._pam(D, 6, random_state=0)
    assert len(medoids) == 6
    npt.assert_allclose(inertia, 0.0, atol=1e-12)


def test_pam_rejects_bad_arguments():
    D = np.zeros((4, 4))
    with pytest.raises(ValueError):
        cb._pam(D, 0)
    with pytest.raises(ValueError):
        cb._pam(D, 5)
    with pytest.raises(ValueError):
        cb._pam(np.zeros((4, 3)), 2)


# ----------------------------------------------------------------------
# build_template_catalog (public API)
# ----------------------------------------------------------------------
# Three distinct shape archetypes in (amplitude, phase) space.
_ARCHETYPES = [
    (np.array([1.0, 0.5, 0.33, 0.25]), np.array([0.0, 0.10, 0.05, 0.02])),  # harmonic-rich
    (np.array([1.0, 0.05, 0.0, 0.0]), np.array([0.0, 0.00, 0.00, 0.00])),   # near-sinusoid
    (np.array([0.2, 1.0, 0.10, 0.05]), np.array([0.0, 0.00, 0.00, 0.00])),  # 2f-dominated
]


def _archetype_population(n_per=20, noise=0.02, seed=0):
    """Noisy, randomly phase-shifted variants of the 3 archetypes."""
    rng = np.random.RandomState(seed)
    templates, true = [], []
    for label, (c, s) in enumerate(_ARCHETYPES):
        for _ in range(n_per):
            t = Template(c + noise * rng.randn(len(c)),
                         s + noise * rng.randn(len(s)))
            templates.append(_shift(t, rng.rand()))   # random phase origin
            true.append(label)
    order = rng.permutation(len(templates))
    return [templates[i] for i in order], np.array(true)[order]


def test_build_template_catalog_recovers_archetypes():
    templates, true = _archetype_population(seed=1)
    vocab, diag = cb.build_template_catalog(
        templates, 3, metric='orbit', random_state=0, return_diagnostics=True)
    assert len(vocab) == 3
    assert all(isinstance(t, Template) for t in vocab)
    assert _label_accuracy(true, diag.labels, 3) >= 0.95
    # clusters are balanced (~20 each)
    assert min(diag.cluster_sizes) >= 15
    assert diag.total_cost > 0.0
    # each medoid's own archetype is the closest archetype to it
    arch_templates = [Template(c, s) for c, s in _ARCHETYPES]
    for med in vocab:
        dists = [cb._orbit_distance(med, a) for a in arch_templates]
        true_arch = true[[i for i, t in enumerate(templates)
                          if t is med or np.allclose(t.c_n, med.c_n)][0]]
        assert np.argmin(dists) == true_arch


def test_build_template_catalog_returns_unit_energy_templates():
    templates, _ = _archetype_population(seed=2)
    vocab = cb.build_template_catalog(templates, 4, random_state=0)
    for t in vocab:
        npt.assert_allclose(np.sum(t.c_n ** 2 + t.s_n ** 2), 1.0, atol=1e-9)


def test_build_template_catalog_is_deterministic():
    templates, _ = _archetype_population(seed=3)
    v1, d1 = cb.build_template_catalog(templates, 3, random_state=5,
                                       return_diagnostics=True)
    v2, d2 = cb.build_template_catalog(templates, 3, random_state=5,
                                       return_diagnostics=True)
    npt.assert_array_equal(d1.medoid_indices, d2.medoid_indices)
    npt.assert_array_equal(d1.labels, d2.labels)


def test_build_template_catalog_chart_metric_agreement():
    templates, _ = _archetype_population(seed=4)
    vocab, diag = cb.build_template_catalog(
        templates, 3, metric='orbit', random_state=0, return_diagnostics=True)
    # on this clean (non-singular) population the cheap chart largely agrees
    assert diag.orbit_chart_agreement is not None
    assert diag.orbit_chart_agreement >= 0.8


def test_build_template_catalog_n_harmonics_truncation():
    # mismatched H is rejected unless n_harmonics is given
    mixed = [_random_template(H=3, seed=1), _random_template(H=5, seed=2)]
    with pytest.raises(ValueError):
        cb.build_template_catalog(mixed, 1)
    vocab = cb.build_template_catalog(mixed, 1, n_harmonics=3)
    assert len(vocab[0].c_n) == 3
    npt.assert_allclose(np.sum(vocab[0].c_n ** 2 + vocab[0].s_n ** 2), 1.0,
                        atol=1e-9)


def test_build_template_catalog_rejects_bad_arguments():
    templates = [_random_template(H=4, seed=s) for s in range(4)]
    with pytest.raises(ValueError):
        cb.build_template_catalog(templates, 0)
    with pytest.raises(ValueError):
        cb.build_template_catalog(templates, 5)
    with pytest.raises(ValueError):
        cb.build_template_catalog(templates, 2, metric='bogus')
    with pytest.raises(ValueError):
        cb.build_template_catalog(templates, 2, method='bogus')
    with pytest.raises(NotImplementedError):
        cb.build_template_catalog(templates, 2, method='greedy')


def test_templates_from_sampled_uniform_harmonics():
    rng = np.random.RandomState(0)
    phase = np.linspace(0, 1, 128, endpoint=False)
    Y = np.array([np.cos(2 * np.pi * phase + rng.rand()) +
                  0.3 * np.cos(2 * np.pi * 2 * phase + rng.rand())
                  for _ in range(5)])
    templates = cb.templates_from_sampled(Y, nharmonics=4)
    assert len(templates) == 5
    assert all(len(t.c_n) == 4 for t in templates)
    # uniform H feeds straight into build_template_catalog
    vocab = cb.build_template_catalog(templates, 2, random_state=0)
    assert len(vocab) == 2


# ----------------------------------------------------------------------
# Sesar 2010 loader (stdlib only; network download skips cleanly offline)
# ----------------------------------------------------------------------
def test_fetch_sesar_templates_smoke():
    try:
        templates = cb.fetch_sesar_templates(nharmonics=6, bands=['r'])
    except Exception as exc:  # no network and not cached -> skip, not fail
        pytest.skip("Sesar template fetch unavailable: %s" % exc)
    assert len(templates) > 0
    assert all(len(t.c_n) == 6 for t in templates)
    for t in templates:
        npt.assert_allclose(np.sum(t.c_n ** 2 + t.s_n ** 2), 1.0, atol=1e-9)
    vocab, diag = cb.build_template_catalog(templates, 3, random_state=0,
                                            return_diagnostics=True)
    assert len(vocab) == 3
    assert sum(diag.cluster_sizes) == len(templates)


# ----------------------------------------------------------------------
# End-to-end: a built catalog drives the multiband periodogram (keystone)
# ----------------------------------------------------------------------
def _simulate_multiband(template, f0=0.123, amp=2.0,
                        band_offsets=(0.0, -0.6, 0.4), n_per_band=60,
                        noise=0.03, seed=1):
    rng = np.random.RandomState(seed)
    t, y, bands, dy = [], [], [], []
    for k, off in enumerate(band_offsets):
        tk = np.sort(rng.rand(n_per_band) * 80)
        sig = noise * (1 + 0.5 * k)
        yk = amp * template((f0 * tk) % 1.0) + off + sig * rng.randn(n_per_band)
        t.append(tk)
        y.append(yk)
        bands.append(np.full(n_per_band, k, dtype=int))
        dy.append(np.full(n_per_band, sig))
    t, y = np.concatenate(t), np.concatenate(y)
    bands, dy = np.concatenate(bands), np.concatenate(dy)
    order = np.random.RandomState(seed + 7).permutation(len(t))
    return t[order], y[order], bands[order], dy[order]


def test_catalog_drives_multiband_periodogram():
    from ftperiodogram import (FastMultibandTemplatePeriodogram,
                               build_template_catalog)
    templates, _ = _archetype_population(seed=1)
    vocab = build_template_catalog(templates, 3, random_state=0)

    # a held-out multiband light curve drawn from one archetype's shape
    arch = Template(*_ARCHETYPES[0])
    t, y, bands, dy = _simulate_multiband(arch, seed=5)

    model = FastMultibandTemplatePeriodogram(vocab, mode='floating_offsets')
    model.fit(t, y, bands, dy)
    f, p = model.autopower(minimum_frequency=0.05, maximum_frequency=0.30)

    # the injected frequency is recovered ...
    pk = int(np.argmax(p))
    assert abs(f[pk] - 0.123) < 1e-2
    # ... and the winning vocabulary template is the one orbit-closest to arch
    expected = int(np.argmin([cb._orbit_distance(v, arch) for v in vocab]))
    assert model.best_model.template_set_index == expected


def test_public_api_exports():
    import ftperiodogram
    for name in ('build_template_catalog', 'templates_from_sampled',
                 'fetch_sesar_templates', 'CatalogDiagnostics'):
        assert hasattr(ftperiodogram, name)
