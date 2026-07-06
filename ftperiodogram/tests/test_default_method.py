"""WP C4 pins: the scan+polish maximizer is the DEFAULT method;
``method='eigvals'`` is retained permanently as the reference/validation
mode. These tests pin (a) that the default really is the scan path, and
(b) default == eigvals on a fixed well-conditioned fixture set, so the two
implementations can never drift apart silently.

Scoping (VERIFICATION.md WP C3 finding 4b + WP C3.5): the two-sided
default == eigvals equivalence is guaranteed on well-conditioned data --
the fixtures here are screened to circle conditioning min|MM|/max|MM|
above 1e-3. At deep |MM| dips the single-band exact fallback is
equal-treatment (not bitwise; up to ~1e-8 at conditioning ~1e-9) and
multiband shared_phase is ONE-SIDED by design post-C3.5
(default/scan >= eigvals) -- pinned separately below.
"""
import numpy as np
import pytest

from ..core import (YM_MM_from_sums, _eval_polys_on_circle,
                    template_periodogram)
from ..modeler import FastTemplatePeriodogram
from ..multiband import DEFAULT_METHOD, FastMultibandTemplatePeriodogram
from ..summations import direct_summations
from ..template import Template
from ..utils import weights
from .test_scan_polish import (_deepdip_multiband_fixture, _simulate,
                               _simulate_multiband)

GATE_TOL = 1e-12
PIN_HARMONICS = [2, 5, 8]
PIN_SEED = 7


def _assert_well_conditioned(t, y, dy, template, freqs, floor=1e-3):
    """Screen the fixture: every frequency's circle conditioning must be
    comfortably outside the deep-dip regime, so the two-sided pin below is
    within its guaranteed scope."""
    H = len(template.c_n)
    w = weights(dy)
    M = max(128, 32 * H)
    for s in direct_summations(t, y, w, freqs, H):
        _, MM, _ = YM_MM_from_sums(template.c_n, template.s_n, s)
        absM = np.abs(_eval_polys_on_circle(MM.coef[np.newaxis], M))
        assert float(np.min(absM) / np.max(absM)) > floor


def test_default_method_constant_is_scan():
    assert DEFAULT_METHOD == 'scan'


@pytest.mark.parametrize('H', PIN_HARMONICS)
def test_default_is_scan_single_band(H):
    """The no-kwarg default output is bitwise the method='scan' output."""
    t, y, dy, template, freqs = _simulate(H, 40, PIN_SEED)
    p_def, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                    freqs)
    p_scan, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                     freqs, method='scan')
    assert np.array_equal(p_def, p_scan)


@pytest.mark.parametrize('H', PIN_HARMONICS)
def test_default_matches_eigvals_single_band(H):
    """default == eigvals to the gate on screened well-conditioned data,
    with identical argmax -- the permanent anti-drift pin."""
    t, y, dy, template, freqs = _simulate(H, 40, PIN_SEED)
    _assert_well_conditioned(t, y, dy, template, freqs)
    p_def, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                    freqs)
    p_eig, _ = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                    freqs, method='eigvals')
    assert float(np.max(np.abs(p_def - p_eig))) <= GATE_TOL
    assert int(np.argmax(p_def)) == int(np.argmax(p_eig))


def test_modeler_autopower_default_is_scan():
    t, y, dy, template, _ = _simulate(3, 40, PIN_SEED)
    ftp = FastTemplatePeriodogram(template=template).fit(t, y, dy)
    kw = dict(minimum_frequency=0.5, maximum_frequency=3.0,
              samples_per_peak=3, save_best_model=False)
    _, p_def = ftp.autopower(**kw)
    _, p_scan = ftp.autopower(method='scan', **kw)
    _, p_eig = ftp.autopower(method='eigvals', **kw)
    assert np.array_equal(p_def, p_scan)
    assert float(np.max(np.abs(p_def - p_eig))) <= GATE_TOL


@pytest.mark.parametrize('mode', ['independent', 'shared_phase',
                                  'floating_offsets', 'sesar'])
def test_default_multiband_is_scan_and_matches_eigvals(mode):
    """All four sharing modes: the default is bitwise the scan path, and
    matches eigvals to the gate on well-conditioned (40/band) fixtures."""
    t, y, bands, dy, tmpl, freqs, labels = _simulate_multiband(4, 40,
                                                               PIN_SEED)
    kw = ({'relative_offsets': {b: 0.5 * j for j, b in enumerate(labels)}}
          if mode == 'sesar' else {})
    m = FastMultibandTemplatePeriodogram(templates=tmpl, mode=mode,
                                         **kw).fit(t, y, bands, dy)
    p_def = m.power(freqs, fast=False, save_best_model=False)
    p_scan = m.power(freqs, fast=False, save_best_model=False,
                     method='scan')
    p_eig = m.power(freqs, fast=False, save_best_model=False,
                    method='eigvals')
    assert np.array_equal(p_def, p_scan)
    assert float(np.max(np.abs(p_def - p_eig))) <= GATE_TOL


def test_default_one_sided_at_shared_phase_deep_dips():
    """At deep dips the default (scan) is one-sided vs the eigvals
    reference by design (C3.5 max-merge): never below, may recover the
    reference's documented residual deficit."""
    t, y, bands, dy, tmpl = _deepdip_multiband_fixture(40005)
    m = FastMultibandTemplatePeriodogram(
        templates=tmpl, mode='shared_phase').fit(t, y, bands, dy)
    freqs = np.linspace(0.8, 1.2, 25)
    p_def = m.power(freqs, fast=False, save_best_model=False)
    p_eig = m.power(freqs, fast=False, save_best_model=False,
                    method='eigvals')
    assert np.all(p_def >= p_eig - 1e-15)
    # non-vacuity: the known deficit rows are genuinely recovered
    assert float(np.max(p_def - p_eig)) >= 1e-5
