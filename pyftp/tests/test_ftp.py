import numpy as np
from numpy.testing import assert_allclose

from .. import fast_template_periodogram as ftp

import pytest


def M_slow(t, cos_phi, omega, cn, sn, sgn=1):
    cn, sn = np.broadcast_arrays(cn, sn)
    n = np.arange(1, len(sn) + 1)[:, None]
    phi = sgn * np.arccos(cos_phi)

    return (np.dot(cn, np.cos(n * (omega * t - phi))) +
            np.dot(sn, np.sin(n * (omega * t - phi))))


@pytest.mark.parametrize('nterms', [1, 2, 3])
@pytest.mark.parametrize('cos_phi', [1, 0, -0.5, -1])
@pytest.mark.parametrize('sgn', [+1, -1])
@pytest.mark.parametrize('omega', [0.9, 1.0, 1.1])
def test_M_basic(nterms, cos_phi, sgn, omega, rseed=42):
    # test that M correctly creates a truncated Fourier series
    rand = np.random.RandomState(rseed)
    t = np.linspace(0, 3, 10)
    cn = rand.randn(nterms)
    sn = rand.randn(nterms)

    M1 = M_slow(t, cos_phi, omega, cn, sn, sgn)
    M2 = ftp.M(t, cos_phi, omega, cn, sn, sgn)

    assert_allclose(M1, M2)


@pytest.mark.parametrize('nterms', [1, 2, 3])
@pytest.mark.parametrize('sgn', [+1, -1])
def test_AB_derivs(nterms, sgn, eps=1E-8, rseed=42):
    rand = np.random.RandomState(rseed)
    c = rand.randn(nterms)
    s = rand.randn(nterms)
    x = np.linspace(0, 1, endpoint=False)

    # Test that dA appropriately computes the derivative
    A2 = ftp.Avec(x + eps, c, s, sgn)
    A1 = ftp.Avec(x, c, s, sgn)
    dA = ftp.dAvec(x, c, s, sgn)
    assert_allclose(dA, (A2 - A1) / eps, rtol=1E-5)

    # Test that dB computes the correct derivative
    B2 = ftp.Bvec(x + eps, c, s, sgn)
    B1 = ftp.Bvec(x, c, s, sgn)
    dB = ftp.dBvec(x, c, s, sgn)
    assert_allclose(dA, (A2 - A1) / eps, rtol=1E-5)
