import numpy as np
from numpy.testing import assert_allclose

from .. import utils

import pytest

@pytest.mark.parametrize('nterms', [1, 2, 3])
@pytest.mark.parametrize('sgn', [+1, -1])
def test_AB_derivs(nterms, sgn, eps=1E-8, rseed=42):
    rand = np.random.RandomState(rseed)
    c = rand.randn(nterms)
    s = rand.randn(nterms)
    x = np.linspace(0, 1, endpoint=False)[:, None]

    # Test that dA appropriately computes the derivative
    A2 = utils.Avec(x + eps, c, s, sgn)
    A1 = utils.Avec(x, c, s, sgn)
    dA = utils.dAvec(x, c, s, sgn)
    assert_allclose(dA, (A2 - A1) / eps, rtol=1E-5)

    # Test that dB computes the correct derivative
    B2 = utils.Bvec(x + eps, c, s, sgn)
    B1 = utils.Bvec(x, c, s, sgn)
    dB = utils.dBvec(x, c, s, sgn)
    assert_allclose(dA, (A2 - A1) / eps, rtol=1E-5)
