import numpy as np
from numpy.testing import assert_allclose

from .. import utils

import pytest


def test_get_diags(n=10):
    m = np.random.rand(n * n).reshape((n, n))

    diags_manual = np.zeros(2 * n)

    for i in range(len(diags_manual)):
        for j in range(i + 1):
            if i < n:
                diags_manual[i] += m[i-j][j]
            else:
                diags_manual[i] += m[(n - 1) - j][i - (n - 1) + j]

    np.testing.assert_allclose(diags_manual, utils.get_diags(m))
