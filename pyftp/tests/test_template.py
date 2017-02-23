import numpy as np
from ..modeler import Template
from numpy.testing import assert_allclose


def test_template_from_coefficients():
    rng = np.random.RandomState(42)
    cn = rng.randn(4)
    sn = rng.randn(4)

    t = Template(cn, sn)
    t.precompute()

    assert_allclose(t.phase, np.linspace(0, 1, 100))
