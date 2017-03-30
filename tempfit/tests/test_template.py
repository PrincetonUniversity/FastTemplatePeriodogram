import numpy as np
from ..template import Template

from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize('nharmonics', [1, 2, 3, 4])
def test_template_round_trip(nharmonics, rseed=42):
    rng = np.random.RandomState(rseed)

    c_n, s_n = rng.randn(2, nharmonics)
    template = Template(c_n, s_n)

    phase = np.linspace(0, 1, 100, endpoint=False)
    y = template(phase)
    template2 = Template.from_sampled(y, nharmonics)

    assert_allclose(c_n, template2.c_n)
    assert_allclose(s_n, template2.s_n)


def test_template_infer_harmonics(rseed=42):
    rng = np.random.RandomState(rseed)
    c_n, s_n = rng.randn(2, 10)
    template = Template(c_n, s_n)
    phase = np.linspace(0, 1, 100, endpoint=False)
    y = template(phase)

    # Make sure we recover the right number of harmonics
    variance = np.cumsum(c_n ** 2 + s_n ** 2)
    eps = 1E-3

    for i, nharmonics in enumerate(variance[:-1] / variance[-1]):
        template = Template.from_sampled(y, nharmonics + eps)
        assert len(template.c_n) == i + 1
