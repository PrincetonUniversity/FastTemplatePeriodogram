import numpy as np
from numpy.polynomial import polynomial as pol
from numpy.testing import assert_allclose
from ..pseudo_poly import PseudoPolynomial

import pytest


def random_polynomial(Np=1, Nq=1, r=0, rseed=42):
    rand = np.random.RandomState(rseed)
    return PseudoPolynomial(p=rand.randint(-5, 5, Np),
                            q=rand.randint(-5, 5, Nq),
                            r=r)


@pytest.mark.parametrize('r', [0, -1, -2])
def test_pseudopoly_eval(r, rseed=42):
    rand = np.random.RandomState(rseed)
    x = rand.rand(10)
    p = rand.randint(-5, 5, 3)
    q = rand.randint(-5, 5, 2)
    rfac_p = (1 - x ** 2) ** r
    rfac_q = (1 - x ** 2) ** (r + 0.5)

    poly_p = pol.Polynomial(p)
    poly_q = pol.Polynomial(q)

    # empty
    poly = PseudoPolynomial(r=r)
    assert_allclose(poly(x), 0)

    # p only
    poly = PseudoPolynomial(p=p, r=r)
    assert_allclose(poly(x), rfac_p * poly_p(x))

    # q only
    poly = PseudoPolynomial(q=q, r=r)
    assert_allclose(poly(x), rfac_q * poly_q(x))

    # p and q
    poly = PseudoPolynomial(p=p, q=q, r=r)
    assert_allclose(poly(x), rfac_p * poly_p(x) + rfac_q * poly_q(x))


@pytest.mark.parametrize('r', [0, -1, -2])
def test_pseudopoly_addition(r, rseed=42):
    # polynomial + polynomial, same r value
    p1 = random_polynomial(2, 3, r=r, rseed=rseed)
    p2 = random_polynomial(2, 3, r=r, rseed=rseed + 1)
    p12 = PseudoPolynomial(p1.p + p2.p, p1.q + p2.q, r)
    assert p1 + p2 == p12


@pytest.mark.parametrize('r', [0, -1, -2])
def test_pseudopoly_subtraction(r, rseed=42):
    # polynomial - polynomial, same r value
    p1 = random_polynomial(2, 3, r=r, rseed=rseed)
    p2 = random_polynomial(2, 3, r=r, rseed=rseed + 1)
    p12 = PseudoPolynomial(p1.p - p2.p, p1.q - p2.q, r)
    assert p1 - p2 == p12


@pytest.mark.parametrize('r', [0, -1, -2])
@pytest.mark.parametrize('A', [0, -1, 2.5])
def test_pseudopoly_multiplication(A, r, rseed=42):
    # float * polynomial
    poly = random_polynomial(2, 3, r=r, rseed=rseed)
    poly2 = PseudoPolynomial(A * poly.p, A * poly.q, r=r)

    assert A * poly == poly2
    assert poly * A == poly2

    # polynomial * pseudopoly
    A = [1, 3, 2, 4]
    result = PseudoPolynomial(p=pol.polymul(A, poly.p),
                              q=pol.polymul(A, poly.q),
                              r=poly.r)
    assert A * poly == result
    assert poly * A == result


@pytest.mark.parametrize('r1', [0, -1, -2])
@pytest.mark.parametrize('r2', [0, -1, -2])
@pytest.mark.parametrize('Np1', [1, 2])
@pytest.mark.parametrize('Np2', [1, 2, 3])
@pytest.mark.parametrize('Nq1', [1, 2])
@pytest.mark.parametrize('Nq2', [1, 2, 3])
def test_pseudopoly_operations(r1, r2, Np1, Np2, Nq1, Nq2, rseed=42):
    rand = np.random.RandomState(rseed)
    x = rand.rand(10)
    A = rand.randn()

    poly1 = random_polynomial(Np1, Nq1, r1, rseed=rand.randint(1000))
    poly2 = random_polynomial(Np2, Nq2, r2, rseed=rand.randint(1000))

    op = A * poly1 - poly2
    assert_allclose(op(x), A * poly1(x) - poly2(x))

    op = poly1 - A
    assert_allclose(op(x), poly1(x) - A)

    op = poly1 * poly2
    assert_allclose(op(x), poly1(x) * poly2(x))
