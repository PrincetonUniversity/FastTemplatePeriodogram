import numpy as np
from numpy.polynomial import polynomial as pol
from numpy.testing import assert_allclose
from ..pseudo_poly import PseudoPolynomial
from numpy.polynomial.polynomial import Polynomial
import pytest


def random_pseudo_poly(Np=1, Nq=1, r=0, rseed=42):
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

    poly_p = Polynomial(p)
    poly_q = Polynomial(q)

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
    pp1 = random_pseudo_poly(2, 3, r=r, rseed=rseed)
    pp2 = random_pseudo_poly(2, 3, r=r, rseed=rseed + 1)
    pp12 = PseudoPolynomial(pp1.p + pp2.p, pp1.q + pp2.q, r)
    assert pp1 + pp2 == pp12


@pytest.mark.parametrize('r', [0, -1, -2])
def test_pseudopoly_subtraction(r, rseed=42):
    # polynomial - polynomial, same r value
    pp1 = random_pseudo_poly(2, 3, r=r, rseed=rseed)
    pp2 = random_pseudo_poly(2, 3, r=r, rseed=rseed + 1)
    pp12 = PseudoPolynomial(pp1.p - pp2.p, pp1.q - pp2.q, r)
    assert pp1 - pp2 == pp12


@pytest.mark.parametrize('r', [0, -1, -2])
@pytest.mark.parametrize('A', [0, -1, 2.5])
def test_pseudopoly_multiplication(A, r, rseed=42):
    # float * polynomial
    ppoly = random_pseudo_poly(2, 3, r=r, rseed=rseed)
    
    #A = Polynomial([1, 3, 2, 4])

    ppoly2 = PseudoPolynomial(A * ppoly.p, A * ppoly.q, r=r)

    assert A * ppoly == ppoly2
    assert ppoly * A == ppoly2

    # polynomial * pseudopoly
    B = [1, 2, 3, 4]
    result = PseudoPolynomial(p=(Polynomial(B) * ppoly.p),
                              q=(Polynomial(B) * ppoly.q),
                              r=ppoly.r)


    assert B * ppoly == result
    assert ppoly * B == result


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

    ppoly1 = random_pseudo_poly(Np1, Nq1, r1, rseed=rand.randint(1000))
    ppoly2 = random_pseudo_poly(Np2, Nq2, r2, rseed=rand.randint(1000))

    op = A * ppoly1 - ppoly2
    assert_allclose(op(x), A * ppoly1(x) - ppoly2(x))

    op = ppoly1 - A
    assert_allclose(op(x), ppoly1(x) - A)

    op = ppoly1 * ppoly2
    assert_allclose(op(x), ppoly1(x) * ppoly2(x))


@pytest.mark.parametrize('r', [0, -1, -2])
@pytest.mark.parametrize('Np', [1, 2])
@pytest.mark.parametrize('Nq', [1, 2])
def test_pseudopoly_root_finding(r, Np, Nq, rseed=42):
    rand = np.random.RandomState(rseed)

    poly = random_pseudo_poly(Np, Nq, r, rseed=rand.randint(1000))

    zeros = poly.complex_roots()

    pzeros = [ abs(poly(z)) for z in zeros ]

    assert_allclose(pzeros, np.zeros_like(pzeros), atol=1E-9)

@pytest.mark.parametrize('r', [0, -1, -2])
@pytest.mark.parametrize('delta', [ 1E-5, 1E-3, 1E-1 ])
@pytest.mark.parametrize('nroots', [ 1, 2, 3, 4, 5, 10 ])
def test_pathological_roots(r, nroots, delta, rseed=42, tol=1E-5):
    rand = np.random.RandomState(rseed)
    roots = [ root for root in rand.rand(max([int(nroots/2), 1])) ]
    for i in range(nroots - len(roots)):
        rt = roots[i] + np.sign(0.5 - rand.rand()) * delta
        if abs(rt) < 1:
            roots.append(rt)

    roots = np.sort(roots)
    p = pol.polyfromroots(roots)
    q = pol.polyfromroots(roots)

    pp = PseudoPolynomial(p=p, q=q, r=r)

    pproots = np.sort(pp.real_roots(use_newton=False))

    print("original roots: ", roots)
    print("pp roots      : ",pproots)

    proots = pol.polyroots(p)
    qroots = pol.polyroots(q)

    for root in proots:
        dr = min(np.absolute(np.array(roots) - root))
        assert(dr < tol)

    for root in qroots:
        dr = min(np.absolute(np.array(roots) - root))
        assert(dr < tol)

    for root in pproots:
        assert( abs(pp(root)) < tol)

    for root in roots:
        dr = min(np.absolute(np.array(pproots) - root) / max([ tol, np.absolute(root) ]))
        assert( abs(pp(root)) < tol and dr < tol )
