from __future__ import print_function

import numpy as np
from time import time
from scipy.special import chebyu, chebyt
from numbers import Number, Integral
from scipy.optimize import newton, brentq
from numpy.polynomial.polynomial import Polynomial
import numpy.polynomial.polynomial as pol

def remove_zeros(p, tol=1E-10):
    for i, coeff in enumerate(p):
        if abs(coeff) < tol:
            p[i] = 0.
        
    return p


class PseudoPolynomial(object):
    """
    Convenience class for doing algebra with rational functions
    containing factors of $\sqrt{1 - x^2}$ and with (1 - x^2)^(-r)
    in the denominator

    PP = (1 - x^2)^r * [polynomial(coeffs_1)
                        + (1 - x^2)^(1/2) * polynomial(coeffs_2)]

    Parameters
    ----------
    p : np.ndarray
        Coefficients of polynomial (1)
    q : np.ndarray
        Coefficients of polynomial (2)
    r : int <= 0
        Factor in $(1 - x^2)^r * (p(x) + sqrt(1 - x^2)*q(x))$.

    """
    def __init__(self, p=0, q=0, r=0):
        self.p = remove_zeros(np.atleast_1d(p))
        self.q = remove_zeros(np.atleast_1d(q))
        self.r = r

        if r > 0 or int(r) != r:
            raise ValueError("r must be a negative integer")

        if self.p.ndim != 1:
            raise ValueError('p must be one-dimensional')
        if self.p.dtype == object:
            raise ValueError('p must be a numerical array')

        if self.q.ndim != 1:
            raise ValueError('q must be one-dimensional')
        if self.q.dtype == object:
            raise ValueError('q must be a numerical array')


    @classmethod
    def coerce(cls, obj):
        """Coerce an object into a PseudoPolynomial if possible"""
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, Polynomial):
            return cls(obj.coef)
        else:
            return cls(obj)
            obj_arr = np.atleast_1d(obj)
            if obj_arr.ndim == 1:
                return cls(obj_arr)
            else:
                raise ValueError("Object of type {0} cannot be coerced "
                                 "into a PseudoPolynomial".format(type(obj)))

    def __eq__(self, other):
        if not isinstance(other, PseudoPolynomial):
            raise NotImplementedError("comparison with {0}".format(type(other)))

        return (np.all(self.p == other.p) and
                np.all(self.q == other.q) and
                (self.r == other.r))

    def __add__(self, other):
        other = self.coerce(other)

        if other.r < self.r:
            self, other = other, self
        p1, p2, q1, q2 = map(Polynomial, (self.p, other.p, self.q, other.q))
        r1, r2 = self.r, other.r
        one_minus_x_squared = Polynomial([1, 0, -1])

        p12 = p1 + p2 * one_minus_x_squared ** (r2 - r1)
        q12 = q1 + q2 * one_minus_x_squared ** (r2 - r1)
        r12 = r1

        return self.__class__(remove_zeros(p12.coef), remove_zeros(q12.coef), r12)

    def __radd__(self, other):
        # addition is commutative
        return self.__add__(other)

    def __mul__(self, other):
        other = self.coerce(other)

        p1, p2, q1, q2 = map(Polynomial, (self.p, other.p, self.q, other.q))
        r1, r2 = self.r, other.r
        one_minus_x_squared = Polynomial([1, 0, -1])

        p12 = p1 * p2 + one_minus_x_squared * q1 * q2
        q12 = p1 * q2 + q1 * p2
        r12 = r1 + r2

        return self.__class__(remove_zeros(p12.coef), remove_zeros(q12.coef), r12)

    def __rmul__(self, other):
        # multiplication is commutative
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return (-1) * self

    def __pos__(self):
        return 1 * self

    def __repr__(self):
        return 'PseudoPolynomial(p=%s, q=%s, r=%s)'%(repr(self.p), repr(self.q), repr(self.r))

    def deriv(self):
        """ compute first derivative

        Returns
        -------
        dPP : PseudoPolynomial
            d(PP)/dx represented as a PseudoPolynomial
        """
        dp = pol.polyder(self.p)
        dq = pol.polyder(self.q)


        p = pol.polysub(pol.polymul((1, 0, -1), dp), pol.polymul((0, 2 * self.r), self.p))
        q = pol.polysub(pol.polymul((1, 0, -1), dq), pol.polymul((0, 2 * self.r + 1), self.q))
        r = self.r - 1

        return PseudoPolynomial(p=remove_zeros(p), q=remove_zeros(q), r=r)

    def root_finding_poly(self):
        """ p^2 - (1 - x^2) * q^2

        Returns
        -------
        coef : np.ndarray
            Coefficients of a polynomial that has the
            same number of roots as the PP

        """
        return  remove_zeros(pol.polysub(pol.polymul(self.p, self.p),
                            pol.polymul(pol.polymul(self.q, self.q), (1, 0, -1))))

    def _roots0(self):
        p  = self.root_finding_poly()
        dp = remove_zeros(pol.polyder(p))


        roots_p = pol.polyroots(self.p)
        roots_q = pol.polyroots(self.q)

        roots = []
        for root in roots_p:
            if any([ abs(rq - root) < 1E-5 for rq in roots_q ]):
                roots.append(root)


        pr    = remove_zeros(pol.polyfromroots(roots))
        p2, _ = pol.polydiv(p, pol.polymul(pr, pr))
        p2    = remove_zeros(p2)

        new_roots = list(pol.polyroots(p2))
        
        roots.extend(new_roots)

        return roots, p, dp

    def complex_roots(self):
        roots0, p, dp = self._roots0()

        roots = []
        for root in roots0:
            if abs(self(root)) < 1E-9:
                roots.append(root)

        return roots

    def real_roots_pm(self, use_newton=False):
        roots0, p, dp = self._roots0()

        f = lambda x, p=p : pol.polyval( x, p)
        fprime = lambda x, dp=dp : pol.polyval( x, dp )

        return correct_real_roots(roots0, f, fprime=fprime, use_newton=use_newton)

    def real_roots(self, use_newton=False):
        roots0, p, dp = self._roots0()

        f = lambda x, p=p : pol.polyval( x, p)
        fprime = lambda x, dp=dp : pol.polyval( x, dp )

        criterion = lambda x, P=self.p, Q=self.q : pol.polyval(x, P) * pol.polyval(x, Q) < 1E-8

        return correct_real_roots(roots0, f, fprime=fprime, criterion=criterion, use_newton=use_newton)

    def eval(self, x, tol=1E-4):

        #if (not hasattr(x, '__iter__') and abs(x) >= 1) \
        #    or (hasattr(x, '__iter__') and any(np.absolute(x) >= 1)):
        #    raise RuntimeError("Cannot evaluate PseudoPolynomial outside of (-1, 1).")

        p, q = pol.Polynomial(self.p), pol.Polynomial(self.q)
        #lmx2 = 1 if abs(x) < tol * tol else 1 - np.power(x, 2)

        lmx2 = 1 - np.power(x, 2)

        num = p(x) + np.sqrt(lmx2) * q(x)
        denom = lmx2 ** self.r

        return num * denom

    def __call__(self, x):
        return self.eval(x)


# An (or Bn) as a PseudoPolynomial
Afunc_pp = lambda n, p, q, sgn : PseudoPolynomial(   \
                                        p=         p * np.array(chebyt(n).coef)[::-1],
                                        q= - sgn * q * np.array(chebyu(n-1).coef)[::-1] \
                                               if n > 0 else np.array([0]),
                                        r=   0)

# Vector A or B in PseudoPolynomial form
ABpoly = lambda c, s, sgn, kind : [ Afunc_pp(n+1, C if kind == 'A' else  S,
                                                  S if kind == 'A' else -C, sgn) \
                                       for n, (C, S) in enumerate(zip(c, s)) ]

# Hardcoded, should probably be double checked but this
# empirically works
get_poly_len = lambda H : 6 * H + 2


def pseudo_poly_tensor(P1, P2, P3):
    """
    Compute coefficients of all products of P1, P2, P3

    Parameters
    ----------
    P1: list of PseudoPolynomial
        Usually A or B (represented as PseudoPolynomial)
    P2: list of PseudoPolynomial
    P3: list of PseudoPolynomial
        Usually d(A)/dx or d(B)/dx

    Returns
    -------
    P: np.ndarray, shape=(H,H,H,L)
        L is defined by `get_poly_len`. (P) Polynomial coefficients
        for outer product of all 3 vectors of PseudoPolynomials
    Q: np.ndarray, shape=(H,H,H,L)
        L is defined by `get_poly_len`. (Q) Polynomial coefficients
        for outer product of all 3 vectors of PseudoPolynomials

    """
    H = len(P1)
    L = get_poly_len(H)
    P, Q = np.zeros((H, H, H, L)), np.zeros((H, H, H, L))
    for i, p1 in enumerate(P1):
        for j, p2 in enumerate(P2):
            PP = p1 * p2
            for k, p3 in enumerate(P3):
                PPP = PP * p3

                #lpolp = max([ i for i in range(L) if abs(PPP.p[i]) > 1E-9 ])
                #lpolq = max([ i for i in range(L) if abs(PPP.q[i]) > 1E-9 ])
                #print len(P[i][j][k]), len(PPP.p), len(PPP.q)
                P[i][j][k][:len(PPP.p)] = PPP.p[:]
                Q[i][j][k][:len(PPP.q)] = PPP.q[:]

    return P, Q


def compute_polynomial_tensors(A, B, dA, dB):
    """
    returns coefficients of all

    (A or B)_n * (A or B)_m * d(A or B)_k,

    pseudo polynomial products
    """
    AAdAp, AAdAq = pseudo_poly_tensor(A, A, dA)
    AAdBp, AAdBq = pseudo_poly_tensor(A, A, dB)
    ABdAp, ABdAq = pseudo_poly_tensor(A, B, dA)
    ABdBp, ABdBq = pseudo_poly_tensor(A, B, dB)
    BBdAp, BBdAq = pseudo_poly_tensor(B, B, dA)
    BBdBp, BBdBq = pseudo_poly_tensor(B, B, dB)

    return AAdAp, AAdAq, AAdBp, AAdBq, ABdAp, ABdAq, \
           ABdBp, ABdBq, BBdAp, BBdAq, BBdBp, BBdBq

def get_polynomial_vectors(cn, sn, sgn=1):
    """
    returns list of PseudoPolynomials corresponding to
    A_n, B_n, and their derivatives
    """
    A = ABpoly(cn, sn, sgn, 'A')
    B = ABpoly(cn, sn, sgn, 'B')

    dA = [ a.deriv() for a in A ]
    dB = [ b.deriv() for b in B ]

    return A, B, dA, dB

def correct_real_roots(roots0, func, fprime=None,use_newton=True, criterion=None, tol=1E-5):
    """ 
    Uses Newton's method to determine roots of `func`
    with `roots0` as initial guesses.

    """

    corr_roots = []
    check = criterion if not criterion is None else lambda x : True
    
    for r0 in roots0:
        if abs(r0) > 1 - tol:
            continue


        if abs(func(r0.real)) < tol:
            if not any([ abs(r0.real - cr) < 1E-6 for cr in corr_roots ]):
                if not check(r0.real):
                    continue
                corr_roots.append(r0.real)
            continue

        if use_newton:
            try: 
                nz = newton(func, r0.real, maxiter=50, fprime=fprime)
                all_bad = False
                if not any([ abs(nz - cr) < tol for cr in corr_roots ]):
                    if not check(nz):
                        continue
                    corr_roots.append(nz)

            except RuntimeError:
                corr_roots.append(r0.real)
        else:
            if abs(r0.imag) < tol:
                corr_roots.append(r0.real)

    return corr_roots


def get_final_ppoly(ptensors, sums):

    AAdAp, AAdAq, \
    AAdBp, AAdBq, \
    ABdAp, ABdAq, \
    ABdBp, ABdBq, \
    BBdAp, BBdAq, \
    BBdBp, BBdBq = ptensors

    H = len(AAdAp)

    """
    Kaada = np.einsum('i,jk->ijk', sums.YC[:H], sums.CC[:H,:H]) \
          - np.einsum('k,ji->ijk', sums.YC[:H], sums.CC[:H,:H])

    Kaadb = np.einsum('i,jk->ijk', sums.YC[:H], sums.CS[:H,:H]) \
          - np.einsum('k,ji->ijk', sums.YS[:H], sums.CC[:H,:H])

    Kabda = np.einsum('i,kj->ijk', sums.YC[:H], sums.CS[:H,:H]) \
          + np.einsum('j,ki->ijk', sums.YS[:H], sums.CC[:H,:H])

    Kabdb = np.einsum('i,jk->ijk', sums.YC[:H], sums.SS[:H,:H]) \
          + np.einsum('j,ik->ijk', sums.YS[:H], sums.CS[:H,:H])

    Kbbda = np.einsum('i,kj->ijk', sums.YS[:H], sums.CS[:H,:H]) \
          - np.einsum('k,ij->ijk', sums.YC[:H], sums.SS[:H,:H])

    Kbbdb = np.einsum('i,jk->ijk', sums.YS[:H], sums.SS[:H,:H]) \
          - np.einsum('k,ji->ijk', sums.YS[:H], sums.SS[:H,:H])
    """

    Kaada = np.einsum('i,jk->ijk', sums.YC[:H], sums.CC[:H,:H]) - np.einsum('k,ij->ijk', sums.YC[:H], sums.CC[:H,:H])
    Kaadb = np.einsum('i,jk->ijk', sums.YC[:H], sums.CS[:H,:H]) - np.einsum('k,ij->ijk', sums.YS[:H], sums.CC[:H,:H])
    Kabda = np.einsum('i,kj->ijk', sums.YC[:H], sums.CS[:H,:H]) + np.einsum('j,ik->ijk', sums.YS[:H], sums.CC[:H,:H])
    Kabdb = np.einsum('i,jk->ijk', sums.YC[:H], sums.SS[:H,:H]) + np.einsum('j,ik->ijk', sums.YS[:H], sums.CS[:H,:H])
    Kbbda = np.einsum('i,kj->ijk', sums.YS[:H], sums.CS[:H,:H]) - np.einsum('k,ij->ijk', sums.YC[:H], sums.SS[:H,:H])
    Kbbdb = np.einsum('i,jk->ijk', sums.YS[:H], sums.SS[:H,:H]) - np.einsum('k,ij->ijk', sums.YS[:H], sums.SS[:H,:H])


    Pp  = np.einsum('ijkl,ijk->l', AAdAp, Kaada)
    Pp += np.einsum('ijkl,ijk->l', AAdBp, Kaadb)
    Pp += np.einsum('ijkl,ijk->l', ABdAp, Kabda)
    Pp += np.einsum('ijkl,ijk->l', ABdBp, Kabdb)
    Pp += np.einsum('ijkl,ijk->l', BBdAp, Kbbda)
    Pp += np.einsum('ijkl,ijk->l', BBdBp, Kbbdb)

    Pq  = np.einsum('ijkl,ijk->l', AAdAq, Kaada)
    Pq += np.einsum('ijkl,ijk->l', AAdBq, Kaadb)
    Pq += np.einsum('ijkl,ijk->l', ABdAq, Kabda)
    Pq += np.einsum('ijkl,ijk->l', ABdBq, Kabdb)
    Pq += np.einsum('ijkl,ijk->l', BBdAq, Kbbda)
    Pq += np.einsum('ijkl,ijk->l', BBdBq, Kbbdb)
    
    #P = pol.polysub(pol.polymul(Pp, Pp), pol.polymul(pol.polymul(Pq, Pq), (1, 0, -1)))
    PP = PseudoPolynomial(p=Pp, q=Pq, r=0)


    return PP


def compute_zeros(ptensors, sums, old_b=None, loud=False, small = 1E-5):
    """
    Compute frequency-dependent polynomial coefficients,
    then find real roots

    Parameters
    ----------
    ptensors: np.ndarray
        generated by :compute_polynomial_tensors: and contains
        coefficients unique to each template
    sums : Summations
        ordered dictionary containing CC, CS, SS, YC, YS
    loud : bool (default: False)
        Print timing information

    Returns
    -------
    roots: list
        list of cos(omega * tau) values corresponding to
        (real) roots of the generated polynomial.

    """
    
    PP = get_final_ppoly(ptensors, sums)

    if not old_b is None:
        p = pol.Polynomial(PP.root_finding_poly())
        pprime = p.deriv()
        roots0 = [ -old_b, old_b ]
        return correct_real_roots(roots0, p, fprime=pprime, tol=1E-3)
    
    return PP.real_roots_pm()

    