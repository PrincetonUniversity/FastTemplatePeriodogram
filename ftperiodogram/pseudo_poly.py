from __future__ import print_function

import numpy as np
from scipy.special import chebyu, chebyt
from scipy.optimize import newton
from numpy.polynomial.polynomial import Polynomial
import numpy.polynomial.polynomial as pol
from collections import namedtuple


one_minus_x_squared = Polynomial([1, 0, -1])
x_poly = Polynomial([ 0, 1 ])


class PseudoPolynomial(object):
    """
    Convenience class for doing algebra with rational functions
    containing factors of $\sqrt{1 - x^2}$ and with (1 - x^2)^(-r)
    in the denominator

    PP = (1 - x^2)^r * [p + (1 - x^2)^(1/2) * q]

    where p and q are simple polynomials, and r <= 0.

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
        
        if r > 0 or int(r) != r:
            raise ValueError("r must be a negative integer")

        self.r = r

        if isinstance(p, Polynomial):
            self.p = p
        else:
            self.p = np.atleast_1d(p)
            if self.p.ndim != 1:
                raise ValueError('p must be one-dimensional')
            if self.p.dtype == object:
                raise ValueError('p must be a numerical array')
            self.p = Polynomial(self.p)

        if isinstance(q, Polynomial):
            self.q = q
        else:
            self.q = np.atleast_1d(q)
            if self.q.ndim != 1:
                raise ValueError('q must be one-dimensional')
            if self.q.dtype == object:
                raise ValueError('q must be a numerical array')
            self.q = Polynomial(self.q)
        
    @classmethod
    def coerce(cls, obj):
        """Coerce an object into a PseudoPolynomial if possible"""
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, Polynomial):
            return cls(obj)
        else:
            coeff = np.atleast_1d(obj)
            if coeff.ndim == 1:
                return cls(coeff)
            else:
                raise ValueError("Object of type {0} cannot be coerced "
                                 "into a PseudoPolynomial".format(type(obj)))

    def __eq__(self, other):
        if not isinstance(other, PseudoPolynomial):
            raise NotImplementedError("comparison with {0}".format(type(other)))

        return (self.p.has_samecoef(other.p) and\
                self.q.has_samecoef(other.q) and\
                self.r == other.r)

    def __add__(self, other):
        other = self.coerce(other)

        if other.r < self.r:
            self, other = other, self
        p1, p2, q1, q2 = self.p, other.p, self.q, other.q
        r1, r2 = self.r, other.r

        p12 = p1 + p2 * one_minus_x_squared ** (r2 - r1)
        q12 = q1 + q2 * one_minus_x_squared ** (r2 - r1)
        r12 = r1

        return self.__class__(p12, q12, r12)

    def __radd__(self, other):
        # addition is commutative
        return self.__add__(other)

    def __mul__(self, other):
        other = self.coerce(other)

        p1, p2, q1, q2 = self.p, other.p, self.q, other.q
        r1, r2 = self.r, other.r
        

        p12 = p1 * p2 + one_minus_x_squared * q1 * q2
        q12 = p1 * q2 + q1 * p2
        r12 = r1 + r2

        return self.__class__(p12, q12, r12)

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
        return 'PseudoPolynomial(p=%s, q=%s, r=%s)'%(repr(self.p),
                                                     repr(self.q),
                                                     repr(self.r))

    def deriv(self):
        """ compute first derivative

        Returns
        -------
        dPP : PseudoPolynomial
            d(PP)/dx represented as a PseudoPolynomial
        """
        dp = self.p.deriv()
        dq = self.q.deriv()
        

        p = one_minus_x_squared * dp \
               -  2 * self.r * x_poly * self.p

        q = one_minus_x_squared * dq \
               - (2 * self.r + 1) * x_poly * self.q

        r = self.r - 1

        return PseudoPolynomial(p=p, q=q, r=r)

    def root_finding_poly(self):
        """
        R = p^2 - (1 - x^2) * q^2

        Returns
        -------
        prf : Polynomial
            Numpy `Polynomial` instance of a root-finding polynomial `R(x)`. 
            Every zero of the `PseudoPolynomial` is also a zero of `R(x)`.
        """
        p2 = self.p * self.p
        q2 = self.q * self.q
        return p2 - one_minus_x_squared * q2

    def _roots0(self, tol=1E-6, deflate_common_roots=True):
        """
        Complex roots of pseudo-poly using `np.polynomial.polyroots` method,
        which finds the eigenvalues of the companion matrix of the root-finding
        poly.

        First finds common roots between `p` and `q` polynomials, then divides
        these roots from the `root_finding_poly`.

        Parameters
        ----------
        tol : float, optional (default = 1E-6)
            Small number for comparing distance between roots. if abs(root1 - root2) < tol,
            they are considered equal
        deflate_common_roots: bool, optional (default=True)
            If true, the greatest common polynomial factor between `self.p` and `self.q` is
            divided out of the root-finding polynomial. This helps stabilize the root finding
            process when there are closely-spaced roots or roots with multiplicities > 1.
        

        Returns
        -------
        roots : list (of floats)
            (Complex) roots of the `root_finding_poly`
        p : Polynomial
            numpy `Polynomial` instance of the root finding polynomial
        
        """
        p  = self.root_finding_poly()
        roots = []
        prf = p

        if deflate_common_roots:
            roots_p = self.p.roots()
            roots_q = self.q.roots()
            for root in roots_p:
                if any([ abs(rq - root) < tol for rq in roots_q ]):
                    roots.append(root)

            pr = Polynomial([1])
            if len(roots) > 0:
                pr = Polynomial.fromroots(roots)

            pr *= pr

            prf = p / pr

        roots.extend(list(prf.roots()))
        return roots, p

    def complex_roots(self, tol=1E-6):
        """
        Finds all complex roots of the pseudo-polynomial,
        confirmed by testing that self(root) is close to
        zero at each root returned by `_roots0`

        Parameters
        ----------
        tol : float
            If `abs(self(root)) < tol`, `root` is considered a
            root of the polynomial

        Returns
        -------
        roots : array_like
            List of (unique) complex roots of the pseudopolynomial
        """
        roots0, p = self._roots0()

        roots = []
        for root in roots0:
            if abs(self(root)) < tol and \
                      not any([ abs(r - root) < tol for r in roots0 ]):
                roots.append(root)

        return roots

    def real_roots_pm(self, tol=1E-6):
        """
        Finds all real roots of the root-finding polynomial,
        without confirming that the `PseudoPolynomial` evaluates
        as zero at each root (i.e. these are complex roots of
        `p(x) +/- sqrt(1 - x^2)q(x)`

        Returns
        -------
        roots : array_like
            List of (unique) real roots of the root-finding poly
        """
        roots0, p = self._roots0()
        return [ r for r in roots0 if abs(r.imag) < tol ]

    def real_roots(self, tol=1E-8, use_newton=False):
        """
        Finds all real roots of the root-finding polynomial,
        confirming that each root is in fact a zero by evaluating `self(root)`.

        Parameters
        ----------
        tol: float, optional (default=1E-8)
            Small number for comparing roots and evaluations of the PseudoPolynomial.
        use_newton: bool, optional (default = False)
            If true, uses Halley's method to improve estimates of zeros.

        Returns
        -------
        roots : array_like
            List of (unique) real roots of the PseudoPolynomial
        """
        roots0, p = self._roots0()
        roots = []
        rroots = np.sort([ r.real for r in roots0])

        dpp, d2pp = None, None
        if use_newton:
            dpp = self.deriv()
            d2pp = dpp.deriv()

        for r in rroots:
            if abs(r) > 1:
                continue

            r_new = r

            if use_newton:
                try:
                    r_new = newton(self, r, fprime=dpp, fprime2=d2pp)

                except:
                    raise Warning("Newtons method failed to converge for root %.3e"%(r))

            
            if not abs(self(r_new)) < tol:
                continue

            if len(roots) == 0:
                roots.append(r_new)
                continue

            if abs(r_new - roots[-1]) > tol:
                roots.append(r_new)

        return roots

    def eval(self, x):
        """Evaluate the polynomial at the given value"""

        lmx2 = one_minus_x_squared(x)

        num = self.p(x) + np.sqrt(lmx2) * self.q(x)
        denom = lmx2 ** self.r

        return num * denom

    def conj(self):
        """ The 'conjugate' of the PseudoPolynomial: p(x) - sqrt(1-x^2)q(x) """
        return PseudoPolynomial(p=self.p, q=-self.q, r=self.r)

    def __call__(self, x):
        return self.eval(x)


# An (or Bn) as a PseudoPolynomial
def Afunc_pp(n, p, q):
    return PseudoPolynomial(p=p * np.array(chebyt(n).coef)[::-1],
                            q=-q * np.array(chebyu(n-1).coef)[::-1] \
                               if n > 0 else np.array([0]),
                            r=0)

# Vector A or B in PseudoPolynomial form
def ABpoly(c, s, kind):
    return [Afunc_pp(n + 1,
                     C if kind == 'A' else  S,
                     S if kind == 'A' else -C)
            for n, (C, S) in enumerate(zip(c, s))]

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
                P[i][j][k][:len(PPP.p.coef)] = PPP.p.coef[:]
                Q[i][j][k][:len(PPP.q.coef)] = PPP.q.coef[:]

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
    A = ABpoly(cn, sn, 'A')
    B = ABpoly(cn, sn, 'B')

    if sgn == -1:
        A = A.conj()
        B = B.conj()

    dA = [ a.deriv() for a in A ]
    dB = [ b.deriv() for b in B ]

    return A, B, dA, dB

def get_final_ppoly_components(ptensors, sums):
    """
    Calculates the `PseudoPolynomial` used to determine
    a set of candidates for the optimal phase-shift

    Parameters
    ----------
    ptensors: np.ndarray
        generated by :compute_polynomial_tensors: and contains
        coefficients unique to each template
    sums : Summations
        ordered dictionary containing CC, CS, SS, YC, YS

    Returns
    -------
    PPfinal : PseudoPolynomial
        PseudoPolynomial for which the zeros are the
        candidate phase shifts
    """
    AAdAp, AAdAq, \
    AAdBp, AAdBq, \
    ABdAp, ABdAq, \
    ABdBp, ABdBq, \
    BBdAp, BBdAq, \
    BBdBp, BBdBq = ptensors

    H = len(AAdAp)

    Kaada = np.einsum('i,jk->ijk', sums.YC[:H], sums.CC[:H,:H]) \
          - np.einsum('k,ij->ijk', sums.YC[:H], sums.CC[:H,:H])

    Kaadb = np.einsum('i,jk->ijk', sums.YC[:H], sums.CS[:H,:H]) \
          - np.einsum('k,ij->ijk', sums.YS[:H], sums.CC[:H,:H])

    Kabda = np.einsum('i,kj->ijk', sums.YC[:H], sums.CS[:H,:H]) \
          + np.einsum('j,ik->ijk', sums.YS[:H], sums.CC[:H,:H])

    Kabdb = np.einsum('i,jk->ijk', sums.YC[:H], sums.SS[:H,:H]) \
          + np.einsum('j,ik->ijk', sums.YS[:H], sums.CS[:H,:H])

    Kbbda = np.einsum('i,kj->ijk', sums.YS[:H], sums.CS[:H,:H]) \
          - np.einsum('k,ij->ijk', sums.YC[:H], sums.SS[:H,:H])

    Kbbdb = np.einsum('i,jk->ijk', sums.YS[:H], sums.SS[:H,:H]) \
          - np.einsum('k,ij->ijk', sums.YS[:H], sums.SS[:H,:H])

    p  = np.einsum('ijkl,ijk->l', AAdAp, Kaada)
    p += np.einsum('ijkl,ijk->l', AAdBp, Kaadb)
    p += np.einsum('ijkl,ijk->l', ABdAp, Kabda)
    p += np.einsum('ijkl,ijk->l', ABdBp, Kabdb)
    p += np.einsum('ijkl,ijk->l', BBdAp, Kbbda)
    p += np.einsum('ijkl,ijk->l', BBdBp, Kbbdb)

    q  = np.einsum('ijkl,ijk->l', AAdAq, Kaada)
    q += np.einsum('ijkl,ijk->l', AAdBq, Kaadb)
    q += np.einsum('ijkl,ijk->l', ABdAq, Kabda)
    q += np.einsum('ijkl,ijk->l', ABdBq, Kabdb)
    q += np.einsum('ijkl,ijk->l', BBdAq, Kbbda)
    q += np.einsum('ijkl,ijk->l', BBdBq, Kbbdb)

    return p, q

def get_final_roots_faster(p, q, tol=1E-6):

    p2 = pol.polymul(p, p)
    q2 = pol.polymul(q, q)

    P = pol.polysub(p2, pol.polymul(q2, (1, 0, -1)))

    roots = pol.polyroots(P)

    rroots = np.sort([ r.real for r in roots if not \
                 abs(r.imag) > tol and abs(r.real) < 1 - tol ]).tolist()
    
    nroots = []
    for r in rroots:
        if len(nroots) == 0 or not abs(r - nroots[-1]) < tol:
            nroots.append(r)
    return nroots

def get_final_ppoly(ptensors, sums):
    p, q = get_final_ppoly_components(ptensors, sums)
    return PseudoPolynomial(p=p, q=q, r=0)


def compute_zeros_multifrequency(ptensors, sums, tol=1E-6):
    
    """
    Compute real zeros for multiple frequencies simultaneously
    then find real roots

    Parameters
    ----------
    ptensors: np.ndarray
        generated by :compute_polynomial_tensors: and contains
        coefficients unique to each template
    sums : Summations
        ordered dictionary containing CC, CS, SS, YC, YS
    b_guess : float, optional
        Guess for the location of the optimal phase shift.
        Uses Newton's method to search around `+/-b_guess`
        for a root to the `PseudoPolynomial`
    tol : float (default : 1E-3)
        Tolerance value passed to `correct_real_roots`, only
        needed if `b_guess` is specified.

    Returns
    -------
    roots: list
        List of real roots of the `PseudoPolynomial`
    """

    AAdAp, AAdAq, \
    AAdBp, AAdBq, \
    ABdAp, ABdAq, \
    ABdBp, ABdBq, \
    BBdAp, BBdAq, \
    BBdBp, BBdBq = ptensors

    H = len(AAdAp)

    nf = len(sums)

    YC = np.ravel([ s.YC[:] for s in sums ]).reshape((nf, H))
    YS = np.ravel([ s.YS[:] for s in sums ]).reshape((nf, H))
    CC = np.ravel([ s.CC[:,:] for s in sums ]).reshape((nf, H, H))
    CS = np.ravel([ s.CS[:,:] for s in sums ]).reshape((nf, H, H))
    SS = np.ravel([ s.SS[:,:] for s in sums ]).reshape((nf, H, H))


    Kaada = np.einsum('fi,fjk->fijk', YC, CC) \
          - np.einsum('fk,fij->fijk', YC, CC)

    Kaadb = np.einsum('fi,fjk->fijk', YC, CS) \
          - np.einsum('fk,fij->fijk', YS, CC)

    Kabda = np.einsum('fi,fkj->fijk', YC, CS) \
          + np.einsum('fj,fik->fijk', YS, CC)

    Kabdb = np.einsum('fi,fjk->fijk', YC, SS) \
          + np.einsum('fj,fik->fijk', YS, CS)

    Kbbda = np.einsum('fi,fkj->fijk', YS, CS) \
          - np.einsum('fk,fij->fijk', YC, SS)

    Kbbdb = np.einsum('fi,fjk->fijk', YS, SS) \
          - np.einsum('fk,fij->fijk', YS, SS)

    p  = np.einsum('ijkl,fijk->fl', AAdAp, Kaada)
    p += np.einsum('ijkl,fijk->fl', AAdBp, Kaadb)
    p += np.einsum('ijkl,fijk->fl', ABdAp, Kabda)
    p += np.einsum('ijkl,fijk->fl', ABdBp, Kabdb)
    p += np.einsum('ijkl,fijk->fl', BBdAp, Kbbda)
    p += np.einsum('ijkl,fijk->fl', BBdBp, Kbbdb)

    q  = np.einsum('ijkl,fijk->fl', AAdAq, Kaada)
    q += np.einsum('ijkl,fijk->fl', AAdBq, Kaadb)
    q += np.einsum('ijkl,fijk->fl', ABdAq, Kabda)
    q += np.einsum('ijkl,fijk->fl', ABdBq, Kabdb)
    q += np.einsum('ijkl,fijk->fl', BBdAq, Kbbda)
    q += np.einsum('ijkl,fijk->fl', BBdBq, Kbbdb)


    all_roots = []
    for P, Q in zip(p, q):

        p2 = pol.polymul(P, P)
        q2 = pol.polymul(Q, Q)

        Pf = pol.polysub(p2, pol.polymul(q2, (1, 0, -1)))

        roots = pol.polyroots(Pf)

        rroots = np.sort([ r.real for r in roots if \
               not abs(r.imag) > tol \
               and abs(r.real) < 1 - tol ])
    
        nroots = []
        for r in rroots:
            if len(nroots) == 0 or not abs(r - nroots[-1]) < tol:
                nroots.append(r)
        all_roots.append(nroots)

    return all_roots


def compute_zeros(ptensors, sums, tol=1E-6):
    AAdAp, AAdAq, \
    AAdBp, AAdBq, \
    ABdAp, ABdAq, \
    ABdBp, ABdBq, \
    BBdAp, BBdAq, \
    BBdBp, BBdBq = ptensors

    H = len(AAdAp)

    Kaada = np.einsum('i,jk->ijk', sums.YC, sums.CC) \
          - np.einsum('k,ij->ijk', sums.YC, sums.CC)

    Kaadb = np.einsum('i,jk->ijk', sums.YC, sums.CS) \
          - np.einsum('k,ij->ijk', sums.YS, sums.CC)

    Kabda = np.einsum('i,kj->ijk', sums.YC, sums.CS) \
          + np.einsum('j,ik->ijk', sums.YS, sums.CC)

    Kabdb = np.einsum('i,jk->ijk', sums.YC, sums.SS) \
          + np.einsum('j,ik->ijk', sums.YS, sums.CS)

    Kbbda = np.einsum('i,kj->ijk', sums.YS, sums.CS) \
          - np.einsum('k,ij->ijk', sums.YC, sums.SS)

    Kbbdb = np.einsum('i,jk->ijk', sums.YS, sums.SS) \
          - np.einsum('k,ij->ijk', sums.YS, sums.SS)

    p  = np.einsum('ijkl,ijk->l', AAdAp, Kaada)
    p += np.einsum('ijkl,ijk->l', AAdBp, Kaadb)
    p += np.einsum('ijkl,ijk->l', ABdAp, Kabda)
    p += np.einsum('ijkl,ijk->l', ABdBp, Kabdb)
    p += np.einsum('ijkl,ijk->l', BBdAp, Kbbda)
    p += np.einsum('ijkl,ijk->l', BBdBp, Kbbdb)

    q  = np.einsum('ijkl,ijk->l', AAdAq, Kaada)
    q += np.einsum('ijkl,ijk->l', AAdBq, Kaadb)
    q += np.einsum('ijkl,ijk->l', ABdAq, Kabda)
    q += np.einsum('ijkl,ijk->l', ABdBq, Kabdb)
    q += np.einsum('ijkl,ijk->l', BBdAq, Kbbda)
    q += np.einsum('ijkl,ijk->l', BBdBq, Kbbdb)

    p2 = pol.polymul(p, p)
    q2 = pol.polymul(q, q)

    P = pol.polysub(p2, pol.polymul(q2, (1, 0, -1)))

    roots = pol.polyroots(P)

    rroots = np.sort([ r.real for r in roots if \
                   not abs(r.imag) > tol \
                   and abs(r.real) < 1 - tol ])
    
    nroots = []
    for r in rroots:
        if len(nroots) == 0 or not abs(r - nroots[-1]) < tol:
            nroots.append(r)
    return nroots