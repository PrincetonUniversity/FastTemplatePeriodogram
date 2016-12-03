import numpy as np
from time import time
from scipy.special import chebyu, chebyt
from numbers import Number, Integral
from scipy.optimize import newton, brentq
from numpy.polynomial.polynomial import Polynomial
import numpy.polynomial.polynomial as pol


# ZERO FINDING UTILITIES. #################################################
# Using Sturm chains and bracketing zeros
#    this method is slower than the numpy polyroots() function that uses
#    the eigenvalues of the polynomial companion matrix.

def linzero(xlo, xhi, ylo, yhi):
	""" approximate the location of the zero contained within
	    (xlo, xhi)
	"""
	m = (yhi - ylo) / (xhi - xlo)
	b = ylo - m * xlo
	return -b/m

def hone_in(p, lo, hi, stop, count, max_count=10):
	""" improve estimate of the location of a zero
	    by iteratively applying the secant method
	    and refining the bracket
	"""

	y_hi = pol.polyval(hi, p)
	y_lo = pol.polyval(lo, p)

	if y_hi * y_lo >= 0:
		raise Exception("y_lo and y_hi need different signs.")

	zero = linzero(lo, hi, y_lo, y_hi)

	fz = pol.polyval(zero, p)
	if zero - lo < stop or hi - zero < stop \
	           or fz == 0 or count == max_count:
		return zero, count

	if fz * y_lo < 0:
		return hone_in(p, lo, zero, stop, count+1, max_count)

	else:
		return hone_in(p, zero, hi, stop, count+1, max_count)

def secant_zero(p, lo, hi, acc=1E-5):
	z, c = hone_in(p, lo, hi, acc, 0, max_count=100)
	return z
	
def brent_zero(p, lo, hi, acc=1E-5):
	f = lambda x, p=p : pol.polyval(x, p)
	return brentq(f, lo, hi)

def newton_zero(p, lo, hi, acc=1E-5):

	ylo = pol.polyval(lo, p)
	yhi = pol.polyval(hi, p)

	x0 = linzero(lo, hi, ylo, yhi)

	dp = pol.polyder(p)
	f = lambda x, p=p : pol.polyval(x, p)
	df = lambda x, dp=dp : pol.polyval(x, dp)

	return newton(f, x0, fprime=df)

def rem(p, d):
	return pol.polydiv(p, d)[1]

def n_sign_changes(p):
	n = 0
	for i in range(len(p) - 1):
		if p[i] * p[i+1] < 0: n += 1
	return n

def sigma(schain, x):
	return n_sign_changes([ pol.polyval(x, p) for p in schain ])

def sturm_chain(p):
	# Sturm chains:
	#   p0, p1, p2, ..., pm
	#
	#   p0 = p
	#   p1 = dp/dx
	#   p2 = -rem(p0, p1)
	#   p3 = -rem(p1, p2)
	#     ...
	#    0 = -rem(p(m-1), pm)
	#
	chains = [ p, pol.polyder(p) ]
	while True:
		pn = -rem(chains[-2], chains[-1])
		if len(pn) == 1 and abs(pn[0]) < 1E-14:
			break
		chains.append(pn)
	return chains

def bisect(a, b):
	d = 0.5 * (b - a)
	return [ (a, a + d), (a + d, b) ]

def sturm_zeros(p, a, b, acc=1E-5, zero_finder=brent_zero):
	chains = sturm_chain(p)

	brackets = [ (a, b) ]
	zeros = []
	while len(brackets) > 0:
		#print brackets
		x1, x2 = brackets.pop()
		if x2 - x1 < acc:
			zeros.append(x1)
			continue

		# use Sturm's theorem to get # of distinct
		# real zeros within [ x1, x2 ]
		n = sigma(chains, x1) - sigma(chains, x2)
		#print n, x1, x2

		if n == 0:
			continue
		elif n == 1 and pol.polyval(x1, p) * pol.polyval(x2, p) < 0:
			zeros.append(zero_finder(p, x1, x2, acc=acc))
		else:
			brackets.extend(bisect(x1, x2))

	return zeros

class PseudoPolynomial(object):
	""" 
	Convenience class for doing algebra with polynomials
	containing factors of $\sqrt{1 - x^2}$

	PP = polynomial(coeffs_1) 
	      + (1 - x^2)^(r + 1/2) * polynomial(coeffs_2)

	Parameters
	----------
	p : np.ndarray
		Coefficients of polynomial (1)
	q : np.ndarray
		Coefficients of polynomial (2)
	r : int <= 0
		Factor in $\sqrt{1 - x^2}^{r + 1/2}$.

	"""
	def __init__(self, p=None, q=None, r=None):

		# Uses arrays instead of Polynomial instances
		# for better performance
		assert(p is None or isinstance(p, np.ndarray))
		assert(q is None or isinstance(q, np.ndarray))
		assert(r is None or isinstance(r, Integral))

		assert(r <= 0)

		self.p = np.array([0]) if p is None else p
		self.q = np.array([0]) if q is None else q
		self.r = 0 if r is None else r

	def __add__(self, PP):

		if not (isinstance(PP, type(self)) or isinstance(PP, np.ndarray)):
			raise TypeError("Can only add Polynomial or PseudoPolynomial "
							"to another PseudoPolynomial, not %s"%(str(type(PP))))
		PP_ = PP
		if not isinstance(PP, type(self)):
			PP_ = PseudoPolynomial(p=PP)

		p1, p2 = (self, PP_) if self.r <= PP_.r else (PP_, self)
		
		x = 1 if p1.r == p2.r else pol.polypow((1, 0, -1), p2.r - p1.r)

		p = pol.polyadd(self.p, PP.p)
		q = pol.polyadd(p1.q, pol.polymul(x, p2.q))
		r = p1.r

		return PseudoPolynomial(p=p, q=q, r=r)


	def __mul__(self, PP):

		if isinstance(PP, np.ndarray) or isinstance(PP, Number):
			return PseudoPolynomial(p=pol.polymul(self.p, PP), 
				                    q=pol.polymul(self.q, PP), 
				                    r=self.r)
		
		if not isinstance(PP, type(self)):
			raise TypeError("Can only multiply PseudoPolynomial "
			                "by a number, numpy Polynomial, or "
			                "another PseudoPolynomial, not %s"%(str(type(PP))))

		p1, p2 = (self, PP) if self.r <= PP.r else (PP, self)

		x1 = pol.polypow((1, 0, -1), p1.r + p2.r + 1)
		x2 = 1 if p2.r == p1.r else pol.polypow((1, 0, -1), p2.r - p1.r)

		p = pol.polyadd(pol.polymul(p1.p, p2.p), 
			            pol.polymul(pol.polymul(x1, p1.q), p2.q))
		q = pol.polyadd(pol.polymul(pol.polymul(x2,p1.p), p2.q), 
			            pol.polymul(p2.p, p1.q))
		r = p1.r

		return PseudoPolynomial(p=p, q=q, r=r)


	def __sub__(self, PP):
		return self.__add__(PP * (-1))

	def __repr__(self):
		return 'PseudoPolynomial(p=%s, q=%s, r=%s)'%(repr(self.p), repr(self.q), repr(self.r))

	def deriv(self):
		""" compute first derivative

		Returns
		-------
		dPP : PseudoPolynomial
			d(PP)/dx represented as a PseudoPolynomial
		"""
		p = pol.polyder(self.p)
		q = pol.polysub(pol.polymul((1, 0, -1), 
			                        pol.polyder(self.q)), 
		                (2 * self.r + 1) * pol.polymul((0, 1),  self.q))
		r = self.r - 1

		return PseudoPolynomial(p=p, q=q, r=r)


	def root_finding_poly(self):
		""" (1 - x^2)^(-(2 * r + 1)) * p^2 - q^2 

		Returns
		-------
		coef : np.ndarray
			Coefficients of a polynomial that has the
			same number of roots as the PP

		"""
		return  pol.polysub(pol.polymul(pol.polypow(self.p, 2), 
			                            pol.polypow((1, 0, -1), 
			                            	       -(2 * self.r + 1))),  
		                    pol.polypow(self.q, 2))

	def roots(self):
		return self.root_finding_poly().roots()

	def eval(self, x):
		return pol.polyval(x, self.p) + pow(1 - x*x, self.r + 0.5) \
		                                * pol.polyval(x, self.q)


# An (or Bn) as a PseudoPolynomial
Afunc_pp = lambda n, p, q, sgn : PseudoPolynomial(   \
                                        p=         p * np.array(chebyt(n).coef)[::-1], 
	                                    q= - sgn * q * np.array(chebyu(n-1).coef)[::-1] \
	                                           if n > 0 else np.array([0]),
	                                    r=   0)

# Vector A or B in PseudoPolynomial form
ABpoly = lambda c, s, sgn, alpha : [ Afunc_pp(n+1, C if alpha == 0 else  S, 
	                                             S if alpha == 0 else -C, sgn) \
                                       for n, (C, S) in enumerate(zip(c, s)) ]

# Hardcoded, should probably be double checked but this 
# empirically works
get_poly_len = lambda H : 6 * H - 2

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
	A = ABpoly(cn, sn, sgn, 0)
	B = ABpoly(cn, sn, sgn, 1)

	dA = [ a.deriv() for a in A ]
	dB = [ b.deriv() for b in B ]

	return A, B, dA, dB



def compute_zeros(ptensors, sums, loud=False):
	""" 
	Compute frequency-dependent polynomial coefficients, 
	then find real roots 
		
	Parameters
	----------
	ptensors: np.ndarray
		generated by :compute_polynomial_tensors: and contains
		coefficients unique to each template
	summations: tuple or array-like
		C, S, YC, YS, CChat, CShat, SShat; see documentation for
		definitions of these quantities.
	loud: bool (default: False)
		Print timing information

	Returns
	-------
	roots: list
		list of cos(omega * tau) values corresponding to
		(real) roots of the generated polynomial.

	"""
	t0 = None
	if loud: t0 = time()

	AAdAp, AAdAq, \
	AAdBp, AAdBq, \
	ABdAp, ABdAq, \
	ABdBp, ABdBq, \
	BBdAp, BBdAq, \
	BBdBp, BBdBq = ptensors

	H = len(AAdAp)

	if loud:
		dt = time() - t0
		print "   ", dt, " seconds for bookkeeping"

	if loud: t0 = time()
	Kaada = np.einsum('i,jk->ijk', sums.YC[:H], sums.CCh[:H,:H]) - np.einsum('k,ij->ijk', sums.YC, sums.CCh[:H,:H])
	Kaadb = np.einsum('i,jk->ijk', sums.YC[:H], sums.CSh[:H,:H]) - np.einsum('k,ij->ijk', sums.YS, sums.CCh[:H,:H])
	Kabda = np.einsum('i,kj->ijk', sums.YC[:H], sums.CSh[:H,:H]) + np.einsum('j,ik->ijk', sums.YS, sums.CCh[:H,:H])
	Kabdb = np.einsum('i,jk->ijk', sums.YC[:H], sums.SSh[:H,:H]) + np.einsum('j,ik->ijk', sums.YS, sums.CSh[:H,:H])
	Kbbda = np.einsum('i,kj->ijk', sums.YS[:H], sums.CSh[:H,:H]) - np.einsum('k,ij->ijk', sums.YC, sums.SSh[:H,:H])
	Kbbdb = np.einsum('i,jk->ijk', sums.YS[:H], sums.SSh[:H,:H]) - np.einsum('k,ij->ijk', sums.YS, sums.SSh[:H,:H])
	
	if loud: 
		dt = time() - t0
		print "   ", dt, " seconds to make constants"


	# Note: the first and last einsums for both Pp and Pq might not be necessary.
	#       see the docs for more information

	if loud: t0 = time()
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
	if loud:
		dt = time() - t0
		print "   ", dt, " seconds to make coefficients of pseudo-polynomial"

	if loud: t0 = time()
	P = pol.polysub(pol.polymul((1, 0, -1), pol.polymul(Pp, Pp)), pol.polymul(Pq, Pq))
	if loud:
		dt = time() - t0
		print "   ",dt, " seconds to get final polynomial"

	if loud: t0 = time()

	#c = max(np.absolute(P))
	#c = 1./c if c > 0 else 1.0
	#if P[-1] < 0: c *= -1
	c = 1.0
	R = pol.polyroots(np.array(P) * c)
	#R = sturm_zeros(P, -1, 1)
	if loud:
		dt = time() - t0
		print "   ", dt, " seconds to find roots of polynomial"

	return [ r.real for r in R if abs(r.imag) < 1E-6 and abs(r.real) < 1 ]