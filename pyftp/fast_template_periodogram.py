""" 
FAST TEMPLATE PERIODOGRAM (prototype)

Uses NFFT to make the template periodogram scale as H*N log(H*N) 
where H is the number of harmonics in which to expand the template and
N is the number of observations.

Previous routines scaled as N^2 and used non-linear least-squares
minimization (e.g. Levenberg-Marquardt) at each frequency.

(c) John Hoffman 2016

"""

import numpy as np
from math import *
from scipy.special import eval_chebyt, eval_chebyu, chebyu, chebyt
from pynfft.nfft import NFFT
from numpy.polynomial.polynomial import Polynomial
import numpy.polynomial.polynomial as pol
from scipy.optimize import curve_fit
from numbers import Number, Integral
import sys
import os
from time import time

# shortcuts for the Chebyshev polynomials
Un = lambda n, x : eval_chebyu(n, x) if n >= 0 else 0
Tn = lambda n, x : eval_chebyt(n, x) if n >= 0 else 0

# A and dA expressions
Afunc    = lambda n, x, p, q, positive=True :      \
                       p * Tn(n  , x) - (1 if positive else -1) \
                                         * q * Un(n-1, x) * np.sqrt(1 - min([ 1, x*x ])) 

dAfunc   = lambda n, x, p, q, positive=True : \
                       n * (p * Un(n-1, x) + (1 if positive else -1) \
                       	                 * q * Tn(n  , x) / np.sqrt(1 - min([ 1, x*x ])))



# returns vector expressions of A, B and their derivatives
Avec        = lambda x, c, s, positive=True :  np.array([  \
	                      Afunc(n, x, c[n-1],  s[n-1], positive=positive) \
	                             for n in range(1, len(s)+1) ])
Bvec        = lambda x, c, s, positive=True :  np.array([ \
                          Afunc(n, x, s[n-1], -c[n-1], positive=positive) \
                                 for n in range(1, len(s)+1) ])
dAvec       = lambda x, c, s, positive=True :  np.array([ \
	                     dAfunc(n, x, c[n-1],  s[n-1], positive=positive) \
	                             for n in range(1, len(s)+1) ])
dBvec       = lambda x, c, s, positive=True :  np.array([ \
	                     dAfunc(n, x, s[n-1], -c[n-1], positive=positive) \
	                             for n in range(1, len(s)+1) ])

def M(t, b, w, cn, sn, positive=True):
	""" evaluate the shifted template at a given time """

	A = Avec(b, cn, sn, positive=positive)
	B = Bvec(b, cn, sn, positive=positive)
	Xc = np.array([ np.cos(-n * w * t) \
		              for n in range(1, len(cn)+1) ], dtype=np.float64)
	Xs = np.array([ np.sin(-n * w * t) \
		              for n in range(1, len(cn)+1) ], dtype=np.float64)

	return np.dot(A, Xc) + np.dot(B, Xs)

def pdg_ftp(a, b, cn, sn, YY, YC, YS, positive=True):
	""" evaluate the periodogram for a given a, b """
	A = Avec(b, cn, sn, positive=positive)
	B = Bvec(b, cn, sn, positive=positive)

	return (a / YY) * (np.dot(A, YC) + np.dot(B, YS))

def fitfunc(x, positive, w, cn, sn, a, b, c):
	""" aM(t - tau) + c """
	return a * np.array(map(lambda X : M(X, b, w, cn, sn, \
		                        positive=positive), x)) + c

def weights(err):
	""" converts sigma_i -> w_i \equiv (1/W) * (1/sigma_i^2) """
	w = np.power(err, -2)
	w/= np.sum(w)
	return w

def get_a_from_b(b, cn, sn, sums, A=None, B=None, positive=True):
	""" return the optimal amplitude & offset for a given value of b """

	C, S, YC, YS, CCh, CSh, SSh = sums

	if A is None:
		A = Avec(b, cn, sn, positive=positive)
	if B is None:
		B = Bvec(b, cn, sn, positive=positive)

	a = np.dot(A, YC) + np.dot(B, YS)
	a /= (np.dot(A, np.dot(CCh, A)) + 2 * np.dot(A, np.dot(CSh, B)) \
		  + np.dot(B, np.dot(SSh, B)))

	#c = -a * (np.dot(A, C) + np.dot(B, S))

	return a


def shift_t_for_nfft(t, ofac):
	""" transforms times to [-1/2, 1/2] interval """

	r = ofac * (max(t) - min(t))
	eps = 1E-5
	a = 0.5 - eps

	return 2 * a * (t - min(t)) / r - a

def compute_summations(x, y, err, H, ofac=5, hfac=1):
	""" 
	Computes C, S, YC, YS, CC, CS, SS using
	pyNFFT
	"""
	# convert errs to weights
	w = weights(err)

	# number of frequencies (+1 for 0 freq)
	N = int(floor(0.5 * len(x) * ofac * hfac))
	
	# shift times to [ -1/2, 1/2 ]
	t = shift_t_for_nfft(x, ofac)

	# compute angular frequencies
	T = max(x) - min(x)
	df = 1. / (ofac * T)
	#print df, N*df

	omegas = np.array([ i * 2 * np.pi * df for i in range(1, N) ])
	
	# compute weighted mean
	ybar = np.dot(w, y)

	# subtract off weighted mean
	u = np.multiply(w, y - ybar)

	# weighted variance
	YY = np.dot(w, np.power(y-ybar, 2))

	# plan NFFT's and precompute
	plan = NFFT(4 * H * N, len(x))
	plan.x = t
	plan.precompute()

	plan2 = NFFT(2 * H * N, len(x))
	plan2.x = t
	plan2.precompute()

	# evaluate NFFT for w
	plan.f = w
	f_hat_w = plan.adjoint()[2 * H * N + 1:]

	# evaluate NFFT for y - ybar
	plan2.f = u
	f_hat_u = plan2.adjoint()[H * N + 1:]

	summations = []
	# Now compute the summation values at each frequency
	for i in range(N-1):
		C_ = np.zeros(2 * H)
		S_ = np.zeros(2 * H)
		YC_ = np.zeros(H)
		YS_ = np.zeros(H)
		CC_ = np.zeros((H, H))
		CS_ = np.zeros((H, H))
		SS_ = np.zeros((H, H))


		for j in range(2 * H):
			# This sign factor is necessary 
			# but I don't know why.
			s = (-1)**(i+1) if (j % 2 == 0) else 1.0
			C_[j] =  f_hat_w[(j+1)*(i+1)-1].real * s
			S_[j] =  f_hat_w[(j+1)*(i+1)-1].imag * s
			if j < H:
				YC_[j] =  f_hat_u[(j+1)*(i+1)-1].real * s
				YS_[j] =  f_hat_u[(j+1)*(i+1)-1].imag * s

		for j in range(H):
			for k in range(H):
				
				Sn, Cn = None, None

				if j == k:
					Sn = 0
					Cn = 1
				elif k > j:
					Sn =  S_[k - j - 1] 
					Cn =  C_[k - j - 1]
				else:
					Sn = -S_[j - k - 1]
					Cn =  C_[j - k - 1]

				Sp = S_[j + k + 1]
				Cp = C_[j + k + 1]
				
				CC_[j][k] = 0.5 * ( Cn + Cp ) - C_[j]*C_[k]
				CS_[j][k] = 0.5 * ( Sn + Sp ) - C_[j]*S_[k]
				SS_[j][k] = 0.5 * ( Cn - Cp ) - S_[j]*S_[k]


		summations.append((C_[:H], S_[:H], YC_, YS_, CC_, CS_, SS_))
		
	return omegas, summations, YY, w, ybar

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

	def __getstate__(self):
		return dict(p=self.p, q=self.q, r=self.r)

	def __setstate__(self, state):
		self.r = state['r']
		self.q = state['q']
		self.p = state['p']

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


	def deriv(self):
		""" compute first derivative

		Returns
		-------
		dPP : PseudoPolynomial
			The d(PP)/dx as another PseudoPolynomial
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
                                        p=   p * np.array(chebyt(n)), 
	                                    q= - sgn * q * np.array(chebyu(n-1)) \
	                                           if n > 0 else np.array([0]),
	                                    r=   0)

# Vector A or B in PseudoPolynomial form
ABpoly = lambda c, s, sgn, alpha : [ Afunc_pp(n, C if alpha == 0 else  S, 
	                                             S if alpha == 0 else -C, sgn) \
                                       for n, (C, S) in enumerate(zip(c, s)) ]



def pseudo_poly_tensor(P1, P2, P3):
	""" 
	compute coefficients of all products of P1, P2, P3, where
	P1, P2, and P3 are vectors of PseudoPolynomials
	"""
	H = len(P1)
	L = 3 * H - 1
	P, Q = np.zeros((H, H, H, L)), np.zeros((H, H, H, L))
	for i, p1 in enumerate(P1):
		for j, p2 in enumerate(P2):
			PP = p1 * p2
			for k, p3 in enumerate(P3):
				PPP = PP * p3

				P[i][j][k][:len(PPP.p)] = PPP.p[:]
				Q[i][j][k][:len(PPP.q)] = PPP.q[:]

	return P, Q


def compute_polynomial_tensors(A, B, dA, dB):
	"""
	returns coefficients of all 
	
	(A or B)_n * (A or B)_m * d(A or B)_k, 
	
	pseudo polynomial products
	"""

	AAdBp, AAdBq = pseudo_poly_tensor(A, A, dB)
	ABdAp, ABdAq = pseudo_poly_tensor(A, B, dA)
	ABdBp, ABdBq = pseudo_poly_tensor(A, B, dB)
	BBdAp, BBdAq = pseudo_poly_tensor(B, B, dA)
	
	return AAdBp, AAdBq, ABdAp, ABdAq, ABdBp, ABdBq, BBdAp, BBdAq

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


def compute_zeros(ptensors, summations):
	""" Computes frequency-dependent polynomial coefficients """
	C, S, YC, YS, CCh, CSh, SSh = summations
	AAdBp, AAdBq, ABdAp, ABdAq, ABdBp, ABdBq, BBdAp, BBdAq = ptensors

	Pp  = np.tensordot(np.tensordot(AAdBp, YC, axes=(0, 0)), 
		                                  CSh, axes=([0, 1], [0, 1]))
	Pp -= np.tensordot(np.tensordot(AAdBp, YS, axes=(2, 0)), 
		                                  CCh, axes=([0, 1], [0, 1]))

	Pp += np.tensordot(np.tensordot(ABdAp, YC, axes=(0, 0)), 
		                                  CSh, axes=([0, 1], [1, 0]))
	Pp += np.tensordot(np.tensordot(ABdAp, YS, axes=(1, 0)), 
		                                  CCh, axes=([0, 1], [0, 1]))

	Pp += np.tensordot(np.tensordot(ABdBp, YC, axes=(0, 0)), 
		                                  SSh, axes=([0, 1], [0, 1]))
	Pp += np.tensordot(np.tensordot(ABdBp, YS, axes=(1, 0)), 
		                                  CSh, axes=([0, 1], [0, 1]))

	Pp += np.tensordot(np.tensordot(BBdAp, YS, axes=(0, 0)), 
		                                  CSh, axes=([0, 1], [1, 0]))
	Pp -= np.tensordot(np.tensordot(BBdAp, YC, axes=(2, 0)), 
		                                  SSh, axes=([0, 1], [0, 1]))


	Pq  = np.tensordot(np.tensordot(AAdBq, YC, axes=(0, 0)), 
		                                  CSh, axes=([0, 1], [0, 1]))
	Pq -= np.tensordot(np.tensordot(AAdBq, YS, axes=(2, 0)), 
		                                  CCh, axes=([0, 1], [0, 1]))

	Pq += np.tensordot(np.tensordot(ABdAq, YC, axes=(0, 0)), 
		                                  CSh, axes=([0, 1], [1, 0]))
	Pq += np.tensordot(np.tensordot(ABdAq, YS, axes=(1, 0)), 
		                                  CCh, axes=([0, 1], [0, 1]))

	Pq += np.tensordot(np.tensordot(ABdBq, YC, axes=(0, 0)), 
		                                  SSh, axes=([0, 1], [0, 1]))
	Pq += np.tensordot(np.tensordot(ABdBq, YS, axes=(1, 0)), 
		                                  CSh, axes=([0, 1], [0, 1]))

	Pq += np.tensordot(np.tensordot(BBdAq, YS, axes=(0, 0)), 
		                                  CSh, axes=([0, 1], [1, 0]))
	Pq -= np.tensordot(np.tensordot(BBdAq, YC, axes=(2, 0)), 
		                                  SSh, axes=([0, 1], [0, 1]))


	P = pol.polysub(pol.polymul((1, 0, -1), pol.polypow(Pp, 2)), pol.polypow(Pq, 2)) 
	R = pol.polyroots(P)

	return R

def fastTemplatePeriodogram(x, y, err, cn, sn, ofac=10, hfac=1, 
	                          polytens_p=None, polytens_n=None, 
	                          pvectors_p=None, pvectors_n=None):
	if pvectors_p is None:
		pvectors_p = get_polynomial_vectors(cn, sn, sgn=  1)
	if pvectors_n is None:
		pvectors_n = get_polynomial_vectors(cn, sn, sgn= -1)
	
	if polytens_p is None:
		polytens_p = compute_polynomial_tensors(*pvectors_p)
	if polytens_n is None:
		polytens_n = compute_polynomial_tensors(*pvectors_n)

	# compute sums using NFFT
	omegas, summations, YY, w, ybar = \
		compute_summations(x, y, err, len(cn), ofac=ofac, hfac=hfac)
	
	FTP = np.zeros(len(omegas))
	for i, (omega, sums) in enumerate(zip(omegas, summations)):

		afromb = lambda B, p : get_a_from_b(B, cn, sn, sums, positive=p)
		
		# Get zeros for sin(wt) > 0
		Bpz = compute_zeros(polytens_p, sums)
		# ''            sin(wt) < 0
		Bnz = compute_zeros(polytens_n, sums)
		
		# get a from b (solve rest of non-linear system of equations)
		Z = []
		for Bz, p in [ (Bpz, True), (Bnz, False) ]:
			for bz in Bz:
				if bz.imag != 0:
					continue
				Az = afromb(bz.real, p)
				if Az < 0:
					continue

				Z.append((Az, bz.real, p))
		
		# Compute periodogram values
		Pz = [ pdg_ftp( Av, Bv, cn, sn, YY, sums[2], sums[3], positive=p) \
		            for Av, Bv, p in Z ]

		# Periodogram value is the global max of P_{-} and P_{+}.
		FTP[i] = 0 if len(Pz) == 0 else max(Pz)

	return omegas / (2 * np.pi), FTP
	
