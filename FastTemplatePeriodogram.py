""" 
FAST TEMPLATE PERIODOGRAM (prototype)

Uses NFFT to make the template periodogram scale as H*N log(H*N) 
where H is the number of harmonics to expand the template in and
N is the number of observations.

Previous routines scaled as N^2 and used non-linear least-squares
minimization (e.g. Levenberg-Marquardt) at each frequency.

(c) John Hoffman 2016

"""

import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt, eval_chebyu, chebyu, chebyt
from pynfft.nfft import NFFT
from gatspy.periodic import LombScargleFast, RRLyraeTemplateModeler
import gatspy.datasets.rrlyrae as rrl
from gatspy import datasets
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
                                         * q * Un(n-1, x) * np.sqrt(1 - x*x) \
                                         * (0 if x * x > 1 else 1) 
dAfunc   = lambda n, x, p, q, positive=True : \
                       n * (p * Un(n-1, x) + (1 if positive else -1) \
                       	                 * q * Tn(n  , x) / np.sqrt(1 - x*x))\
                       					 * (0 if x * x > 1 else 1) 



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



# ZERO FINDING UTILITIES. #################################################
def linzero(xlo, xhi, ylo, yhi):
	""" approximate the location of the zero contained within
	    (xlo, xhi)
	"""
	m = (yhi - ylo) / (xhi - xlo)
	b = ylo - m * xlo
	return -b/m

def hone_in(lo, hi, stop, func, count, max_count):
	""" improve estimate of the location of a zero
	    by iteratively applying the secant method
	    and refining the bracket
	"""

	y_hi = func(hi)
	y_lo = func(lo)

	if y_hi * y_lo >= 0:
		raise Exception("y_lo and y_hi need different signs.")

	zero = linzero(lo, hi, y_lo, y_hi)

	fz = func(zero)
	if zero - lo < stop or hi - zero < stop \
	           or fz == 0 or count == max_count:
		return zero

	if fz * y_lo < 0:
		return hone_in(lo, zero, stop, func, count+1, max_count)

	else:
		return hone_in(zero, hi, stop, func, count+1, max_count)


def find_zeros(x, y, func=None, stop=1E-6, max_count=100):
	""" find all zeros of a function over (min(x), max(x))
	    if func is None, just uses one evaluation of the secant
	    method. Otherwise, improves iteratively by evaluating the function
	"""
	inds = [ i for i in range(len(y) - 1) if y[i] * y[i+1] < 0 ]
	zeros = []
	for i in inds:
		if not func is None:
			try:
				z = hone_in(x[i], x[i+1], stop, func, 0, max_count)
			except RuntimeError, e:
				raise Warning("iterative zero finder broke, using only "
							  "first approximation (i={i}, x0={x0}, "
							  "x1={x1})".format(i=i, x0=x[i], x1=x[i+1]))

				z = linzero(x[i], x[i+1], y[i], y[i+1])
			
		else:
			z = linzero(x[i], x[i+1], y[i], y[i+1])
			

		zeros.append(z)
	return zeros
# ZERO FINDING UTILITIES. #################################################

def shift_t_for_nfft(t, ofac):
	""" transforms times to [-1/2, 1/2] interval """

	r = ofac * (max(t) - min(t))
	eps = 1E-5
	a = 0.5 - eps

	return 2 * a * (t - min(t)) / r - a

def compute_summations(x, y, err, H, ofac=5, hfac=1):

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

def get_a_from_b_poly(pvectors, summations):

	A, B, dA, dB = pvectors
	C, S, YC, YS, CCh, CSh, SSh = summations

	N_p   = np.zeros(len(A))
	D_p   = np.zeros(2 * len(A))

	N_q   = np.zeros(len(A))
	D_q   = np.zeros(2 * len(A))

	N_r = A[1].r
	D_r = (A[1] * A[1]).r

	H = len(A)
	for n, (An, Bn) in enumerate(zip(A, B)):
		#print An.p
		ANP = np.pad(An.p, (0, H - len(An.p)), mode='constant', constant_values=0)
		BNP = np.pad(Bn.p, (0, H - len(Bn.p)), mode='constant', constant_values=0)
		ANQ = np.pad(An.q, (0, H - len(An.q)), mode='constant', constant_values=0)
		BNQ = np.pad(Bn.q, (0, H - len(Bn.q)), mode='constant', constant_values=0)

		N_p += ANP * YC[n] + BNP * YS[n]
		N_q += ANQ * YC[n] + BNQ * YS[n]
	for n, (An, Bn) in enumerate(zip(A, B)):
		for m, (Am, Bm) in enumerate(zip(A, B)):
			AA = An * Am
			AB = An * Bm
			BB = Bn * Bm

			AAP = np.pad(AA.p, (0, 2 * H - len(AA.p)), 
				                   mode='constant', constant_values=0)
			ABP = np.pad(AB.p, (0, 2 * H - len(AB.p)), 
				                   mode='constant', constant_values=0)
			BBP = np.pad(BB.p, (0, 2 * H - len(BB.p)), 
				                   mode='constant', constant_values=0)

			AAQ = np.pad(AA.q, (0, 2 * H - len(AA.q)), 
				                   mode='constant', constant_values=0)
			ABQ = np.pad(AB.q, (0, 2 * H - len(AB.q)), 
				                   mode='constant', constant_values=0)
			BBQ = np.pad(BB.q, (0, 2 * H - len(BB.q)), 
				                   mode='constant', constant_values=0)

			D_p += AAP * CCh[n][m] + 2 * ABP * CSh[n][m] + BBP * SSh[n][m]
			D_q += AAQ * CCh[n][m] + 2 * ABQ * CSh[n][m] + BBQ * SSh[n][m]

	PPN = PseudoPolynomial(p=N_p, q=N_q, r=N_r)
	PPD = PseudoPolynomial(p=D_p, q=D_q, r=D_r)


	func = lambda b, PPN=PPN, PPD=PPD : PPN.eval(b) / PPD.eval(b)

	return func

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

	A = ABpoly(cn, sn, sgn, 0)
	B = ABpoly(cn, sn, sgn, 1)

	dA = [ a.deriv() for a in A ]
	dB = [ b.deriv() for b in B ]

	return A, B, dA, dB


def compute_zeros(ptensors, summations):
	C, S, YC, YS, CCh, CSh, SSh = summations
	AAdBp, AAdBq, ABdAp, ABdAq, ABdBp, ABdBq, BBdAp, BBdAq = ptensors

	#t0 = time()
	
	#"""
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

	#

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



	"""
	H = len(YC)
	L = 3*H - 1
	Pp = np.zeros(L)
	Pq = np.zeros(L)
	for i in range(H):
		for j in range(H):
			for k in range(H):
				Pp += AAdBp[i][j][k] * (YC[i] * CSh[j][k] - YS[k] * CCh[i][j])
				Pp += ABdAp[i][j][k] * (YC[i] * CSh[k][j] + YS[j] * CCh[i][k])
				Pp += ABdBp[i][j][k] * (YC[i] * SSh[j][k] + YS[j] * CSh[i][k])
				Pp += BBdAp[i][j][k] * (YS[i] * CSh[k][j] - YC[k] * SSh[i][j])

				Pq += AAdBq[i][j][k] * (YC[i] * CSh[j][k] - YS[k] * CCh[i][j])
				Pq += ABdAq[i][j][k] * (YC[i] * CSh[k][j] + YS[j] * CCh[i][k])
				Pq += ABdBq[i][j][k] * (YC[i] * SSh[j][k] + YS[j] * CSh[i][k])
				Pq += BBdAq[i][j][k] * (YS[i] * CSh[k][j] - YC[k] * SSh[i][j])
	"""
	#dt = time() - t0 
	#print dt, " seconds for computing coefficients"

	#t0 = time()
	P = pol.polysub(pol.polymul((1, 0, -1), pol.polypow(Pp, 2)), pol.polypow(Pq, 2)) 
	#dt = time() - t0
	#print dt, " seconds for computing polynomial"

	#t0 = time()
	#P = PseudoPolynomial(p = Pp, q= Pq, r=-1).root_finding_poly()
	#dt = time() - t0
	#print dt, " seconds for computing polynomial via root_finding_poly"
	
	#t0 = time()
	R = pol.polyroots(P)
	#dt = time() - t0
	#print dt, " seconds for finding roots of that polynomial"

	return R


def bfunc(b, cn, sn, YC, YS, CCh, CSh, SSh, positive=True, polys=None):
	""" this should be a polynomial in b. The maximum periodogram value
		(and thus the optimal solution for b) should be located at one 
		of the zeros of this polynomial.

		For now, I've just implemented this using linear algebra (not efficient)
	"""
	A = Avec(b, cn, sn, positive=positive)
	B = Bvec(b, cn, sn, positive=positive)
	dA = dAvec(b, cn, sn, positive=positive)
	dB = dBvec(b, cn, sn, positive=positive)
	
	X1  = np.outer(YC, np.dot(A, CCh) + np.dot(B, np.transpose(CSh)))
	X1 -= np.outer(    np.dot(np.transpose(CCh), A), YC)
	X2  = np.outer(YS, np.dot(A, CCh) + np.dot(B, np.transpose(CSh)))
	X2 -= np.outer(2 * np.dot(np.transpose(CSh), A) + np.dot(np.transpose(SSh), B), YC)
	X3  = np.outer(YC, np.dot(A, CSh) + np.dot(B, SSh))
	X3 -= np.outer(    np.dot(np.transpose(CCh), A), YS)
	X4  = np.outer(YS, np.dot(A, CSh) + np.dot(B, SSh))
	X4 -= np.outer(2 * np.dot(np.transpose(CSh), A) + np.dot(np.transpose(SSh), B), YS)

	return np.dot(A, np.dot(X1, dA)) + np.dot(B, np.dot(X2, dA)) \
	     + np.dot(A, np.dot(X3, dB)) + np.dot(B, np.dot(X4, dB))

def fasterTemplatePeriodogram(x, y, err, cn, sn, ofac=10, hfac=1, nprint=1, 
	                          polytens_p=None, polytens_n=None, 
	                          pvectors_p=None, pvectors_n=None):
	#t0 = time()
	if pvectors_p is None:
		pvectors_p = get_polynomial_vectors(cn, sn, sgn=  1)
	if pvectors_n is None:
		pvectors_n = get_polynomial_vectors(cn, sn, sgn= -1)
	
	if polytens_p is None:
		polytens_p = compute_polynomial_tensors(*pvectors_p)
	if polytens_n is None:
		polytens_n = compute_polynomial_tensors(*pvectors_n)

	t0 = time()
	# compute necessary arrays
	omegas, summations, YY, w, ybar = \
		compute_summations(x, y, err, len(cn), ofac=ofac, hfac=hfac)
	dt = time() - t0
	print dt, "seconds to compute summations", dt/len(omegas), " per freq"

	FTP = np.zeros(len(omegas))
	for i, (omega, sums) in enumerate(zip(omegas, summations)):

		afromb = lambda B, p : get_a_from_b(B, cn, sn, sums, positive=p)
		#t0 = time()
		#afromb_p = get_a_from_b_poly(pvectors_p, sums)
		#afromb_n = get_a_from_b_poly(pvectors_n, sums)
		#afromb = lambda B, p : afromb_p(B) if p else afromb_n(B)
		#dt = time() - t0
		#if i < nprint:
		#	print dt, " seconds for getting polynomial expression for a from b"

		t0 = time()
		Bpz = compute_zeros(polytens_p, sums)
		Bnz = compute_zeros(polytens_n, sums)
		dt = time() - t0
		if i < nprint:
			print dt, " seconds for root finding; ", dt/(len(Bpz) + len(Bnz)), " per root"
		
		t0 = time()
		# get a, c from b (solve rest of non-linear system of equations)
		Z = []
		for Bz, p in [ (Bpz, True), (Bnz, False) ]:
			for bz in Bz:
				if bz.imag != 0:
					continue
				Az = afromb(bz.real, p)
				if Az < 0:
					continue

				Z.append((Az, bz.real, p))
		
		dt = time() - t0
		if i < nprint:
			print dt, " seconds to get amplitudes for b values"
		t0 = time()
		# Compute periodogram values
		Pz = [ pdg_ftp( Av, Bv, cn, sn, YY, sums[2], sums[3], positive=p) \
		            for Av, Bv, p in Z ]
		dt = time() - t0
		if i < nprint:
			print dt, " seconds to evaluate periodogram at each zero"

		# Periodogram value is the global max of P_{-} and P_{+}.
		FTP[i] = 0 if len(Pz) == 0 else max(Pz)


	return omegas / (2 * np.pi), FTP


def fastTemplatePeriodogram(x, y, err, cn, sn, ofac=10, hfac=1, nb=50, stop=1E-5, max_count=10):
	
	# define array of b values to numerically find bracketed areas containing zeros
	b = np.linspace(-1 + stop, 1 - stop, nb)

	t0 = time()
	# compute necessary arrays
	omegas, summations, YY, w, ybar = \
		compute_summations(x, y, err, len(cn), ofac=ofac, hfac=hfac)
	dt = time() - t0
	print dt, "seconds to compute summations", dt/len(omegas), " per freq"

	FTP = np.zeros(len(omegas))
	for i, (omega, (C, S, YC, YS, CCh, CSh, SSh)) in enumerate(zip(omegas, summations)):

		# Shortcut functions
		bf      = lambda B, p :        bfunc(B, cn, sn, 
			                           YC, YS, CCh, CSh, SSh, positive=p)
		afromb  = lambda B, p : get_a_from_b(B, cn, sn, 
			                          (C, S, YC, YS, CCh, CSh, SSh), positive=p)

		t0 = time()
		# find zeros (very inefficiently)
		FBp = map(lambda B : bf(B, True ), b)
		FBn = map(lambda B : bf(B, False), b)
		dt = time() - t0
		if i == 0:
			print dt, " seconds to run %d * 2 bfunc"\
					  " evaluations for a rough grid search"%(nb), dt/(2*nb), \
					  " sec per func call"
		
		t0 = time()
		Bpz = find_zeros(b, FBp, func= lambda BV : bf(BV, True),  
			                                      stop=stop, max_count=max_count)
		Bnz = find_zeros(b, FBn, func= lambda BV : bf(BV, False), 
			                                      stop=stop, max_count=max_count)
		dt = time() - t0
		if i == 0:
			print dt, " seconds for root finding; ", \
			              dt/(len(Bpz) + len(Bnz)), " per root"
		
		t0 = time()
		# get a, c from b (solve rest of non-linear system of equations)
		Zp, Zn = [], []
		for bz in Bpz:
			Apz = afromb(bz, True)

			#ignore negative amplitude solutions
			if Apz < 0:
				continue

			Zp.append((Apz, bz))

		for bz in Bnz:
			Anz = afromb(bz, False)
			if Anz < 0:
				continue

			Zn.append((Anz, bz))
		dt = time() - t0
		if i == 0:
			print dt, " seconds to get amplitudes for b values"
		t0 = time()
		# Compute periodogram values
		Pzp = [ pdg_ftp( Av, Bv, cn, sn, YY, YC, YS, positive=True) for Av, Bv in Zp ]
		Pzn = [ pdg_ftp( Av, Bv, cn, sn, YY, YC, YS, positive=False) for Av, Bv in Zn ]
		dt = time() - t0
		if i == 0:
			print dt, " seconds to evaluate periodogram at each zero"

		# Periodogram value is the global max of P_{-} and P_{+}.
		FTP[i] = max([ max(Pzp) if len(Pzp) > 0 else 0, max(Pzn) if len(Pzn) > 0 else 0 ])


	return omegas / (2 * np.pi), FTP

def LMfit_ac(x, y, err, b, cn, sn, w, positive=True):
	""" fits a, b, c with Levenberg-Marquardt """

	ffunc = lambda X, *pars : fitfunc(X, positive, w, cn, sn, pars[0], b, pars[1])
	p0 = [ np.std(y), 0.0, np.mean(y) ]
	bounds = ([0, -np.inf], [ np.inf, np.inf])
	popt, pcov = curve_fit(ffunc, np.array(x, dtype=float), np.array(y, dtype=float), 
		                    sigma=np.array(err, dtype=float), p0=p0, 
                            absolute_sigma=True, bounds=bounds, 
                            method='trf')
	a, c = popt

	return a, c


def LMfit(x, y, err, cn, sn, w, positive=True):
	""" fits a, b, c with Levenberg-Marquardt """

	ffunc = lambda X, *pars : fitfunc(X, positive, w, cn, sn, *pars)
	p0 = [ np.std(y), 0.0, np.mean(y) ]
	bounds = ([0, -1, -np.inf], [ np.inf, 1, np.inf])
	popt, pcov = curve_fit(ffunc, np.array(x, dtype=float), np.array(y, dtype=float), 
		                    sigma=np.array(err, dtype=float), p0=p0, 
                            absolute_sigma=True, bounds=bounds, 
                            method='trf')
	a, b, c = popt

	return a, b, c

def rms_resid_over_rms(CN, SN, Tt, Yt):
	"""
	f, ax = plt.subplots()
	ax.plot(Tt, Yt, color='k')
	print len(CN)
	errs = []
	for H in range(len(CN)):
		cn, sn = CN[:(H+1)], SN[:(H+1)]
		a, b, c = LMfit(Tt, Yt, np.ones(len(Tt))*0.0001, cn, sn, 2 * np.pi)

		Ym = fitfunc(Tt, True, 2 * np.pi, cn, sn, a, b, c)

		er = sqrt(np.mean(np.power(Ym - Yt, 2)) / np.mean(np.power(Yt, 2)))
		errs.append(er)
		ax.plot(Tt, Ym, label='H=%d'%(H + 1))
		
	for i, er in enumerate(errs):
		print "H = %d"%(i+1), " err = ", er
	
	ax.legend(loc='best', fontsize=9)
	ax.set_xlabel('phase')
	ax.set_ylabel('scaled mag')
	plt.show()
	"""

	# This is fairly slow; is there a better way to get best fit pars?
	a, c = LMfit_ac(Tt, Yt, np.ones(len(Tt))*0.0001, 0.0, CN, SN, 2 * np.pi)
	Ym = fitfunc(Tt, True, 2 * np.pi, CN, SN, a, 0.0, c)

	"""
	f, ax = plt.subplots()
	ax.plot(Tt, Yt, color='k')
	ax.plot(Tt, Ym, label='H=%d'%(len(CN)))
	ax.legend(loc='best', fontsize=9)
	plt.show()
	"""
	return sqrt(np.mean(np.power(Ym - Yt, 2)) / np.mean(np.power(Yt, 2)))


def approximate_template(Tt, Yt, errfunc=rms_resid_over_rms, stop=1E-2):
	""" Fourier transforms template, returning the first H components """

	fft = np.fft.fft(Yt)
	cn, sn = zip(*[ (p.real/len(Tt), p.imag/len(Tt)) for i,p in enumerate(fft) \
		             if i > 0 ])

	h = 1
	while errfunc(cn[:h], sn[:h], Tt, Yt) > stop:
		h+=1

	return cn[:h], sn[:h]

def get_rrlyr_templates(template_fname=None, errfunc=rms_resid_over_rms, 
	                    stop=1E-2, filts='r'):

	# Obtain RR Lyrae templates
	templates = rrl.fetch_rrlyrae_templates()
	
	# Select the right ID's
	IDS = [ t for t in templates.ids if t[-1] in list(filts) ]

	# Get (phase, amplitude) data for each template
	Ts, Ys = zip(*[ templates.get_template(ID) for ID in IDS ])

	# Transform templates into harmonics and precompute
	# template-specific data
	all_polyvecs_polytens = []
	all_cn_sn = []
	if template_fname is None or not os.path.exists(template_fname):
		print "precomputing templates..."
		t0 = time()
		for i, (T, Y) in enumerate(zip(Ts, Ys)):
			print "  template ", i
			# The templates appear to be .... reversed?
			CN, SN = approximate_template(T, Y[::-1], stop=stop, errfunc=errfunc)
			print "  ", len(CN), " harmonics kept."

			pvectors_p = get_polynomial_vectors(CN, SN, sgn=  1)
			pvectors_n = get_polynomial_vectors(CN, SN, sgn= -1)
		
			polytens_p = compute_polynomial_tensors(*pvectors_p)
			polytens_n = compute_polynomial_tensors(*pvectors_n)

			all_polyvecs_polytens.append((pvectors_n, pvectors_p, 
				                          polytens_n, polytens_p))
			all_cn_sn.append((CN, SN))
		dt = time() - t0
		print dt, " seconds to precompute templates"

		if not template_fname is None:
			pickle.dump((all_cn_sn, all_polyvecs_polytens), 
			        open(template_fname, 'wb'))

	else:
		all_cn_sn, all_polyvecs_polytens \
		       = pickle.load(open(template_fname, 'rb'))

	return all_cn_sn, all_polyvecs_polytens


def RRLyrModeler(x, y, err, filts='r', loud=True, 
	              ofac=10, hfac=1, template_fname=None, 
	              errfunc=rms_resid_over_rms, stop=1E-2):

	
	all_cn_sn, all_polyvecs_polytens = \
		get_rrlyr_templates(template_fname=template_fname, 
			                 stop=stop, filts=filts, errfunc=errfunc)

	# Compute periodograms for each template
	all_ftps = []
	for i, ((CN, SN), (pvectors_n, pvectors_p, polytens_n, polytens_p)) \
	            in enumerate(zip(all_cn_sn, all_polyvecs_polytens)):

		if loud: print i + 1, "/", len(all_cn_sn)
		all_ftps.append(fasterTemplatePeriodogram(x, y, err, CN, SN, 
			         ofac=ofac, hfac=hfac, polytens_p=polytens_p,
			         polytens_n=polytens_n, pvectors_p=pvectors_p,
			         pvectors_n=pvectors_n))	

	freqs, ftps = zip(*all_ftps)
	FREQS = freqs[0]

	# RR lyr periodogram is the maximum periodogram value at each frequency
	FTP = [ max([ ftp[i] for ftp in ftps ]) for i in range(len(FREQS)) ]

	return FREQS, FTP

if __name__ == '__main__':
	import cPickle as pickle
	# Runs the RRLyrModeler against the gatspy RRLyraeTemplateModeler

	# Number of harmonics to keep
	#H = 8
	stop=2E-2
	"""
	# Get some data
	rrlyrae = datasets.fetch_rrlyrae()

	# Pick an arbitary lightcurve
	lens = { }
	for lcid in rrlyrae.ids:
		t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
		lens[lcid] = len(t[(filts=='r')])
	ML = max(lens.values())
	lcid = None
	for ID, l in lens.iteritems():
		if l == ML:
			lcid = ID

	#lcid = rrlyrae.ids[2]
	t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
	mask = (filts == 'r')
	x, y, err = t[mask], mag[mask], dmag[mask]

	"""

	#x,y,err = pickle.load(open('xyerr.pkl', 'rb'))
	#N = len(x)
	N=
	freq = 3.
	x = np.sort(np.random.rand(N))
	err = np.absolute(np.random.normal(loc=0.2, scale=0.1, size=N))
	y = np.cos(freq * x) + np.sin(3 * freq * x - 0.44) \
	            + np.array([ np.random.normal(loc=0, scale=s) for s in err ])

	#import cPickle as pickle
	#pickle.dump((x, y, err), open('xyerr.pkl', 'wb'))
	
	# Run our RRLyr modeler
	df = 0.005
	max_f = 1./(0.2)

	T = max(x) - min(x)
	ofac = 1./(T * df)
	hfac = max_f * T / float(len(x))
	print ofac, hfac

	t0 = time()
	freqs, ftp = RRLyrModeler(x, y, err, filts='r', loud=True, 
	                          ofac=ofac, hfac=hfac, stop=stop, errfunc=rms_resid_over_rms,
	                          template_fname='templates_RR_stop%.2e.pkl'%(stop))
	dt = time() - t0
	print "ftp: ", dt, "seconds"

	# Now run the gatspy modeler
	periods = np.power(freqs, -1)[::-1]

	t0 = time()
	model = RRLyraeTemplateModeler(filts='r').fit(x, y, err)
	FTP_GATSPY = model.periodogram(periods)
	dt = time() - t0
	print "gatspy: ", dt, " seconds"

	# and PLOT!
	FTP_GATSPY = FTP_GATSPY[::-1]
	f, ax = plt.subplots()
	ax.plot(freqs, ftp, label="our RRLTM")
	ax.plot(freqs, FTP_GATSPY, label="gatspy RRLTM")
	ax.set_xlabel('freq')
	ax.set_ylabel('fast template periodogram')
	ax.set_title('stop=%.3e, N=%d'%(stop, len(x)))
	ax.legend(loc='best')
	plt.show()
	