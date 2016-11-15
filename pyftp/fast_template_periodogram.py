""" 
FAST TEMPLATE PERIODOGRAM (prototype)

Uses NFFT to make the template periodogram scale as H*N log(H*N) 
where H is the number of harmonics in which to expand the template and
N is the number of observations.

Previous routines scaled as N^2 and used non-linear least-squares
minimization (e.g. Levenberg-Marquardt) at each frequency.

(c) John Hoffman 2016

"""
import sys
import os
import cmath
from math import *
from time import time
import numpy as np
from scipy.special import eval_chebyt,\
                          eval_chebyu

from pynfft.nfft import NFFT
from pseudo_poly import compute_polynomial_tensors,\
                        get_polynomial_vectors,\
                        compute_zeros


# shortcuts for the Chebyshev polynomials
Un = lambda n, x : eval_chebyu(n, x) if n >= 0 else 0
Tn = lambda n, x : eval_chebyt(n, x) if n >= 0 else 0

# A and dA expressions
Afunc    = lambda n, x, p, q, positive=True :      \
                       p * Tn(n  , x) - (1 if positive else -1) \
                                         * q * Un(n-1, x) * np.sqrt(1 -  x*x) 

dAfunc   = lambda n, x, p, q, positive=True : \
                       n * (p * Un(n-1, x) + (1 if positive else -1) \
                       	                 * q * Tn(n  , x) / np.sqrt(1 -  x*x))



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

def getAB(b, cn, sn):
	SQ =  sqrt(1 - min([ 1-1E-8, b*b ]))
	H = len(cn)
	TN = np.array([ Tn(n+1, b) for n in range(H) ])
	UN = np.array([ Un(n, b) * SQ for n in range(H) ])
	snUN, cnUN = sn * UN, cn * UN
	Ap = cn * TN - snUN
	An = Ap + 2 * snUN

	Bp = sn * TN + cnUN
	Bn = Bp - 2 * cnUN
	
	return Ap, An, Bp, Bn

def M(t, b, w, cn, sn, positive=True):
	""" evaluate the shifted template at a given time """

	A = Avec(b, cn, sn, positive=positive)
	B = Bvec(b, cn, sn, positive=positive)
	Xc = np.array([ np.cos(n * w * t) for n in range(1, len(cn)+1) ])
	Xs = np.array([ np.sin(n * w * t) for n in range(1, len(cn)+1) ])

	return np.dot(A, Xc) + np.dot(B, Xs)

def pdg_ftp(a, b, cn, sn, YY, YC, YS, A=None, B=None, 
	        AYCBYS=None, positive=True):
	""" evaluate the periodogram for a given a, b """

	if A is None:
		A = Avec(b, cn, sn, positive=positive)
	if B is None:
		B = Bvec(b, cn, sn, positive=positive)
	if AYCBYS is None:
		AYCBYS = (np.dot(A, YC) + np.dot(B, YS))

	return (a / YY) * AYCBYS

def fitfunc(x, positive, w, cn, sn, a, b, c):
	""" aM(t - tau) + c """
	m = lambda B : lambda X : M(X, B, w, cn, sn, positive=positive)
	return a * np.array(map(m(b), x)) + c

def weights(err):
	""" converts sigma_i -> w_i \equiv (1/W) * (1/sigma_i^2) """
	w = np.power(err, -2)
	w/= np.sum(w)
	return w

def get_a_from_b(b, cn, sn, sums, A=None, B=None, 
	             AYCBYS=None, positive=True):
	""" return the optimal amplitude & offset for a given value of b """

	C, S, YC, YS, CCh, CSh, SSh = sums

	H = len(cn)
	if A is None:
		A = Avec(b, cn, sn, positive=positive)
	if B is None:
		B = Bvec(b, cn, sn, positive=positive)
	if AYCBYS is None:
		AYCBYS = np.dot(A, YC[:H]) + np.dot(B, YS[:H])

	a = AYCBYS / (       np.einsum('i,j,ij', A, A, CCh[:H][:H]) \
		           + 2 * np.einsum('i,j,ij', A, B, CSh[:H][:H]) \
		           +     np.einsum('i,j,ij', B, B, SSh[:H][:H]))

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

	omegas = np.array([ 2 * np.pi * i * df for i in range(1, N) ])
	
	# compute weighted mean
	ybar = np.dot(w, y)

	# subtract off weighted mean
	u = np.multiply(w, y - ybar)
	
	# weighted variance
	YY = np.dot(w, np.power(y - ybar, 2))

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
			s = (-1 if ((i % 2)==0) and ((j % 2) == 0) else 1)
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


def fastTemplatePeriodogram(x, y, err, cn, sn, ofac=10, hfac=1, 
	                          pvectors=None, polytens=None,
	                          omegas=None, summations=None, YY=None, w=None, 
	                          ybar=None, loud=False):

	if pvectors is None:
		pvectors = get_polynomial_vectors(cn, sn, sgn=  1)
	
	if polytens is None:
		polytens = compute_polynomial_tensors(*pvectors)

	t0 = None
	if loud: 
		t0 = time()
	if summations is None:
		# compute sums using NFFT
		omegas, summations, YY, w, ybar = \
			compute_summations(x, y, err, len(cn), ofac=ofac, hfac=hfac)

	if loud: 
		dt = time() - t0
		print "*", dt / len(omegas), " s / freqs to get summations"

	FTP = np.zeros(len(omegas))
	for i, (omega, sums) in enumerate(zip(omegas, summations)):
		
		afromb = lambda B, p : get_a_from_b(B, cn, sn, sums, positive=p)
		
		
		if loud: 
			t0 = time()
		# Get zeros (zeros are same for both +/- sinwtau)
		Bz = compute_zeros(polytens, sums, loud=(i==0 and loud))

		if len(Bz) == 0:
			FTP[i] = 0
			continue
		
		if loud and i == 0:
			dt = time() - t0
			print "*", dt, " s / freqs to get zeros"

		if loud: t0 = time()
		# get a from b (solve rest of non-linear system of equations)
		Pz = []
		Ap, An, Bp, Bn = zip(*[ getAB(bz, cn, sn) for bz in Bz ]) 
		Aycbysp, Aycbysn = zip(*[ (np.dot(ap, sums[2]) + np.dot(bp, sums[3]), \
			       np.dot(an, sums[2]) + np.dot(bn, sums[3])) \
		              for ap, an, bp, bn in zip(Ap, An, Bp, Bn) ])

		for bz, ap, an, bp, bn, aycbysp, aycbysn in \
					zip(Bz, Ap, An, Bp, Bn, Aycbysp, Aycbysn):

			amp_p = get_a_from_b(bz, cn, sn, sums, A=ap, B=bp, AYCBYS=aycbysp)
			amp_n = get_a_from_b(bz, cn, sn, sums, A=an, B=bn, AYCBYS=aycbysn)

			pdg_p = amp_p * aycbysp / YY
			pdg_n = amp_n * aycbysn / YY
			
			if amp_p > 0:
				Pz.append(pdg_p)
			if amp_n > 0:
				Pz.append(pdg_n)

		if loud and i == 0:
			dt = time() - t0
			print "*", dt, " s / freq to investigate each zero"

		# Periodogram value is the global max of P_{-} and P_{+}.
		FTP[i] = 0 if len(Pz) == 0 else max(Pz)
		
	return omegas / (2 * np.pi), FTP
	
