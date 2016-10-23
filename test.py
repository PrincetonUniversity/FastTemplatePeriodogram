import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt, eval_chebyu
from gatspy import periodic
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from pynfft.nfft import NFFT
from pynfft.solver import Solver
import sys
from time import time

Un = lambda n, x : eval_chebyu(n, np.float64(x)) if n >= 0 else 0
Tn = lambda n, x : eval_chebyt(n, np.float64(x)) if n >= 0 else 0

def weights(err, normed=True):
	w = np.power(err, -2)
	if normed: 
		w /= np.sum(w)
	return w

def XX_naive(x, y, w, omega, func1, func2, H=1):
	XX = np.zeros((H, H), dtype=np.float64)
	for n in range(1, H+1):
		for m in range(1, H+1):
			XX[n-1][m-1] = np.sum( np.multiply(w, np.multiply(func1(n * omega * x), func2(m * omega * x))) )
	return XX

def YX_naive(x, y, w, omega, func, H=1):
	YX = np.zeros(H, dtype=np.float64)
	for n in range(1, H+1):
		YX[n-1] = np.sum(np.multiply(w, np.multiply(y, func(n * omega * x))))
	return YX

def X_naive(x, y, w, omega, func, H=1):
	X = np.zeros(H, dtype=np.float64)
	for n in range(1, H+1):
		X[n-1] = np.sum(np.multiply(w, func(n * omega * x)))
	return X

def XXh_from_X(x, y, w, omega, kind='CC', H=1, make_hat_last=False, make_hat=True):
	C = X_naive(x, y, w, omega, np.cos, H=2*H)
	S = X_naive(x, y, w, omega, np.sin, H=2*H)
	XY = np.zeros((H,H))
	for i in range(H):
		for j in range(H):
			Cn, Sn = None, None
			if j == i:
				Cn = 1
				Sn = 0
			elif j > i:
				Cn =  C[j - i - 1] 
				Sn = -S[j - i - 1]
			else:
				Cn = C[i - j - 1]
				Sn = S[i - j - 1]
			Cp = C[i + j + 1]
			Sp = S[i + j + 1]
			print i, j, Cn, Cp, Sn, Sp

			if kind=='CC':
				XY[i][j] = 0.5 * (Cn + Cp) - (C[i]*C[j] if (make_hat and not make_hat_last) else 0)
			elif kind == 'SS':
				XY[i][j] = 0.5 * (Cn - Cp) - (S[i]*S[j] if (make_hat and not make_hat_last) else 0)
			else:
				XY[i][j] = 0.5 * (Sn + Sp) - (C[i]*S[j] if (make_hat and not make_hat_last) else 0)

	if make_hat_last:
		if kind == 'CC':
			XY -= np.outer(C[:H], C[:H])
		elif kind == 'SS':
			XY -= np.outer(S[:H], S[:H])
		else:
			XY -= np.outer(C[:H], S[:H])

	return XY

def XXhat(XX, X1, X2):
	XXh = np.zeros((len(XX), len(XX)), dtype=np.float64)
	for i in range(len(XX)):
		for j in range(len(XX)):
			XXh[i][j] = XX[i][j] - X1[i] * X2[j]
	return XXh

def YXhat(YX, X, ybar):
	YXh = np.zeros(len(YX), dtype=np.float64)
	for i in range(len(YX)):
		YXh[i] = YX[i] - ybar * X[i]
	return YXh

def tinterval(t, ofac):
	r = ofac * (max(t) - min(t))
	eps_border = 1E-5
	a = 0.5 - eps_border
	return 2 * a * (t - min(t)) / r - a


Afunc    = lambda n, x, p, q, plus=True :      p * Tn(n  , x) - (1 if plus else -1) * q * Un(n-1, x) * np.sqrt(1 - x*x)  
dAfunc   = lambda n, x, p, q, plus=True : n * (p * Un(n-1, x) + (1 if plus else -1) * q * Tn(n  , x) / np.sqrt(1 - x*x)) 

CC_naive = lambda x, y, w, omega, H=1 : XX_naive(x, y, w, omega, np.cos, np.cos, H=H)
CS_naive = lambda x, y, w, omega, H=1 : XX_naive(x, y, w, omega, np.cos, np.sin, H=H)
SS_naive = lambda x, y, w, omega, H=1 : XX_naive(x, y, w, omega, np.sin, np.sin, H=H)
YC_naive = lambda x, y, w, omega, H=1 : YX_naive(x, y, w, omega, np.cos, H=H)
YS_naive = lambda x, y, w, omega, H=1 : YX_naive(x, y, w, omega, np.sin, H=H)
S_naive  = lambda x, y, w, omega, H=1 :  X_naive(x, y, w, omega, np.sin, H=H)
C_naive  = lambda x, y, w, omega, H=1 :  X_naive(x, y, w, omega, np.cos, H=H)



Avec        = lambda x, c, s, plus=True :  np.array( [  Afunc(n, x, c[n-1],  s[n-1], plus=plus) for n in range(1, len(s)+1) ], dtype=np.float64)
Bvec        = lambda x, c, s, plus=True :  np.array( [  Afunc(n, x, s[n-1], -c[n-1], plus=plus) for n in range(1, len(s)+1) ], dtype=np.float64)
dAvec       = lambda x, c, s, plus=True :  np.array( [ dAfunc(n, x, c[n-1],  s[n-1], plus=plus) for n in range(1, len(s)+1) ], dtype=np.float64)
dBvec       = lambda x, c, s, plus=True :  np.array( [ dAfunc(n, x, s[n-1], -c[n-1], plus=plus) for n in range(1, len(s)+1) ], dtype=np.float64)

CChat = lambda CC, C : XXhat(CC, C, C)
CShat = lambda CS, C, S : XXhat(CS, C, S)
SShat = lambda SS, S : XXhat(SS, S, S)
YShat = lambda YS, S, ybar : YXhat(YS, S, ybar)
YChat = lambda YC, C, ybar : YXhat(YC, C, ybar)

def get_a_and_c_from_b(b, cn, sn, C, S, YC, YS, CCh, CSh, SSh, plus=True):

	A = Avec(b, cn, sn, plus=plus)
	B = Bvec(b, cn, sn, plus=plus)

	a = np.dot(A, YC) + np.dot(B, YS)
	a /= (np.dot(A, np.dot(CCh, A)) + 2 * np.dot(A, np.dot(CSh, B)) \
		  + np.dot(B, np.dot(SSh, B)))

	c = -a * (np.dot(A, C) + np.dot(B, S))

	return a, c

def M(t, b, w, cn, sn, plus=True):
	A = Avec(b, cn, sn, plus=plus)
	B = Bvec(b, cn, sn, plus=plus)
	Xc = np.array([ np.cos(n * w * t) for n in range(1, len(cn)+1) ], dtype=np.float64)
	Xs = np.array([ np.sin(n * w * t) for n in range(1, len(cn)+1) ], dtype=np.float64)

	return np.dot(A, Xc) + np.dot(B, Xs)

def fitfunc(x, plus, w, cn, sn, a, b, c):
	return a * np.array(map(lambda X : M(X, b, w, cn, sn, plus=plus), x)) + c

def chi2(y, ymodel, err):
	return np.sum(np.power(np.divide(y - ymodel, err), 2))

def LMfit(x, y, err, cn, sn, w, plus=True):
	ffunc = lambda X, *pars : fitfunc(X, plus, w, cn, sn, *pars)
	p0 = [ np.std(y), 0.0, np.mean(y) ]
	bounds = ([0, -1, -np.inf], [ np.inf, 1, np.inf])
	popt, pcov = curve_fit(ffunc, np.array(x, dtype=float), np.array(y, dtype=float), 
		                    sigma=np.array(err, dtype=float), p0=p0, 
                            absolute_sigma=True, bounds=bounds, 
                            method='trf')
	a, b, c = popt
	return a, b, c

def LMfit_from_b(x, y, err, cn, sn, w, b, plus=True):
	ffunc = lambda X, *pars : fitfunc(X, plus, w, cn, sn, pars[0], b, pars[1])
	p0 = [ np.std(y), np.mean(y) ]
	bounds = ([0, -np.inf], [ np.inf, np.inf])
	popt, pcov = curve_fit(ffunc, np.array(x, dtype=float), np.array(y, dtype=float), 
		                    sigma=np.array(err, dtype=float), p0=p0, 
                            absolute_sigma=True, bounds=bounds, 
                            method='trf')
	a, c = popt
	return a, c


def pdg(x, y, err, cn, sn, a, b, c, omega, plus=True):
	popt, pcov = curve_fit(lambda X, C : C,  x.astype(np.float64),
						y.astype(np.float64), sigma=err.astype(np.float64), 
						p0=[ np.mean(y.astype(np.float64)) ], absolute_sigma=True)

	ycons = np.ones(len(y)) * popt[0]
	c2c = chi2(y, ycons, err)

	ymod = fitfunc(x, plus, omega, cn, sn, a, b, c)
	c2m = chi2(y, ymod , err)
	return (c2c - c2m) / c2c


def pdg_full_nonlin(x, y, err, cn, sn, omega, plus=True):
	a, b, c = LMfit(x, y, err, cn, sn, omega, plus=plus)
	return pdg(x, y, err, cn, sn, a, b, c, omega, plus=plus)



def pdg_nonlin_from_b(x, y, err, cn, sn, b, omega, C, S, YC, YS, CCh, CSh, SSh, plus=True):
	a, c = LMfit_from_b(x, y, err, cn, sn, omega, b, plus=plus)
	return pdg(x, y, err, cn, sn, a, b, c, omega, plus=plus)

def pdg_ftp_full(b, cn, sn, YY, YC, YS, CCh, CSh, SSh, plus=True, worry=False):
	A = Avec(b, cn, sn, plus=plus)
	B = Bvec(b, cn, sn, plus=plus)

	num =  (np.dot(np.dot(A, np.outer(YC, YC)), A) + 2 * np.dot(np.dot(A, np.outer(YC, YS)), B) + np.dot(np.dot(B, np.outer(YS, YS)), B) )
	denom = (np.dot(A, np.dot(CCh, A)) + 2 * np.dot(A, np.dot(CSh, B)) + np.dot(B, np.dot(SSh, B)))
	return num / (denom * YY)

def pdg_ftp(a, b, cn, sn, YY, YC, YS, plus=True, worry=False):
	A = Avec(b, cn, sn, plus=plus)
	B = Bvec(b, cn, sn, plus=plus)

	return (a / YY) * (np.dot(A, YC) + np.dot(B, YS))

def linzero(xlo, xhi, ylo, yhi):
	m = (yhi - ylo) / (xhi - xlo)
	b = ylo - m * xlo
	return -b/m

def hone_in(lo, hi, stop, func, count, max_count):
	y_hi = func(hi)
	y_lo = func(lo)

	if y_hi * y_lo >= 0:
		raise Exception("y_lo and y_hi need different signs.")

	zero = linzero(lo, hi, y_lo, y_hi)

	#print lo, zero, hi
	fz = func(zero)
	if zero - lo < stop or hi - zero < stop or fz == 0 or count == max_count:
		return zero

	if fz * y_lo < 0:
		return hone_in(lo, zero, stop, func, count+1, max_count)

	else:
		return hone_in(zero, hi, stop, func, count+1, max_count)


def find_zeros(x, y, func=None, stop=1E-6, max_count=100):
	inds = [ i for i in range(len(y) - 1) if y[i] * y[i+1] < 0 ]
	zeros = []
	for i in inds:
		
		if not func is None:
			try:
				z = hone_in(x[i], x[i+1], stop, func, 0, max_count)
				#print "good zero"
			except RuntimeError, e:
				print "bad zero finder", "(",x[i],",",y[i],")", "(",x[i+1],",",y[i+1],")"
				z = linzero(x[i], x[i+1], y[i], y[i+1])
			#print "ZERO IS ", z
			#sys.exit()

		else:
			z = linzero(x[i], x[i+1], y[i], y[i+1])
			

		zeros.append(z)
	return zeros

def get_N_df_omegas(x, ofac, hfac):

	N = int(floor(0.5 * len(x) * ofac * hfac))
	t = tinterval(x, ofac)
	T = max(x) - min(x)
	df = 1. / (ofac * T)
	omegas = np.array([ i * 2 * np.pi * df for i in range(1, N) ])
	return N, df, omegas


def get_arrs_fast(x, y, err, H, ofac=5, hfac=1):

	N, df, omegas = get_N_df_omegas(x, ofac, hfac)
	t = tinterval(x, ofac)
	
	w = weights(err, normed=True)
	W = 1./sum(w)

	ybar = np.dot(w, y) * W
	u = np.multiply(w, y - ybar)

	#print "med(w), med(u) = ", np.median(w), np.median(u)
	YY = np.dot(w, np.power(y - ybar, 2)) * W
	#print w, u

	plan = NFFT(4 * H * N, len(x))
	plan.x = t
	plan.precompute()

	plan2 = NFFT(2 * H * N, len(x))
	plan2.x = t
	plan2.precompute()

	plan.f = w
	f_hat_w = plan.adjoint()[2 * H * N + 1:]

	plan2.f = u
	f_hat_u = plan2.adjoint()[H * N + 1:]


	C, S, YC, YS, CCh, CSh, SSh = [], [], [], [], [], [], []

	for i in range(N-1):
		C_ = np.zeros(2 * H)
		S_ = np.zeros(2 * H)
		YC_ = np.zeros(H)
		YS_ = np.zeros(H)
		CC_ = np.zeros((H, H))
		CS_ = np.zeros((H, H))
		SS_ = np.zeros((H, H))


		for j in range(2 * H):
			##s = (1 if j % 2 == 0 else -1)
			s = (-1)**(i+1) if (j % 2 == 0) else 1.0
			C_[j] = f_hat_w[(j+1)*(i+1)-1].real * W * s
			S_[j] = f_hat_w[(j+1)*(i+1)-1].imag * W * s

		for j in range(H):
			s = (-1)**(i+1) if (j % 2 == 0) else 1.0
			YC_[j] =  f_hat_u[(j+1)*(i+1)-1].real * W * s
			YS_[j] =  f_hat_u[(j+1)*(i+1)-1].imag * W * s
			for k in range(H):
				
				Sn, Cn = None, None

				if j == k:
					Sn = 0
					Cn = 1
				elif k > j:
					Sn = S_[k - j - 1] 
					Cn = C_[k - j - 1]
				else:
					Sn = -S_[j - k - 1]
					Cn =  C_[j - k - 1]

				Sp = S_[j + k + 1]
				Cp = C_[j + k + 1]
				
				CC_[j][k] = 0.5 * ( Cn + Cp ) - C_[j]*C_[k]
				CS_[j][k] = 0.5 * ( Sn + Sp ) - C_[j]*S_[k]
				SS_[j][k] = 0.5 * ( Cn - Cp ) - S_[j]*S_[k]


		C_ = C_[:H]
		S_ = S_[:H]
		C.append(C_)		
		S.append(S_)
		CCh.append(CC_ )#- np.outer(C_, C_))
		CSh.append(CS_ )#- np.outer(C_, S_))
		SSh.append(SS_ )#- np.outer(S_, S_))
		YC.append(YC_)
		YS.append(YS_)

		
	return omegas, C, S, YC, YS, CCh, CSh, SSh, YY, W, ybar

def bfunc(b, cn, sn, YC, YS, CCh, CSh, SSh, plus=True):
	""" this should be a polynomial in b. The maximum periodogram value
		(and thus the optimal solution for b) should be located at one 
		of the zeros of this polynomial.

		For now, I've just implemented this using linear algebra (not efficient)
	"""
	A = Avec(b, cn, sn, plus=plus)
	B = Bvec(b, cn, sn, plus=plus)
	dA = dAvec(b, cn, sn, plus=plus)
	dB = dBvec(b, cn, sn, plus=plus)
	
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


def fast_bfunc(b, coeffs):
	val = 0
	B = 1
	for c in coeffs:
		val += c * B
		B *= b
	return val

#def get_coeffs(cn, sn, YC, YS, CCh, CSh, SSh):


def test_bfunc_speed(b, cn, sn, YC, YS, CCh, CSh, SSh):
	nb = len(b)
	
	H = len(cn)
	
	t0 = time()
	Bfunc_slow   = lambda B, p :         bfunc(B, CN, SN, YC, YS, CCh, CSh, SSh, plus=p)

	FBp = map(lambda B : Bfunc_slow(B, True), b)
	FBn = map(lambda B : Bfunc_slow(B, False), b)

	tz0 = time()
	Bpz = find_zeros(b, FBp, func= lambda BV : Bfunc_slow(BV, True), stop=1E-3, max_count=10)
	Bnz = find_zeros(b, FBn, func= lambda BV : Bfunc_slow(BV, False), stop=1E-3, max_count=10)
	dtz_s = time() - tz0

	dt = (time() - t0) / nb

	coeffs = np.random.randn(6*H - 2)

	t0 = time()

	Bfunc_fast   = lambda B : fast_bfunc(B, coeffs)

	# for grid search
	t0gs = time()
	#FBp = map(Bfunc_fast, b)
	#FBn = map(Bfunc_fast, b)
	dtgs = time() - t0gs

	# calculating coeffs
	t0cc = time()
	Bfunc_slow(b[4], True)
	Bfunc_slow(b[4], False)
	dtcc = time() - t0cc

	# find zeros
	tz0 = time()
	#Bpz_f = find_zeros(b, FBp, func=Bfunc_fast, stop=1E-3, max_count=10)
	#Bnz_f = find_zeros(b, FBn, func=Bfunc_fast, stop=1E-3, max_count=10)
	Bpz_f = np.roots(coeffs)
	Bnz_f = np.roots(coeffs)
	dtz_f = time() - tz0

	dt_f = (time() - t0) / nb

	print nb * dt, " slow ", nb * dt_f + dt, " fast "
	Nzs = (len(Bpz) + len(Bnz))
	Nzf = (len(Bpz_f) + len(Bnz_f))
	print dtz_f / Nzf , " s per zero (x", Nzf, ") (fast), ", dtz_s / Nzs, " s per zero (x", Nzs, ") (slow)"
	print dtcc, " s calculating coefficients"
	print dtgs, " s getting grid search"

def get_arrs(x, y, err, H, omega):
	w = weights(err, normed=True)
	W = 1.0 / sum(w)
	ybar = sum(np.multiply(w, y)) * W

	Y = y - ybar
	YY = np.dot(w, np.power(Y, 2)) * W

	S  =  S_naive(x, Y, w * W, omega, H=H)
	C  =  C_naive(x, Y, w * W, omega, H=H)
	YC = YC_naive(x, Y, w * W, omega, H=H)
	YS = YS_naive(x, Y, w * W, omega, H=H)
	CC = CC_naive(x, Y, w * W, omega, H=H)
	SS = SS_naive(x, Y, w * W, omega, H=H)
	CS = CS_naive(x, Y, w * W, omega, H=H)
	
	SSh = SS - np.outer(S, S)
	CCh = CC - np.outer(C, C)
	CSh = CS - np.outer(C, S)

	#CC_ = XXh_from_X(x, y, w * W, omega, kind='CC', H=H, make_hat_last=True, make_hat=True)
	#print CCh, CC_
	#sys.exit()

	return C, S, YC, YS, CCh, CSh, SSh, YY, W, ybar

		
def fastTemplatePeriodogram(x, y, err, cn, sn, ofac=10, hfac=1, nb=50, use_slow=True, enforce_positive_amplitude=True, pdgmode='full', worry=False, plot_bad=False):
	DT = np.float64

	# convert to high precision
	CN = np.array(cn, dtype=DT)
	SN = np.array(sn, dtype=DT)
	X, Y, ERR = np.array(x, dtype=DT), \
	            np.array(y, dtype=DT), \
	            np.array(err, dtype=DT)

	# Translate uncertainties into weights
	w  = weights(err)

	eps = 1E-7
	small = 1E-4
	b = np.linspace(-1 + eps, 1 - eps, nb)


	# compute necessary arrays
	omegas, Cx, Sx  = None, None, None
	YCx, YSx, CChx = None, None, None
	CShx, SShx, YY, W, ybar = None, None, None, None, None
	t0 = time()
	if not use_slow:
		omegas, Cx, Sx, YCx, YSx, CChx, CShx, SShx, YY, W, ybar = \
			get_arrs_fast(X, Y, ERR, len(CN), ofac=ofac, hfac=hfac)
	else:
		N, df, omegas = get_N_df_omegas(x, ofac, hfac)
		Cx, Sx  = [], []
		YCx, YSx = [], []
		CShx, SShx, CChx  = [], [], []

		for omega in omegas:
			CS, SS_, YCS, YSS, CChS, CShS, SShS, YY, W, ybar = get_arrs(X, Y, ERR, len(CN), omega)
			Cx.append(CS)
			Sx.append(SS_)
			YCx.append(YCS)
			YSx.append(YSS)
			CChx.append(CChS)
			CShx.append(CShS)
			SShx.append(SShS)
	dt = time() - t0
	print " getting (C, S, ...) arrays took ", dt, " seconds"
	
	#print YY
	#sys.exit()
	"""

	omegas, Cf, Sf, YCf, YSf, CChf, CShf, SShf, YY, W, ybar = \
		get_arrs_fast(X, Y, ERR, len(CN), ofac=ofac, hfac=hfac)
	
	N, df, omegas = get_N_df_omegas(x, ofac, hfac)
	Cs, Ss  = [], []
	YCs, YSs = [], []
	CShs, SShs, CChs  = [], [], []

	for omega in omegas:
		CS, SS_, YCS, YSS, CChS, CShS, SShS, YY, W, ybar = get_arrs(X, Y, ERR, len(CN), omega)
		Cs.append(CS)
		Ss.append(SS_)
		YCs.append(YCS)
		YSs.append(YSS)
		CChs.append(CChS)
		CShs.append(CShS)
		SShs.append(SShS)

	f, ax = plt.subplots()
	XX_slow, XX_fast, name = SShs, SShf, 'SShat'
	lss = [ '-', '--', ':', '-.'  ]
	colors = [ 'b', 'r', 'c', 'm', 'g' ]
	for i in range(H):
		ax.plot(omegas,  [ CF[i] for CF, CS in zip(Cf, Cs) ], color=colors[i], ls=lss[(2*i)%(len(lss))], label='C  %d (fast)'%(i))
		ax.plot(omegas,  [ CS[i] for CF, CS in zip(Cf, Cs) ], color=colors[i], ls=lss[(2 * i + 1)%(len(lss))], label='C %d (slow)'%(i))

		#for j in range(i, H):
	
			#ax.plot(omegas, [ (XXf[i][j] - XXs[i][j]) for XXs, XXf in zip(XX_slow, XX_fast) ], ls=lss[i%len(lss)], color=colors[j%len(lss)], label='(%s) slow, %d,%d'%(name, i, j))
			#ax.plot(omegas, [ XXs[i][j] for XXs, XXf in zip(XX_slow, XX_fast) ], ls=lss[(2*i)%len(lss)], color=colors[j%len(lss)], label='(%s) slow, %d,%d'%(name, i, j), alpha=0.5)
			#ax.plot(omegas, [ XXf[i][j] for XXs, XXf in zip(XX_slow, XX_fast) ], ls=lss[(2*i+1)%len(lss)], color=colors[j%len(lss)], label='(%s) fast, %d,%d'%(name, i, j), alpha=0.5)

			#ax.plot(omegas, [ XXs[j] for XXs, XXf in zip(YSs, YSf) ], ls=lss[(2*i)%len(lss)], color=colors[j%len(lss)], label='(%s) slow, %d'%('YS', i), alpha=0.5)
			#ax.plot(omegas, [ XXf[j] for XXs, XXf in zip(YSs, YSf) ], ls=lss[(2*i+1)%len(lss)], color=colors[j%len(lss)], label='(%s) fast, %d'%('YS', i), alpha=0.5)

	ax.legend(loc='best')
	plt.show()
	sys.exit()
	"""
	


	FTP = np.zeros(len(omegas), dtype=DT)
	for i, (omega, C, S, YC, YS, CCh, CSh, SSh) \
	         in enumerate(zip(omegas, Cx, Sx, YCx, YSx, CChx, CShx, SShx)):

		
		#test_bfunc_speed(b, CN, SN, YC, YS, CCh, CSh, SSh)
		#sys.exit()

		# Shortcut functions
		Bfunc   = lambda B, p :         bfunc(B, CN, SN, YC, YS, CCh, CSh, SSh, plus=p)
		acfromb = lambda B, p : get_a_and_c_from_b(B, CN, SN, C, S, YC, YS, CCh, CSh, SSh, plus=p)

		t0 = time()
		# find zeros (very inefficiently)
		FBp = map(lambda B : Bfunc(B, True), b)
		FBn = map(lambda B : Bfunc(B, False), b)
		if i == 0:
			dt = time() - t0
			print " evaluating b func took ", dt, " seconds"

		t0 = time()
		Bpz = find_zeros(b, FBp, func= lambda BV : Bfunc(BV, True), stop=1E-5, max_count=10)
		Bnz = find_zeros(b, FBn, func= lambda BV : Bfunc(BV, False), stop=1E-5, max_count=10)
		if i == 0:
			dt = time() - t0
			print " finding zeros took ", dt, " seconds"

		t0 = time()
		# get a, c from b (solve rest of non-linear system of equations)
		Zp, Zn = [], []
		for bz in Bpz:
			Apz, Cpz = acfromb(bz, True)
			if Apz < -small and enforce_positive_amplitude:
				continue
			Zp.append((Apz, bz, Cpz))

		for bz in Bnz:
			Anz, Cnz = acfromb(bz, False)
			if Anz < -small and enforce_positive_amplitude:
				continue
			Zn.append((Anz, bz, Cnz))
		if i == 0:
			dt = time() - t0
			print " selecting zeros took ", dt, " seconds"

		t0 = time()
		# Compute periodogram values
		Pzp, Pzn = None, None
		if pdgmode == 'full':
			Pzp = [ pdg_ftp_full( Bv, CN, SN, W * YY, YC, YS,CCh, CSh, SSh, plus=True, worry=worry) for Av, Bv, Cv in Zp ]
			Pzn = [ pdg_ftp_full( Bv, CN, SN, W * YY, YC, YS,CCh, CSh, SSh, plus=False, worry=worry) for Av, Bv, Cv in Zn ]
		elif pdgmode == 'from_a':
			Pzp = [ pdg_ftp( acfromb(Bv, True)[0], Bv, CN, SN, W * YY, YC, YS, plus=True, worry=worry) for Av, Bv, Cv in Zp ]
			Pzn = [ pdg_ftp( acfromb(Bv, False)[0], Bv, CN, SN, W * YY, YC, YS, plus=False, worry=worry) for Av, Bv, Cv in Zn ]
		elif pdgmode == 'nonlin_from_b':
			Pzp = [ pdg_nonlin_from_b( X, Y, ERR, CN, SN, Bv, omega, C, S, YC, YS, CCh, CSh, SSh, plus=True) for Av, Bv, Cv in Zp ]
			Pzn = [ pdg_nonlin_from_b( X, Y, ERR, CN, SN, Bv, omega, C, S, YC, YS, CCh, CSh, SSh, plus=False) for Av, Bv, Cv in Zn ]
		elif pdgmode == 'full_nonlin':
			Pzp = [ pdg_full_nonlin( X, Y, ERR, CN, SN, omega, plus=True)  ]
			Pzn = [ pdg_full_nonlin( X, Y, ERR, CN, SN, omega, plus=False) ]
		else:
			raise Exception(" dont know what pdgmode='%s' is"%(pdgmode) )
		if i == 0:
			dt = time() - t0
			print " evaluating periodogram at each zero took ", dt, " seconds"

		# Periodogram value is the global max of this.
		FTP[i] = max([ max(Pzp) if len(Pzp) > 0 else 0, max(Pzn) if len(Pzn) > 0 else 0 ])

		if (FTP[i] < -1E-3 or FTP[i] > 1 + 1E-3) and plot_bad:
			print "BAD: omega = ", omega, " i = ", i, " P = ", FTP[i]
			Ppf = map(lambda bval : pdg_ftp_full(bval, CN, SN, YY * W, YC, YS, CCh, CSh, SSh, plus=True, worry=False), b)
			Pnf = map(lambda bval : pdg_ftp_full(bval, CN, SN, YY * W, YC, YS, CCh, CSh, SSh, plus=False, worry=False), b)
			#Pp = map(lambda bval : pdg_ftp(acfromb(bval, True)[0], bval, CN, SN, YY * W, YC, YS, plus=True, worry=False), b)
			#Pn = map(lambda bval : pdg_ftp(acfromb(bval, False)[0], bval, CN, SN, YY * W, YC, YS, plus=False, worry=False), b)
			Ppnl = map(lambda bval : pdg_nonlin_from_b( X, Y, ERR, CN, SN, bval, omega, C, S, YC, YS, CCh, CSh, SSh, plus=True), b)
			Pnnl = map(lambda bval : pdg_nonlin_from_b( X, Y, ERR, CN, SN, bval, omega, C, S, YC, YS, CCh, CSh, SSh, plus=False), b)

			#print Ppnl
			f, ax = plt.subplots()
			ax.scatter([ z[1] for z in Zp ], np.zeros(len(Zp)), color='b')
			ax.scatter([ z[1] for z in Zn ], np.zeros(len(Zn)), color='m')

			#FB3 = map(func3, b)
			#ax.axvline(b1n, color='m', lw=2, ls='--', alpha=0.5, label='fit b (p)')
			#ax.axvline(b1p, color='b', lw=2, ls='--', alpha=0.5, label='fit b (n)')
			ax.plot(b, FBn, color='m', label='POLYNOMIAL (n)')
			ax.plot(b, FBp, color='b', label='POLYNOMIAL (p)')
			ax.axhline(0, color='k')
			#ax.axvline(B_0, color='k', label="ACTUAL", alpha=0.5, lw=2, ls='-')
			B_ = None
			maxp = None
			for P, (Av, Bv, Cv) in zip(Pzp, Zp):
				if maxp is None or P > maxp:
					B_ = Bv
					maxp = P
			for P, (Av, Bv, Cv) in zip(Pzn, Zn):
				if maxp is None or P > maxp:
					B_ = Bv
					maxp = P

			ax.axvline(B_, color='g', label="FTP", alpha=0.5, lw=2, ls='--')

			ax.axhline(1, ls='--', color='k')

			#ax.plot(b, Pn, color='m', lw=2, ls=':', label='pdg (n)')
			ax.plot(b, Pnf, color='m', lw=2, ls='--', label='pdg full (n)')
			ax.plot(b, Pnnl, color='m', lw=2, ls='-.', label="pdg nonlin (n)")

			#ax.plot(b, Pp, color='b', lw=2, ls=':', label='pdg (p)')
			ax.plot(b, Ppf, color='b', lw=2, ls='--', label='pdg full (p)')
			ax.plot(b, Ppnl, color='b', lw=2, ls='-.', label="pdg nonlin (p)")
			#L = 1.2 * max([ np.std(FBn), np.std(FBp) ])
			ax.set_ylim(-5, 5)

			ax.legend(loc='best', fontsize=10)

			plt.show()


	return omegas / (2 * np.pi), FTP

if __name__ == '__main__':
	N = 25
	H = 1
	freq = 3.0
	#freq = 3.56

	#phi = 0.5 * (0.8 * np.random.rand() + 0.1) / freq
	phi = 0.1
	#CN = np.array([np.random.rand() for i in range(H) ], dtype=np.float64)#, 0.34, 0.67, 0.44, 0.32]
	#SN = np.array([np.random.rand() for i in range(H) ], dtype=np.float64)#, 0.21, 0.75, 0.99, 0.11]
	CN = [ 0.8, 0.6, 0.3, 0.4 , 0.6]
	SN = [ 0.4, 0.7, 0.1 , 0.2, 0.8]
	CN = [ cn for i, cn in enumerate(CN) if i < H ]
	SN = [ sn for i, sn in enumerate(SN) if i < H ]


	#CN = [ 0.5 ]
	#SN = [ 0.5 ]



	#CN = np.array([ 0.73, 0.45 ], dtype=np.float64)
	#SN = np.array([ 0.20, 0.67 ], dtype=np.float64)
	#CN = np.array([ 1.0 ])
	#SN = np.array([ 0.0 ])

	A_0 = 3.0
	W_0 = 2 * np.pi * freq
	B_0 = cos(W_0 * phi)
	#C_0 = 5.0 * np.random.rand()
	C_0 = 5.0
	dW  = 0
	
	tvals = np.linspace(0, 5.0, 200)
	#tvals = np.array(sorted(5 * np.random.rand(N)))


	"""
	yfit1 = fitfunc(tvals, True, W_0, CN, SN, A_0, 1.0, C_0)
	yfit2 = fitfunc(tvals, True, W_0, CN, SN, A_0, sqrt(2) / 2., C_0)
	yfit3 = fitfunc(tvals, True, W_0, CN, SN, A_0, 0.0, C_0)
	yfit4 = fitfunc(tvals, True, W_0, CN, SN, A_0, -sqrt(2)/2.0, C_0)
	yfit5 = fitfunc(tvals, True, W_0, CN, SN, A_0, -1.0, C_0)
	yfit1n = fitfunc(tvals, False, W_0, CN, SN, A_0, -1.0, C_0)
	yfit2n = fitfunc(tvals, False, W_0, CN, SN, A_0, -sqrt(2.0)/2.0, C_0)
	yfit3n = fitfunc(tvals, False, W_0, CN, SN, A_0, 0.0, C_0)
	yfit4n = fitfunc(tvals, False, W_0, CN, SN, A_0, sqrt(2.0)/2.0, C_0)
	yfit5n = fitfunc(tvals, False, W_0, CN, SN, A_0, 1.0, C_0)

	F, AX = plt.subplots()
	AX.plot(tvals * freq, yfit1, lw=0.5, color='k')
	AX.plot(tvals * freq, yfit2, lw=0.75, color='k')
	AX.plot(tvals * freq, yfit3, lw=1.0, color='k')
	AX.plot(tvals * freq, yfit4, lw=1.25, color='k')
	AX.plot(tvals * freq, yfit5, lw=1.5, color='k')
	AX.plot(tvals * freq, yfit1n, lw=1.75, color='k')
	AX.plot(tvals * freq, yfit2n, lw=2.0, color='k')
	AX.plot(tvals * freq, yfit3n, lw=2.25, color='k')
	AX.plot(tvals * freq, yfit4n, lw=2.5, color='k')
	AX.plot(tvals * freq, yfit5n, lw=2.75, color='k')
	plt.show()
	"""
	#sys.exit()
	
	signal = lambda X : np.array(map(lambda T : A_0 * M(T, B_0, W_0, CN, SN, plus=(W_0 * phi < np.pi)) + C_0, X), dtype=np.float64)
	signal_error = lambda E : np.array([ np.random.normal(scale=Eval) for Eval in E ], dtype=np.float64)

	#x = np.array(sorted(np.random.random(N)))
	x = np.linspace(0, 1, N, dtype=np.float64)
	errs = np.array(np.absolute(np.random.normal(loc=0.2, scale=0.2, size=len(x))), dtype=np.float64)
	
	y = signal(x) + signal_error(errs)



	print "computing periodogram"
	
	use_slow=False
	t0 = time()
	cases = { #'full, slow' : dict(pdgmode='full', use_slow=True) ,
				#'full_nonlin, slow' : dict(pdgmode='full_nonlin', use_slow=True) ,
				'ftp: direct summation' : dict(pdgmode='from_a', use_slow=True, plot_bad=False) ,
				'ftp: using nfft' : dict(pdgmode='from_a', use_slow=False, plot_bad=False) ,
			  #'full, fast' : dict(pdgmode='full', use_slow=False)
	}



	ftps = { case : None for case in cases }
	freqs = None
	for case, kwargs in cases.iteritems():
		print case
		t0 = time()
		freqs, ftps[case] = fastTemplatePeriodogram(x, y, errs, CN, SN, **kwargs)
		dt = time() - t0
		print dt, " seconds, ", dt/len(freqs), " per freq"

	LSP = periodic.LombScargleFast()
	LSP.fit(x, y, errs)
	ftps_ls = LSP.periodogram(np.power(freqs, -1)[::-1])[::-1]

	fftp, axftp = plt.subplots()
	for case in cases:
		axftp.plot(freqs, ftps[case], label=case)
	axftp.plot(freqs, ftps_ls, label="gatspy LombScargleFast")
	axftp.legend(loc='best')
	axftp.axvline(freq, color='k', ls='--')
	axftp.set_xlabel('freq')
	axftp.set_title('H=%d, N=%d'%(H, N))
	axftp.set_ylabel('periodogram')
	fftp.savefig("ftp_test_H%d_N%d.png"%(H,N))
	plt.show()
	sys.exit()


	eps = 1E-6
	b = np.linspace(-1 + eps, 1 - eps, 200)


	print "getting arrs (fast)"
	t0 = time()
	omegas, Cf, Sf, YCf, YSf, CChf, CShf, SShf, YYf, Wf, ybarf= get_arrs_fast(x, y, errs, len(CN), ofac=100, hfac=1)
	print "DONE: ",(time() - t0) / len(omegas), " seconds per freq."
	fno = np.argmin(np.absolute(omegas - (W_0 + dW)))
	om = omegas[fno]

	print om, W_0

	print "getting arrs (slow)"
	t0 = time()
	C, S, YC, YS, CCh, CSh, SSh, YY, W, ybar = get_arrs(x, y, errs, len(CN), om )
	print "DONE: ", time() - t0, " seconds per freq."

	print C, Cf[fno]
	print S, Sf[fno]
	print YC, YCf[fno]
	print YS, YSf[fno]


	#C, S, YC, YS, CCh, CSh, SSh, YY, W, ybar = \
	#	Cf[fno], Sf[fno], YCf[fno], YSf[fno], CChf[fno], CShf[fno], SShf[fno], YYf, Wf, ybarf
	#print C, Cf[fno]
	#print S, Sf[fno]


	#print "func2"
	#func2 = Func(x, y, errs, CN, SN, omega2, plus=True)
	print "lmfit"
	t0 = time()
	a1p, b1p, c1p = LMfit(x, y, errs, CN, SN, W_0 + dW, plus=True)
	a1n, b1n, c1n = LMfit(x, y, errs, CN, SN, W_0 + dW, plus=False)
	print  "LMFIT : ", time() - t0, " seconds"

	
	print "getting zeros"
	t0 = time()

	func1p = lambda B : bfunc(B, CN, SN, YC, YS, CCh, CSh, SSh, plus=True)
	func1n = lambda B : bfunc(B, CN, SN, YC, YS, CCh, CSh, SSh, plus=False)

	FBp = map(func1p, b)
	FBn = map(func1n, b)

	Bpz = find_zeros(b, FBp)
	Bnz = find_zeros(b, FBn)

	Zp, Zn = [], []
	for bz in Bpz:
		Apz, Cpz = get_a_and_c_from_b(bz, CN, SN, C, S, YC, YS, CCh, CSh, SSh, plus=True)
		Zp.append((Apz, bz, Cpz))

	for bz in Bnz:
		Anz, Cnz = get_a_and_c_from_b(bz, CN, SN, C, S, YC, YS, CCh, CSh, SSh, plus=False)
		Zn.append((Anz, bz, Cnz))
	
	#Pzp = [ pdg(x, y, errs, CN, SN, Av, Bv, Cv + ybar, W_0, plus=True) for Av, Bv, Cv in Zp ]
	#Pzn = [ pdg(x, y, errs, CN, SN, Av, Bv, Cv + ybar, W_0, plus=False) for Av, Bv, Cv in Zn ]

	Pzp = [ pdg_ftp_full(Bv, CN, SN, YY * W, YC, YS, CCh, CSh, SSh, plus=True ) for Av, Bv, Cv in Zp ]
	Pzn = [ pdg_ftp_full(Bv, CN, SN, YY * W, YC, YS, CCh, CSh, SSh, plus=False) for Av, Bv, Cv in Zn ]

	A_p, B_p, C_p = Zp[np.argmax(Pzp)]
	A_n, B_n, C_n = Zn[np.argmax(Pzn)]
	
	tf_ftp = (max(Pzp) > max(Pzn))
 
	A_, B_, C_ = (A_p, B_p, C_p + ybar) if tf_ftp else (A_n, B_n, C_n + ybar) 
	print "ZEROS : ", time() - t0, " seconds"
	

	
	print "calc periodogram (pos)"
	#P1p = map(lambda bval : pdg(x, y, errs, CN, SN, a1p, bval, c1p, W_0, plus=True), b)
	P1p = map(lambda bval : pdg_ftp_full(bval, CN, SN, YY * W, YC, YS, CCh, SSh, CSh, plus=True), b)

	print "calc periodogram (neg)"
	#P1n = map(lambda bval : pdg(x, y, errs, CN, SN, a1n, bval, c1n, W_0, plus=False), b)
	P1n = map(lambda bval : pdg_ftp_full(bval, CN, SN, YY * W, YC, YS, CCh, SSh, CSh, plus=False), b)

	"""
	P1n_func = interp1d(b, P1n)
	P1p_func = interp1d(b, P1p)
	"""
	i1 = np.argmax(P1n)
	i2 = np.argmax(P1p)
	
	#pdgp = pdg(x, y, errs, CN, SN, a1p, b1p, c1p, W_0, plus=True)
	#pdgn = pdg(x, y, errs, CN, SN, a1n, b1n, c1n, W_0, plus=False)

	pdgp = pdg_ftp_full(b1p, CN, SN, YY * W, YC, YS, CCh, SSh, CSh, plus=True)
	pdgn = pdg_ftp_full(b1n, CN, SN, YY * W, YC, YS, CCh, SSh, CSh, plus=False)


	tf_fit = (pdgp > pdgn)

	AF, BF, CF = (a1p, b1p, c1p) if tf_fit else (a1n, b1n, c1n)

	print "A (0, fit, ftp): %15.5e %15.5e %15.5e"%(A_0, AF, A_)
	print "B (0, fit, ftp): %15.5e %15.5e %15.5e"%(B_0, BF, B_)
	print "C (0, fit, ftp): %15.5e %15.5e %15.5e"%(C_0, CF, C_)

	

	frq = (W_0 + dW) / (2 * np.pi)
	y0 = signal(tvals/frq)
	ymod  = fitfunc(tvals / frq, tf_fit, W_0 + dW, CN, SN, AF, BF, CF)
	ymod_ = fitfunc(tvals / frq, tf_ftp, W_0 + dW, CN, SN, A_, B_, C_)

	pvals, y0pf = zip(*sorted([(T%1.0, Y) for T, Y in zip( tvals, y0 ) ], key=lambda ent : ent[0]))
	pvals, ymodpf = zip(*sorted([(T%1.0, Y) for T, Y in zip( tvals, ymod ) ], key=lambda ent : ent[0]))
	pvals, ymodpf_ = zip(*sorted([(T%1.0, Y) for T, Y in zip( tvals, ymod_ ) ], key=lambda ent : ent[0]))

	f3, ax3 = plt.subplots()
	PHI = (frq * x) % 1.0

	ax3.errorbar(PHI, y, yerr=errs, fmt='o', color='b', alpha=0.5, markersize=10, label="data")
	ax3.plot(pvals, ymodpf, color='r', label="fit")
	ax3.plot(pvals, y0pf, color='k', label="actual")
	ax3.plot(pvals, ymodpf_, color='g', label="ftp")
	ax3.legend(loc='best')

	ax3.set_xlabel('phase')
	ax3.set_ylabel('Y or Y_model')
	
	f, ax = plt.subplots()
	ax.scatter([ z[1] for z in Zp ], np.zeros(len(Zp)), color='b')
	ax.scatter([ z[1] for z in Zn ], np.zeros(len(Zn)), color='m')

	#FB3 = map(func3, b)
	ax.axvline(b1n, color='m', lw=2, ls='--', alpha=0.5, label='fit b (p)')
	ax.axvline(b1p, color='b', lw=2, ls='--', alpha=0.5, label='fit b (n)')
	ax.plot(b, FBn, color='m', label='POLYNOMIAL (n)')
	ax.plot(b, FBp, color='b', label='POLYNOMIAL (p)')
	ax.axhline(0, color='k')
	#ax.axvline(B_0, color='k', label="ACTUAL", alpha=0.5, lw=2, ls='-')
	ax.axvline(B_, color='g', label="FTP", alpha=0.5, lw=2, ls='--')

	ax.axhline(1, ls='--', color='k')

	ax.axvline(b[i1], color='m', lw=3, ls='-.', label='max pdg (n)')
	ax.axvline(b[i2], color='b', lw=3, ls='-.', label='max pdg (p)')

	ax.plot(b, P1n, color='m', lw=2, ls=':', label='pdg (n)')
	ax.plot(b, P1p, color='b', lw=2, ls=':', label='pdg (p)')
	L = 1.2 * max([ np.std(FBn), np.std(FBp) ])
	ax.set_ylim(-L, L)

	ax.legend(loc='best', fontsize=10)

	plt.show()
	#LSP = periodic.LombScargleFast()
	#LSP.fit(x, y, errs)
	#freqs = np.linspace(0.5, 1000, 10000)
	#periods = np.power(freqs, -1)[::-1]
	#P = LSP.periodogram(periods)
