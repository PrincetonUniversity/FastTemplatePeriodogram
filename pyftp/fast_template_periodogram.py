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
from collections import namedtuple
from scipy.special import eval_chebyt,\
                          eval_chebyu

from pynfft.nfft import NFFT
from pseudo_poly import compute_polynomial_tensors,\
                        get_polynomial_vectors,\
                        compute_zeros


Summations = namedtuple('Summations', [ 'C', 'S', 'YC', 'YS', 
                                        'CCh', 'CSh', 'SSh'])

ModelFitParams = namedtuple('ModelFitParams', [ 'a', 'b', 'c', 'sgn' ])

# shortcuts for the Chebyshev polynomials
Un = lambda n, x : eval_chebyu(n, x) if n >= 0 else 0
Tn = lambda n, x : eval_chebyt(n, x) if n >= 0 else 0

# A and dA expressions
Afunc    = lambda n, x, p, q, sgn=1 :      \
            p * Tn(n  , x) - sgn * q * Un(n-1, x) * np.sqrt(1 -  x*x) 

dAfunc   = lambda n, x, p, q, sgn=1 : \
            n * (p * Un(n-1, x) + sgn * q * Tn(n  , x) / np.sqrt(1 -  x*x))



# returns vector expressions of A, B and their derivatives
Avec        = lambda x, c, s, sgn=1 :  np.array([  \
                          Afunc(n, x, c[n-1],  s[n-1], sgn=sgn) \
                                 for n in range(1, len(s)+1) ])
Bvec        = lambda x, c, s, sgn=1 :  np.array([ \
                          Afunc(n, x, s[n-1], -c[n-1], sgn=sgn) \
                                 for n in range(1, len(s)+1) ])
dAvec       = lambda x, c, s, sgn=1 :  np.array([ \
                         dAfunc(n, x, c[n-1],  s[n-1], sgn=sgn) \
                                 for n in range(1, len(s)+1) ])
dBvec       = lambda x, c, s, sgn=1 :  np.array([ \
                         dAfunc(n, x, s[n-1], -c[n-1], sgn=sgn) \
                                 for n in range(1, len(s)+1) ])

def getAB(b, cn, sn):
    """ efficient computation of A, B vectors for both +/- sin(wtau) """
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

def M(t, b, omega, cn, sn, sgn=1):
    """ evaluate the shifted template at a given time """

    A = Avec(b, cn, sn, sgn=sgn)
    B = Bvec(b, cn, sn, sgn=sgn)
    Xc = np.array([ np.cos(n * omega * t) for n in range(1, len(cn)+1) ])
    Xs = np.array([ np.sin(n * omega * t) for n in range(1, len(cn)+1) ])

    return np.dot(A, Xc) + np.dot(B, Xs)

def fitfunc(x, sgn, omega, cn, sn, a, b, c):
    """ aM(t - tau) + c """
    m = lambda b_ : lambda x_ : M(x_, b_, omega, cn, sn, sgn=sgn)
    return a * np.array(map(m(b), x)) + c

def weights(err):
    """ converts sigma_i -> w_i \equiv (1/W) * (1/sigma_i^2) """
    w = np.power(err, -2)
    w/= np.sum(w)
    return w

def get_a_from_b(b, cn, sn, sums, A=None, B=None, 
                 AYCBYS=None, sgn=1):
    """ return the optimal amplitude & offset for a given value of b """

    if A is None:
        A = Avec(b, cn, sn, sgn=sgn)
    if B is None:
        B = Bvec(b, cn, sn, sgn=sgn)
    if AYCBYS is None:
        AYCBYS = np.dot(A, sums.YC) + np.dot(B, sums.YS)

    a = AYCBYS / (       np.einsum('i,j,ij', A, A, sums.CCh) \
                   + 2 * np.einsum('i,j,ij', A, B, sums.CSh) \
                   +     np.einsum('i,j,ij', B, B, sums.SSh))
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

    all_computed_sums = []
    # Now compute the summation values at each frequency
    for i in range(N-1):

        computed_sums = Summations(C=np.zeros(H),
                                   S=np.zeros(H),
                                   YC=np.zeros(H),
                                   YS=np.zeros(H),
                                   CCh=np.zeros((H,H)),
                                   CSh=np.zeros((H,H)),
                                   SSh=np.zeros((H,H)))

        C_, S_ = np.zeros(2 * H), np.zeros(2 * H)
        for j in range(2 * H):
            # This sign factor is necessary 
            # but I don't know why.
            s = (-1 if ((i % 2)==0) and ((j % 2) == 0) else 1)
            C_[j] =  f_hat_w[(j+1)*(i+1)-1].real * s
            S_[j] =  f_hat_w[(j+1)*(i+1)-1].imag * s
            if j < H:
                computed_sums.YC[j] =  f_hat_u[(j+1)*(i+1)-1].real * s
                computed_sums.YS[j] =  f_hat_u[(j+1)*(i+1)-1].imag * s

        for j in range(H):
            for k in range(H):
                
                Sn, Cn = None, None

                if j == k:
                    Sn = 0
                    Cn = 1
                else:
                    Sn =  np.sign(k - j) * S_[int(abs(k - j)) - 1] 
                    Cn =  C_[int(abs(k - j)) - 1]

                Sp = S_[j + k + 1]
                Cp = C_[j + k + 1]

                computed_sums.CCh[j][k] = 0.5 * ( Cn + Cp ) - C_[j] * C_[k]
                computed_sums.CSh[j][k] = 0.5 * ( Sn + Sp ) - C_[j] * S_[k]
                computed_sums.SSh[j][k] = 0.5 * ( Cn - Cp ) - S_[j] * S_[k]

        computed_sums.C[:] = C_[:H]
        computed_sums.S[:] = S_[:H]

        all_computed_sums.append(computed_sums)
        
    return omegas, all_computed_sums, YY, w, ybar


def fastTemplatePeriodogram(x, y, err, cn, sn, ofac=10, hfac=1, 
                              pvectors=None, ptensors=None,
                              omegas=None, summations=None, YY=None, w=None, 
                              ybar=None, loud=False, return_best_fit_pars=False):

    H = len(cn)
    
    if pvectors is None:
        pvectors = get_polynomial_vectors(cn, sn, sgn=  1)
    
    if ptensors is None:
        ptensors = compute_polynomial_tensors(*pvectors)

    t0 = None
    if loud: 
        t0 = time()

    if summations is None:
        # compute sums using NFFT
        omegas, summations, YY, w, ybar = \
            compute_summations(x, y, err, H, ofac=ofac, hfac=hfac)

    if loud: 
        dt = time() - t0
        print "*", dt / len(omegas), " s / freqs to get summations"

    FTP = np.zeros(len(omegas))
    best_fit_pars = []

    # Iterate through frequency values (sums contains C, S, YC, ...)
    for i, (omega, sums) in enumerate(zip(omegas, summations)):
               
        if loud and i == 0: 
            t0 = time()

        # Get zeros of polynomial (zeros are same for both +/- sinwtau)
        zeros = compute_zeros(ptensors, sums, loud=(i==0 and loud))

        # Maybe we should throw an exception/warning here...
        if len(zeros) == 0:
            FTP[i] = 0
            continue
        
        if loud and i == 0:
            dt = time() - t0
            print "*", dt, " s / freqs to get zeros"
            t0 = time()

        # Pz is the list of p_+(w), p_-(w) values at each zero
        Pz = []
        fit_pars = []
        for z in zeros:
            for sgn in [ -1, 1 ]:
                A = Avec(z, cn, sn, sgn=sgn)
                B = Bvec(z, cn, sn, sgn=sgn)

                AYCBYS = np.dot(A, sums.YC[:H]) + np.dot(B, sums.YS[:H])

                # Obtain amplitude for a given b=cos(wtau) and sign(sin(wtau))
                amplitude = get_a_from_b(z, cn, sn, sums, A=A, B=B, AYCBYS=AYCBYS)

                # skip negative amplitude solutions
                if amplitude < 0: continue
                
		# record the best-fit parameters for this template 
                if return_best_fit_pars:
                    c = -amplitude * (np.dot(A, sums.C[:H]) + np.dot(B, sums.S[:H]))
                    fit_pars.append(ModelFitParams(a=amplitude, b=z, c=c, sgn=sgn))
               
                Pz.append(amplitude * AYCBYS / YY)

        if loud and i == 0:
            dt = time() - t0
            print "*", dt, " s / freq to investigate each zero"

        # Periodogram value is the global max of P_{-} and P_{+}.
        j = None if len(Pz) == 0 else np.argmax(Pz)
        FTP[i] = 0 if j is None else Pz[j]
        if return_best_fit_pars:
            best_fit_pars.append((0,0,0) if j is None else fit_pars[j])
    if not return_best_fit_pars:
        return omegas / (2 * np.pi), FTP
    else:
        return omegas / (2 * np.pi), FTP, best_fit_pars
    
