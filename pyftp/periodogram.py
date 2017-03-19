"""
FAST TEMPLATE PERIODOGRAM (prototype)

Uses NFFT to make the template periodogram scale as H*N log(H*N)
where H is the number of harmonics in which to expand the template and
N is the number of observations.

Previous routines scaled as N^2 and used non-linear least-squares
minimization (e.g. Levenberg-Marquardt) at each frequency.

(c) John Hoffman 2016

"""
from __future__ import print_function

import sys
import os
from math import *

from time import time
import numpy as np
from scipy.special import eval_chebyt,\
                          eval_chebyu

from .summations import fast_summations, direct_summations
from .pseudo_poly import compute_polynomial_tensors,\
                         get_polynomial_vectors,\
                         compute_zeros
from .utils import Un, Tn, Avec, Bvec, dAvec, dBvec,\
                    Summations, ModelFitParams, weights



def get_a_from_b(b, cn, sn, sums, A=None, B=None,
                 AYCBYS=None, sgn=1):
    """ return the optimal amplitude & offset for a given value of b """

    if A is None:
        A = Avec(b, cn, sn, sgn=sgn)
    if B is None:
        B = Bvec(b, cn, sn, sgn=sgn)
    if AYCBYS is None:
        AYCBYS = np.dot(A, sums.YC) + np.dot(B, sums.YS)

    a = AYCBYS / (       np.einsum('i,j,ij', A, A, sums.CC) \
                   + 2 * np.einsum('i,j,ij', A, B, sums.CS) \
                   +     np.einsum('i,j,ij', B, B, sums.SS))
    return a


def fit_template(t, y, dy, template, freq, allow_negative_amplitudes=True):
    H = len(template.cn)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar)**2)
    cn, sn = template.cn, template.sn

    sums = direct_summations(t, y, weights(dy), freq, H)

    zeros = compute_zeros(template.ptensors, sums)

    # Check boundaries, too
    if not  1 in zeros: zeros.append(1)
    if not -1 in zeros: zeros.append(-1)

    # compute phase shift due to non-zero x[0]
    tshift = (freq * t[0]) % 1.0


    print(zeros)

    max_pz, bfpars = None, None
    for bz in zeros:
        for sgn_ in [ -1, 1 ]:
            A = Avec(bz, cn, sn, sgn=sgn_)
            B = Bvec(bz, cn, sn, sgn=sgn_)

            AYCBYS = np.dot(A, sums.YC[:H]) + np.dot(B, sums.YS[:H])
            ACBS   = np.dot(A, sums.C[:H])  + np.dot(B, sums.S[:H])

            # Obtain amplitude for a given b=cos(wtau) and sign(sin(wtau))
            amplitude = get_a_from_b(bz, cn, sn, sums, A=A, B=B, AYCBYS=AYCBYS)

            # Skip negative amplitude solutions
            if amplitude < 0 and not allow_negative_amplitudes:
                continue

            # Compute periodogram
            pz = amplitude * AYCBYS / YY

            # Record the best-fit parameters for this template
            if max_pz is None or pz > max_pz:
                # Get offset
                c = ybar - amplitude * ACBS

                # Correct for the fact that we've shifted t -> t - t0
                # during the NFFT
                wtauz = np.arccos(bz)
                if sgn_ < 0:
                    wtauz = 2 * np.pi - wtauz
                wtauz += tshift * 2 * np.pi

                # Store best-fit parameters
                bfpars = ModelFitParams(a=amplitude, b=np.cos(wtauz),
                                        c=c, sgn=int(np.sign(np.sin(wtauz))))


                max_pz = pz

    if bfpars is None:
        raise Warning("could not find positive amplitude solution for freq = %.3e"%(freq))
        return ModelFitParams(a=0, b=1, c=ybar, sgn=1), 0.

    return max_pz, bfpars



def fast_template_periodogram(t, y, dy, cn, sn, freqs, pvectors=None, ptensors=None,
                              summations=None, loud=False, return_best_fit_pars=False,
                              allow_negative_amplitudes=True):
    H = len(cn)
    w = weights(dy)
    ybar = np.dot(w, y)
    YY = np.dot(w, (y - ybar)**2)

    if pvectors is None:
        pvectors = get_polynomial_vectors(cn, sn, sgn=  1)

    if ptensors is None:
        ptensors = compute_polynomial_tensors(*pvectors)

    t0 = None
    if loud:
        t0 = time()

    if summations is None:
        # compute sums using NFFT
        summations = fast_summations(t, y, w, freqs, H)

    if loud:
        dt = time() - t0
        print("*", dt / len(freqs), " s / freqs to get summations")

    FTP = np.zeros(len(freqs))
    best_fit_pars = []

    # Iterate through frequency values (sums contains C, S, YC, ...)
    for i, (freq, sums) in enumerate(zip(freqs, summations)):

        if loud and i == 0:
            t0 = time()

        # Get zeros of polynomial (zeros are same for both +/- sinwtau)
        zeros = compute_zeros(ptensors, sums, loud=(i==0 and loud))

        # Check boundaries, too
        if not  1 in zeros: zeros.append(1)
        if not -1 in zeros: zeros.append(-1)

        if loud and i == 0:
            dt = time() - t0
            print("*", dt, " s / freqs to get zeros")
            t0 = time()

        bfpars = None
        max_pz = None

        # compute phase shift due to non-zero x[0]
        tshift = (freq * t[0]) % 1.0

        for bz in zeros:
            for sgn_ in [ -1, 1 ]:
                A = Avec(bz, cn, sn, sgn=sgn_)
                B = Bvec(bz, cn, sn, sgn=sgn_)

                AYCBYS = np.dot(A, sums.YC[:H]) + np.dot(B, sums.YS[:H])
                ACBS   = np.dot(A, sums.C[:H])  + np.dot(B, sums.S[:H])

                # Obtain amplitude for a given b=cos(wtau) and sign(sin(wtau))
                amplitude = get_a_from_b(bz, cn, sn, sums, A=A, B=B, AYCBYS=AYCBYS)

                # Skip negative amplitude solutions

                if amplitude < 0 and not allow_negative_amplitudes: 
                    continue

                # Compute periodogram
                pz = amplitude * AYCBYS / YY

                # Record the best-fit parameters for this template
                if max_pz is None or pz > max_pz:
                    if return_best_fit_pars:

                        # Get offset
                        c = ybar - amplitude * ACBS

                        # Correct for the fact that we've shifted t -> t - t0
                        # during the NFFT
                        wtauz = np.arccos(bz)
                        if sgn_ < 0:
                            wtauz = 2 * np.pi - wtauz
                        wtauz += tshift * 2 * np.pi

                        # Store best-fit parameters
                        sgn = int(np.sign(np.sin(wtauz)))
                        b   = np.cos(wtauz)

                        bfpars = ModelFitParams(a=amplitude, b=b, c=c, sgn=sgn)

                    max_pz = pz

        if return_best_fit_pars and bfpars is None:
            # Could not find any positive amplitude solutions ... usually means this is a poor fit,
            # so instead of a more exhaustive search for the best positive amplitude solution,
            # we simply set parameters to 0 and periodogram to 0
            FTP[i] = 0
            bfpars = ModelFitParams(a=0, b=1, c=ybar, sgn=1)
            raise Warning("could not find positive amplitude solution for freq = %.3e (ftp[%d])"%(freq, i))


        if loud and i == 0:
            dt = time() - t0
            print("*", dt, " s / freq to investigate each zero")

        # Periodogram value is the global max of P_{-} and P_{+}.
        FTP[i] = max_pz
        if return_best_fit_pars:
            best_fit_pars.append(bfpars)

    if not return_best_fit_pars:
        return FTP
    else:
        return FTP, best_fit_pars
