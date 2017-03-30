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

from numpy.testing import assert_allclose


def get_a_from_b(b, cn, sn, sums, A=None, B=None,
                 AYCBYS=None, sgn=1):
    """ return the optimal amplitude & offset for a given value of b """

    if A is None:
        A = Avec(b, cn, sn, sgn=sgn)
    if B is None:
        B = Bvec(b, cn, sn, sgn=sgn)
    if AYCBYS is None:
        AYCBYS = np.dot(A, sums.YC) + np.dot(B, sums.YS)

    D = (    np.einsum('i,j,ij', A, A, sums.CC) \
       + 2 * np.einsum('i,j,ij', A, B, sums.CS) \
       +     np.einsum('i,j,ij', B, B, sums.SS))

    return AYCBYS / D


def fit_template(t, y, dy, cn, sn, ptensors, freq, sums=None, 
                       allow_negative_amplitudes=True):
    nh   = len(cn)
    w    = weights(dy)
    ybar = np.dot(w, y)
    yy   = np.dot(w, (y - ybar)**2)

    if sums is None:
        sums   = direct_summations(t, y, w, freq, nh) 

    zeros = compute_zeros(ptensors, sums)

    # Check boundaries, too
    for edge in [1, -1]:
        if not edge in zeros:
            zeros.append(edge)

    max_pz, bfpars = None, None
    
    for bz in zeros:
        for sgn in [ -1, 1 ]:
            A = Avec(bz, cn, sn, sgn=sgn)
            B = Bvec(bz, cn, sn, sgn=sgn)

            AYCBYS = np.dot(A, sums.YC[:nh]) + np.dot(B, sums.YS[:nh])
            ACBS   = np.dot(A, sums.C[:nh])  + np.dot(B, sums.S[:nh])

            # Obtain amplitude for a given b=cos(wtau) and sign(sin(wtau))
            a = get_a_from_b(bz, cn, sn, sums, A=A, B=B, AYCBYS=AYCBYS)

            # Skip negative amplitude solutions
            if a < 0 and not allow_negative_amplitudes:
                continue

            # Compute periodogram
            pz = a * AYCBYS / yy
                    
            # Record the best-fit parameters for this template
            if max_pz is None or pz > max_pz:
                # Get offset
                c = ybar - a * ACBS

                # Store best-fit parameters
                bfpars = ModelFitParams(a=a, b=bz, c=c, sgn=sgn)

                max_pz = pz
    
    if bfpars is None:
        return 0, ModelFitParams(a=0, b=1, c=ybar, sgn=1)

    return max_pz, bfpars


def template_periodogram(t, y, dy, cn, sn, freqs, ptensors=None,
                              summations=None, loud=False,
                              allow_negative_amplitudes=True, fast=True):
    nh = len(cn)
    w = weights(dy)

    if ptensors is None:
        pvectors = get_polynomial_vectors(cn, sn, sgn=1)
        ptensors = compute_polynomial_tensors(*pvectors)

    if summations is None:
        # compute sums using NFFT
        if fast:
            summations = fast_summations(t, y, w, freqs, nh)
        else:
            summations = direct_summations(t, y, w, freqs, nh)

    power, best_fit_pars = [], []

    # Iterate through frequency values (sums contains C, S, YC, ...)
    for frq, sums in zip(freqs, summations):

        p_max, bfpars = fit_template(t, y, dy, cn, sn, ptensors, frq, sums=sums,
                          allow_negative_amplitudes=allow_negative_amplitudes)

        best_fit_pars.append(bfpars)
        power.append(p_max)


    return np.array(power), best_fit_pars
