from pynfft.nfft import NFFT
from .utils import Summations
import numpy as np
from math import floor


def inspect_freqs(freqs):
    """Validate that frequencies lie on a regular grid at multiples of df"""
    nf = len(freqs)
    df = freqs[1] - freqs[0]
    dnf = int(round(freqs[0] / df))

    if not np.allclose(freqs[0], dnf * df):
        raise ValueError("Minimum frequency must be an integer multiple of df")

    if not np.allclose(np.diff(freqs), df):
        raise ValueError("frequencies must lie on a regular grid")

    return nf, df, dnf


def direct_summations_single_freq(t, y, w, freq, nharmonics):
    """
    Compute summations (C, S, CC, ...) via direct summation
    for a single frequency
    """

    ybar = np.dot(w, y)

    wt = 2 * np.pi * freq * t


    YC = np.array([ np.dot(w, np.multiply(y-ybar, np.cos(wt * (h+1))))\
                                 for h in range(nharmonics) ])

    YS = np.array([ np.dot(w, np.multiply(y-ybar, np.sin(wt * (h+1))))\
                                 for h in range(nharmonics) ])

    C = np.array([ np.dot(w, np.cos(wt * (h+1)))\
                                 for h in range(nharmonics) ])

    S = np.array([ np.dot(w, np.sin(wt * (h+1)))\
                                 for h in range(nharmonics) ])

    CC = np.zeros((nharmonics, nharmonics))
    CS = np.zeros((nharmonics, nharmonics))
    SS = np.zeros((nharmonics, nharmonics))

    for h1 in range(nharmonics):
        for h2 in range(nharmonics):
            CC[h1][h2] = np.dot(w, np.multiply(np.cos(wt * (h1+1)),
                                               np.cos(wt * (h2+1))))

            CS[h1][h2] = np.dot(w, np.multiply(np.cos(wt * (h1+1)),
                                               np.sin(wt * (h2+1))))

            SS[h1][h2] = np.dot(w, np.multiply(np.sin(wt * (h1+1)),
                                               np.sin(wt * (h2+1))))

            CC[h1][h2] -= C[h1] * C[h2]
            CS[h1][h2] -= C[h1] * S[h2]
            SS[h1][h2] -= S[h1] * S[h2]

    return Summations(C=C, S=S, YC=YC, YS=YS, CC=CC, CS=CS, SS=SS)


def direct_summations(t, y, w, freqs, nh):
    """
    Compute summations (C, S, CC, ...) via direct summation
    for one or more frequencies
    """

    multi_freq = hasattr(freqs, '__iter__')

    if multi_freq:
        return [ direct_summations_single_freq(t, y, w, frq, nh)\
                                                      for frq in freqs ]
    else:
        return direct_summations_single_freq(t, y, w, freqs, nh)


def fast_summations(t, y, w, freqs, nh, eps=1E-5):
    """
    Computes C, S, YC, YS, CC, CS, SS using
    pyNFFT
    """

    nf, df, dnf = inspect_freqs(freqs)
    tmin = min(t)

    # infer samples per peak
    baseline = max(t) - tmin
    samples_per_peak = 1./(baseline * df)

    eps = 1E-5
    a = 0.5 - eps
    r = 2 * a / df

    tshift = a * (2 * (t - tmin) / r - 1)

    # number of frequencies needed for NFFT
    # need nf_nfft_u / 2 - 1 =  H * (nf - 1 + dnf)
    #      nf_nfft_w / 2 - 1 = 2H * (nf - 1 + dnf)
    nf_nfft_u = 2 * (     nh * (nf + dnf - 1) + 1)
    nf_nfft_w = 2 * ( 2 * nh * (nf + dnf - 1) + 1)
    n_w0 = int(floor(nf_nfft_w/2))
    n_u0 = int(floor(nf_nfft_u/2))

    # transform y -> w_i * y_i - ybar
    ybar = np.dot(w, y)
    u = np.multiply(w, y - ybar)

    # plan NFFT's and precompute
    plan = NFFT(nf_nfft_w, len(tshift))
    plan.x = tshift
    plan.precompute()

    plan2 = NFFT(nf_nfft_u, len(tshift))
    plan2.x = tshift
    plan2.precompute()

    # NFFT(weights)
    plan.f = w

    f_hat_w = plan.adjoint()[n_w0:]

    # NFFT(y - ybar)
    plan2.f = u
    f_hat_u = plan2.adjoint()[n_u0:]

    # now correct for phase shift induced by transforming t -> (-1/2, 1/2)
    beta = -a * (2 * tmin / r + 1)
    I = 0. + 1j
    twiddles = np.exp(- I * 2 * np.pi * np.arange(0, n_w0) * beta)
    f_hat_u *= twiddles[:len(f_hat_u)]
    f_hat_w *= twiddles[:len(f_hat_w)]

    all_computed_sums = []

    # Now compute the summation values at each frequency
    for i in range(nf):
        j = np.arange(2 * nh)
        k = (j + 1) * (i + dnf)
        C = f_hat_w[k].real
        S = f_hat_w[k].imag
        YC = f_hat_u[k[:nh]].real
        YS = f_hat_u[k[:nh]].imag

        #-------------------------------
        # Note: redefining j and k here!
        k = np.arange(nh)
        j = k[:, np.newaxis]

        Sn  = np.sign(k - j) * S[abs(k - j) - 1]
        Sn.flat[::nh + 1] = 0  # set diagonal to zero

        Cn = C[abs(k - j) - 1]
        Cn.flat[::nh + 1] = 1  # set diagonal to one

        Sp = S[j + k + 1]
        Cp = C[j + k + 1]

        CC = 0.5 * (Cn + Cp) - C[j] * C[k]
        CS = 0.5 * (Sn + Sp) - C[j] * S[k]
        SS = 0.5 * (Cn - Cp) - S[j] * S[k]

        all_computed_sums.append(Summations(C=C[:nh], S=S[:nh],
                                            YC=YC, YS=YS,
                                            CC=CC, CS=CS, SS=SS))

    return all_computed_sums
