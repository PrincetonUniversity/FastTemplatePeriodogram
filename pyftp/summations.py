from pynfft.nfft import NFFT
from .utils import Summations, weights
import numpy as np

def shift_t_for_nfft(t, samples_per_peak):
    """ transforms t to [-1/2, 1/2] interval """

    r = samples_per_peak * (max(t) - min(t))
    eps = 1E-5
    a = 0.5 - eps

    return 2 * a * (t - min(t)) / r - a

def direct_summations(t, y, w, freq, nharmonics):
    """ for testing against NFFT implementation """

    # shift the data to [-1/2, 1/2); then scale so that
    # frequency * t gives expected phase
    #tt = t - t[0]#shift_t_for_nfft(t, 1) * (max(t) - min(t))
    ybar = np.dot(w, y)

    wt = 2 * np.pi * freq * shift_t_for_nfft(t, 1) * (max(t) - min(t))


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

def assert_close(x, y, tol=1E-3):
    assert( abs(x - y) < tol * 0.5 * (x + y) )


def fast_summations(t, y, w, freqs, nharmonics):
    """
    Computes C, S, YC, YS, CC, CS, SS using
    pyNFFT
    """

    df = freqs[1] - freqs[0]
    nf = len(freqs)
    
    assert_close((nf-1) * df + freqs[0], freqs[-1])

    if abs(freqs[0] / df - round(freqs[0] / df)) > 1E-3:
        raise ValueError("Minimum frequency must be a multiple of df")

    if not all([ abs(freqs[i] - freqs[i-1] - df) < 1E-3*df for i in range(1, len(freqs)) ]):
        raise ValueError("Frequencies are not evenly spaced!")

    # offset for minimum frequency
    dnf = int(round(freqs[0] / df))

    # infer samples per peak
    baseline = max(t) - min(t)
    samples_per_peak = 1./(baseline * df)

    # number of frequencies needed for NFFT
    # need nf_nfft_u / 2 - 1 =  H * (nf - 1 + dnf)
    #      nf_nfft_w / 2 - 1 = 2H * (nf - 1 + dnf)
    nf_nfft_u = 2 * ( nharmonics * (nf + dnf - 1) + 1)
    nf_nfft_w = 2 * ( 2 * nharmonics * (nf + dnf - 1) + 1)

    # shift times to [ -1/2, 1/2 ]
    tshift = shift_t_for_nfft(t, samples_per_peak)

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
    f_hat_w = plan.adjoint()[nf_nfft_w/2 :]

    # NFFT(y - ybar)
    plan2.f = u
    f_hat_u = plan2.adjoint()[nf_nfft_u/2 :]

    all_computed_sums = []

    # Now compute the summation values at each frequency
    for i in range(0, nf):
        computed_sums = Summations(C=np.zeros(nharmonics),
                                   S=np.zeros(nharmonics),
                                   YC=np.zeros(nharmonics),
                                   YS=np.zeros(nharmonics),
                                   CC=np.zeros((nharmonics,nharmonics)),
                                   CS=np.zeros((nharmonics,nharmonics)),
                                   SS=np.zeros((nharmonics,nharmonics)))

        C_, S_ = np.zeros(2 * nharmonics), np.zeros(2 * nharmonics)
        for j in range(2 * nharmonics):
            k = (j + 1) * (i + dnf)
            C_[j] =  f_hat_w[k].real
            S_[j] =  f_hat_w[k].imag
            if j < nharmonics:

                computed_sums.YC[j] =  f_hat_u[k].real
                computed_sums.YS[j] =  f_hat_u[k].imag

        for j in range(nharmonics):
            for k in range(nharmonics):
                Sn, Cn = None, None

                if j == k:
                    Sn = 0
                    Cn = 1
                else:
                    Sn =  np.sign(k - j) * S_[int(abs(k - j)) - 1]
                    Cn =  C_[int(abs(k - j)) - 1]

                Sp = S_[j + k + 1]
                Cp = C_[j + k + 1]

                computed_sums.CC[j][k] = 0.5 * ( Cn + Cp ) - C_[j] * C_[k]
                computed_sums.CS[j][k] = 0.5 * ( Sn + Sp ) - C_[j] * S_[k]
                computed_sums.SS[j][k] = 0.5 * ( Cn - Cp ) - S_[j] * S_[k]

        computed_sums.C[:] = C_[:nharmonics]
        computed_sums.S[:] = S_[:nharmonics]

        all_computed_sums.append(computed_sums)

    return all_computed_sums

