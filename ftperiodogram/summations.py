import warnings

from nfft import nfft_adjoint
from .utils import Summations
import numpy as np
from math import floor


def _warn_if_concentrated_weights(w):
    """Warn when the inverse-variance weights are concentrated enough to
    trigger catastrophic cancellation in the ``E[xy] - E[x]E[y]`` moment sums.

    The fast (NFFT) path computes the raw moments and subtracts the means, so
    when a handful of points (fewer than ~the model dof) dominate the weight
    budget the subtraction cancels catastrophically.  The direct path avoids
    this with two-pass centered sums, hence the ``fast=False`` advice.
    """
    w = np.asarray(w)
    if w.size < 2:
        return
    total = np.sum(w)
    if not (total > 0):
        return
    wmax = np.max(w)
    wmin = np.min(w)
    share = wmax / total
    ratio = wmax / wmin if wmin > 0 else np.inf
    if share > 1 - 1e-6 or ratio > 1e6:
        warnings.warn(
            "Inverse-variance weights are highly concentrated "
            "(largest single-point weight share = {0:.6g}, "
            "max(w)/min(w) = {1:.3g}). The NFFT fast path can lose "
            "precision to catastrophic cancellation in the moment sums "
            "when fewer points than the model dof dominate the weight "
            "budget; consider fast=False (direct summations, which use "
            "centered two-pass sums).".format(share, ratio),
            UserWarning)


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
    for a single frequency.

    The covariance sums are computed as two-pass *centered* moments: first
    the weighted first moments (C, S, ybar), then moments of the residuals
    (cos - C, sin - S, y - ybar).  This is analytically identical to the
    uncentered ``E[xy] - E[x]E[y]`` form (the weights are normalized) but
    avoids its catastrophic cancellation when the inverse-variance weight
    concentrates in fewer points than ~the model dof.
    """
    ybar = np.dot(w, y)
    wt = 2 * np.pi * freq * t
    h = 1 + np.arange(nharmonics)[:, np.newaxis]

    ch = np.cos(h * wt)
    sh = np.sin(h * wt)

    # first pass: weighted first moments
    C = np.dot(ch, w)
    S = np.dot(sh, w)

    # second pass: moments of the centered residuals
    dc = ch - C[:, np.newaxis]
    ds = sh - S[:, np.newaxis]
    yres = y - ybar

    YC = np.dot(dc, w * yres)
    YS = np.dot(ds, w * yres)

    wdc = w * dc
    CC = np.dot(wdc, dc.T)
    CS = np.dot(wdc, ds.T)
    SS = np.dot(w * ds, ds.T)

    return Summations(C=C, S=S, YC=YC, YS=YS, CC=CC, CS=CS, SS=SS)


def _direct_summations_single_freq_uncentered(t, y, w, freq, nharmonics):
    """Uncentered (single-pass) covariance sums: ``E[xy] - E[x]E[y]``.

    Retained as the regression reference for the weight-conditioning tests
    (tests/test_weight_conditioning.py): this form cancels catastrophically
    under concentrated inverse-variance weights.  Production code uses the
    centered :func:`direct_summations_single_freq`.
    """
    ybar = np.dot(w, y)
    wt = 2 * np.pi * freq * t
    h = 1 + np.arange(nharmonics)[:, np.newaxis]

    C = np.dot(np.cos(h * wt), w)
    S = np.dot(np.sin(h * wt), w)

    YC = np.dot((y - ybar) * np.cos(h * wt), w)
    YS = np.dot((y - ybar) * np.sin(h * wt), w)

    hT = h[:, :, np.newaxis]

    CC = np.dot(np.cos(hT * wt) * np.cos(h * wt), w)
    CS = np.dot(np.cos(hT * wt) * np.sin(h * wt), w)
    SS = np.dot(np.sin(hT * wt) * np.sin(h * wt), w)

    CC -= C[:, np.newaxis] * C
    CS -= C[:, np.newaxis] * S
    SS -= S[:, np.newaxis] * S

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



def fast_summations(t, y, w, freqs, nh, sigma=2, tol=1E-7, m=None, 
                        kernel='gaussian', use_fft=True, truncated=True):
    """
    Computes C, S, YC, YS, CC, CS, SS using
    nfft Python implementation by Jake Vanderplas
    """
    _warn_if_concentrated_weights(w)

    nfft_kwargs = dict(sigma=sigma, tol=tol, m=m,
                        kernel=kernel, use_fft=use_fft, 
                        truncated=truncated)

    nf, df, dnf = inspect_freqs(freqs)
    tmin = min(t)

    # infer samples per peak
    baseline = max(t) - tmin
    samples_per_peak = 1./(baseline * df)

    a = 0.5 - 1E-8
    r = 2 * a / df

    tshift = a * (2 * (t - tmin) / r - 1)

    # number of frequencies needed for NFFT
    # need nf_nfft_u / 2 - 1 =  H * (nf - 1 + dnf)
    #      nf_nfft_w / 2 - 1 = 2H * (nf - 1 + dnf)
    nf_nfft_u = 2 * (     nh * (nf + dnf - 1) + 1)
    nf_nfft_w = 2 * ( 2 * nh * (nf + dnf - 1) + 1)

    # transform y -> w_i * y_i - ybar
    ybar = np.dot(w, y)
    u = np.multiply(w, y - ybar)

    
    n_w0 = int(floor(nf_nfft_w/2))
    n_u0 = int(floor(nf_nfft_u/2))
    f_hat_u = nfft_adjoint(tshift, u, nf_nfft_u, **nfft_kwargs )[n_u0:]
    f_hat_w = nfft_adjoint(tshift, w, nf_nfft_w, **nfft_kwargs )[n_w0:]

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
