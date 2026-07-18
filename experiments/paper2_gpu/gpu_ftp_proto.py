"""xp-agnostic (numpy / CuPy) batched multiband floating_offsets FTP.

A GPU-portable prototype of the batched pipeline in ftperiodogram.multiband,
written so `xp = numpy` (CPU reference) or `xp = cupy` (GPU) both work. Batched
over BOTH sources and frequencies -- the Paper-2 throughput regime (many sparse
griz light curves). Validated on CPU against the ftperiodogram package, then
benchmarked on a datacenter FP64 GPU.

Scope: the dominant Stage-2 (assembly) + Stage-3 (scan+Newton) kernels on
well-conditioned data. The deep-|MM|-dip exact-root fallback is omitted (it
fires ~0% on well-conditioned griz at the multiband deferral threshold); a
production port routes those rare rows through a batched Aberth-Ehrlich rooter
(validated separately). FP64 throughout.
"""
import numpy as np


def assemble_YM_MM_AC(xp, cn, sn, C, S, YC, YS, CC, CS, SS):
    """Batched sums -> (YM, MM, AC) coefficient stacks. Leading axes of the sum
    arrays are an arbitrary batch B...; C/S/YC/YS end in H, CC/CS/SS in (H,H).
    Port of core.batched_YM_MM_from_sums (vectorized np.trace diagonal sums)."""
    H = len(cn)
    cn = xp.asarray(cn); sn = xp.asarray(sn)
    alpha = 0.5 * (cn + 1j * sn)                                   # (H,)

    aYC = alpha * (YC - 1j * YS)                                  # (B..., H)
    zeros = xp.zeros(aYC.shape[:-1] + (1,), dtype=xp.complex128)
    YM = xp.concatenate((xp.flip(xp.conj(aYC), axis=-1), zeros, aYC), axis=-1)

    UU = CC + 1j * CS
    VV = SS + 1j * xp.swapaxes(CS, -1, -2)
    aa = xp.outer(alpha, alpha)
    aac = xp.outer(alpha, xp.conj(alpha))
    CCk = (xp.conj(UU) - VV) * aa
    CSk = (UU + xp.conj(VV)) * aac
    SSk = xp.conj(CCk)

    CCk = xp.ascontiguousarray(xp.flip(CCk, axis=-2))
    CSk = xp.ascontiguousarray(xp.swapaxes(CSk, -1, -2))
    SSk = xp.ascontiguousarray(xp.flip(SSk, axis=-1))

    offs = range(-H + 1, H)
    def diags(M):
        return xp.stack([xp.trace(M, offset=o, axis1=-2, axis2=-1)
                         for o in offs], axis=-1)
    CC_d = diags(CCk); CS_d = diags(CSk); SS_d = diags(SSk)

    B = YM.shape[:-1]
    MM = xp.zeros(B + (4 * H + 1,), dtype=xp.complex128)
    inds = xp.arange(2 * H - 1)
    MM[..., inds] += SS_d
    MM[..., inds + H + 1] += 2 * CS_d
    MM[..., inds + 2 * H + 2] += CC_d

    AC = alpha * (C - 1j * S)
    return YM, MM, AC


def _eval_circle(xp, coefs, M):
    """Evaluate stacked polys at M uniform circle angles via zero-padded ifft
    on the last axis. coefs (..., ncoef) -> (..., M)."""
    return M * xp.fft.ifft(coefs, n=M, axis=-1)


def _horner(xp, coefs, phi):
    """coefs (..., ncoef), phi (...,) -> (...,) complex."""
    out = xp.zeros(phi.shape, dtype=xp.complex128)
    for j in range(coefs.shape[-1] - 1, -1, -1):
        out = out * phi + coefs[..., j]
    return out


def scan_powers(xp, YM, MM, YY, H, n_newton=8, n_angles=None):
    """Batched circle-scan + Newton-polish maximizer -> powers (B...,).
    Simplified: brackets every circular local max of P and polishes each; no
    dip candidates, no exact fallback (well-conditioned regime). Matches
    core.scan_polish_from_coefs powers to ~1e-9 on well-conditioned data."""
    M = max(128, 32 * H) if n_angles is None else n_angles
    flat = YM.reshape(-1, YM.shape[-1])
    flatM = MM.reshape(-1, MM.shape[-1])
    nB = flat.shape[0]
    # YY is per-source (batch) -- broadcast across the frequency axis, then flatten
    YYf = xp.broadcast_to(xp.asarray(YY), YM.shape[:-1]).reshape(-1)

    Yv = _eval_circle(xp, flat, M)                               # (nB, M)
    Mv = _eval_circle(xp, flatM, M)
    Pg = xp.real(Yv * Yv / Mv) / YYf[:, None]
    Pg = xp.where(xp.isfinite(Pg), Pg, -xp.inf)

    left = xp.roll(Pg, 1, axis=1); right = xp.roll(Pg, -1, axis=1)
    ismax = (Pg >= left) & (Pg >= right) & (Pg > -xp.inf)        # (nB, M)

    # For a fixed candidate budget (GPU-friendly), keep the top-C maxima per row.
    Cmax = 4 * H
    Pmask = xp.where(ismax, Pg, -xp.inf)
    # indices of top-Cmax angles per row
    order = xp.flip(xp.argsort(Pmask, axis=1), axis=1)[:, :Cmax]         # (nB, Cmax)
    theta0 = (2 * np.pi / M) * order                            # seed angles
    valid = xp.take_along_axis(Pmask, order, axis=1) > -xp.inf

    kY = xp.arange(flat.shape[1]); kM = xp.arange(flatM.shape[1])
    YMc = flat[:, None, :]; MMc = flatM[:, None, :]             # (nB,1,ncoef)
    dY1 = YMc * kY; dY2 = YMc * (kY * kY)
    dM1 = MMc * kM; dM2 = MMc * (kM * kM)
    half = 2 * np.pi / M
    theta = theta0.copy()
    for _ in range(n_newton):
        phi = xp.exp(1j * theta)                                # (nB, Cmax)
        Y = _horner(xp, YMc, phi); Y1 = _horner(xp, dY1, phi); Y2 = _horner(xp, dY2, phi)
        Mm = _horner(xp, MMc, phi); M1 = _horner(xp, dM1, phi); M2 = _horner(xp, dM2, phi)
        Bc = 2.0 * Y * Y1 * Mm - Y * Y * M1
        dP = xp.real(1j * Bc / (Mm * Mm))
        d2P = xp.real((-2.0 * (Y1 * Y1 + Y * Y2) * Mm + Y * Y * M2) / (Mm * Mm)
                      + 2.0 * Bc * M1 / (Mm * Mm * Mm))
        step = dP / d2P
        step = xp.where(xp.isfinite(step), step, 0.0)
        theta = xp.clip(theta - step, theta0 - half, theta0 + half)

    phi = xp.exp(1j * theta)
    Yp = _horner(xp, YMc, phi); Mp = _horner(xp, MMc, phi)
    Pp = xp.real(Yp * Yp / Mp) / YYf[:, None]
    P0 = xp.take_along_axis(Pg, order, axis=1)
    Pc = xp.where(xp.isfinite(Pp) & (Pp >= P0), Pp, P0)
    Pc = xp.where(valid, Pc, -xp.inf)
    best = xp.max(Pc, axis=1)
    best = xp.where(xp.isfinite(best), best, 0.0)
    return best.reshape(YM.shape[:-1])


def direct_sums(xp, t, w, u, freqs, H):
    """Batched direct summations for a batch of sources in ONE band.
    t/w/u: (B, N) padded (w=0 on padding); u = w*(y-ybar). freqs (m,).
    Returns stacked sums with leading (B, m): C,S,YC,YS (...,2H or H),
    CC,CS,SS (...,H,H). Same uncentered product-to-sum form as the package."""
    t = xp.asarray(t); w = xp.asarray(w); u = xp.asarray(u); fr = xp.asarray(freqs)
    B, N = t.shape; m = fr.shape[0]
    ph = 2 * np.pi * (fr[None, :, None] * t[:, None, :])          # (B, m, N)
    mh = xp.arange(1, 2 * H + 1)
    ang = mh[None, None, :, None] * ph[:, :, None, :]             # (B, m, 2H, N)
    cosA = xp.cos(ang); sinA = xp.sin(ang)
    C = xp.einsum('bmhn,bn->bmh', cosA, w)                       # (B, m, 2H)
    S = xp.einsum('bmhn,bn->bmh', sinA, w)
    YC = xp.einsum('bmhn,bn->bmh', cosA[:, :, :H, :], u)         # (B, m, H)
    YS = xp.einsum('bmhn,bn->bmh', sinA[:, :, :H, :], u)
    k = xp.arange(H); j = k[:, None]
    ak = xp.abs(k - j)
    Sn = xp.sign(k - j) * S[..., ak - 1]                         # (B,m,H,H)
    idx = xp.arange(H)
    Sn[..., idx, idx] = 0
    Cn = C[..., ak - 1].copy()
    Cn[..., idx, idx] = 1
    Sp = S[..., j + k + 1]; Cp = C[..., j + k + 1]
    Cj = C[..., :H, None]; Ck = C[..., None, :H]
    Sj = S[..., :H, None]; Sk = S[..., None, :H]
    CC = 0.5 * (Cn + Cp) - Cj * Ck
    CS = 0.5 * (Sn + Sp) - Cj * Sk
    SS = 0.5 * (Cn - Cp) - Sj * Sk
    return C[..., :H], S[..., :H], YC, YS, CC, CS, SS


def floating_offsets_powers(xp, band_arrays, template, freqs, chunk=2048,
                            n_newton=8):
    """Full batched floating_offsets power spectrum for a batch of sources.
    band_arrays: list over bands of dicts {t,w,u,W,ybar_global,YY} where t/w/u
    are (B,N_b) padded, W (B,), ybar_global (B,), YY (B,) shared across bands.
    template: (cn, sn). Returns powers (B, nfreq)."""
    cn, sn = template
    H = len(cn)
    B = band_arrays[0]['t'].shape[0]
    nfreq = len(freqs)
    YY = xp.asarray(band_arrays[0]['YY'])                        # (B,)
    out = xp.empty((B, nfreq), dtype=xp.float64)
    for i0 in range(0, nfreq, chunk):
        fr = freqs[i0:i0 + chunk]; mm = len(fr)
        YM = MM = None
        for bd in band_arrays:
            C, S, YC, YS, CC, CS, SS = direct_sums(xp, bd['t'], bd['w'], bd['u'], fr, H)
            YM_k, MM_k, AC_k = assemble_YM_MM_AC(xp, cn, sn, C, S, YC, YS, CC, CS, SS)
            W = xp.asarray(bd['W'])[:, None, None]              # (B,1,1)
            YM = W * YM_k if YM is None else YM + W * YM_k
            MM = W * MM_k if MM is None else MM + W * MM_k
        pw = scan_powers(xp, YM, MM, YY[:, None], H, n_newton=n_newton)  # (B, mm)
        out[:, i0:i0 + mm] = pw
    return out
