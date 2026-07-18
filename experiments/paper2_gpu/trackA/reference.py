"""Pure-numpy bit-mirrors of the Track-A CUDA kernels (kernels.py).

Each function here re-implements ONE kernel's algorithm with the SAME
operation ORDER as the CUDA C source (thread loops written as Python
loops / vectorized elementwise ops), so the GPU math can be validated
LOCALLY (no GPU on this machine) against the real ftperiodogram package,
and the GPU outputs can later be compared against these mirrors on the
pod at near-machine tolerances.

Mirror discipline (why this file looks low-level):
- Complex multiply is spelled out via ``_cmul`` (plain 4-mul/2-add formula)
  because numpy's complex-multiply ufunc may use fused (FCMLA/FMA) SIMD
  paths whose rounding differs from the kernel's ``--fmad=false`` code.
  Complex add/sub/scale-by-real are componentwise and fusion-free, so
  plain numpy operators are used for those.
- ``Re(a/b)`` is ``_creal_div`` (numerator ``a.re*b.re + a.im*b.im`` over
  ``|b|^2``), NOT numpy's Smith-algorithm complex division: the kernel
  uses the same explicit formula.
- ``|z|`` is ``sqrt(re^2 + im^2)`` (NOT ``np.abs`` = hypot): matches the
  kernel. This can differ from the package by ~1 ulp; validate_local
  checks that no fixture row's deferral/dip classification flips.
- ``phi = e^{i theta}`` is built as ``cos + i sin`` (kernel: ``sincos``).
- Reductions that must match the kernel (Horner, diagonal sums, epoch
  accumulation) are explicit serial loops in the same order; np.min/max
  are used only where the reduction is exact (min/max of finite doubles).

Residual mirror-vs-kernel differences (documented, gated on the pod by
tolerance rather than bitwise equality): libm-vs-CUDA sin/cos (<= ~1-2
ulp), pocketfft-vs-cuFFT circle evaluation, and any numpy real-ufunc SIMD
reassociation. Everything else is order-identical IEEE FP64.

Behavioral reference: ftperiodogram.core.scan_polish_from_coefs (WP C2/C4
frozen scan), ftperiodogram.core.batched_YM_MM_from_sums, and
ftperiodogram.summations._direct_stacked_sums_chunk. See kernels.py for
the CUDA sources these functions mirror line-by-line.
"""
import numpy as np

TWO_PI = 2.0 * np.pi

# Scan constants -- keep in sync with ftperiodogram.core (asserted by
# validate_local.py against the package values).
SCAN_MIN_ANGLES = 128
SCAN_ANGLES_PER_H = 32
SCAN_NEWTON_STEPS = 8
SCAN_MAX_CANDIDATES_PER_H = 4
SCAN_DIP_RTOL = 0.5
SCAN_EXACT_RTOL = 0.15          # host routes rows below this to the exact path
BATCHED_DEFER_RTOL = 3e-3       # multiband batched deferral threshold

# Status flags (keep in sync with kernels.py STATUS_* -- validate asserts).
FLAG_FLAT = 1          # no usable candidate -> flat zero-power fit
FLAG_WINNER_DIP = 2    # winning candidate was a dip candidate
FLAG_CAP_MAX = 4       # grid-maxima candidate list hit the kcap trim
FLAG_CAP_DIP = 8       # dip candidate list hit the kcap trim
FLAG_K18_CHANGED = 16  # K.18 filter changed the winner vs unfiltered argmax
FLAG_DIP_CAND = 32     # row had at least one dip candidate


# ----------------------------------------------------------------------
# Complex helpers with kernel-identical operation order
# ----------------------------------------------------------------------
def _cplx(re, im):
    """Exact complex construction from real/imag parts (no arithmetic)."""
    re = np.asarray(re, dtype=np.float64)
    im = np.asarray(im, dtype=np.float64)
    z = np.empty(np.broadcast(re, im).shape, dtype=np.complex128)
    z.real, z.imag = re, im
    return z


def _cmul(a, b):
    """Plain-formula complex multiply: (ar*br - ai*bi, ar*bi + ai*br).

    Mirrors the kernel's ``cmul`` compiled with ``--fmad=false``: two
    products and one add/sub per component, each individually rounded.
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    return _cplx(a.real * b.real - a.imag * b.imag,
                 a.real * b.imag + a.imag * b.real)


def _cconj(a):
    a = np.asarray(a, dtype=np.complex128)
    return _cplx(a.real, -a.imag)


def _creal_div(a, b):
    """Re(a / b) = (ar*br + ai*bi) / (br^2 + bi^2), kernel ``creal_div``."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    return (a.real * b.real + a.imag * b.imag) / (b.real * b.real +
                                                  b.imag * b.imag)


def _cabs(z):
    """|z| = sqrt(re^2 + im^2), kernel ``cabs_`` (NOT hypot)."""
    z = np.asarray(z, dtype=np.complex128)
    return np.sqrt(z.real * z.real + z.imag * z.imag)


def _cpow_int(z, e):
    """Binary exponentiation z**e (e >= 0 int), kernel ``cpow_int``."""
    r = np.ones_like(np.asarray(z, dtype=np.complex128))
    b = np.array(z, dtype=np.complex128, copy=True)
    e = int(e)
    while e:
        if e & 1:
            r = _cmul(r, b)
        b = _cmul(b, b)
        e >>= 1
    return r


def _horner_c(coefs, phi):
    """Horner evaluation; kernel ``horner_c``.

    coefs: (..., ncoef) rows aligned with phi (...,).
    """
    coefs = np.asarray(coefs, dtype=np.complex128)
    a = np.zeros(np.asarray(phi).shape, dtype=np.complex128)
    for j in range(coefs.shape[-1] - 1, -1, -1):
        a = _cmul(a, phi) + coefs[..., j]
    return a


def _horner012(coefs, phi):
    """Fused Horner of p, its k-weighted, and k^2-weighted companions;
    kernel ``horner012``. The k / k^2 derivative weights are applied
    on the fly (float(j), float(j*j)) -- no pre-materialized dY1/dY2
    coefficient arrays, matching design requirement (iii)."""
    coefs = np.asarray(coefs, dtype=np.complex128)
    shape = np.asarray(phi).shape
    a0 = np.zeros(shape, dtype=np.complex128)
    a1 = np.zeros(shape, dtype=np.complex128)
    a2 = np.zeros(shape, dtype=np.complex128)
    for j in range(coefs.shape[-1] - 1, -1, -1):
        c = coefs[..., j]
        a0 = _cmul(a0, phi) + c
        a1 = _cmul(a1, phi) + float(j) * c
        a2 = _cmul(a2, phi) + float(j * j) * c
    return a0, a1, a2


# ----------------------------------------------------------------------
# Stage 1 -- direct summations (mirror of stage1_direct_sums kernel)
# ----------------------------------------------------------------------
def stage1_direct_sums_ref(t, w, u, freqs, H):
    """Direct sums for a padded batch of sources in ONE band.

    t/w/u: (B, Nmax) zero-padded (w = u = 0 on padding; padded epochs
    contribute exact +0.0, identical to the kernel's n < N loop up to the
    sign of zero). u = w * (y - ybar_band). freqs: (m,).

    Returns dict(C, S, YC, YS) of (B, m, H) and (CC, CS, SS) of
    (B, m, H, H) -- the ftperiodogram.summations.direct_summations_batched
    per-source output layout (uncentered product-to-sum covariance form).

    Kernel correspondence: one block per source; each thread owns one
    frequency; per epoch the base angle ``ph = TWO_PI * (f * t_n)`` costs
    ONE sincos, then the 2H-harmonic ladder is the complex-exponential
    (Chebyshev) recurrence ``e^{i(h+1)ph} = e^{i h ph} e^{i ph}`` --
    design requirement: no 2H transcendental calls per epoch-freq.
    Accumulation order: epochs outer, harmonics inner.
    """
    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    B, Nmax = t.shape
    m = freqs.shape[0]
    H2 = 2 * H

    f = freqs[None, :]                                  # (1, m)
    Cf = [np.zeros((B, m)) for _ in range(H2)]
    Sf = [np.zeros((B, m)) for _ in range(H2)]
    YCa = [np.zeros((B, m)) for _ in range(H)]
    YSa = [np.zeros((B, m)) for _ in range(H)]

    for n in range(Nmax):
        tn = t[:, n][:, None]                           # (B, 1)
        wn = w[:, n][:, None]
        un = u[:, n][:, None]
        ph = TWO_PI * (f * tn)                          # (B, m)
        c1 = np.cos(ph)
        s1 = np.sin(ph)
        ch = c1.copy()
        sh = s1.copy()
        for h in range(H2):
            if h > 0:
                chn = ch * c1 - sh * s1
                sh = sh * c1 + ch * s1                  # uses OLD ch
                ch = chn
            Cf[h] = Cf[h] + wn * ch
            Sf[h] = Sf[h] + wn * sh
            if h < H:
                YCa[h] = YCa[h] + un * ch
                YSa[h] = YSa[h] + un * sh

    C = np.stack(Cf[:H], axis=-1)                       # (B, m, H)
    S = np.stack(Sf[:H], axis=-1)
    YC = np.stack(YCa, axis=-1)
    YS = np.stack(YSa, axis=-1)

    CC = np.empty((B, m, H, H))
    CS = np.empty((B, m, H, H))
    SS = np.empty((B, m, H, H))
    for j in range(H):
        for k in range(H):
            if j == k:
                cn_ = 1.0
                sn_ = 0.0
            else:
                a = abs(k - j)
                cn_ = Cf[a - 1]
                sn_ = Sf[a - 1] if k > j else -Sf[a - 1]
            cp = Cf[j + k + 1]
            sp = Sf[j + k + 1]
            # uncentered E[xy] - E[x]E[y] moment form (the documented
            # weight-concentration cancellation lives in these subtractions;
            # operation order: halved product-to-sum, then subtract the
            # mean product -- identical to summations.py)
            CC[:, :, j, k] = 0.5 * (cn_ + cp) - Cf[j] * Cf[k]
            CS[:, :, j, k] = 0.5 * (sn_ + sp) - Cf[j] * Sf[k]
            SS[:, :, j, k] = 0.5 * (cn_ - cp) - Sf[j] * Sf[k]

    return dict(C=C, S=S, YC=YC, YS=YS, CC=CC, CS=CS, SS=SS)


# ----------------------------------------------------------------------
# Stage 2 -- YM/MM/AC assembly (mirror of stage2_assemble kernel)
# ----------------------------------------------------------------------
def stage2_assemble_ref(sums, cn, sn, H=None):
    """Per-band sums -> (YM, MM, AC) coefficient stacks for one band.

    sums: dict with C/S/YC/YS of shape (..., Hs) and CC/CS/SS of
    (..., Hs, Hs). cn/sn: template Fourier coefficients, length H <= Hs
    (H < Hs = the subtype H-prefix reuse path: the top-left H x H block
    of an Hs-assembled covariance IS the H assembly, since CC_{jk} only
    involves harmonics <= j + k + 2 <= 2H).

    Returns (YM, MM, AC): (..., 2H+1), (..., 4H+1), (..., H) complex128.

    Mirrors core.batched_YM_MM_from_sums with DOCUMENTED operation order
    (this is the numerically load-bearing stage -- GPU_FEASIBILITY par.7
    risk #1):
      1. alpha = 0.5*(cn + i sn); aYC = alpha * (YC - i YS);
         YM = [conj(aYC) reversed, 0, aYC].
      2. X  = (CC - SS) - i(CS + CS^T)   [= conj(UU) - VV]
         Yc = (CC + SS) + i(CS - CS^T)   [= UU + conj(VV)]
         CCk = X * (alpha_j alpha_k); CSk = Yc * (alpha_j conj(alpha_k));
         SSk = conj(CCk) (conjugated per element).
      3. Diagonal sums traverse each diagonal with ascending row index i
         SERIALLY (np.trace uses pairwise summation for length-8 rows --
         a last-ulp difference absorbed by the validation gates):
           CC_d(o) = sum_i CCk[H-1-i, i+o]      (flip rows, trace)
           CS_d(o) = sum_i CSk[i+o, i]          (transpose, trace)
           SS_d(o) = sum_i conj(CCk[i, H-1-i-o]) (conj, flip cols, trace)
      4. MM accumulation order: ALL SS_d into slots [0, 2H-2], then ALL
         2*CS_d into [H+1, 3H-1], then ALL CC_d into [2H+2, 4H]
         (core's exact ordering; the slots overlap, so order matters at
         the last ulp).
      5. AC = alpha * (C - i S).
    """
    cn = np.asarray(cn, dtype=np.float64)
    sn = np.asarray(sn, dtype=np.float64)
    if H is None:
        H = len(cn)
    assert len(cn) == H and len(sn) == H

    C = np.asarray(sums['C'], dtype=np.float64)[..., :H]
    S = np.asarray(sums['S'], dtype=np.float64)[..., :H]
    YC = np.asarray(sums['YC'], dtype=np.float64)[..., :H]
    YS = np.asarray(sums['YS'], dtype=np.float64)[..., :H]
    CC = np.asarray(sums['CC'], dtype=np.float64)[..., :H, :H]
    CS = np.asarray(sums['CS'], dtype=np.float64)[..., :H, :H]
    SS = np.asarray(sums['SS'], dtype=np.float64)[..., :H, :H]

    lead = C.shape[:-1]
    alpha = _cplx(0.5 * cn, 0.5 * sn)                   # (H,)

    # -- YM -----------------------------------------------------------
    aYC = _cmul(alpha, _cplx(YC, -YS))                  # (..., H)
    YM = np.zeros(lead + (2 * H + 1,), dtype=np.complex128)
    YM[..., :H] = _cconj(aYC[..., ::-1])
    YM[..., H + 1:] = aYC

    # -- MM -----------------------------------------------------------
    CSt = np.swapaxes(CS, -1, -2)
    X = _cplx(CC - SS, -(CS + CSt))
    Yc = _cplx(CC + SS, CS - CSt)
    axa = _cmul(alpha[:, None], alpha[None, :])         # (H, H)
    axc = _cmul(alpha[:, None], _cconj(alpha)[None, :])
    CCk = _cmul(X, axa)
    CSk = _cmul(Yc, axc)

    MM = np.zeros(lead + (4 * H + 1,), dtype=np.complex128)
    # section 1: SS diagonals -> slots [0, 2H-2]
    for d in range(2 * H - 1):
        o = d - (H - 1)
        i0, i1 = max(0, -o), min(H - 1, H - 1 - o)
        acc = np.zeros(lead, dtype=np.complex128)
        for i in range(i0, i1 + 1):
            acc = acc + _cconj(CCk[..., i, H - 1 - i - o])
        MM[..., d] = MM[..., d] + acc
    # section 2: 2 * CS diagonals -> slots [H+1, 3H-1]
    for d in range(2 * H - 1):
        o = d - (H - 1)
        i0, i1 = max(0, -o), min(H - 1, H - 1 - o)
        acc = np.zeros(lead, dtype=np.complex128)
        for i in range(i0, i1 + 1):
            acc = acc + CSk[..., i + o, i]
        MM[..., d + H + 1] = MM[..., d + H + 1] + 2.0 * acc
    # section 3: CC diagonals -> slots [2H+2, 4H]
    for d in range(2 * H - 1):
        o = d - (H - 1)
        i0, i1 = max(0, -o), min(H - 1, H - 1 - o)
        acc = np.zeros(lead, dtype=np.complex128)
        for i in range(i0, i1 + 1):
            acc = acc + CCk[..., H - 1 - i, i + o]
        MM[..., d + 2 * H + 2] = MM[..., d + 2 * H + 2] + acc

    # -- AC -----------------------------------------------------------
    AC = _cmul(alpha, _cplx(C, -S))

    return YM, MM, AC


def combine_bands_ref(per_band, W_rows):
    """W_k-weighted band combination (mirror of the stage-2 kernel's
    accumulate mode): first band writes Wk*val, later bands add Wk*val.

    per_band: list over bands of (YM, MM, AC) with leading shape (R, ...);
    W_rows: list over bands of (R,) per-row weights (per-source W_k
    broadcast over the frequency axis by the caller).
    """
    YM = MM = AC = None
    for (YM_k, MM_k, AC_k), wk in zip(per_band, W_rows):
        wk = np.asarray(wk, dtype=np.float64)
        wY = wk[..., None] * YM_k
        wM = wk[..., None] * MM_k
        wA = wk[..., None] * AC_k
        if YM is None:
            YM, MM, AC = wY, wM, wA
        else:
            YM = YM + wY
            MM = MM + wM
            AC = AC + wA
    return YM, MM, AC


# ----------------------------------------------------------------------
# Circle evaluation (host-side in the GPU design: batched cuFFT)
# ----------------------------------------------------------------------
def eval_circle_ref(coefs, M):
    """p(e^{2 pi i m / M}) for every row: M * ifft(coefs, n=M), identical
    to core._eval_polys_on_circle (and to the cupy wrapper's
    ``M * cupy.fft.ifft``). Stays OUTSIDE the stage-3 kernel."""
    coefs = np.asarray(coefs, dtype=np.complex128)
    if coefs.shape[-1] > M:
        raise ValueError("n_angles must be >= ncoef")
    return M * np.fft.ifft(coefs, n=M, axis=-1)


# ----------------------------------------------------------------------
# Stage 3 -- scan + Newton polish maximizer (mirror of stage3_scan_polish)
# ----------------------------------------------------------------------
def stage3_scan_polish_ref(YM, MM, YY, H, n_angles=None,
                           n_newton=SCAN_NEWTON_STEPS,
                           positive_amplitude=True, kcap=None,
                           dip_rtol=SCAN_DIP_RTOL, Yv=None, Mv=None):
    """Per-row candidate detection + Newton polish + K.18 filter + argmax.

    YM: (nrow, 2H+1), MM: (nrow, 4H+1), YY: (nrow,) complex/real stacks.
    Yv/Mv: optional precomputed circle values (nrow, M) -- the kernel
    consumes these plus the coefficient rows; when omitted they are
    computed here exactly as the host wrapper would.

    Returns dict(power, theta, theta1, cond_r, status) of (nrow,) arrays.

    The kernel FLAGS ill-conditioned rows (cond_r) and never solves them:
    the host routes rows with cond_r < SCAN_EXACT_RTOL (scan contract)
    and cond_r < BATCHED_DEFER_RTOL (multiband batched contract) to the
    exact/per-frequency reference paths (host_route_exact_rows below).

    Candidate compaction (audit hunt win #7): only actual circular local
    maxima of P plus deep-|MM|-dip candidates are polished -- never a
    fixed Cmax = 4H top-k.
    """
    YM = np.ascontiguousarray(np.asarray(YM, dtype=np.complex128))
    MM = np.ascontiguousarray(np.asarray(MM, dtype=np.complex128))
    YY = np.asarray(YY, dtype=np.float64)
    nrow = YM.shape[0]
    if kcap is None:
        kcap = SCAN_MAX_CANDIDATES_PER_H * H
    floor = max(SCAN_MIN_ANGLES, SCAN_ANGLES_PER_H * H)
    M = floor if n_angles is None else max(int(n_angles), floor)
    halfw = TWO_PI / M

    if Yv is None:
        Yv = eval_circle_ref(YM, M)
        Mv = eval_circle_ref(MM, M)

    # -- grid scan ----------------------------------------------------
    with np.errstate(all='ignore'):
        Pg = _creal_div(_cmul(Yv, Yv), Mv) / YY[:, None]
    Pg[~np.isfinite(Pg)] = -np.inf
    absM = _cabs(Mv)
    minA = np.min(absM, axis=1)
    maxA = np.max(absM, axis=1)
    with np.errstate(all='ignore'):
        cond_r = minA / maxA

    ismax = ((Pg >= np.roll(Pg, 1, axis=1)) &
             (Pg >= np.roll(Pg, -1, axis=1)) & (Pg > -np.inf))
    ismin = ((absM <= np.roll(absM, 1, axis=1)) &
             (absM <= np.roll(absM, -1, axis=1)) &
             (absM < dip_rtol * maxA[:, None]))

    # -- candidate list building (kernel: thread-0 serial walk) --------
    status = np.zeros(nrow, dtype=np.int64)
    fidx_l, g_l, isdip_l, ncand = [], [], [], np.zeros(nrow, dtype=np.int64)
    for r in range(nrow):
        gm = np.flatnonzero(ismax[r])
        if gm.size > kcap:
            status[r] |= FLAG_CAP_MAX
            # keep the kcap largest Pg, ties -> lower angle index, then
            # restore ascending-angle candidate order (kernel: selection
            # rounds with strict >, then an ascending re-walk)
            order = np.lexsort((gm, -Pg[r, gm]))
            gm = np.sort(gm[order[:kcap]])
        gd = np.flatnonzero(ismin[r])
        if gd.size:
            status[r] |= FLAG_DIP_CAND
        if gd.size > kcap:
            status[r] |= FLAG_CAP_DIP
            order = np.lexsort((gd, absM[r, gd]))
            gd = np.sort(gd[order[:kcap]])
        # per-row candidate order: maxima (ascending angle) then dips
        # (ascending angle) -- core's stable row-major ordering
        fidx_l.append(np.full(gm.size + gd.size, r))
        g_l.append(np.concatenate([gm, gd]))
        isdip_l.append(np.concatenate([np.zeros(gm.size, dtype=bool),
                                       np.ones(gd.size, dtype=bool)]))
        ncand[r] = gm.size + gd.size

    out = dict(power=np.zeros(nrow), theta=np.zeros(nrow),
               theta1=np.zeros(nrow), cond_r=cond_r, status=status,
               n_candidates=ncand)
    if ncand.sum() == 0:
        status |= FLAG_FLAT
        return out

    fidx = np.concatenate(fidx_l)
    g = np.concatenate(g_l)
    isdip = np.concatenate(isdip_l)

    theta0 = (TWO_PI / M) * g
    P0 = Pg[fidx, g]
    Yfb = Yv[fidx, g].copy()
    Mfb = Mv[fidx, g].copy()

    # -- dip refinement: clamped Newton on |MM|^2 ----------------------
    dm = np.flatnonzero(isdip)
    if dm.size:
        MMd = MM[fidx[dm]]
        th0d = theta0[dm]
        th = th0d.copy()
        for _ in range(n_newton):
            phi = _cplx(np.cos(th), np.sin(th))
            Mm, M1, M2 = _horner012(MMd, phi)
            with np.errstate(all='ignore'):
                q1 = 2.0 * (M1.real * Mm.imag - M1.imag * Mm.real)
                q2 = 2.0 * ((M1.real * M1.real + M1.imag * M1.imag) -
                            (Mm.real * M2.real + Mm.imag * M2.imag))
                step = q1 / q2
            step[~np.isfinite(step)] = 0.0
            th = np.clip(th - step, th0d - halfw, th0d + halfw)
        phi = _cplx(np.cos(th), np.sin(th))
        Yd = _horner_c(YM[fidx[dm]], phi)
        Md = _horner_c(MMd, phi)
        with np.errstate(all='ignore'):
            Pd = _creal_div(_cmul(Yd, Yd), Md) / YY[fidx[dm]]
        Pd[~np.isfinite(Pd)] = -np.inf
        # the refined angle becomes the polish seed AND clamp center
        theta0[dm] = th
        P0[dm] = Pd
        Yfb[dm] = Yd
        Mfb[dm] = Md

    # -- Newton polish on dP/dtheta ------------------------------------
    YMc = YM[fidx]
    MMc = MM[fidx]
    YYc = YY[fidx]
    th = theta0.copy()
    for _ in range(n_newton):
        phi = _cplx(np.cos(th), np.sin(th))
        Y, Y1, Y2 = _horner012(YMc, phi)
        Mm, M1, M2 = _horner012(MMc, phi)
        with np.errstate(all='ignore'):
            Bc = _cmul(_cmul(2.0 * Y, Y1), Mm) - _cmul(_cmul(Y, Y), M1)
            Mm2 = _cmul(Mm, Mm)
            Mm3 = _cmul(Mm2, Mm)
            iB = _cplx(-Bc.imag, Bc.real)
            dP = _creal_div(iB, Mm2)
            num = (_cmul(-2.0 * (_cmul(Y1, Y1) + _cmul(Y, Y2)), Mm) +
                   _cmul(_cmul(Y, Y), M2))
            d2P = _creal_div(num, Mm2) + _creal_div(_cmul(2.0 * Bc, M1), Mm3)
            step = dP / d2P
        step[~np.isfinite(step)] = 0.0
        th = np.clip(th - step, theta0 - halfw, theta0 + halfw)

    # -- evaluate polished P, seed fallback ----------------------------
    phi = _cplx(np.cos(th), np.sin(th))
    Yp = _horner_c(YMc, phi)
    Mp = _horner_c(MMc, phi)
    with np.errstate(all='ignore'):
        Pp = _creal_div(_cmul(Yp, Yp), Mp) / YYc
    use_seed = ~np.isfinite(Pp) | (Pp < P0)
    th = np.where(use_seed, theta0, th)
    phi0 = _cplx(np.cos(theta0), np.sin(theta0))
    phi = np.where(use_seed, phi0, phi)
    Yc = np.where(use_seed, Yfb, Yp)
    Mc = np.where(use_seed, Mfb, Mp)
    Pc = np.where(use_seed, P0, Pp)

    with np.errstate(all='ignore'):
        th1 = _creal_div(_cmul(_cpow_int(phi, H), Yc), Mc)

    # -- K.18 positive-amplitude filter + first-wins argmax ------------
    # (kernel: thread-0 serial over the row's candidates)
    with np.errstate(invalid='ignore'):
        pos = th1 >= 0.0                       # NaN -> False
    off = 0
    for r in range(nrow):
        nc = int(ncand[r])
        sl = slice(off, off + nc)
        off += nc
        if nc == 0:
            status[r] |= FLAG_FLAT
            continue
        Pc_r = Pc[sl]
        pos_r = pos[sl]
        win_u, best_u = -1, -np.inf
        for c in range(nc):
            if Pc_r[c] > best_u:
                best_u, win_u = Pc_r[c], c
        anyp = bool(np.any(pos_r)) if positive_amplitude else False
        win, best = -1, -np.inf
        for c in range(nc):
            pe = Pc_r[c]
            if positive_amplitude and anyp and not pos_r[c]:
                pe = -np.inf
            if pe > best:
                best, win = pe, c
        if win < 0:
            status[r] |= FLAG_FLAT
            continue
        out['power'][r] = Pc_r[win]
        out['theta'][r] = th[sl][win]
        out['theta1'][r] = th1[sl][win]
        if isdip[sl][win]:
            status[r] |= FLAG_WINNER_DIP
        if win != win_u:
            status[r] |= FLAG_K18_CHANGED
    return out


# ----------------------------------------------------------------------
# Host-side exact routing (NOT a kernel: the kernel flags, the host solves)
# ----------------------------------------------------------------------
def host_route_exact_rows(YM, MM, AC, H, ybar_rows, YY_rows, result,
                          rtol=SCAN_EXACT_RTOL, positive_amplitude=True):
    """Overwrite rows whose conditioning fell below ``rtol`` with the exact
    root-path solution (core.roots_from_YM_MM), exactly as the production
    host would after downloading the flagged coefficient rows. Modifies
    ``result`` in place; returns the flagged row indices.

    NaN cond_r (all-zero MM) is treated as flagged: ``~(cond_r >= rtol)``.
    """
    import numpy.polynomial as pol
    from ftperiodogram import core as pdg

    cond_r = result['cond_r']
    with np.errstate(invalid='ignore'):
        flagged = np.flatnonzero(~(cond_r >= rtol))
    for i in flagged:
        params, pw, bphi = pdg.roots_from_YM_MM(
            pol.Polynomial(YM[i]), pol.Polynomial(MM[i]), AC[i], H,
            float(ybar_rows[i]), float(YY_rows[i]),
            positive_amplitude=positive_amplitude)
        result['power'][i] = pw
        result['theta'][i] = np.imag(np.log(bphi)) % TWO_PI
        result['theta1'][i] = params.a
    return flagged


# ----------------------------------------------------------------------
# Full pipeline mirror (mirrors the kernels.py host driver)
# ----------------------------------------------------------------------
def full_pipeline_ref(band_arrays, templates, freqs, chunk=512,
                      n_newton=SCAN_NEWTON_STEPS, positive_amplitude=True,
                      route_exact=True, rtol=SCAN_EXACT_RTOL,
                      collect=False, Hs=None):
    """Batched multiband floating_offsets powers for K templates with
    device-sums reuse (mirror of kernels.gpu_multiband_powers).

    band_arrays : list over bands of dicts with
        t, w, u : (B, Nmax) zero-padded per-band arrays
        W       : (B,) band weight totals (stats.W[band])
        YY      : (B,) combined variance  (stats.YY_combined; same array
                  for every band)
        ybar    : (B,) global weighted mean (stats.ybar_global)
    templates : list of (cn, sn) pairs; each length H_t <= Hs. Stage-1
        sums are computed ONCE per frequency chunk at ladder depth Hs
        (default max H_t) and reused across all templates AND subtype
        H-prefixes -- the multi-template contract.
    freqs : (nfreq,)

    Returns dict with powers (K, B, nfreq), cond_r, status, theta, theta1
    (same shape), and (if collect) the stage-1 sums / stage-2 coefs of the
    LAST chunk for fixture capture.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    nfreq = len(freqs)
    B = band_arrays[0]['t'].shape[0]
    K = len(templates)
    tH = [len(np.asarray(c)) for c, s in templates]
    if Hs is None:
        Hs = max(tH)
    assert max(tH) <= Hs
    YY = np.asarray(band_arrays[0]['YY'], dtype=np.float64)
    ybar = np.asarray(band_arrays[0]['ybar'], dtype=np.float64)

    shape = (K, B, nfreq)
    res = dict(powers=np.zeros(shape), theta=np.zeros(shape),
               theta1=np.zeros(shape), cond_r=np.zeros(shape),
               status=np.zeros(shape, dtype=np.int64))
    collected = {}

    for i0 in range(0, nfreq, chunk):
        i1 = min(i0 + chunk, nfreq)
        fr = freqs[i0:i1]
        m = i1 - i0
        # ---- stage 1 once per band at ladder depth Hs (sums reuse) ----
        sums_by_band = [stage1_direct_sums_ref(bd['t'], bd['w'], bd['u'],
                                               fr, Hs)
                        for bd in band_arrays]
        for kt, (cn, sn) in enumerate(templates):
            Ht = tH[kt]
            per_band, W_rows = [], []
            for bd, sums in zip(band_arrays, sums_by_band):
                flat = {key: (val.reshape(B * m, *val.shape[2:]))
                        for key, val in sums.items()}
                per_band.append(stage2_assemble_ref(flat, cn, sn, H=Ht))
                W_rows.append(np.repeat(np.asarray(bd['W'],
                                                   dtype=np.float64), m))
            YMt, MMt, ACt = combine_bands_ref(per_band, W_rows)
            YY_rows = np.repeat(YY, m)
            r3 = stage3_scan_polish_ref(YMt, MMt, YY_rows, Ht,
                                        n_newton=n_newton,
                                        positive_amplitude=positive_amplitude)
            if route_exact:
                host_route_exact_rows(YMt, MMt, ACt, Ht,
                                      np.repeat(ybar, m), YY_rows, r3,
                                      rtol=rtol,
                                      positive_amplitude=positive_amplitude)
            res['powers'][kt, :, i0:i1] = r3['power'].reshape(B, m)
            res['theta'][kt, :, i0:i1] = r3['theta'].reshape(B, m)
            res['theta1'][kt, :, i0:i1] = r3['theta1'].reshape(B, m)
            res['cond_r'][kt, :, i0:i1] = r3['cond_r'].reshape(B, m)
            res['status'][kt, :, i0:i1] = r3['status'].reshape(B, m)
            if collect and kt == 0:
                collected = dict(sums_by_band=sums_by_band, YM=YMt, MM=MMt,
                                 AC=ACt, i0=i0, i1=i1)
    if collect:
        res['collected'] = collected
    return res


def make_band_arrays(t, y, bands, dy, H, mode='floating_offsets'):
    """Package one source's flat (t, y, bands, dy) into the padded
    band_arrays list consumed by full_pipeline_ref / the GPU wrappers,
    using the package's own _prepare_bands (single source of truth for
    weights / per-band centering). Returns (band_arrays, stats)."""
    from ftperiodogram.multiband import _prepare_bands
    band_data, stats = _prepare_bands(t, y, bands, dy, mode, None, H)
    out = []
    for band in stats.bands:
        t_k, y_k, w_k = band_data[band]
        ybar_k = np.dot(w_k, y_k)
        u_k = w_k * (y_k - ybar_k)
        out.append(dict(band=band, t=t_k, w=w_k, u=u_k,
                        W=stats.W[band], YY=stats.YY_combined,
                        ybar=stats.ybar_global))
    return out, stats


def stack_sources(per_source_band_arrays):
    """Pad + stack per-source band_arrays (from make_band_arrays) into the
    batched (B, Nmax) layout. All sources must share the band list order.
    Returns list over bands of dicts(t, w, u, N, W, YY, ybar)."""
    nband = len(per_source_band_arrays[0])
    B = len(per_source_band_arrays)
    out = []
    for kb in range(nband):
        Ns = [len(src[kb]['t']) for src in per_source_band_arrays]
        Nmax = max(Ns)
        t = np.zeros((B, Nmax))
        w = np.zeros((B, Nmax))
        u = np.zeros((B, Nmax))
        for b, src in enumerate(per_source_band_arrays):
            n = Ns[b]
            t[b, :n] = src[kb]['t']
            w[b, :n] = src[kb]['w']
            u[b, :n] = src[kb]['u']
        out.append(dict(
            band=per_source_band_arrays[0][kb]['band'],
            t=t, w=w, u=u, N=np.asarray(Ns, dtype=np.int32),
            W=np.asarray([src[kb]['W']
                          for src in per_source_band_arrays]),
            YY=np.asarray([src[kb]['YY']
                           for src in per_source_band_arrays]),
            ybar=np.asarray([src[kb]['ybar']
                             for src in per_source_band_arrays])))
    return out
