"""Track-A CuPy RawKernel CUDA sources + host-side launch wrappers.

Importable WITHOUT cupy (the CUDA sources are plain strings; the cupy
import is guarded) so the kernels can be authored/reviewed locally and
shipped to a GPU pod unchanged. reference.py holds the numpy bit-mirrors
of each kernel (same operation order); validate_local.py must PASS before
anything ships to a pod; pod/validate_gpu.py compares compiled-kernel
outputs against the mirror outputs saved by validate_local.py.

Design decisions baked in from AUDIT_2026-07-18 par.2/par.4 (GPU wins 7-10)
and the behavioral reference core.scan_polish_from_coefs:

stage3_scan_polish (one block per (source,freq) row):
  (i)   MANDATORY K.18 positive-amplitude filter: keep candidates with
        th1 = Re(phi^H Y / M) >= 0, best-fitting among those; if NO
        candidate is non-negative, fall back to the global power
        maximizer -- exactly core.scan_polish_from_coefs' semantics.
        (Its omission in the cupy proto biased 31% of off-peak
        frequencies -- FINDINGS corrections.)
  (ii)  Candidate compaction: polish only actual circular local maxima
        of P plus deep-|MM|-dip candidates -- never a fixed Cmax=4H
        top-k (audit win #7, 2-2.5x on the proto).
  (iii) Shared-memory staging of the YM/MM coefficient rows; k and k^2
        derivative weights applied on the fly in the fused Horner
        (audit win #8) -- no dY1/dY2/dM1/dM2 arrays exist anywhere.
  (iv)  FP64 throughout; compiled with --fmad=false by default so the
        arithmetic follows the written operation order (bit-mirror
        fidelity vs reference.py); --fmad=true is a bench-only knob.
  (v)   Outputs per-row conditioning r = min|MM|/max|MM| on the circle;
        the HOST routes rows below core._SCAN_EXACT_RTOL (0.15) /
        multiband._BATCHED_DEFER_RTOL (3e-3) to the exact path. The
        kernel flags; it does not solve them.
  (vi)  Dip-candidate seeding: deep circular local minima of |MM| below
        _SCAN_DIP_RTOL (0.5) of the row max, refined by clamped Newton
        on |MM|^2, then polished like any candidate -- core's exact
        geometry (narrow P spikes hide inside |MM| dips).
  (vii) Newton clamp +/- 2pi/M around each seed, n_newton steps
        (default 8 = core._SCAN_NEWTON_STEPS), seed fallback when the
        polished P does not improve on the seed P.
  Circle evaluation of Y/M on the M-angle grid stays OUTSIDE the kernel:
  the host wrapper computes Yv/Mv = M * cupy.fft.ifft(coefs, n=M)
  (batched cuFFT; a ZGEMM variant is a pod-tunable alternative). The
  kernel consumes the grid values plus the coefficient rows.

stage2_assemble: per-band sums -> YM/MM/AC coefficient stacks, FP64,
  mirroring core.batched_YM_MM_from_sums with documented operation order
  (see reference.stage2_assemble_ref's docstring; this is the numerically
  load-bearing kernel -- the moment-sum cancellation under concentrated
  weights, GPU_FEASIBILITY par.7 risk #1). Wk-weighted accumulate mode
  combines bands in place. Supports H <= Hs (subtype H-prefix reuse: the
  H x H sub-block of Hs-assembled sums IS the H assembly).

stage1_direct_sums: one block per source, threads own frequencies;
  direct sums C,S,YC,YS,CC,CS,SS over 2H harmonics with ONE sincos per
  (epoch,freq) and a complex-exponential (Chebyshev) recurrence for the
  harmonic ladder (audit win #10) -- never 2H transcendental calls per
  epoch-freq. Output layout matches summations.direct_summations_batched.

Multi-template: gpu_multiband_powers keeps the Stage-1 sums device-
  resident and reuses them across the K templates and across subtype
  scans (per-template H <= Hs) -- assemble+scan per template with NO
  re-upload and NO sums recompute (audit win #9).
"""
import numpy as np

try:
    import cupy as cp
except ImportError:          # local authoring machine has no GPU
    cp = None

# Status flags -- keep in sync with reference.py FLAG_* (validate asserts).
STATUS_FLAT = 1
STATUS_WINNER_DIP = 2
STATUS_CAP_MAX = 4
STATUS_CAP_DIP = 8
STATUS_K18_CHANGED = 16
STATUS_DIP_CAND = 32

SCAN_MIN_ANGLES = 128
SCAN_ANGLES_PER_H = 32
SCAN_NEWTON_STEPS = 8
SCAN_MAX_CANDIDATES_PER_H = 4
SCAN_DIP_RTOL = 0.5
SCAN_EXACT_RTOL = 0.15
BATCHED_DEFER_RTOL = 3e-3


# ----------------------------------------------------------------------
# CUDA C sources (plain strings; no external headers -- NVRTC-safe)
# ----------------------------------------------------------------------
COMMON_SRC = r"""
#ifndef FTP_MAX_H
#define FTP_MAX_H 8
#endif
#define TWO_PI 6.283185307179586

typedef double2 cplx;

__device__ __forceinline__ cplx cmk(double x, double y)
{ cplx z; z.x = x; z.y = y; return z; }
__device__ __forceinline__ cplx cadd(cplx a, cplx b)
{ return cmk(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ cplx csub(cplx a, cplx b)
{ return cmk(a.x - b.x, a.y - b.y); }
/* plain 4-mul/2-add complex product; --fmad=false keeps each product
   individually rounded, matching reference._cmul */
__device__ __forceinline__ cplx cmul(cplx a, cplx b)
{ return cmk(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
__device__ __forceinline__ cplx cscale(cplx a, double s)
{ return cmk(s * a.x, s * a.y); }
__device__ __forceinline__ cplx cconjc(cplx a)
{ return cmk(a.x, -a.y); }
/* Re(a / b) without a full complex division (reference._creal_div) */
__device__ __forceinline__ double creal_div(cplx a, cplx b)
{ return (a.x * b.x + a.y * b.y) / (b.x * b.x + b.y * b.y); }
/* |z| as sqrt(re^2 + im^2) -- NOT hypot (reference._cabs) */
__device__ __forceinline__ double cabs_(cplx a)
{ return sqrt(a.x * a.x + a.y * a.y); }
__device__ __forceinline__ cplx cpow_int(cplx z, int e)
{
    cplx r = cmk(1.0, 0.0);
    cplx b = z;
    while (e) {
        if (e & 1) r = cmul(r, b);
        b = cmul(b, b);
        e >>= 1;
    }
    return r;
}
/* finite test without header dependencies: false for NaN and +/-inf */
__device__ __forceinline__ bool finited(double x)
{ return fabs(x) <= 1.7976931348623157e308; }
__device__ __forceinline__ double neg_inf()
{ return __longlong_as_double(0xfff0000000000000LL); }

/* Horner evaluation (reference._horner_c) */
__device__ __forceinline__ cplx horner_c(const cplx* cf, int n, cplx phi)
{
    cplx a = cmk(0.0, 0.0);
    for (int j = n - 1; j >= 0; --j) a = cadd(cmul(a, phi), cf[j]);
    return a;
}
/* fused Horner of p and its k-/k^2-weighted companions; the derivative
   weights are applied ON THE FLY (design req (iii)); reference._horner012 */
__device__ __forceinline__ void horner012(const cplx* cf, int n, cplx phi,
                                          cplx* p0, cplx* p1, cplx* p2)
{
    cplx a0 = cmk(0.0, 0.0), a1 = a0, a2 = a0;
    for (int j = n - 1; j >= 0; --j) {
        const cplx c = cf[j];
        a0 = cadd(cmul(a0, phi), c);
        a1 = cadd(cmul(a1, phi), cscale(c, (double)j));
        a2 = cadd(cmul(a2, phi), cscale(c, (double)j * (double)j));
    }
    *p0 = a0; *p1 = a1; *p2 = a2;
}
"""

STAGE1_SRC = r"""
/* stage1_direct_sums: one block per source; each thread owns one
   frequency (grid-stride over the chunk). Per epoch: ONE sincos of the
   base angle ph = TWO_PI * (f * t_n), then the complex-exponential
   harmonic ladder e^{i(h+1)ph} = e^{i h ph} * e^{i ph} (Chebyshev
   recurrence). Accumulation order: epochs outer, harmonics inner
   (reference.stage1_direct_sums_ref). Covariance assembly is the
   uncentered product-to-sum form of summations.py:
       CC_jk = 0.5*(Cn + Cp) - C_j C_k   (etc.)
   Output layout matches direct_summations_batched per source:
   C/S/YC/YS (B, m, H); CC/CS/SS (B, m, H, H). */
extern "C" __global__ void stage1_direct_sums(
    const double* __restrict__ t,      /* (B, Nmax) zero-padded */
    const double* __restrict__ w,
    const double* __restrict__ u,
    const int*    __restrict__ nobs,   /* (B,) true epoch counts */
    const double* __restrict__ freqs,  /* (m,) */
    double* __restrict__ C,            /* (B, m, H) */
    double* __restrict__ S,
    double* __restrict__ YC,
    double* __restrict__ YS,
    double* __restrict__ CC,           /* (B, m, H, H) */
    double* __restrict__ CS,
    double* __restrict__ SS,
    const int B, const int Nmax, const int m, const int H)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int N = nobs[b];
    const int H2 = 2 * H;

    extern __shared__ double smem[];
    double* st = smem;
    double* sw = st + Nmax;
    double* su = sw + Nmax;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        st[n] = t[(size_t)b * Nmax + n];
        sw[n] = w[(size_t)b * Nmax + n];
        su[n] = u[(size_t)b * Nmax + n];
    }
    __syncthreads();

    double Cf[2 * FTP_MAX_H], Sf[2 * FTP_MAX_H];
    double YCa[FTP_MAX_H], YSa[FTP_MAX_H];

    for (int jf = threadIdx.x; jf < m; jf += blockDim.x) {
        const double f = freqs[jf];
        for (int h = 0; h < H2; ++h) { Cf[h] = 0.0; Sf[h] = 0.0; }
        for (int h = 0; h < H;  ++h) { YCa[h] = 0.0; YSa[h] = 0.0; }

        for (int n = 0; n < N; ++n) {
            const double tn = st[n], wn = sw[n], un = su[n];
            const double ph = TWO_PI * (f * tn);
            double s1, c1;
            sincos(ph, &s1, &c1);
            double ch = c1, sh = s1;
            for (int h = 0; h < H2; ++h) {
                if (h > 0) {
                    const double chn = ch * c1 - sh * s1;
                    sh = sh * c1 + ch * s1;   /* uses OLD ch */
                    ch = chn;
                }
                Cf[h] += wn * ch;
                Sf[h] += wn * sh;
                if (h < H) { YCa[h] += un * ch; YSa[h] += un * sh; }
            }
        }

        const size_t row = (size_t)b * m + jf;
        for (int h = 0; h < H; ++h) {
            C[row * H + h]  = Cf[h];
            S[row * H + h]  = Sf[h];
            YC[row * H + h] = YCa[h];
            YS[row * H + h] = YSa[h];
        }
        for (int j = 0; j < H; ++j) {
            for (int k = 0; k < H; ++k) {
                double cn_, sn_;
                if (j == k) { cn_ = 1.0; sn_ = 0.0; }
                else {
                    const int a = (k > j) ? (k - j) : (j - k);
                    cn_ = Cf[a - 1];
                    sn_ = (k > j) ? Sf[a - 1] : -Sf[a - 1];
                }
                const double cp = Cf[j + k + 1];
                const double sp = Sf[j + k + 1];
                const size_t o = (row * H + j) * H + k;
                /* moment-sum cancellation lives in these subtractions */
                CC[o] = 0.5 * (cn_ + cp) - Cf[j] * Cf[k];
                CS[o] = 0.5 * (sn_ + sp) - Cf[j] * Sf[k];
                SS[o] = 0.5 * (cn_ - cp) - Sf[j] * Sf[k];
            }
        }
    }
}
"""

STAGE2_SRC = r"""
/* stage2_assemble: one thread per (source,freq) row. Operation order is
   documented in reference.stage2_assemble_ref (steps 1-5); the diagonal
   sums traverse ascending i SERIALLY and the MM slots are filled in
   core's section order (all SS, then all 2*CS, then all CC -- the slots
   overlap, so order matters at the last ulp). Wk/accum implement the
   band combination YM' = sum_k W_k YM_k in band order.
   Hs = ladder depth of the sums; H <= Hs = template harmonics (subtype
   H-prefix reuse reads the H x H sub-block of the Hs x Hs covariance). */
extern "C" __global__ void stage2_assemble(
    const double* __restrict__ C,      /* (nrow, Hs) */
    const double* __restrict__ S,
    const double* __restrict__ YC,
    const double* __restrict__ YS,
    const double* __restrict__ CCs,    /* (nrow, Hs, Hs) */
    const double* __restrict__ CSs,
    const double* __restrict__ SSs,
    const double* __restrict__ alr,    /* (H,) alpha = 0.5*(cn + i sn) */
    const double* __restrict__ ali,
    const double* __restrict__ Wk,     /* (nrow,) per-row band weight */
    cplx* __restrict__ YM,             /* (nrow, 2H+1) */
    cplx* __restrict__ MM,             /* (nrow, 4H+1) */
    cplx* __restrict__ AC,             /* (nrow, H) */
    const int accum, const int nrow, const int Hs, const int H)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrow) return;
    const double wk = Wk[i];
    const size_t r1 = (size_t)i * Hs;
    const size_t r2 = (size_t)i * Hs * Hs;

    cplx alpha[FTP_MAX_H];
    for (int h = 0; h < H; ++h) alpha[h] = cmk(alr[h], ali[h]);

    /* ---- step 1: aYC, YM row --------------------------------------- */
    cplx aYC[FTP_MAX_H];
    for (int h = 0; h < H; ++h)
        aYC[h] = cmul(alpha[h], cmk(YC[r1 + h], -YS[r1 + h]));
    {
        const size_t ob = (size_t)i * (2 * H + 1);
        for (int h = 0; h < 2 * H + 1; ++h) {
            cplx v;
            if (h < H)       v = cconjc(aYC[H - 1 - h]);
            else if (h == H) v = cmk(0.0, 0.0);
            else             v = aYC[h - H - 1];
            v = cscale(v, wk);
            YM[ob + h] = accum ? cadd(YM[ob + h], v) : v;
        }
    }

    /* ---- steps 2-4: MM row ------------------------------------------
       X_jk  = (CC - SS) - i(CS_jk + CS_kj); CCk = X * (a_j a_k)
       Yc_jk = (CC + SS) + i(CS_jk - CS_kj); CSk = Yc * (a_j conj(a_k))
       SSk_jk = conj(CCk_jk), conjugated per element. */
    cplx mm[4 * FTP_MAX_H + 1];
    for (int d = 0; d < 4 * H + 1; ++d) mm[d] = cmk(0.0, 0.0);

    /* section 1: SS diagonals -> slots [0, 2H-2] */
    for (int d = 0; d < 2 * H - 1; ++d) {
        const int o = d - (H - 1);
        const int i0 = (o < 0) ? -o : 0;
        const int i1 = (o < 0) ? (H - 1) : (H - 1 - o);
        cplx acc = cmk(0.0, 0.0);
        for (int ii = i0; ii <= i1; ++ii) {
            const int j = ii, k = H - 1 - ii - o;
            const size_t jk = r2 + (size_t)j * Hs + k;
            const size_t kj = r2 + (size_t)k * Hs + j;
            const cplx X = cmk(CCs[jk] - SSs[jk], -(CSs[jk] + CSs[kj]));
            const cplx axa = cmul(alpha[j], alpha[k]);
            acc = cadd(acc, cconjc(cmul(X, axa)));
        }
        mm[d] = cadd(mm[d], acc);
    }
    /* section 2: 2 * CS diagonals -> slots [H+1, 3H-1] */
    for (int d = 0; d < 2 * H - 1; ++d) {
        const int o = d - (H - 1);
        const int i0 = (o < 0) ? -o : 0;
        const int i1 = (o < 0) ? (H - 1) : (H - 1 - o);
        cplx acc = cmk(0.0, 0.0);
        for (int ii = i0; ii <= i1; ++ii) {
            const int j = ii + o, k = ii;
            const size_t jk = r2 + (size_t)j * Hs + k;
            const size_t kj = r2 + (size_t)k * Hs + j;
            const cplx Yc = cmk(CCs[jk] + SSs[jk], CSs[jk] - CSs[kj]);
            const cplx axc = cmul(alpha[j], cconjc(alpha[k]));
            acc = cadd(acc, cmul(Yc, axc));
        }
        mm[d + H + 1] = cadd(mm[d + H + 1], cscale(acc, 2.0));
    }
    /* section 3: CC diagonals -> slots [2H+2, 4H] */
    for (int d = 0; d < 2 * H - 1; ++d) {
        const int o = d - (H - 1);
        const int i0 = (o < 0) ? -o : 0;
        const int i1 = (o < 0) ? (H - 1) : (H - 1 - o);
        cplx acc = cmk(0.0, 0.0);
        for (int ii = i0; ii <= i1; ++ii) {
            const int j = H - 1 - ii, k = ii + o;
            const size_t jk = r2 + (size_t)j * Hs + k;
            const size_t kj = r2 + (size_t)k * Hs + j;
            const cplx X = cmk(CCs[jk] - SSs[jk], -(CSs[jk] + CSs[kj]));
            const cplx axa = cmul(alpha[j], alpha[k]);
            acc = cadd(acc, cmul(X, axa));
        }
        mm[d + 2 * H + 2] = cadd(mm[d + 2 * H + 2], acc);
    }
    {
        const size_t ob = (size_t)i * (4 * H + 1);
        for (int d = 0; d < 4 * H + 1; ++d) {
            const cplx v = cscale(mm[d], wk);
            MM[ob + d] = accum ? cadd(MM[ob + d], v) : v;
        }
    }

    /* ---- step 5: AC row --------------------------------------------- */
    {
        const size_t ob = (size_t)i * H;
        for (int h = 0; h < H; ++h) {
            const cplx v = cscale(
                cmul(alpha[h], cmk(C[r1 + h], -S[r1 + h])), wk);
            AC[ob + h] = accum ? cadd(AC[ob + h], v) : v;
        }
    }
}
"""

STAGE3_SRC = r"""
#define ST_FLAT        1
#define ST_WINNER_DIP  2
#define ST_CAP_MAX     4
#define ST_CAP_DIP     8
#define ST_K18_CHANGED 16
#define ST_DIP_CAND    32

/* stage3_scan_polish: one block per (source,freq) row.
   Phases:
     A (parallel): stage YM/MM rows in shared memory; compute the grid
       P = Re(Y^2/M)/YY (non-finite -> -inf) and |M| for all M angles.
     B (thread 0, serial -- deterministic): min/max |M| -> cond_r output
       (HOST routes rows below the exact/deferral thresholds; the kernel
       flags, it does not solve); circular-local-max candidates of P and
       deep circular-local-min dip candidates of |M| (< dip_rtol * max),
       each capped at kcap (largest-P / smallest-|M| with ties -> lower
       angle), candidate order = maxima ascending angle then dips
       ascending angle (core's row-major ordering).
     C (parallel over candidates): dip refine = clamped Newton on |MM|^2
       from the dip grid angle; the REFINED angle becomes the polish seed
       and clamp center (core lines 628-663).
     D (parallel over candidates): n_newton clamped Newton steps on
       dP/dtheta with analytic first/second derivatives (fused Horner,
       on-the-fly k/k^2 weights); evaluate polished P; fall back to the
       seed when not improved (core lines 680-713).
     E (thread 0, serial): th1 = Re(phi^H Y/M); K.18 positive-amplitude
       filter with fall-back-to-global-max semantics (core lines 715-737);
       first-wins argmax; flat fallback when no usable candidate. */
extern "C" __global__ void stage3_scan_polish(
    const cplx* __restrict__ YMc,      /* (nrow, 2H+1) */
    const cplx* __restrict__ MMc,      /* (nrow, 4H+1) */
    const cplx* __restrict__ Yv,       /* (nrow, M) circle values */
    const cplx* __restrict__ Mv,       /* (nrow, M) */
    const double* __restrict__ YYr,    /* (nrow,) */
    double* __restrict__ power,
    double* __restrict__ theta_best,
    double* __restrict__ theta1,
    double* __restrict__ cond_r,
    int* __restrict__ status,
    const int nrow, const int H, const int M,
    const int n_newton, const int kcap, const int pos_amp,
    const double dip_rtol)
{
    const int row = blockIdx.x;
    if (row >= nrow) return;
    const int nym = 2 * H + 1;
    const int nmm = 4 * H + 1;
    const int ncmax = 2 * kcap;
    const double NEG_INF = neg_inf();

    extern __shared__ double smem[];
    double* p = smem;
    cplx* s_ym  = (cplx*)p;  p += 2 * nym;
    cplx* s_mm  = (cplx*)p;  p += 2 * nmm;
    cplx* c_Yfb = (cplx*)p;  p += 2 * ncmax;
    cplx* c_Mfb = (cplx*)p;  p += 2 * ncmax;
    double* s_Pg   = p;  p += M;
    double* s_absM = p;  p += M;
    double* c_th0  = p;  p += ncmax;
    double* c_P0   = p;  p += ncmax;
    double* c_th   = p;  p += ncmax;
    double* c_Pc   = p;  p += ncmax;
    double* c_th1  = p;  p += ncmax;
    int* s_int  = (int*)p;
    int* s_keep = s_int;               /* M */
    int* c_dip  = s_int + M;           /* ncmax */
    int* s_ctl  = s_int + M + ncmax;   /* [0]=ncand [1]=status */

    const double YY = YYr[row];
    const double halfw = TWO_PI / (double)M;

    /* ---- phase A ---------------------------------------------------- */
    for (int j = threadIdx.x; j < nym; j += blockDim.x)
        s_ym[j] = YMc[(size_t)row * nym + j];
    for (int j = threadIdx.x; j < nmm; j += blockDim.x)
        s_mm[j] = MMc[(size_t)row * nmm + j];
    for (int mi = threadIdx.x; mi < M; mi += blockDim.x) {
        const cplx Y  = Yv[(size_t)row * M + mi];
        const cplx Mm = Mv[(size_t)row * M + mi];
        double pg = creal_div(cmul(Y, Y), Mm) / YY;
        if (!finited(pg)) pg = NEG_INF;
        s_Pg[mi] = pg;
        s_absM[mi] = cabs_(Mm);
    }
    __syncthreads();

    /* ---- phase B (thread 0, serial) --------------------------------- */
    if (threadIdx.x == 0) {
        double mn = s_absM[0], mx = s_absM[0];
        for (int mi = 1; mi < M; ++mi) {
            mn = fmin(mn, s_absM[mi]);
            mx = fmax(mx, s_absM[mi]);
        }
        cond_r[row] = mn / mx;
        int st = 0;

        /* grid maxima of P */
        int nmax = 0;
        for (int mi = 0; mi < M; ++mi) {
            const double pg = s_Pg[mi];
            const int prv = (mi == 0) ? (M - 1) : mi - 1;
            const int nxt = (mi == M - 1) ? 0 : mi + 1;
            const int is = (pg >= s_Pg[prv]) && (pg >= s_Pg[nxt]) &&
                           (pg > NEG_INF);
            s_keep[mi] = is;
            nmax += is;
        }
        if (nmax > kcap) {   /* degenerate plateaus: keep kcap largest P */
            st |= ST_CAP_MAX;
            for (int sel = 0; sel < kcap; ++sel) {
                int bi = -1; double bv = 0.0;
                for (int mi = 0; mi < M; ++mi)
                    if (s_keep[mi] == 1 && (bi < 0 || s_Pg[mi] > bv)) {
                        bi = mi; bv = s_Pg[mi];
                    }
                if (bi >= 0) s_keep[bi] = 2;
            }
            for (int mi = 0; mi < M; ++mi)
                s_keep[mi] = (s_keep[mi] == 2);
        }
        int nc = 0;
        for (int mi = 0; mi < M; ++mi) if (s_keep[mi]) {
            c_th0[nc] = (TWO_PI / (double)M) * (double)mi;
            c_P0[nc] = s_Pg[mi];
            c_Yfb[nc] = Yv[(size_t)row * M + mi];
            c_Mfb[nc] = Mv[(size_t)row * M + mi];
            c_dip[nc] = 0;
            ++nc;
        }

        /* deep dip minima of |M| */
        const double thr = dip_rtol * mx;
        int ndip = 0;
        for (int mi = 0; mi < M; ++mi) {
            const double a = s_absM[mi];
            const int prv = (mi == 0) ? (M - 1) : mi - 1;
            const int nxt = (mi == M - 1) ? 0 : mi + 1;
            const int is = (a <= s_absM[prv]) && (a <= s_absM[nxt]) &&
                           (a < thr);
            s_keep[mi] = is;
            ndip += is;
        }
        if (ndip > 0) st |= ST_DIP_CAND;
        if (ndip > kcap) {   /* keep kcap smallest |M| */
            st |= ST_CAP_DIP;
            for (int sel = 0; sel < kcap; ++sel) {
                int bi = -1; double bv = 0.0;
                for (int mi = 0; mi < M; ++mi)
                    if (s_keep[mi] == 1 && (bi < 0 || s_absM[mi] < bv)) {
                        bi = mi; bv = s_absM[mi];
                    }
                if (bi >= 0) s_keep[bi] = 2;
            }
            for (int mi = 0; mi < M; ++mi)
                s_keep[mi] = (s_keep[mi] == 2);
        }
        for (int mi = 0; mi < M; ++mi) if (s_keep[mi]) {
            c_th0[nc] = (TWO_PI / (double)M) * (double)mi;
            c_dip[nc] = 1;
            ++nc;
        }
        s_ctl[0] = nc;
        s_ctl[1] = st;
    }
    __syncthreads();

    const int nc = s_ctl[0];

    /* ---- phase C: dip refinement (parallel over candidates) --------- */
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        if (!c_dip[c]) continue;
        const double th0 = c_th0[c];
        double th = th0;
        for (int it = 0; it < n_newton; ++it) {
            double si, co;
            sincos(th, &si, &co);
            const cplx phi = cmk(co, si);
            cplx Mm, M1, M2;
            horner012(s_mm, nmm, phi, &Mm, &M1, &M2);
            const double q1 = 2.0 * (M1.x * Mm.y - M1.y * Mm.x);
            const double q2 = 2.0 * ((M1.x * M1.x + M1.y * M1.y) -
                                     (Mm.x * M2.x + Mm.y * M2.y));
            double stp = q1 / q2;
            if (!finited(stp)) stp = 0.0;
            th = fmin(fmax(th - stp, th0 - halfw), th0 + halfw);
        }
        double si, co;
        sincos(th, &si, &co);
        const cplx phi = cmk(co, si);
        const cplx Yd = horner_c(s_ym, nym, phi);
        const cplx Md = horner_c(s_mm, nmm, phi);
        double Pd = creal_div(cmul(Yd, Yd), Md) / YY;
        if (!finited(Pd)) Pd = NEG_INF;
        c_th0[c] = th;      /* refined angle = polish seed + clamp center */
        c_P0[c] = Pd;
        c_Yfb[c] = Yd;
        c_Mfb[c] = Md;
    }
    __syncthreads();

    /* ---- phase D: Newton polish + seed fallback --------------------- */
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        const double th0 = c_th0[c];
        double th = th0;
        for (int it = 0; it < n_newton; ++it) {
            double si, co;
            sincos(th, &si, &co);
            const cplx phi = cmk(co, si);
            cplx Y, Y1, Y2, Mm, M1, M2;
            horner012(s_ym, nym, phi, &Y, &Y1, &Y2);
            horner012(s_mm, nmm, phi, &Mm, &M1, &M2);
            const cplx Bc = csub(cmul(cmul(cscale(Y, 2.0), Y1), Mm),
                                 cmul(cmul(Y, Y), M1));
            const cplx Mm2 = cmul(Mm, Mm);
            const cplx Mm3 = cmul(Mm2, Mm);
            const cplx iB = cmk(-Bc.y, Bc.x);
            const double dP = creal_div(iB, Mm2);
            const cplx num = cadd(
                cmul(cscale(cadd(cmul(Y1, Y1), cmul(Y, Y2)), -2.0), Mm),
                cmul(cmul(Y, Y), M2));
            const double d2P = creal_div(num, Mm2) +
                               creal_div(cmul(cscale(Bc, 2.0), M1), Mm3);
            double stp = dP / d2P;
            if (!finited(stp)) stp = 0.0;
            th = fmin(fmax(th - stp, th0 - halfw), th0 + halfw);
        }
        double si, co;
        sincos(th, &si, &co);
        cplx phi = cmk(co, si);
        cplx Yp = horner_c(s_ym, nym, phi);
        cplx Mp = horner_c(s_mm, nmm, phi);
        double Pp = creal_div(cmul(Yp, Yp), Mp) / YY;
        const bool seed = !finited(Pp) || (Pp < c_P0[c]);
        if (seed) {
            th = th0;
            double s0, c0;
            sincos(th0, &s0, &c0);
            phi = cmk(c0, s0);
            Yp = c_Yfb[c];
            Mp = c_Mfb[c];
            Pp = c_P0[c];
        }
        c_th[c] = th;
        c_Pc[c] = Pp;
        c_th1[c] = creal_div(cmul(cpow_int(phi, H), Yp), Mp);
    }
    __syncthreads();

    /* ---- phase E: K.18 filter + first-wins argmax (thread 0) -------- */
    if (threadIdx.x == 0) {
        int st = s_ctl[1];
        int win_u = -1;
        double best_u = NEG_INF;
        for (int c = 0; c < nc; ++c)
            if (c_Pc[c] > best_u) { best_u = c_Pc[c]; win_u = c; }
        bool anyp = false;
        if (pos_amp)
            for (int c = 0; c < nc; ++c)
                if (c_th1[c] >= 0.0) { anyp = true; break; }
        int win = -1;
        double best = NEG_INF;
        for (int c = 0; c < nc; ++c) {
            double pe = c_Pc[c];
            if (pos_amp && anyp && !(c_th1[c] >= 0.0)) pe = NEG_INF;
            if (pe > best) { best = pe; win = c; }
        }
        if (win < 0) {
            power[row] = 0.0;
            theta_best[row] = 0.0;
            theta1[row] = 0.0;
            st |= ST_FLAT;
        } else {
            power[row] = c_Pc[win];
            theta_best[row] = c_th[win];
            theta1[row] = c_th1[win];
            if (c_dip[win]) st |= ST_WINNER_DIP;
            if (win != win_u) st |= ST_K18_CHANGED;
        }
        status[row] = st;
    }
}
"""


def full_source():
    """Concatenated CUDA source for all three kernels."""
    return COMMON_SRC + STAGE1_SRC + STAGE2_SRC + STAGE3_SRC


# ----------------------------------------------------------------------
# Host-side wrappers (require cupy)
# ----------------------------------------------------------------------
_module_cache = {}


def _require_cp():
    if cp is None:
        raise RuntimeError("cupy is not installed -- GPU wrappers are "
                           "pod-only. The CUDA sources and reference "
                           "mirrors are importable without it.")


def get_module(max_h=8, fmad=False):
    """Compile (and cache) the RawModule for a given FTP_MAX_H.

    fmad=False (default) disables FMA contraction so the arithmetic
    follows the written operation order (bit-mirror fidelity vs
    reference.py). fmad=True is a throughput knob for bench_gpu only --
    never for validation.
    """
    _require_cp()
    key = (int(max_h), bool(fmad))
    if key not in _module_cache:
        opts = ['-DFTP_MAX_H={0}'.format(int(max_h))]
        if not fmad:
            opts.append('--fmad=false')
        _module_cache[key] = cp.RawModule(code=full_source(),
                                          options=tuple(opts))
    return _module_cache[key]


def scan_n_angles(H, n_angles=None):
    floor = max(SCAN_MIN_ANGLES, SCAN_ANGLES_PER_H * H)
    return floor if n_angles is None else max(int(n_angles), floor)


class GpuBandData(object):
    """Device-resident padded per-band epoch data for a batch of sources.

    Upload ONCE; reused across every frequency chunk, template, and
    subtype scan (multi-template contract).
    """

    def __init__(self, band_dict):
        _require_cp()
        self.band = band_dict.get('band')
        self.t = cp.asarray(band_dict['t'], dtype=cp.float64)
        self.w = cp.asarray(band_dict['w'], dtype=cp.float64)
        self.u = cp.asarray(band_dict['u'], dtype=cp.float64)
        self.N = cp.asarray(band_dict['N'], dtype=cp.int32)
        self.W = cp.asarray(band_dict['W'], dtype=cp.float64)   # (B,)
        self.B, self.Nmax = self.t.shape


def gpu_stage1(band, freqs_dev, H, block=128, module=None):
    """Run stage1_direct_sums for one band over a device freqs chunk.

    Returns dict of device arrays C/S/YC/YS (B, m, H), CC/CS/SS
    (B, m, H, H).
    """
    _require_cp()
    if module is None:
        module = get_module(max_h=max(H, 8))
    kern = module.get_function('stage1_direct_sums')
    B, Nmax = band.B, band.Nmax
    m = int(freqs_dev.shape[0])
    C = cp.empty((B, m, H))
    S = cp.empty((B, m, H))
    YC = cp.empty((B, m, H))
    YS = cp.empty((B, m, H))
    CC = cp.empty((B, m, H, H))
    CS = cp.empty((B, m, H, H))
    SS = cp.empty((B, m, H, H))
    smem = 3 * Nmax * 8
    kern((B,), (block,),
         (band.t, band.w, band.u, band.N, freqs_dev,
          C, S, YC, YS, CC, CS, SS,
          np.int32(B), np.int32(Nmax), np.int32(m), np.int32(H)),
         shared_mem=smem)
    return dict(C=C, S=S, YC=YC, YS=YS, CC=CC, CS=CS, SS=SS)


def gpu_stage2(sums, cn, sn, Wk_rows, accum, out=None, H=None,
               block=128, module=None):
    """Run stage2_assemble on one band's device sums.

    sums: dict from gpu_stage1 at ladder depth Hs (arrays (B, m, Hs...)).
    cn/sn: host template coefficients, length H <= Hs.
    Wk_rows: (nrow,) device per-row band weights.
    accum: False for the first band (write), True to accumulate.
    out: (YM, MM, AC) device arrays to accumulate into (required when
    accum=True). Returns (YM, MM, AC).
    """
    _require_cp()
    cn = np.asarray(cn, dtype=np.float64)
    sn = np.asarray(sn, dtype=np.float64)
    if H is None:
        H = len(cn)
    Hs = int(sums['C'].shape[-1])
    assert H <= Hs
    if module is None:
        module = get_module(max_h=max(Hs, 8))
    kern = module.get_function('stage2_assemble')
    lead = sums['C'].shape[:-1]
    nrow = int(np.prod(lead))
    if out is None:
        YM = cp.empty((nrow, 2 * H + 1), dtype=cp.complex128)
        MM = cp.empty((nrow, 4 * H + 1), dtype=cp.complex128)
        AC = cp.empty((nrow, H), dtype=cp.complex128)
    else:
        YM, MM, AC = out
    alr = cp.asarray(0.5 * cn)
    ali = cp.asarray(0.5 * sn)
    grid = (nrow + block - 1) // block
    kern((grid,), (block,),
         (sums['C'], sums['S'], sums['YC'], sums['YS'],
          sums['CC'], sums['CS'], sums['SS'],
          alr, ali, Wk_rows, YM, MM, AC,
          np.int32(1 if accum else 0), np.int32(nrow),
          np.int32(Hs), np.int32(H)))
    return YM, MM, AC


def gpu_stage3(YM, MM, YY_rows, H, n_angles=None,
               n_newton=SCAN_NEWTON_STEPS, positive_amplitude=True,
               kcap=None, dip_rtol=SCAN_DIP_RTOL, block=128, module=None):
    """Run stage3_scan_polish on device coefficient stacks.

    Circle evaluation happens HERE (host wrapper, batched cuFFT):
    Yv/Mv = M * cupy.fft.ifft(coefs, n=M) -- outside the kernel by design.
    Returns dict of device arrays power/theta/theta1/cond_r/status.
    """
    _require_cp()
    if module is None:
        module = get_module(max_h=max(H, 8))
    kern = module.get_function('stage3_scan_polish')
    if kcap is None:
        kcap = SCAN_MAX_CANDIDATES_PER_H * H
    M = scan_n_angles(H, n_angles)
    nrow = int(YM.shape[0])
    nym, nmm = 2 * H + 1, 4 * H + 1
    ncmax = 2 * kcap

    Yv = cp.ascontiguousarray(M * cp.fft.ifft(YM, n=M, axis=1))
    Mv = cp.ascontiguousarray(M * cp.fft.ifft(MM, n=M, axis=1))

    power = cp.empty(nrow)
    theta = cp.empty(nrow)
    th1 = cp.empty(nrow)
    cond_r = cp.empty(nrow)
    status = cp.empty(nrow, dtype=cp.int32)

    # shared layout (see STAGE3_SRC): cplx arrays first (16B alignment),
    # then doubles, then ints; +8 ints of slack for the control words
    smem = (2 * nym + 2 * nmm + 4 * ncmax) * 8        # cplx as double pairs
    smem += (2 * M + 5 * ncmax) * 8                   # doubles
    smem += (M + ncmax + 8) * 4                       # ints
    kern((nrow,), (block,),
         (YM, MM, Yv, Mv, YY_rows,
          power, theta, th1, cond_r, status,
          np.int32(nrow), np.int32(H), np.int32(M),
          np.int32(n_newton), np.int32(kcap),
          np.int32(1 if positive_amplitude else 0),
          np.float64(dip_rtol)),
         shared_mem=smem)
    return dict(power=power, theta=theta, theta1=th1,
                cond_r=cond_r, status=status)


def gpu_multiband_powers(band_data_list, templates, YY, freqs, chunk=1024,
                         n_newton=SCAN_NEWTON_STEPS,
                         positive_amplitude=True, Hs=None, reuse_sums=True,
                         fmad=False, block=128):
    """Full batched multiband floating_offsets detection scan on the GPU
    for K templates over B sources -- powers only (host reconstructs
    parameters at peaks via the CPU per-frequency path).

    band_data_list: list of GpuBandData (or dicts, uploaded here).
    templates: list of (cn, sn); per-template H_t <= Hs. Stage-1 sums are
    computed once per chunk at ladder depth Hs and REUSED across all
    templates and subtype H-prefixes (reuse_sums=False recomputes them
    per template -- bench comparison only).
    YY: (B,) host or device combined variances.

    Returns dict of device arrays powers/theta/theta1/cond_r/status of
    shape (K, B, nfreq). Rows with cond_r below SCAN_EXACT_RTOL /
    BATCHED_DEFER_RTOL must be routed by the caller to the CPU exact /
    per-frequency reference paths (reference.host_route_exact_rows) --
    the kernel only flags them.
    """
    _require_cp()
    bands = [b if isinstance(b, GpuBandData) else GpuBandData(b)
             for b in band_data_list]
    tH = [len(np.asarray(c)) for c, s in templates]
    if Hs is None:
        Hs = max(tH)
    module = get_module(max_h=max(Hs, 8), fmad=fmad)
    freqs = np.asarray(freqs, dtype=np.float64)
    nfreq = len(freqs)
    B = bands[0].B
    K = len(templates)
    YY = cp.asarray(YY, dtype=cp.float64)

    out = {k: cp.zeros((K, B, nfreq),
                       dtype=(cp.int32 if k == 'status' else cp.float64))
           for k in ('powers', 'theta', 'theta1', 'cond_r', 'status')}

    freqs_dev = cp.asarray(freqs)
    for i0 in range(0, nfreq, chunk):
        i1 = min(i0 + chunk, nfreq)
        m = i1 - i0
        fr = freqs_dev[i0:i1]
        YY_rows = cp.repeat(YY, m)
        W_rows = [cp.repeat(b.W, m) for b in bands]
        sums_shared = ([gpu_stage1(b, fr, Hs, block=block, module=module)
                        for b in bands] if reuse_sums else None)
        for kt, (cn, sn) in enumerate(templates):
            Ht = tH[kt]
            sums_list = (sums_shared if reuse_sums else
                         [gpu_stage1(b, fr, Hs, block=block, module=module)
                          for b in bands])
            coefs = None
            for kb, sums in enumerate(sums_list):
                coefs = gpu_stage2(sums, cn, sn, W_rows[kb],
                                   accum=(kb > 0),
                                   out=coefs, H=Ht, block=block,
                                   module=module)
            YMt, MMt, ACt = coefs
            r3 = gpu_stage3(YMt, MMt, YY_rows, Ht, n_newton=n_newton,
                            positive_amplitude=positive_amplitude,
                            block=block, module=module)
            out['powers'][kt, :, i0:i1] = r3['power'].reshape(B, m)
            out['theta'][kt, :, i0:i1] = r3['theta'].reshape(B, m)
            out['theta1'][kt, :, i0:i1] = r3['theta1'].reshape(B, m)
            out['cond_r'][kt, :, i0:i1] = r3['cond_r'].reshape(B, m)
            out['status'][kt, :, i0:i1] = r3['status'].reshape(B, m)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="dump CUDA source (for "
                                 "offline syntax checks)")
    ap.add_argument('--dump', action='store_true')
    a = ap.parse_args()
    if a.dump:
        print(full_source())
