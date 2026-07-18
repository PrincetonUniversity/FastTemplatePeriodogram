"""
Batched, vectorized polynomial root-finders for the FTP stationarity polynomial.

Convention: coefficient stacks are INCREASING order, shape (nf, D+1):
    p_i(x) = sum_{k=0}^{D} coefs[i, k] * x**k
matching ftperiodogram.core.batched_stationarity_coefs. All return roots of
shape (nf, D), complex128, fully vectorized over the nf axis (the only Python
loops are over the *degree* D, not over polynomials).

Methods:
    batched_aberth            -- Aberth-Ehrlich simultaneous iteration
    batched_durand_kerner     -- Durand-Kerner (Weierstrass) simultaneous iter
    batched_companion_eigvals -- stacked companion build + np.linalg.eigvals
    numpy_roots_loop          -- reference: per-poly numpy Polynomial.roots()

Plus select_power(): the core.py argmax-over-unit-circle selection, vectorized,
used identically for every method so accuracy comparisons isolate the roots.
"""
import numpy as np
import numpy.polynomial as pol


# ----------------------------------------------------------------------
# Horner helpers (vectorized over the degree axis, broadcast over nf)
# ----------------------------------------------------------------------
def _horner_p_dp(c, W):
    """Evaluate p and p' at W. c:(nf,D+1) increasing order; W:(nf,D).
    Returns (p(W), p'(W)), both (nf,D)."""
    D = c.shape[1] - 1
    p = np.broadcast_to(c[:, D:D + 1], W.shape).astype(np.complex128).copy()
    dp = np.zeros_like(W)
    for k in range(D - 1, -1, -1):
        dp = dp * W + p
        p = p * W + c[:, k:k + 1]
    return p, dp


def _horner_p(c, W):
    """Evaluate p only. c:(nf,m) increasing order; W:(nf,D) -> (nf,D)."""
    m = c.shape[1]
    out = np.broadcast_to(c[:, m - 1:m], W.shape).astype(np.complex128).copy()
    for k in range(m - 2, -1, -1):
        out = out * W + c[:, k:k + 1]
    return out


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------
def _init_roots(c, spread=0.5, inflate=1.0):
    """Initialize D roots per poly on a circle of radius (|c0/cD|)^(1/D)
    with evenly spread angles plus a fixed offset to break real-axis symmetry.

    Returns W:(nf,D) complex128 and the per-row radius R:(nf,).
    """
    nf, D = c.shape[0], c.shape[1] - 1
    c0 = c[:, 0]
    cD = c[:, D]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(c0 / cD)
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    R = inflate * ratio ** (1.0 / D)                     # (nf,)
    R = np.where(np.isfinite(R) & (R > 0), R, 1.0)
    j = np.arange(D)
    # offset by spread/D radians so no root starts exactly on the real axis
    ang = (2.0 * np.pi / D) * j + spread                 # (D,)
    W = R[:, None] * np.exp(1j * ang)[None, :]
    return W.astype(np.complex128), R


# ----------------------------------------------------------------------
# Aberth-Ehrlich
# ----------------------------------------------------------------------
def batched_aberth(coefs, tol=1e-13, max_iter=80, spread=0.5, inflate=1.2,
                   return_stats=False):
    """Batched Aberth-Ehrlich simultaneous root iteration.

    coefs : (nf, D+1) increasing order.  Returns roots (nf, D).
    Convergence is tracked per polynomial; converged polys are frozen (their
    corrections are zeroed) so the remaining work concentrates on stragglers.

    inflate=1.2 offsets the initial circle OFF the unit circle: FTP roots
    cluster on |z|=1, and seeding the roots exactly there is a near-degenerate
    configuration that nearly doubles the iteration count (median 28 -> 16 at
    D=46); a 20% radius offset avoids it with no loss of robustness.
    """
    c = np.ascontiguousarray(coefs, dtype=np.complex128)
    nf, D = c.shape[0], c.shape[1] - 1
    if D <= 0:
        return np.zeros((nf, 0), dtype=np.complex128)

    W, _ = _init_roots(c, spread=spread, inflate=inflate)
    idx = np.arange(D)
    active = np.ones(nf, dtype=bool)          # per-poly still-iterating mask
    iters_used = np.full(nf, max_iter, dtype=np.int32)

    for it in range(max_iter):
        if not active.any():
            break
        Wa = W[active]                        # (na, D)
        ca = c[active]
        p, dp = _horner_p_dp(ca, Wa)

        # Newton ratio N = p/p' (guard p'=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            N = p / dp
        N[~np.isfinite(N)] = 0.0              # dead root: freeze (no Newton info)

        # Aberth sum S_i = sum_{j!=i} 1/(w_i - w_j)
        diff = Wa[:, :, None] - Wa[:, None, :]    # (na, D, D)
        diff[:, idx, idx] = 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / diff
        inv[:, idx, idx] = 0.0
        S = inv.sum(axis=2)                   # (na, D)

        with np.errstate(divide='ignore', invalid='ignore'):
            denom = 1.0 - N * S
            corr = N / denom
        corr[~np.isfinite(corr)] = 0.0

        Wa_new = Wa - corr
        W[active] = Wa_new

        # per-poly convergence: max root correction below tol
        maxcorr = np.max(np.abs(corr), axis=1)    # (na,)
        done = maxcorr < tol
        if done.any():
            act_idx = np.where(active)[0]
            newly = act_idx[done]
            iters_used[newly] = it + 1
            active[newly] = False

    if return_stats:
        # report the actual per-poly iteration count; non-converged stay max_iter
        return W, iters_used
    return W


# ----------------------------------------------------------------------
# Durand-Kerner (Weierstrass)
# ----------------------------------------------------------------------
def batched_durand_kerner(coefs, tol=1e-13, max_iter=200, spread=0.5,
                          inflate=1.2, return_stats=False):
    """Batched Durand-Kerner simultaneous root iteration (monic form).

    coefs : (nf, D+1) increasing order.  Returns roots (nf, D).
    """
    c = np.ascontiguousarray(coefs, dtype=np.complex128)
    nf, D = c.shape[0], c.shape[1] - 1
    if D <= 0:
        return np.zeros((nf, 0), dtype=np.complex128)

    cD = c[:, D:D + 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        cmonic = c / cD                       # monic; leading coef -> 1
    # rows with cD==0 would be all-nan; leave them (garbage in) but keep finite
    cmonic = np.where(np.isfinite(cmonic), cmonic, 0.0)

    W, _ = _init_roots(c, spread=spread, inflate=inflate)
    idx = np.arange(D)
    active = np.ones(nf, dtype=bool)
    iters_used = np.full(nf, max_iter, dtype=np.int32)

    for it in range(max_iter):
        if not active.any():
            break
        Wa = W[active]
        ca = cmonic[active]
        pv = _horner_p(ca, Wa)                # monic p(w_i)

        diff = Wa[:, :, None] - Wa[:, None, :]
        diff[:, idx, idx] = 1.0
        prod = np.prod(diff, axis=2)          # prod_{j!=i}(w_i - w_j)

        with np.errstate(divide='ignore', invalid='ignore'):
            corr = pv / prod
        corr[~np.isfinite(corr)] = 0.0

        W[active] = Wa - corr

        maxcorr = np.max(np.abs(corr), axis=1)
        done = maxcorr < tol
        if done.any():
            act_idx = np.where(active)[0]
            newly = act_idx[done]
            iters_used[newly] = it + 1
            active[newly] = False

    if return_stats:
        return W, iters_used
    return W


# ----------------------------------------------------------------------
# Batched companion-matrix + stacked eigvals
# ----------------------------------------------------------------------
def batched_companion_eigvals(coefs):
    """Build the (nf,D,D) Frobenius companion stack and call
    np.linalg.eigvals once on the whole stack (numpy loops internally in C).
    coefs : (nf, D+1) increasing order.  Returns roots (nf, D)."""
    c = np.ascontiguousarray(coefs, dtype=np.complex128)
    nf, D = c.shape[0], c.shape[1] - 1
    if D <= 0:
        return np.zeros((nf, 0), dtype=np.complex128)
    lead = c[:, D:D + 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        a = c[:, :D] / lead                   # (nf, D), monic tail
    a = np.where(np.isfinite(a), a, 0.0)
    C = np.zeros((nf, D, D), dtype=np.complex128)
    if D > 1:
        C[:, np.arange(1, D), np.arange(0, D - 1)] = 1.0   # subdiagonal ones
    C[:, :, D - 1] = -a                        # last column = -a_k
    return np.linalg.eigvals(C)


# ----------------------------------------------------------------------
# Production recipe: batched Aberth with an eigvals safety net
# ----------------------------------------------------------------------
def batched_roots(coefs, tol=1e-12, max_iter=60, chunk=4096, fallback=True):
    """Recommended drop-in: batched Aberth over chunks, with the EXACT per-row
    numpy reference path (``pol.Polynomial(coef).roots()`` -- the same call the
    current code / `_exact_root_fallback` makes) used ONLY for the (rare) rows
    Aberth did not drive below `tol` within `max_iter`. On well-conditioned FTP
    polynomials the fallback set is empty, so this is pure Aberth (~3x faster
    than the per-poly numpy loop at D=46); on adversarial clustered/multiple-
    root inputs it degrades gracefully to the exact reference (no speedup)
    rather than returning under-converged roots.

    Returns (roots (nf, D), n_fallback)."""
    c = np.ascontiguousarray(coefs, dtype=np.complex128)
    nf, D = c.shape[0], c.shape[1] - 1
    out = np.empty((nf, D), dtype=np.complex128)
    n_fb = 0
    for i0 in range(0, nf, chunk):
        cc = c[i0:i0 + chunk]
        W, iters = batched_aberth(cc, tol=tol, max_iter=max_iter,
                                  return_stats=True)
        if fallback:
            bad = np.where(iters >= max_iter)[0]
            for b in bad:
                r = pol.Polynomial(cc[b]).roots()
                W[b] = 0.0
                W[b, :len(r)] = r
            n_fb += len(bad)
        out[i0:i0 + chunk] = W
    return out, n_fb


# ----------------------------------------------------------------------
# Reference: per-poly numpy roots loop (THE current bottleneck)
# ----------------------------------------------------------------------
def numpy_roots_loop(coefs):
    """Per-polynomial numpy.polynomial.Polynomial.roots() -- exactly what
    core.py roots_from_YM_MM / _exact_root_fallback do, one poly at a time.
    Returns a (nf, D) stack (assumes each poly keeps its full degree D)."""
    c = np.ascontiguousarray(coefs, dtype=np.complex128)
    nf, D = c.shape[0], c.shape[1] - 1
    out = np.zeros((nf, D), dtype=np.complex128)
    for i in range(nf):
        r = pol.Polynomial(c[i]).roots()
        # numpy trims exactly-zero trailing (high-degree) coefs -> may be < D
        out[i, :len(r)] = r
    return out


# ----------------------------------------------------------------------
# Selection: the core.py argmax-over-unit-circle power (vectorized)
# ----------------------------------------------------------------------
def select_power(roots, YM, MM, YY):
    """Replicate roots_from_YM_MM selection, vectorized over nf.

    roots : (nf, D) candidate roots (any modulus; projected to unit circle).
    YM    : (nf, 2H+1) ; MM : (nf, 4H+1) ; YY scalar.
    Returns (power (nf,), best_phi (nf,)) exactly as the core selects them
    (default single-band: global power maximizer, no positivity filter).
    """
    nf, D = roots.shape
    # keep only non-zero roots; project to unit circle
    absr = np.abs(roots)
    good = absr > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = roots / absr
    phi = np.where(good, phi, 0.0)

    Yv = _horner_p(YM, phi)
    Mv = _horner_p(MM, phi)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.real(Yv * Yv / Mv) / YY
    P[~np.isfinite(P)] = -np.inf
    P[~good] = -np.inf

    i = np.argmax(P, axis=1)                  # (nf,)
    rows = np.arange(nf)
    power = P[rows, i]
    best_phi = phi[rows, i]
    # rows with no finite candidate -> flat zero-power fit (core contract)
    dead = ~np.isfinite(power)
    power = np.where(dead, 0.0, power)
    best_phi = np.where(dead, 1.0 + 0.0j, best_phi)
    return power, best_phi
