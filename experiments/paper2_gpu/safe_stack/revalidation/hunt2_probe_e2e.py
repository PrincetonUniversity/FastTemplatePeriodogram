"""Hunt2 E2E A/B: stock vs reduceat-assembly _combine_stacked_shared_amp.

Monkeypatches mb._combine_stacked_shared_amp with the fused prep+segment-sum
version (prep hoisted per band_chunks object across the K templates), then
interleaved median-of-5 timing on the production shapes:
  mb_catalog_h4 / h2 / h8 (K=4 griz, nfreq=8000), mb_batched_h4 (K=1).
Also: power parity stats and the chunk-size bitwise-invariance pin.
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/Users/johnhoffman/Documents/fast_template_periodogram/'
            'FastTemplatePeriodogram')
sys.path.insert(0, str(REPO / 'experiments' / 'paper2_gpu' / 'safe_stack'))

import fixtures  # noqa: E402
import ftperiodogram.multiband as mb  # noqa: E402
from ftperiodogram.multiband import FastMultibandTemplatePeriodogram  # noqa: E402

ORIG = mb._combine_stacked_shared_amp

# ---------------------------------------------------------------- fast path
_PERM = {}          # (H, nb) -> (perm_cc, st_cc, perm_cs, st_cs)
_PREP = [None, None]   # [band_chunks_obj, prep dict]
_WTS = {}           # id(template_dict) -> (ref, wcc, wcs, alphas)


def _diag_perm(H, nb):
    key = (H, nb)
    if key not in _PERM:
        r = np.arange(H)
        rr, cc = np.meshgrid(r, r, indexing='ij')
        dsum = np.concatenate([(rr + cc).ravel()] * nb)
        ddif = np.concatenate([(rr - cc + H - 1).ravel()] * nb)
        perm_cc = np.argsort(dsum, kind='stable')
        perm_cs = np.argsort(ddif, kind='stable')
        ncols = 2 * H - 1
        st_cc = np.searchsorted(dsum[perm_cc], np.arange(ncols))
        st_cs = np.searchsorted(ddif[perm_cs], np.arange(ncols))
        _PERM[key] = (perm_cc, st_cc, perm_cs, st_cs)
    return _PERM[key]


def _prep(band_chunks, bands, H):
    if _PREP[0] is band_chunks:
        return _PREP[1]
    nb = len(bands)
    perm_cc, st_cc, perm_cs, st_cs = _diag_perm(H, nb)
    Pcc_l, Pcs_l, percol = [], [], {}
    for b in bands:
        s = band_chunks[b]
        nf = s.CC.shape[0]
        CCf = s.CC.reshape(nf, H * H)
        SSf = s.SS.reshape(nf, H * H)
        CSf = s.CS.reshape(nf, H * H)
        CSTf = np.ascontiguousarray(
            np.swapaxes(s.CS, 1, 2)).reshape(nf, H * H)
        Pcc_l.append((CCf - SSf) - 1j * (CSf + CSTf))
        Pcs_l.append((CCf + SSf) + 1j * (CSf - CSTf))
        percol[b] = (s.YC - 1j * s.YS, s.C - 1j * s.S)
    Pa = np.concatenate(Pcc_l, axis=1)
    Ps = np.concatenate(Pcs_l, axis=1)
    out = (np.ascontiguousarray(Pa[:, perm_cc]),
           np.ascontiguousarray(Ps[:, perm_cs]), percol)
    _PREP[0] = band_chunks
    _PREP[1] = out
    return out


def _weights_for(template_dict, bands, W, H, nb):
    key = id(template_dict)
    hit = _WTS.get(key)
    if hit is not None and hit[0] is template_dict:
        return hit[1], hit[2], hit[3]
    perm_cc, _, perm_cs, _ = _diag_perm(H, nb)
    wcc_l, wcs_l, alphas = [], [], {}
    for b in bands:
        tm = template_dict[b]
        a = 0.5 * (np.asarray(tm.c_n) + 1j * np.asarray(tm.s_n))
        alphas[b] = a
        wk = W[b]
        wcc_l.append(wk * np.outer(a, a).ravel())
        wcs_l.append(wk * np.outer(a, np.conj(a)).ravel())
    wcc = np.concatenate(wcc_l)[perm_cc]
    wcs = np.concatenate(wcs_l)[perm_cs]
    _WTS[key] = (template_dict, wcc, wcs, alphas)
    return wcc, wcs, alphas


def fast_combine(template_dict, transforms, stats, i0, i1, freqs, mode,
                 band_chunks=None):
    if mode != 'floating_offsets' or band_chunks is None:
        return ORIG(template_dict, transforms, stats, i0, i1, freqs, mode,
                    band_chunks=band_chunks)
    bands, H = stats.bands, stats.H
    nb = len(bands)
    _, st_cc, _, st_cs = _diag_perm(H, nb)
    Pa_g, Ps_g, percol = _prep(band_chunks, bands, H)
    wcc, wcs, alphas = _weights_for(template_dict, bands, stats.W, H, nb)

    CCd = np.add.reduceat(Pa_g * wcc, st_cc, axis=1)
    CSd = np.add.reduceat(Ps_g * wcs, st_cs, axis=1)
    SSd = np.conj(CCd)[:, ::-1]
    nf = CCd.shape[0]
    MM = np.zeros((nf, 4 * H + 1), dtype=np.complex128)
    MM[:, :2 * H - 1] += SSd
    MM[:, H + 1:3 * H] += 2.0 * CSd
    MM[:, 2 * H + 2:] += CCd

    aYC = None
    AC = None
    AC_by_band = {}
    for b in bands:
        yc, cs = percol[b]
        a, wk = alphas[b], stats.W[b]
        AC_k = a * cs
        AC_by_band[b] = AC_k
        t1 = (wk * a) * yc
        aYC = t1 if aYC is None else aYC + t1
        t2 = wk * AC_k
        AC = t2 if AC is None else AC + t2
    YM = np.empty((nf, 2 * H + 1), dtype=np.complex128)
    YM[:, :H] = np.conj(aYC)[:, ::-1]
    YM[:, H] = 0.0
    YM[:, H + 1:] = aYC
    return YM, MM, AC, AC_by_band


# ---------------------------------------------------------------- harness
def bench_case(label, ftp, freqs, reps=5):
    def run():
        return ftp.power(freqs, save_best_model=False, fast=True)

    # parity first (also warms caches)
    mb._combine_stacked_shared_amp = ORIG
    p_stock = run()
    mb._combine_stacked_shared_amp = fast_combine
    p_fast = run()
    d = np.abs(p_stock - p_fast)
    sc = np.max(np.abs(p_stock))
    n13 = int(np.sum(d > 1e-13 * max(sc, 1.0)))

    tA, tB = [], []
    for _ in range(reps):
        mb._combine_stacked_shared_amp = ORIG
        t0 = time.perf_counter(); run(); tA.append(time.perf_counter() - t0)
        mb._combine_stacked_shared_amp = fast_combine
        t0 = time.perf_counter(); run(); tB.append(time.perf_counter() - t0)
    mb._combine_stacked_shared_amp = ORIG
    mA, mB = np.median(tA), np.median(tB)
    print('%-16s stock %7.4fs  fast %7.4fs  speedup %.3fx  '
          'maxAbsDiff %.2e (rel %.2e)  rows>1e-13rel: %d'
          % (label, mA, mB, mA / mB, d.max(), d.max() / sc, n13))
    return mA, mB


def main():
    nfreq = 8000
    for H, sub, K in ((4, 'RRab', 4), (2, 'RRc', 4), (8, 'RRab', 4),
                      (4, 'RRab', 1)):
        fx = fixtures.mb_fixture(sub, H=H, nfreq=nfreq)
        tset = fx['vocab'] if K == 4 else fx['vocab'][0]
        ftp = FastMultibandTemplatePeriodogram(templates=tset,
                                               mode='floating_offsets')
        ftp.fit(fx['t'], fx['y'], fx['bands'], fx['dy'])
        bench_case('mb_%s_h%d_K%d' % (sub, H, K), ftp, fx['freqs'])

    # chunk-size bitwise pin (patched path), small grid
    fx = fixtures.mb_fixture('RRab', H=4, nfreq=128)
    vocab = [mb.build_template_set(s, fx['bands']) for s in fx['vocab'][:2]]
    transforms, stats, freqs = mb._prepare_band_transforms(
        fx['t'], fx['y'], fx['bands'], fx['dy'], fx['freqs'], 4,
        'floating_offsets', None, fast=True)
    mb._combine_stacked_shared_amp = fast_combine
    p64 = mb.multiband_power_spectra_batched(vocab, transforms, stats, freqs,
                                             chunk_size=64)
    p4096 = mb.multiband_power_spectra_batched(vocab, transforms, stats,
                                               freqs, chunk_size=4096)
    mb._combine_stacked_shared_amp = ORIG
    print('patched chunk-size bitwise invariant (64 vs 4096):',
          np.array_equal(p64, p4096))


if __name__ == '__main__':
    main()
