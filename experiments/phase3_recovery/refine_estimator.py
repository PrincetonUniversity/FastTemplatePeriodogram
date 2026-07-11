"""Peak-refinement wrapper for the pluggable period estimators (WP B8-closure b).

B8 arm (c) showed the production 10k grid is NOT converged for the H=8 and
binned methods (multiharmonic peak width ~ Rayleigh/H; 10k gives ~0.29 grid
points per H=8 peak width), so absolute sparse-N rates were quoted as lower
bounds "at the 10k grid".  Rather than a ~4x-cost plain dense rerun (which arm
(c) suggests may STILL not converge for MHLS), this wraps any estimator with
local peak refinement -- the plan of record in SUMMARY_c-grid20k.md:

1. evaluate the base estimator's spectrum over the coarse grid;
2. take the top ``top_m`` local extrema (plateau-safe: runs of equal values
   collapse to one representative, so CE's sparse-N entropy plateaus cannot
   eat the candidate budget);
3. refine each candidate on a fine window ``+/- window * df`` at spacing
   ``df / factor`` using the base's own spectrum;
4. return ``1 / f_best`` over coarse + all fine evaluations.

Cost is ~``1 + top_m * (2 * window * factor) / n_freq`` of a coarse pass
(~1.4x for the defaults on the 10k grid), and the returned peak value is the
continuum-limit one for any peak the coarse grid detects at all.

Spectrum seam per method: CE wraps ``entropy_spectrum`` with ``better='min'``;
GLS / MHLS / MBLS wrap ``power_spectrum`` (their per-frequency designs accept
arbitrary grids); FTP wraps ``power_spectrum`` too -- its fast NFFT summations
require an aligned uniform grid (``freqs[0]`` an integer multiple of ``df``),
which every window grid here satisfies BY CONSTRUCTION: the coarse grid is
NFFT-aligned (``validation._align_grid``), so a coarse peak ``f0 = n * df`` is
also ``n * factor`` fine steps, and the window points ``f0 + k * df/factor``
are all integer multiples of the fine spacing.

The refined search never leaves the declared band: window points are clipped
to the coarse ``[freqs[0], freqs[-1]]`` (explicit [f_min, f_max] discipline).
Instances are picklable (plain attributes + module-level class), so they ship
through the ``spawn``-pool ``score_estimator`` seam like every baseline.
"""
import numpy as np


def _candidate_peaks(values, top_m, better):
    """Indices of the top ``top_m`` local extrema of ``values``, best first.

    Plateau-safe: a point qualifies when it is >= both neighbors (missing
    neighbors count as -inf, so a rising band edge qualifies), and runs of
    consecutive qualifying indices (flat plateaus) are collapsed to their first
    member before ranking.  ``better='min'`` finds minima.  The global optimum
    is always a candidate by construction.
    """
    v = np.asarray(values, dtype=float)
    if better == 'min':
        v = -v
    ext = np.r_[-np.inf, v, -np.inf]
    idx = np.flatnonzero((ext[1:-1] >= ext[:-2]) & (ext[1:-1] >= ext[2:]))
    if idx.size == 0:                                  # all-NaN spectrum guard
        return idx
    idx = idx[np.r_[True, np.diff(idx) > 1]]           # collapse plateaus
    order = np.argsort(-v[idx], kind='stable')         # best first, grid-stable
    return idx[order[:int(top_m)]]


class RefinedEstimator(object):
    """Wrap a base period estimator with local peak refinement.

    ``base`` must expose the spectrum method named by ``better``:
    ``power_spectrum`` for ``better='max'`` (FTP / GLS / MHLS / MBLS) or
    ``entropy_spectrum`` for ``better='min'`` (conditional entropy).  The
    wrapped object keeps the estimator seam ``__call__(t, y, bands, dy, freqs)
    -> P_rec`` so it drops into ``RecoveryScorer.score_estimator`` unchanged.
    """

    def __init__(self, base, top_m=32, window=2.0, factor=32, better='max'):
        if better not in ('max', 'min'):
            raise ValueError("better must be 'max' or 'min'; got %r" % (better,))
        attr = 'power_spectrum' if better == 'max' else 'entropy_spectrum'
        if not hasattr(base, attr):
            raise ValueError("base estimator %r lacks %s (needed for better=%r)"
                             % (type(base).__name__, attr, better))
        if int(factor) < 2:
            raise ValueError("factor must be >= 2; got %r" % (factor,))
        if float(window) <= 0 or int(top_m) < 1:
            raise ValueError("need window > 0 and top_m >= 1")
        self.base = base
        self.top_m = int(top_m)
        self.window = float(window)
        self.factor = int(factor)
        self.better = better
        self._attr = attr

    def _spectrum(self, t, y, bands, dy, freqs):
        return np.asarray(getattr(self.base, self._attr)(t, y, bands, dy, freqs),
                          dtype=float)

    def __call__(self, t, y, bands, dy, freqs):
        freqs = np.asarray(freqs, dtype=float)
        df = freqs[1] - freqs[0]
        spec = self._spectrum(t, y, bands, dy, freqs)
        sign = 1.0 if self.better == 'max' else -1.0
        s = sign * spec
        best_i = int(np.argmax(s))
        best_val, best_f = s[best_i], freqs[best_i]

        fine_df = df / self.factor
        half = int(round(self.window * self.factor))   # fine steps per side
        for i in _candidate_peaks(spec, self.top_m, self.better):
            # NFFT-safe by construction: freqs[i] is an integer multiple of
            # fine_df because the coarse grid is aligned and factor is integer
            k0 = int(round(freqs[i] / fine_df))
            wf = (k0 + np.arange(-half, half + 1)) * fine_df
            wf = wf[(wf >= freqs[0]) & (wf <= freqs[-1])]  # stay in the band
            if wf.size < 2:                            # NFFT needs >= 2 points
                continue
            ws = sign * self._spectrum(t, y, bands, dy, wf)
            j = int(np.argmax(ws))
            if ws[j] > best_val:
                best_val, best_f = ws[j], wf[j]
        return 1.0 / best_f
