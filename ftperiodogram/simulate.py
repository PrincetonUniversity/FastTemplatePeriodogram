"""Light-curve simulator and observing-cadence model (Phase 3.2 harness).

Given a :class:`~ftperiodogram.template.Template`, an injected period, an observing
:class:`Cadence`, and a photometric noise model, produce single-band ``(t, y, dy)``
and multiband ``(t, y, bands, dy)`` light curves whose underlying signal matches the
model :class:`~ftperiodogram.modeler.TemplateModel` fits::

    y(t) = mean_mag + amplitude * template(t / period - tau) + noise

The :class:`Cadence` abstraction is a one-method protocol (``sample(rng) ->
CadenceSample``) so a *real* ZTF DR22 cadence drops in later as a subclass with
zero churn to the simulators or the recovery metrics.  The bundled
:class:`SyntheticCadence` produces random epochs inside a multi-year baseline with
yearly seasonal gaps, per-band epoch counts, and a heteroscedastic
error-vs-magnitude relation (fainter => larger sigma).

stdlib + numpy only.  Observing windows are set by explicit time spans, never via a
Nyquist heuristic.
"""
import os
from collections import namedtuple

import numpy as np


def _as_rng(random_state):
    """Coerce ``None`` / int / RandomState into a ``np.random.RandomState``."""
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


# ----------------------------------------------------------------------
# Error-vs-magnitude noise model
# ----------------------------------------------------------------------
def exp_mag_error(sigma_floor=0.01, sigma_scale=0.005, mag_ref=20.0,
                  mag_scale=1.0, sigma_max=0.5):
    """Heteroscedastic photometric error vs magnitude (fainter => larger sigma).

    Returns a callable ``mag -> sigma`` so a measured ZTF ``sigma(mag)`` table can
    later replace it with no API change::

        sigma(mag) = clip(sigma_floor + sigma_scale * exp((mag - mag_ref)/mag_scale),
                          sigma_floor, sigma_max)

    Defaults give ~0.01 mag at the bright end rising toward the survey limit, with
    a clip so the weights ``dy**-2`` never blow up.
    """
    def _sigma(mag):
        mag = np.asarray(mag, dtype=float)
        s = sigma_floor + sigma_scale * np.exp((mag - mag_ref) / mag_scale)
        return np.clip(s, sigma_floor, sigma_max)
    return _sigma


# ----------------------------------------------------------------------
# Cadence abstraction
# ----------------------------------------------------------------------
#: Result of sampling a cadence: observation epochs, per-epoch band labels
#: (``None`` for single-band), and the error-vs-mag relation to apply.
CadenceSample = namedtuple('CadenceSample', ['t', 'bands', 'mag_err_model'])


def _in_season(t, t0, season_length_days, season_period_days):
    """Boolean mask: epoch falls inside a visible (yearly) observing season."""
    phase = np.mod(np.asarray(t, dtype=float) - t0, season_period_days)
    return phase < season_length_days


class Cadence(object):
    """Abstract observing cadence.

    Subclass and implement :meth:`sample` to map a request to observation epochs
    plus an error-vs-magnitude relation.  A real ZTF DR22 cadence is a drop-in
    subclass that returns its recorded epochs through the same
    :class:`CadenceSample`.
    """

    def sample(self, rng=None):
        """Return a :class:`CadenceSample`.  Implementations MUST return globally
        ascending-sorted ``t``.  ``rng`` may be ``None`` / int / RandomState."""
        raise NotImplementedError

    @property
    def baseline(self):
        """A-priori observing time span ``T`` (days), if known before sampling."""
        raise NotImplementedError


class SyntheticCadence(Cadence):
    """Random epochs in a multi-year baseline with yearly seasonal gaps.

    Parameters
    ----------
    n_epochs : int or dict {band: int}, default 60
        Per-band epoch counts.  An int means the same count in every band.
    bands : sequence of hashable or None, default None
        Band labels.  ``None`` => single-band (``CadenceSample.bands`` is ``None``).
    baseline_days : float, default 3 * 365.25
        Total observing window ``T``; epochs are drawn in ``[t0, t0 + baseline_days]``.
    t0 : float, default 0.0
        Window start (days).
    season_length_days : float, default 270.0
        Visible length of each yearly observing season (the rest is a gap).
    season_period_days : float, default 365.25
        Spacing between season starts (one season per year).
    err_model : callable mag -> sigma or None
        Error-vs-magnitude relation; defaults to :func:`exp_mag_error`.
    random_state : int or RandomState or None
        Default seed used by :meth:`sample` when no ``rng`` is passed.
    max_draw_factor : int, default 20
        Cap on rejection-sampling batches before giving up (degenerate duty cycle).
    """

    def __init__(self, n_epochs=60, bands=None, baseline_days=3 * 365.25, t0=0.0,
                 season_length_days=270.0, season_period_days=365.25,
                 err_model=None, random_state=None, max_draw_factor=20):
        if baseline_days <= 0:
            raise ValueError("baseline_days must be positive")
        if not 0 < season_length_days <= season_period_days:
            raise ValueError("require 0 < season_length_days <= season_period_days")
        if bands is None and isinstance(n_epochs, dict):
            raise ValueError("dict n_epochs requires explicit `bands`")
        self.n_epochs = n_epochs
        self.bands = None if bands is None else list(bands)
        self.baseline_days = float(baseline_days)
        self.t0 = float(t0)
        self.season_length_days = float(season_length_days)
        self.season_period_days = float(season_period_days)
        self.err_model = exp_mag_error() if err_model is None else err_model
        self.random_state = random_state
        self.max_draw_factor = int(max_draw_factor)

    @property
    def baseline(self):
        return self.baseline_days

    def _draw_epochs(self, count, rng):
        """Rejection-sample ``count`` in-season epochs; sorted ascending."""
        if count <= 0:
            raise ValueError("per-band epoch count must be positive; got %r"
                             % (count,))
        kept, n_kept, attempts = [], 0, 0
        batch = max(count * 4, 16)
        while n_kept < count and attempts < self.max_draw_factor:
            cand = self.t0 + rng.rand(batch) * self.baseline_days
            cand = cand[_in_season(cand, self.t0, self.season_length_days,
                                   self.season_period_days)]
            kept.append(cand)
            n_kept += len(cand)
            attempts += 1
        t = np.concatenate(kept) if kept else np.array([])
        if len(t) < count:
            raise ValueError("could not draw %d in-season epochs in %d batches "
                             "(duty cycle too low?)" % (count, self.max_draw_factor))
        return np.sort(t[:count])

    def sample(self, rng=None):
        rng = _as_rng(self.random_state if rng is None else rng)
        if self.bands is None:
            if isinstance(self.n_epochs, dict):
                raise ValueError("dict n_epochs requires explicit `bands`")
            t = self._draw_epochs(int(self.n_epochs), rng)
            return CadenceSample(t=t, bands=None, mag_err_model=self.err_model)

        if isinstance(self.n_epochs, dict):
            counts = {b: int(self.n_epochs[b]) for b in self.bands}
        else:
            counts = {b: int(self.n_epochs) for b in self.bands}
        t_parts, b_parts = [], []
        for b in self.bands:
            tb = self._draw_epochs(counts[b], rng)
            t_parts.append(tb)
            b_parts.append(np.full(counts[b], b))
        t = np.concatenate(t_parts)
        bands = np.concatenate(b_parts)
        order = np.argsort(t, kind='mergesort')
        return CadenceSample(t=t[order], bands=bands[order],
                             mag_err_model=self.err_model)


class RealZTFCadence(Cadence):
    """A *real* observing cadence taken from recorded per-band ZTF epochs.

    This is the drop-in real-data sibling of :class:`SyntheticCadence`: instead of
    rejection-sampling epochs inside a parametric seasonal window, it replays the
    actual MJDs of a recorded ZTF DR light curve (with its real seasonal gaps,
    per-band epoch counts, and clumping), plus an *empirical* error-vs-magnitude
    relation measured from that survey's ``magerr`` vs ``mag`` scatter.  Because it
    is a :class:`Cadence` subclass returning the same :class:`CadenceSample`, every
    simulator and recovery driver consumes it with zero churn -- a population is
    simulated by injecting different truth shapes onto this one *fixed* sampling
    pattern, exactly as with the synthetic cadence.

    Parameters
    ----------
    epochs_by_band : dict {band: 1-D array of MJD}
        Recorded observation epochs (days) per band.  Each array must be non-empty;
        order is irrelevant (epochs are concatenated and globally mergesorted).
    err_model : callable mag -> sigma or None
        Empirical photometric error-vs-magnitude relation (e.g. from
        :func:`ftperiodogram.simulate.exp_mag_error` or a measured ZTF table).
        ``None`` falls back to :func:`exp_mag_error` defaults (synthetic), which is
        only sensible for a smoke test -- pass the real one in production.
    object_id : hashable or None
        Provenance tag (e.g. the ZTF ``oid`` / sky position the cadence came from).
    metadata : dict or None
        Free-form provenance (release, RA/Dec, ngoodobs, ...); not used in sampling.

    Notes
    -----
    :meth:`sample` ignores ``rng`` (the cadence is deterministic -- the *noise* is
    added by the simulators, not here), so a frozen population built on a
    :class:`RealZTFCadence` reuses the identical real sampling for every source.
    """

    def __init__(self, epochs_by_band, err_model=None, object_id=None,
                 metadata=None):
        if not epochs_by_band:
            raise ValueError("epochs_by_band must be a non-empty {band: epochs} dict")
        self.bands = list(epochs_by_band.keys())
        self._epochs = {}
        for b in self.bands:
            arr = np.sort(np.asarray(epochs_by_band[b], dtype=float))
            if arr.size == 0:
                raise ValueError("band %r has no epochs" % (b,))
            if not np.all(np.isfinite(arr)):
                raise ValueError("band %r has non-finite epochs" % (b,))
            self._epochs[b] = arr
        self.err_model = exp_mag_error() if err_model is None else err_model
        self.object_id = object_id
        self.metadata = dict(metadata or {})

        all_t = np.concatenate([self._epochs[b] for b in self.bands])
        self._t_min = float(all_t.min())
        self._t_max = float(all_t.max())

    @property
    def baseline(self):
        """Total recorded observing span ``T = max(t) - min(t)`` (days)."""
        return self._t_max - self._t_min

    def epoch_counts(self):
        """Per-band recorded epoch counts ``{band: n}``."""
        return {b: int(self._epochs[b].size) for b in self.bands}

    def sample(self, rng=None):
        """Replay the recorded epochs as a globally ascending-sorted multiband
        :class:`CadenceSample` (deterministic; ``rng`` is ignored)."""
        t_parts, b_parts = [], []
        for b in self.bands:
            tb = self._epochs[b]
            t_parts.append(tb)
            b_parts.append(np.full(tb.size, b))
        t = np.concatenate(t_parts)
        bands = np.concatenate(b_parts)
        order = np.argsort(t, kind='mergesort')
        return CadenceSample(t=t[order], bands=bands[order],
                             mag_err_model=self.err_model)

    @classmethod
    def from_cache(cls, oid, *, err_model=None, bands=None, data_home=None,
                   catflags_max=0, max_epochs_per_band=None):
        """Build a :class:`RealZTFCadence` from one cached ZTF object's epochs.

        Reads the per-object record written by
        ``experiments/phase3_recovery/fetch_ztf_cadence.py`` (a ``.npz`` under
        ``~/.ftperiodogram_data/ztf_cadence_sample/`` keyed by sky-grouped ``oid``).
        Lazily imports nothing network-y -- it only touches the local cache, so the
        core package stays pure numpy/scipy.

        Parameters
        ----------
        oid : str
            The grouped-object id (cache filename stem ``<oid>.npz``).
        err_model : callable mag -> sigma or None
            Error model to attach (defaults to :func:`exp_mag_error`).
        bands : sequence of str or None
            Keep only these bands (e.g. ``['g', 'r']``); ``None`` keeps all present.
        data_home : str or None
            Cache root (defaults to ``~/.ftperiodogram_data/ztf_cadence_sample``).
        catflags_max : int
            Keep only epochs with ``catflags <= catflags_max`` (0 = clean only).
        max_epochs_per_band : int or None
            If set, *evenly* thin each band to at most this many epochs across the
            full baseline (``np.linspace`` index subset).  This preserves the
            seasonal-gap structure and full time span while capping the per-band
            count.  NB (2026-07-18 re-audit): aggressive thinning does NOT
            preserve intra-night clumping -- across the 99-cadence sample the
            same-night (<0.5 d) consecutive-pair count drops from 27,549 (full)
            to 97 (n=20) to 9 (n=8), so heavily-thinned cadences behave as
            sparse quasi-random sampling, not as intra-night-structured data.
            ``None`` keeps every recorded epoch.
        """
        home = (os.path.join(os.path.expanduser("~"), ".ftperiodogram_data",
                             "ztf_cadence_sample")
                if data_home is None else data_home)
        path = os.path.join(home, "%s.npz" % oid)
        if not os.path.exists(path):
            raise FileNotFoundError("no cached ZTF cadence at %s" % path)
        with np.load(path, allow_pickle=True) as rec:
            t = np.asarray(rec['mjd'], dtype=float)
            bnd = np.asarray(rec['band']).astype(str)
            catf = (np.asarray(rec['catflags'], dtype=int) if 'catflags' in rec
                    else np.zeros(t.size, dtype=int))
            meta = (rec['metadata'].item() if 'metadata' in rec.files
                    else {})
        keep = catf <= int(catflags_max)
        t, bnd = t[keep], bnd[keep]
        present = list(dict.fromkeys(bnd.tolist()))
        wanted = present if bands is None else [b for b in bands if b in present]
        if not wanted:
            raise ValueError("none of bands=%r present for oid=%s (have %r)"
                             % (bands, oid, present))
        epochs_by_band = {}
        for b in wanted:
            tb = np.sort(t[bnd == b])
            if (max_epochs_per_band is not None
                    and tb.size > int(max_epochs_per_band)):
                idx = np.linspace(0, tb.size - 1, int(max_epochs_per_band))
                tb = tb[np.unique(np.round(idx).astype(int))]
            epochs_by_band[b] = tb
        return cls(epochs_by_band, err_model=err_model, object_id=oid,
                   metadata=meta)


# ----------------------------------------------------------------------
# Simulators
# ----------------------------------------------------------------------
SimulatedLightCurve = namedtuple(
    'SimulatedLightCurve',
    ['t', 'y', 'dy', 'period', 'frequency', 'params', 'cadence'])

SimulatedMultibandLightCurve = namedtuple(
    'SimulatedMultibandLightCurve',
    ['t', 'y', 'bands', 'dy', 'period', 'frequency', 'params', 'cadence'])


def simulate_lightcurve(template, period, cadence, amplitude=0.5, mean_mag=15.0,
                        tau=0.0, random_state=None, add_noise=True):
    """Single-band light curve from a ``template`` on a (single-band) ``cadence``.

    ``y = mean_mag + amplitude * template(t/period - tau)``; per-epoch ``dy`` comes
    from the cadence's error-vs-mag model evaluated at the clean magnitude, and
    Gaussian noise of that scale is added when ``add_noise``.  ``t`` is returned
    ascending-sorted (the single-band modeler contract).  ``tau`` is a phase offset
    in turns.
    """
    period = float(period)
    if period <= 0:
        raise ValueError("period must be positive")
    rng = _as_rng(random_state)
    sample = cadence.sample(rng)
    if sample.bands is not None:
        raise ValueError("simulate_lightcurve needs a single-band cadence; use "
                         "simulate_multiband_lightcurve for a multiband cadence")
    t = np.sort(np.asarray(sample.t, dtype=float))
    if t.size < 2:
        raise ValueError("need at least 2 epochs to simulate a light curve")
    y_clean = mean_mag + amplitude * np.asarray(template(t / period - tau))
    dy = np.maximum(np.asarray(sample.mag_err_model(y_clean), dtype=float), 1e-12)
    y = y_clean + dy * rng.randn(t.size) if add_noise else y_clean
    params = dict(amplitude=amplitude, mean_mag=mean_mag, tau=tau,
                  add_noise=add_noise)
    return SimulatedLightCurve(t=t, y=y, dy=dy, period=period,
                               frequency=1.0 / period, params=params,
                               cadence=cadence)


def simulate_multiband_lightcurve(template, period, cadence, amplitude=0.5,
                                  mean_mag=15.0, tau=0.0, band_amplitudes=None,
                                  band_offsets=None, random_state=None,
                                  add_noise=True, shuffle=True):
    """Multiband light curve sharing one shape across bands (floating-offsets regime).

    For band ``b``::

        y_b = mean_mag + band_offsets[b]
              + amplitude * band_amplitudes[b] * template(t/period - tau)

    ``band_amplitudes`` / ``band_offsets`` are dicts keyed by the cadence's band
    labels (``None`` => 1.0 / 0.0 everywhere, i.e. a pure shared shape that exactly
    matches the modeler's ``floating_offsets`` mode).  With ``shuffle`` the flat
    arrays are returned in random global order (the multiband modeler needs no
    sorting).
    """
    period = float(period)
    if period <= 0:
        raise ValueError("period must be positive")
    rng = _as_rng(random_state)
    sample = cadence.sample(rng)
    if sample.bands is None:
        raise ValueError("simulate_multiband_lightcurve needs a multiband cadence")
    t = np.asarray(sample.t, dtype=float)
    bands = np.asarray(sample.bands)
    uniq = list(dict.fromkeys(bands.tolist()))
    if band_amplitudes is not None:
        missing = [b for b in uniq if b not in band_amplitudes]
        if missing:
            raise ValueError("band_amplitudes missing bands %r" % (missing,))
    if band_offsets is not None:
        missing = [b for b in uniq if b not in band_offsets]
        if missing:
            raise ValueError("band_offsets missing bands %r" % (missing,))
    amp_vec = np.array([1.0 if band_amplitudes is None else band_amplitudes[b]
                        for b in bands.tolist()], dtype=float)
    off_vec = np.array([0.0 if band_offsets is None else band_offsets[b]
                        for b in bands.tolist()], dtype=float)
    shape = np.asarray(template(t / period - tau))
    y_clean = mean_mag + off_vec + amplitude * amp_vec * shape
    dy = np.maximum(np.asarray(sample.mag_err_model(y_clean), dtype=float), 1e-12)
    y = y_clean + dy * rng.randn(t.size) if add_noise else y_clean
    if shuffle:
        order = rng.permutation(t.size)
        t, y, bands, dy = t[order], y[order], bands[order], dy[order]
    params = dict(amplitude=amplitude, mean_mag=mean_mag, tau=tau,
                  band_amplitudes=band_amplitudes, band_offsets=band_offsets,
                  add_noise=add_noise)
    return SimulatedMultibandLightCurve(t=t, y=y, bands=bands, dy=dy,
                                        period=period, frequency=1.0 / period,
                                        params=params, cadence=cadence)
