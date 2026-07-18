"""Shared harness for the Paper-2 Track-B measurements (HANDOFF.md S4).

Locks the "projected" cost levers on real template shapes + realistic cadences:
  B1  recovery-vs-H at each H's converged grid, RRab & RRc separately
  B3  vocabulary-size vs recall (confirm the flat-in-K rerun result)
  B4  deep-|MM'|-dip fallback rate on realistic cadences (feeds Track-A #4)
  B5  PS1 nfreq / cost inputs

Everything here is fresh code (no reuse of the paper-repo scripts); it leans only
on the public ftperiodogram API and the two locally-cached data sources:
  - Baeza-Villagra et al. (2025) DECam *griz* RR Lyrae templates (typed RRab/RRc),
    cached in ~/.ftperiodogram_data/{RRab,RRc}_normalized.zip
  - the 99-object real ZTF cadence sample in ~/.ftperiodogram_data/ztf_cadence_sample/

No real PS1 3pi cadence exists yet (WP P0.2 CasJobs not run) -> the PS1 arm uses a
faithful synthetic griz cadence (few epochs/band, multi-year seasonal), and the
real-cadence stressor is the cached ZTF sample. The real-PS1 measurement stays a
flagged gap pending P0.2.
"""
import os
import numpy as np

from ftperiodogram.catalog_builder import (fetch_baeza_villagra_templates,
                                           build_template_catalog)
from ftperiodogram.simulate import (SyntheticCadence, RealZTFCadence,
                                     simulate_multiband_lightcurve, exp_mag_error)
from ftperiodogram.multiband import FastMultibandTemplatePeriodogram
from ftperiodogram import recovery as rec

GRIZ = ('g', 'r', 'i', 'z')

# Bailey period boxes (days) for the two subtypes -- the realistic injection ranges.
PERIOD_BOX = {'RRab': (0.44, 0.90), 'RRc': (0.20, 0.45)}
# griz amplitude ratios relative to r (rough RRL colour-amplitude trend; g largest).
BAND_AMP = {'g': 1.25, 'r': 1.0, 'i': 0.78, 'z': 0.66}

# True g-band peak-to-peak amplitude ranges (mag) -- 2026-07-18 audit fix:
# Template is unit-Fourier-energy normalized, so the simulator 'amplitude' is
# NOT mag p2p; inject() now draws a target p2p from these and rescales by the
# shape's measured p2p. (RRab ~0.5-1.2 mag in g, RRc ~0.2-0.5.)
P2P_G = {'RRab': (0.5, 1.2), 'RRc': (0.2, 0.5)}

# PS1 3pi single-epoch 5-sigma depths (AB mag) -- audit fix: the default
# exp_mag_error is ~4-5x optimistic at the faint end.
PS1_M5 = {'g': 22.0, 'r': 21.8, 'i': 21.5, 'z': 20.9}
_SIGMA_SYS = 0.013     # bright-end systematic floor (mag)


def ps1_mag_error(band):
    """PS1 3pi single-epoch error model for one band:
    sigma = sqrt(sigma_sys^2 + (1.0857 / SNR)^2), SNR = 5 * 10^(-0.4 (m - m5))."""
    m5 = PS1_M5[band]

    def _sigma(mag):
        mag = np.asarray(mag, dtype=float)
        snr = 5.0 * 10.0 ** (-0.4 * (mag - m5))
        return np.sqrt(_SIGMA_SYS ** 2 + (1.0857 / snr) ** 2)
    return _sigma


def _ps1_err_model_flat(bands_of_epoch):
    """Error model callable for a flat multiband cadence: needs the band of
    each epoch, captured at cadence-build time (see Ps1PairCadence)."""
    sig_fns = {b: ps1_mag_error(b) for b in PS1_M5}

    def _sigma(mag):
        mag = np.asarray(mag, dtype=float)
        out = np.empty(mag.shape, dtype=float)
        for b in np.unique(bands_of_epoch):
            m = bands_of_epoch == b
            out[m] = sig_fns[str(b)](mag[m])
        return out
    return _sigma


def template_p2p(template, n_phase=512):
    """Measured peak-to-peak of a (unit-Fourier-energy) template shape."""
    ph = np.arange(n_phase) / float(n_phase)
    vals = np.asarray(template(ph), dtype=float)
    return float(vals.max() - vals.min())


# ----------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------
_BANK_CACHE = {}


def load_bank(subtype, H, bands=GRIZ, n_phase=256):
    """BV-2025 templates of one subtype, one Template per (star, band) truncated
    to ``H`` harmonics. Cached per (subtype, H, bands)."""
    key = (subtype, int(H), tuple(bands))
    if key not in _BANK_CACHE:
        tmpls = fetch_baeza_villagra_templates(
            subtypes=(subtype,), bands=list(bands), normalized=True,
            nharmonics=int(H), n_phase=n_phase)
        _BANK_CACHE[key] = tmpls
    return _BANK_CACHE[key]


def _band_of(template_id):
    return template_id.rsplit('-', 1)[-1]


def _star_of(template_id):
    return template_id.rsplit('-', 1)[0]


def unique_stars(bank):
    """Ordered unique star ids in a (star,band) BV bank."""
    seen, out = set(), []
    for t in bank:
        s = _star_of(t.template_id)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def shared_shape_template(bank, star):
    """One representative shared shape (the 'r'-band template, else the first
    band present) for a BV star -- the injected shared griz shape."""
    per_band = {_band_of(t.template_id): t for t in bank if _star_of(t.template_id) == star}
    return per_band.get('r', next(iter(per_band.values())))


def split_stars(subtype, seed=0, frac_vocab=0.5):
    """Deterministic held-out split of the BV stars (2026-07-18 audit fix:
    truths and vocab must not share stars for absolute rates / the K lever).
    Returns (vocab_stars, truth_stars)."""
    stars = unique_stars(load_bank(subtype, 8))
    rng = np.random.RandomState(seed + 424243)
    order = rng.permutation(len(stars))
    n_voc = int(round(frac_vocab * len(stars)))
    voc = [stars[i] for i in order[:n_voc]]
    tru = [stars[i] for i in order[n_voc:]]
    return voc, tru


def detection_vocab(subtype, H, K, bands=GRIZ, seed=0, stars=None):
    """K-medoid RRL detection vocabulary of one subtype at order H.

    Clusters the shared-shape (one template per star) set with
    build_template_catalog (PAM, phase-shift-invariant orbit distance).
    ``stars`` restricts the candidate pool (pass the held-out vocab split);
    None uses the full bank (in-sample; fine for paired H-comparisons only).
    When K >= pool size the whole pool is returned."""
    bank = load_bank(subtype, H, bands=bands)
    pool = unique_stars(bank) if stars is None else list(stars)
    shapes = [shared_shape_template(bank, s) for s in pool]
    if K >= len(shapes):
        return shapes
    return build_template_catalog(shapes, n_clusters=int(K), metric='orbit',
                                  method='pam', random_state=seed)


# ----------------------------------------------------------------------
# Cadences
# ----------------------------------------------------------------------
class Ps1PairCadence(SyntheticCadence):
    """PS1-3pi-like griz cadence with same-night TTI pairs (2026-07-18 audit fix).

    Real PS1 3pi visits one filter per night in a Transient-Time-Interval pair:
    two exposures ~15-40 min apart, ~n_per_band/2 nights per filter over the
    multi-year baseline, seasonally windowed. The pair structure (intra-night
    Delta t ~ 25 min, inter-night gaps of days-months) drives the 1-day alias
    ladder and the phase clustering that stresses |MM| conditioning -- the
    uniform-in-season SyntheticCadence draw lacks both.

    Odd n_per_band gets (n-1)/2 pairs + 1 single. Per-band error model is the
    PS1 single-epoch relation (ps1_mag_error).
    """

    def __init__(self, n_per_band, baseline_days=1600.0, bands=GRIZ,
                 season_length_days=250.0, pair_dt_min=(15.0, 40.0), seed=0):
        counts = ({b: int(n_per_band[b]) for b in bands}
                  if isinstance(n_per_band, dict)
                  else {b: int(n_per_band) for b in bands})
        SyntheticCadence.__init__(
            self, n_epochs=counts, bands=list(bands),
            baseline_days=baseline_days,
            season_length_days=season_length_days, random_state=seed)
        self.pair_dt_min = pair_dt_min

    def sample(self, rng=None):
        from ftperiodogram.simulate import CadenceSample, _as_rng
        rng = _as_rng(self.random_state if rng is None else rng)
        lo, hi = self.pair_dt_min
        t_parts, b_parts = [], []
        for b in self.bands:
            n = self.n_epochs[b]
            n_pairs, extra = divmod(n, 2)
            nights = self._draw_epochs(n_pairs + extra, rng)   # night anchors
            nights = np.floor(nights) + 0.25 + 0.5 * rng.rand(nights.size)
            tb = []
            for k, t0 in enumerate(nights):
                tb.append(t0)
                if k < n_pairs:                                # TTI partner
                    tb.append(t0 + rng.uniform(lo, hi) / 1440.0)
            tb = np.sort(np.asarray(tb, dtype=float))
            t_parts.append(tb)
            b_parts.append(np.full(tb.size, b))
        t = np.concatenate(t_parts)
        bands = np.concatenate(b_parts)
        order = np.argsort(t, kind='mergesort')
        t, bands = t[order], bands[order]
        return CadenceSample(t=t, bands=bands,
                             mag_err_model=_ps1_err_model_flat(bands))


def ps1_cadence(n_per_band, baseline_days=1600.0, bands=GRIZ,
                season_length_days=250.0, seed=0):
    """PS1-3pi-like griz cadence: TTI same-night pairs, seasonal windows,
    per-band PS1 single-epoch errors. ``n_per_band`` int or {band: int}."""
    return Ps1PairCadence(n_per_band, baseline_days=baseline_days, bands=bands,
                          season_length_days=season_length_days, seed=seed)


def ztf_cadence_ids(data_home=None):
    home = (os.path.join(os.path.expanduser("~"), ".ftperiodogram_data",
                         "ztf_cadence_sample") if data_home is None else data_home)
    return sorted(f[:-4] for f in os.listdir(home) if f.endswith('.npz'))


def load_ztf_cadence(oid, bands=None, max_epochs_per_band=None):
    """A real ZTF cadence (g/r) from the local sample, optionally thinned."""
    return RealZTFCadence.from_cache(oid, bands=bands, catflags_max=0,
                                     max_epochs_per_band=max_epochs_per_band)


# ----------------------------------------------------------------------
# Injection
# ----------------------------------------------------------------------
def inject(truth_shape, subtype, cadence, seed, mean_mag=20.5, p2p_g=None):
    """Inject a shared-shape RRL onto a (multiband) cadence with griz band
    amplitudes + random per-band offsets, at a random Bailey-box period.

    2026-07-18 audit fix: amplitude is specified as TRUE g-band peak-to-peak
    (mag). ``p2p_g=None`` draws from the subtype range P2P_G; the simulator
    amplitude is p2p_g / (BAND_AMP['g'] * p2p(shape)) so the injected g-band
    signal really spans ``p2p_g`` mag. Returns dict with t,y,bands,dy,P_true,
    p2p_g,baseline."""
    rng = np.random.RandomState(seed)
    lo, hi = PERIOD_BOX[subtype]
    period = rng.uniform(lo, hi)
    tau = rng.uniform(0, 1)
    band_off = {b: rng.uniform(-0.3, 0.3) for b in GRIZ}
    if p2p_g is None:
        p2p_g = rng.uniform(*P2P_G[subtype])
    amplitude = float(p2p_g) / (BAND_AMP['g'] * template_p2p(truth_shape))
    lc = simulate_multiband_lightcurve(
        truth_shape, period, cadence, amplitude=amplitude, mean_mag=mean_mag,
        tau=tau, band_amplitudes=dict(BAND_AMP), band_offsets=band_off,
        random_state=rng, add_noise=True, shuffle=True)
    T = float(np.max(lc.t) - np.min(lc.t))
    return dict(t=np.asarray(lc.t, float), y=np.asarray(lc.y, float),
                bands=np.asarray(lc.bands), dy=np.asarray(lc.dy, float),
                P_true=period, p2p_g=float(p2p_g), baseline=T)


# ----------------------------------------------------------------------
# Frequency grid (each H at its own converged resolution)
# ----------------------------------------------------------------------
def converged_grid(f_lo, f_hi, H, baseline, oversample=4.0):
    """Grid whose step resolves the template-periodogram peak at order ``H``.

    The FTP peak half-width in frequency is ~1/(H*T) (higher harmonics sharpen
    the peak), so a converged grid needs df <= 1/(oversample * H * T). One grid
    built for the largest H in a sweep is therefore converged for every smaller H
    as well; the H=8 convergence is checked explicitly by doubling ``oversample``.
    """
    df = 1.0 / (float(oversample) * float(H) * float(baseline))
    # the NFFT fast path requires freqs[0] to be an integer multiple of df
    # (ftperiodogram.summations.inspect_freqs); snap f_lo down to the grid.
    dnf = int(np.floor(f_lo / df))
    f_start = dnf * df
    n = int(np.ceil((f_hi - f_start) / df)) + 1
    return f_start + df * np.arange(n)


# ----------------------------------------------------------------------
# FTP recovery
# ----------------------------------------------------------------------
def ftp_best_period(vocab, src, freqs, mode='floating_offsets'):
    """Argmax-power period of the FTP over a template ``vocab`` (bank max at each
    frequency). Uses the batched floating_offsets path (fast=True)."""
    model = FastMultibandTemplatePeriodogram(list(vocab), mode=mode)
    model.fit(src['t'], src['y'], src['bands'], src['dy'])
    powers = model.power(np.asarray(freqs, float), fast=True,
                         save_best_model=False)
    return 1.0 / freqs[int(np.argmax(powers))], powers


def score(P_rec, P_true, baseline, rtol=0.01, delta_phi_max=0.5):
    """Both recovery conventions, exact-or-harmonic. Returns a dict of bools."""
    frac = rec.classify_recovery(P_rec, P_true, baseline=baseline,
                                 criterion='fractional', rtol=rtol,
                                 count_harmonics=True)
    phase = rec.classify_recovery(P_rec, P_true, baseline=baseline,
                                  criterion='phase_coherence',
                                  delta_phi_max=delta_phi_max,
                                  count_harmonics=True)
    return dict(frac_exact=frac.exact, frac_recovered=frac.recovered,
                frac_alias=frac.alias_name,
                phase_exact=phase.exact, phase_recovered=phase.recovered,
                phase_alias=phase.alias_name)
