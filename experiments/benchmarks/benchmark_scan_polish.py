"""WP C2 gate-5 speedup benchmark: method='scan' vs method='eigvals'.

End-to-end ``template_periodogram`` wall time (NFFT summations included) on
a fixed random fixture, repeated and the minimum taken. Gate: scan is
>= 15x faster at H = 8 and >= 3x at H = 2. Run:

    .venv/bin/python experiments/benchmarks/benchmark_scan_polish.py
"""
import time

import numpy as np

from ftperiodogram.core import template_periodogram
from ftperiodogram.template import Template

N_OBS = 300
N_FREQ = 20000
N_REPEATS = 3
SEED = 0


def _fixture(H):
    rng = np.random.RandomState(SEED)
    template = Template(rng.randn(H), rng.randn(H))
    t = np.sort(10.0 * rng.rand(N_OBS))
    dy = 0.05 * (1 + rng.rand(N_OBS))
    y = 1.5 * template((t / 0.77) % 1.0) + dy * rng.randn(N_OBS)
    freqs = 0.001 * (100 + np.arange(N_FREQ))
    return t, y, dy, template, freqs


def _time(method, t, y, dy, template, freqs):
    best = np.inf
    p = None
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        p, _prm = template_periodogram(t, y, dy, template.c_n, template.s_n,
                                       freqs, fast=True, method=method)
        best = min(best, time.perf_counter() - t0)
    return best, p


def main():
    print("N_obs={0}, N_freq={1}, best of {2} repeats, seed={3}".format(
        N_OBS, N_FREQ, N_REPEATS, SEED))
    for H, gate in ((2, 3.0), (8, 15.0), (10, None)):
        t, y, dy, template, freqs = _fixture(H)
        t_eig, p_eig = _time('eigvals', t, y, dy, template, freqs)
        t_scan, p_scan = _time('scan', t, y, dy, template, freqs)
        dp = float(np.max(np.abs(p_eig - p_scan)))
        speedup = t_eig / t_scan
        gate_str = ("PASS" if speedup >= gate else "FAIL") if gate else "info"
        print("H={0:2d}: eigvals {1:7.2f} s | scan {2:6.2f} s | "
              "speedup {3:6.1f}x (gate {4}: {5}) | max|dP| {6:.2e}".format(
                  H, t_eig, t_scan, speedup,
                  '>= {0:.0f}x'.format(gate) if gate else 'n/a',
                  gate_str, dp))


if __name__ == '__main__':
    main()
