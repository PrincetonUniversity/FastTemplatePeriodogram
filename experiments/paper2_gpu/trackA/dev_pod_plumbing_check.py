#!/usr/bin/env python
"""LOCAL dev check of pod/validate_gpu.py plumbing (no GPU needed).

Substitutes the GPU layer (kernels.gpu_*) with the numpy mirrors and a
minimal fake ``cupy`` module, then runs pod/validate_gpu.py's main().
Every comparison must PASS with ~zero diffs: this proves the fixture
.npz keys, shapes, row ordering, mask logic and gate arithmetic are
correct BEFORE any pod time is spent. It validates none of the CUDA --
that is what the pod session is for.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))

import numpy as np                                     # noqa: E402
import reference as R                                  # noqa: E402


# ---- fake cupy -------------------------------------------------------
def _make_fake_cupy():
    cp = types.ModuleType('cupy')
    cp.asarray = np.asarray
    cp.asnumpy = np.asarray
    cp.ones = np.ones
    cp.repeat = np.repeat
    cp.zeros = np.zeros

    class _Runtime(object):
        @staticmethod
        def getDeviceProperties(i):
            return {'name': b'FAKE-LOCAL-MIRROR'}

        @staticmethod
        def memGetInfo():
            return (int(60e9), int(80e9))

    class _Device(object):
        def synchronize(self):
            pass

    cuda = types.SimpleNamespace(runtime=_Runtime, Device=_Device)
    cp.cuda = cuda
    cp.fft = types.SimpleNamespace(ifft=np.fft.ifft)
    cp.ascontiguousarray = np.ascontiguousarray
    cp.float64 = np.float64
    cp.complex128 = np.complex128
    cp.int32 = np.int32
    return cp


sys.modules['cupy'] = _make_fake_cupy()

import kernels as K                                    # noqa: E402
K.cp = sys.modules['cupy']


# ---- mirror-backed fakes of the GPU wrappers -------------------------
class FakeBandData(object):
    def __init__(self, d):
        self.t = np.asarray(d['t'], dtype=np.float64)
        self.w = np.asarray(d['w'], dtype=np.float64)
        self.u = np.asarray(d['u'], dtype=np.float64)
        self.N = np.asarray(d['N'], dtype=np.int32)
        self.W = np.asarray(d['W'], dtype=np.float64)
        self.B, self.Nmax = self.t.shape


def fake_get_module(max_h=8, fmad=False):
    class _M(object):
        def get_function(self, name):
            return lambda *a, **k: None
    return _M()


def fake_stage1(band, freqs_dev, H, block=128, module=None):
    return R.stage1_direct_sums_ref(band.t, band.w, band.u,
                                    np.asarray(freqs_dev), H)


def fake_stage2(sums, cn, sn, Wk_rows, accum, out=None, H=None,
                block=128, module=None):
    cn = np.asarray(cn, dtype=np.float64)
    sn = np.asarray(sn, dtype=np.float64)
    if H is None:
        H = len(cn)
    Hs = int(np.asarray(sums['C']).shape[-1])
    lead = np.asarray(sums['C']).shape[:-1]
    nrow = int(np.prod(lead))
    flat = {k: np.asarray(v).reshape((nrow,) +
                                     np.asarray(v).shape[len(lead):])
            for k, v in sums.items()}
    YMk, MMk, ACk = R.stage2_assemble_ref(flat, cn, sn, H=H)
    w = np.asarray(Wk_rows, dtype=np.float64)[:, None]
    if not accum or out is None:
        return (w * YMk, w * MMk, w * ACk)
    YM, MM, AC = out
    return (YM + w * YMk, MM + w * MMk, AC + w * ACk)


def fake_stage3(YM, MM, YY_rows, H, n_angles=None,
                n_newton=R.SCAN_NEWTON_STEPS, positive_amplitude=True,
                kcap=None, dip_rtol=R.SCAN_DIP_RTOL, block=128,
                module=None, row_chunk=None):
    return R.stage3_scan_polish_ref(
        np.asarray(YM), np.asarray(MM), np.asarray(YY_rows), H,
        n_angles=n_angles, n_newton=n_newton,
        positive_amplitude=positive_amplitude, kcap=kcap,
        dip_rtol=dip_rtol)


def fake_multiband_powers(band_data_list, templates, YY, freqs,
                          chunk=1024, n_newton=R.SCAN_NEWTON_STEPS,
                          positive_amplitude=True, Hs=None,
                          reuse_sums=True, fmad=False, block=128):
    band_arrays = []
    for b in band_data_list:
        band_arrays.append(dict(t=b.t, w=b.w, u=b.u, N=b.N, W=b.W,
                                YY=np.asarray(YY, dtype=np.float64),
                                ybar=np.zeros(b.B)))
    res = R.full_pipeline_ref(band_arrays, templates, freqs, chunk=chunk,
                              n_newton=n_newton,
                              positive_amplitude=positive_amplitude,
                              route_exact=False, Hs=Hs)
    return res


K.GpuBandData = FakeBandData
K.get_module = fake_get_module
K.gpu_stage1 = fake_stage1
K.gpu_stage2 = fake_stage2
K.gpu_stage3 = fake_stage3
K.gpu_multiband_powers = fake_multiband_powers

sys.path.insert(0, os.path.join(HERE, 'pod'))
import validate_gpu                                    # noqa: E402

if __name__ == '__main__':
    print('=== pod plumbing check: validate_gpu.py driven by the numpy '
          'mirrors (expect ALL PASS, ~zero diffs) ===')
    code = 0
    try:
        validate_gpu.main()
    except SystemExit as exc:
        code = int(exc.code or 0)
    # the results json slot belongs to REAL pod runs -- do not leave a
    # fake-local one lying around to be mistaken for pod output
    fake_json = os.path.join(HERE, 'pod', 'validate_gpu_results.json')
    if os.path.exists(fake_json):
        os.unlink(fake_json)
    sys.exit(code)
