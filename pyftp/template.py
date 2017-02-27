import numpy as np
from . import fast_template_periodogram as ftp


class Template(object):
    """Fourier-based Template Approximation

    The model is

    y(t) = sum_n( c_n cos(nwt) + s_n sin(nwt) )

    Parameters
    ----------
    c_n, s_n : array_like
        One-dimensional arrays of model coefficients
    """
    def __init__(self, c_n, s_n, template_id=None):
        self.c_n, self.s_n = np.broadcast_arrays(c_n, s_n)
        self.template_id = template_id
        if self.c_n.ndim != 1:
            raise ValueError("c_n and s_n must be one-dimensional")
        self._computed = {}

    @property
    def cn(self):
        return self.c_n

    @property
    def sn(self):
        return self.s_n

    @classmethod
    def from_sampled(cls, y, nharmonics=0.9, **kwargs):
        """Create a template from a regularly sampled function

        Parameters
        ----------
        y : array_like
            equally-spaced template. If N = len(y), then y[n] is the template
            evaluated at phase = n / N
        nharmonics : float or int
            If integer, specify the number of harmonics to use.
            If float between 0 and 1, then specify the relative variance to
            preserve in selecting the number of harmonics
        """
        yhat = np.fft.rfft(y)[1:]

        # automatically determine number of harmonics
        if 0 < nharmonics < 1:
            cuml_var = np.cumsum(abs(yhat) ** 2)
            ind = np.searchsorted(cuml_var, nharmonics * cuml_var[-1])
            nharmonics = np.clip(ind, 1, len(yhat))

        coeffs = 2 * yhat[:nharmonics] / len(y)

        c_n, s_n = coeffs.real, -coeffs.imag
        return cls(c_n, s_n, **kwargs)

    def precompute(self, force_recompute=False):
        if force_recompute:
            self._computed = {}
        # properties are computed by referencing them
        pvectors, ptensors = self.pvectors, self.ptensors

    @property
    def pvectors(self):
        if 'pvectors' not in self._computed:
            self._computed['pvectors'] =\
                ftp.get_polynomial_vectors(self.c_n, self.s_n, sgn=1)
        return self._computed['pvectors']

    @property
    def ptensors(self):
        if 'ptensors' not in self._computed:
            self._computed['ptensors'] =\
                ftp.compute_polynomial_tensors(*self.pvectors)
        return self._computed['ptensors']

    def __call__(self, phase):
        # evaluate the template
        phase = np.asarray(phase)[..., np.newaxis]
        n = np.arange(1, len(self.c_n) + 1)
        return (np.dot(np.cos(2 * np.pi * n * phase), self.c_n) +
                np.dot(np.sin(2 * np.pi * n * phase), self.s_n))
