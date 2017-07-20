Fast Template Periodogram
=========================

.. image:: http://img.shields.io/travis/PrincetonUniversity/FastTemplatePeriodogram/master.svg?style=flat
	:target: https://travis-ci.org/PrincetonUniversity/FastTemplatePeriodogram

.. image:: https://codecov.io/gh/PrincetonUniversity/FastTemplatePeriodogram/coverage.svg?branch=master
	:target: https://codecov.io/gh/PrincetonUniversity/FastTemplatePeriodogram

:Authors:
	John Hoffman (mailto:jah5@princeton.edu)
	Jake Vanderplas (mailto:jakevdp@uw.edu)

:Version:
	0.9.5.dev

Check out the `Scipy 2017 talk <https://www.youtube.com/watch?v=7STeeVnfYFM>`_

Description
-----------

.. image:: plots/templates_and_periodograms.png

The Fast Template Periodogram extends the Lomb-Scargle
periodogram ([Barning1963]_, [Vanicek1971]_, [Lomb1976]_, [Scargle1982]_, [ZechmeisterKurster2009]_) for arbitrary (periodic) signal shapes. It
naturally handles non-uniformly sampled time-series data.

:Template:
	Periodic signal shape, expressed as a truncated Fourier series of length ``H``.

:Periodogram:
	Least-squares estimator of the power spectral density (Lomb-Scargle); for more
	general models, proportional to the "goodness of fit" statistic for the best-fit
	model at that frequency. See [VanderPlas2017]_ for more details.

Uses the the `non-equispaced Fast Fourier Transform <https://www-user.tu-chemnitz.de/~potts/nfft>`_ to efficiently compute frequency-dependent sums.

The ``ftperiodogram`` library is complete with API documentation and consistency
checks using ``py.test``.


Installing
----------

**THIS IS NOT TRUE YET!**

As long as you have ``scipy`` and ``numpy`` installed, you should be able to run
``pip install ftperiodogram`` and everything should work fine.

If this doesn't work, consult the instructions in ``CONDA_INSTALL.md`` for installing ``ftperiodogram`` and its dependencies with with
`conda <https://www.continuum.io/downloads>`_.

Examples
--------

See the ``Examples.ipynb`` located in the ``notebooks/`` directory.

To run this notebook, use the ``jupyter notebook`` command from
inside the ``notebooks/`` directory::

	$ cd notebooks/
	$ jupyter notebook


Updates
-------

* See the `issues <https://github.com/PrincetonUniversity/FastTemplatePeriodogram/issues>`_section for known bugs! You can also submit bugs through this interface.


More information
================

Previous implementations
------------------------

The `gatspy <http://www.astroml.org/gatspy/>`_ library has an implementation of
both single and multiband template fitting, however this implementation
uses non-linear least-squares fitting to compute the optimal parameters
(amplitude, phase, constant offset) of the template fit at each frequency. That
process scales as ``N_obs*N_f``, where ``N`` is the number of observations and
``N_f`` is the number of frequencies at which to calculate the periodogram.

This is more or less the procedure used in [Sesar2017]_ to perform
template fits to Pan-STARRS photometry, however they used a more sophisticated
multiband model that locked the phases, amplitudes and
offsets of all bands together. They found that template fitting was significantly more accurate for estimating periods of RR Lyrae stars, but the computational resources
needed for these fits were enormous (~30 minutes per object per CPU core).

How does the fast template periodogram improve things?
------------------------------------------------------

By rederiving periodic template fitting (or periodic matched filter analysis)
in the context of least-squares spectral analysis, we found a significantly
better way to perform these fits. Details will be presented in a paper
(Hoffman *et al.* 2017, *in prep*), but the important part is you can reduce
the non-linearity of the problem to the following:

- Finding the zeros of an order ``6H-1`` complex polynomial at each trial frequency
	- This is done via the ``numpy.polynomial`` library, which performs singular-value decomposition on the polynomial "companion matrix", and scales as ``O(H^3)``.
- Computing the coefficients of these polynomials for all trial frequencies simultaneously by leveraging the non-equispaced fast Fourier transform, a process that scales as ``O(HN_f log(HN_f))``.

This provides two advantages:

:Improved computational speed and scaling:
	.. image:: plots/timing_vs_ndata_const_freq.png
	Speed comparison for a test case using a constant
	number of trial frequencies but varying the number
	of observations.

:Numerically stable and accurate:
	.. image:: plots/correlation_with_nonlinopt.png
	Accuracy comparison between the fast template periodogram
	and a ``gatspy``-like method that uses the ``scipy.optimize.minimize``
	function to find the optimal phase shift parameter. The minimization
	method is given 10 random starting values and the best result is kept.
	Though in most cases the truly optimal solution is found, in many cases
	a sub-optimal solution is chosen instead (i.e. only a locally optimal
	solution was chosen).


How is this different than the multi-harmonic periodogram?
----------------------------------------------------------

The multi-harmonic periodogram ([Bretthorst1988]_,[SchwarzenbergCzerny1996]_) is another
extension of Lomb-Scargle that fits a truncated Fourier series to the data
at each trial frequency. This algorithm can also be made to scale as
``HN_f logHN_f`` [Palmer2009]_.

However, the multi-harmonic periodogram is fundamentally different than template fitting.
In template fitting, the relative amplitudes and phases of the Fourier series are *fixed*.
In a multi-harmonic periodogram, the relative amplitudes and phases of the Fourier series are *free parameters*.

The multiharmonic periodogram is more flexible than the template periodogram, but less
sensitive to a given signal. If you're hoping to find a non-sinusoidal signal with an
unknown shape, it might make more sense to use a multi-harmonic periodogram.]

For more discussion of the multiharmonic periodogram and related extensions, see [VanderPlas_etal_2015]_ and [VanderPlas2017]_.

TODO
----

* Multi-band extensions
* Speed improvements


References
----------


.. [ZechmeisterKurster2009] `Paper <http://adsabs.harvard.edu/abs/2009A%26A...496..577Z>`_

.. [Lomb1976] `Least-squares frequency analysis of unequally spaced data <http://adsabs.harvard.edu/abs/1976Ap%26SS..39..447L>`_

.. [Scargle1982] `Studies in astronomical time series analysis. II - Statistical aspects of spectral analysis of unevenly spaced data <http://adsabs.harvard.edu/abs/1982ApJ...263..835S>`_

.. [Barning1963] `The numerical analysis of the light-curve of 12 Lacertae <http://adsabs.harvard.edu/abs/1963BAN....17...22B>`_

.. [Vanicek1971] `Further Development and Properties of the Spectral Analysis by Least-Squares <http://adsabs.harvard.edu/abs/1971Ap%26SS..12...10V>`_

.. [VanderPlas2017] `Understanding the Lomb-Scargle Periodogram <https://arxiv.org/abs/1703.09824>`_

.. [Sesar2017] https://arxiv.org/abs/1611.08596

.. [Bretthorst1988] https://link.springer.com/book/10.1007%2F978-1-4684-9399-3

.. [SchwarzenbergCzerny1996] http://iopscience.iop.org/article/10.1086/309985/meta

.. [Palmer2009] http://iopscience.iop.org/article/10.1088/0004-637X/695/1/496/meta

.. [VanderPlas_etal_2015] http://adsabs.harvard.edu/abs/2015ApJ...812...18V