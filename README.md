Fast Template Periodogram
=========================

John Hoffman, 2016
jah5@princeton.edu

Description
-----------
The Fast Template Periodogram extends the Generalised Lomb Scargle
periodogram ([Zechmeister and Kurster 2009](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:0901.2573]) 
for arbitrary (periodic) signal shapes. A template is first approximated
by a truncated Fourier series of length `H`. The Nonequispaced Fast Fourier Transform
[NFFT](https://www-user.tu-chemnitz.de/~potts/nfft/) is used
to efficiently compute frequency-dependent sums.

Because the FTP is a non-linear extension of the GLS, the zeros of 
a polynomial of order ~`6H` must be computed at each frequency.

The [gatspy](http://www.astroml.org/gatspy/) library has an implementation of
both single and multiband template fitting, however this implementation
uses non-linear least-squares fitting to compute the optimal parameters 
(amplitude, phase, constant offset) of the template fit at each frequency. That
process scales as `N*Nfreq`, where `N` is the number of observations and
`Nfreq` is the number of frequencies at which to calculate the periodogram.

However, the template periodogram, expressed as a truncated Fourier series,
can be derived exactly, with the caveat that the optimal solution is a root 
of a polynomial of order ~`6*H`.

This method then scales as `N * H * log(N * H)`, assuming `N` ~ `Nfreq`. 
For small problems (N~Nfreq <~ 200), the fast template periodogram is only marginally
faster than the slow `N*Nfreq` algorithms, and for very small problems may actually be slower.

However, for time series with large amounts of data (HATNet has ~10,000 observations
per source), the non-linear template modeler is computationally impractical (~28 hours of 
computation time per source), while the fast template periodogram takes about 25 minutes
in its current state.

Requirements
------------

* [pyNFFT](https://pypi.python.org/pypi/pyNFFT) is required, but this program is thorny to install.
	* Do NOT use `pip install pynfft`; this will almost definitely not work.
	* You need to install NFFT <= 3.2.4 (NOT the latest version)
	* use `./configure --enable-openmp` when installing NFFT
	* NFFT also requires [FFTW3](http://www.fftw.org)
	* You may have to manually add the directory containing NFFT `.h` files  to the `include_dirs` variable in the pyNFFT `setup.py` file.
* The [Scipy stack](http://www.scipy.org/install.html)
* [gatspy](http://www.astroml.org/gatspy/) 
	* RRLyrae modeler needs this to obtain templates
	* Used to check accuracy/performance of the FTP

Updates
-------
* Template periodogram python implementation now works

* Speed has been improved by:
	* computing template-specific quantities before-hand
	* more efficient root finding by computing coefficients of polynomial
		* However, memory requirements (and speed) scale as ~H ** 4
		* PseudoPolynomial class for bookkeeping and code simplification
		* This is still the slowest process for small cases N <~ 200

Timing
------
	
* CASE: 7 harmonics, 60 observations, 500 frequencies (len of ffts = `4 * H * Nf` and `2 * H * Nf`)
	* 3.1E-3 s / freq for root finding
		* roots for both positive/negative sin(omega*tau)
		* 7.5E-4 s for computing coefficients of "PseudoPolynomial"
		* 1.5E-4 s for computing final polynomial
		* 5.9E-4 s for finding roots of polynomial
	* 5.9E-4 s / freq to compute amplitudes at each real zero
		* To enforce amplitude > 0 (flipping the template over isn't allowed)
	* 2.6E-4 s / freq to evaluate periodogram at each real zero for which A > 0
	* 2.0E-4 s / freq for summations

**Compared with gatspy** 

![timing](timing/timing.png "Timing compared to gatspy RRLyraeTemplateModeler")

NOTES
-----
* For portability and eventual coding in C (and later CUDA), it might
  make sense to do root-finding directly with our own implementation.
	* May also be faster -- np.roots computes eigenvalues of 
          companion matrix but maybe there are more optimal methods in our case?
	
* For testing against the gatspy template periodogram, there's an RR Lyrae modeler 
  included in FastTemplatePeriodgram.py that uses the same templates as the gatspy
  RRLyrae modeler.
	* Our implementation is (now) generally faster than the gatspy RRLyrae 
          template modeler for small cases and when using a reasonable number of 
          harmonics (~6ish). Not by much, however.

