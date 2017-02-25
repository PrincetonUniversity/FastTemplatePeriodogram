# Fast Template Periodogram [![build status](http://img.shields.io/travis/PrincetonUniversity/FastTemplatePeriodogram/master.svg?style=flat)](https://travis-ci.org/PrincetonUniversity/FastTemplatePeriodogram) [![codecov.io](https://codecov.io/gh/PrincetonUniversity/FastTemplatePeriodogram/coverage.svg?branch=master)](https://codecov.io/gh/PrincetonUniversity/FastTemplatePeriodogram)

John Hoffman

Jake Vanderplas

(c) 2016

Description
-----------

![examples](plots/templates_and_periodograms.png "Examples")

The Fast Template Periodogram extends the Generalized Lomb-Scargle
periodogram ([Zechmeister and Kurster 2009](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:0901.2573])) 
for arbitrary (periodic) signal shapes. A template is first approximated
by a truncated Fourier series of length `H`. The Non-equispaced Fast Fourier Transform
[NFFT](https://www-user.tu-chemnitz.de/~potts/nfft/) is used
to efficiently compute frequency-dependent sums.

Because the FTP is a non-linear extension of the GLS, the zeros of 
a polynomial of order `~6H` must be computed at each frequency.

The [gatspy](http://www.astroml.org/gatspy/) library has an implementation of
both single and multiband template fitting, however this implementation
uses non-linear least-squares fitting to compute the optimal parameters 
(amplitude, phase, constant offset) of the template fit at each frequency. That
process scales as `N_obs*N_f`, where `N` is the number of observations and
`N_f` is the number of frequencies at which to calculate the periodogram.

This process is extremely slow. [Sesar et al. (2016)](https://arxiv.org/abs/1611.08596) applied a similar
template fitting procedure to multiband Pan-STARRS photometry and found that
(1) template fitting was significantly more accurate for estimating periods
of RR Lyrae stars, but that (2) it required a substantial amount of 
computational resources to perform these fits.

However, if the templates are sufficiently smooth (or can be adequately 
approximated by a sufficiently smooth template) the template can be
represented by a short truncated Fourier series of length `H`. Using this 
representation, the optimal parameters (amplitude, phase, offset) 
of the template fit can then be found exactly after finding the roots of 
a polynomial at each trial frequency.

The coefficients of these polynomials involve sums that can be efficiently
computed with (non-equispaced) fast Fourier transforms. These sums
can be computed in `HN_f log(HN_f)` time.

In its current state, the root-finding procedure is the rate limiting step.
This unfortunately means that for now the fast template periodogram scales as 
`N_f*(H^4)`. We are working to reduce the computation time so that the entire 
procedure scales as `HN_f log(HN_f)` for reasonable values of `H` (`< 10`).

However, even for small cases where `H=6` and `N_obs=10`, this procedure is 
about twice as fast as the `gatspy` template modeler. And, the speedup over
`gatspy` grows linearly with `N_obs`! 


How is this different than the multi-harmonic periodogram?
----------------------------------------------------------


The multi-harmonic periodogram ([Schwarzenberg-Czerny (1996)](http://iopscience.iop.org/article/10.1086/309985/meta)) is another 
extension of Lomb-Scargle that fits a truncated Fourier series to the data 
at each trial frequency. This is nice if you have a strong non-sinusoidal signal 
and a large dataset. This algorithm can also be made to scale as
`HN_f logHN_f` ([Palmer 2009](http://iopscience.iop.org/article/10.1088/0004-637X/695/1/496/meta)).

However, the multi-harmonic periodogram is fundamentally different than template fitting. 
In template fitting, the relative amplitudes and phases of the Fourier series are *fixed*. 
In a multi-harmonic periodogram, the relative amplitudes and phases of the Fourier series are 
*free parameters*. These extra free parameters mean that (1) you need a larger
number of observations `N_obs` to reach the same signal to noise, and (2) you are
more likely to detect a multiple of the true frequency. For a discussion of this
effect, possible remedies with Tikhonov regularization, and an illuminating review
of periodograms in general, see [Vanderplas et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...812...18V).

Requirements
------------

* [pyNFFT](https://pypi.python.org/pypi/pyNFFT) is required, but this program is thorny to install.
	* Do NOT use `pip install pynfft`; this will almost definitely not work.
	* You need to install [NFFT](https://www-user.tu-chemnitz.de/~potts/nfft/) <= 3.2.4 (NOT the latest version)
	* use `./configure --enable-openmp` when installing NFFT
	* NFFT also requires [FFTW3](http://www.fftw.org)
	* You may have to manually add the directory containing NFFT `.h` files to the `include_dirs` variable in the pyNFFT `setup.py` file.
* The [Scipy stack](http://www.scipy.org/install.html)
* [gatspy](http://www.astroml.org/gatspy/) 
	* RRLyrae modeler needs this to obtain templates
	* Used to check accuracy/performance of the FTP

Installation
------------
These instructions assume a `*nix` operating system (i.e. Mac, Linux, BSD, etc.). 

#### If you have a **Mac** operating system

* I have not tested the default `clang` compilers; I would highly recommend 
  installing [macports]() and installing `gcc` compilers.
	`sudo port install gcc`
* After this, you may need to 'select' the gcc compilers: run 
    `sudo port select --list gcc`
  then pick the option that is not "none" by running 
    `sudo port select --set gcc mp-gcc6`
  where `mp-gcc6` is the option besides `none` (it may be different if you 
  install another version).
* If you run into any other trouble or find other dependencies, please let us know!

#### Installing FFTW3

* First download the FFTW3 [source](http://www.fftw.org), (the latest version should be fine, I have `3.3.5` on my machine)
* Unzip the downloaded `.tar.gz` file
* `./configure --enable-openmp --enable-threads` from inside the directory
* `sudo make install`

#### Installing NFFT

* Download [NFFT](https://www-user.tu-chemnitz.de/~potts/nfft/) **version <= 3.2.4** (NOT the latest version)
* unzip `.tar.gz` file
* `./configure --enable-openmp --with-fftw3-includedir=/usr/local/include --with-fftw3-libdir=/usr/local/lib`
	* optionally, use `--with-window=gaussian`, which should be faster (I haven't actually tested this).
* `sudo make install`

#### Installing pyNFFT
* `pip download pynfft`
* unzip `.tar.gz` file
* open the `setup.py` file in a text editor, add `/usr/local/include` to the `include_dirs` variable; i.e., change
	```python
	# Define utility functions to build the extensions
	def get_common_extension_args():
	    import numpy
	    common_extension_args = dict(
	        libraries=['nfft3_threads', 'nfft3', 'fftw3_threads', 'fftw3', 'm'],
	        library_dirs=[],
	        include_dirs=[numpy.get_include()], #THIS LINE
	```

	to 

	```python
	# Define utility functions to build the extensions
	def get_common_extension_args():
	    import numpy
	    common_extension_args = dict(
	        libraries=['nfft3_threads', 'nfft3', 'fftw3_threads', 'fftw3', 'm'],
	        library_dirs=[],
	        include_dirs=[numpy.get_include(), '/usr/local/include'], #added /usr/local/include
	```

* then `python setup.py install`.

#### Installing gatspy
* `pip install gatspy` should work!

#### Installing this code
* `git clone https://github.com/PrincetonUniversity/FastTemplatePeriodogram.git`
* Change into the newly created `FastTemplatePeriodogram` directory
* `python setup.py install`

Example usage
-------------

```python
from pyftp import modeler
import numpy as np

# define your template by its Fourier coefficients
cn = np.array([ 1.0, 0.5, 0.2 ])
sn = np.array([ 1.0, -0.2, 0.5 ])

# create a Template object
template = modeler.Template(cn=cn, sn=sn)

# Precompute some quantities for speed
template.precompute()

# create a FastTemplateModeler
model = modeler.FastTemplateModeler()

# add the template(s) to your modeler
model.add_templates([ template ])

# get some data
t, mag, err = get_your_data()

# feed the data to the modeler
model.fit(t, mag, err)

# get your template periodogram!
# ofac -- the oversampling factor: df = 1 / (ofac * (max(t) - min(t)))
# hfac -- the nyquist factor: f_max = hfac * N_obs / (max(t) - min(t))
freqs, periodogram = model.periodogram(ofac=20, hfac=1)

# What are the parameters of the best fit?
template, params = model.get_best_model()
```

There is also a built-in RR Lyrae modeler that pulls RR Lyrae templates 
from Gatspy (templates are from [Sesar et al. (2010)](http://iopscience.iop.org/article/10.1088/0004-637X/708/1/717/meta)).

```python
from pyftp import rrlyrae

# create a FastTemplateModeler
model = rrlyrae.FastRRLyraeTemplateModeler(filts='r')

# get some data
t, mag, err = get_your_data()

# feed the data to the modeler
model.fit(t, mag, err)

# get your template periodogram!
freqs, periodogram = model.periodogram(ofac=20, hfac=1)

```

 
Updates
-------

* See the [issues](https://github.com/PrincetonUniversity/FastTemplatePeriodogram/issues) 
section for known bugs! You can also submit bugs through this interface.

Timing
------

![timing](plots/timing_vs_ndata.png "Timing compared to gatspy")

The Fast Template Periodogram seems to do better than Gatspy
for virtually all reasonable cases (reasonable meaning a small-ish
number of harmonics are needed to accurately approximate the template,
small-ish meaning less than about 10).

It may be surprising that FTP appears to scale as `NH`, instead of
`NH log NH`, but that's because the NFFT is not the limiting factor (yet).
Most of the computation time is spent calculating polynomial coefficients,
and this computation scales as roughly `NH^4`. 

![timingnh](plots/timing_vs_nharm.png "Timing vs harmonics")

The FTP scales sub-linearly to linearly with the number of harmonics `H`
for `H < 10`, and for larger number of harmonics scales as `H^4`. This
is the main limitation of FTP.


Accuracy
--------

Compared with the Gatspy template modeler, the FTP provides improved accuracy as well as speed. 
For large values of `p(freq)`, the FTP correlates strongly with the Gatspy template algorithm; however,
since Gatspy uses non-linear function fitting (Levenberg-Marquardt), the predicted value for
`p(freq)` may not be optimal if the data is poorly modeled by the template. FTP, on the other 
hand, solves for the optimal solution directly, and thus tends to find equally good or 
better solutions when `p(freq)` is small.

![corrwithgats](plots/correlation_with_gatspy.png "Correlation to gatspy")
![accuracy](plots/correlation_with_large_H.png "How many harmonics do we need?")

For some frequencies, the Gatspy modeler finds no improvement over a constant fit 
(`p_gatspy(freq) = 0`). However, for these frequencies, the FTP consistently finds better 
solutions.

At frequencies where the template models the data at least moderately well (`p(freq) ~> 0.01`),
the Gatspy modeler and the FTP are in good agreement.

Assuming, then, that the FTP is indeed producing the "correct" periodogram, we can then
ask how many harmonics we must use in order to achieve an estimate of the periodogram to
a given accuracy.

TODO
----

* Extending this to a multiband template periodogram is a top priority after fixing bugs and
  providing adequate documentation!
* Unit testing
* Improve performance!
