import os
import sys
from time import time
from math import *
import numpy as np
import fast_template_periodogram as ftp
from scipy.optimize import curve_fit
import cPickle as pickle

def LMfit(x, y, err, cn, sn, w, positive=True):
	""" fits a, b, c with Levenberg-Marquardt """

	ffunc = lambda X, *pars : ftp.fitfunc(X, positive, w, cn, sn, *pars)
	p0 = [ np.std(y), 0.0, np.mean(y) ]
	bounds = ([0, -1, -np.inf], [ np.inf, 1, np.inf])
	popt, pcov = curve_fit(ffunc, x, y, sigma=err, p0=p0, 
                            absolute_sigma=True, bounds=bounds, 
                            method='trf')
	a, b, c = popt

	return a, b, c

def rms_resid_over_rms(CN, SN, Tt, Yt):
	# This is fairly slow; is there a better way to get best fit pars?
	a, b, c = LMfit(Tt, Yt, np.ones(len(Tt))*0.0001, CN, SN, 2 * np.pi, positive=True)
	Ym = ftp.fitfunc(Tt, True, 2 * np.pi, CN, SN, a, b, c)

	S = sqrt(np.mean(np.power(Yt, 2)))

	Rp = sqrt(np.mean(np.power(Ym - Yt, 2))) / S

	a, b, c = LMfit(Tt, Yt, np.ones(len(Tt))*0.0001, CN, SN, 2 * np.pi, positive=False)
	Ym = ftp.fitfunc(Tt, False, 2 * np.pi, CN, SN, a, b, c)

	Rn = sqrt(np.mean(np.power(Ym - Yt, 2))) / S
	return min([ Rn, Rp ])

rms = lambda x : sqrt(np.mean(np.power(x, 2)))

def match_up_truncated_template(CN, SN, Tt, Yt):
	Ym = ftp.fitfunc(Tt, True, 2 * np.pi, CN, SN, 2.0, 0.0, 0.0)

	# Align the maxima of truncated and full templates
	di = np.argmax(Ym) - np.argmax(Yt)

	# Add some 'wiggle room', since maxima may be offset by 1
	Ym = [ np.array([ Ym[(j + (di + k))%len(Ym)] for j in range(len(Ym)) ]) for k in [ -1, 0, 1 ] ]

	# Align the heights of the truncated and full templates
	Ym = [ Y + (Yt[0] - Y[0]) for Y in Ym ]

	# Keep the best fit
	return Ym[np.argmin( [ rms(Y - Yt) for Y in Ym ] )]

def rms_resid_over_rms_fast(CN, SN, Tt, Yt):
	Ym = match_up_truncated_template(CN, SN, Tt, Yt)
	return rms(Yt - Ym) / rms(Yt)

def approximate_template(Tt, Yt, errfunc=rms_resid_over_rms, stop=1E-2, nharmonics=None):
	""" Fourier transforms template, returning the first H components """

	fft = np.fft.fft(Yt[::-1])
	
	cn, sn = None, None
	if not nharmonics is None and int(nharmonics) > 0:
		cn, sn = zip(*[ (p.real/len(Tt), p.imag/len(Tt)) for i,p in enumerate(fft) \
		             if i > 0 and i <= int(nharmonics) ])
		
	else:

		cn, sn = zip(*[ (p.real/len(Tt), p.imag/len(Tt)) for i,p in enumerate(fft) \
		             if i > 0 ])

		h = 1
		while errfunc(cn[:h], sn[:h], Tt, Yt) > stop:
			#print "h -> ", h
			h+=1

		cn, sn = cn[:h], sn[:h]
	return cn, sn

class Template(object):
	"""
	Template class
	
	y(t) = sum[n]( c[n]cos(nwt) + s[n]sin(nwt) )

	Parameters
	----------
	cn: array-like, optional
		Truncated Fourier coefficients (cosine) 
	sn: array-like, optional
		Truncated Fourier coefficients (sine) 
	phase: array-like, optional
		phase-values, must contain floating point numbers in [0,1]
	y: array-like, optional
		amplitude of template at each phase value
	stop: float, optional (default: 2E-2)
		will pick minimum number of harmonics such that 
		rms(trunc(template) - template) / rms(template) < stop
	nharmonics: None or int, optional (default: None)
		Keep a constant number of harmonics
	fname: str, optional
		Filename to load/save template
	errfunc: callable, optional (default: rms_resid_over_rms)
		A function returning some measure of error resulting
		from approximating the template with a given number 
		of harmonics
	template_id: str, optional
		Name of template

	"""
	def __init__(self, cn=None, sn=None, phase=None, y=None, 
		               stop=2E-2, nharmonics=None, fname=None,
				       errfunc=rms_resid_over_rms, template_id=None):

		self.phase = phase
		self.y = y

		self.fname = fname
		self.stop = stop
		self.nharmonics = nharmonics
		self.errfunc = errfunc
		self.cn = None
		self.sn = None
		self.pvectors = None
		self.ptensors = None
		self.template_id = template_id
		self.best_fit_y = None


	def is_saved(self):

		return (not self.fname is None and os.path.exists(self.fname))

	def precompute(self):
		self.cn, self.sn = approximate_template(self.phase, self.y, 
			                            stop=self.stop, errfunc=self.errfunc, 
										nharmonics=self.nharmonics)


		self.nharmonics = len(self.cn)

		self.pvectors = ftp.get_polynomial_vectors(self.cn, self.sn, sgn=1)

		self.ptensors = ftp.compute_polynomial_tensors(*self.pvectors)

		self.best_fit_y = match_up_truncated_template(self.cn, self.sn, self.phase, self.y)

		self.rms_resid_over_rms = rms(self.best_fit_y - self.y) / rms(self.y)

		return self

	def load(self, fname=None):
		fn = fname if not fname is None else self.fname
		self.__dict__.update(pickle.load(open(fn, 'rb')))

	def save(self, fname=None):
		fn = fname if not fname is None else self.fname
		pickle.dump(self.__dict__, open(fn, 'wb'))

	def add_plot_to_axis(self, ax):
		ax.plot(self.phase, self.y, color='k', label="original")
		ax.plot(self.phase, self.best_fit_y, color='r', 
			      label="best-fit approx; rms_resid/rms=%.3e, H=%d"%(self.rms_resid_over_rms, self.nharmonics))		

	def plot(self, plt):
		f, ax = plt.subplots()
		self.add_plot_to_axis(ax)
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)
		ax.set_xlabel('phase')
		ax.set_ylabel('y')
		ax.set_title('"%s", stop = %.3e, H = %d'%(self.template_id, 
			                                 self.stop, self.nharmonics))
		ax.legend(loc='best', fontsize=9)
		plt.show()
		plt.close(f)

class FastTemplateModeler(object):

	"""
	Base class for template modelers

	Parameters
	----------

	loud: boolean (default: True), optional
		print status
	ofac: float, optional (default: 10)
		oversampling factor -- higher values of ofac decrease 
		the frequency spacing (by increasing the size of the FFT)
	hfac: float, optional (default: 1)
		high-frequency factor -- higher values of hfac increase
		the maximum frequency of the periodogram at the 
		expense of larger frequency spacing.
	errfunc: callable, optional (default: rms_resid_over_rms)
		A function returning some measure of error resulting
		from approximating the template with a given number 
		of harmonics
	stop: float, optional (default: 0.01)
		A stopping criterion. Once `errfunc` returns a number
		that is smaller than `stop`, the harmonics up to that point
		are kept. If not, another harmonic is added.
	nharmonics: None or int, optional (default: None)
		Number of harmonics to keep if a constant number of harmonics
		is desired

	"""
	def __init__(self, **kwargs):
		self.params = { key : value for key, value in kwargs.iteritems() }
		self.templates = {}
		self.omegas = None
		self.summations = None
		self.YY = None
		self.max_harm = 0
		self.w = None
		self.ybar = None

	def _get_template_by_id(self, template_id):
		assert(template_id in self.templates)
		return self.templates[template_id]

	def _template_ids(self):
		return self.templates.keys()

	def add_templates(self, templates, template_ids=None):
		
		self.templates.update(
					{ TEMP.template_id if not TEMP.template_id is None \
					    else (i if template_ids is None \
					    	      else template_ids[i]) \
					        : TEMP for i, TEMP in enumerate(templates) })

		self.max_harm = max([ T.nharmonics for T in self.templates.values() ])
		return self

	def remove_templates(self, template_ids):
		for ID in template_ids:
			assert ID in self.templates
			del self.templates[ID]
		return self

	def set_params(self, **new_params):
		self.params.update(new_params)
		return self

	def fit(self, x, y, err):
		"""
		Parameters
		----------
		x: np.ndarray, list
			independent variable (time)
		y: np.ndarray, list
			array of observations
		err: np.ndarray
			array of observation uncertainties
		"""
		self.x = x
		self.y = y
		self.err = err
		self.summations = None
		return self

	def compute_sums(self):
		
		self.omegas, self.summations, \
		self.YY, self.w, self.ybar = \
			ftp.compute_summations(self.x, self.y, self.err, self.max_harm, 
								ofac=self.params['ofac'], hfac=self.params['hfac'])

		return self


		
	def periodogram(self, **kwargs):
		self.params.update(kwargs)

		#if self.summations is None:
		#	self.compute_sums()
		loud = False if not 'loud' in self.params else self.params['loud']
		all_ftps = []
		for template_id, template in self.templates.iteritems():
			args = (self.x, self.y, self.err, template.cn, template.sn)
			kwargs = dict(ofac       = self.params['ofac'], 
						  hfac       = self.params['hfac'], 
						  ptensors   = template.ptensors, 
						  pvectors   = template.pvectors, 
						  omegas     = self.omegas, 
						  summations = self.summations, 
						  YY         = self.YY, 
						  ybar       = self.ybar, 
						  w          = self.w,
						  loud       = loud)
			all_ftps.append(ftp.fastTemplatePeriodogram(*args, **kwargs))	

		freqs, ftps = zip(*all_ftps)
		FREQS = freqs[0]

		# Periodogram is the maximum periodogram value at each frequency
		return FREQS,  np.array([ max([ f[i] for f in ftps ]) for i in range(len(FREQS)) ])


