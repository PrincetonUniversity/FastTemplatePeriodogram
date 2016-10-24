import os
import sys
from time import time
from math import *
import numpy as np
import fast_template_periodogram as ftp
import gatspy.datasets.rrlyrae as rrl
from scipy.optimize import curve_fit
import cPickle as pickle

def LMfit(x, y, err, cn, sn, w, positive=True):
	""" fits a, b, c with Levenberg-Marquardt """

	ffunc = lambda X, *pars : ftp.fitfunc(X, positive, w, cn, sn, *pars)
	p0 = [ np.std(y), 0.0, np.mean(y) ]
	bounds = ([0, -1, -np.inf], [ np.inf, 1, np.inf])
	popt, pcov = curve_fit(ffunc, np.array(x, dtype=float), np.array(y, dtype=float), 
		                    sigma=np.array(err, dtype=float), p0=p0, 
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


def approximate_template(Tt, Yt, errfunc=rms_resid_over_rms, stop=1E-2):
	""" Fourier transforms template, returning the first H components """

	fft = np.fft.fft(Yt)
	cn, sn = zip(*[ (p.real/len(Tt), p.imag/len(Tt)) for i,p in enumerate(fft) \
		             if i > 0 ])

	h = 1
	while errfunc(cn[:h], sn[:h], Tt, Yt) > stop:
		h+=1

	return cn[:h], sn[:h]

def get_rrlyr_templates(template_fname=None, errfunc=rms_resid_over_rms, 
	                    stop=1E-2, filts='r'):
	"""
	read templates for a given filter(s) from 
	the gatspy.datasets.rrlyrae package, approximate each 
	template with the minimal number of harmonics such 
	that `errfunc(Cn, Sn, T, Y)` < `stop` and save values
	in a pickled file given by template_fname if template_fname
	is not None.
	"""
	# Obtain RR Lyrae templates
	templates = rrl.fetch_rrlyrae_templates()
	
	# Select the right ID's
	IDS = [ t for t in templates.ids if t[-1] in list(filts) ]

	# Get (phase, amplitude) data for each template
	Ts, Ys = zip(*[ templates.get_template(ID) for ID in IDS ])

	# Transform templates into harmonics and precompute
	# template-specific data
	all_polyvecs_polytens = []
	all_cn_sn = []
	if template_fname is None or not os.path.exists(template_fname):
		print "precomputing templates..."
		t0 = time()
		
		for i, (T, Y) in enumerate(zip(Ts, Ys)):
			print "  template ", i
			# The templates appear to be .... reversed?
			CN, SN = approximate_template(T, Y[::-1], stop=stop, errfunc=errfunc)
			#CN, SN = approximate_template(T, Y, stop=stop, errfunc=errfunc)
			print "  ", len(CN), " harmonics kept."

			pvectors_p = ftp.get_polynomial_vectors(CN, SN, sgn=  1)
			pvectors_n = ftp.get_polynomial_vectors(CN, SN, sgn= -1)
		
			polytens_p = ftp.compute_polynomial_tensors(*pvectors_p)
			polytens_n = ftp.compute_polynomial_tensors(*pvectors_n)

			all_polyvecs_polytens.append((pvectors_n, pvectors_p, 
				                          polytens_n, polytens_p))
			all_cn_sn.append((CN, SN))
		dt = time() - t0
		print dt, " seconds to precompute templates"

		if not template_fname is None:
			pickle.dump((all_cn_sn, all_polyvecs_polytens), 
			        open(template_fname, 'wb'))

	else:
		all_cn_sn, all_polyvecs_polytens \
		       = pickle.load(open(template_fname, 'rb'))

	return all_cn_sn, all_polyvecs_polytens


def RRLyrModeler(x, y, err, filts='r', loud=True, 
	              ofac=10, hfac=1, template_fname=None, 
	              errfunc=rms_resid_over_rms, stop=1E-2):
	"""
	RR Lyrae Template modeler

	Parameters
	----------
	x: np.ndarray, list
		independent variable (time)
	y: np.ndarray, list
		array of observations
	err: np.ndarray
		array of observation uncertainties
	filts: str (default: 'r')
		string containing one or more of 'ugriz'
	loud: boolean (default: True), optional
		print status
	ofac: float, optional (default: 10)
		oversampling factor -- higher values of ofac decrease 
		the frequency spacing (by increasing the size of the FFT)
	hfac: float, optional (default: 1)
		high-frequency factor -- higher values of hfac increase
		the maximum frequency of the periodogram at the 
		expense of larger frequency spacing.
	template_fname: str, optional (default: None)
		filename of pickle file to load/save the precomputed template
		values. If the file does not exist, one is created.
	errfunc: callable, optional (default: rms_resid_over_rms)
		A function returning some measure of error resulting
		from approximating the template with a given number 
		of harmonics
	stop: float, optional (default: 0.01)
		A stopping criterion. Once `errfunc` returns a number
		that is smaller than `stop`, the harmonics up to that point
		are kept. If not, another harmonic is added.
	
	Returns
	-------
	freqs: np.ndarray
		Array of frequencies corresponding to the periodogram values
	ftp: np.ndarray
		Array of periodogram values evaluated at each frequency

	"""
	all_cn_sn, all_polyvecs_polytens = \
		get_rrlyr_templates(template_fname=template_fname, 
			                 stop=stop, filts=filts, errfunc=errfunc)

	# Compute periodograms for each template
	all_ftps = []
	for i, ((CN, SN), (pvectors_n, pvectors_p, polytens_n, polytens_p)) \
	            in enumerate(zip(all_cn_sn, all_polyvecs_polytens)):

		if loud: print i + 1, "/", len(all_cn_sn)
		all_ftps.append(ftp.fastTemplatePeriodogram(x, y, err, CN, SN, 
			         ofac=ofac, hfac=hfac, polytens_p=polytens_p,
			         polytens_n=polytens_n, pvectors_p=pvectors_p,
			         pvectors_n=pvectors_n))	

	freqs, ftps = zip(*all_ftps)
	FREQS = freqs[0]

	# RR lyr periodogram is the maximum periodogram value at each frequency
	FTP = [ max([ f[i] for f in ftps ]) for i in range(len(FREQS)) ]

	return FREQS, FTP
