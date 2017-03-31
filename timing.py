import numpy as np
import matplotlib.pyplot as plt
from tempfit.modeler import FastTemplateModeler
from tempfit.template import Template
from tempfit.utils import weights 

import tempfit.periodogram as pdg
import tempfit.pseudo_poly as ppol
import tempfit.summations as tsums

import pstats
from time import time
import cProfile

rand = np.random.RandomState(42)

def get_data(n=30):
	t = np.sort(rand.rand(n))
	y = rand.randn(n)
	y_err = np.ones_like(y)

	return t, y, y_err

def get_template(nh=3):
	cn = 2 * rand.rand(nh) - 1
	sn = 2 * rand.rand(nh) - 1

	return Template(cn, sn)

def get_modeler(ndata, nh, precompute=True):

	t, y, yerr = get_data(n=ndata)
	template = get_template(nh=nh)
	if precompute:
		template.precompute()

	modeler = FastTemplateModeler(template=template)
	modeler.fit(t, y, yerr)

	return modeler



def wrap_timer(func, nfreqs=None):

	def wfunc(*args, **kwargs):
		t0 = time()
		rvals = func(*args, **kwargs)
		dt = time() - t0
		ftext = "" if nfreqs is None else "(%.3e s / freq)"%(dt/nfreqs)
		print(" %-30s: %.3e s %s"%(func.__name__, dt, ftext))
		return rvals
	return wfunc

def wrap_timer_avg(func, name=None):
	def wfunc(agen, kwgen, ngen, name=name):
		dts = []
		rvals = []
		name = func.__name__ if name is None else name
		for i in range(ngen):
			args = agen(i)
			kwargs = kwgen(i)

			t0 = time()
			rvals.append(func(*args, **kwargs))
			dts.append(time() - t0)

		print(" %-30s: %.3e s (%.3e +/- %.3e avg)"%(name, sum(dts), np.mean(dts), np.std(dts)))
		return rvals
	return wfunc


def timing(mod):
	w = weights(mod.dy)
	freqs = mod.autofrequency()
	nh = len(mod.template.c_n)
	ptensors = mod.template.ptensors

	direct_summations = wrap_timer(tsums.direct_summations, nfreqs=len(freqs))
	fast_summations   = wrap_timer(tsums.fast_summations, nfreqs=len(freqs))
	get_final_ppoly   = wrap_timer_avg(ppol.get_final_ppoly)
	get_final_ppoly_components = wrap_timer_avg(ppol.get_final_ppoly_components)
	get_final_roots_faster = wrap_timer_avg(ppol.get_final_roots_faster)
	compute_zeros     = wrap_timer_avg(ppol.compute_zeros)
	autopower         = wrap_timer(mod.autopower, nfreqs=len(freqs))


	# direct_summations
	dsums = direct_summations(mod.t, mod.y, w, freqs, nh)

	# fast summations
	fsums = fast_summations(mod.t, mod.y, w, freqs, nh)

	# compute_zeros
	agen = lambda i, fsums=fsums, ptensors=ptensors : [ ptensors, fsums[i] ]
	kgen = lambda i : { }
	#zeros = compute_zeros(agen, kgen, len(freqs))

	# get_final_ppoly (coefficients)
	pps    = get_final_ppoly(agen, kgen, len(freqs))

	# real_roots_pm (finding roots from coefficients)
	func = lambda pp : pp.real_roots_pm()
	
	real_roots_pm = wrap_timer_avg(func, name='real_roots_pm')
	
	agenrr = lambda i, pps=pps : [ pps[i] ]
	kgenrr = lambda i : {}

	real_roots_pm(agenrr, kgenrr, len(freqs))

	print("alternative root finding (bypasses PseudoPolynomial)")
	print("----------------------------------------------------")
	# get_final_ppoly_components (coefficients, no PP)
	ps_and_qs    = get_final_ppoly_components(agen, kgen, len(freqs))
	agenrf = lambda i, paq=ps_and_qs : paq[i]

	# get final roots faster (bypasses the PseudoPolynomial module)
	roots  = get_final_roots_faster(agenrf, kgen, len(freqs))

	print("----------------------------------------------------")
	# modeler.autopower()
	frqs, powers = autopower()

ndata = 300
nh    = 5
timing(get_modeler(ndata, nh))

