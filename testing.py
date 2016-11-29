from time import time
import cPickle as pickle
import numpy as np
from math import *
import sys
import matplotlib.pyplot as plt 
from gatspy.periodic import RRLyraeTemplateModeler, LombScargleFast
import gatspy.datasets as datasets
import pyftp.rrlyrae as rrlm
from pyftp.modeler import FastTemplateModeler, Template
from gatspy.periodic.template_modeler import BaseTemplateModeler
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import pyftp.fast_template_periodogram as ftp


class GatspyTemplateModeler(BaseTemplateModeler):
	"""
	Convenience class for the gatspy BaseTemplateModeler
	"""
	def __init__(self, templates=None, **kwargs):
		self.ftp_templates = None

		if not templates is None:
			if isinstance(templates, dict):
				self.ftp_templates = templates.copy()
			elif isinstance(templates, list):
				self.ftp_templates = { t.template_id : t for t in templates }
			else:
				self.ftp_templates = { 0 : templates }
		BaseTemplateModeler.__init__(self, **kwargs)


	def _template_ids(self):
		return self.ftp_templates.keys()

	def _get_template_by_id(self, template_id):
		assert(template_id in self.ftp_templates)
		t = self.ftp_templates[template_id]

		return t.phase, t.y

rms = lambda y : sqrt(np.mean(np.power(y, 2)))
rror = lambda ym, y : rms(y - ym) / rms(y)


def set_model_templates(templates, stop=None, nharmonics=None):
	assert(not (stop is None and nharmonics is None))
	for template in templates:
		if nharmonics is None:
			template.stop = stop
		else:
			template.nharmonics = nharmonics

		template.precompute()


def accuracy_vs_stop(model, x, y, err, nharms=np.arange(1,10), nharm_answer=10, 
					use_gatspy=False, plot=True, label=None):

	p_ans = None
	p_ftps = []
	template_errors = []
	for i, nharm in enumerate(nharms):

		if 'loud' in model.params and model.params['loud']:
			print "H = ", nharm

		# set number of harmonics for each template
		set_model_templates(model.templates.values(), nharmonics=nharm)

		# Get template errors for each template
		template_errors.append([ t.rms_resid_over_rms for t in model.templates.values() ])

		# Run FTP
		model.fit(x, y, err)
		frq, p = model.periodogram()
		p_ftps.append(p)

		# possibly run gatspy
		if i == 0 and use_gatspy:
			periods = np.power(frq, -1)[::-1]
			gmodel = GatspyTemplateModeler(templates=model.templates)
			gmodel.fit(x, y, err)
			p_ans = gmodel.periodogram(periods)[::-1]

	if not use_gatspy:
		set_model_templates(model.templates.values(), nharmonics=nharm_answer)
		model.fit(x, y, err)
		frq, p_ans = model.periodogram()

	if not plot:
		return frq, template_errors, p_ftps, p_ans

	# PLOT
	f, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
	p_errs = [ rror(p, p_ans) for p in p_ftps ]
	t_errs = [ np.mean(terr) for terr in template_errors ]
	ax.plot(t_errs, p_errs, 'o')
	ax.set_xlabel('mean template error rms(yfit - y)/rms(y)')
	ax.set_ylabel('periodogram error')
	if use_gatspy: 
		ax.text(0.05, 0.95, 'Using gatspy as ground truth', 
									ha='left', va='top', fontsize=12, transform=ax.transAxes)
	else:
		ax.text(0.05, 0.95, 'Using H=%d as ground truth'%(nharm_answer), 
									ha='left', va='top', fontsize=12, transform=ax.transAxes)

	if not label is None: ax.set_title(label)
	for H, te, pe in zip(nharms, t_errs, p_errs):
		ax.text((1 - 0.05) * te , (1 + 0.05) *pe, 'H=%d'%(H), fontsize=9, ha='right', va='bottom')

	if not use_gatspy: ax.set_yscale('log')
	ax.set_xscale('log')

	ax2.plot(frq, p_ans, color='r', label='H=%d'%(nharm_answer))

	da = 0.5 / (len(template_errors) + 1)
	for i, (pe, p, h) in enumerate(zip(template_errors, p_ftps, nharms)):
		ax2.plot(frq, p, color='k', alpha=da * (i + 1), label="H = %d"%(h))

	ax2.set_xlabel('freq')
	ax2.set_ylabel('periodogram')
	ax2.legend(loc='best', fontsize=9)
	ax2.set_ylim(0, 1)
	ax2.set_xlim(0, max(frq))
	#plt.show()

	f.savefig('plots/accuracy_gt%s.png'%('gatspy' if use_gatspy else 'H%d'%(nharm_answer)))



def test_rrlyr_modeler(x, y, err, **kwargs):

	fmodel = rrlm.FastRRLyraeTemplateModeler(**kwargs)
	fmodel.fit(x, y, err)
	frq, p_ftp = fmodel.periodogram()

	gmodel = RRLyraeTemplateModeler(filts=kwargs['filts'])
	gmodel.fit(x, y, err)
	p_gats = gmodel.periodogram(np.power(frq, -1)[::-1])[::-1]

	return frq, p_ftp, p_gats


	
def generate_random_signal(n, sigma, freq=1.0, ttemp=None, ytemp=None):
	x = np.sort(np.random.rand(n))

	y = None
	if not ttemp is None:
		ttemp_ = list(ttemp[:])
		ytemp_ = list(ytemp[:])
		if all([ t < 1.0 for t in ttemp_ ]):
			ttemp_.append(1.0)
			ytemp_.append(ytemp_[0])
		tmodel = interp1d(ttemp_, ytemp_)
		y = tmodel(((x * freq)%1.0)) + sigma * np.random.normal(size=n)

	else:
		y = np.cos(freq * x) + sigma * np.random.normal(size=n)

	err = np.ones(n) * sigma

	return x, y, err

if __name__ == '__main__':
	N = 200
	ftp_template_id = '100r' # non-sinusoidal
	nharms = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
	nharm_answer = 10
	ofac = 10
	hfac = 1
	sigma=0.3
	freq = 50.0

	# Obtain template from RR Lyrae dataset
	rrl_templates = datasets.rrlyrae.fetch_rrlyrae_templates()
	
	Ttemp, Ytemp = rrl_templates.get_template(ftp_template_id)
	template = Template(phase=Ttemp, y=Ytemp)

	# build model
	model = FastTemplateModeler(ofac=ofac, hfac=hfac, loud=True)
	model.add_templates([ template ])


	x, y, err = generate_random_signal(N, sigma, freq=freq)
	#accuracy_vs_stop(model, x, y, err, nharms=nharms, use_gatspy=True, 
	#	                                        label="N=%d, sinusoid"%(N))
	#accuracy_vs_stop(model, x, y, err, nharms=nharms, use_gatspy=False, 
	#	             nharm_answer=nharm_answer, label="N=%d, sinusoid"%(N))

	template.nharmonics = 10
	template.precompute()
	model.fit(x, y, err)
	frq, p = model.periodogram()


	gmodel = GatspyTemplateModeler(templates=model.templates)
	gmodel.fit(x, y, err)
	pg = gmodel.periodogram(np.power(frq[::-1], -1))[::-1]

	f, (axp, axt) = plt.subplots(1, 2, figsize=(10, 5))
	delta=1E-4
	axp.scatter(np.array(pg) + delta, np.array(p) + delta, color='k', alpha=0.3)
	axp.plot(np.linspace(delta, 1), np.linspace(delta, 1), color='k', ls=':')
	axp.set_xlim(delta, 1)
	axp.set_ylim(delta, 1)
	axp.set_xlabel('Gatspy periodogram + %.1e'%(delta))
	axp.set_ylabel('FTP periodogram + %.1e'%(delta)) 
	axp.set_xscale('log')
	axp.set_yscale('log')
	axp.text(0.05, 0.95, "R=%.5f"%(pearsonr(pg, p)[0]), 
		transform=axp.transAxes, ha='left', va='top')

	axp.set_title("N=%d, sinusoid"%(N))
	template.add_plot_to_axis(axt)
	axt.set_xlabel('phase')
	axt.set_ylabel('y')
	axt.set_xlim(0, 1)
	axt.set_ylim(0, 1)
	#axt.legend(loc='best', fontsize=9)
	axt.set_title('ID = %s, H = %d'%(ftp_template_id, template.nharmonics))
	axt.text(0.5, 0.05, '$\\left<\\left(\\hat{T} - T\\right)^2\\right>^{1/2}'\
						'/\\left<T^2\\right>^{1/2}=%.5f$'%(template.rms_resid_over_rms), transform=axt.transAxes, ha='center', va='bottom')
	axt.invert_yaxis()

	f.savefig('plots/accuracy_corr_with_gatspy.png')
	plt.show()



