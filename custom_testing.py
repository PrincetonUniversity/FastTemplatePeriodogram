from pyftp.summations import direct_summations, fast_summations
from pyftp.utils import weights
from pyftp.template import Template
from pyftp.modeler import FastTemplateModeler, FastMultiTemplateModeler, TemplateModel
from pyftp.utils import ModelFitParams, weights
from pyftp.fast_template_periodogram import fit_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt


def template_function(phase,
                      c_n=[-0.181, -0.075, -0.020],
                      s_n=[-0.110, 0.000,  0.030]):
	n = 1 + np.arange(len(c_n))[:, np.newaxis]
	return (np.dot(c_n, np.cos(2 * np.pi * n * phase)) +
	        np.dot(s_n, np.sin(2 * np.pi * n * phase)))


def data(N=100, T=10, period=0.9, coeffs=(5, 10),
	     yerr=0.0001, rseed=150, phi0=0.0):

	t = T * rand.rand(N)
	y = coeffs[0] + coeffs[1] * template_function(t / period - phi0)
	y += yerr * rand.randn(N)
	return t, y, yerr * np.ones_like(y)

def template_model_data(temp, params, N=1000, T=10, period=0.9, yerr=0.001, rseed=150):
	model = TemplateModel(temp, frequency=(1./period), parameters=params)

	rand = np.random.RandomState(rseed)
	t = np.sort(T * rand.rand(N))

	y = model(t) + yerr * rand.randn(N)

	return t, y, yerr * np.ones_like(y)


def template():
    phase = np.linspace(0, 1, 100, endpoint=False)
    return phase, template_function(phase)

def get_frequencies(dat, samples_per_peak, nyquist_factor):
	t, y, yerr = dat
	df = 1. / (t.max() - t.min()) / samples_per_peak
	Nf = int(0.5 * samples_per_peak * nyquist_factor * len(t))
	return df * (1 + np.arange(Nf))

def test_compute_summations_is_accurate(nharmonics):
	samples_per_peak, nyquist_factor = 1, 1
	dat = data()
	t, y, yerr = dat
	w = weights(yerr)
	freqs = get_frequencies(dat, samples_per_peak, nyquist_factor)
	all_fast_sums =  fast_summations(t, y, w, freqs, nharmonics)

	all_slow_sums = [ direct_summations(t, y, w, freq, nharmonics) for freq in freqs ]

	return freqs, all_fast_sums, all_slow_sums


def truncate_template(phase, y, nharmonics):
	fft = np.fft.fft(y[::-1])
	c_n, s_n = zip(*[ (p.real/len(phase), p.imag/len(phase)) for i,p in enumerate(fft) \
	             if i > 0 and i <= int(nharmonics) ])

	return c_n, s_n

def test_template_model(temp, ntau = 150):
	temp.precompute()
	
	phi = np.linspace(0, 1, 100)
	freq = 1./0.9
	omega = 2 * np.pi * freq

	shift = 0.5
	f, ax = plt.subplots()
	ax.set_xlim(0, 1)
	ax.set_ylabel('template')
	ax.set_xlabel('phase')

	# plot of generated data
	data_line, = ax.plot([], [], 'o', color='k')

	# plot of the signal that generated the data
	signal_line, = ax.plot([], [], '-', color='g', label='signal')

	# plot of the "best_model" reported by FTP
	model_line, = ax.plot([], [], '-', color='r', label='model')
	
	####################
	# periodogram plot
	ax_pdg = f.add_axes([ 0.22, 0.65, 0.2, 0.2 ])

	ax_pdg.spines['top'].set_visible(False)
	ax_pdg.spines['right'].set_visible(False)

	
	ax_pdg.set_xlim(0, 1)
	ax_pdg.set_ylim(0.7, 1.2)

	ax_pdg.set_ylabel('$P(\\omega)$')
	ax_pdg.set_xlabel('$f\\tau_0$')
	#ax_pdg.axhline(0, ls=':', color='k', alpha=0.5)

	# plot corresponding to manually computing P(omega) for original_signal
	pdg_line_0, = ax_pdg.plot([], [], color='g')

	# plot corresponding to manually computing P(omega) for "best_model" returned by FTP
	pdg_line_m, = ax_pdg.plot([], [], color='r')

	# plot corresponding to the reported P(omega) from ftp
	pdg_line_pdg, = ax_pdg.plot([], [], color='k')

	######################################
	# plot of ftau(model) - ftau(signal)
	ax_taud = f.add_axes([ 0.55, 0.65, 0.2, 0.2 ])
	
	ax_taud.spines['top'].set_visible(False)
	ax_taud.spines['right'].set_visible(False)

	ax_taud.set_xlim(0, 1)
	ax_taud.set_ylim(-1, 1)
	ax_taud.axhline(0, ls='--', color='k')
	ax_taud.axhline(0.5, ls=':', color='0.5')
	ax_taud.axhline(-0.5, ls=':', color='0.5')

	ax_taud.set_ylabel('$f(\\tau_m - \\tau_0)$')
	ax_taud.set_xlabel('$f\\tau_0$')

	tau_diff, = ax_taud.plot([], [], color='k')


	# Lists for plot data
	pdg_line_0_data = []
	pdg_line_m_data = []
	pdg_line_pdg_data = []
	tau_data = []
	tau_diff_data = []

	def gen_tau(n=0, ntau=ntau):
		while n < ntau:
			yield (float(n % ntau) / ntau) / freq
			n += 1

	def gen_index(n=0, ntau=ntau):
		while n < ntau:
			yield n
			n += 1

	# precompute all data
	def init():
		datas, models, params, pdgs = [],[],[],[]
		for tau in gen_tau():

			b = np.cos(omega * tau)
			sgn = np.sign(np.sin(omega * tau))

			pars = ModelFitParams(a=1.0, c=0.0, b=b, sgn=sgn)

			t, y, yerr = template_model_data(temp, pars)

			# uses direct summations!
			p_ftp, best_fit_pars = fit_template(t, y, yerr, temp, freq, allow_negative_amplitudes=True)


			model_true = TemplateModel(temp, frequency=freq, parameters=pars)
			model_ftp = TemplateModel(temp, frequency=freq, parameters=best_fit_pars)

			y_best_model = model_ftp((phi - shift) / freq)
			y_true = model_true(phi / freq)

			models.append((y_true, y_best_model))
			datas.append((t, y, yerr))
			params.append((pars, best_fit_pars))

			w = weights(yerr)
			ybar = np.dot(w, y)
			chi2_0 = np.dot(w, (y - ybar)**2)
			chi2_m = np.dot(w, (y - model_ftp(t - shift / freq))**2)
			chi2_s = np.dot(w, (y - model_true(t))**2)

			p_model = 1 - chi2_m / chi2_0
			p_true = 1 - chi2_s / chi2_0

			pdgs.append((p_ftp, p_model, p_true))
		return datas, models, params, pdgs

	datas, models, params, pdgs = init()

	def run(index):

		pars, mpars = params[index]
		t, y, yerr = datas[index]
		y_true, y_best_model = models[index]
		p_pdg, p_model, p_true = pdgs[index]

		omtm = np.arccos(mpars.b)
		if mpars.sgn < 0:
			omtm = 2 * np.pi - omtm

		omt0 = np.arccos(pars.b)
		if pars.sgn < 0:
			omt0 = 2 * np.pi - omt0

		ftau_model = omtm / (2 * np.pi) - shift
		ftau_true = omt0 / (2 * np.pi)


		phase_data = (t * freq) % 1.0

		data_line.set_data(phase_data, y)
		signal_line.set_data(phi, y_true)
		model_line.set_data(phi, y_best_model)

		#ax.legend(loc='best')
		ax.set_title('$f\\tau_0=%.3f$ (%d); $f\\tau_m=%.3f$ (%d) [%04d/%04d]'%(ftau_true, pars.sgn, ftau_model, mpars.sgn, index+1, ntau))
		ax.legend(loc='best')
		
		# update data
		pdg_line_0_data.append(p_true)
		pdg_line_m_data.append(p_model)
		pdg_line_pdg_data.append(p_pdg)
		tau_data.append(ftau_true)
		tau_diff_data.append(ftau_model - ftau_true)

		# update plot with updated data
		pdg_line_0.set_data(tau_data, pdg_line_0_data)
		pdg_line_m.set_data(tau_data, pdg_line_m_data)
		pdg_line_pdg.set_data(tau_data, pdg_line_pdg_data)
		tau_diff.set_data(tau_data, tau_diff_data)

		# manually set ylims to something reasonable
		ymin = min([ min(y_true), min(y_best_model), min(y) ])
		ymax = max([ max(y_true), max(y_best_model), max(y) ])

		print(index)
		dy = ymax - ymin

		ax.set_ylim(ymin - 0.1 * dy, ymax + 0.8 * dy)


	ani = animation.FuncAnimation(f, run, frames=ntau, blit=False, interval=100,
                              repeat=False)
	ani.save('model_discrepancy_H%d_phi0_%.2f.mp4'%(len(temp.c_n), shift), writer='ffmpeg', fps= ntau / 10)
	f.savefig('last_frame_H%d_phi0_%.2f.png'%(len(temp.c_n), shift))

if __name__ == '__main__':
	nharmonics = 3
	phase, y_phase = template()
	c_n, s_n = truncate_template(phase, y_phase, nharmonics)
	temp = Template(c_n, s_n)

	test_template_model(temp, ntau=100)
	