from time import time
import numpy as np
from math import * 
from gatspy.periodic import RRLyraeTemplateModeler
import pyftp.rrlyr_modeler as rrlm

ofac = 5
hfac = 1
stop = 2E-2

Ns = [ 15, 30, 40, 60, 80, 90, 120  ]
for N in Ns:

	# generate signal
	freq = 3.
	x = np.sort(np.random.rand(N))
	err = np.absolute(np.random.normal(loc=0.2, scale=0.1, size=N))
	y = np.cos(freq * x) + np.sin(3 * freq * x - 0.44) \
	            + np.array([ np.random.normal(loc=0, scale=s) for s in err ])
	
	# Time the FTP rrlyrae modeler
	t0 = time()
	freqs, per = rrlm.RRLyrModeler(x, y, err, filts='r', loud=False, 
                          ofac=ofac, hfac=hfac, stop=stop, errfunc=rrlm.rms_resid_over_rms,
                          template_fname='../saved_templates/templates_RR_stop%.2e.pkl'%(stop))
	dt_ftp = time() - t0

	# convert freqs to periods
	periods = np.power(freqs[::-1], -1)

	# time the gatspy RR Lyr modeler
	t0 = time()
	model = RRLyraeTemplateModeler(filts='r').fit(x, y, err)
	FTP_GATSPY = model.periodogram(periods)
	dt = time() - t0

	# print
	print N, len(freqs), dt, dt_ftp
