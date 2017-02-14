from time import time
import numpy as np
from math import * 
from gatspy.periodic import RRLyraeTemplateModeler
from gatspy.periodic.template_modeler import BaseTemplateModeler
import gatspy.datasets as datasets
from pyftp.modeler import Template, FastTemplateModeler, rms_resid_over_rms_fast
from pyftp.gatspy_template_modeler import GatspyTemplateModeler

ofac = 5
hfac = 1
#stop = 2E-2
redo = False
nharmonics = 6
time_gatspy = False

Ns = [ 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 300, 500, 1000, 2000, 5000  ]

# Get template from Gatspy
rrl_templates = datasets.rrlyrae.fetch_rrlyrae_templates()
ftp_template_id = '100r' # very non-sinusoidal
Ttemp, Ytemp = rrl_templates.get_template(ftp_template_id)

# Load template or generate new one from gatspy template
fname = 'saved_templates/%s_nh%d.pkl'%(ftp_template_id, nharmonics)
ftp_template = Template(fname=fname, template_id=ftp_template_id, 
	      nharmonics=nharmonics, errfunc=rms_resid_over_rms_fast)

if ftp_template.is_saved() and not redo:
	print "loading template"
	ftp_template.load()

else:
	print "generating template"
	ftp_template.phase = Ttemp
	ftp_template.y     = Ytemp
	ftp_template.precompute()
	ftp_template.save()

templates = [ ftp_template ]
#print len(ftp_template.cn), " harmonics"
ftpmodel = FastTemplateModeler(ofac=ofac, hfac=hfac)
ftpmodel.add_templates(templates)

# initialize gatspy template modeler 
gmodel = None if not time_gatspy else GatspyTemplateModeler(templates=templates)

if time_gatspy:
	print "# gatspy"
else:
	print "# fast template modeler (H = %d)"%(nharmonics)

for N in Ns:

	# generate signal
	freq = 3.
	x = np.sort(np.random.rand(N))
	err = np.absolute(np.random.normal(loc=0.2, scale=0.1, size=N))
	y = np.cos(freq * x) + np.sin(3 * freq * x - 0.44) \
	            + np.array([ np.random.normal(loc=0, scale=s) for s in err ])
	


	# Time the FTP modeler
	t0 = time()
	ftpmodel.fit(x, y, err).compute_sums()
	freqs, per = ftpmodel.periodogram()
	dt = time() - t0

	if not time_gatspy:
		print N, len(freqs), dt

	if time_gatspy:
		periods = np.power(freqs, -1)[::-1]
	
		# Time the Gatspy template modeler
		t0 = time()
		perg = gmodel.fit(x, y, err).periodogram(periods)
		dt = time() - t0

		print N, len(freqs), dt
