from time import time
import numpy as np
from math import * 
from gatspy.periodic import RRLyraeTemplateModeler
from gatspy.periodic.template_modeler import BaseTemplateModeler
import gatspy.datasets as datasets
from pyftp.modeler import Template, FastTemplateModeler, rms_resid_over_rms_fast

ofac = 5
hfac = 1
stop = 2E-2
redo = False

Ns = [ 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 300, 500, 1000, 2000, 5000  ]

# Get template from Gatspy
rrl_templates = datasets.rrlyrae.fetch_rrlyrae_templates()
ftp_template_id = '100r' # very non-sinusoidal
Ttemp, Ytemp = rrl_templates.get_template(ftp_template_id)

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


# Load template or generate new one from gatspy template
fname = 'saved_templates/%s_stop%.3e.pkl'%(ftp_template_id, stop)
ftp_template = Template(fname=fname, template_id=ftp_template_id, 
	                 stop=stop, errfunc=rms_resid_over_rms_fast)

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
print len(ftp_template.cn), " harmonics"
ftpmodel = FastTemplateModeler(ofac=ofac, hfac=hfac)
ftpmodel.add_templates(templates)

# initialize gatspy template modeler 
gmodel = GatspyTemplateModeler(templates=templates)

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
	dt_ftp = time() - t0

	periods = np.power(freqs, -1)[::-1]
	print float(len(periods))/float(len(x))

	# Time the Gatspy template modeler
	t0 = time()
	perg = gmodel.fit(x, y, err).periodogram(periods)
	dt_gatspy = time() - t0

	# print
	print N, len(freqs), dt_ftp, dt_gatspy
