from time import time
import cPickle as pickle
import numpy as np
from math import *
import matplotlib.pyplot as plt 
from gatspy.periodic import RRLyraeTemplateModeler
import gatspy.datasets as datasets
import pyftp.rrlyr_modeler as rrlm
import pyftp.fast_template_periodogram as ftp
# Runs the RRLyrModeler against the gatspy RRLyraeTemplateModeler

#stops = np.array([ 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2, 0.5 ])
stops= [ 5E-3, 1E-2, 2E-2, 5E-2, 1E-1, 2E-1, 5E-1 ]

# Get some data
rrlyrae = datasets.fetch_rrlyrae()

# Pick an arbitary lightcurve
lcid = rrlyrae.ids[0]

t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
mask = (filts == 'r')
x, y, err = t[mask], mag[mask], dmag[mask]

#x,y,err = pickle.load(open('xyerr.pkl', 'rb'))
N = len(x)



# Run our RRLyr modeler
df = 0.005
max_f = 1./(0.2)

T = max(x) - min(x)
ofac = 1./(T * df)
hfac = max_f * T / float(len(x))


FTPS = []
TIMES = []
for stop in stops:
	t0 = time()
	freqs, p = rrlm.RRLyrModeler(x, y, err, filts='r', loud=True, 
                          ofac=ofac, hfac=hfac, stop=stop, errfunc=rrlm.rms_resid_over_rms,
                          template_fname='saved_templates/templates_RR_stop%.2e.pkl'%(stop))
	dt = time() - t0
	print "ftp: ", dt, "seconds; stop = ", stop
	TIMES.append(dt)
	FTPS.append(p)


# Now run the gatspy modeler
periods = np.power(freqs, -1)[::-1]

t0 = time()
model = RRLyraeTemplateModeler(filts='r').fit(x, y, err)
FTP_GATSPY = model.periodogram(periods)
dt = time() - t0
print "gatspy: ", dt, " seconds"

# and PLOT!
FTP_GATSPY = FTP_GATSPY[::-1]
SG = sqrt(np.mean(np.power(FTP_GATSPY, 2)))

DFTPS = [ ftp - FTP_GATSPY for ftp in FTPS ]
ERRS = [ sqrt(np.mean(np.power(df, 2))) / SG for df in DFTPS ]

#dEdT = [ (ERRS[i+1] - ERRS[i]) / (TIMES[i] - TIMES[i+1]) for i in range(len(ERRS) - 1) ]

"""
f2, ax2 = plt.subplots()
ax2.plot(stops[:-1], dEdT)
ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.set_xlabel('max rms(fit - template)/rms(template)')
ax2.set_ylabel('derr / dtime')
#ax2.invert_xaxis()

plt.show()
"""

f, ax = plt.subplots()
da = 0. if (len(stops) == 1) else 0.75 / (len(stops) - 1)

for i, (stop, err, dftp) in enumerate(zip(stops, ERRS, DFTPS)):
	print "%.5f = rms(resid) / rms, stop = %.5f"%(err, stop)
	ax.plot(freqs, dftp, color='k', alpha=1.0 - da * i, label="stop=%.5f"%(stop))

ax.set_xlabel('freq')
ax.set_ylabel('rms(resid) / rms(gatspy per)')
#ax.set_title('stop=%.3e, N=%d'%(stop, len(x)))
ax.legend(loc='best')
plt.show()

