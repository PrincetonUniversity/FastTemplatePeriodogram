import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from math import *

# Fit n^2 and nlogn
NlogN = lambda x, a, b, c : a * np.multiply(x, np.log(x)) + b* x + c
N2    = lambda x, a, b, c : a * np.power(x, 2) + b * x +  c
N     = lambda x, a, c    : a * x + c

n = np.logspace(0, 4)
bounds=([0, 0, 0], [np.inf, np.inf, np.inf])

gats_t = np.loadtxt('timing.txt', dtype=np.dtype([('N', int), ('Nf', int), ('t_ftp', float), ('dt', float)]))

ftp_t = {}
popt_ftp = {}
pcov_ftp = {}
ftp_nlogn = {}
for nh in range(1, 7):
	ftp_t[nh] = np.loadtxt('timing_ftp_h%d.txt'%(nh), dtype=np.dtype([('N', int), ('Nf', int), ('dt', float)]))
	popt_ftp[nh], pcov_ftp[nh] = curve_fit(NlogN, ftp_t[nh]['N'], ftp_t[nh]['dt'], bounds=bounds)
	ftp_nlogn[nh] = NlogN(n, *(popt_ftp[nh]))

popt_gats, pcov_gats = curve_fit(N2, gats_t['N'], gats_t['dt'],bounds=bounds )
gats_n2 = N2(n, *popt_gats)

# plot!
f, ax = plt.subplots()
ax.plot(gats_t['N'], gats_t['dt'], 'o', label='Gatspy')
ax.plot(n, gats_n2, color='k', ls=':', label="%.2e $N^2$ + %.2e$N$ + %.2e"%(popt_gats[0], popt_gats[1], popt_gats[2]))
H = [ 5 ]
for nh in H:
	ax.plot(ftp_t[nh]['N'], ftp_t[nh]['dt'], 's', color='g', label='FTP (%d harmonics)'%(nh))
	ax.plot(n, ftp_nlogn[nh], color='g', ls='--', label='%.2e $N\\logN$ + %.2e$N$ + %.2e'%(popt_ftp[nh][0], popt_ftp[nh][1], popt_ftp[nh][2]))

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=12)
ax.set_xlabel('$N = (2/5) N_{\\rm freq}$', fontsize=15)
ax.set_ylabel('Exec. time (sec)', fontsize=15)
ax.set_xlim(10, 10000)
ax.set_ylim(1E-2, 1E5)
f.savefig('timing.png')

plt.show()
plt.close(f)

# plot!
f, ax = plt.subplots()

H = range(1, 7)
N = ftp_t[H[0]]['N'][8]
T = [ ftp_t[h]['dt'][8] for h in H ]

ax.plot(H, T, 's', color='g', label='FTP (Nobs = %d)'%(N))

ax.legend(loc='upper left', fontsize=12)
ax.set_xlabel('$H$ (number of harmonics)', fontsize=15)
ax.set_ylabel('Exec. time (sec)', fontsize=15)
#ax.set_xlim(0, 10)
#ax.set_ylim(1E-2, 1E5)

f.savefig('timing_nh.png')

plt.show()