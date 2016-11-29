import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from math import *

tdata = np.loadtxt('timing.txt', dtype=np.dtype([('N', int), ('Nf', int), ('t_ftp', float), ('t_gatspy', float)]))

# Fit n^2 and nlogn
NlogN = lambda x, a, b, c : a * np.multiply(x, np.log(x)) + b* x + c
N2    = lambda x, a, b, c : a * np.power(x, 2) + b * x +  c
N     = lambda x, a, c    : a * x + c

bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
popt_ftp, pcov_ftp = curve_fit(NlogN, tdata['N'], tdata['t_ftp'], bounds=bounds)
popt_gats, pcov_gats = curve_fit(N2, tdata['N'], tdata['t_gatspy'],bounds=bounds )

n = np.logspace(0, 4)
gats_n2 = N2(n, *popt_gats)
ftp_nlogn = NlogN(n, *popt_ftp)

# plot!
f, ax = plt.subplots()
ax.plot(tdata['N'], tdata['t_gatspy'], 'o', label='Gatspy')
ax.plot(n, gats_n2, color='k', ls=':', label="%.2e $N^2$ + %.2e$N$ + %.2e"%(popt_gats[0], popt_gats[1], popt_gats[2]))
ax.plot(tdata['N'], tdata['t_ftp'], 's', color='g', label='FTP (6 harmonics)')
ax.plot(n, ftp_nlogn, color='g', ls='--', label='%.2e $N\\logN$ + %.2e$N$ + %.2e'%(popt_ftp[0], popt_ftp[1], popt_ftp[2]))

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=12)
ax.set_xlabel('$N = (2/5) N_{\\rm freq}$', fontsize=15)
ax.set_ylabel('Exec. time (sec)', fontsize=15)
ax.set_xlim(10, 10000)
ax.set_ylim(1E-2, 1E5)
f.savefig('timing.png')

plt.show()
