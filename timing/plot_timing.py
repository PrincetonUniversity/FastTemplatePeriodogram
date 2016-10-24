import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

tdata = np.loadtxt('timing.txt', dtype=np.dtype([('N', int), ('Nf', int), ('t_gatspy', float), ('t_ftp', float)]))

# Fit n^2 and nlogn
NlogN = lambda x, a, b : a * np.multiply(x, np.log(x)) + b
N2    = lambda x, a, b, c : a * np.power(x - b, 2) + c

popt_ftp, pcov_ftp = curve_fit(NlogN, tdata['N'], tdata['t_ftp'])
popt_gats, pcov_gats = curve_fit(N2, tdata['N'], tdata['t_gatspy'])

n = np.logspace(1, 4)
gats_n2 = N2(n, *popt_gats)
ftp_nlogn = NlogN(n, *popt_ftp)
###################

# plot!
f, ax = plt.subplots()

ax.plot(tdata['N'], tdata['t_gatspy'], 'o', label='Gatspy RR Lyr')
ax.plot(tdata['N'], tdata['t_ftp'], 's', label='FTP RR Lyr (stop=0.02)')
ax.plot(n, gats_n2, color='k', ls=':', label="%.2e $(N - (%.2e))^2$ + %.2e"%(popt_gats[0], popt_gats[1], popt_gats[2]))
ax.plot(n, ftp_nlogn, color='k', ls='--', label='%.2e $N\\logN$ + %.2e'%(popt_ftp[0], popt_ftp[1]))

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlabel('$N = (2/5) N_{\\rm freq}$')
ax.set_ylabel('Exec. time (sec)')
ax.set_xlim(10, 10000)
f.savefig('timing.png')

plt.show()
