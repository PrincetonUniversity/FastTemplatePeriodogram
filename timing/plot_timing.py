import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

tdata = np.loadtxt('timing_v4.txt', dtype=np.dtype([('N', int), ('Nf', int), ('t_ftp', float), ('t_gatspy', float)]))
#tdata2 = np.loadtxt('timing_v2.txt', dtype=np.dtype([('N', int), ('Nf', int), ('t_ftp', float)]))
#tdata3 = np.loadtxt('timing_v3.txt', dtype=np.dtype([('N', int), ('Nf', int), ('t_ftp', float)]))


# Fit n^2 and nlogn
NlogN = lambda x, a, c : a * np.multiply(x, np.log(x)) + c#b * x + c
N2    = lambda x, a, c : a * np.power(x, 2) + 0 * x + c

#popt_ftp, pcov_ftp = curve_fit(NlogN, tdata['N'], tdata['t_ftp'])
#popt_ftp2, pcov_ftp2 = curve_fit(NlogN, tdata2['N'], tdata2['t_ftp'])
#popt_ftp3, pcov_ftp3 = curve_fit(NlogN, tdata3['N'], tdata3['t_ftp'])
popt_ftp, pcov_ftp = curve_fit(NlogN, tdata['N'], tdata['t_ftp'])
popt_gats, pcov_gats = curve_fit(N2, tdata['N'], tdata['t_gatspy'])

n = np.logspace(0, 4)
gats_n2 = N2(n, *popt_gats)
ftp_nlogn = NlogN(n, *popt_ftp)
#ftp2_nlogn = NlogN(n, *popt_ftp2)
#ftp3_nlogn = NlogN(n, *popt_ftp3)
###################

# plot!
f, ax = plt.subplots()

ax.plot(tdata['N'], tdata['t_gatspy'], 'o', label='Gatspy RR Lyr')
ax.plot(n, gats_n2, color='k', ls=':', label="%.2e $N^2$ + %.2e $N$ + %.2e"%(popt_gats[0], 0, popt_gats[1]))
ax.plot(tdata['N'], tdata['t_ftp'], 's', color='g', label='FTP (Oct 22, 2016) RR Lyr (stop=0.02)')
ax.plot(n, ftp_nlogn, color='g', ls='--', label='%.2e $N\\logN$ + %.2e'%(popt_ftp[0], popt_ftp[1]))
#ax.plot(tdata2['N'], tdata2['t_ftp'], 's', color='c', label='FTP (Oct 24, 2016) RR Lyr (stop=0.02)')
#ax.plot(n, ftp2_nlogn, color='c', ls='--', label='%.2e $N\\logN$ + %.2e'%(popt_ftp2[0], popt_ftp2[1]))
#ax.plot(tdata3['N'], tdata3['t_ftp'], 's', color='m', label='FTP (Oct 26, 2016) RR Lyr (stop=0.02)')
#ax.plot(n, ftp3_nlogn, color='m', ls='--', label='%.2e $N\\logN$ + %.2e'%(popt_ftp3[0], popt_ftp3[1]))

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlabel('$N = (2/5) N_{\\rm freq}$')
ax.set_ylabel('Exec. time (sec)')
ax.set_xlim(10, 10000)
f.savefig('timing.png')

plt.show()
