import numpy as np
import numpy.polynomial as pol

from ftperiodogram.summations import direct_summations_single_freq
from ftperiodogram.template import Template
from ftperiodogram.core import fit_template
import matplotlib.pyplot as plt
#(t, y, w, freq, nharmonics):

get_yhat = lambda thetas, model : lambda phi, thetas=thetas, model=model : thetas[0] * model(phi - thetas[1]) + thetas[2]

def pdg_and_poly(cn, sn, sums, ybar):
    H = len(cn)

    alpha = 0.5 * (np.asarray(cn) + 1j * np.asarray(sn))
    alpha_conj = np.conj(alpha)

    CC =  (sums.CC - sums.SS - 1j * (sums.CS + sums.CS.T)) * np.outer(alpha, alpha)
    CS =  2 * ((sums.CC + sums.SS + 1j * (sums.CS - sums.CS.T)) * np.outer(alpha, alpha_conj))#[:,::-1]
    SS = np.conj(CC)#[::-1, ::-1]

    YC = np.array(sums.YC - 1j * sums.YS, dtype=np.complex64)

    aYC = alpha * YC
    YM = pol.Polynomial(np.concatenate((np.conj(aYC)[::-1], [0], aYC)).astype(np.complex64))

    MM = np.zeros(4 * H + 1, dtype=np.complex64)

    #for k in range(0, 2 * H - 1):
    #    n0 = max([ 0, k - (H-1) ])
    #    m0 = min([ H-1, k ])

    #    inds = np.arange(k + 1)
    #    if k + 1 > H:
    #        inds = np.arange(2 * H - k - 1)


    #    MM[k + 2 * H + 2] = np.sum(CC[n0 + inds, m0 - inds])
    #    MM[k + H + 1]     = np.sum(CS[n0 + inds, m0 - inds])
    #    MM[k]             = np.sum(SS[n0 + inds, m0 - inds])
    for n in range(H):
        for m in range(H):
            MM[2*H + (n + 1) + (m + 1)] += CC[n][m]
            MM[2*H + (n + 1) - (m + 1)] += CS[n][m]
            MM[2*H - (n + 1) - (m + 1)] += SS[n][m]


    MM = pol.Polynomial(MM)

    AC = alpha * (sums.C - 1j * sums.S)
    alpha_phi = pol.Polynomial(np.concatenate(([0], AC)))

    #phiH = np.zeros(H+1)
    #phiH[-1] = 1.

    #YMphiH = pol.Polynomial(phiH) * YM

    p = 2 * MM * YM.deriv() - MM.deriv() * YM

    roots = p.roots()
    #roots = np.array([ np.imag(np.log(r))%(2 * np.pi) for r in roots ])

    pdg_phi = lambda phi : np.real(YM(phi) ** 2 / MM(phi))

    best_phi = np.argmax(pdg_phi(roots))

    
    theta_1 = np.real(YMphiH(best_phi) /  MM(best_phi))
    theta_2 = np.imag(np.log(best_phi)) % (2 * np.pi)

    mbar = 2 * np.real(alpha_phi(best_phi))
    theta_3 = ybar - mbar * theta_1

    return (theta_1, theta_2, theta_3), pdg_phi(best_phi)

    """
    
    pdg_th2 = lambda theta_2 : pdg_phi(np.exp(1j * theta_2))
    p_th2 = lambda theta_2 : p(np.exp(1j * theta_2))

    mbar = lambda phi : 2 * np.real(alpha_phi(phi))
    theta_1 = lambda phi : np.real(YMphiH(phi) /  MM(phi))
    theta_2 = lambda phi : np.imag(np.log(phi))
    theta_3 = lambda phi, mbar=mbar : ybar - mbar(phi) * theta_1(phi)

    return pdg_th2, p_th2, theta_1, theta_2, theta_3, roots
    """


def get_template(cn, sn):
    tmpl = Template(cn, sn)

    template = lambda phi, tmpl=tmpl : tmpl((phi / (2 * np.pi))%1.0)

    return template

rand = np.random.RandomState()
def simulate_data(thetas, omega, model, ndata=100, sigma=0.01):
    yhat = get_yhat(thetas, model)

    t = np.sort(rand.rand(ndata))
    y = yhat(omega * t)
    y += sigma * np.random.randn(ndata)
    dy = sigma * np.ones_like(y)
    return t, y, dy

def pdg_manual(t, y, dy, omega, thetas, model):
    w = np.power(dy, -2)
    w /= sum(w)

    yhat = get_yhat(thetas, model)(omega * t)
    ybar = np.dot(w, y)

    chi2 = np.dot(w, np.power(y - yhat, 2))
    chi20 = np.dot(w, np.power(y - ybar, 2))

    return 1. - chi2 / chi20

def test():

    thetas = ( 1.0, 5.0, 0.0 )
    omega = 2 * np.pi * 10

    cn = np.array([0.8, 0.7, 0.1, 0.1, 0.2, 0.9, 0.9, .01, 0.2, .01, .01, .01])
    sn = np.array([0.7, 0.2, 0.6, 0.1, 0.8, 0.9, 0.9, 0.3, 0.1, 1.0, .01, .01])

    tmp = Template(cn, sn)
    cn, sn = tmp.c_n, tmp.s_n

    print(len(cn), len(sn))


    model = get_template(cn, sn)

    t, y, dy = simulate_data(thetas, omega, model)

    w = np.power(dy, -2)
    w /= sum(w)
    ybar = np.dot(w, y)

    YY = np.dot(w, np.power(y - ybar, 2))



    #sums = direct_summations_single_freq(t, y, w, omega / (2 * np.pi), len(cn))
    #thetas, power = pdg_and_poly(cn, sn, sums, ybar)
    #power /= YY

    fit_template(t, y, dy, cn, sn, None, omega / (2 * np.pi))



    """
    pdg_th2, p_th2, theta_1, theta_2, theta_3, roots = pdg_and_poly(cn, sn, sums, ybar)
    print(roots)
    #roots = [ r / abs(r) for r in roots if abs(r - 1) < 0.1 ]
    print(roots)
    th2_roots = [ np.imag(np.log(r)) for r in roots ]
    th2_roots = [ (r + (2 * np.pi if r < 2 else 0))%(2 * np.pi) for r in th2_roots ]


    theta_2 = np.linspace(0, 2 * np.pi, 200)
    poly = p_th2(theta_2)
    all_thetas = [ (theta_1(np.exp(1j * th2)), th2, theta_3(np.exp(1j * th2))) for th2 in theta_2 ]

    f, ax = plt.subplots()

    th1, th2, th3 = zip(*all_thetas)
    ax.plot(th2, th1, color='r')
    ax.plot(th2, th3, color='b')
    ax.plot(theta_2, pdg_th2(theta_2) / YY, color='k')
    ax.plot(theta_2, poly, color='c')

    ax.axhline(thetas[0], color='r', ls=':')
    ax.axhline(thetas[2], color='b', ls=':')
    ax.axvline(thetas[1], color='k', ls=':')

    for th2r in th2_roots:
        ax.axvline(th2r, color='0.5', ls='-', lw=2)


    ax.axhline(pdg_manual(t, y, dy, omega, thetas, model), color='k', ls='--')
    #ax.axhline(theta_2, pdg_th2(theta_2) / YY, color='k')
    plt.show()
    """

if __name__ == '__main__':
    test()
