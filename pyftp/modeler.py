from __future__ import print_function

import os
import sys
from time import time
from math import *
import numpy as np
from scipy.optimize import curve_fit

try:
    # Python 2.x only
    import cPickle as pickle
except ImportError:
    import pickle

from . import fast_template_periodogram as ftp


def LMfit(x, y, err, cn, sn, omega, sgn=1):
    """ fits a, b, c with Levenberg-Marquardt """

    ffunc = lambda X, *pars : ftp.fitfunc(X, sgn, omega, cn, sn, *pars)
    p0 = [ np.std(y), 0.0, np.mean(y) ]
    bounds = ([0, -1, -np.inf], [ np.inf, 1, np.inf])
    popt, pcov = curve_fit(ffunc, x, y, sigma=err, p0=p0,
                            absolute_sigma=True, bounds=bounds,
                            method='trf')
    a, b, c = popt

    return a, b, c


def rms_resid_over_rms(cn, sn, Tt, Yt):
    # This is fairly slow; is there a better way to get best fit pars?
    a, b, c = LMfit(Tt, Yt, np.ones(len(Tt))*0.0001, cn, sn, 2 * np.pi, sgn=True)
    Ym = ftp.fitfunc(Tt, 1, 2 * np.pi, cn, sn, a, b, c)

    S = sqrt(np.mean(np.power(Yt, 2)))

    Rp = sqrt(np.mean(np.power(Ym - Yt, 2))) / S

    a, b, c = LMfit(Tt, Yt, np.ones(len(Tt))*0.0001, cn, sn, 2 * np.pi, sgn=False)
    Ym = ftp.fitfunc(Tt, -1, 2 * np.pi, cn, sn, a, b, c)

    Rn = sqrt(np.mean(np.power(Ym - Yt, 2))) / S
    return min([ Rn, Rp ])

rms = lambda x : sqrt(np.mean(np.power(x, 2)))


def match_up_truncated_template(cn, sn, Tt, Yt):
    Ym = ftp.fitfunc(Tt, 1, 2 * np.pi, cn, sn, 2.0, 0.0, 0.0)

    # Align the maxima of truncated and full templates
    di = np.argmax(Ym) - np.argmax(Yt)

    # Add some 'wiggle room', since maxima may be offset by 1
    Ym = [ np.array([ Ym[(j + (di + k))%len(Ym)] for j in range(len(Ym)) ]) for k in [ -1, 0, 1 ] ]

    # Align the heights of the truncated and full templates
    Ym = [ Y + (Yt[0] - Y[0]) for Y in Ym ]

    # Keep the best fit
    return Ym[np.argmin( [ rms(Y - Yt) for Y in Ym ] )]


def rms_resid_over_rms_fast(cn, sn, Tt, Yt):
    Ym = match_up_truncated_template(cn, sn, Tt, Yt)
    return rms(Yt - Ym) / rms(Yt)


def approximate_template(Tt, Yt, errfunc=rms_resid_over_rms, stop=1E-2, nharmonics=None):
    """ Fourier transforms template, returning the first H components """

    #print "fft"
    fft = np.fft.fft(Yt[::-1])

    cn, sn = None, None
    if not nharmonics is None and int(nharmonics) > 0:
        #print "creating cn and sn"
        cn, sn = zip(*[ (p.real/len(Tt), p.imag/len(Tt)) for i,p in enumerate(fft) \
                     if i > 0 and i <= int(nharmonics) ])

    else:

        cn, sn = zip(*[ (p.real/len(Tt), p.imag/len(Tt)) for i,p in enumerate(fft) \
                     if i > 0 ])

        h = 1
        while errfunc(cn[:h], sn[:h], Tt, Yt) > stop:
            #print "h -> ", h
            h+=1

        cn, sn = cn[:h], sn[:h]
    return cn, sn

normfac = lambda cn, sn : 1./np.sqrt(sum([ ss*ss + cc*cc for cc, ss in zip(cn, sn) ]))


class Template(object):
    """
    Template class

    y(t) = sum[n]( c[n]cos(nwt) + s[n]sin(nwt) )

    Parameters
    ----------
    cn: array-like, optional
        Truncated Fourier coefficients (cosine)
    sn: array-like, optional
        Truncated Fourier coefficients (sine)
    phase: array-like, optional
        phase-values, must contain floating point numbers in [0,1]
    y: array-like, optional
        amplitude of template at each phase value
    stop: float, optional (default: 2E-2)
        will pick minimum number of harmonics such that
        rms(trunc(template) - template) / rms(template) < stop
    nharmonics: None or int, optional (default: None)
        Keep a constant number of harmonics
    fname: str, optional
        Filename to load/save template
    errfunc: callable, optional (default: rms_resid_over_rms)
        A function returning some measure of error resulting
        from approximating the template with a given number
        of harmonics
    template_id: str, optional
        Name of template

    """
    def __init__(self, cn=None, sn=None, phase=None, y=None,
                       stop=2E-2, nharmonics=None, fname=None,
                       errfunc=rms_resid_over_rms, template_id=None):

        self.phase = phase
        self.y = y

        self.fname = fname
        self.stop = stop
        self.nharmonics = nharmonics
        self.errfunc = errfunc
        self.cn = None
        self.sn = None
        self.pvectors = None
        self.ptensors = None
        self.template_id = template_id
        self.best_fit_y = None

        self.cn_full = None
        self.sn_full = None


        self.defined_by_ty = cn is None and sn is None
        if self.defined_by_ty and (phase is None or y is None):
            raise Exception("Need to define either (phase, y) or (cn, sn) for template")

        if not self.defined_by_ty:

            assert(not (cn is None or sn is None))

            self.cn_full = np.array(cn).copy()
            self.sn_full = np.array(sn).copy()

            if nharmonics is None:
                self.nharmonics = len(cn)

            assert(self.nharmonics <= len(cn))

            self.cn = self.cn_full[:self.nharmonics].copy()
            self.sn = self.sn_full[:self.nharmonics].copy()


            self.cn *= normfac(self.cn, self.sn)
            self.sn *= normfac(self.cn, self.sn)

    def is_saved(self):

        return (not self.fname is None and os.path.exists(self.fname))

    def precompute(self):


        if self.defined_by_ty:
            #print "approximating template"
            self.cn, self.sn = approximate_template(self.phase, self.y,
                                        stop=self.stop, errfunc=self.errfunc,
                                        nharmonics=self.nharmonics)

            #print "getting best fit y"
            self.best_fit_y = match_up_truncated_template(self.cn, self.sn, self.phase, self.y)

            #print "computing rms_resid/rms"
            self.rms_resid_over_rms = rms(self.best_fit_y - self.y) / rms(self.y)

        else:

            assert(self.nharmonics <= len(self.cn_full))

            self.cn = self.cn_full[:self.nharmonics].copy()
            self.sn = self.sn_full[:self.nharmonics].copy()

            self.cn *= normfac(self.cn, self.sn)
            self.sn *= normfac(self.cn, self.sn)

            nph = 100
            self.phase = np.linspace(0, 1, nph)
            self.y = ftp.fitfunc(self.phase, 1, 2 * np.pi, self.cn_full, self.sn_full, 2.0, 0.0, 1.0)
            self.best_fit_y = ftp.fitfunc(self.phase, 1, 2 * np.pi, self.cn, self.sn, 2.0, 0.0, 1.0)

            self.rms_resid_over_rms = rms(self.best_fit_y - self.y) / rms(self.y)

        #print "computing pvectors"
        #print self.cn
        #print self.sn
        self.pvectors = ftp.get_polynomial_vectors(self.cn, self.sn, sgn=1)

        #print "computing ptensors"
        self.ptensors = ftp.compute_polynomial_tensors(*self.pvectors)

        return self

    def load(self, fname=None):
        fn = fname if not fname is None else self.fname
        self.__dict__.update(pickle.load(open(fn, 'rb')))

    def save(self, fname=None):
        fn = fname if not fname is None else self.fname
        pickle.dump(self.__dict__, open(fn, 'wb'))

    def add_plot_to_axis(self, ax):
        ax.plot(self.phase, self.y, color='k', label="original")
        ax.plot(self.phase, self.best_fit_y,
                  label="truncated (H=%d)"%(self.nharmonics))

    def set_nharmonics(self, nharmonics):
        self.nharmonics = nharmonics
        self.precompute()
        return self

    def plot(self, plt):
        f, ax = plt.subplots()
        self.add_plot_to_axis(ax)
        ax.set_xlim(0, 1)
        #ax.set_ylim(0, 1)
        ax.set_xlabel('phase')
        ax.set_ylabel('y')
        ax.set_title('"%s", stop = %.3e, H = %d'%(self.template_id,
                                             self.stop, self.nharmonics))
        ax.legend(loc='best', fontsize=9)
        plt.show()
        plt.close(f)


class FastTemplateModeler(object):

    """
    Base class for template modelers

    Parameters
    ----------

    loud: boolean (default: True), optional
        print status
    ofac: float, optional (default: 10)
        oversampling factor -- higher values of ofac decrease
        the frequency spacing (by increasing the size of the FFT)
    hfac: float, optional (default: 3)
        high-frequency factor -- higher values of hfac increase
        the maximum frequency of the periodogram at the
        expense of larger frequency spacing.
    errfunc: callable, optional (default: rms_resid_over_rms)
        A function returning some measure of error resulting
        from approximating the template with a given number
        of harmonics
    stop: float, optional (default: 0.01)
        A stopping criterion. Once `errfunc` returns a number
        that is smaller than `stop`, the harmonics up to that point
        are kept. If not, another harmonic is added.
    nharmonics: None or int, optional (default: None)
        Number of harmonics to keep if a constant number of harmonics
        is desired

    """
    def __init__(self, **kwargs):
        self.params = { key : value for key, value in kwargs.iteritems() }
        self.templates = {}
        self.omegas = None
        self.summations = None
        self.YY = None
        self.max_harm = 0
        self.w = None
        self.ybar = None

        defaults = dict(ofac=10, hfac=3)

        # set defaults
        for key, value in defaults.iteritems():
            if not key in self.params:
                self.params[key] = value
        if 'templates' in self.params:
            self.add_templates(self.params['templates'])
            del self.params['templates']

    def _get_template_by_id(self, template_id):
        assert(template_id in self.templates)
        return self.templates[template_id]

    def _template_ids(self):
        return self.templates.keys()

    def get_new_template_id(self):
        i = 0
        while i in self.templates:
            i += 1
        return i

    def add_template(self, template, template_id=None):
        if template_id is None:
            if template.template_id is None:
                ID = self.get_new_template_id()
                self.templates[ID] = template
            else:
                self.templates[template.template_id] = template
        else:
            self.templates[template_id] = template
        return self

    def add_templates(self, templates, template_ids=None):

        if isinstance(templates, dict):
            for ID, template in templates.iteritems():
                self.add_template(template, template_id=ID)

        elif isinstance(templates, list):
            if template_ids is None:
                for template in templates:
                    self.add_template(template)
            else:
                for ID, template in zip(template_ids, templates):
                    self.add_template(template, template_id=ID)
        elif not hasattr(templates, '__iter__'):
            self.add_template(templates, template_id=template_ids)
        else:
            raise Exception("did not recognize type of 'templates' passed to add_templates")

        return self

    def remove_templates(self, template_ids):
        for ID in template_ids:
            assert ID in self.templates
            del self.templates[ID]
        return self

    def set_params(self, **new_params):
        self.params.update(new_params)
        return self

    def fit(self, x, y, err):
        """
        Parameters
        ----------
        x: np.ndarray, list
            independent variable (time)
        y: np.ndarray, list
            array of observations
        err: np.ndarray
            array of observation uncertainties
        """
        self.x = x
        self.y = y
        self.err = err

        # Set all old values to none
        self.summations = None
        self.freqs_ = None
        self.best_template_id = None
        self.best_template = None
        self.best_model_params = None
        self.periodogram_ = None
        self.model_params_ = None
        self.periodogram_all_templates_ = None
        return self

    def compute_sums(self):

        self.omegas, self.summations, \
        self.YY, self.w, self.ybar = \
            ftp.compute_summations(self.x, self.y, self.err, self.max_harm,
                                ofac=self.params['ofac'], hfac=self.params['hfac'])

        return self



    def periodogram(self, **kwargs):
        self.params.update(kwargs)

        #if self.summations is None:
        #    self.compute_sums()
        loud = False if not 'loud' in self.params else self.params['loud']

        all_ftps = []
        for template_id, template in self.templates.iteritems():
            args = (self.x, self.y, self.err, template.cn, template.sn)
            kwargs = dict(ofac       = self.params['ofac'],
                          hfac       = self.params['hfac'],
                          ptensors   = template.ptensors,
                          pvectors   = template.pvectors,
                          omegas     = self.omegas,
                          summations = self.summations,
                          YY         = self.YY,
                          ybar       = self.ybar,
                          w          = self.w,
                          loud       = loud,
                          return_best_fit_pars=True)
            all_ftps.append((template_id, ftp.fast_template_periodogram(*args, **kwargs)))

        template_ids, all_ftps_ = zip(*all_ftps)
        freqs, ftps, modelpars  = zip(*all_ftps_)
        freqs = freqs[0]

        self.periodogram_ = np.array([ max([ f[i] for f in ftps ]) for i in range(len(freqs)) ])
        self.freqs_ = freqs
        self.periodogram_all_templates_ = zip(template_ids, ftps)
        self.model_params_ = zip(template_ids, modelpars)

        # Periodogram is the maximum periodogram value at each frequency
        return self.freqs_, self.periodogram_

    def get_best_model(self, **kwargs):

        ibest = np.argmax(self.periodogram_)
        tbest = np.argmax([ f[ibest] for t, f in self.periodogram_all_templates_ ])

        self.best_freq = self.freqs_[ibest]
        self.best_template_id, self.best_model_params = self.model_params_[tbest]

        self.best_model_params = self.best_model_params[ibest]
        self.best_template = self.templates[self.best_template_id]

        return self.best_template, self.best_model_params
