from __future__ import print_function

import os
import sys
from time import time
from math import *
import numpy as np
from scipy.optimize import curve_fit
from .utils import weights, ModelFitParams
from .template import Template
#from summations import fast_summations, direct_summations

try:
    # Python 2.x only
    import cPickle as pickle
except ImportError:
    import pickle

from . import fast_template_periodogram as ftp



class TemplateModel(object):
    """
    Template for models of the form :math:`a * M(t - tau) + c` for
    some template M.

    Parameters
    ----------
    template : Template
        Template for the model

    frequency : float, optional (default = 1.0)
        Frequency of the signal

    parameters : ModelFitParams
        Parameters for the model (a, b, c, sgn); must be a `ModelFitParams`
        instance


    Examples
    --------

    >>> params = ModelFitParams(a=1, b=1, c=0, sgn=1)
    >>> template = Template(cn=[ 1.0, 0.4, 0.2], sn=[0.1, 0.9, 0.2])
    >>> model = TemplateModel(template, frequency=1.0, parameters=params)
    >>> t = np.linspace(0, 10, 100)
    >>> y_fit = model(t)
    """
    def __init__(self, template, frequency=1.0, parameters=None):
        self.template = template
        self.frequency = frequency
        self.parameters = parameters

    def __call__(self, t):
        if not isinstance(self.template, Template):
            raise TypeError("template must be a Template instance")
        if not isinstance(self.parameters, ModelFitParams):
            raise TypeError("parameters must be ModelFitParams instance")

        wtau = np.arccos(self.parameters.b)
        if self.parameters.sgn == -1:
            wtau = 2 * np.pi - wtau


        tau = wtau / (2 * np.pi * self.frequency)
        phase = (self.frequency * (t - tau)) % 1.0
        
        return self.parameters.a * self.template(phase) + self.parameters.c


class FastTemplateModeler(object):

    """
    Base class for template modelers

    Fits a single template to the data. For 
    fitting multiple templates, use the FastMultiTemplateModeler

    Parameters
    ----------
    template : Template
        Template to fit (must be Template instance)

    allow_negative_amplitudes : bool (optional, default=True)
        if False, then negative optimal template amplitudes
        will be replaced with zero-amplitude solutions. A False
        value prevents the modeler from fitting an inverted
        template to the data, but does not attempt to find the
        best positive amplitude solution, which would require
        substantially more computational resources.

    """
    def __init__(self, template=None, allow_negative_amplitudes=True):
        
        self.template = template
        self.allow_negative_amplitudes = allow_negative_amplitudes
        self.t, self.y, self.dy = None, None, None
        #self.summations = None
        self.best_model = None

    def _validate_template(self, template):
        pass

    def _validate_data(self, t, y, dy=None):
        pass
        
    def _validate_frequencies(self, frequencies):
        pass

    def fit(self, t, y, dy=None):
        """
        Parameters
        ----------
        t: array_like
            sequence of observation times
        y: array_like
            sequence of observations associated with times `t`
        dy: float or array_like (optional, default=None)
            error(s)/uncertaint(ies) associated with observed values `y`.
            If scalar, all observations are weighted equally, which is
            effectively the same as setting `dy=None`.

        """
        self.t = np.array(t)
        self.y = np.array(y)
        self.dy = np.array(dy)
        
        return self

    def autofrequency(self, nyquist_factor=5, samples_per_peak=5, minimum_frequency=None,
                        maximum_frequency = None):
        """
        Determine a suitable frequency grid for data.

        Note that this assumes the peak width is driven by the observational
        baseline, which is generally a good assumption when the baseline is
        much larger than the oscillation period.
        If you are searching for periods longer than the baseline of your
        observations, this may not perform well.

        Even with a large baseline, be aware that the maximum frequency
        returned is based on the concept of "average Nyquist frequency", which
        may not be useful for irregularly-sampled data. The maximum frequency
        can be adjusted via the nyquist_factor argument, or through the
        maximum_frequency argument.

        Parameters
        ----------
        samples_per_peak : float (optional, default=5)
            The approximate number of desired samples across the typical peak
        nyquist_factor : float (optional, default=5)
            The multiple of the average nyquist frequency used to choose the
            maximum frequency if maximum_frequency is not provided.
        minimum_frequency : float (optional)
            If specified, then use this minimum frequency rather than one
            chosen based on the size of the baseline.
        maximum_frequency : float (optional)
            If specified, then use this maximum frequency rather than one
            chosen based on the average nyquist frequency.

        Returns
        -------
        frequency : ndarray or Quantity
            The heuristically-determined optimal frequency bin
        """

        if any([ X is None for X in [ self.t, self.y, self.dy ] ]):
            raise ValueError("One or more of t, y, dy is None; "
                             "fit(t, y, dy) must be called before autofrequency.")

        baseline = self.t.max() - self.t.min()
        n_samples = self.t.size

        df = 1. / baseline / samples_per_peak

        if minimum_frequency is not None:
            f0 = df * np.floor(minimum_frequency / df)
        else:
            f0 = df

        if maximum_frequency is not None:
            Nf = int(np.ceil((maximum_frequency - f0) / df))
        else:
            Nf = int(0.5 * samples_per_peak * nyquist_factor * n_samples)

        return f0 + df * np.arange(Nf)

    def autopower(self, save_best_model=True, **kwargs):
        """
        Compute template periodogram at automatically-determined frequencies

        Parameters
        ----------
        save_best_model : optional, bool (default = True)
            Save a TemplateModel instance corresponding to the best-fit model found

        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------

        frequency, power : ndarray, ndarray
            The frequency and template periodogram power

        """
        assert(not any([ x is None for x in [ self.t, self.y, self.dy, self.template ]]))

        frequency = self.autofrequency(**kwargs)
        results = ftp.fast_template_periodogram(self.t, self.y, self.dy, self.template.cn, 
                            self.template.sn, frequency, pvectors=self.template.pvectors, 
                            ptensors=self.template.ptensors, return_best_fit_pars=save_best_model,
                            allow_negative_amplitudes=self.allow_negative_amplitudes)

        if save_best_model:
            p, bfpars = results
            i = np.argmax(p)
            self.best_model = TemplateModel(self.template, frequency = frequency[i], 
                                                          parameters = bfpars[i])

            return frequency, p
            

        else:
            p = results
            return frequency, p

    def power(self, frequency, save_best_model=True):
        """
        Compute template periodogram at a given set of frequencies; slower than
        `autopower`, but frequencies are not restricted to being evenly spaced

        Parameters
        ----------
        frequency : float or array_like
            Frequenc(ies) at which to determine template periodogram power

        save_best_model : optional, bool (default=True)
            Save best model fit, accessible via the `best_model` attribute

        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------

        power : float or ndarray
            The frequency and template periodogram power, a

        """
        fitter = lambda frq : ftp.fit_template(self.t, self.y, self.dy, self.template, frq, 
                                allow_negative_amplitudes=self.allow_negative_amplitudes)

        multiple_frequencies = hasattr(frequency, '__iter__')
        if multiple_frequencies:
            bfpars, p = zip(*map(fitter, frequency))
            p = np.array(p)
        else:
            bfpars, p = fitter(frequency)

        if save_best_model:
            if multiple_frequencies:
                i = np.argmax(p)
                self.best_model = TemplateModel(self.template, frequency = frequency[i], 
                                                          parameters = bfpars[i])
            else:
                self.best_model = TemplateModel(self.template, frequency = frequency, 
                                                          parameters = bfpars)

        return p

class FastMultiTemplateModeler(FastTemplateModeler):

    """
    Template modeler that fits multiple templates

    Parameters
    ----------
    templates : list of Template
        Templates to fit (must be list of Template instances)

    allow_negative_amplitudes : bool (optional, default=True)
        if False, then negative optimal template amplitudes
        will be replaced with zero-amplitude solutions. A False
        value prevents the modeler from fitting an inverted
        template to the data, but does not attempt to find the
        best positive amplitude solution, which would require
        substantially more computational resources.

    """
    def __init__(self, templates=None, allow_negative_amplitudes=True):
        
        self.templates = templates
        self.allow_negative_amplitudes = allow_negative_amplitudes
        self.t, self.y, self.dy = None, None, None
        #self.summations = None
        self.best_model = None

    def autopower(self, save_best_model=True, **kwargs):
        """
        Compute template periodogram at automatically-determined frequencies

        Parameters
        ----------
        save_best_model : optional, bool (default = True)
            Save a TemplateModel instance corresponding to the best-fit model found

        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------

        frequency, power : ndarray, ndarray
            The frequency and template periodogram power

        """
        assert(not any([ x is None for x in [ self.t, self.y, self.dy, self.templates ]]))
        assert(isinstance(self.templates, list))
        assert(len(self.templates) > 0)

        frequency = self.autofrequency(**kwargs)

        results = [ ftp.fast_template_periodogram(self.t, self.y, self.dy, template.cn, 
                            template.sn, frequency, pvectors=template.pvectors, 
                            ptensors=template.ptensors, return_best_fit_pars=save_best_model,
                            allow_negative_amplitudes=self.allow_negative_amplitudes)\
                        for template in self.templates ]
        p = None
        if save_best_model:
            p, bfpars = zip(*results)

            #print p
            maxes = [ max(P) for P in p ]
            print("maxes = ", maxes)

            i = np.argmax(maxes)
            j = np.argmax(p[i])

            self.best_model = TemplateModel(self.templates[i], frequency = frequency[j], 
                                                          parameters = bfpars[i][j])
            
        else:
            p = results
        
        
        return frequency, np.max(p, axis=0)

    def power(self, frequency, save_best_model=True):
        """
        Compute template periodogram at a given set of frequencies

        Parameters
        ----------
        frequency : float or array_like
            Frequenc(ies) at which to determine template periodogram power

        save_best_model : optional, bool (default=True)
            Save best model fit, accessible via the `best_model` attribute

        **kwargs : optional, dict
            Passed to `autofrequency`

        Returns
        -------

        power : float or ndarray
            The frequency and template periodogram power, a

        """

        assert(not any([ x is None for x in [ self.t, self.y, self.dy, self.templates ]]))
        assert(isinstance(self.templates, list))
        assert(len(self.templates) > 0)

        fitters =[ lambda frq : ftp.fit_template(self.t, self.y, self.dy, template, frq, 
                                allow_negative_amplitudes=self.allow_negative_amplitudes)\
                    for template in self.templates ]
        p_final = None

        multiple_frequencies = hasattr(frequency, '__iter__')
        if multiple_frequencies:
            p, bfpars = zip(*[ zip(*map(fitter, frequency)) for fitter in fitters ])

            p_final = np.max(p, axis=0)
        else:
            p, bfpars = zip(*[ fitter(frequency) for fitter in fitters ])
            p_final = max(p)

        if save_best_model:
            if multiple_frequencies:
                j = np.argmax(p_final)
                i = np.argmax([ P[j] for P in p ])

                self.best_model = TemplateModel(self.templates[i], frequency = frequency[j], 
                                                          parameters = bfpars[i][j])
            else:
                i = np.argmax(p)
                self.best_model = TemplateModel(self.templates[i], frequency = frequency, 
                                                          parameters = bfpars[i])

        return p_final
