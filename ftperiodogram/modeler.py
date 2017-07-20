from __future__ import print_function

from functools import wraps
import warnings

import numpy as np
from scipy import optimize

from .utils import weights, ModelFitParams, AltModelFitParams, power_from_fit
from .template import Template
from . import core as pdg


# function wrappers for performing checks
def requires_templates(func):
    @wraps(func)
    def wrap(self, *args, **kwargs):
        self._validate_templates()
        return func(self, *args, **kwargs)
    return wrap


def requires_template(func):
    @wraps(func)
    def wrap(self, *args, **kwargs):
        self._validate_template()
        return func(self, *args, **kwargs)
    return wrap


def requires_data(func):
    @wraps(func)
    def wrap(self, *args, **kwargs):
        self._validate_data()
        return func(self, *args, **kwargs)
    return wrap


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
    parameters : ``ModelFitParams`` or ``AltModelFitParams``
        Parameters for the model (a, b, c, sgn); must be a ``ModelFitParams``
        instance or an ``AltModelFitParams`` instance
        (``theta_1``, ``theta_2``, ``theta_3``)

    Examples
    --------
    >>> params = ModelFitParams(a=1, b=1, c=0, sgn=1)
    >>> template = Template([ 1.0, 0.4, 0.2], [0.1, 0.9, 0.2])
    >>> model = TemplateModel(template, frequency=1.0, parameters=params)
    >>> t = np.linspace(0, 10, 100)
    >>> y_fit = model(t)
    >>> params_alt = ModelFitParams(theta_1=1, theta_2=0, theta_3=0)
    >>> model_alt = TemplateModel(template, frequency=1.0,
    ...                                     parameters=params_alt)
    >>> assert(max(np.absolute(model(t) - model_alt(t))) < 1E-9)
    """
    def __init__(self, template, frequency=1.0, parameters=None):
        self.template = template
        self.frequency = frequency
        self.parameters = parameters

        self._thetas = self.convert_to_theta(self.parameters)

    def _validate(self):
        if not isinstance(self.template, Template):
            raise TypeError("template must be a Template instance")
        if not isinstance(self.parameters, ModelFitParams) and\
           not isinstance(self.parameters, AltModelFitParams):
            raise TypeError("parameters must be `ModelFitParams` or "
                            "`AltModelFitParams` instance "
                            "(type={0})".format(type(self.parameters)))

    @staticmethod
    def convert_to_theta(parameters):
        """
        Convert `ModelFitParams` instance to `AltModelFitParams`
        instance
        """
        if isinstance(parameters, AltModelFitParams):
            return parameters

        if not isinstance(parameters, ModelFitParams):
            raise TypeError("parameters must be `ModelFitParams` or "
                            "`AltModelFitParams` instance "
                            "(type={0})".format(type(parameters)))

        theta_2 = np.arccos(parameters.b)
        if parameters.sgn == -1:
            theta_2 = 2 * np.pi - theta_2

        return AltModelFitParams(theta_1=parameters.a,
                                 theta_2=theta_2,
                                 theta_3=parameters.c)

    def __call__(self, t):
        self._validate()

        wt = t * self.frequency

        th1, th2, th3 = self._thetas

        return th1 * self.template(wt - th2) + th3


class FastTemplatePeriodogram(object):
    """Base class for template periodogram instances

    Fits a single template to the data. For
    fitting multiple templates, use the FastMultiTemplatePeriodogram

    Parameters
    ----------
    template : Template, optional
        Template to fit (must be Template instance). Doesn't have to be
        supplied to `__init__` but must be set before running `power`,
        `autopower`, or `fit_model`.

    Notes
    -----
    `allow_negative_amplitudes` is deprecated as of `v1.0.1`
    to simplify things. Earlier versions were inconsistent in
    whether or not they actually checked for negative amplitudes,
    since this was an experimental feature -- don't trust that
    earlier versions do anything useful with this parameter!
    """
    def __init__(self, template=None):
        self.template = template
        self.t, self.y, self.dy = None, None, None
        self.best_model = None

    def _validate_template(self):
        if self.template is None:
            raise ValueError("No template set.")
        if not isinstance(self.template, Template):
            raise ValueError("template is not a Template instance.")

    def _validate_data(self):
        if any([X is None for X in [self.t, self.y, self.dy]]):
            raise ValueError("One or more of t, y, dy is None; "
                             "fit(t, y, dy) must be called first.")
        inds = np.arange(len(self.t) - 1)
        if any(self.t[inds] > self.t[inds+1]):
            raise ValueError("One or more observations are not consecutive.")

        if not (len(self.t) == len(self.y) and len(self.y) == len(self.dy)):
            raise ValueError("One or more of (t, y, dy) arrays are"
                             " unequal lengths")

    def _validate_frequencies(self, frequencies):
        raise NotImplementedError()

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

        Returns
        -------
        self : FastTemplatePeriodogram
            Returns self
        """
        # TODO: validate dy when it is float or None

        self.t = np.array(t)
        self.y = np.array(y)
        self.dy = np.array(dy)
        return self

    @requires_data
    @requires_template
    def fit_model(self, freq, **kwargs):
        """Fit a template model to data.

        y_model(t) = th1 * template(freq * t - th2) + th3

        Parameters
        ----------
        freq : float
            Frequency at which to fit a template model
        **kwargs: dict, optional
            Passed to ``fit_template``

        Returns
        -------
        model : TemplateModel
            The best-fit model at this frequency
        """
        freq = float(freq)
        p, parameters = pdg.fit_template(self.t, self.y, self.dy,
                                         self.template.c_n, self.template.s_n,
                                         freq, **kwargs)
        return TemplateModel(self.template, parameters=parameters,
                             frequency=freq)

    @requires_data
    def autofrequency(self, nyquist_factor=5, samples_per_peak=5,
                      minimum_frequency=None, maximum_frequency=None,
                      **kwargs):
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
        baseline = self.t.max() - self.t.min()
        n_samples = self.t.size

        df = 1. / (baseline * samples_per_peak)

        if minimum_frequency is not None:
            nf0 = min([1, np.floor(minimum_frequency / df)])
        else:
            nf0 = 1

        if maximum_frequency is not None:
            Nf = int(np.ceil(maximum_frequency / df - nf0))
        else:
            Nf = int(0.5 * samples_per_peak * nyquist_factor * n_samples)

        return df * (nf0 + np.arange(Nf))

    @requires_data
    @requires_template
    def autopower(self, save_best_model=True, fast=True, **kwargs):
        """
        Compute template periodogram at automatically-determined frequencies

        Parameters
        ----------
        save_best_model : optional, bool (default = True)
            Save a TemplateModel instance corresponding to the best-fit model
            found
        **kwargs : optional, dict
            Passed to ``autofrequency`` and ``template_periodogram``

        Returns
        -------
        frequency, power : ndarray, ndarray
            The frequency and template periodogram power
        """
        frequency = self.autofrequency(**kwargs)
        p, bfpars = pdg.template_periodogram(self.t, self.y, self.dy,
                                             self.template.c_n,
                                             self.template.s_n,
                                             frequency, fast=fast, **kwargs)

        if save_best_model:
            i = np.argmax(p)
            self._save_best_model(TemplateModel(self.template,
                                                frequency=frequency[i],
                                                parameters=bfpars[i]))
        return frequency, p

    @requires_data
    @requires_template
    def power(self, frequency, save_best_model=True, **kwargs):
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
            Passed to ``fit_template``

        Returns
        -------
        power : float or ndarray
            The frequency and template periodogram power, a
        """
        # Allow inputs of any shape; we'll reshape output to match
        frequency = np.asarray(frequency)
        shape = frequency.shape
        frequency = frequency.ravel()

        def fitter(freq):
            return pdg.fit_template(self.t, self.y, self.dy,
                                    self.template.c_n, self.template.s_n,
                                    freq, **kwargs)

        p, bfpars = zip(*map(fitter, frequency))
        p = np.array(p)

        if save_best_model:
            i = np.argmax(p)
            best_model = TemplateModel(self.template,
                                       frequency=frequency[i],
                                       parameters=bfpars[i])
            self._save_best_model(best_model)

        return p.reshape(shape)

    def _save_best_model(self, model, overwrite=False):
        if overwrite or self.best_model is None:
            self.best_model = model
        else:
            # if there is an existing best model, replace
            # with new best model only if new model improves fit
            y_fit = model(self.t)
            y_best = self.best_model(self.t)

            p_fit = power_from_fit(self.y, self.dy, y_fit)
            p_best = power_from_fit(self.y, self.dy, y_best)

            if p_fit > p_best:
                self.best_model = model


class FastMultiTemplatePeriodogram(FastTemplatePeriodogram):
    """
    Template modeler that fits multiple templates

    Parameters
    ----------
    templates : list of Template
        Templates to fit (must be list of Template instances)

    Notes
    -----
    ``allow_negative_amplitudes`` is deprecated as of ``v1.0.1``
    to simplify things. Earlier versions were inconsistent in
    whether or not they actually checked for negative amplitudes,
    since this was an experimental feature -- don't trust that
    earlier versions do anything useful with this parameter!
    """
    def __init__(self, templates=None, **kwargs):
        self.templates = templates
        self.t, self.y, self.dy = None, None, None
        self.best_model = None

    def _validate_templates(self):
        if self.templates is None:
            raise ValueError("No templates set.")
        if not hasattr(self.templates, '__iter__'):
            raise ValueError("<object>.templates must be iterable.")
        if len(self.templates) == 0:
            raise ValueError("No templates.")
        for template in self.templates:
            if not isinstance(template, Template):
                raise ValueError("One or more templates are not "
                                 "Template instances.")

    @requires_data
    @requires_templates
    def fit_model(self, freq, **kwargs):
        """Fit a template model to data.

        y_model(t) = th1 * template(freq * t - th2) + th3

        Parameters
        ----------
        freq : float
            Frequency at which to fit a template model
        **kwargs : dict, optional
            Passed to ``fit_template``

        Returns
        -------
        model : TemplateModel
            The best-fit model at this frequency
        """
        if not any([isinstance(freq, type_)
                    for type_ in [float, np.floating]]):
            raise ValueError('fit_model requires float argument')

        p, parameters = zip(*[pdg.fit_template(self.t, self.y, self.dy,
                                               template.c_n, template.s_n,
                                               freq, **kwargs)
                              for template in self.templates])

        i = np.argmax(p)
        params = parameters[i]
        template = self.templates[i]

        return TemplateModel(template, parameters=params, frequency=freq)

    @requires_data
    @requires_templates
    def autopower(self, save_best_model=True, fast=True, **kwargs):
        """
        Compute template periodogram at automatically-determined frequencies

        Parameters
        ----------
        save_best_model : optional, bool (default = True)
            Save a TemplateModel instance corresponding to the best-fit model
            found
        **kwargs : optional, dict
            Passed to ``autofrequency`` and ``template_periodogram``

        Returns
        -------
        frequency, power : ndarray, ndarray
            The frequency and template periodogram power
        """
        frequency = self.autofrequency(**kwargs)

        results = [pdg.template_periodogram(self.t, self.y, self.dy,
                                            template.c_n, template.s_n,
                                            frequency, fast=fast, **kwargs)
                   for template in self.templates]

        p, bfpars = zip(*results)
        if save_best_model:
            maxes = [max(P) for P in p]

            i = np.argmax(maxes)
            j = np.argmax(p[i])

            self._save_best_model(TemplateModel(self.templates[i],
                                                frequency=frequency[j],
                                                parameters=bfpars[i][j]))

        return frequency, np.max(p, axis=0)

    @requires_data
    def power_from_single_template(self, frequency, template, fast=False,
                                   save_best_model=True, **kwargs):
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

        # Allow inputs of any shape; we'll reshape output to match
        frequency = np.asarray(frequency)
        shape = frequency.shape
        frequency = frequency.ravel()

        p, bfpars = pdg.template_periodogram(self.t, self.y, self.dy,
                                             template.c_n, template.s_n,
                                             frequency, fast=fast, **kwargs)
        p = np.asarray(p)
        if save_best_model:
            i = np.argmax(p)
            best_model = TemplateModel(template, frequency=frequency[i],
                                       parameters=bfpars[i])
            self._save_best_model(best_model)

        return p.reshape(shape)

    @requires_data
    @requires_templates
    def power(self, frequency, save_best_model=True, fast=False, **kwargs):
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

        power = self.power_from_single_template
        all_power = [power(frequency, template, fast=fast,
                           save_best_model=save_best_model)
                     for template in self.templates]

        return np.max(all_power, axis=0)


class SlowTemplatePeriodogram(object):
    """Slow periodogram built from a template model.

    When computing the periodogram, this performs a nonlinear optimization at
    each frequency. This is used mainly for testing the faster method
    available in FastTemplatePeriodogram

    Parameters
    ----------
    template : Template object
        callable object that returns the template value as a function of phase
    nguesses : int, optional (default: 10)
        number of initial guesses for the phase shift parameter (to avoid local
        minima)
    ** minimize_kwargs: dict, optional
        Passed to the `scipy.optimize.minimize` function (or `minimize_scalar`
        if `nguesses` is None)
    """
    # TODO: match the full API of FastTemplateModeler.
    # Perhaps factor-out common routines into a base class?

    def __init__(self, template=None, nguesses=10, **minimize_kwargs):
        self.template = template
        self.nguesses = nguesses
        self.minimize_kwargs = minimize_kwargs

        if 'bounds' not in self.minimize_kwargs:
            # minimize_scalar takes a single tuple
            if self.nguesses is None:
                self.minimize_kwargs['bounds'] = (0, 1)

            # minimize() takes a list of tuples (for each dimension)
            else:
                self.minimize_kwargs['bounds'] = [(0, 1)]

    def fit(self, t, y, dy=None):
        """Fit periodogram to given data

        Parameters
        ----------
        t : array_like
            sequence of observation times
        y : array_like
            sequence of observations associated with times t
        dy : float, array_like (optional)
            error or sequence of observational errors associated with times t
        """
        self.t, self.y, self.dy = self._validate_inputs(t, y, dy)
        return self

    def _validate_inputs(self, t, y, dy):
        """ Run consistency checks on data """
        if dy is None:
            # TODO: handle dy = None case more efficiently
            t, y, dy = np.broadcast_arrays(t, y, 1.0)
        else:
            t, y, dy = np.broadcast_arrays(t, y, dy)
        if t.ndim != 1:
            raise ValueError("Inputs (t, y, dy) must be 1-dimensional")
        return t, y, dy

    def _chi2_ref(self):
        """ Compute the reference chi-square """
        weights = self.dy ** -2
        weights /= weights.sum()
        ymean = np.dot(weights, self.y)
        return np.sum((self.y - ymean) ** 2 / self.dy ** 2)

    def _minimize_chi2_at_single_freq(self, freq, nguesses=None):
        """
        Finds optimal phase-shift of template via non-linear optimization
        with `scipy.optimize.minimize` or `scipy.optimize.minimize_scalar`.

        Parameters
        ----------
        freq: float
            Frequency of signal
        nguesses: int, optional (default is self.nguesses)
            Number of initial guesses to start the non-linear minimization
        """
        # at each phase, use a linear model to find best [offset, amplitude]
        # and then minimize this scalar function of phase
        def chi2(phase):
            shifted = self.template(self.t * freq - phase)
            X = np.vstack([np.ones_like(shifted), shifted]).T
            offset, amp = np.linalg.solve(np.dot(X.T, X),
                                          np.dot(X.T, self.y))
            y_model = offset + amp * shifted
            return np.sum((self.y - y_model) ** 2 / self.dy ** 2)

        nguesses = nguesses if nguesses is not None else self.nguesses

        # User can opt to run minimize scalar (faster)
        if nguesses is None:
            return optimize.minimize_scalar(chi2, **self.minimize_kwargs)

        # initial guesses for phase shift (linearly spaced)
        guesses = np.linspace(0 + 0.5 / nguesses, 1, nguesses)

        def local_minimum(x0):
            return optimize.minimize(chi2, x0,
                                     **self.minimize_kwargs)

        # get solutions for each initial guess
        local_minima = [local_minimum(guess) for guess in guesses]

        # keep the successful solutions
        local_minima = [res for res in local_minima if res.success]

        # if no optimizations are successful, just return results from
        # `minimize_scalar`
        if len(local_minima) == 0:
            warnings.warn(" ".join(["`scipy.optimize.minimize` did not find",
                                    "an optimal solution for",
                                    "freq={freq};".format(freq=freq),
                                    "now running `minimize_scalar` with",
                                    "bounds=(0, 1)"]))
            return optimize.minimize_scalar(chi2, bounds=(0, 1))

        # return the best one
        return local_minima[np.argmin([res.fun for res in local_minima])]

    def power(self, freq):
        """Compute a template-based periodogram at the given frequencies

        Parameters
        ----------
        freq : array_like
            frequencies at which to evaluate the template periodogram

        Returns
        -------
        power : np.ndarray
            normalized power spectrum computed at the given frequencies
        """
        freq = np.asarray(freq)
        results = list(map(self._minimize_chi2_at_single_freq, freq.flat))
        failures = sum([not res.success for res in results])
        if failures:
            raise RuntimeError("{0}/{1} frequency values failed to converge"
                               "".format(failures, freq.size))
        chi2 = np.array([res.fun for res in results])
        return np.reshape(1 - chi2 / self._chi2_ref(), freq.shape)
