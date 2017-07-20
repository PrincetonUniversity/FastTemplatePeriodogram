"""Utility functions"""
import numpy as np
from collections import namedtuple


# ----------------------------------------------------------------------
# Named tuples for storing values used in the package


Summations = namedtuple('Summations', ['C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS'])
ModelFitParams = namedtuple('ModelFitParams', ['a', 'b', 'c', 'sgn'])

AltModelFitParams = namedtuple('AltModelFitParams', ['theta_1', 'theta_2',
                                                     'theta_3'])

# ----------------------------------------------------------------------
# Misc. functions


def weights(err):
    """convert sigma_i -> w_i \equiv (1/W) * (1/sigma_i^2)"""
    w = np.power(err, -2)
    w /= np.sum(w)
    return w


def get_diags(mat):
    """get a sequence of diagonal traces for a matrix"""
    return np.array([sum(mat.diagonal(i))
                     for i in range(-mat.shape[0]+1, mat.shape[1])])


def power_from_fit(y, dy, yfit):
    """
    Periodogram value for a given model and set of observations

    Parameters
    ----------
    t: array_like
        Observation times.
    y: array_like
        Observations.
    dy: array_like
        Observation uncertainties.
    yfit: array_like
        Model fit.

    Returns
    -------
    power: float
        `1 - chi2 / chi2_0`; periodogram power for given model
    """
    w = weights(dy)
    ybar = np.dot(w, y)

    chi2_0 = np.dot(w, (y-ybar)**2)
    chi2 = np.dot(w, (y-yfit)**2)

    return 1. - (chi2 / chi2_0)
