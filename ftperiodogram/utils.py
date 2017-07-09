"""Utility Functions"""
import numpy as np
from collections import namedtuple


#----------------------------------------------------------------------
# Named tuples for storing values used in the package
Summations = namedtuple('Summations', ['C', 'S', 'YC', 'YS', 'CC', 'CS', 'SS'])
ModelFitParams = namedtuple('ModelFitParams', ['a', 'b', 'c', 'sgn'])


#----------------------------------------------------------------------
# Misc. functions

def weights(err):
    """ converts sigma_i -> w_i \equiv (1/W) * (1/sigma_i^2) """
    w = np.power(err, -2)
    w/= np.sum(w)
    return w
