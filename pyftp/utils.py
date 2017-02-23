"""Utility Functions"""
import numpy as np
from scipy.special import eval_chebyu, eval_chebyt

#----------------------------------------------------------------------
# Functions to compute A, B, and their derivatives

# shortcuts for the Chebyshev polynomials
def Un(n, x):
    return np.where(np.asarray(n) >= 0, eval_chebyu(n, x), 0)

def Tn(n, x):
    return np.where(np.asarray(n) >= 0, eval_chebyt(n, x), 0)


def Afunc(n, x, p, q, sgn=1):
    return p * Tn(n, x) - sgn * q * Un(n-1, x) * np.sqrt(1 -  x*x)


def dAfunc(n, x, p, q, sgn=1):
    return n * (p * Un(n-1, x) + sgn * q * Tn(n, x) / np.sqrt(1 -  x*x))


def Avec(x, c, s, sgn=1):
    """Vector expression of A"""
    n = np.arange(1, len(s) + 1).reshape(s.shape)
    return Afunc(n, x, c, s, sgn=sgn)


def Bvec(x, c, s, sgn=1):
    """Vector expression of B"""
    n = np.arange(1, len(s) + 1).reshape(s.shape)
    return Afunc(n, x, s, -c, sgn=sgn)


def dAvec(x, c, s, sgn=1):
    """Vector expression of the derivative of A"""
    n = np.arange(1, len(s) + 1).reshape(s.shape)
    return dAfunc(n, x, c, s, sgn=sgn)


def dBvec(x, c, s, sgn=1):
    """Vector expression of the derivative of B"""
    n = np.arange(1, len(s) + 1).reshape(s.shape)
    return dAfunc(n, x, s, -c, sgn=sgn)
