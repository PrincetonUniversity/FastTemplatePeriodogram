from __future__ import print_function

from scipy.optimize import newton, brentq
from numpy.polynomial.polynomial import Polynomial
import numpy.polynomial.polynomial as pol

# ZERO FINDING UTILITIES. #################################################
# Using Sturm chains and bracketing zeros
#    this method is slower than the numpy polyroots() function that uses
#    the eigenvalues of the polynomial companion matrix.

def linzero(xlo, xhi, ylo, yhi):
    """ approximate the location of the zero contained within
        (xlo, xhi)
    """
    m = (yhi - ylo) / (xhi - xlo)
    b = ylo - m * xlo
    return -b/m


def hone_in(p, lo, hi, stop, count, max_count=10):
    """ improve estimate of the location of a zero
        by iteratively applying the secant method
        and refining the bracket
    """

    y_hi = pol.polyval(hi, p)
    y_lo = pol.polyval(lo, p)

    if y_hi * y_lo >= 0:
        raise Exception("y_lo and y_hi need different signs.")

    zero = linzero(lo, hi, y_lo, y_hi)

    fz = pol.polyval(zero, p)
    if zero - lo < stop or hi - zero < stop \
               or fz == 0 or count == max_count:
        return zero, count

    if fz * y_lo < 0:
        return hone_in(p, lo, zero, stop, count+1, max_count)

    else:
        return hone_in(p, zero, hi, stop, count+1, max_count)


def secant_zero(p, lo, hi, acc=1E-5):
    z, c = hone_in(p, lo, hi, acc, 0, max_count=100)
    return z


def brent_zero(p, lo, hi, acc=1E-5):
    f = lambda x, p=p : pol.polyval(x, p)
    return brentq(f, lo, hi)


def newton_zero(p, lo, hi, acc=1E-5):

    ylo = pol.polyval(lo, p)
    yhi = pol.polyval(hi, p)

    x0 = linzero(lo, hi, ylo, yhi)

    dp = pol.polyder(p)
    f = lambda x, p=p : pol.polyval(x, p)
    df = lambda x, dp=dp : pol.polyval(x, dp)

    return newton(f, x0, fprime=df)


def rem(p, d):
    return pol.polydiv(p, d)[1]


def n_sign_changes(p):
    n = 0
    for i in range(len(p) - 1):
        if p[i] * p[i+1] < 0: n += 1
    return n


def sigma(schain, x):
    return n_sign_changes([ pol.polyval(x, p) for p in schain ])


def sturm_chain(p):
    # Sturm chains:
    #   p0, p1, p2, ..., pm
    #
    #   p0 = p
    #   p1 = dp/dx
    #   p2 = -rem(p0, p1)
    #   p3 = -rem(p1, p2)
    #     ...
    #    0 = -rem(p(m-1), pm)
    #
    chains = [ p, pol.polyder(p) ]
    while True:
        pn = -rem(chains[-2], chains[-1])
        if len(pn) == 1 and abs(pn[0]) < 1E-14:
            break
        chains.append(pn)
    return chains


def bisect(a, b):
    d = 0.5 * (b - a)
    return [ (a, a + d), (a + d, b) ]


def sturm_zeros(p, a, b, acc=1E-5, zero_finder=brent_zero):
    chains = sturm_chain(p)

    brackets = [ (a, b) ]
    zeros = []
    while len(brackets) > 0:
        #print brackets
        x1, x2 = brackets.pop()
        if x2 - x1 < acc:
            zeros.append(x1)
            continue

        # use Sturm's theorem to get # of distinct
        # real zeros within [ x1, x2 ]
        n = sigma(chains, x1) - sigma(chains, x2)
        #print n, x1, x2

        if n == 0:
            continue
        elif n == 1 and pol.polyval(x1, p) * pol.polyval(x2, p) < 0:
            zeros.append(zero_finder(p, x1, x2, acc=acc))
        else:
            brackets.extend(bisect(x1, x2))

    return zeros
