"""Bivariate polynomials for standard ensembles and atomic measures."""
import sympy as sp

from .core import numden
from .symbols import c as _c
from .symbols import g, m, z
from .transforms import Lgz2Lmz

__all__ = ["wignerpol", "wishartpol", "atomLmz", "equiLmz"]


def wignerpol():
    """Semicircle law (Wigner matrix): m^2 + m*z + 1."""
    return m**2 + m * z + 1


def wishartpol(c=_c):
    """Marcenko-Pastur law (Wishart matrix W = G*G'/N, c = n/N).

    ``c`` may be numeric or symbolic (default: the symbol ``c``).
    """
    c = sp.nsimplify(c, rational=True)
    return sp.collect(m * (1 - c - c * m * z - z) - 1, m)


def atomLmz(masses, weights):
    """Atomic measure with atoms at ``masses`` and weights ``weights``.

    Example: ``atomLmz([1, 2], [0.5, 0.5])`` is a matrix with half its
    eigenvalues equal to 1 and the rest equal to 2.
    """
    if len(masses) != len(weights):
        raise ValueError("masses and weights must have the same length")
    total = 0
    for a, w in zip(masses, weights):
        a = sp.nsimplify(a, rational=True)
        w = sp.nsimplify(w, rational=True)
        total += w / (a - z)
    return numden(m - total)


def equiLmz(t, M):
    """Equilibrium measure for the potential V(x) = t * x^(2M).

    Example: ``equiLmz(1, 1)`` returns a (scaled) semicircle law.
    """
    t = sp.nsimplify(t, rational=True)
    M = int(M)
    a = sp.Integer(1)
    for l in range(1, M + 1):
        a = a * sp.Rational(2 * l - 1, 2 * l)
    a = (a * M * t) ** sp.Rational(-1, 2 * M)

    h1 = z ** (2 * M - 2)
    for j in range(1, M):
        scaling = sp.Integer(1)
        for l in range(1, j + 1):
            scaling = scaling * sp.Rational(2 * l - 1, 2 * l)
        h1 = h1 + z ** (2 * M - 2 - 2 * j) * a ** (2 * j) * scaling

    Lgz = sp.simplify((g / (M * t) - z ** (2 * M - 1)) ** 2 - (z**2 - a**2) * h1**2)
    return sp.collect(numden(Lgz2Lmz(Lgz)), m)
