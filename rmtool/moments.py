"""Moment enumeration from a bivariate polynomial."""
import warnings

import sympy as sp

from .symbols import m, myu, z
from .transforms import Lmz2Lmyuz

__all__ = ["Lmz2MomS", "Lmz2MomF", "TLmz"]

_MAX_BRANCHES = 16


def _strip_z(expr):
    """Divide a polynomial in (myu, z) by the largest power of z dividing it."""
    expr = sp.expand(expr)
    if expr == 0:
        return expr
    P = sp.Poly(expr, myu, z)
    k = min(mon[1] for mon in P.monoms())
    if k:
        expr = sp.expand(sp.cancel(expr / z**k))
    return expr


def _is_one(v):
    return sp.simplify(v - 1) == 0


def Lmz2MomS(Lmz, max_moment=4):
    """Enumerate the first ``max_moment`` moments of the measure encoded by ``Lmz``.

    Returns a list ``[1, M1, M2, ..., M_max_moment]``.  If several power-series
    expansions about zero start with coefficient 1, all are returned as a list
    of lists (use positivity arguments to select the correct one).

    The algorithm substitutes ``myu -> M_k + myu*z`` iteratively into the
    moment polynomial ``Lmyuz`` and solves for the next coefficient.
    """
    Lmyuz = _strip_z(Lmz2Lmyuz(Lmz))
    L0 = Lmyuz.subs(z, 0)
    first = sp.solve(L0, myu) if L0.has(myu) else []
    if not any(_is_one(v) for v in first):
        warnings.warn(
            "No moments exist. The bivariate polynomial does not encode a valid "
            "probability measure."
        )
        return []

    sequences = []

    def recurse(L, seq):
        if len(sequences) >= _MAX_BRANCHES:
            return
        if len(seq) == max_moment + 1:
            sequences.append(seq)
            return
        Lnext = _strip_z(L.subs(myu, seq[-1] + myu * z))
        Lz0 = sp.expand(Lnext.subs(z, 0))
        if Lz0 == 0 or not Lz0.has(myu):
            return  # dead branch: no expansion
        roots = sp.solve(Lz0, myu)
        for root in roots:
            if root.is_real is False:
                continue
            recurse(Lnext, seq + [sp.simplify(root)])

    recurse(Lmyuz, [sp.Integer(1)])

    if not sequences:
        warnings.warn(
            "No moments exist. The bivariate polynomial either does not encode a "
            "valid probability measure or the measure is non-compact."
        )
        return [sp.Integer(1)] + [sp.nan] * max_moment
    if len(sequences) > 1:
        warnings.warn(
            f"There are {len(sequences)} expansions about zero with initial "
            "coefficient equal to 1. All have been returned. Eliminate some of "
            "them by positivity arguments."
        )
        return sequences
    return sequences[0]


def Lmz2MomF(Lmz, max_moment=4):
    """Fast moment enumeration.

    The MATLAB version relied on Maple's ``gfun[algeqtoseries]``; in Python
    this simply delegates to :func:`Lmz2MomS`.
    """
    return Lmz2MomS(Lmz, max_moment)


def TLmz(Lmz):
    """(Dm+1) x (Dz+1) coefficient matrix: entry (i, j) is the coefficient of
    ``m**i * z**j`` in ``Lmz``."""
    P = sp.Poly(sp.expand(Lmz), m, z)
    Dm, Dz = P.degree(m), P.degree(z)
    C = sp.zeros(Dm + 1, Dz + 1)
    for (i, j), coeff in zip(P.monoms(), P.coeffs()):
        C[i, j] = coeff
    return C
