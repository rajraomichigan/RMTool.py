"""Operations on random-matrix models expressed on their bivariate polynomials.

Every function takes/returns an ``Lmz`` polynomial: the bivariate polynomial
``L(m, z) = 0`` satisfied by the Stieltjes transform ``m(z)`` of the limiting
spectral measure.
"""
import sympy as sp

from .core import L1plusL2, L1timesL2, irreducLuv
from .symbols import m, r, s as _s, z
from .transforms import (Lmz2Lgz, Lmz2Lrg, Lmz2Lsy, Lrg2Lgz, Lrg2Lmz, Lsy2Lmz)

__all__ = [
    "AplusB", "AtimesB", "AblockB", "compressA", "mobiusA", "invA", "scaleA",
    "shiftA", "transposeA", "squareA", "AtimesWish", "AgramWish", "corrWish",
    "addAdtimes", "AplusBkernel", "AtimesBkernel",
]


def AplusB(LmzA, LmzB):
    """C = A + Q*B*Q' with Q Haar unitary (free additive convolution).

    Example: ``AplusB(atomLmz([1,-1],[1/2,1/2]), atomLmz([1,-1],[1/2,1/2]))``
    returns the arc-sine law.
    """
    LrgA = Lmz2Lrg(LmzA)
    LrgB = LrgA if sp.expand(LmzA - LmzB) == 0 else Lmz2Lrg(LmzB)
    LrgC = L1plusL2(LrgA, LrgB, r)
    return Lrg2Lmz(LrgC)


def AtimesB(LmzA, LmzB):
    """C = A * Q*B*Q' with Q Haar unitary (free multiplicative convolution).

    Assumes A*B has real eigenvalues (e.g. A, B positive semi-definite).
    """
    LsyA = Lmz2Lsy(LmzA)
    LsyB = Lmz2Lsy(LmzB)
    LsyC = L1timesL2(LsyA, LsyB, _s)
    return Lsy2Lmz(LsyC)


def AblockB(LmzA, LmzB, c):
    """C = diag(A, B) with c = size(A)/size(C)."""
    c = sp.nsimplify(c, rational=True)
    if c == 1:
        return LmzA
    if c == 0:
        return LmzB
    if sp.expand(LmzA - LmzB) == 0:
        return irreducLuv(LmzA, m, z)
    LmzA1 = LmzA.subs(m, m / c)
    LmzB1 = LmzB.subs(m, m / (1 - c))
    return irreducLuv(L1plusL2(LmzA1, LmzB1, m), m, z)


def compressA(LmzA, factor):
    """B = top n-by-n block of the N-by-N matrix Q*A*Q', factor = n/N < 1."""
    factor = sp.nsimplify(factor, rational=True)
    LmzT = scaleA(LmzA, factor)
    LrgT = Lmz2Lrg(LmzT)
    LrgB = LrgT.subs(r, r * factor)
    return Lrg2Lmz(LrgB)


def mobiusA(LmzA, p, q, rr, s):
    """B = (p*A + q*I) / (rr*A + s*I)."""
    p, q, rr, s = (sp.nsimplify(v, rational=True) for v in (p, q, rr, s))
    alpha = (q - s * z) / (p - rr * z)
    beta = 1 / (p - rr * z)
    temp = LmzA.subs(z, -alpha)
    temp = temp.subs(m, ((m / beta) - rr) / (s - rr * alpha))
    return irreducLuv(temp, m, z)


def invA(LmzA):
    """B = inv(A)."""
    return mobiusA(LmzA, 0, 1, 1, 0)


def scaleA(LmzA, alpha):
    """B = alpha * A."""
    return mobiusA(LmzA, alpha, 0, 0, 1)


def shiftA(LmzA, alpha):
    """B = A + alpha * I."""
    return mobiusA(LmzA, 1, alpha, 0, 1)


def transposeA(LmzA, c):
    """B = X'*X where A = X*X' and c = size(A)/size(B)."""
    c = sp.nsimplify(c, rational=True)
    LmzB = LmzA.subs(m, (1 - 1 / c) * (1 / (0 - z)) + m / c)
    return irreducLuv(LmzB, m, z)


def squareA(LmzA):
    """B = A^2."""
    sz = sp.sqrt(z)
    Lmz1 = LmzA.subs(z, sz).subs(m, 2 * m * sz)
    Lmz2 = LmzA.subs(z, -sz).subs(m, -2 * m * sz)
    LmzB = L1plusL2(sp.expand(Lmz1), sp.expand(Lmz2), m)
    return irreducLuv(LmzB, m, z)


def AtimesWish(LmzA, c):
    """B = A x W(c), a Wishart matrix with covariance A.

    W(c) = G*G'/N with G = randn(n, N) and c = n/N.
    """
    c = sp.nsimplify(c, rational=True)
    z1 = sp.Dummy("z1")
    temp = LmzA.subs(m, m * (1 - c - c * z1 * m))
    temp = temp.subs(z, z1 / (1 - c - c * z1 * m))
    temp = temp.subs(z1, z)
    LmzB, minimal = irreducLuv(temp, m, z, return_minimal=True)
    if not minimal:
        LmzB = irreducLuv(LmzB / (1 - c - c * z * m), m, z)
    return LmzB


def AgramWish(LmzA, c, s):
    """B = (A_s + sqrt(s)*G)(A_s + sqrt(s)*G)' with G = randn(n, N), c = n/N."""
    c = sp.nsimplify(c, rational=True)
    s = sp.nsimplify(s, rational=True)
    temp = LmzA.subs(m, m / (1 + s * c * m))
    temp = temp.subs(z, (1 + s * c * m) * (z * (1 + s * c * m) + s * (c - 1)))
    LmzB, minimal = irreducLuv(temp, m, z, return_minimal=True)
    if not minimal:
        LmzB = irreducLuv(LmzB / (1 + s * c * m), m, z)
    return LmzB


def corrWish(LmzA, LmzB, c):
    """Spatio-temporally correlated Wishart: C = Y*Y' with
    Y = A^{1/2} G B^{1/2}, G = randn(n, N), c = n/N."""
    c = sp.nsimplify(c, rational=True)
    LmzW = m * (1 - c - c * m * z - z) - 1
    LmzWt = transposeA(LmzW, c)
    LmzT = AtimesB(LmzWt, LmzB)
    LmzTt = transposeA(LmzT, 1 / c)
    return AtimesB(LmzA, LmzTt)


def addAdtimes(LmzA, d):
    """Measure whose R-transform is d times that of A (R_B = d * R_A)."""
    d = sp.nsimplify(d, rational=True)
    LrgA = Lmz2Lrg(LmzA)
    LrgB = LrgA.subs(r, r / d)
    return Lrg2Lmz(LrgB)


# ---------------------------------------------------------------------------
# Experimental kernel polynomials (ports of AplusBkernel / AtimesBkernel).
# ---------------------------------------------------------------------------

def AplusBkernel(LmzA1, LmzA2):
    """Experimental: trivariate polynomial L(m, x, z) for the A + B kernel.

    Returns ``(Lmxz, LmzSum)``.
    """
    gg, ff, xx = sp.symbols("g f x")
    LgzA1 = Lmz2Lgz(LmzA1)
    LrgA1 = Lmz2Lrg(LmzA1)
    LrgA2 = Lmz2Lrg(LmzA2)
    LrgSum = L1plusL2(LrgA1, LrgA2, r)
    LmzSum = Lrg2Lmz(LrgSum)
    LgzSum = Lmz2Lgz(LmzSum)
    LgfA1 = LgzA1.subs(z, ff)
    Lfz = sp.resultant(sp.expand(LgfA1), sp.expand(LgzSum), gg)
    Lmxz = Lfz.subs(ff, xx - 1 / m)
    return irreducLuv(Lmxz, m, z), LmzSum


def AtimesBkernel(LmzA1, LmzA2):
    """Experimental: trivariate polynomial L(m, x, z) for the A x B kernel."""
    gg, ff, xx = sp.symbols("g f x")
    LmzProd = AtimesB(LmzA1, LmzA2)
    LgzProd = Lmz2Lgz(LmzProd)
    LgzA1 = Lmz2Lgz(LmzA1)
    LgfA2 = irreducLuv(LgzA1.subs({gg: gg * ff, z: 1 / ff}, simultaneous=True), gg, ff)
    LgzProd2 = irreducLuv(LgzProd.subs({gg: gg / z}, simultaneous=True), gg, z)
    Lfz = sp.resultant(sp.expand(LgfA2), sp.expand(LgzProd2), gg)
    Lmxz = Lfz.subs(ff, (1 + 1 / (z * m)) / xx)
    return irreducLuv(Lmxz, m, z)

