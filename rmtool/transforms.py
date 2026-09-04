"""Conversions between the bivariate polynomial representations of the
Stieltjes (m), Cauchy (g), R (r), S (s,y), moment (myu) and eta transforms."""
from .core import irreducLuv
from .symbols import eta, g, m, myu, r, s, y, z

__all__ = [
    "Lmz2Lgz", "Lgz2Lmz", "Lgz2Lrg", "Lrg2Lgz", "Lmz2Lrg", "Lrg2Lmz",
    "Lmz2Lsy", "Lsy2Lmz", "Lmz2Lmyuz", "Lmyuz2Lmz", "Lmz2Letaz", "Letaz2Lmz",
]


def Lmz2Lgz(Lmz):
    """Stieltjes (m) -> Cauchy (g = -m)."""
    return Lmz.subs(m, -g)


def Lgz2Lmz(Lgz):
    """Cauchy (g) -> Stieltjes (m = -g)."""
    return Lgz.subs(g, -m)


def Lgz2Lrg(Lgz):
    """Cauchy (g,z) -> R-transform (r,g): z = r + 1/g."""
    return irreducLuv(Lgz.subs(z, r + 1 / g), r, g)


def Lrg2Lgz(Lrg):
    """R-transform (r,g) -> Cauchy (g,z): r = z - 1/g."""
    return irreducLuv(Lrg.subs(r, z - 1 / g), g, z)


def Lmz2Lrg(Lmz):
    """Stieltjes (m,z) -> R-transform (r,g)."""
    return Lgz2Lrg(Lmz2Lgz(Lmz))


def Lrg2Lmz(Lrg):
    """R-transform (r,g) -> Stieltjes (m,z)."""
    return irreducLuv(Lgz2Lmz(Lrg2Lgz(Lrg)), m, z)


def Lmz2Lsy(Lmz):
    """Stieltjes (m,z) -> S-transform (s,y)."""
    Lsy = Lmz.subs(m, -y * s)
    Lsy = Lsy.subs(z, (y + 1) / (y * s))
    return irreducLuv(Lsy, s, y)


def Lsy2Lmz(Lsy):
    """S-transform (s,y) -> Stieltjes (m,z)."""
    Lmz = Lsy.subs(s, -m / y)
    Lmz = Lmz.subs(y, -z * m - 1)
    return irreducLuv(Lmz, m, z)


def Lmz2Lmyuz(Lmz):
    """Stieltjes (m,z) -> moment series (myu,z): myu(z) = sum_k M_k z^k."""
    Lmyuz = Lmz.subs(z, 1 / z)
    Lmyuz = Lmyuz.subs(m, -myu * z)
    return irreducLuv(Lmyuz, myu, z)


def Lmyuz2Lmz(Lmyuz):
    """Moment series (myu,z) -> Stieltjes (m,z)."""
    Lmz = Lmyuz.subs(z, 1 / z)
    Lmz = Lmz.subs(myu, -m * z)
    return irreducLuv(Lmz, m, z)


def Lmz2Letaz(Lmz):
    """Stieltjes (m,z) -> eta-transform (eta,z)."""
    Letaz = Lmz.subs(z, -1 / z)
    Letaz = Letaz.subs(m, z * eta)
    return irreducLuv(Letaz, eta, z)


def Letaz2Lmz(Letaz):
    """eta-transform (eta,z) -> Stieltjes (m,z)."""
    Lmz = Letaz.subs(z, -1 / z)
    Lmz = Lmz.subs(eta, -z * m)
    return irreducLuv(Lmz, m, z)
