"""Numerical extraction of the density from a bivariate polynomial (Lmz2pdf)."""
import warnings
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from .symbols import m, z

__all__ = ["PdfInfo", "Lmz2pdf"]


@dataclass
class PdfInfo:
    """Result of :func:`Lmz2pdf`.

    Attributes
    ----------
    range : ndarray
        Points at which the polynomial was evaluated.
    rr : ndarray, shape (len(range), Dm)
        All roots in ``m`` at each point of ``range``.
    density : ndarray
        Normalised imaginary parts (imag / pi) of the roots.  For degree
        Dm <= 3 this is 1-D and equals the density; for Dm > 3 it is 2-D with
        one column per root, and the "right root" must be isolated manually.
    real : ndarray
        Real part of the same roots.
    poles : ndarray
        Real roots of the leading coefficient of ``Lmz`` in ``m`` (locations
        of possible atoms / poles).
    multipleroots : ndarray
        Real roots of the discriminant of ``Lmz`` w.r.t. ``m``
        (candidate boundary points of the support).
    """

    range: np.ndarray
    rr: np.ndarray
    density: np.ndarray
    real: np.ndarray
    poles: np.ndarray = field(default_factory=lambda: np.array([]))
    multipleroots: np.ndarray = field(default_factory=lambda: np.array([]))

    def best_density(self):
        """Largest imag/pi root at each point (a heuristic for Dm > 3)."""
        if self.density.ndim == 1:
            return self.density
        d = np.where(np.isnan(self.density), -np.inf, self.density)
        out = d.max(axis=1)
        out[~np.isfinite(out)] = np.nan
        return out


def _real_roots(expr, var):
    """Real roots of a univariate polynomial expression (numeric)."""
    P = sp.Poly(sp.expand(expr), var)
    if P.degree() <= 0:
        return np.array([])
    coeffs = np.array([complex(cf) for cf in P.all_coeffs()])
    rts = np.roots(coeffs)
    rts = rts[np.abs(rts.imag) < 1e-9].real
    return np.sort(rts)


def Lmz2pdf(Lmz, xx=None, params=None):
    """Evaluate the algebraic curve ``Lmz(m, z) = 0`` along the real axis.

    Parameters
    ----------
    Lmz : sympy expression
        Bivariate polynomial in ``m`` and ``z``.
    xx : array-like, optional
        Grid on which to compute the density.  If omitted, a grid covering the
        poles and candidate boundary points is chosen automatically.
    params : dict, optional
        Values for any extra symbolic parameters (e.g. ``{c: 0.5}``).  Any
        remaining free symbol is set to 0.5 with a warning.

    Returns
    -------
    PdfInfo
    """
    Lmz = sp.sympify(Lmz)
    if params:
        Lmz = Lmz.subs(params)
    extra = sorted(Lmz.free_symbols - {m, z}, key=str)
    if extra:
        warnings.warn(
            f"Lmz has symbolic variables other than m and z; setting {extra} to 0.5"
        )
        Lmz = Lmz.subs({sym: sp.Rational(1, 2) for sym in extra})
    if not (Lmz.has(m) and Lmz.has(z)):
        raise ValueError(f"Input argument Lmz = {Lmz} is not a polynomial in m and z")

    P = sp.Poly(sp.expand(Lmz), m)
    Dm = P.degree()
    coeff_exprs = P.all_coeffs()  # highest degree first
    poles = _real_roots(coeff_exprs[0], z) if coeff_exprs[0].has(z) else np.array([])

    disc = sp.discriminant(sp.expand(Lmz), m)
    if not disc.has(z):
        boundary = np.array([])
        warnings.warn(
            "Possibly atomic measure encoded by the bivariate polynomial. "
            "Examine pdfinfo.poles and pdfinfo.real."
        )
    else:
        boundary = _real_roots(disc, z)

    if xx is None:
        pts = np.concatenate([boundary, poles])
        if pts.size == 0:
            pts = np.array([-1.0, 1.0])
        xmin, xmax = pts.min() - 1, pts.max() + 1
        step = min((xmax - xmin) / 100, 0.05)
        xx = np.arange(xmin, xmax + step / 2, step)
        warnings.warn(
            f"Region of support not specified. Using range [{xmin:g}, {xmax:g}] "
            f"in steps of {step:g}. Inspect pdfinfo.multipleroots to refine."
        )
    xx = np.asarray(xx, dtype=float)

    coeff_fns = [sp.lambdify(z, cf, "numpy") for cf in coeff_exprs]
    rr = np.zeros((len(xx), Dm), dtype=complex)
    for i, xv in enumerate(xx):
        cs = np.array([complex(fn(xv)) for fn in coeff_fns])
        rts = np.roots(cs) if np.any(cs != 0) else np.array([])
        rr[i, : len(rts)] = rts
        if len(rts) < Dm:
            rr[i, len(rts):] = np.nan

    if Dm > 3:
        warnings.warn("All roots returned -- isolate correct root manually")
        root = rr.real + 1j * rr.imag / np.pi
    else:
        root = np.full(len(xx), np.nan, dtype=complex)
        for i in range(len(xx)):
            im = np.nan_to_num(rr[i].imag)
            pos = rr[i][im > 1e-10]
            if pos.size:
                best = pos[np.argmax(pos.imag)]
                root[i] = best.real + 1j * best.imag / np.pi

    return PdfInfo(
        range=xx,
        rr=rr,
        density=root.imag,
        real=root.real,
        poles=poles,
        multipleroots=boundary,
    )
