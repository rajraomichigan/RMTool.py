"""Core bivariate-polynomial machinery (ports of irreducLuv, Luv2Cu, L1plusL2, L1timesL2)."""
import warnings

import sympy as sp

__all__ = ["numden", "irreducLuv", "Luv2Cu", "L1plusL2", "L1timesL2"]


def numden(expr):
    """Clear the denominator of a rational expression and return the numerator.

    MATLAB equivalent: ``numden``.  Floating point constants such as ``0.5``
    are converted to exact rationals so that factorisation works reliably.
    """
    expr = sp.nsimplify(sp.sympify(expr), rational=True)
    num, _ = sp.fraction(sp.cancel(sp.together(expr)))
    return sp.expand(num)


def irreducLuv(Luv, u, v, return_minimal=False):
    """Make a bivariate polynomial ``Luv`` irreducible with respect to ``u`` and ``v``.

    Denominators are cleared, the polynomial is factored, and spurious factors
    (constants, powers of ``v`` alone, repeated factors) are discarded.  If more
    than one factor involving ``u`` survives, their product is returned and
    ``minimal`` is ``False``; the "right factor" may then need to be identified
    by hand (e.g. by checking which factor admits a valid moment expansion).

    Parameters
    ----------
    Luv : sympy expression
    u, v : sympy symbols
    return_minimal : bool
        If True, return ``(Luv, minimal)``.
    """
    Luv = numden(Luv)
    if Luv == 0:
        return (Luv, True) if return_minimal else Luv

    _, factors = sp.factor_list(Luv)
    keep = [fac for fac, _mult in factors if fac.has(u)]
    if not keep:
        result = sp.collect(Luv, u)
        return (result, True) if return_minimal else result

    both = [fac for fac in keep if fac.has(v)]
    if both:
        keep = both

    minimal = len(keep) == 1
    result = keep[0]
    for fac in keep[1:]:
        result = result * fac
    result = sp.collect(sp.expand(result), u)

    if not minimal and not return_minimal:
        warnings.warn(
            "Polynomial can be factored further; it may be useful to identify "
            "the correct factor. Use sympy.factor(L) to see the factorization."
        )
    return (result, minimal) if return_minimal else result


def Luv2Cu(Luv, u):
    """Companion matrix of ``Luv`` viewed as a (monic) polynomial in ``u``.

    The characteristic polynomial of the returned matrix equals ``Luv``
    divided by its leading coefficient in ``u``.
    """
    P = sp.Poly(sp.expand(Luv), u)
    Du = P.degree()
    coeffs = P.all_coeffs()  # highest degree first
    lead = coeffs[0]
    monic = [sp.cancel(cf / lead) for cf in coeffs]
    Cu = sp.zeros(Du, Du)
    for i in range(Du - 1):
        Cu[i, i + 1] = 1
    for i in range(Du):
        Cu[Du - 1, i] = -monic[Du - i]  # coefficient of u**i
    return Cu


def _same(L1, L2):
    return sp.expand(L1 - L2) == 0


def L1plusL2(Luv1, Luv2, u):
    """Bivariate polynomial whose zeros in ``u`` are the sums of the zeros of
    ``Luv1`` and ``Luv2`` (computed with a resultant).

    When ``Luv1 == Luv2`` the zeros are simply doubled, which avoids
    introducing spurious branches.
    """
    if _same(Luv1, Luv2):
        return numden(Luv1.subs(u, u / 2))
    t = sp.Dummy("t")
    res = sp.resultant(sp.expand(Luv1), sp.expand(Luv2.subs(u, t - u)), u)
    return sp.expand(res.subs(t, u))


def L1timesL2(Luv1, Luv2, u):
    """Bivariate polynomial whose zeros in ``u`` are the products of the zeros
    of ``Luv1`` and ``Luv2`` (computed with a resultant).

    When ``Luv1 == Luv2`` the zeros are simply squared.
    """
    if _same(Luv1, Luv2):
        w = sp.Dummy("w")
        res = sp.resultant(sp.expand(Luv1.subs(u, w)), w**2 - u, w)
        return sp.expand(res)
    t = sp.Dummy("t")
    Du2 = sp.Poly(sp.expand(Luv2), u).degree()
    L2t = sp.expand(u**Du2 * Luv2.subs(u, t / u))
    res = sp.resultant(sp.expand(Luv1), L2t, u)
    return sp.expand(res.subs(t, u))
