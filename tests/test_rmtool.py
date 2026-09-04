import warnings

import numpy as np
import sympy as sp

from rmtool import *  # noqa: F401,F403

warnings.simplefilter("ignore")
half = sp.Rational(1, 2)


def test_wishart_moments():
    assert Lmz2MomS(wishartpol(c), 4) == [1, 1, c + 1, c**2 + 3 * c + 1, c**3 + 6 * c**2 + 6 * c + 1]


def test_wigner_moments():
    assert Lmz2MomS(wignerpol(), 6) == [1, 0, 1, 0, 2, 0, 5]


def test_arcsine_from_free_sum():
    A = atomLmz([1, -1], [half, half])
    L = AplusB(A, A)
    assert sp.expand(L - (m**2 * z**2 - 4 * m**2 - 1)) == 0


def test_wigner_plus_wishart_moments():
    W = AplusB(wignerpol(), wishartpol(c))
    assert Lmz2MomS(W, 3) == [1, 1, c + 2, c**2 + 3 * c + 4]


def test_AtimesWish_identity_is_wishart():
    L = AtimesWish(atomLmz([1], [1]), c)
    assert sp.simplify(L / wishartpol(c)).is_number


def test_squareA_wigner_is_MP1():
    assert Lmz2MomS(squareA(wignerpol()), 4) == [1, 1, 2, 5, 14]


def test_invA():
    assert Lmz2MomS(invA(atomLmz([1, 2], [half, half])), 2) == [1, sp.Rational(3, 4), sp.Rational(5, 8)]


def test_compressA_arcsine01():
    L = compressA(atomLmz([1, 0], [half, half]), half)
    assert Lmz2MomS(L, 3) == [1, half, sp.Rational(3, 8), sp.Rational(5, 16)]


def test_transposeA_wishart():
    L1 = transposeA(wishartpol(c), c)
    L2 = scaleA(wishartpol(1 / c), c)
    assert sp.simplify(L1 / L2).is_number


def test_roundtrips():
    for L in (wishartpol(c), wignerpol()):
        for fwd, bwd in ((Lmz2Lsy, Lsy2Lmz), (Lmz2Letaz, Letaz2Lmz), (Lmz2Lmyuz, Lmyuz2Lmz), (Lmz2Lrg, Lrg2Lmz)):
            assert sp.simplify(bwd(fwd(L)) / L).is_number


def test_density_mass():
    info = Lmz2pdf(wishartpol(half), np.arange(-0.05, 5, 0.01))
    d = np.nan_to_num(info.density)
    assert abs(np.trapezoid(d, info.range) - 1) < 1e-2
    assert abs(np.trapezoid(d * info.range, info.range) - 1) < 1e-2
    assert np.allclose(info.multipleroots, [(1 - np.sqrt(0.5)) ** 2, (1 + np.sqrt(0.5)) ** 2])


def test_theory_vs_simulation():
    rng = np.random.default_rng(0)
    e = np.concatenate([np.linalg.eigvalsh(wigner(100, rng=rng) + wishart(100, 200, rng=rng)) for _ in range(100)])
    info = Lmz2pdf(AplusB(wignerpol(), wishartpol(half)))
    cent, h = histw(e, 40, ax=False)
    th = np.interp(cent, info.range, np.nan_to_num(info.density))
    assert np.abs(h - th).max() < 0.05


def test_TLmz():
    assert TLmz(wishartpol(c)) == sp.Matrix([[-1, 0], [1 - c, -1], [0, -c]])
