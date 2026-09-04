"""RMTool: a random matrix and free probability calculator (Python port).

Original MATLAB toolbox by N. Raj Rao (MIT, 2006), based on
N. R. Rao and A. Edelman, "The polynomial method for random matrices",
Foundations of Computational Mathematics, 2008 (arXiv:math/0601389).

Quick start::

    from rmtool import *
    b = wishartpol(0.5)
    Lmz2MomS(b, 10)
    info = Lmz2pdf(b, np.arange(-0.05, 5, 0.01))
    plt.plot(info.range, info.density)
"""
from .core import L1plusL2, L1timesL2, Luv2Cu, irreducLuv, numden
from .density import Lmz2pdf, PdfInfo
from .ensembles import atomLmz, equiLmz, wignerpol, wishartpol
from .moments import Lmz2MomF, Lmz2MomS, TLmz
from .operations import (AblockB, AgramWish, AplusB, AplusBkernel, AtimesB,
                         AtimesBkernel, AtimesWish, addAdtimes, compressA,
                         corrWish, invA, mobiusA, scaleA, shiftA, squareA,
                         transposeA)
from .sampling import haar, histw, theoryvssim, wigner, wishart
from .symbols import c, eta, g, m, myu, r, s, y, z
from .transforms import (Letaz2Lmz, Lgz2Lmz, Lgz2Lrg, Lmyuz2Lmz, Lmz2Letaz,
                         Lmz2Lgz, Lmz2Lmyuz, Lmz2Lrg, Lmz2Lsy, Lrg2Lgz,
                         Lrg2Lmz, Lsy2Lmz)

__version__ = "1.0.0"

__all__ = [
    # symbols
    "m", "z", "c", "r", "g", "s", "y", "myu", "eta",
    # core
    "numden", "irreducLuv", "Luv2Cu", "L1plusL2", "L1timesL2",
    # transforms
    "Lmz2Lgz", "Lgz2Lmz", "Lgz2Lrg", "Lrg2Lgz", "Lmz2Lrg", "Lrg2Lmz",
    "Lmz2Lsy", "Lsy2Lmz", "Lmz2Lmyuz", "Lmyuz2Lmz", "Lmz2Letaz", "Letaz2Lmz",
    # ensembles
    "wignerpol", "wishartpol", "atomLmz", "equiLmz",
    # operations
    "AplusB", "AtimesB", "AblockB", "compressA", "mobiusA", "invA", "scaleA",
    "shiftA", "transposeA", "squareA", "AtimesWish", "AgramWish", "corrWish",
    "addAdtimes", "AplusBkernel", "AtimesBkernel",
    # moments & density
    "Lmz2MomS", "Lmz2MomF", "TLmz", "Lmz2pdf", "PdfInfo",
    # sampling
    "wigner", "wishart", "haar", "histw", "theoryvssim",
]
