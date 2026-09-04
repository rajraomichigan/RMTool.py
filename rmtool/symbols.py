"""Symbolic variables used throughout RMTool.

MATLAB equivalent: ``syms m z c r g s y myu eta``.

* ``m``   : Stieltjes transform variable, m(z) = int f(x)/(x - z) dx
* ``z``   : complex spectral variable
* ``g``   : Cauchy transform, g = -m
* ``r``   : R-transform variable
* ``s, y``: S-transform variables
* ``myu`` : moment generating variable (Lmyuz representation)
* ``eta`` : eta-transform variable
* ``c``   : generic Wishart aspect-ratio parameter (rows/columns)
"""
import sympy as sp

m, z, c, r, g, s, y, myu, eta = sp.symbols("m z c r g s y myu eta")

__all__ = ["m", "z", "c", "r", "g", "s", "y", "myu", "eta"]
