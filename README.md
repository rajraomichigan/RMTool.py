# RMTool — a random matrix and free probability calculator (Python)

Python port of the [RMTool](http://www.mit.edu/~raj/rmtool) MATLAB toolbox by
N. Raj Rao. RMTool computes the *limiting spectral measure* of a large class of
random matrix models **symbolically**, using the polynomial method of

> N. R. Rao and A. Edelman, *The polynomial method for random matrices*,
> Foundations of Computational Mathematics 8 (2008). [arXiv:math/0601389](https://arxiv.org/abs/math/0601389)

A probability measure is represented by the bivariate polynomial `L(m, z) = 0`
satisfied by its Stieltjes transform `m(z) = ∫ f(x)/(x − z) dx`. Operations on
random matrices (free sums, free products, Wishart-type transformations, Möbius
transforms, compressions, …) become operations on these polynomials, carried out
with [SymPy](https://www.sympy.org). Densities and moments are then extracted
from the resulting polynomial.

## Installation

```bash
pip install rmtool            # core (sympy + numpy)
pip install "rmtool[plot]"    # adds matplotlib for histw / theoryvssim
```

## Quick start

```python
import numpy as np, matplotlib.pyplot as plt
from rmtool import *

b = wishartpol(0.5)                          # Marcenko-Pastur, c = n/N = 0.5
print(Lmz2MomS(b, 10))                       # [1, 1, 3/2, 5/2, 37/8, ...]
info = Lmz2pdf(b, np.arange(-0.05, 5, 0.01))
plt.plot(info.range, info.density)

b = AplusB(wignerpol(), wishartpol(c))       # free sum, c kept symbolic
print(b)            # c*m**3 + c*m**2*z + c*m + m**2 + m*z - m + 1
print(Lmz2MomS(b))  # [1, 1, c + 2, c**2 + 3*c + 4, ...]

# theory vs. simulation
rng = np.random.default_rng(0)
e = np.concatenate([np.linalg.eigvalsh(wigner(100, rng=rng) + wishart(100, 200, rng=rng))
                    for _ in range(1000)])
info = Lmz2pdf(AplusB(wignerpol(), wishartpol(0.5)))
theoryvssim(e, 40, info.density, info.range)
```

Custom atomic measures are built with `atomLmz` (or `numden`):

```python
L1 = atomLmz([1, 2], [0.5, 0.5])            # half the eigenvalues at 1, half at 2
L1 = numden(m - sp.Rational(1,2)/(1 - z) - sp.Rational(1,2)/(2 - z))   # same thing
```

## Function reference

Names follow the MATLAB toolbox so existing scripts translate line-by-line.

| Function | Operation |
|---|---|
| `wignerpol()` | semicircle law |
| `wishartpol(c)` | Marcenko-Pastur law, `c` = rows/columns (may be symbolic) |
| `atomLmz(masses, weights)` | atomic measure |
| `equiLmz(t, M)` | equilibrium measure for the potential `t·x^(2M)` |
| `invA(L)` | `inv(A)` |
| `shiftA(L, alpha)` | `A + alpha·I` |
| `scaleA(L, alpha)` | `alpha·A` |
| `mobiusA(L, p, q, r, s)` | `(p·A + q·I)/(r·A + s·I)` |
| `transposeA(L, c)` | `X'X` given `A = XX'`, `c` = size(A)/size(B) |
| `squareA(L)` | `A²` |
| `AtimesWish(L, c)` | `A × Wishart(c)` (Wishart with covariance A) |
| `AgramWish(L, c, s)` | `(A_s + √s·G)(A_s + √s·G)'` |
| `corrWish(La, Lb, c)` | spatio-temporally correlated Wishart |
| `AplusB(La, Lb)` | `A + QBQ'` (free additive convolution) |
| `AtimesB(La, Lb)` | `A × QBQ'` (free multiplicative convolution) |
| `AblockB(La, Lb, c)` | `diag(A, B)` |
| `compressA(L, c)` | random compression of `A` by a factor `c < 1` |
| `addAdtimes(L, d)` | measure with R-transform `d·R_A` |
| `Lmz2MomS(L, n)` / `Lmz2MomF(L, n)` | first `n` moments |
| `Lmz2pdf(L, xx)` | density along the grid `xx` (returns a `PdfInfo`) |
| `TLmz(L)` | coefficient matrix of the bivariate polynomial |
| `Lmz2Lgz, Lgz2Lrg, Lmz2Lsy, Lmz2Lmyuz, Lmz2Letaz, …` | transform conversions |
| `irreducLuv, Luv2Cu, L1plusL2, L1timesL2, numden` | low-level polynomial tools |
| `wigner(n), wishart(n, m), haar(n)` | sample random matrices (NumPy) |
| `histw(e, nbins), theoryvssim(...)` | normalised histogram / overlay plot |

## Notes for MATLAB users

* The symbols `m, z, c, r, g, s, y, myu, eta` are exported by the package
  (`from rmtool import *`), replacing `syms m z c`.
* `Lmz2pdf` returns a `PdfInfo` dataclass with fields `range, rr, density,
  real, poles, multipleroots`. For polynomials of degree > 3 in `m`,
  `density` has one column per root — use `info.best_density()` as a
  heuristic or isolate the right root manually, as in the MATLAB version.
* `Lmz2MomF` (which used Maple's `gfun`) simply calls `Lmz2MomS`.
* Floating-point constants such as `0.5` are converted to exact rationals so
  that factorisation is reliable; prefer `sympy.Rational` for other values.
* `AplusBkernel` / `AtimesBkernel` are ported but, as in the original, experimental.
* `pretty` → `sympy.pretty`, `latex` → `sympy.latex`.

## License

GPL-2.0-or-later, as the original toolbox. If you use RMTool in a publication,
please cite the paper above and acknowledge the software.
