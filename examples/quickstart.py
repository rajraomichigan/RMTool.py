"""Python version of the RMTool 'Quick start' from the user's guide."""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from rmtool import *

# Marcenko-Pastur: polynomial, moments, density
b = wishartpol(0.5)
print("Marcenko-Pastur moments:", Lmz2MomS(b, 10))
info = Lmz2pdf(b, np.arange(-0.05, 5, 0.01))
plt.figure(1); plt.plot(info.range, info.density, linewidth=2); plt.title("Marcenko-Pastur, c = 0.5")

# Wigner semicircle
b = wignerpol()
print("Wigner moments:", Lmz2MomS(b, 10))
info = Lmz2pdf(b, np.arange(-4, 4, 0.01))
plt.figure(2); plt.plot(info.range, info.density, linewidth=2); plt.title("Semicircle")

# Free sum with a symbolic parameter c
b = AplusB(wignerpol(), wishartpol(c))
print("Wigner + Wishart(c):", b)
print("moments:", Lmz2MomS(b))
print(sp.latex(b))

# Theory vs. simulation
rng = np.random.default_rng(0)
e = np.concatenate([np.linalg.eigvalsh(wigner(100, rng=rng) + wishart(100, 200, rng=rng)) for _ in range(1000)])
info = Lmz2pdf(AplusB(wignerpol(), wishartpol(0.5)))
plt.figure(3); theoryvssim(e, 40, info.density, info.range)
plt.show()
