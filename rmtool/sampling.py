"""Random matrix generators and plotting helpers (wigner, wishart, haar, histw)."""
import numpy as np

__all__ = ["wigner", "wishart", "haar", "histw", "theoryvssim"]


def wigner(n, isreal=True, rng=None):
    """n x n symmetric (Hermitian) Wigner matrix with semicircle spectrum on [-2, 2]."""
    rng = np.random.default_rng(rng)
    if isreal:
        g = rng.standard_normal((n, n))
    else:
        g = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    return (g + g.conj().T) / np.sqrt(2 * n)


def wishart(n, m=None, isreal=True, rng=None):
    """n x n Wishart matrix W = G*G'/m with G of size n x m (c = n/m)."""
    rng = np.random.default_rng(rng)
    if m is None:
        m = n
    if isreal:
        g = rng.standard_normal((n, m))
    else:
        g = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))) / np.sqrt(2)
    return g @ g.conj().T / m


def haar(n, rng=None):
    """n x n Haar-distributed unitary matrix."""
    rng = np.random.default_rng(rng)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q @ np.diag(np.exp(2j * np.pi * rng.random(n)))


def histw(y, nbins=40, ax=None, **bar_kwargs):
    """Normalised histogram of the data ``y`` using ``nbins`` boxes.

    Returns ``(centers, heights)`` and draws a bar plot (matplotlib) unless
    ``ax=False``.
    """
    y = np.asarray(y, dtype=float).ravel()
    dy = (y.max() - y.min()) / nbins
    edges = y.min() + dy * np.arange(nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.floor((y - y.min()) / dy).astype(int), 0, nbins - 1)
    h = np.bincount(idx, minlength=nbins).astype(float)
    h /= (centers[1] - centers[0]) * len(y)
    if ax is not False:
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.bar(centers, h, width=dy, **bar_kwargs)
    return centers, h


def theoryvssim(e, nbins, imagpdf, pdfrange, ax=None):
    """Overlay the theoretical density on a histogram of sampled eigenvalues."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    histw(e, nbins, ax=ax)
    ax.plot(pdfrange, imagpdf, "r", linewidth=2)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    return ax
