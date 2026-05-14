"""Linear elasticity, 3D — matches `surfing_amr_sota_3d.py`."""

from __future__ import annotations
from ufl import Identity, grad, inner, sqrt, sym, tr


def epsilon(v):
    return sym(grad(v))


def sigma(em, mu, lmbda):
    return 2.0 * mu * em + lmbda * tr(em) * Identity(3)


def dgd(z, eta):
    return z ** 2 + eta


def energy(em, mu, lmbda):
    return mu * inner(em, em) + 0.5 * lmbda * tr(em) ** 2


def I1_0(em, mu, lmbda):
    return tr(sigma(em, mu, lmbda))


def sigmavm(em, mu, lmbda):
    sig = sigma(em, mu, lmbda)
    I1  = tr(sig)
    dev = sig - (1.0 / 3.0) * I1 * Identity(3)
    return sqrt(0.5 * inner(dev, dev) + 1e-30)
