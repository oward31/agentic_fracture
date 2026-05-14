"""Linear elasticity, 2D plane strain.

All formulas match `bench_amr.py` exactly:
  I1_0 uses the (1+nu)·tr(sigma) promotion that accounts for the out-of-plane
  stress sigma_33 = nu·(sigma_11 + sigma_22).
"""

from __future__ import annotations
from ufl import Identity, grad, inner, sqrt, sym, tr


def epsilon(v):
    return sym(grad(v))


def sigma(em, mu, lmbda):
    return 2.0 * mu * em + lmbda * tr(em) * Identity(2)


def dgd(z, eta):
    return z ** 2 + eta


def energy(em, mu, lmbda):
    return mu * inner(em, em) + 0.5 * lmbda * tr(em) ** 2


def I1_0(em, mu, lmbda, nu):
    return (1.0 + nu) * tr(sigma(em, mu, lmbda))


def sigmavm(em, mu, lmbda, nu):
    sig    = sigma(em, mu, lmbda)
    I1     = (1.0 + nu) * tr(sig)
    dev2d  = sig - (1.0 / 3.0) * I1 * Identity(2)
    return sqrt(0.5 * (inner(dev2d, dev2d)
                      + ((2.0 * nu / 3.0 - 1.0 / 3.0) ** 2) * I1 ** 2) + 1e-30)
