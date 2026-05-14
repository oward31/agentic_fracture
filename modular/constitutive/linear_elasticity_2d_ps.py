"""Linear elasticity, 2D plane stress.

Matches `dyn_branch_amr.py` exactly: sigma uses the (1-2ν)/(1-ν) correction,
energy picks up the ν/(1-ν) out-of-plane contribution.
"""

from __future__ import annotations
from ufl import Identity, grad, inner, sqrt, sym, tr


def epsilon(v):
    return sym(grad(v))


def sigma(em, mu, lmbda, nu):
    return (2.0 * mu * em
            + lmbda * tr(em) * (1.0 - 2.0 * nu) / (1.0 - nu) * Identity(2))


def dgd(z, eta):
    return z ** 2 + eta


def energy(em, mu, lmbda, nu):
    return (mu * (inner(em, em) + (nu / (1.0 - nu)) ** 2 * tr(em) ** 2)
            + 0.5 * lmbda * ((1.0 - 2.0 * nu) / (1.0 - nu)) ** 2 * tr(em) ** 2)


def I1_0(em, mu, lmbda, nu):
    return tr(sigma(em, mu, lmbda, nu))


def sigmavm(em, mu, lmbda, nu):
    sig   = sigma(em, mu, lmbda, nu)
    I1    = tr(sig)
    dev2d = sig - (1.0 / 3.0) * I1 * Identity(2)
    return sqrt(0.5 * (inner(dev2d, dev2d) + (1.0 / 9.0) * I1 ** 2) + 1e-30)
