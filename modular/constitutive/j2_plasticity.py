"""J2 plasticity with isotropic power-law hardening — matches ductile_amr.py.

The degradation function is ductility-aware: `dgd(z, p) = z^(2·(p/p_sat)^2) + eta`
which drives the phase-field only once accumulated plastic strain is
significant. `p_sat = 0.12` follows the existing code (empirically tuned).

This module provides:
  • Tensor helpers (strain promoted to 3D, Voigt ↔ tensor conversions)
  • Radial return (`compute_stress_update`)
  • Degradation and elastic energy of the driving term
  • A `build_problem_spaces` convenience that sets up the quadrature
    function spaces for plastic state storage.
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import basix.ufl
from dolfinx import fem
from ufl import (Identity, as_tensor, as_vector, conditional, dev, grad,
                 gt, inner, sqrt, sym, tr)


# --- Tensor helpers ------------------------------------------------------- #
def strain_2d_planestrain_as_3d(v):
    """Plane-strain 3D strain tensor from 2D displacement field."""
    e = sym(grad(v))
    return as_tensor([[e[0, 0], e[0, 1], 0.0],
                      [e[0, 1], e[1, 1], 0.0],
                      [0.0,     0.0,     0.0]])


def strain_3d(v):
    return sym(grad(v))


def voigt_to_tensor_2d(X):
    """Voigt (4,) [xx, yy, zz, xy] → full 3D tensor (plane-strain)."""
    return as_tensor([[X[0], X[3], 0.0],
                      [X[3], X[1], 0.0],
                      [0.0,  0.0,  X[2]]])


def tensor_to_voigt_2d(X):
    return as_vector([X[0, 0], X[1, 1], X[2, 2], X[0, 1]])


def voigt_to_tensor_3d(X):
    """Voigt (6,) [xx, yy, zz, xy, xz, yz] → full 3D tensor."""
    return as_tensor([[X[0], X[3], X[4]],
                      [X[3], X[1], X[5]],
                      [X[4], X[5], X[2]]])


def tensor_to_voigt_3d(X):
    return as_vector([X[0, 0], X[1, 1], X[2, 2],
                      X[0, 1], X[0, 2], X[1, 2]])


# --- Volumetric / deviatoric stress --------------------------------------- #
def stress_volumetric(eps_el, mu, lmbda):
    return (1.0 / 3.0) * (3.0 * lmbda + 2.0 * mu) * tr(eps_el) * Identity(3)


def stress_deviatoric(eps_el, mu):
    return 2.0 * mu * dev(eps_el)


def von_mises(sig):
    s = dev(sig)
    return sqrt(1.5 * inner(s, s))


# --- Isotropic power-law hardening ---------------------------------------- #
def yield_stress(p, sig0, e0, nhard):
    return sig0 * (1.0 + p / e0) ** (1.0 / nhard)


def hardening_modulus(p, sig0, e0, nhard):
    return sig0 / nhard / e0 * (1.0 + p / e0) ** ((1.0 / nhard) - 1.0)


# --- Ductility-aware degradation ------------------------------------------ #
P_SAT = 0.12   # same constant as ductile_amr.py


def dgd_ductile(z, p, eta):
    return z ** (2.0 * (p / P_SAT) ** 2) + eta


def elastoplastic_energy(u, eps_pl_tensor, mu, lmbda, *, strain_fn):
    """Driving energy for the phase field: ψ(ε_el), ε_el = ε(u) − ε_pl."""
    eps_el = strain_fn(u) - eps_pl_tensor
    return mu * inner(eps_el, eps_el) + 0.5 * lmbda * tr(eps_el) ** 2


# --- Radial return -------------------------------------------------------- #
def compute_stress_update(u, eps_pl_tensor, p, z, mat, *, strain_fn):
    """One-step radial return (ductile 'Model 2', degraded yield)."""
    mu, lmbda = mat["mu"], mat["lmbda"]
    sig_vol = stress_volumetric(strain_fn(u) - eps_pl_tensor, mu, lmbda)
    sig_dev = stress_deviatoric(strain_fn(u) - eps_pl_tensor, mu)
    sig_trial = sig_vol + sig_dev

    gz  = dgd_ductile(z, p, mat["eta"])
    vm  = von_mises(sig_trial)

    sig_y = yield_stress(p, mat["sigma_y0"], mat["sigma_y0"] / mat["H_hardening"],
                         mat["n_hardening"])
    H_p   = hardening_modulus(p, mat["sigma_y0"],
                               mat["sigma_y0"] / mat["H_hardening"],
                               mat["n_hardening"])

    f_yield = gz * vm - sig_y
    dp = conditional(f_yield < 0.0, 0.0,
                     f_yield / (3.0 * mu * gz + H_p))
    Z33 = as_tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    N   = conditional(gt(vm, 1e-10), dev(sig_trial) / vm, Z33)
    new_dev = conditional(f_yield < 0.0, sig_dev,
                          sig_dev - 2.0 * mu * 1.5 * dp * N)
    return sig_vol + new_dev, dp, 1.5 * dp * N


# --- Quadrature spaces for plastic state storage -------------------------- #
def make_quadrature_spaces(msh, *, q_deg: int = 2, voigt_dim: int = 4):
    """Build scalar + tensor (Voigt) quadrature spaces.

    The scheme is left as basix's default; callers must use the same
    `q_deg` in the integration `Measure` metadata.
    """
    cell_name = msh.topology.cell_name()
    W_tensor = fem.functionspace(
        msh,
        basix.ufl.quadrature_element(
            cell_name, degree=q_deg, value_shape=(voigt_dim,)),
    )
    W_scalar = fem.functionspace(
        msh,
        basix.ufl.quadrature_element(cell_name, degree=q_deg),
    )
    return W_tensor, W_scalar


def interpolate_quadrature(ufl_expr, func: fem.Function):
    """Evaluate a UFL expression at the quadrature points of `func`."""
    ipts = func.function_space.element.interpolation_points()
    e = fem.Expression(ufl_expr, ipts)
    func.interpolate(e)
    func.x.scatter_forward()
