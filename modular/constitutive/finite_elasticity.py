"""Lopez-Pamies Ogden-type hyperelasticity.

Helpers that build the kinematics, energy density psi1 and undegraded energy
psi11, and the stress invariants sigma_I1 and sigma_vm needed by the
Drucker-Prager driving term. Valid in both 2D (plane stress via F33 solve) and
3D (F fully resolved).

All expressions are UFL — they produce forms that DOLFINx compiles.
"""

from __future__ import annotations
from typing import Dict

import ufl
from ufl import (Identity, det, diff, grad, inv, sqrt, tr, variable, ln)


# ========================================================================== #
# 2D plane stress — F33 solved from sigma_33 = 0 analytically.
# ========================================================================== #
def kinematics_2d_plane_stress(u, mat: Dict):
    """Return a dict with F, J_m, I1_m, I2_m and the principal invariants i1, i2.

    The 3-3 deformation is obtained by enforcing P_33 = 0 analytically — an
    explicit formula specific to the Lopez-Pamies free energy.
    """
    mu1, mu2 = mat["mu1"], mat["mu2"]
    a1, a2   = mat["alpha1"], mat["alpha2"]
    kappa    = mat["kappa"]

    I_  = Identity(2)
    F_  = I_ + grad(u)
    C_  = F_.T * F_
    Ic  = tr(C_)
    J_  = det(F_)

    F33 = ((kappa + mu1 + (a1 * mu1) / 3.0 + mu2 + (a2 * mu2) / 3.0
            - (3.0 ** (1 - a1) * (Ic + J_ ** (-2)) ** (a1 - 1) * mu1
             + 3.0 ** (1 - a2) * (Ic + J_ ** (-2)) ** (a2 - 1) * mu2) / J_ ** 2)
           / (J_ * (kappa + (a1 * mu1) / 3.0 + (a2 * mu2) / 3.0)))

    I1m = Ic + F33 ** 2
    I2m = 0.5 * (I1m ** 2 - tr(C_ * C_) - F33 ** 4)
    Jm  = J_ * F33

    # Cardano-like invariants needed for sigma_I1, sigma_I2 in Drucker-Prager.
    chi1 = (2.0 ** 5 / 27.0) * (2.0 * I1m ** 3 - 9.0 * I1m * I2m + 27.0 * Jm ** 2)
    chi2 = (2.0 ** 10 / 27.0) * (4.0 * I2m ** 3 - (I1m * I2m) ** 2
                                  + 4.0 * I1m ** 3 * Jm ** 2
                                  - 18.0 * I1m * I2m * Jm ** 2
                                  + 27.0 * Jm ** 4)
    chi3 = (-2.0 / 3.0 * I1m
            + abs(chi1 + sqrt(abs(chi2))) ** (1.0 / 3.0)
            + abs(chi1 - sqrt(abs(chi2))) ** (1.0 / 3.0))

    i1 = 0.5 * (sqrt(abs(2.0 * I1m + chi3))
              + sqrt(abs(2.0 * I1m - chi3
                         + 16.0 * Jm / sqrt(abs(2.0 * I1m + chi3)))))
    i2 = sqrt(abs(I2m + 2.0 * i1 * Jm))

    return {"F": F_, "J": J_, "Ic": Ic, "F33": F33,
            "I1m": I1m, "I2m": I2m, "Jm": Jm, "i1": i1, "i2": i2}


# ========================================================================== #
# 2D plane strain — F33 ≡ 1.
# ========================================================================== #
def kinematics_2d_plane_strain(u, mat: Dict):
    I_ = Identity(2)
    F_ = I_ + grad(u)
    C_ = F_.T * F_
    Ic = tr(C_)
    J_ = det(F_)
    I1m = Ic + 1.0
    I2m = 0.5 * (I1m ** 2 - tr(C_ * C_) - 1.0)
    Jm  = J_

    # Cardano invariants (same formula as 2D-PS).
    chi1 = (2.0 ** 5 / 27.0) * (2.0 * I1m ** 3 - 9.0 * I1m * I2m + 27.0 * Jm ** 2)
    chi2 = (2.0 ** 10 / 27.0) * (4.0 * I2m ** 3 - (I1m * I2m) ** 2
                                  + 4.0 * I1m ** 3 * Jm ** 2
                                  - 18.0 * I1m * I2m * Jm ** 2
                                  + 27.0 * Jm ** 4)
    chi3 = (-2.0 / 3.0 * I1m
            + abs(chi1 + sqrt(abs(chi2))) ** (1.0 / 3.0)
            + abs(chi1 - sqrt(abs(chi2))) ** (1.0 / 3.0))

    i1 = 0.5 * (sqrt(abs(2.0 * I1m + chi3))
              + sqrt(abs(2.0 * I1m - chi3
                         + 16.0 * Jm / sqrt(abs(2.0 * I1m + chi3)))))
    i2 = sqrt(abs(I2m + 2.0 * i1 * Jm))
    return {"F": F_, "J": J_, "Ic": Ic, "F33": 1.0,
            "I1m": I1m, "I2m": I2m, "Jm": Jm, "i1": i1, "i2": i2}


# ========================================================================== #
# 3D — F ≡ I + grad(u) fully.
# ========================================================================== #
def kinematics_3d(u, mat: Dict):
    I_ = Identity(3)
    F_ = I_ + grad(u)
    C_ = F_.T * F_
    Ic = tr(C_)
    IIc = 0.5 * (Ic ** 2 - tr(C_ * C_))
    J_ = det(F_)
    return {"F": F_, "J": J_, "Ic": Ic,
            "I1m": Ic, "I2m": IIc, "Jm": J_,
            "i1": sqrt(abs(Ic + 2.0 * sqrt(abs(IIc)))) ,   # crude proxy
            "i2": sqrt(abs(IIc + 2.0 * J_))}


# ========================================================================== #
# Shared energy / stress-invariant helpers.
# ========================================================================== #
def energy_density(z, k, mat: Dict):
    """Degraded Lopez-Pamies energy (isochoric + volumetric with two etas)."""
    mu1, mu2 = mat["mu1"], mat["mu2"]
    a1, a2   = mat["alpha1"], mat["alpha2"]
    kappa    = mat["kappa"]
    eta1, eta2 = mat["eta1"], mat["eta2"]
    I1m, Jm = k["I1m"], k["Jm"]
    return ((z ** 2 + eta1)
            * ((3.0 ** (1 - a1) / a1) * (mu1 / 2.0) * (I1m ** a1 - 3.0 ** a1)
             + (3.0 ** (1 - a2) / a2) * (mu2 / 2.0) * (I1m ** a2 - 3.0 ** a2)
             - (mu1 + mu2) * (Jm - 1.0))
          + (z ** 2 + eta2)
            * (kappa / 2.0 + (3.0 - 2.0 * a1) * mu1 / 6.0
                           + (3.0 - 2.0 * a2) * mu2 / 6.0) * (Jm - 1.0) ** 2)


def energy_density_undegraded(k, mat: Dict):
    """Undegraded (z=1) Lopez-Pamies energy — used in the PF driving force."""
    mu1, mu2 = mat["mu1"], mat["mu2"]
    a1, a2   = mat["alpha1"], mat["alpha2"]
    kappa    = mat["kappa"]
    I1m, Jm = k["I1m"], k["Jm"]
    return ((3.0 ** (1 - a1) / a1) * (mu1 / 2.0) * (I1m ** a1 - 3.0 ** a1)
          + (3.0 ** (1 - a2) / a2) * (mu2 / 2.0) * (I1m ** a2 - 3.0 ** a2)
          - (mu1 + mu2) * (Jm - 1.0)
          + (kappa / 2.0 + (3.0 - 2.0 * a1) * mu1 / 6.0
                         + (3.0 - 2.0 * a2) * mu2 / 6.0) * (Jm - 1.0) ** 2)


def stress_invariants(k, mat: Dict):
    """Return (sigma_I1, sigma_vm) for the Drucker-Prager driving term."""
    mu1, mu2 = mat["mu1"], mat["mu2"]
    a1, a2   = mat["alpha1"], mat["alpha2"]
    kappa    = mat["kappa"]
    I1m, Jm, i1, i2 = k["I1m"], k["Jm"], k["i1"], k["i2"]

    dWdJ  = -(mu1 + mu2) + (kappa
                           + (3.0 - 2.0 * a1) * mu1 / 3.0
                           + (3.0 - 2.0 * a2) * mu2 / 3.0) * (Jm - 1.0)
    dWdI1 = 0.5 * (3.0 ** (1 - a1) * mu1 * I1m ** (a1 - 1)
                 + 3.0 ** (1 - a2) * mu2 * I1m ** (a2 - 1))

    sigma_I1 = 2.0 * i1 * dWdI1 + i2 * dWdJ
    sigma_I2 = (4.0 * i2 * dWdI1 ** 2 + i1 * Jm * dWdJ ** 2
              + 2.0 * (i1 * i2 - 3.0 * Jm) * dWdJ * dWdI1)
    sigma_vm = sqrt(abs(sigma_I1 ** 2 / 3.0 - sigma_I2))
    return sigma_I1, sigma_vm


# ========================================================================== #
# First Piola for reaction-force post-processing.
# ========================================================================== #
def first_piola_2d_plane_stress(u, z, mat: Dict):
    mu1, mu2 = mat["mu1"], mat["mu2"]
    a1, a2   = mat["alpha1"], mat["alpha2"]
    kappa    = mat["kappa"]
    eta1, eta2 = mat["eta1"], mat["eta2"]

    F = variable(Identity(2) + grad(u))
    C = F.T * F
    Ic = tr(C)
    J_ = det(F)
    F33 = ((kappa + mu1 + (a1 * mu1) / 3.0 + mu2 + (a2 * mu2) / 3.0
            - (3.0 ** (1 - a1) * (Ic + J_ ** (-2)) ** (a1 - 1) * mu1
             + 3.0 ** (1 - a2) * (Ic + J_ ** (-2)) ** (a2 - 1) * mu2) / J_ ** 2)
           / (J_ * (kappa + (a1 * mu1) / 3.0 + (a2 * mu2) / 3.0)))
    I1m = Ic + F33 ** 2
    Jm  = J_ * F33
    psi = ((z ** 2 + eta1)
           * ((3.0 ** (1 - a1) / a1) * (mu1 / 2.0) * (I1m ** a1 - 3.0 ** a1)
            + (3.0 ** (1 - a2) / a2) * (mu2 / 2.0) * (I1m ** a2 - 3.0 ** a2)
            - (mu1 + mu2) * ln(Jm))
         + (z ** 2 + eta2)
           * (kappa / 2.0 - a1 * mu1 / 3.0 - a2 * mu2 / 3.0) * (Jm - 1.0) ** 2)
    return diff(psi, F)


def first_piola_2d_plane_strain(u, z, mat: Dict):
    """First Piola stress for 2D plane strain (F33 ≡ 1)."""
    mu1, mu2 = mat["mu1"], mat["mu2"]
    a1, a2   = mat["alpha1"], mat["alpha2"]
    kappa    = mat["kappa"]
    eta1, eta2 = mat["eta1"], mat["eta2"]

    F = variable(Identity(2) + grad(u))
    C = F.T * F
    Ic = tr(C)
    J_ = det(F)
    I1m = Ic + 1.0
    psi = ((z ** 2 + eta1)
           * ((3.0 ** (1 - a1) / a1) * (mu1 / 2.0) * (I1m ** a1 - 3.0 ** a1)
            + (3.0 ** (1 - a2) / a2) * (mu2 / 2.0) * (I1m ** a2 - 3.0 ** a2)
            - (mu1 + mu2) * ln(J_))
         + (z ** 2 + eta2)
           * (kappa / 2.0 - a1 * mu1 / 3.0 - a2 * mu2 / 3.0) * (J_ - 1.0) ** 2)
    return diff(psi, F)


def first_piola_3d(u, z, mat: Dict):
    mu1, mu2 = mat["mu1"], mat["mu2"]
    a1, a2   = mat["alpha1"], mat["alpha2"]
    kappa    = mat["kappa"]
    eta1, eta2 = mat["eta1"], mat["eta2"]

    F = variable(Identity(3) + grad(u))
    C = F.T * F
    Ic = tr(C)
    J_ = det(F)
    psi = ((z ** 2 + eta1)
           * ((3.0 ** (1 - a1) / a1) * (mu1 / 2.0) * (Ic ** a1 - 3.0 ** a1)
            + (3.0 ** (1 - a2) / a2) * (mu2 / 2.0) * (Ic ** a2 - 3.0 ** a2)
            - (mu1 + mu2) * ln(J_))
         + (z ** 2 + eta2)
           * (kappa / 2.0 - a1 * mu1 / 3.0 - a2 * mu2 / 3.0) * (J_ - 1.0) ** 2)
    return diff(psi, F)
