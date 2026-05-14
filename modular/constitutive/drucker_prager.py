"""Drucker-Prager strength surface — shared across all linear/finite variants.

Returns three ingredients used by every phase-field residual:

  1. Coefficients  (delta, beta1, beta2)     — depend on (Gc, eps, sts, shs,
                                                 Wts, Whs, h_min).
  2. Driving force  ce                        — UFL expression f(z, psi1,
                                                 I1_deg, SQJ2_deg, I1_undeg).
  3. AMR indicator                            — {0, 1} UFL expression marking
                                                 cells to be refined.

The formulas match the user's existing codes (bench, surfing, dyn_branch,
slant) verbatim so results reproduce bit-for-bit.
"""

from __future__ import annotations
from typing import Dict

from ufl import conditional, grad, inner, lt, max_value


def drucker_prager_coefficients(mat: Dict, *, eps: float, h_min: float) -> Dict[str, float]:
    """Compute delta, beta1, beta2 from material properties.

    mat needs: Gc, sigma_ts, sigma_hs, Wts, Whs
    """
    Gc    = mat["Gc"]
    sts   = mat["sigma_ts"]
    shs   = mat["sigma_hs"]
    Wts   = mat["Wts"]
    Whs   = mat["Whs"]

    corr  = (1.0 + 3.0 * h_min / (8.0 * eps)) ** (-1)
    delta = (corr ** 2 * ((sts + (1.0 + 2.0 * 3.0 ** 0.5) * shs)
                          / ((8.0 + 3.0 * 3.0 ** 0.5) * shs))
             * 3.0 * Gc / (16.0 * Wts * eps)
             + corr * (2.0 / 5.0))
    beta1 = (-(delta * Gc) / (shs * 8.0 * eps)
             + (2.0 * Whs) / (3.0 * shs))
    beta2 = (-(3.0 ** 0.5 * (3.0 * shs - sts) * delta * Gc) / (shs * sts * 8.0 * eps)
             - (2.0 * Whs) / (3.0 ** 0.5 * shs)
             + (2.0 * 3.0 ** 0.5 * Wts) / sts)
    return {"delta": delta, "beta1": beta1, "beta2": beta2}


def drucker_prager_driving_force(z, psi1, I1_deg, SQJ2_deg, I1_undeg, dp):
    """Drucker-Prager crack driving term ce."""
    return (dp["beta2"] * SQJ2_deg
            + dp["beta1"] * I1_deg
            + conditional(lt(I1_undeg, 0.0), 2.0 * z * psi1, 0.0))


def pf_flag_indicator(z, psi1, ce, *, Gc: float, eps: float, delta: float,
                      threshold: float = -0.1):
    """AMR indicator — 1 where the cell is crack-like.

    Fixed threshold −0.1 (user-mandated).
    Two triggers (ORed):
      (a) pf_flag > threshold
      (b) z < 0.2  (already damaged)
    """
    pf_flag = (2.0 * z * psi1 - ce - 3.0 * delta * Gc / 8.0 / eps) \
              / (3.0 * delta * Gc / 8.0 / eps)
    return max_value(conditional(pf_flag > threshold, 1.0, 0.0),
                     conditional(z < 0.2, 1.0, 0.0))


def pf_residual_gradient_term(y, z, *, Gc: float, eps: float, delta: float):
    """The gradient / regularisation term common to all phase-field residuals:

        + (3δGc/8) · ( −y/eps + 2·eps·∇z·∇y ) · dx

    Returned as a UFL integrand (caller multiplies by dx).
    """
    return 3.0 * delta * Gc / 8.0 * (-y / eps + 2.0 * eps * inner(grad(z), grad(y)))
