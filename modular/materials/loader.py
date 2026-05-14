"""Material loader with derived quantities.

A material entry in materials.json lists primary physical properties keyed by
`problem_type`. Derived quantities — shear/Lame moduli, bulk modulus, critical
energy densities Wts/Whs, Irwin characteristic length lch, and the canonical
mesh sizes — are added on the fly by this loader so the JSON stays minimal.

Convention for the regularisation length and mesh sizes (user-specified):
    lch   = 3 * Gc / (16 * Wts)
    eps   = lch
    h0    = 2 * eps   = 2 * lch
    h_min = h0 / 8

For finite elasticity, Wts is obtained by solving the 1-D Lopez-Pamies
uniaxial-tension problem (fsolve on the nonlinear BVP) — we cache the result
in the returned dict.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np


_DEFAULT_DB = Path(__file__).with_name("materials.json")


def load_material(name: str, db_path: str | Path | None = None) -> Dict[str, Any]:
    """Load material `name`, augment it with derived quantities."""
    path = Path(db_path) if db_path is not None else _DEFAULT_DB
    with open(path, "r") as fh:
        db = json.load(fh)
    if name not in db:
        raise KeyError(f"Material '{name}' not found in {path}. "
                       f"Available: {[k for k in db if not k.startswith('_')]}")
    m = dict(db[name])
    m["name"] = name

    ptype = m["problem_type"]
    if ptype in ("linear_elasticity", "dynamic_linear_elasticity", "ductile"):
        _augment_linear(m)
    elif ptype == "finite_elasticity":
        _augment_finite_elasticity(m)
    else:
        raise ValueError(f"Unknown problem_type '{ptype}' for material '{name}'.")
    m.update(derive_length_scales(m))
    return m


# --------------------------------------------------------------------------- #
# Linear / ductile elastic moduli and strength surface.
# --------------------------------------------------------------------------- #
def _augment_linear(m: Dict[str, Any]) -> None:
    """Fill mu, lambda, kappa, Wts, Whs, sigma_hs for linear/ductile cases."""
    E, nu = m["E"], m["nu"]
    m["mu"]    = E / (2.0 * (1.0 + nu))
    m["lmbda"] = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    m["kappa"] = E / (3.0 * (1.0 - 2.0 * nu))
    m.update(derive_strength_surface(m))


def derive_strength_surface(m: Dict[str, Any]) -> Dict[str, float]:
    """Drucker-Prager hydrostatic strength & critical energy densities.

    For the ductile case we build sigma_ts from sigma_y0 * sigma_ts_factor
    unless sigma_ts is already present (mirrors the existing ductile code).
    """
    if m["problem_type"] == "ductile" and "sigma_ts" not in m:
        m["sigma_ts"] = m["sigma_y0"] * m.get("sigma_ts_factor", 2.0)
    if m["problem_type"] == "ductile" and "sigma_cs" not in m:
        m["sigma_cs"] = 3.0 * m["sigma_ts"]

    sts, scs = m["sigma_ts"], m["sigma_cs"]
    shs = (2.0 / 3.0) * sts * scs / (scs - sts)
    Wts = 0.5 * sts**2 / m["E"]
    Whs = 0.5 * shs**2 / m["kappa"]
    return {"sigma_hs": shs, "Wts": Wts, "Whs": Whs}


# --------------------------------------------------------------------------- #
# Lopez-Pamies Ogden: solve for Wts, Whs at the supplied strengths.
# --------------------------------------------------------------------------- #
def _augment_finite_elasticity(m: Dict[str, Any]) -> None:
    """Fill Wts, Whs by solving the Lopez-Pamies BVP at sigma_ts, sigma_hs."""
    from scipy.optimize import fsolve

    mu1, mu2       = m["mu1"], m["mu2"]
    a1,  a2        = m["alpha1"], m["alpha2"]
    kappa          = m["kappa"]
    sts, shs       = m["sigma_ts"], m["sigma_hs"]

    m["lmbda"] = kappa - 2.0 / 3.0 * (mu1 + mu2)
    m["mu"]    = mu1 + mu2
    m["nu"]    = 0.5 / (1.0 + (mu1 + mu2) / m["lmbda"])
    m["E"]     = 2.0 * m["mu"] * (1.0 + m["nu"])

    def W_of(I1, J):
        return ((3.0**(1 - a1) / a1) * (mu1 / 2.0) * (I1**a1 - 3.0**a1)
              + (3.0**(1 - a2) / a2) * (mu2 / 2.0) * (I1**a2 - 3.0**a2)
              - (mu1 + mu2) * (J - 1.0)
              + (kappa / 2.0 + (3.0 - 2.0 * a1) * mu1 / 6.0
                             + (3.0 - 2.0 * a2) * mu2 / 6.0) * (J - 1.0)**2)

    # --- Uniaxial tension ---
    def ut(x):
        l1, l2 = x
        I1 = l1**2 + 2.0 * l2**2
        J  = l1 * l2**2
        dI = 0.5 * (3.0**(1 - a1) * mu1 * I1**(a1 - 1)
                  + 3.0**(1 - a2) * mu2 * I1**(a2 - 1))
        dJ = -(mu1 + mu2) + (kappa + (3.0 - 2.0 * a1) * mu1 / 3.0
                                   + (3.0 - 2.0 * a2) * mu2 / 3.0) * (J - 1.0)
        return [2.0 * dI * l1 + dJ * J / l1 - sts,
                2.0 * dI * l2 + dJ * J / l2]
    (l1, l2), _, ok_ut, msg_ut = fsolve(ut, [1.5, 0.8], full_output=True)
    if ok_ut != 1:
        raise RuntimeError(f"Lopez-Pamies UT solver failed: {msg_ut}")
    m["Wts"] = W_of(l1**2 + 2.0 * l2**2, l1 * l2**2)

    # --- Hydrostatic tension ---
    def ht(x):
        lam = x[0]
        I1 = 3.0 * lam**2
        J  = lam**3
        dI = 0.5 * (3.0**(1 - a1) * mu1 * I1**(a1 - 1)
                  + 3.0**(1 - a2) * mu2 * I1**(a2 - 1))
        dJ = -(mu1 + mu2) + (kappa + (3.0 - 2.0 * a1) * mu1 / 3.0
                                   + (3.0 - 2.0 * a2) * mu2 / 3.0) * (J - 1.0)
        return [2.0 * dI * lam + dJ * J / lam - shs]
    (lam,), _, ok_ht, msg_ht = fsolve(ht, [1.01], full_output=True)
    if ok_ht != 1:
        raise RuntimeError(f"Lopez-Pamies HT solver failed: {msg_ht}")
    m["Whs"] = W_of(3.0 * lam**2, lam**3)


# --------------------------------------------------------------------------- #
# Length scales — single source of truth for h0 and h_min.
# --------------------------------------------------------------------------- #
def derive_length_scales(m: Dict[str, Any]) -> Dict[str, float]:
    """Return lch, eps, h0, h_min with the user's fixed convention.

    Convention:
        lch   = 3 * Gc / (16 * Wts)
        eps   = lch
        h0    = 2 * eps   = 2 * lch
        h_min = h0 / 8
    """
    lch   = 3.0 * m["Gc"] / (16.0 * m["Wts"])
    eps   = lch
    h0    = 2.0 * eps
    h_min = h0 / 8.0
    return {"lch": lch, "eps": eps, "h0": h0, "h_min": h_min}


# --------------------------------------------------------------------------- #
# Approximate Rayleigh wave speed for dynamic time step.
# --------------------------------------------------------------------------- #
def rayleigh_speed(m: Dict[str, Any]) -> float:
    """Approximate Rayleigh wave speed c_R ≈ (0.862 + 1.14 nu)/(1+nu) * c_S."""
    cS = float(np.sqrt(m["mu"] / m["rho"]))
    nu = m["nu"]
    return (0.862 + 1.14 * nu) / (1.0 + nu) * cS
