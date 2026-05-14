"""Per-step XDMF snapshot writer.

Fields written every step: u, z, sigma_vm (projected); plus p (equivalent
plastic strain) for ductile runs when `include_plastic=True`.
"""

from __future__ import annotations
import os
from typing import Dict

from dolfinx import fem
from dolfinx.io import XDMFFile

from .reaction import custom_project


class XDMFWriter:
    """Writes one `step_<N>.xdmf` per call. Scalar fields are L2-projected
    to the CG1 scalar space `P["Y"]` so ParaView sees node-based data."""

    def __init__(self, output_dir: str, *, include_plastic: bool = False,
                 include_stress: bool = True):
        self.output_dir      = output_dir
        self.include_plastic = include_plastic
        self.include_stress  = include_stress
        os.makedirs(output_dir, exist_ok=True)

    def write(self, P: Dict, t: float, step: int):
        msh   = P["msh"]
        fname = os.path.join(self.output_dir, f"step_{step:06d}.xdmf")
        with XDMFFile(msh.comm, fname, "w") as xf:
            xf.write_mesh(msh)
            xf.write_function(_maybe_cg1_vector(P, P["u"], "u"), t)
            xf.write_function(P["z"], t)
            if self.include_stress and "sigma_vm" in P:
                vm = custom_project(P.get("dgd", 1.0) * P["sigma_vm"],
                                    P["Y"], P["dx"])
                vm.name = "sigma_vm"
                xf.write_function(vm, t)
            if self.include_plastic and "p" in P:
                p_proj = custom_project(P["p"], P["Y"], P["dx"])
                p_proj.name = "p_eq"
                xf.write_function(p_proj, t)


def _maybe_cg1_vector(P, u_func, name):
    """If the builder advertises a CG1 plot space (used by finite-elasticity
    CR elements), interpolate onto it; otherwise write u directly."""
    if "V_plot" in P:
        u_cg = fem.Function(P["V_plot"])
        u_cg.interpolate(fem.Expression(u_func, P["V_plot"].element.interpolation_points()))
        u_cg.name = name
        return u_cg
    u_func.name = name
    return u_func
