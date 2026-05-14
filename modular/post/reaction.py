"""Reaction-force assembly helpers.

`reaction_form_from_sigma_Nd(P, facet_name, component)` returns a callable
`rf(P) -> fem.Form` that, when assembled and MPI-reduced, yields the
reaction force (Fx or Fy) on a given boundary.

We use `t = σ · n, Fi = ∫_∂Ω_tag t_i ds`.

For the ductile problem `sigma` is degraded × Voigt stress; for linear and
dynamic cases it's just the UFL Cauchy stress.  For finite elasticity a
dedicated `stress_form` is computed inside the problem builder.
"""

from __future__ import annotations
from typing import Callable

from dolfinx import fem
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, dot, inner


def custom_project(expr, V, dx):
    """L2 projection onto `V` (same pattern as the existing codes)."""
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(u, v) * dx
    L = inner(expr, v) * dx
    A = fem.petsc.assemble_matrix(fem.form(a)); A.assemble()
    b = fem.petsc.assemble_vector(fem.form(L)); b.assemble()
    ksp = PETSc.KSP().create(V.mesh.comm); ksp.setOperators(A)
    ksp.setType("preonly"); ksp.getPC().setType("lu")
    out = fem.Function(V)
    ksp.solve(b, out.x.petsc_vec); out.x.scatter_forward()
    ksp.destroy(); A.destroy(); b.destroy()
    return out


def reaction_form_from_sigma_2d(facet_name: str, component: int) -> Callable:
    """2D reaction-force factory — assumes the problem dict has a UFL `sigma`
    expression, `n` outward normal, `ds` measure and `name_to_tag` lookup.
    """
    def rf(P):
        sig = P.get("dgd", 1.0) * P["sigma"]
        n = P["n"]
        traction = dot(sig, n)
        tag = P["name_to_tag"][facet_name]
        return fem.form(traction[component] * P["ds"](tag))
    return rf


def reaction_form_from_sigma_3d(facet_name: str, component: int) -> Callable:
    def rf(P):
        sig = P.get("dgd", 1.0) * P["sigma"]
        n = P["n"]
        traction = dot(sig, n)
        tag = P["name_to_tag"][facet_name]
        return fem.form(traction[component] * P["ds"](tag))
    return rf
