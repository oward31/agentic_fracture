"""Pieces shared by every problem builder (function spaces, facet tags,
stagger norms, AMR indicator wiring)."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from dolfinx import fem, mesh as dmesh
from petsc4py.PETSc import ScalarType
from ufl import (CellDiameter, FacetNormal, Measure, inner)

from ..common.bcs import build_facet_tags


# -------------------------------------------------------------------------- #
# AMR-safe BC ramp materialisation
# -------------------------------------------------------------------------- #
def materialise_bc_constants(msh, bc_spec_u: Optional[List[Tuple]]
                              ) -> Tuple[Optional[List[Tuple]], Dict[str, Any]]:
    """Replace ``"@ramp:N"`` string markers in ``bc_spec_u`` with fresh
    ``fem.Constant`` objects on the build mesh ``msh``.

    Returns ``(fresh_spec, bc_value_constants)``:
      * ``fresh_spec`` — same list with markers swapped for live Constants
        (passable to ``build_dirichlet_bcs``).
      * ``bc_value_constants`` — ``{marker_str: fem.Constant on msh}`` for the
        driver to update over time.

    **Why this exists.**  Earlier versions of fracture_agent's templates created
    ``fem.Constant(msh_coarse, 0.0)`` slots in the driver and passed them
    directly into ``bc_spec_u``.  That worked for a fixed mesh, but AMR
    rebuilds the problem on a *new* mesh — the old Constants are stale,
    the new BCs reference them, and the driver's ramp proxy stops
    propagating changes.  By always materialising the Constants on the
    *current* build mesh inside the modular builder, every AMR rebuild
    creates a fresh, mesh-correct Constant that the after-rebuild hook
    can re-bind.

    Compatible with the legacy "raw fem.Constant in bc_spec_u" pattern —
    those values pass through unchanged (and remain AMR-fragile, but no
    fracture_agent-generated driver still emits that pattern after this change).
    """
    if bc_spec_u is None:
        return None, {}
    fresh: List[Tuple] = []
    bc_value_constants: Dict[str, Any] = {}
    for tag, comp, val in bc_spec_u:
        if isinstance(val, str) and val.startswith("@ramp:"):
            c = fem.Constant(msh, ScalarType(0.0))
            bc_value_constants[val] = c
            fresh.append((tag, comp, c))
        else:
            fresh.append((tag, comp, val))
    return fresh, bc_value_constants


def function_spaces(msh, *, vector_dim: int):
    """Standard spaces: Lagrange-1 vector, Lagrange-1 scalar, DG-0 scalar."""
    V  = fem.functionspace(msh, ("Lagrange", 1, (vector_dim,)))
    Y  = fem.functionspace(msh, ("Lagrange", 1))
    Yv = fem.functionspace(msh, ("DG", 0))
    return V, Y, Yv


def tag_and_measures(msh, markers_spec, *, quadrature_degree: int = 2):
    """Build facet tags plus `dx`, `ds`, outward normal."""
    fdim = msh.topology.dim - 1
    for d in (fdim - 1, fdim, msh.topology.dim):
        msh.topology.create_connectivity(d, msh.topology.dim)
    ft, name_to_tag, tag_to_facets = build_facet_tags(msh, markers_spec)
    dx = Measure("dx", domain=msh,
                 metadata={"quadrature_degree": quadrature_degree,
                           "quadrature_scheme": "default"})
    ds = Measure("ds", domain=msh, subdomain_data=ft)
    n  = FacetNormal(msh)
    return ft, name_to_tag, tag_to_facets, dx, ds, n


def make_phasefield_fields(msh, Y):
    """z ∈ [0,1] with bounds."""
    z       = fem.Function(Y, name="phasefield")
    z_trial = fem.Function(Y)
    z_lb    = fem.Function(Y, name="Lower bound")
    z_ub    = fem.Function(Y, name="Upper bound")
    z_lb.x.array[:] = 0.0; z_lb.x.scatter_forward()
    z_ub.x.array[:] = 1.0; z_ub.x.scatter_forward()
    z.x.array[:]    = 1.0; z.x.scatter_forward()
    return z, z_trial, z_lb, z_ub


def cache_cell_size_and_indicator(msh, Yv, indicator_expr):
    """Cache the cell-size function (DG0) and pre-compiled indicator expr."""
    ipts = Yv.element.interpolation_points()
    cell_h = fem.Function(Yv)
    cell_h.interpolate(fem.Expression(CellDiameter(msh), ipts))
    ind_expr = fem.Expression(indicator_expr, ipts)
    return cell_h, ind_expr


def stagger_residual_forms(u, z, V, Y, dx):
    """Return (u_trial, u_diff, u_diff_norm_form,
               z_diff, z_diff_norm_form)."""
    u_trial = fem.Function(V)
    u_diff  = fem.Function(V)
    z_diff  = fem.Function(Y)
    u_form = fem.form(inner(u_diff, u_diff) * dx)
    z_form = fem.form(inner(z_diff, z_diff) * dx)
    return u_trial, u_diff, u_form, z_diff, z_form
