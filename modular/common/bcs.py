"""Boundary-tag and Dirichlet-BC helpers.

The mesh modules return `markers_spec = [(tag, name, locator), ...]`; this
module turns that into a `meshtags` object and gives you DOF lookups by
component.

`bcs_z = []` by convention — no Dirichlet BCs on the phase field.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Callable, Any

import numpy as np
from dolfinx import fem, mesh as dmesh
from petsc4py.PETSc import ScalarType


# -------------------------------------------------------------------------- #
# Facet tagging
# -------------------------------------------------------------------------- #
def build_facet_tags(msh, markers_spec):
    """Return (facet_tags, name_to_tag, tag_to_facets).

    `markers_spec` is a list of `(tag, name, locator)` triples.
    """
    fdim = msh.topology.dim - 1
    tag_ids, names, facet_arrays = [], [], []
    tag_to_facets = {}
    name_to_tag = {}
    for tag, name, loc in markers_spec:
        facets = dmesh.locate_entities_boundary(msh, fdim, loc)
        tag_ids.append(tag)
        names.append(name)
        facet_arrays.append(facets)
        tag_to_facets[tag] = facets
        name_to_tag[name] = tag

    indices = np.hstack(facet_arrays).astype(np.int32)
    values  = np.hstack([np.full_like(fa, tg)
                         for fa, tg in zip(facet_arrays, tag_ids)]).astype(np.int32)
    order   = np.argsort(indices)
    ft = dmesh.meshtags(msh, fdim, indices[order], values[order])
    return ft, name_to_tag, tag_to_facets


# -------------------------------------------------------------------------- #
# Vertex / edge entity markers (for pinning a single point, e.g. top-right).
# -------------------------------------------------------------------------- #
def locate_corner(msh, x_target, y_target, z_target=None, tol=0.01):
    """Locate the (fdim - 1) entity closest to a given corner, used for
    point-pinning tricks (fixing u_x on the top-right corner of the plate)."""
    fdim = msh.topology.dim
    if msh.geometry.dim == 2:
        def marker(x):
            return (np.abs(x[0] - x_target) < tol) & (np.abs(x[1] - y_target) < tol)
        return dmesh.locate_entities(msh, fdim - 2, marker)
    else:
        def marker(x):
            ok = (np.abs(x[0] - x_target) < tol) & (np.abs(x[1] - y_target) < tol)
            if z_target is not None:
                ok &= (np.abs(x[2] - z_target) < tol)
            return ok
        return dmesh.locate_entities(msh, fdim - 3, marker)


# -------------------------------------------------------------------------- #
# Assemble Dirichlet BCs from a compact, named specification.
# -------------------------------------------------------------------------- #
def build_dirichlet_bcs(
    V: fem.FunctionSpace,
    msh,
    name_to_tag_to_facets: Dict[int, np.ndarray],
    spec: List[Tuple[str, int | None, Any]],
):
    """Build a list of `dirichletbc` objects from a spec list.

    Each entry in `spec` is (facet_key, component, value) where:
      - facet_key: either the numeric tag (int) or the string name from
        `name_to_tag_to_facets` lookup done by the caller (here we take the
        facet array directly).
      - component: None → full vector BC; 0/1/2 → one scalar component.
      - value: either a float/int (ScalarType), or a `fem.Constant`, or a
        `fem.Function` already on the right space for full-vector BCs.
    """
    bcs = []
    fdim = msh.topology.dim - 1
    for facet_key, comp, val in spec:
        facets = (name_to_tag_to_facets[facet_key]
                  if isinstance(facet_key, int)
                  else name_to_tag_to_facets[facet_key])
        if comp is None:
            dofs = fem.locate_dofs_topological(V, fdim, facets)
            if isinstance(val, (int, float)):
                raise TypeError("Vector BC requires a fem.Constant, fem.Function,"
                                " or numpy array, not a scalar.")
            bcs.append(fem.dirichletbc(val, dofs)
                       if isinstance(val, fem.Function)
                       else fem.dirichletbc(val, dofs, V))
        else:
            Vsub = V.sub(comp)
            dofs = fem.locate_dofs_topological(Vsub, fdim, facets)
            value = (val if isinstance(val, fem.Constant)
                     else np.array(val, dtype=ScalarType))
            bcs.append(fem.dirichletbc(value, dofs, Vsub))
    return bcs


# -------------------------------------------------------------------------- #
# Traction and body forces for weak forms.
# -------------------------------------------------------------------------- #
def traction_form(v, ds, tractions: Dict[int, Any]):
    """Weak-form contribution  Σ_i ∫_∂Ω_i (t_i · v) ds_i.

    `tractions = {facet_tag: traction_vector_UFL}` — each entry is a UFL
    vector-valued expression (e.g. `press * n`, `as_vector([0.0, p])`).
    Returns None when the dict is empty so the caller can keep the form
    compact.
    """
    from ufl import inner
    if not tractions:
        return None
    term = None
    for tag, t in tractions.items():
        contrib = inner(t, v) * ds(tag)
        term = contrib if term is None else term + contrib
    return term


def body_force_form(v, dx, b):
    """Weak-form contribution ∫_Ω (b · v) dx. `b` is a UFL vector or None."""
    from ufl import inner
    return None if b is None else inner(b, v) * dx
