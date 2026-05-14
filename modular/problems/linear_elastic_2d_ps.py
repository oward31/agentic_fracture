"""Linear elasticity, 2D plane stress — displacement-controlled phase-field."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from ufl import TestFunction, TrialFunction, derivative, inner

from ..common.bcs import build_dirichlet_bcs, locate_corner, traction_form, body_force_form
from ..common.snes import SNESSolver, OPTS_U, OPTS_Z_VI
from ..constitutive import linear_elasticity_2d_ps as const2D_PS
from ..constitutive.drucker_prager import (drucker_prager_coefficients,
                                           drucker_prager_driving_force,
                                           pf_flag_indicator,
                                           pf_residual_gradient_term)
from ._common import (cache_cell_size_and_indicator, function_spaces,
                      make_phasefield_fields, stagger_residual_forms,
                      tag_and_measures,
                      materialise_bc_constants)


def make_linear_elastic_2d_ps_builder(
    mat: Dict,
    markers_spec: List,
    geom: Dict,
    *,
    eps: float,
    h0: float,
    h_min: float,
    bc_spec_u: Optional[List[Tuple]] = None,
    traction_spec: Optional[Dict[int, any]] = None,
    body_force: Optional[any] = None,
    pin_point: Optional[Tuple[float, float]] = None,
    disp_const: Optional[fem.Constant] = None,
) -> Callable:
    dp_coef = drucker_prager_coefficients(mat, eps=eps, h_min=h_min)

    def build_problem(msh):
        V, Y, Yv = function_spaces(msh, vector_dim=2)
        ft, name_to_tag, tag_to_facets, dx, ds, n = tag_and_measures(msh, markers_spec)

        u = fem.Function(V, name="displacement")
        v = TestFunction(V); du = TrialFunction(V)
        z, z_trial, z_lb, z_ub = make_phasefield_fields(msh, Y)
        y = TestFunction(Y); dz = TrialFunction(Y)

        dc = disp_const if disp_const is not None else fem.Constant(msh, ScalarType(0.0))

        # Materialise any "@ramp:N" string markers in bc_spec_u into
        # fresh fem.Constants on THIS mesh, so the AMR rebuild path
        # doesn't leave the BCs pointing at stale (coarse-mesh) values.
        fresh_bc_spec_u, bc_value_constants = materialise_bc_constants(msh, bc_spec_u)
        if fresh_bc_spec_u is None:
            bcs_u = _default_bcs_u(msh, V, tag_to_facets, dc, pin_point, geom)
        else:
            bcs_u = build_dirichlet_bcs(V, msh, tag_to_facets, fresh_bc_spec_u)
        bcs_z: list = []

        mu, lmbda, nu, eta = mat["mu"], mat["lmbda"], mat["nu"], mat["eta"]
        em      = const2D_PS.epsilon(u)
        sig     = const2D_PS.sigma(em, mu, lmbda, nu)
        dgd     = const2D_PS.dgd(z, eta)
        psi1    = const2D_PS.energy(em, mu, lmbda, nu)
        I1_und  = const2D_PS.I1_0(em, mu, lmbda, nu)
        SQJ2_u  = const2D_PS.sigmavm(em, mu, lmbda, nu)
        I1_deg  = dgd * I1_und
        SQJ2_d  = dgd * SQJ2_u

        ce = drucker_prager_driving_force(z, psi1, I1_deg, SQJ2_d, I1_und, dp_coef)

        R_u = inner(dgd * sig, const2D_PS.epsilon(v)) * dx
        t_form = traction_form(v, ds, traction_spec or {})
        b_form = body_force_form(v, dx, body_force)
        if t_form is not None: R_u = R_u - t_form
        if b_form is not None: R_u = R_u - b_form

        R_z = (y * 2.0 * z * psi1 * dx
               - y * ce * dx
               + pf_residual_gradient_term(y, z, Gc=mat["Gc"], eps=eps,
                                           delta=dp_coef["delta"]) * dx)

        problem_u = SNESSolver(R_u, u, bcs=bcs_u, J_form=derivative(R_u, u, du),
                               petsc_options=OPTS_U)
        problem_z = SNESSolver(R_z, z, bcs=bcs_z, J_form=derivative(R_z, z, dz),
                               petsc_options=OPTS_Z_VI, bounds=(z_lb, z_ub))

        indicator = pf_flag_indicator(z, psi1, ce,
                                       Gc=mat["Gc"], eps=eps,
                                       delta=dp_coef["delta"])
        cell_h, indicator_expr = cache_cell_size_and_indicator(msh, Yv, indicator)
        u_trial, u_diff, u_form, z_diff, z_form = stagger_residual_forms(u, z, V, Y, dx)

        return dict(
            msh=msh, V=V, Y=Y, Yv=Yv,
            u=u, u_trial=u_trial,
            z=z, z_trial=z_trial, z_lb=z_lb, z_ub=z_ub,
            disp_const=dc,
            bc_value_constants=bc_value_constants,
            bcs_u=bcs_u, bcs_z=bcs_z,
            facet_tags=ft, name_to_tag=name_to_tag, tag_to_facets=tag_to_facets,
            dx=dx, ds=ds, n=n,
            R_u=R_u, R_z=R_z,
            problem_u=problem_u, problem_z=problem_z,
            sigma=sig, dgd=dgd, psi1=psi1, sigma_vm=SQJ2_u,
            indicator=indicator, indicator_expr=indicator_expr, cell_h=cell_h,
            u_diff=u_diff, u_diff_norm_form=u_form,
            z_diff=z_diff, z_diff_norm_form=z_form,
            mat=mat, eps=eps, h0=h0, h_min=h_min, dp=dp_coef,
        )

    return build_problem


# -------------------------------------------------------------------------- #
def _default_bcs_u(msh, V, tag_to_facets, disp_const, pin_point, geom):
    """Default BCs: bottom u_y = 0, top u_y = disp_const, top-right corner u_x = 0."""
    fdim = msh.topology.dim - 1
    bot_y = fem.locate_dofs_topological(V.sub(1), fdim, tag_to_facets[4])
    top_y = fem.locate_dofs_topological(V.sub(1), fdim, tag_to_facets[3])
    bcs = [
        fem.dirichletbc(np.array(0.0, dtype=ScalarType), bot_y, V.sub(1)),
        fem.dirichletbc(disp_const, top_y, V.sub(1)),
    ]
    if pin_point is None and "W" in geom and "L" in geom:
        pin_point = (geom["W"], geom["L"] / 2.0)
    if pin_point is not None:
        pin_ents = locate_corner(msh, pin_point[0], pin_point[1])
        pin_dofs = fem.locate_dofs_topological(V.sub(0), fdim - 1, pin_ents)
        bcs.insert(0, fem.dirichletbc(np.array(0.0, dtype=ScalarType), pin_dofs, V.sub(0)))
    return bcs
