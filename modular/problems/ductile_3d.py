"""Ductile phase-field fracture (3D, J2 plasticity).

Straightforward 3D extension of `ductile_2d_pe.py`:
  * Voigt layout is (6,): [σxx, σyy, σzz, σxy, σxz, σyz]
  * Strain is just `sym(grad(u))` — no 2D→3D promotion
  * Default BCs: bottom (y=0) clamped, top u_y ramped, one corner of bottom
    pins u_x/u_z to remove rigid rotation.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from ufl import (TestFunction, TrialFunction, derivative, grad, inner,
                 conditional, max_value)

from ..common.bcs import body_force_form, build_dirichlet_bcs, traction_form
from ..common.snes import SNESSolver, OPTS_U, OPTS_Z_VI
from ..constitutive import j2_plasticity as j2
from ..constitutive.drucker_prager import drucker_prager_coefficients
from ._common import (cache_cell_size_and_indicator, function_spaces,
                      make_phasefield_fields, stagger_residual_forms,
                      tag_and_measures,
                      materialise_bc_constants)


def make_ductile_3d_builder(
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
    disp_const: Optional[fem.Constant] = None,
    q_deg: int = 2,
    p_amr: float = 0.025,
) -> Callable:
    dp_coef = drucker_prager_coefficients(mat, eps=eps, h_min=h_min)

    def build_problem(msh):
        V, Y, Yv = function_spaces(msh, vector_dim=3)
        ft, name_to_tag, tag_to_facets, dx, ds, n = tag_and_measures(
            msh, markers_spec, quadrature_degree=q_deg)

        W_tensor, W_scalar = j2.make_quadrature_spaces(msh, q_deg=q_deg, voigt_dim=6)

        u = fem.Function(V, name="displacement")
        v_test = TestFunction(V); du = TrialFunction(V)
        z, z_trial, z_lb, z_ub = make_phasefield_fields(msh, Y)
        y_test = TestFunction(Y); dz = TrialFunction(Y)

        sig         = fem.Function(W_tensor, name="stress_voigt")
        p           = fem.Function(W_scalar, name="p_eq")
        eps_pl      = fem.Function(W_tensor, name="eps_pl_voigt")
        dp_stored   = fem.Function(W_scalar)
        deps_pl_s   = fem.Function(W_tensor)
        p_old       = fem.Function(W_scalar)
        eps_pl_old  = fem.Function(W_tensor)

        dc = disp_const if disp_const is not None else fem.Constant(msh, ScalarType(0.0))

        # Materialise any "@ramp:N" string markers in bc_spec_u into fresh
        # fem.Constants on THIS mesh — keeps BCs valid after AMR rebuild.
        fresh_bc_spec_u, bc_value_constants = materialise_bc_constants(msh, bc_spec_u)
        if fresh_bc_spec_u is None:
            bcs_u = _default_bcs_u_3d(msh, V, tag_to_facets, dc, geom)
        else:
            bcs_u = build_dirichlet_bcs(V, msh, tag_to_facets, fresh_bc_spec_u)
        bcs_z: list = []

        eps_pl_3D = j2.voigt_to_tensor_3d(eps_pl)
        new_sig, dp_, deps_pl_ = j2.compute_stress_update(
            u, eps_pl_3D, p, z, mat, strain_fn=j2.strain_3d)

        degradation = j2.dgd_ductile(z, p, mat["eta"])
        psi_driving = j2.elastoplastic_energy(u, eps_pl_3D, mat["mu"], mat["lmbda"],
                                               strain_fn=j2.strain_3d)

        R_u = inner(degradation * new_sig, j2.strain_3d(v_test)) * dx
        t_form = traction_form(v_test, ds, traction_spec or {})
        b_form = body_force_form(v_test, dx, body_force)
        if t_form is not None: R_u = R_u - t_form
        if b_form is not None: R_u = R_u - b_form

        P_SAT = j2.P_SAT
        R_z = (y_test * 2.0 * (p / P_SAT) ** 2 * (z ** (2.0 * (p / P_SAT) ** 2 - 1.0))
               * psi_driving * dx
               + mat["Gc"] * (y_test * (z - 1.0) / eps / 2.0
                              + 2.0 * eps * inner(grad(z), grad(y_test))) * dx)

        problem_u = SNESSolver(R_u, u, bcs=bcs_u, J_form=derivative(R_u, u, du),
                               petsc_options=OPTS_U)
        problem_z = SNESSolver(R_z, z, bcs=bcs_z, J_form=derivative(R_z, z, dz),
                               petsc_options=OPTS_Z_VI, bounds=(z_lb, z_ub))

        # Indicator evaluated via L2 projection (p is a quadrature function).
        from ufl import CellDiameter
        indicator = max_value(conditional(p > p_amr, 1.0, 0.0),
                              conditional(z < 0.2, 1.0, 0.0))
        ipts = Yv.element.interpolation_points()
        cell_h = fem.Function(Yv)
        cell_h.interpolate(fem.Expression(CellDiameter(msh), ipts))
        from ..post.reaction import custom_project
        def indicator_proj_fn(P_ref):
            return custom_project(P_ref["indicator"], P_ref["Yv"], P_ref["dx"])
        u_trial, u_diff, u_form, z_diff, z_form = stagger_residual_forms(u, z, V, Y, dx)

        return dict(
            msh=msh, V=V, Y=Y, Yv=Yv, W_tensor=W_tensor, W_scalar=W_scalar,
            u=u, u_trial=u_trial,
            z=z, z_trial=z_trial, z_lb=z_lb, z_ub=z_ub,
            disp_const=dc,
            bc_value_constants=bc_value_constants,
            bcs_u=bcs_u, bcs_z=bcs_z,
            facet_tags=ft, name_to_tag=name_to_tag, tag_to_facets=tag_to_facets,
            dx=dx, ds=ds, n=n,
            R_u=R_u, R_z=R_z,
            problem_u=problem_u, problem_z=problem_z,
            sigma=new_sig, sigma_vm=j2.von_mises(new_sig), dgd=degradation,
            sig=sig, p=p, eps_pl=eps_pl,
            p_old=p_old, eps_pl_old=eps_pl_old,
            dp_stored=dp_stored, deps_pl_stored=deps_pl_s,
            new_sig=new_sig, dp_=dp_, deps_pl_=deps_pl_,
            indicator=indicator, indicator_proj_fn=indicator_proj_fn, cell_h=cell_h,
            u_diff=u_diff, u_diff_norm_form=u_form,
            z_diff=z_diff, z_diff_norm_form=z_form,
            mat=mat, eps=eps, h0=h0, h_min=h_min, dp=dp_coef,
            q_deg=q_deg, voigt_dim=6,
            strain_fn=j2.strain_3d,
            voigt_from_tensor=j2.tensor_to_voigt_3d,
            voigt_to_tensor=j2.voigt_to_tensor_3d,
        )

    return build_problem


def _default_bcs_u_3d(msh, V, tag_to_facets, disp_const, geom):
    fdim = msh.topology.dim - 1
    bot = tag_to_facets[1]  # bottom
    top = tag_to_facets[2]  # top
    bot_y = fem.locate_dofs_topological(V.sub(1), fdim, bot)
    top_y = fem.locate_dofs_topological(V.sub(1), fdim, top)
    bot_x = fem.locate_dofs_topological(V.sub(0), fdim, bot)
    bot_z = fem.locate_dofs_topological(V.sub(2), fdim, bot)
    return [
        fem.dirichletbc(disp_const, top_y, V.sub(1)),
        fem.dirichletbc(np.array(0.0, dtype=ScalarType), bot_y, V.sub(1)),
        fem.dirichletbc(np.array(0.0, dtype=ScalarType), bot_x, V.sub(0)),
        fem.dirichletbc(np.array(0.0, dtype=ScalarType), bot_z, V.sub(2)),
    ]
