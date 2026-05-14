"""Finite elasticity (Lopez-Pamies Ogden), 2D plane stress — matches slant_amr.

Uses Crouzeix-Raviart elements with Nitsche/DG stabilisation, custom Newton
with CG + BoomerAMG, and H1 gradient-flow regularisation (ω).

The 'nice' adaptive-time-step logic lives in the solver module; this builder
just produces the forms, KSPs and the canned `omega_c` Constant.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional

import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from ufl import (CellDiameter, FacetArea, FacetNormal, Measure,
                 TestFunction, TrialFunction, conditional, derivative,
                 dot, grad, inner, jump, lt, max_value)

from ..common.bcs import build_facet_tags
from ..constitutive import finite_elasticity as fe
from ..constitutive.drucker_prager import drucker_prager_coefficients


# Solver tolerances for the CG/BoomerAMG inner KSPs (matches slant_amr.py).
KSP_RTOL   = 1.0e-10
KSP_ATOL   = 1.0e-12
KSP_MAX_IT = 500
OMEGA_INIT = 1.0e6


def make_finite_elastic_2d_ps_builder(
    mat: Dict,
    markers_spec: List,
    geom: Dict,
    *,
    eps: float,
    h0: float,
    h_min: float,
    bc_spec_top: Optional[Callable] = None,
    traction_spec: Optional[Dict[int, any]] = None,
    top_value_fn: Optional[Callable] = None,
    maxdisp: float = 1.0,
) -> Callable:
    """Build a finite-elastic 2D plane-stress problem in slant_amr style.

    `top_value_fn`: optional `(t) -> callable(x) -> (2, n)` used to prescribe
    u on the top face. Default: u = (0, t·maxdisp).
    """
    dp_coef = drucker_prager_coefficients(mat, eps=eps, h_min=h_min)

    def build_problem(msh):
        tdim = msh.topology.dim
        fdim = tdim - 1
        msh.topology.create_connectivity(fdim, tdim)

        d = 2
        V      = fem.functionspace(msh, ("CR", 1, (d,)))
        Y      = fem.functionspace(msh, ("Lagrange", 1))
        Yg     = fem.functionspace(msh, ("DG", 0))
        V_plot = fem.functionspace(msh, ("Lagrange", 1, (d,)))

        # Fields.
        u           = fem.Function(V, name="displacement")
        u_prev      = fem.Function(V)
        u_prev_prev = fem.Function(V)
        u_inc       = fem.Function(V)
        v_          = TestFunction(V); du = TrialFunction(V)

        z      = fem.Function(Y, name="phasefield"); z.x.array[:] = 1.0
        z.x.scatter_forward()
        z_prev = fem.Function(Y); z_prev.x.array[:] = 1.0; z_prev.x.scatter_forward()
        z_inc  = fem.Function(Y)
        z_lb   = fem.Function(Y); z_lb.x.array[:] = 0.0; z_lb.x.scatter_forward()
        z_ub   = fem.Function(Y); z_ub.x.array[:] = 1.0; z_ub.x.scatter_forward()
        y_     = TestFunction(Y); dz = TrialFunction(Y)

        # Facet tags and measures.
        ft, name_to_tag, tag_to_facets = build_facet_tags(msh, markers_spec)
        dx = Measure("dx", domain=msh)
        ds = Measure("ds", domain=msh, subdomain_data=ft)
        dS = Measure("dS", domain=msh)
        n_ = FacetNormal(msh)
        h_ = FacetArea(msh)
        h_avg = (h_("+") + h_("-")) / 2.0

        # Required BC tags: "bottom" (clamped), "top" (prescribed).
        bot = tag_to_facets[name_to_tag["bottom"]]
        top = tag_to_facets[name_to_tag["top"]]

        r_func    = fem.Function(V)   # prescribed top value (t·maxdisp in y)
        zero_func = fem.Function(V)

        bottom_dofs = fem.locate_dofs_topological(V, fdim, bot)
        top_dofs    = fem.locate_dofs_topological(V, fdim, top)
        bcs_u = [fem.dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), bottom_dofs, V),
                 fem.dirichletbc(r_func, top_dofs)]
        bcs_du = [fem.dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), bottom_dofs, V),
                  fem.dirichletbc(zero_func, top_dofs)]

        top_dofs_Y = fem.locate_dofs_topological(Y, fdim, top)
        bot_dofs_Y = fem.locate_dofs_topological(Y, fdim, bot)
        bcs_z = [fem.dirichletbc(PETSc.ScalarType(1.0), top_dofs_Y, Y),
                 fem.dirichletbc(PETSc.ScalarType(1.0), bot_dofs_Y, Y)]
        bcs_dz = [fem.dirichletbc(PETSc.ScalarType(0.0), top_dofs_Y, Y),
                  fem.dirichletbc(PETSc.ScalarType(0.0), bot_dofs_Y, Y)]

        # Kinematics and energy.
        k = fe.kinematics_2d_plane_stress(u, mat)
        psi1  = fe.energy_density(z, k, mat)
        psi11 = fe.energy_density_undegraded(k, mat)
        sigma_I1, sigma_vm = fe.stress_invariants(k, mat)

        # Drucker-Prager driving term.
        delta = dp_coef["delta"]; beta1 = dp_coef["beta1"]; beta2 = dp_coef["beta2"]
        ce = (beta2 * (z ** 2) * sigma_vm
            + beta1 * (z ** 2) * sigma_I1
            + conditional(lt(sigma_I1, 0.0), 2.0 * z * psi11, 0.0))

        # Displacement residual (CR + Nitsche).
        Pi = psi1 * dx
        R1 = derivative(Pi, u, v_)
        R2 = ((5.0 / h_avg) * dot(jump(u), jump(v_)) * dS
            + (5.0 / h_)   * dot(u, v_) * ds(name_to_tag["bottom"])
            + (5.0 / h_)   * dot(u - r_func, v_) * ds(name_to_tag["top"]))
        R_u = R1 + R2
        J_u = derivative(R_u, u, du)

        omega_c = fem.Constant(msh, PETSc.ScalarType(OMEGA_INIT))
        J_reg   = J_u + (1.0 / omega_c) * inner(grad(du), grad(v_)) * dx

        R_u_form   = fem.form(R_u)
        J_reg_form = fem.form(J_reg)
        A_u = fem.petsc.create_matrix(J_reg_form)
        b_u = fem.petsc.create_vector(R_u_form)

        ksp_u = PETSc.KSP().create(msh.comm)
        ksp_u.setType("cg"); ksp_u.setTolerances(rtol=KSP_RTOL, atol=KSP_ATOL, max_it=KSP_MAX_IT)
        pc_u = ksp_u.getPC(); pc_u.setType("hypre"); pc_u.setHYPREType("boomeramg")

        # Phase-field residual.
        Gc = mat["Gc"]
        pen = 1000.0 * (3.0 * Gc / 8.0 / eps) * conditional(lt(delta, 1.0), 1.0, delta)
        Wv  = pen / 2.0 * ((abs(z) - z) ** 2 + (abs(1.0 - z) - (1.0 - z)) ** 2) * dx
        R_z = (y_ * 2.0 * z * psi11 * dx
             - y_ * ce * dx
             + 3.0 * delta * Gc / 8.0 * (y_ * (-1.0) / eps) * dx
             + 3.0 * Gc * delta / 8.0 * (2.0 * eps * inner(grad(z), grad(y_))) * dx
             + derivative(Wv, z, y_))
        J_z = derivative(R_z, z, dz)
        R_z_form = fem.form(R_z)
        J_z_form = fem.form(J_z)
        A_z = fem.petsc.create_matrix(J_z_form)
        b_z = fem.petsc.create_vector(R_z_form)

        ksp_z = PETSc.KSP().create(msh.comm)
        ksp_z.setType("cg"); ksp_z.setTolerances(rtol=KSP_RTOL, atol=KSP_ATOL, max_it=KSP_MAX_IT)
        pc_z = ksp_z.getPC(); pc_z.setType("hypre"); pc_z.setHYPREType("boomeramg")

        # Stress form for reaction-force post-processing.
        P_stress = fe.first_piola_2d_plane_stress(u, z, mat)
        stress_form_top_y = fem.form(P_stress[1, 1] * ds(name_to_tag["top"]))

        # AMR indicator.
        pf_flag = (2.0 * z * psi11 - ce - 3.0 * delta * Gc / 8.0 / eps) \
                  / (3.0 * delta * Gc / 8.0 / eps)
        indicator = max_value(conditional(pf_flag > -0.1, 1.0, 0.0),
                              conditional(z < 0.2, 1.0, 0.0))
        ipts = Yg.element.interpolation_points()
        indicator_expr = fem.Expression(indicator, ipts)
        cell_h = fem.Function(Yg)
        cell_h.interpolate(fem.Expression(CellDiameter(msh), ipts))

        return dict(
            msh=msh, V=V, Y=Y, Yv=Yg, Yg=Yg, V_plot=V_plot,
            u=u, u_prev=u_prev, u_prev_prev=u_prev_prev, u_inc=u_inc,
            z=z, z_prev=z_prev, z_inc=z_inc, z_lb=z_lb, z_ub=z_ub,
            r_func=r_func, zero_func=zero_func,
            bcs_u=bcs_u, bcs_du=bcs_du, bcs_z=bcs_z, bcs_dz=bcs_dz,
            facet_tags=ft, name_to_tag=name_to_tag, tag_to_facets=tag_to_facets,
            dx=dx, ds=ds, dS=dS, n=n_,
            R_u_form=R_u_form, J_reg_form=J_reg_form, A_u=A_u, b_u=b_u,
            R_z_form=R_z_form, J_z_form=J_z_form, A_z=A_z, b_z=b_z,
            ksp_u=ksp_u, ksp_z=ksp_z,
            omega_c=omega_c, maxdisp=maxdisp,
            stress_form=stress_form_top_y,
            indicator=indicator, indicator_expr=indicator_expr, cell_h=cell_h,
            mat=mat, eps=eps, h0=h0, h_min=h_min, dp=dp_coef,
            sigma_vm=sigma_vm,
        )

    return build_problem
