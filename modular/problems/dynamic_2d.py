"""Dynamic phase-field fracture, 2D plane stress — HHT-α time integration.

Mirrors `dyn_branch_amr.py`: traction loading on top & bottom, HHT-α with
α_f = 0.1, α_m = 0. Mass is degraded by z²+η; same Drucker-Prager driving
term as the quasistatic cases.

The `pressure` Constant is exposed via P["pressure"] so the driver can ramp
it over time.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

from dolfinx import fem
from petsc4py.PETSc import ScalarType
from ufl import (TestFunction, TrialFunction, as_vector, derivative, inner)

from ..common.bcs import body_force_form, build_dirichlet_bcs, traction_form
from ..common.snes import SNESSolver, OPTS_U_DYN, OPTS_Z_VI_DYN
from ..constitutive import linear_elasticity_2d_ps as const2D_PS
from ..constitutive.drucker_prager import (drucker_prager_coefficients,
                                           drucker_prager_driving_force,
                                           pf_flag_indicator,
                                           pf_residual_gradient_term)
from ._common import (cache_cell_size_and_indicator, function_spaces,
                      make_phasefield_fields, stagger_residual_forms,
                      tag_and_measures,
                      materialise_bc_constants)


# --- HHT-α parameters ------------------------------------------------------ #
ALPHA_F = 0.1
ALPHA_M = 0.0
GAMMA_NM = 0.5 + ALPHA_F
BETA_NM  = 0.25 * (1.0 + ALPHA_F) ** 2


def _avg(x_old, x_new, alpha, d: int):
    return as_vector([alpha * x_old[i] + (1.0 - alpha) * x_new[i] for i in range(d)])


def update_a_ufl(u_new, u_prev, v_prev, a_prev, dt):
    """Newmark acceleration expression for use in the UFL residual."""
    return ((u_new - u_prev - dt * v_prev) / (BETA_NM * dt ** 2)
            - ((1.0 - 2.0 * BETA_NM) / (2.0 * BETA_NM)) * a_prev)


def update_a_arr(u_arr, u_prev_arr, v_prev_arr, a_prev_arr, dt):
    b = float(BETA_NM)
    return (u_arr - u_prev_arr - dt * v_prev_arr) / (b * dt ** 2) \
           - ((1.0 - 2.0 * b) / (2.0 * b)) * a_prev_arr


def update_v_arr(a_new_arr, u_prev_arr, v_prev_arr, a_prev_arr, dt):
    g = float(GAMMA_NM)
    return v_prev_arr + dt * ((1.0 - g) * a_prev_arr + g * a_new_arr)


def advance_fields(P, dt: float):
    """End-of-step: compute new a, v, then shift u_prev ← u."""
    a = update_a_arr(P["u"].x.array, P["u_prev"].x.array,
                     P["v_prev"].x.array, P["a_prev"].x.array, dt)
    v = update_v_arr(a, P["u_prev"].x.array, P["v_prev"].x.array,
                     P["a_prev"].x.array, dt)
    P["a_prev"].x.array[:] = a
    P["v_prev"].x.array[:] = v
    P["u_prev"].x.array[:] = P["u"].x.array
    for k in ("a_prev", "v_prev", "u_prev"):
        P[k].x.scatter_forward()


# -------------------------------------------------------------------------- #
def make_dynamic_2d_builder(
    mat: Dict,
    markers_spec: List,
    geom: Dict,
    *,
    eps: float,
    h0: float,
    h_min: float,
    dt: float,
    traction_spec: Optional[Dict[int, any]] = None,
    bc_spec_u: Optional[List[Tuple]] = None,
    pressure: Optional[fem.Constant] = None,
) -> Callable:
    """`pressure` is a Constant ramped by the driver; `traction_spec` maps
    facet tags to UFL vectors (e.g. `{3: pressure*n, 4: pressure*n}`).

    Either `traction_spec` or `pressure` must be supplied; the standard
    pattern is:

        pressure = fem.Constant(msh, 0.0)
        tr = {ntop: pressure * n, nbot: pressure * n}
        build = make_dynamic_2d_builder(... , traction_spec=tr, pressure=pressure)
    """
    dp_coef = drucker_prager_coefficients(mat, eps=eps, h_min=h_min)
    dt_const_holder = {"dt": dt}   # mutable so the driver can update it

    def build_problem(msh):
        V, Y, Yv = function_spaces(msh, vector_dim=2)
        ft, name_to_tag, tag_to_facets, dx, ds, n = tag_and_measures(msh, markers_spec)

        u      = fem.Function(V, name="displacement")
        u_prev = fem.Function(V, name="u_prev")
        v_prev = fem.Function(V, name="v_prev")
        a_prev = fem.Function(V, name="a_prev")
        v = TestFunction(V); du = TrialFunction(V)
        z, z_trial, z_lb, z_ub = make_phasefield_fields(msh, Y)
        y = TestFunction(Y); dz = TrialFunction(Y)

        # Constant pressure (ramped by driver).
        press = pressure if pressure is not None else fem.Constant(msh, ScalarType(0.0))
        dt_c  = fem.Constant(msh, ScalarType(dt_const_holder["dt"]))

        # No Dirichlet BCs by default — loading is via traction.
        fresh_bc_spec_u, bc_value_constants = materialise_bc_constants(msh, bc_spec_u)
        bcs_u = (build_dirichlet_bcs(V, msh, tag_to_facets, fresh_bc_spec_u)
                 if fresh_bc_spec_u else [])
        bcs_z: list = []

        # Constitutive (plane stress).
        mu, lmbda, nu, eta, rho = (mat["mu"], mat["lmbda"], mat["nu"],
                                    mat["eta"], mat["rho"])
        em      = const2D_PS.epsilon(u)
        sig     = const2D_PS.sigma(em, mu, lmbda, nu)
        dgd     = const2D_PS.dgd(z, eta)
        psi1    = const2D_PS.energy(em, mu, lmbda, nu)
        I1_und  = const2D_PS.I1_0(em, mu, lmbda, nu)
        SQJ2_u  = const2D_PS.sigmavm(em, mu, lmbda, nu)

        ce = drucker_prager_driving_force(
            z, psi1, dgd * I1_und, dgd * SQJ2_u, I1_und, dp_coef,
        )

        a_new = update_a_ufl(u, u_prev, v_prev, a_prev, dt_c)

        def m_form(u_, v_, z_):
            return dgd * rho * inner(u_, v_) * dx

        def k_form(u_, v_, z_):
            return inner(dgd * const2D_PS.sigma(const2D_PS.epsilon(u_), mu, lmbda, nu),
                         const2D_PS.epsilon(v_)) * dx

        # Traction (default: pressure · n on top & bottom if traction_spec is None).
        if traction_spec is None:
            try:
                ntop = name_to_tag["top"]; nbot = name_to_tag["bottom"]
                tr = {ntop: press * n, nbot: press * n}
            except KeyError:
                tr = {}
        else:
            tr = traction_spec

        R_u = (k_form(_avg(u_prev, u, ALPHA_F, 2), v, z)
             + m_form(_avg(a_prev, a_new, ALPHA_M, 2), v, z))
        t_form = traction_form(v, ds, tr)
        if t_form is not None: R_u = R_u - t_form

        R_z = (y * 2.0 * z * psi1 * dx
               - y * ce * dx
               + pf_residual_gradient_term(y, z, Gc=mat["Gc"], eps=eps,
                                           delta=dp_coef["delta"]) * dx)

        problem_u = SNESSolver(R_u, u, bcs=bcs_u, J_form=derivative(R_u, u, du),
                               petsc_options=OPTS_U_DYN)
        problem_z = SNESSolver(R_z, z, bcs=bcs_z, J_form=derivative(R_z, z, dz),
                               petsc_options=OPTS_Z_VI_DYN, bounds=(z_lb, z_ub))

        indicator = pf_flag_indicator(z, psi1, ce,
                                       Gc=mat["Gc"], eps=eps,
                                       delta=dp_coef["delta"])
        cell_h, indicator_expr = cache_cell_size_and_indicator(msh, Yv, indicator)
        u_trial, u_diff, u_form, z_diff, z_form = stagger_residual_forms(u, z, V, Y, dx)

        return dict(
            msh=msh, V=V, Y=Y, Yv=Yv,
            u=u, u_trial=u_trial, u_prev=u_prev, v_prev=v_prev, a_prev=a_prev,
            z=z, z_trial=z_trial, z_lb=z_lb, z_ub=z_ub,
            pressure=press, dt_const=dt_c,
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
            # Helpers for time-stepping driver.
            advance_fields=advance_fields,
        )

    return build_problem
