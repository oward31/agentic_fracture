"""Problem-builder factories. Each returns `build_problem(msh) -> P`.

Common P dict keys (all problems):
    V, Y, Yv          — function spaces
    u, z, z_lb, z_ub  — primary fields
    z_trial           — stagger-iteration snapshot of z
    bcs_u, bcs_z      — Dirichlet BC lists
    dx, ds, n         — measures and outward normal
    facet_tags        — meshtags
    name_to_tag       — dict mapping facet names to integer tags
    problem_u, problem_z — SNESSolver wrappers
    R_u               — residual UFL form (for reaction force)
    indicator, indicator_expr, cell_h  — AMR fields
    z_diff, z_diff_norm_form, u_diff, u_diff_norm_form — stagger residuals
    mat, eps, h0, h_min, dp — propagated for AMR and post-processing
    disp_const        — (optional) ramping displacement Constant

Dynamic and ductile variants add more keys (u_prev, v_prev, a_prev, sig, p,
eps_pl, ...).
"""

from .linear_elastic_2d_pe     import make_linear_elastic_2d_pe_builder
from .linear_elastic_2d_ps     import make_linear_elastic_2d_ps_builder
from .linear_elastic_3d        import make_linear_elastic_3d_builder
from .dynamic_2d               import make_dynamic_2d_builder
from .ductile_2d_pe            import make_ductile_2d_pe_builder
from .ductile_3d               import make_ductile_3d_builder
from .finite_elastic_2d_ps     import make_finite_elastic_2d_ps_builder
from .finite_elastic_2d_pe     import make_finite_elastic_2d_pe_builder
from .finite_elastic_3d        import make_finite_elastic_3d_builder
