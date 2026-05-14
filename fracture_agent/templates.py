"""Runnable-script renderer.

All nine modular variants follow the same six-step shape:

    load_material → make_mesh → make_*_builder → build_problem → solver → output

so we emit a single parametrically-rendered script instead of nine bespoke
templates.  The Synthesizer feeds this function the Action + CanonicalSpec +
MeshSizePlan; the product is a standalone Python file that the Executor runs
verbatim via WSL.

The script is written relative to the repository root so that
``from modular.* import ...`` works without further path munging.

User-frame vs mesh-frame coordinates
------------------------------------
The architect describes pre-cracks (and any other explicit (x, y[, z])
coordinate) in the **user's frame**: corners at (0, 0) and (W, L).  Most
custom-mesh modules also use this frame, so no transform is needed.  But
some built-in builders centre the y axis: ``make_notched_plate_2d`` and
friends use ``[0, W] × [-L/2, L/2]``, with the pre-crack at y = 0 instead
of y = L/2.  ``_user_to_mesh_xy`` returns the affine shift that maps a
user-frame point to the mesh's frame for the chosen builder, and we apply
it to every pre-crack segment endpoint before emitting the script.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .config import MODULAR
from .knowledge import Variant
from .mesh import MeshSizePlan
from .schema import Action, CanonicalSpec, Constitutive


# ---------------------------------------------------------------------------
# Variant → { builder, post helper, solver-argument signature }
# ---------------------------------------------------------------------------
VARIANT_META: Dict[str, Dict[str, str]] = {
    "linear_elastic_2d_pe": {
        "builder": "make_linear_elastic_2d_pe_builder",
        "reaction_helper": "reaction_form_from_sigma_2d",
        "solver_call": "run_quasistatic",
    },
    "linear_elastic_2d_ps": {
        "builder": "make_linear_elastic_2d_ps_builder",
        "reaction_helper": "reaction_form_from_sigma_2d",
        "solver_call": "run_quasistatic",
    },
    "linear_elastic_3d": {
        "builder": "make_linear_elastic_3d_builder",
        "reaction_helper": "reaction_form_from_sigma_3d",
        "solver_call": "run_quasistatic",
    },
    "dynamic_2d": {
        "builder": "make_dynamic_2d_builder",
        "reaction_helper": "reaction_form_from_sigma_2d",
        "solver_call": "run_dynamic",
    },
    "ductile_2d_pe": {
        "builder": "make_ductile_2d_pe_builder",
        "reaction_helper": "reaction_form_from_sigma_2d",
        "solver_call": "run_ductile",
    },
    "ductile_3d": {
        "builder": "make_ductile_3d_builder",
        "reaction_helper": "reaction_form_from_sigma_3d",
        "solver_call": "run_ductile",
    },
    "finite_elastic_2d_pe": {
        "builder": "make_finite_elastic_2d_pe_builder",
        "reaction_helper": "reaction_form_from_sigma_2d",
        "solver_call": "run_finite_elasticity",
    },
    "finite_elastic_2d_ps": {
        "builder": "make_finite_elastic_2d_ps_builder",
        "reaction_helper": "reaction_form_from_sigma_2d",
        "solver_call": "run_finite_elasticity",
    },
    "finite_elastic_3d": {
        "builder": "make_finite_elastic_3d_builder",
        "reaction_helper": "reaction_form_from_sigma_3d",
        "solver_call": "run_finite_elasticity",
    },
}


# ---------------------------------------------------------------------------
# User-frame -> mesh-frame coordinate transform.
# ---------------------------------------------------------------------------
def _user_to_mesh_xy(action: Action,
                      spec: CanonicalSpec) -> Callable[[float, float], Tuple[float, float]]:
    """Return ``(x, y) -> (x', y')`` mapping the architect's user-frame
    coordinate to the mesh builder's internal frame.

    Most builders use the same frame as the user (corners at (0, 0) and
    (W, L)) so the transform is identity.  ``make_notched_plate_2d`` and
    its 3D sibling centre the y axis at zero — we shift y by ``-L/2``.
    ``make_slant_plate_2d`` does the same with H.
    """
    builder = action.mesh_builder
    dims = spec.geometry.dimensions or {}
    if builder in ("make_notched_plate_2d", "make_notched_plate_3d"):
        L = float(dims.get("L", 1.0))
        return lambda x, y, _L=L: (float(x), float(y) - _L / 2.0)
    if builder == "make_slant_plate_2d":
        H = float(dims.get("H") or dims.get("L") or 1.0)
        return lambda x, y, _H=H: (float(x), float(y) - _H / 2.0)
    return lambda x, y: (float(x), float(y))


# ---------------------------------------------------------------------------
# Inference: derive the notched-plate notch length ``ac`` from the
# architect's pre_crack_segments when the user didn't spell it out.
# ---------------------------------------------------------------------------
def _infer_notched_plate_ac(spec: CanonicalSpec) -> float | None:
    """For an edge-cracked plate the notch length ``ac`` is the x-coordinate
    of the crack tip when the crack enters from the left face at mid-height
    (the notched-plate convention).  We accept any segment whose two
    endpoints share the same y (within a small tolerance) AND have one
    endpoint at x ≈ 0 — the other endpoint's x is then the notch length.

    Returns the maximum such length, or ``None`` if no segment looks like
    a left-edge horizontal crack.
    """
    segs = spec.geometry.pre_crack_segments or []
    if not segs:
        return None
    dims = spec.geometry.dimensions or {}
    W = float(dims.get("W", 0.0)) or 1.0
    L = float(dims.get("L", 0.0)) or 1.0
    tol = 1e-3 * max(W, L)
    best: float | None = None
    for s in segs:
        if len(s) != 2 or len(s[0]) < 2 or len(s[1]) < 2:
            continue
        (x1, y1), (x2, y2) = s[0], s[1]
        if abs(float(y1) - float(y2)) > tol:
            continue  # not horizontal
        # Pick the endpoint at x ≈ 0; the OTHER endpoint's x is the length.
        if abs(float(x1)) <= tol:
            length = abs(float(x2))
        elif abs(float(x2)) <= tol:
            length = abs(float(x1))
        else:
            continue  # neither endpoint at the left face
        if length > 0 and (best is None or length > best):
            best = length
    return best


# ---------------------------------------------------------------------------
# Mesh-builder call snippets — one per geometry kind.
# ---------------------------------------------------------------------------
def _mesh_call(spec: CanonicalSpec, action: Action, h0: float) -> str:
    """Return a multi-line snippet **without leading indent** — the caller
    re-indents it to the body's nesting level."""
    g = spec.geometry
    if action.mesh_builder == "make_notched_plate_2d":
        d = {"W": 1.0, "L": 1.0, "ac": 0.5, "cw": 0.001}
        d.update({k: v for k, v in g.dimensions.items() if k in d})
        # If the architect provided pre_crack_segments but no explicit ``ac``,
        # infer it from the longest left-edge horizontal segment.
        if "ac" not in g.dimensions:
            inferred = _infer_notched_plate_ac(spec)
            if inferred is not None:
                d["ac"] = inferred
        return (f"from modular.meshes import make_notched_plate_2d\n"
                f"msh_coarse, markers, geom = make_notched_plate_2d(\n"
                f"    W={d['W']}, L={d['L']}, ac={d['ac']}, cw={d['cw']},\n"
                f"    h0={h0}, comm=comm)")
    if action.mesh_builder == "make_notched_plate_3d":
        d = {"W": 1.0, "L": 1.0, "H": 0.1, "ac": 0.5, "cw": 0.001}
        d.update({k: v for k, v in g.dimensions.items() if k in d})
        if "ac" not in g.dimensions:
            inferred = _infer_notched_plate_ac(spec)
            if inferred is not None:
                d["ac"] = inferred
        return (f"from modular.meshes import make_notched_plate_3d\n"
                f"msh_coarse, markers, geom = make_notched_plate_3d(\n"
                f"    W={d['W']}, L={d['L']}, H={d['H']}, ac={d['ac']}, cw={d['cw']},\n"
                f"    h0={h0}, comm=comm)")
    if action.mesh_builder == "make_slant_plate_2d":
        d = {"W": 1.0, "L": 1.0, "slant_angle_deg": 30.0}
        d.update({k: v for k, v in g.dimensions.items() if k in d})
        return (f"from modular.meshes import make_slant_plate_2d\n"
                f"msh_coarse, markers, geom = make_slant_plate_2d(\n"
                f"    W={d['W']}, L={d['L']}, slant_angle_deg={d['slant_angle_deg']},\n"
                f"    h0={h0}, comm=comm)")
    if action.mesh_builder == "make_dogbone_2d":
        d = {"L_total": 4.0, "W_grip": 1.0, "W_gauge": 0.5, "L_gauge": 1.0}
        d.update({k: v for k, v in g.dimensions.items() if k in d})
        return (f"from modular.meshes import make_dogbone_2d\n"
                f"msh_coarse, markers, geom = make_dogbone_2d(\n"
                f"    L_total={d['L_total']}, W_grip={d['W_grip']},\n"
                f"    W_gauge={d['W_gauge']}, L_gauge={d['L_gauge']},\n"
                f"    h0={h0}, comm=comm)")
    if action.mesh_builder == "make_dogbone_3d":
        d = {"L_total": 4.0, "W_grip": 1.0, "W_gauge": 0.5, "L_gauge": 1.0,
             "thickness": 0.2}
        d.update({k: v for k, v in g.dimensions.items() if k in d})
        return (f"from modular.meshes import make_dogbone_3d\n"
                f"msh_coarse, markers, geom = make_dogbone_3d(\n"
                f"    L_total={d['L_total']}, W_grip={d['W_grip']},\n"
                f"    W_gauge={d['W_gauge']}, L_gauge={d['L_gauge']},\n"
                f"    thickness={d['thickness']},\n"
                f"    h0={h0}, comm=comm)")
    # Custom path — user-provided script in session dir.
    return (f"from custom_mesh import make_custom_gmsh  # noqa: E402\n"
            f"msh_coarse, markers, geom = make_custom_gmsh(h0={h0}, comm=comm)")


def _indent(block: str, spaces: int = 4) -> str:
    """Prefix every line of ``block`` with ``spaces`` spaces."""
    pad = " " * spaces
    return "\n".join(pad + ln if ln else ln for ln in block.splitlines())


# ---------------------------------------------------------------------------
# BC spec.
#
# The modular builders already implement the default "bottom u_y=0, top u_y=
# disp_const, top-right u_x=0 pin" BC pattern for the notched plate.  We use
# that default whenever the user's spec matches that canonical layout,
# avoiding the forward-reference pitfall with `disp_const`.  For every other
# case we emit an explicit bc_spec_u that fixes user-specified components
# and ramps each LoadingCase.magnitude over the global time variable.
# ---------------------------------------------------------------------------
def _is_canonical_single_pull(spec: CanonicalSpec) -> bool:
    """Return True iff spec wants the default 'bottom fixed, top pulled' BCs
    that the modular builder already implements — lets us keep the historical
    behaviour verbatim for the simplest cases."""
    if spec.geometry.kind not in ("notched_plate", "plate_3d", "slant_plate"):
        return False
    if not spec.bcs.fixed or not spec.bcs.loading:
        return False
    if len(spec.bcs.fixed) != 1 or len(spec.bcs.loading) != 1:
        return False
    f = spec.bcs.fixed[0]
    l = spec.bcs.loading[0]
    y_comp = 1
    return (f.region == "bottom" and (not f.components or y_comp in f.components)
            and l.region == "top" and (l.component in (None, y_comp))
            and l.control.value == "displacement")


def _bc_block(spec: CanonicalSpec) -> str:
    if _is_canonical_single_pull(spec):
        return "bc_spec_u = None  # modular builder defaults match the spec"

    # AMR-safe pattern: every ramped BC value is a STRING MARKER ("@ramp:N").
    # The modular builder sees the marker, creates a fresh fem.Constant on
    # its own mesh, and exposes the live Constant via P["bc_value_constants"].
    # When AMR rebuilds, the new build creates fresh Constants on the new
    # mesh — the proxy below re-binds via the after_rebuild hook so the BC
    # never goes stale.  (Earlier templates used external fem.Constants on
    # msh_coarse here, which became orphaned after AMR refinement.)
    lines = [
        "# Build name -> integer tag lookup from markers_spec",
        "# (modular.common.bcs.build_dirichlet_bcs takes integer tags).",
        "_name_to_tag = {name: tag for tag, name, _ in markers}",
        "def _t(r):",
        "    if r not in _name_to_tag:",
        "        raise KeyError(",
        "            f\"Region {r!r} missing from markers_spec — \"",
        "            f\"available: {sorted(_name_to_tag)}\")",
        "    return _name_to_tag[r]",
        "bc_spec_u = []",
    ]
    dim_is_3d = spec.dimension.value == "3D"

    # Fixed regions.
    for i, f in enumerate(spec.bcs.fixed):
        comps = f.components or ([0, 1, 2] if dim_is_3d else [0, 1])
        for c in comps:
            lines.append(f"bc_spec_u.append((_t({f.region!r}), {c}, 0.0))")

    # Loading cases — only displacement control goes into Dirichlet bc_spec_u.
    any_disp = False
    for i, lc in enumerate(spec.bcs.loading):
        if lc.control.value != "displacement":
            continue
        any_disp = True
        comp = lc.component if lc.component is not None else 1
        mag = lc.magnitude
        # String marker; modular builder materialises a fresh fem.Constant
        # on the build mesh and stores it in P["bc_value_constants"].
        lines.append(
            f"bc_spec_u.append((_t({lc.region!r}), {comp}, '@ramp:{i}'))"
            f"  # will ramp 0 -> {mag}")
    if not any_disp:
        lines.append("# (no displacement BC in loading cases; builder uses its default)")
    return "\n".join(lines)


def _ramp_update_hook(spec: CanonicalSpec) -> str:
    """Produce a snippet plugged into ``on_output`` that sync-updates each
    ramp Constant from the global pseudo-time t (0 → 1 → magnitude)."""
    lines = []
    for i, lc in enumerate(spec.bcs.loading):
        if lc.control.value != "displacement":
            continue
        lines.append(f"_ramps[{i}].value = t * {lc.magnitude}")
    if not lines:
        return ""
    return "# sync ramps with the solver's current pseudo-time t:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level script render.
# ---------------------------------------------------------------------------
def render_script(spec: CanonicalSpec,
                  action: Action,
                  mesh_plan: MeshSizePlan,
                  session_dir: Path,
                  variant: Variant) -> Path:
    """Produce the runnable .py file; returns its absolute path."""
    meta = VARIANT_META[action.variant]

    # ---- material load ---- #
    material_block = _material_block(spec, session_dir)

    # ---- BC block ---- #
    bc_block = _bc_block(spec)
    is_canonical = _is_canonical_single_pull(spec)

    # ---- output + solver kwargs ---- #
    out_tag = spec.geometry.kind
    run_name = f"paraview_{out_tag}_{action.variant}"
    log_name = f"output_{out_tag}_{action.variant}.txt"

    # Set max_disp = max |magnitude| across displacement loading cases so the
    # solver's log column (= t*max_disp) reports the *physical* peak
    # displacement actually applied.  For multi-ramp cases the RampProxy
    # scales each ramp as (magnitude / max_disp) * (t * max_disp) =
    # magnitude * t, which is what we want.
    disp_mags = [abs(l.magnitude) for l in spec.bcs.loading
                 if l.control.value == "displacement" and l.magnitude is not None]
    if is_canonical and spec.bcs.loading:
        max_disp = spec.bcs.loading[0].magnitude
    elif disp_mags:
        max_disp = max(disp_mags)
    else:
        max_disp = 0.006

    # Pick the loaded region / component for the reaction-force helper.
    # Heuristic: the user usually wants the reaction on the **largest
    # incremental load** — for prompts that combine a small held pre-load
    # with a bigger ramped load (e.g. "left edge held at 0.005 mm pre-shear,
    # top edge then pulled by 0.02 mm"), measuring the reaction on the held
    # edge is rarely interesting.  Pick the displacement loading case with
    # the largest |magnitude|; ties break to the latest-listed (closer to
    # the prompt's "then" phase).
    if spec.bcs.loading:
        disp_loads = [l for l in spec.bcs.loading
                      if l.control.value == "displacement"
                      and l.magnitude is not None]
        candidates = disp_loads or list(spec.bcs.loading)
        reaction_lc = max(
            ((i, l) for i, l in enumerate(candidates)),
            key=lambda pair: (abs(pair[1].magnitude or 0.0), pair[0]),
        )[1]
        reaction_region = reaction_lc.region
        reaction_comp = (reaction_lc.component
                         if reaction_lc.component is not None else 1)
    else:
        reaction_region = "top"
        reaction_comp = 1

    # ---- Dynamic / Finite-elasticity solver argument differences ---- #
    if meta["solver_call"] == "run_quasistatic":
        solver_call = (f"run_quasistatic(\n"
                       f"    P, build_problem, msh_coarse,\n"
                       f"    T_total={spec.loading.T_total}, steps={spec.loading.steps},\n"
                       f"    max_stag={spec.max_stag}, tol_stag={spec.tol_stag},\n"
                       f"    max_disp={max_disp},\n"
                       f"    on_output=on_out,\n"
                       f"    reaction_form={meta['reaction_helper']}"
                       f"({reaction_region!r}, component={reaction_comp}),\n"
                       f"    log_path={log_name!r},\n"
                       f")")
    elif meta["solver_call"] == "run_ductile":
        solver_call = (f"run_ductile(\n"
                       f"    P, build_problem, msh_coarse,\n"
                       f"    T_total={spec.loading.T_total}, steps={spec.loading.steps},\n"
                       f"    max_stag={spec.max_stag}, max_disp={max_disp},\n"
                       f"    on_output=on_out,\n"
                       f"    reaction_form={meta['reaction_helper']}"
                       f"({reaction_region!r}, component={reaction_comp}),\n"
                       f"    log_path={log_name!r},\n"
                       f")")
    elif meta["solver_call"] == "run_finite_elasticity":
        solver_call = (f"run_finite_elasticity(\n"
                       f"    P, build_problem, msh_coarse,\n"
                       f"    T_total={spec.loading.T_total}, steps={spec.loading.steps},\n"
                       f"    max_disp={max_disp},\n"
                       f"    on_output=on_out,\n"
                       f"    reaction_form={meta['reaction_helper']}"
                       f"({reaction_region!r}, component={reaction_comp}),\n"
                       f"    log_path={log_name!r},\n"
                       f")")
    else:  # run_dynamic
        solver_call = (f"run_dynamic(\n"
                       f"    P, build_problem, msh_coarse,\n"
                       f"    T_total={spec.loading.T_total},\n"
                       f"    on_output=on_out,\n"
                       f"    log_path={log_name!r},\n"
                       f")")

    # ---- Fracture toggle: inflate Gc so z stays ≈ 1 ---- #
    fracture_patch = ""
    if not spec.fracture_enabled:
        fracture_patch = ("# ---- Fracture disabled: inflate Gc to effectively "
                          "suppress the phase field. ----\n"
                          "mat['Gc'] = mat.get('Gc', 1.0) * 1.0e8\n"
                          "# Re-derive length scales with the inflated Gc so h0 "
                          "grows accordingly, then shrink back so the mesh stays\n"
                          "# reasonable and z remains pinned by the huge surface "
                          "energy.\n"
                          "from modular.materials.loader import derive_length_scales\n"
                          "mat.update(derive_length_scales(mat))\n")

    # ---- Damaged-notch initialisation ---- #
    # If the Architect declared pre-crack segments, emit numpy code that
    # initialises P["z"] (and P["z_lb"] for irreversibility) to z=0 in a
    # thin band along each segment.  Standard smooth profile:
    #     z(x) = 1 - exp(-(d/eps)^2)
    # where d is the signed distance from node x to the nearest crack
    # segment.  This is the "damaged notch boundary condition" used in the
    # variational phase-field-fracture literature; without it, the solver
    # sees an intact specimen and won't propagate below the nucleation
    # threshold even with a geometric slit in the mesh.
    #
    # Suppress when the mesh ITSELF carries a geometric pre-crack — the
    # built-in plate / slant_plate builders always do, and the custom-mesh
    # LLM is instructed (mesh.py CUSTOM_MESH_SYSTEM) to split the affected
    # edge with a tiny mouth-opening cw whenever pre_crack_segments is in
    # the request.  In those cases the geometric gap is the boundary
    # condition; pinning z=0 along the same segment is redundant and over-
    # constrains DOFs that already sit on free crack faces.
    precrack_block = ""
    _MESH_HAS_GEOMETRIC_CRACK = {"notched_plate", "plate_3d",
                                  "slant_plate", "custom"}
    if (spec.geometry.pre_crack_segments
        and spec.geometry.kind in _MESH_HAS_GEOMETRIC_CRACK):
        precrack_block = (
            "# pre-crack already built into the mesh as a geometric gap; "
            "no z=0 VI pin needed.")
    if (spec.geometry.pre_crack_segments
        and spec.geometry.kind not in _MESH_HAS_GEOMETRIC_CRACK):
        # Architect emits pre-crack endpoints in the user's frame (corner at
        # (0,0)).  Some built-in mesh builders (notched_plate_2d/3d,
        # slant_plate_2d) centre the y-axis at zero; transform each endpoint
        # so the VI z=0 pin lands on the actual mesh nodes.
        _xform = _user_to_mesh_xy(action, spec)
        _segs = []
        for s in spec.geometry.pre_crack_segments:
            if len(s) == 2 and len(s[0]) >= 2 and len(s[1]) >= 2:
                p1 = _xform(s[0][0], s[0][1])
                p2 = _xform(s[1][0], s[1][1])
                _segs.append((p1, p2))
        segs_py = ", ".join(
            f"[({p1[0]!r}, {p1[1]!r}), ({p2[0]!r}, {p2[1]!r})]"
            for p1, p2 in _segs
        )
        if segs_py:
            precrack_block = f"""\
# ---- Damaged-notch initialisation (z=0 pinned along pre-crack) ---- #
# Smooth profile  z(x) = 1 - exp(-(d/eps)^2)  sets the initial state.
# The narrow band  d < 0.3*eps  is PINNED at z = 0 via the VI upper bound
# (P["z_ub"] = 0 there) so the phase field cannot heal — this is the
# standard "damaged notch BC" used in variational phase-field fracture.
import numpy as _np
_PRECRACK_SEGS = [{segs_py}]
_pts = P["Y"].tabulate_dof_coordinates()[:, :2]
_z_init = _np.ones(len(_pts))
_d_min = _np.full(len(_pts), _np.inf)
for (x1, y1), (x2, y2) in _PRECRACK_SEGS:
    _dx, _dy = x2 - x1, y2 - y1
    _L2 = _dx * _dx + _dy * _dy
    if _L2 <= 0.0:
        continue
    _t = ((_pts[:, 0] - x1) * _dx + (_pts[:, 1] - y1) * _dy) / _L2
    _t = _np.clip(_t, 0.0, 1.0)
    _cx = x1 + _t * _dx
    _cy = y1 + _t * _dy
    _d = _np.hypot(_pts[:, 0] - _cx, _pts[:, 1] - _cy)
    _d_min = _np.minimum(_d_min, _d)
    _z_init = _np.minimum(_z_init, 1.0 - _np.exp(-(_d / eps) ** 2))
_pin_mask = _d_min < 0.3 * eps        # nodes pinned at z = 0
P["z"].x.array[:] = _z_init
P["z"].x.scatter_forward()
P["z_lb"].x.array[:] = _z_init
P["z_lb"].x.scatter_forward()
# Upper bound: 0 on the pinned band, 1 elsewhere -> z can't heal inside
# the crack, can damage (z -> 0) anywhere.  Combined with z_lb = _z_init
# the VI sandwich gives us the correct damaged-notch BC.
_z_ub_arr = _np.ones(len(_pts))
_z_ub_arr[_pin_mask] = 0.0
P["z_ub"].x.array[:] = _z_ub_arr
P["z_ub"].x.scatter_forward()
print(f"[damaged-notch] {{len(_PRECRACK_SEGS)}} segment(s); "
      f"min z_init = {{_z_init.min():.3e}}; "
      f"{{int(_pin_mask.sum())}} / {{len(_pts)}} DOFs pinned at z = 0",
      flush=True)
"""

    # For non-canonical (multi-ramp) cases emit the RampProxy plumbing.
    # P["disp_const"].value is driven by the solver as t*max_disp.  Each
    # user ramp is piecewise:
    #   t < t_start            → 0
    #   t_start ≤ t ≤ t_end    → magnitude * (t - t_start) / (t_end - t_start)
    #   t > t_end              → magnitude (held)
    # Defaults (t_start=0, t_end=1) recover the linear ramp the old proxy
    # used.  Multi-stage prompts ("first held, then ramped") become two
    # cases with non-overlapping windows.
    #
    # AMR-safe design: the proxy reads/writes the *current* P (held in a
    # mutable list cell) and looks up the ramp Constants via
    # ``P['bc_value_constants']['@ramp:i']`` — those Constants are
    # materialised by the modular builder on the build mesh, so each AMR
    # rebuild creates fresh ones.  The ``after_rebuild`` hook below
    # re-installs the proxy on the new ``P_new['disp_const']`` and
    # re-applies the current load.  Without this, AMR refinements
    # silently freeze the BC at the value it had at refinement time.
    proxy_block = ""
    after_rebuild_block = ""
    after_rebuild_kwarg = ""
    if not is_canonical and spec.bcs.loading:
        proxy_lines = [
            "class _RampProxy:",
            "    \"\"\"AMR-safe ramp proxy.",
            "",
            "    Wraps P['disp_const'].  Solver drives .value = t*max_disp.",
            "    Each user ramp is piecewise: ramps from 0 to magnitude_i over",
            "    [t_start_i, t_end_i] (in pseudo-time t in [0, 1]); held outside.",
            "    On every set, walks P['bc_value_constants'] and updates the",
            "    fresh-on-this-mesh Constants the modular builder created from",
            "    the @ramp:i markers in bc_spec_u.",
            "    \"\"\"",
            "    def __init__(self, P_holder, schedule, max_disp):",
            "        self._holder = P_holder   # mutable: [P]; rebound after AMR",
            "        self._sched  = schedule   # [(marker_str, mag, t0, t1), ...]",
            "        self._max    = max_disp",
            "        self._raw    = 0.0        # last commanded raw value",
            "    @property",
            "    def value(self):",
            "        return self._raw",
            "    @value.setter",
            "    def value(self, v):",
            "        self._raw = v",
            "        t = (v / self._max) if self._max else 0.0",
            "        bcs = self._holder[0].get('bc_value_constants', {})",
            "        for marker, mag, t0, t1 in self._sched:",
            "            if t <= t0:    r = 0.0",
            "            elif t >= t1:  r = mag",
            "            else:          r = mag * (t - t0) / max(t1 - t0, 1e-12)",
            "            c = bcs.get(marker)",
            "            if c is not None:",
            "                c.value = r",
            "",
            "# schedule = (ramp marker, magnitude, t_start, t_end)",
            f"_MAX_DISP = {float(max_disp)!r}",
            "_schedule = [",
        ]
        for i, lc in enumerate(spec.bcs.loading):
            if lc.control.value == "displacement":
                mag = float(lc.magnitude)
                t0 = max(0.0, min(1.0, float(getattr(lc, "t_start", 0.0))))
                t1 = max(0.0, min(1.0, float(getattr(lc, "t_end", 1.0))))
                if t1 < t0:
                    t0, t1 = 0.0, 1.0   # malformed window — fall back to whole run
                proxy_lines.append(
                    f"    ('@ramp:{i}', {mag!r}, {t0!r}, {t1!r}),")
        proxy_lines.extend([
            "]",
            "_P_holder = [P]",
            "_proxy = _RampProxy(_P_holder, _schedule, _MAX_DISP)",
            "P['disp_const'] = _proxy",
        ])
        proxy_block = "\n".join(proxy_lines)

        # The after_rebuild hook fires inside try_amr after every mesh
        # refinement.  The new P_new has fresh bc_value_constants on the
        # new mesh; rebind the proxy to point at P_new and reinstall it
        # as P_new['disp_const'], then replay the current load so the
        # new fem.Constants get the right value.
        after_rebuild_block = (
            "def _after_amr_rebuild(P_new):\n"
            "    _P_holder[0] = P_new\n"
            "    P_new['disp_const'] = _proxy\n"
            "    # Re-apply the current commanded value so the new ramp\n"
            "    # Constants pick up the right ramp values immediately.\n"
            "    _proxy.value = _proxy.value")
        after_rebuild_kwarg = "    after_rebuild=_after_amr_rebuild,"

    # If we built an after_rebuild hook, inject ``after_rebuild=`` into the
    # solver call so AMR rebuilds reach back into the driver to re-install
    # the proxy.  Otherwise leave the solver call untouched.
    if after_rebuild_kwarg:
        # solver_call ends with "...,\n)"; insert the kwarg before the close.
        solver_call = solver_call[:-2] + after_rebuild_kwarg.lstrip() + "\n)"

    # Re-indent every injected multi-line block to 4-space body indent.
    mesh_block     = _indent(_mesh_call(spec, action, mesh_plan.h0), 4)
    bc_block_i     = _indent(bc_block, 4)
    solver_block_i = _indent(solver_call, 4)
    mat_block_i    = material_block  # already pre-indented (see _material_block)
    frac_block_i   = _indent(fracture_patch, 4) if fracture_patch else ""
    proxy_block_i  = _indent(proxy_block, 4) if proxy_block else ""
    after_rebuild_block_i = _indent(after_rebuild_block, 4) if after_rebuild_block else ""
    precrack_block_i = _indent(precrack_block, 4) if precrack_block else ""

    # ---- Full file ---- #
    body = f'''#!/usr/bin/env python3
"""Auto-generated phase-field fracture script.

Variant     : {action.variant}
Mesh        : {action.mesh_builder}
Solver      : {action.solver}
Fracture    : {"enabled" if spec.fracture_enabled else "DISABLED (Gc->inf)"}
Rationale   : {action.rationale}
"""
from __future__ import annotations
import sys, os
# Script lives at  agent_v2/agentic_simulations/<sid>/run_<variant>.py
# Walk up 2 levels to reach agent_v2/, which is the parent of the `modular`
# package imported below.  (An older path "..", "..", ".." landed at
# agent_full_2/, which silently picked up a stale duplicate of `modular/`
# living one level up — patches to agent_v2/modular/ never took effect.)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
# Also allow ``from custom_mesh import ...`` when the session has one.
sys.path.insert(0, _HERE)

from mpi4py import MPI
from modular.materials import load_material
from modular.common    import print_banner, print_mesh_info
from modular.problems  import {meta["builder"]}
from modular.post      import XDMFWriter, {meta["reaction_helper"]}
from modular.solvers   import {meta["solver_call"]}

def main():
    comm = MPI.COMM_WORLD

    # ---- Material ------------------------------------------------------ #
{mat_block_i}
{frac_block_i}
    # The orchestrator picked these length scales (potentially after one or
    # more mesh rescales following the user's (n/target)^(1/dim) rule).
    # Override what the material loader derived so eps, h0 and h_min stay
    # consistent with the mesh we are about to build.
    eps   = {mesh_plan.eps!r}
    h0    = {mesh_plan.h0!r}
    h_min = {mesh_plan.h_min!r}
    mat["eps"], mat["h0"], mat["h_min"] = eps, h0, h_min
    print_banner(f"{action.variant} {{mat['name']}} eps={{eps:.3e}} h0={{h0:.3e}}", comm)

    # ---- Mesh ---------------------------------------------------------- #
{mesh_block}
    print_mesh_info(msh_coarse, comm)
    n_cells_local  = msh_coarse.topology.index_map(msh_coarse.topology.dim).size_local
    n_cells_global = comm.allreduce(n_cells_local, op=MPI.SUM)
    print(f"[mesh-audit] n_cells_global = {{n_cells_global}}", flush=True)
    # Probe-only mode: the orchestrator calls us with --mesh-only first to
    # decide whether to rescale eps BEFORE the expensive solve.  Exit
    # immediately after the mesh-audit print.
    if "--mesh-only" in sys.argv:
        if comm.rank == 0:
            print("[mesh-only] exiting cleanly.", flush=True)
        sys.exit(0)

    # ---- Problem builder ---------------------------------------------- #
{bc_block_i}

    build_problem = {meta["builder"]}(
        mat=mat, markers_spec=markers, geom=geom,
        eps=eps, h0=h0, h_min=h_min,
        bc_spec_u=bc_spec_u,
    )
    P = build_problem(msh_coarse)

{precrack_block_i}
{proxy_block_i}
{after_rebuild_block_i}
    disp_const = P["disp_const"]

    # ---- Output ------------------------------------------------------- #
    writer = XDMFWriter({run_name!r})
    def on_out(P, t, step, dt, extras):
        writer.write(P, t, step)

    # ---- Solve -------------------------------------------------------- #
{solver_block_i}

if __name__ == "__main__":
    main()
'''
    script_path = session_dir / f"run_{action.variant}.py"
    script_path.write_text(body, encoding="utf-8")
    return script_path


# ---------------------------------------------------------------------------
# Material resolution — write an ad-hoc entry into the session dir when the
# user gave raw properties instead of a catalog name.
# ---------------------------------------------------------------------------
def _material_block(spec: CanonicalSpec, session_dir: Path) -> str:
    m = spec.material
    if m.catalog_name:
        return f"    mat = load_material({m.catalog_name!r})"

    # Build a one-entry JSON DB in the session dir and point load_material at it.
    entry = {"problem_type": m.problem_type or _guess_problem_type(spec),
             "units": m.units}
    # Fill non-None fields only.
    extras = {k: v for k, v in m.model_dump().items()
              if v is not None and k not in ("catalog_name", "display_name",
                                             "problem_type", "units")}
    entry.update(extras)
    name = (m.display_name or "UserMaterial").replace(" ", "_")
    db_path = session_dir / "material.json"
    db_path.write_text(json.dumps({name: entry}, indent=2))
    return (f"    mat = load_material({name!r}, "
            f"db_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"
            f" 'material.json'))")


def _guess_problem_type(spec: CanonicalSpec) -> str:
    if spec.constitutive == Constitutive.j2_plasticity:
        return "ductile"
    if spec.constitutive == Constitutive.lopez_pamies:
        return "finite_elasticity"
    if spec.loading.mode.value == "dynamic":
        return "dynamic_linear_elasticity"
    return "linear_elasticity"
