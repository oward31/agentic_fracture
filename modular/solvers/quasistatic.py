"""Adaptive-dt staggered solver for linear-elastic phase-field fracture.

Spec (user-mandated):
  • T_total = 1.0, Totalsteps = 200  → dt_first = 1/200
  • tol_stag = 1e-7, max_stag = 20
  • If z_residual at end of step > 10·tol_stag → halve dt and revert the
    step. Give up (accept) once dt < dt_first/10.

AMR (matches fea_codes/bench_amr.py exactly):
  • Mid-stagger AMR: try_amr fires between every u-solve and z-solve so
    the z-solve always runs on a mesh that resolves the current active
    zone.  No end-of-step AMR.
  • Irreversibility threshold v_th = 0.05: only nodes with z ≤ v_th are
    pinned via the SNES-VI upper bound; intact / lightly-damaged regions
    are left free.
"""

from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple

import sys
import time
import numpy as np
from dolfinx import fem
from mpi4py import MPI

from ..common.amr import try_amr
from ..common.io_utils import open_log, write_log_line
from ..common.norms import norm_L2_cached
from ..common.snes import is_diverged, reason_label
from ..common.transfer import transfer_function


# Irreversibility threshold (matches fea_codes/bench_amr.py).  Only nodes
# with z ≤ V_TH are pinned via the SNES-VI upper bound; everywhere else
# z is left free to relax up to 1.
V_TH = 0.05


def _stagger_step(P: Dict, comm, msh,
                  *, tol_stag: float, max_stag: int,
                  msh_coarse=None,
                  build_problem: Optional[Callable] = None,
                  eps: Optional[float] = None,
                  h0: Optional[float] = None,
                  h_min: Optional[float] = None,
                  t_disp: Optional[float] = None,
                  v_th: float = V_TH,
                  amr_label: str = "AMR",
                  after_rebuild: Optional[Callable] = None,
                  ):
    """Run staggered u-z iterations with mid-stagger AMR.

    Returns ``(P, msh, u_res, z_res, stag, divergence_info)`` where
    ``divergence_info`` is ``None`` on clean runs, or a tuple
    ``(which: 'u'|'z', reason_label_str, iteration_count)`` if either
    SNES inner solve reported a hard divergence (so the caller can halve
    Δt and revert).

    Mid-stagger AMR (matches ``fea_codes/bench_amr.py``) fires between
    every u-solve and z-solve when the AMR plumbing args are provided
    (``msh_coarse``, ``build_problem``, ``eps``, ``h0``, ``h_min``).  When
    any of those is ``None``, AMR is skipped and behaviour reduces to the
    pre-AMR pure stagger (used by callers that handle AMR externally).

    Capturing ``(iters, reason)`` from each SNES solve surfaces silent
    solver no-ops (e.g. SNES VI converged in 0 iters because the active
    set blocked the descent direction at z=z_ub) so the outer driver can
    halve Δt instead of accepting a no-motion step.
    """
    do_amr = (msh_coarse is not None and build_problem is not None
              and eps is not None and h0 is not None and h_min is not None)
    stag = 0
    u_res = z_res = 1e9
    divergence: Optional[Tuple[str, str, int]] = None
    while (stag < max_stag) and (u_res > tol_stag or z_res > tol_stag):
        stag += 1

        # Save previous u iterate (snapshot BEFORE u-solve).
        P["u_trial"].x.array[:] = P["u"].x.array
        P["u_trial"].x.scatter_forward()
        u_iters, u_reason = P["problem_u"].solve()
        if is_diverged(u_reason):
            divergence = ("u", reason_label(u_reason), int(u_iters))
            break

        # ===== Mid-stagger AMR (matches fea_codes/bench_amr.py) =====
        # Refine the mesh between u-solve and z-solve so the z-solve runs
        # on a mesh that resolves the current active zone.  Without this
        # the z field can evolve in unrefined regions during a step and
        # only get refined the NEXT step — the "phase field evolves
        # outside refined regions" symptom the agent surfaced.
        if do_amr:
            msh, P, did_refine = try_amr(
                msh, msh_coarse, P,
                eps=eps, h0=h0, h_min=h_min,
                build_problem=build_problem,
                # Transfer u_trial too so the post-z u_diff is computed
                # on the new mesh (otherwise u_trial would be a fresh
                # zero on P_new["V"] and u_diff would be artificially
                # huge, forcing extra stagger iters).
                extra_V_fields=("u_trial",),
                label=f"{amr_label}-stag{stag}",
                after_rebuild=after_rebuild,
            )
            if did_refine:
                # Re-apply current displacement BC value on the new mesh.
                # Both the canonical fem.Constant path and the multi-stage
                # RampProxy path expose ``.value`` setters.
                if t_disp is not None and "disp_const" in P:
                    try:
                        P["disp_const"].value = t_disp
                    except (AttributeError, TypeError):
                        pass
                # Re-apply v_th irreversibility upper bound on the new
                # mesh.  AMR's field transfer interpolated z and z_ub but
                # didn't re-derive z_ub from the threshold rule.
                mask = P["z"].x.array <= v_th
                P["z_ub"].x.array[mask] = P["z"].x.array[mask]
                P["z_ub"].x.scatter_forward()
                # Re-set SNES VI bounds (mirrors fea_codes/bench_amr.py;
                # the SNES wrapper holds the petsc_vec by reference so
                # array updates already propagate, but this is harmless
                # belt-and-braces).
                P["problem_z"].solver.setVariableBounds(
                    P["z_lb"].x.petsc_vec, P["z_ub"].x.petsc_vec)

        # Save z_trial AFTER mid-stagger AMR so it lives on the (possibly
        # new) mesh; z_diff below is then computed on a single mesh.
        P["z_trial"].x.array[:] = P["z"].x.array
        P["z_trial"].x.scatter_forward()
        z_iters, z_reason = P["problem_z"].solve()
        if is_diverged(z_reason):
            divergence = ("z", reason_label(z_reason), int(z_iters))
            break

        # Residuals = L2 norms of differences between consecutive iterates.
        P["u_diff"].x.array[:] = P["u"].x.array - P["u_trial"].x.array
        P["u_diff"].x.scatter_forward()
        P["z_diff"].x.array[:] = P["z"].x.array - P["z_trial"].x.array
        P["z_diff"].x.scatter_forward()
        u_res = norm_L2_cached(comm, P["u_diff_norm_form"])
        z_res = norm_L2_cached(comm, P["z_diff_norm_form"])

    return P, msh, u_res, z_res, stag, divergence


def run_quasistatic(
    P: Dict,
    build_problem: Callable,
    msh_coarse,
    *,
    T_total: float = 1.0,
    steps: int = 200,
    max_stag: int = 20,
    tol_stag: float = 1.0e-7,
    max_disp: float,
    on_output: Callable = None,
    log_path: str = "output.txt",
    reaction_form: Callable | None = None,
    amr_label: str = "AMR",
    after_rebuild: Callable | None = None,
):
    """Adaptive-dt staggered driver.

    `on_output(P, t, step, stepsize, extras)` — called after each accepted step;
    typical use is to write an XDMF frame.
    `reaction_form(P)` — returns a `fem.form` that, when assembled, gives the
    reaction force on the displacement face.
    `after_rebuild(P_new)` — called by AMR after every refinement, with the
    fresh problem dict on the new mesh.  Drivers that wrap ``P["disp_const"]``
    in a ramp proxy MUST use this hook to re-install the proxy on the new
    ``disp_const`` (and re-bind any ramp Constants that live in
    ``P["bc_value_constants"]``); otherwise the BCs stop tracking after the
    first refinement.  Optional — the default canonical-pull path doesn't
    need it because the modular builder's internal ``dc`` is already
    consumed by the default BCs.
    """
    msh   = P["msh"]
    comm  = msh.comm
    rank  = comm.rank
    eps   = P["eps"]; h0 = P["h0"]; h_min = P["h_min"]

    dt_first = T_total / steps
    dt_floor = dt_first / 10.0
    stepsize = dt_first
    z_trip   = 10.0 * tol_stag

    open_log(log_path,
             "step  time  dt  stag_iters  u_res  z_res  min_z  disp  Fy",
             comm)

    # Snapshots for adaptive-dt revert.  Held in a mutable dict so the
    # mid-stagger AMR after_rebuild hook can re-bind them onto the new
    # mesh — without that, a Δt revert after AMR would crash with a size
    # mismatch.
    saved_state: Dict[str, Optional[fem.Function]] = {
        "u": None, "z": None, "zlb": None}

    def _refresh_saved_state(P_new):
        """Interpolate the revert snapshots onto the new mesh.  Called
        from inside ``try_amr`` (via the after_rebuild hook) when AMR
        fires mid-stagger."""
        for key, V_key in (("u", "V"), ("z", "Y"), ("zlb", "Y")):
            if saved_state[key] is not None:
                saved_state[key] = transfer_function(saved_state[key],
                                                      P_new[V_key])

    def _chained_after_rebuild(P_new, _user=after_rebuild):
        # Snapshot interpolation first (cheap), then user's hook (typically
        # the RampProxy re-installer from fracture_agent's templates.py, which
        # re-binds bc_value_constants on the new mesh).
        _refresh_saved_state(P_new)
        if _user is not None:
            _user(P_new)

    t, step = stepsize, 1

    while t - 1e-15 < T_total and step <= steps + 200:   # +200 safety margin for cuts
        # Snapshot the current converged state (start-of-step values) for
        # possible Δt-cut revert.  Allocate fresh each step because AMR
        # can replace P with a larger-dimensional problem; the snapshots
        # are then re-interpolated by ``_refresh_saved_state`` if AMR
        # fires inside ``_stagger_step``.
        saved_state["u"]   = fem.Function(P["V"])
        saved_state["z"]   = fem.Function(P["Y"])
        saved_state["zlb"] = fem.Function(P["Y"])
        saved_state["u"].x.array[:]   = P["u"].x.array;    saved_state["u"].x.scatter_forward()
        saved_state["z"].x.array[:]   = P["z"].x.array;    saved_state["z"].x.scatter_forward()
        saved_state["zlb"].x.array[:] = P["z_lb"].x.array; saved_state["zlb"].x.scatter_forward()

        # Irreversibility: SNES-VI upper bound tracks the current z ONLY
        # for nodes already past the v_th=0.05 threshold (matches
        # ``fea_codes/bench_amr.py``).  Intact / lightly-damaged regions
        # are left free to relax up to 1.
        mask = P["z"].x.array <= V_TH
        P["z_ub"].x.array[mask] = P["z"].x.array[mask]
        P["z_ub"].x.scatter_forward()

        # Update applied displacement.
        P["disp_const"].value = t * max_disp

        if rank == 0:
            print(f"\n┌── Step {step:<5d}  │  t = {t:.6e}  │  Δt = {stepsize:.2e}"
                  f"  │  {msh.topology.index_map(msh.topology.dim).size_local:>7,} cells")

        # Stagger with mid-stagger AMR (matches fea_codes/bench_amr.py).
        P, msh, u_res, z_res, stag, divergence = _stagger_step(
            P, comm, msh,
            tol_stag=tol_stag, max_stag=max_stag,
            msh_coarse=msh_coarse, build_problem=build_problem,
            eps=eps, h0=h0, h_min=h_min,
            t_disp=t * max_disp, v_th=V_TH,
            amr_label=amr_label,
            after_rebuild=_chained_after_rebuild,
        )

        # Adaptive dt: SNES divergence (any inner solve) → halve and revert.
        # This is a STRICTLY-HARDER condition than the z_res>z_trip cutoff
        # below: a silent SNES no-op (0 iterations because active set blocked
        # the descent direction) used to slip through with z_res=0 and the
        # step would be accepted with no field motion at all.  We now react
        # to the explicit converged-reason signal that PETSc provides.
        if divergence is not None:
            which, reason, iters = divergence
            if stepsize > dt_floor:
                if rank == 0:
                    print(f"    [solver-status] {which}-solve diverged "
                          f"({reason}, {iters} iters) — halving Δt and reverting")
                P["u"].x.array[:]    = saved_state["u"].x.array;   P["u"].x.scatter_forward()
                P["z"].x.array[:]    = saved_state["z"].x.array;   P["z"].x.scatter_forward()
                P["z_lb"].x.array[:] = saved_state["zlb"].x.array; P["z_lb"].x.scatter_forward()
                stepsize = max(stepsize / 2.0, dt_floor)
                continue   # retry same global time with smaller step
            # At dt_floor → log + accept whatever state we have so the run
            # produces a record (the agent's HealthReport will catch this
            # via the new dt_floor / no_z_evolution flags).
            if rank == 0:
                print(f"    [solver-status] {which}-solve diverged "
                      f"({reason}) at Δt floor — accepting; check log")

        # Adaptive dt: if z_res didn't reach 10·tol, cut and revert.
        elif z_res > z_trip and stepsize > dt_floor:
            if rank == 0:
                print(f"    ⚠  z_res = {z_res:.3e} > {z_trip:.1e} — halving Δt and reverting")
            P["u"].x.array[:]    = saved_state["u"].x.array;   P["u"].x.scatter_forward()
            P["z"].x.array[:]    = saved_state["z"].x.array;   P["z"].x.scatter_forward()
            P["z_lb"].x.array[:] = saved_state["zlb"].x.array; P["z_lb"].x.scatter_forward()
            stepsize = max(stepsize / 2.0, dt_floor)
            continue   # retry same global time with smaller step

        # Step accepted.
        min_z = comm.allreduce(P["z"].x.array.min(), op=MPI.MIN)
        Fy = 0.0
        if reaction_form is not None:
            Fy = comm.allreduce(fem.assemble_scalar(reaction_form(P)), op=MPI.SUM)

        if rank == 0:
            print(f"    ✓  u_res = {u_res:.3e}   z_res = {z_res:.3e}   "
                  f"min(z) = {min_z:.3f}   stag = {stag}")
        write_log_line(log_path,
                       f"{step} {t:.6e} {stepsize:.6e} {stag} {u_res:.6e} "
                       f"{z_res:.6e} {min_z:.6f} {t * max_disp:.6e} {Fy:.6e}",
                       comm)

        if on_output is not None:
            on_output(P, t, step, stepsize,
                      {"min_z": min_z, "Fy": Fy, "u_res": u_res, "z_res": z_res})

        # NB: end-of-step AMR removed — AMR now fires mid-stagger inside
        # ``_stagger_step`` (matches fea_codes/bench_amr.py).

        step += 1
        t = min(t + stepsize, T_total + 1e-14)
        if t >= T_total - 1e-14:
            break

    return P
