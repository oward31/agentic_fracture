"""HHT-α dynamic solver.

Time step comes from the characteristic length and Rayleigh wave speed
    dt = 0.1 · L_char / c_R
— applied at the material/geometry level before the time loop starts.

AMR (matches fea_codes/bench_amr.py):
  • Mid-stagger AMR fires between every u-solve and z-solve.  No
    end-of-step AMR.
  • Irreversibility threshold v_th = 0.05: only nodes with z ≤ v_th are
    pinned via the SNES-VI upper bound.

Note on dt: the dynamic driver uses fixed Δt (Rayleigh-wave-speed CFL).
There is no revert-on-divergence path here, so no snapshot bookkeeping
is needed across the AMR boundary — divergence is logged and we proceed.
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
from ..materials import rayleigh_speed


# Irreversibility threshold (matches fea_codes/bench_amr.py).
V_TH = 0.05


def dynamic_time_step(mat: Dict, L_char: float, safety: float = 0.1) -> float:
    """dt = safety · L_char / c_R (Rayleigh wave speed)."""
    return safety * L_char / rayleigh_speed(mat)


def _stagger_step(P: Dict, comm, msh,
                  *, tol_stag: float, max_stag: int,
                  msh_coarse=None,
                  build_problem: Optional[Callable] = None,
                  eps: Optional[float] = None,
                  h0: Optional[float] = None,
                  h_min: Optional[float] = None,
                  v_th: float = V_TH,
                  amr_label: str = "AMR",
                  after_rebuild: Optional[Callable] = None,
                  ):
    """Mirror of quasistatic._stagger_step with mid-stagger AMR.

    Returns ``(P, msh, u_res, z_res, stag, divergence)``.

    Captures SNES converged-reason so the dynamic driver can flag a silent
    solver no-op.  Mid-stagger AMR fires between u-solve and z-solve when
    the AMR plumbing args are provided; transfers Newmark-state fields
    (u_prev, v_prev, a_prev) along with u and u_trial.
    """
    from ..common.snes import is_diverged, reason_label
    do_amr = (msh_coarse is not None and build_problem is not None
              and eps is not None and h0 is not None and h_min is not None)
    stag = 0; u_res = z_res = 1e9
    divergence: Optional[Tuple[str, str, int]] = None
    while (stag < max_stag) and (u_res > tol_stag or z_res > tol_stag):
        stag += 1
        P["u_trial"].x.array[:] = P["u"].x.array; P["u_trial"].x.scatter_forward()
        u_iters, u_reason = P["problem_u"].solve()
        if is_diverged(u_reason):
            divergence = ("u", reason_label(u_reason), int(u_iters))
            break

        # ===== Mid-stagger AMR (matches fea_codes/bench_amr.py) =====
        if do_amr:
            msh, P, did_refine = try_amr(
                msh, msh_coarse, P,
                eps=eps, h0=h0, h_min=h_min,
                build_problem=build_problem,
                # Transfer dynamic-specific state alongside u and u_trial.
                extra_V_fields=("u_trial", "u_prev", "v_prev", "a_prev"),
                label=f"{amr_label}-stag{stag}",
                after_rebuild=after_rebuild,
            )
            if did_refine:
                # Re-apply v_th irreversibility upper bound on the new mesh.
                mask = P["z"].x.array <= v_th
                P["z_ub"].x.array[mask] = P["z"].x.array[mask]
                P["z_ub"].x.scatter_forward()
                P["problem_z"].solver.setVariableBounds(
                    P["z_lb"].x.petsc_vec, P["z_ub"].x.petsc_vec)

        P["z_trial"].x.array[:] = P["z"].x.array; P["z_trial"].x.scatter_forward()
        z_iters, z_reason = P["problem_z"].solve()
        if is_diverged(z_reason):
            divergence = ("z", reason_label(z_reason), int(z_iters))
            break
        P["u_diff"].x.array[:] = P["u"].x.array - P["u_trial"].x.array
        P["u_diff"].x.scatter_forward()
        P["z_diff"].x.array[:] = P["z"].x.array - P["z_trial"].x.array
        P["z_diff"].x.scatter_forward()
        u_res = norm_L2_cached(comm, P["u_diff_norm_form"])
        z_res = norm_L2_cached(comm, P["z_diff_norm_form"])
    return P, msh, u_res, z_res, stag, divergence


def run_dynamic(
    P: Dict,
    build_problem: Callable,
    msh_coarse,
    *,
    T_total: float,
    dt: float,
    max_stag: int = 20,
    tol_stag: float = 1.0e-7,
    pressure_ramp: Callable[[float], float] = lambda t: 1.0,
    on_output: Callable = None,
    log_path: str = "output_dyn.txt",
    amr_label: str = "AMR",
    after_rebuild: Callable | None = None,
):
    """Fixed-dt HHT-α dynamic driver.

    `pressure_ramp(t)` — scalar in [0, p_max] that multiplies the pressure
    constant at time t.
    """
    msh  = P["msh"]
    comm = msh.comm; rank = comm.rank
    eps  = P["eps"]; h0 = P["h0"]; h_min = P["h_min"]

    open_log(log_path,
             "step  time  stag_iters  u_res  z_res  min_z  pressure",
             comm)

    max_steps = int(np.ceil(T_total / dt))
    t, step = dt, 1
    while step <= max_steps and t - dt < T_total:
        # Irreversibility: SNES-VI upper bound tracks the current z ONLY
        # for nodes with z ≤ V_TH=0.05 (matches fea_codes/bench_amr.py).
        # Intact / lightly-damaged regions are left free to relax up to 1.
        mask = P["z"].x.array <= V_TH
        P["z_ub"].x.array[mask] = P["z"].x.array[mask]; P["z_ub"].x.scatter_forward()

        # Update traction pressure.
        P["pressure"].value = pressure_ramp(t)

        if rank == 0:
            print(f"\n┌── Step {step:<6d}  │  t = {t:.6e}  │  Δt = {dt:.2e}  "
                  f"│  p = {float(P['pressure'].value):.3e}")

        # Stagger with mid-stagger AMR (matches fea_codes/bench_amr.py).
        P, msh, u_res, z_res, stag, divergence = _stagger_step(
            P, comm, msh,
            tol_stag=tol_stag, max_stag=max_stag,
            msh_coarse=msh_coarse, build_problem=build_problem,
            eps=eps, h0=h0, h_min=h_min,
            v_th=V_TH, amr_label=amr_label,
            after_rebuild=after_rebuild,
        )

        # The dynamic driver uses fixed Δt (Rayleigh-wave-speed CFL); we
        # can't halve and revert without losing causality.  But we still
        # surface the divergence so the agent's HealthReport can flag it.
        if divergence is not None and rank == 0:
            which, reason, iters = divergence
            print(f"    [solver-status] {which}-solve diverged "
                  f"({reason}, {iters} iters) — Δt fixed (dynamic), continuing")

        # Advance Newmark state.
        P["advance_fields"](P, dt)

        min_z = comm.allreduce(P["z"].x.array.min(), op=MPI.MIN)

        if rank == 0:
            print(f"    ✓  u_res = {u_res:.3e}   z_res = {z_res:.3e}   "
                  f"min(z) = {min_z:.3f}   stag = {stag}")
        write_log_line(log_path,
                       f"{step} {t:.6e} {stag} {u_res:.6e} {z_res:.6e} "
                       f"{min_z:.6f} {float(P['pressure'].value):.6e}",
                       comm)

        if on_output is not None:
            on_output(P, t, step, dt,
                      {"min_z": min_z, "u_res": u_res, "z_res": z_res})

        # NB: end-of-step AMR removed — AMR now fires mid-stagger inside
        # ``_stagger_step`` (matches fea_codes/bench_amr.py).

        step += 1
        t += dt

    return P
