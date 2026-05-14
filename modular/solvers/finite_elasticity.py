"""Finite-elasticity driver — hand-rolled Newton with H1 regularisation ω.

Mirrors `slant_amr.py` exactly:
  * Per-step max dt shrinks near onset (25% of nominal for t < 0.5).
  * Newton on u (up to `newton_max_u`), then AMR mid-stagger, then Newton on z.
  * On Newton divergence: cut dt in half, revert fields, reduce ω; after 3
    failures switch to gradient-flow mode (larger ω stabilisation).
  * Never-die wrapper: any exception rolls back the state and continues.
"""

from __future__ import annotations
from typing import Callable, Dict

import sys
import time
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from ..common.amr import try_amr
from ..common.io_utils import open_log, write_log_line


# -------------------------------------------------------------------------- #
# Newton sub-loops
# -------------------------------------------------------------------------- #
def _newton_u(P, *, newton_max: int, newton_fail: int, atol: float,
              stepsize, minstepsize, grad_flow_mode: bool):
    """Hand-rolled Newton on the regularised u system. Returns (rnorm, terminate)."""
    nIter, rnorm_prev = 0, 1e3
    terminate = 0
    while nIter < newton_max:
        nIter += 1
        if nIter > newton_fail and not grad_flow_mode:
            terminate = 1; break

        with P["b_u"].localForm() as bl: bl.set(0.0)
        fem_petsc.assemble_vector(P["b_u"], P["R_u_form"])
        fem_petsc.apply_lifting(P["b_u"], [P["J_reg_form"]], bcs=[P["bcs_du"]])
        P["b_u"].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(P["b_u"], P["bcs_du"])

        rnorm = P["b_u"].norm()
        if rnorm < atol: return rnorm, 0
        if rnorm > rnorm_prev * 50.0 or np.isnan(rnorm):
            terminate = 1; break
        if nIter == 15 and rnorm > 1.0: terminate = 1; break
        rnorm_prev = rnorm
        P["b_u"].scale(-1.0)

        P["A_u"].zeroEntries()
        fem_petsc.assemble_matrix(P["A_u"], P["J_reg_form"], bcs=P["bcs_du"])
        P["A_u"].assemble()
        P["ksp_u"].setOperators(P["A_u"])
        P["ksp_u"].solve(P["b_u"], P["u_inc"].x.petsc_vec)
        P["u_inc"].x.scatter_forward()
        P["u"].x.petsc_vec.axpy(1.0, P["u_inc"].x.petsc_vec)
        P["u"].x.scatter_forward()

    return rnorm_prev, terminate


def _newton_z(P, *, newton_max: int, atol: float, minstepsize, stepsize, step):
    """Newton on z. Returns (rnorm_z, terminate)."""
    nIter_z, rnorm_z_prev = 0, 1e3
    terminate = 0
    while nIter_z < newton_max:
        nIter_z += 1
        with P["b_z"].localForm() as bl: bl.set(0.0)
        fem_petsc.assemble_vector(P["b_z"], P["R_z_form"])
        fem_petsc.apply_lifting(P["b_z"], [P["J_z_form"]], bcs=[P["bcs_dz"]])
        P["b_z"].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(P["b_z"], P["bcs_dz"])
        rnorm_z = P["b_z"].norm()
        if rnorm_z < atol: return rnorm_z, 0
        if step > 1 and (rnorm_z > rnorm_z_prev * 1e4 or np.isnan(rnorm_z)):
            if stepsize <= minstepsize:
                break
            else:
                terminate = 1; break
        rnorm_z_prev = rnorm_z
        P["b_z"].scale(-1.0)
        P["A_z"].zeroEntries()
        fem_petsc.assemble_matrix(P["A_z"], P["J_z_form"], bcs=P["bcs_dz"])
        P["A_z"].assemble()
        P["ksp_z"].setOperators(P["A_z"])
        P["ksp_z"].solve(P["b_z"], P["z_inc"].x.petsc_vec)
        P["z_inc"].x.scatter_forward()
        P["z"].x.petsc_vec.axpy(1.0, P["z_inc"].x.petsc_vec)
        P["z"].x.scatter_forward()
    return rnorm_z_prev, terminate


# -------------------------------------------------------------------------- #
def run_finite_elasticity(
    P: Dict,
    build_problem: Callable,
    msh_coarse,
    *,
    T_end: float = 1.0,
    Totalsteps: int = 250,
    nostagiter: int = 10,
    stag_tol: float = 1.0e-6,
    newton_max_u: int = 25,
    newton_fail_u: int = 20,
    newton_max_z: int = 25,
    newton_atol_u: float = 1.0e-8,
    newton_atol_z: float = 1.0e-8,
    omega_init: float = 1.0e6,
    on_output: Callable = None,
    reaction_form: Callable | None = None,
    log_path: str = "output_fe.txt",
    amr_label: str = "AMR",
    top_disp_fn: Callable[[float], Callable] | None = None,
    after_rebuild: Callable | None = None,
):
    """Finite-elasticity quasistatic driver.

    `top_disp_fn(t)` — returns an `interpolate`-friendly function `(x) -> (d, n)`
    that describes the prescribed displacement on the top face at time t.
    Default: u_y = t · maxdisp.
    """
    msh  = P["msh"]
    comm = msh.comm; rank = comm.rank
    eps  = P["eps"]; h0 = P["h0"]; h_min = P["h_min"]
    maxdisp = P.get("maxdisp", 1.0)

    if top_disp_fn is None:
        vec_dim = P["V"].dofmap.index_map_bs
        def default_top_disp_fn(t):
            def _eval(x):
                vals = np.zeros((vec_dim, x.shape[1]))
                vals[1] = t * maxdisp
                return vals
            return _eval
        top_disp_fn = default_top_disp_fn

    open_log(log_path, "t  lambda_y  Fy  z_min", comm)

    minstepsize = 1.0 / Totalsteps / 10000.0
    maxstepsize = 1.0 / Totalsteps * 1.0
    stepsize      = 1.0 / Totalsteps
    stepsize_prev = stepsize

    t, step = stepsize, 1
    samesizecount, nrcount, gfcount = 1, 0, 0
    omega, terminate2 = omega_init, 0
    have_two_solutions = False

    while t - stepsize < T_end:
        try:
            maxstepsize = 1.0 / Totalsteps * (0.25 if t < 0.5 else 1.0)

            if terminate2 == 0:
                omega = omega_init
            else:
                gfcount += 1
                if omega > omega_init: omega = omega_init

            if gfcount > 10:
                gfcount = 0; omega *= 10.0

            P["omega_c"].value = omega

            if rank == 0:
                nc = msh.topology.index_map(msh.topology.dim).size_local
                print(f"\n┌── Step {step:<6d}  │  t = {t:.6e}  │  Δt = {stepsize:.2e}  "
                      f"│  ω = {omega:.1e}  │  {nc:>6,} cells")

            # Update prescribed top displacement.
            P["r_func"].interpolate(top_disp_fn(t)); P["r_func"].x.scatter_forward()
            fem_petsc.set_bc(P["u"].x.petsc_vec, P["bcs_u"]); P["u"].x.scatter_forward()

            # Staggered Newton.
            stag_iter, rnorm_stag, terminate = 0, 1.0, 0
            while stag_iter < nostagiter and rnorm_stag > stag_tol:
                stag_iter += 1
                rnorm_u, term_u = _newton_u(
                    P, newton_max=newton_max_u, newton_fail=newton_fail_u,
                    atol=newton_atol_u, stepsize=stepsize,
                    minstepsize=minstepsize, grad_flow_mode=bool(terminate2))
                if term_u == 1:
                    if stepsize <= minstepsize:
                        terminate = 0   # accept at min dt
                    else:
                        terminate = 1; break

                # Mid-stagger AMR.  Chain the user's after_rebuild (ramp
                # proxy re-install) AFTER the internal omega_c re-set.
                def _chained_rebuild(Pn, _user_cb=after_rebuild, _omega=omega):
                    Pn["omega_c"].value = _omega
                    if _user_cb is not None:
                        _user_cb(Pn)
                msh, P, did = try_amr(msh, msh_coarse, P,
                                      eps=eps, h0=h0, h_min=h_min,
                                      build_problem=build_problem,
                                      extra_V_fields=("u_prev", "u_prev_prev"),
                                      extra_Y_fields=("z", "z_lb", "z_ub", "z_prev"),
                                      after_rebuild=_chained_rebuild,
                                      label=f"{amr_label}-s{stag_iter}")
                if did:
                    P["r_func"].interpolate(top_disp_fn(t)); P["r_func"].x.scatter_forward()
                    fem_petsc.set_bc(P["u"].x.petsc_vec, P["bcs_u"]); P["u"].x.scatter_forward()

                rnorm_z, term_z = _newton_z(P, newton_max=newton_max_z,
                                             atol=newton_atol_z,
                                             minstepsize=minstepsize,
                                             stepsize=stepsize, step=step)
                if term_z == 1:
                    terminate = 1; break

                # Recompute u residual for stagger convergence test.
                with P["b_u"].localForm() as bl: bl.set(0.0)
                fem_petsc.assemble_vector(P["b_u"], P["R_u_form"])
                fem_petsc.apply_lifting(P["b_u"], [P["J_reg_form"]], bcs=[P["bcs_du"]])
                P["b_u"].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                fem_petsc.set_bc(P["b_u"], P["bcs_du"])
                rnorm_stag = P["b_u"].norm()

            # Accept / reject.
            if terminate == 1 and stepsize > minstepsize:
                P["u"].x.array[:] = P["u_prev"].x.array; P["u"].x.scatter_forward()
                P["z"].x.array[:] = P["z_prev"].x.array; P["z"].x.scatter_forward()

            if terminate != 1 or stepsize <= minstepsize:
                P["u_prev_prev"].x.array[:] = P["u_prev"].x.array; P["u_prev_prev"].x.scatter_forward()
                P["u_prev"].x.array[:]      = P["u"].x.array;       P["u_prev"].x.scatter_forward()
                P["z_prev"].x.array[:]      = P["z"].x.array;       P["z_prev"].x.scatter_forward()
                stepsize_prev = stepsize; have_two_solutions = True

                Fy = comm.allreduce(fem.assemble_scalar(P["stress_form"]), op=MPI.SUM)
                zmin = comm.allreduce(P["z"].x.array.min(), op=MPI.MIN)
                write_log_line(log_path, f"{t:.6e} {t*maxdisp:.6e} {Fy:.6e} {zmin:.6e}", comm)
                if rank == 0:
                    print(f"    ✓  Fy = {Fy:.6e}   min(z) = {zmin:.4f}")

                if on_output is not None:
                    on_output(P, t, step, stepsize, {"Fy": Fy, "min_z": zmin})

            # Adaptive time step.
            if terminate == 1:
                if stepsize > minstepsize:
                    t -= stepsize
                    stepsize = max(stepsize / 2.0, minstepsize)
                    t += stepsize
                    samesizecount = 1
                    if gfcount > 0: gfcount -= 1
                    omega *= 0.1
                    if terminate2 == 0: nrcount += 1
                    if nrcount > 3:
                        terminate2 = 1; nrcount = 0
                else:
                    step += 1; samesizecount = 1; t += stepsize
            else:
                if samesizecount < 2:
                    step += 1
                    if t + stepsize <= T_end:
                        samesizecount += 1; t += stepsize
                    else:
                        samesizecount = 1; stepsize = T_end - t; t += stepsize
                else:
                    step += 1
                    if stepsize * 2 <= maxstepsize and t + stepsize * 2 <= T_end:
                        stepsize *= 2; t += stepsize
                    elif stepsize * 2 > maxstepsize and t + maxstepsize <= T_end:
                        stepsize = maxstepsize; t += stepsize
                    else:
                        stepsize = T_end - t; t += stepsize
                    samesizecount = 1

        except Exception as e:
            if rank == 0:
                print(f"    !!! UNEXPECTED ERROR: {e}  — rolling back and halving Δt")
                sys.stdout.flush()
            P["u"].x.array[:] = P["u_prev"].x.array; P["u"].x.scatter_forward()
            P["z"].x.array[:] = P["z_prev"].x.array; P["z"].x.scatter_forward()
            t -= stepsize
            stepsize = max(stepsize / 2.0, minstepsize)
            t += stepsize
            samesizecount = 1

    return P
