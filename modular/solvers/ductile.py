"""Ductile driver — adaptive-dt staggered + quadrature-point transfers.

Key difference vs the linear quasistatic driver: after each u-solve inside
the stagger loop, we interpolate the radial-return expressions onto the
quadrature-point storage (`sig`, `dp_stored`, `deps_pl_stored`) and
accumulate `p` and `eps_pl`. During AMR the plastic state must be
transferred through a direct QP→QP barycentric fit.

AMR (matches fea_codes/ductile_amr.py — Ortiz & Quigley 1991 single-transfer):
  • End-of-step AMR (NOT mid-stagger).  After step n converges, if the AMR
    indicator triggers, build the refined mesh and transfer (p, eps_pl)
    from converged step n to the new QPs via direct QP-to-QP barycentric
    mapping; transfer (u, z) as the initial guess for step n+1; recompute
    stress for consistency; proceed to step n+1 on the new mesh.
  • Irreversibility uses ``mask = z <= 1`` (track z_ub everywhere).
    Damage progression is gated by plastic-strain-driven degradation
    ``z^(2(p/0.12)^2)`` so the v_th=0.05 threshold from the brittle case
    does not apply here.
"""

from __future__ import annotations
from typing import Callable, Dict

import sys
import time
import numpy as np
from dolfinx import fem
from mpi4py import MPI
from scipy import spatial

from ..common.amr import try_amr
from ..common.io_utils import open_log, write_log_line
from ..common.norms import norm_L2_cached
from ..constitutive.j2_plasticity import interpolate_quadrature


# -------------------------------------------------------------------------- #
# Plastic-state QP transfer (Ortiz & Quigley 1991 direct mapping).
# -------------------------------------------------------------------------- #
def transfer_plastic_state_qp(P_old: Dict, P_new: Dict):
    """Transfer (p, eps_pl) from old to new mesh via local linear poly fit.

    Works for triangles (2D, 3 QPs at degree 2) and tetrahedra (3D, 4 QPs
    at degree 2).  We fit a linear polynomial to each cell's QP values and
    evaluate it at the new QP locations.  When a new QP falls outside the
    old mesh, the nearest cell is used (extrapolation).
    """
    msh_old = P_old["msh"]; msh_new = P_new["msh"]
    tdim = msh_old.topology.dim
    gdim = msh_old.geometry.dim

    # --- Old mesh: barycentres + per-cell vertex coords ------------------ #
    n_old = msh_old.topology.index_map(tdim).size_local
    dofmap_o = msh_old.geometry.dofmap
    x_o = msh_old.geometry.x
    cell_verts_o = np.array([x_o[dofmap_o[c]][:, :gdim] for c in range(n_old)])
    centroids_o = cell_verts_o.mean(axis=1)

    # --- Gather old-mesh data to all ranks for cross-partition lookup --- #
    all_centroids = np.vstack(msh_old.comm.allgather(centroids_o))
    all_cell_verts = np.vstack(msh_old.comm.allgather(cell_verts_o))

    ncomp   = P_old["voigt_dim"]
    p_old_all   = np.concatenate(msh_old.comm.allgather(P_old["p"].x.array))
    eps_old_all = np.concatenate(msh_old.comm.allgather(P_old["eps_pl"].x.array))

    q_deg = P_old["q_deg"]
    # Reference QPs (same for all cells with the same cell type).
    ipts = P_old["p"].function_space.element.interpolation_points()
    ref_qpts = np.asarray(ipts)
    nqp = ref_qpts.shape[0]

    # Basis for linear polynomial on simplex: [1, ξ1, ξ2, (ξ3)].
    V_mat = np.column_stack([np.ones(nqp), *[ref_qpts[:, i] for i in range(tdim)]])
    V_inv = np.linalg.pinv(V_mat)        # (tdim+1) × nqp

    # --- New mesh: build physical coords of each new QP ------------------ #
    n_new = msh_new.topology.index_map(tdim).size_local
    dofmap_n = msh_new.geometry.dofmap
    x_n = msh_new.geometry.x
    cell_verts_n = np.array([x_n[dofmap_n[c]][:, :gdim] for c in range(n_new)])

    # Affine map: phys = V0 + J · xi (with appropriate shape functions).
    phi_lin = np.column_stack([1.0 - ref_qpts.sum(axis=1)] +
                              [ref_qpts[:, i] for i in range(tdim)])  # (nqp, tdim+1)
    phys_new = np.einsum("qk,ckd->cqd", phi_lin, cell_verts_n[:, :tdim + 1])
    # Fallback if extra nodes (not a simplex) — just use vertices above.

    # --- Locate each new QP in the old mesh ------------------------------ #
    tree = spatial.cKDTree(all_centroids)
    p_new  = np.zeros(n_new * nqp, dtype=np.float64)
    e_new  = np.zeros(n_new * nqp * ncomp, dtype=np.float64)

    for c in range(n_new):
        for q in range(nqp):
            pt = phys_new[c, q]
            _, nearest = tree.query(pt, k=min(6, all_centroids.shape[0]))
            nearest = np.atleast_1d(nearest)
            chosen = int(nearest[0])
            # Ref coord inside chosen old cell (assume simplex).
            verts = all_cell_verts[chosen, :tdim + 1]
            A = np.column_stack([verts[i + 1] - verts[0] for i in range(tdim)])
            ref = np.linalg.solve(A, pt - verts[0])
            ev = np.array([1.0 - ref.sum()] + [ref[i] for i in range(tdim)])

            # Evaluate linear fit.
            base_p = chosen * nqp
            p_new[c * nqp + q] = ev @ (V_mat.T @ (V_mat @ np.linalg.pinv(V_mat.T @ V_mat)) @
                                         np.zeros(1))  # stub — simpler path below
            old_p = p_old_all[base_p:base_p + nqp]
            p_new[c * nqp + q] = ev @ (V_inv @ np.ones(tdim + 1))  # placeholder
            # Correct evaluation — nodal interpolation:
            p_new[c * nqp + q] = ev @ np.linalg.lstsq(V_mat, old_p, rcond=None)[0]
            for k in range(ncomp):
                old_e = eps_old_all[(base_p + np.arange(nqp)) * ncomp + k]
                e_new[(c * nqp + q) * ncomp + k] = ev @ np.linalg.lstsq(V_mat, old_e,
                                                                         rcond=None)[0]

    P_new["p"].x.array[:n_new * nqp]              = np.maximum(p_new, 0.0)
    P_new["p"].x.scatter_forward()
    P_new["p_old"].x.array[:] = P_new["p"].x.array
    P_new["p_old"].x.scatter_forward()
    P_new["eps_pl"].x.array[:n_new * nqp * ncomp] = e_new
    P_new["eps_pl"].x.scatter_forward()
    P_new["eps_pl_old"].x.array[:] = P_new["eps_pl"].x.array
    P_new["eps_pl_old"].x.scatter_forward()


# -------------------------------------------------------------------------- #
def _stagger_ductile(P: Dict, comm, *, tol_stag: float, max_stag: int):
    """Staggered u-z with radial return after each u solve.

    Returns ``(u_res, z_res, stag, divergence)`` — same protocol as
    quasistatic._stagger_step so the outer driver can halve Δt on a
    silent SNES no-op (active-set blocking the descent direction at
    z=z_ub, line-search failure, etc.).

    AMR is end-of-step (matches fea_codes/ductile_amr.py — Ortiz & Quigley
    single-transfer) and lives in ``run_ductile``; this function does
    NOT touch the mesh.
    """
    from ..common.snes import is_diverged, reason_label
    stag = 0; u_res = z_res = 1e9
    divergence = None
    while (stag < max_stag) and (u_res > tol_stag or z_res > tol_stag):
        stag += 1

        # Solve u.
        P["u_trial"].x.array[:] = P["u"].x.array; P["u_trial"].x.scatter_forward()
        u_iters, u_reason = P["problem_u"].solve()
        if is_diverged(u_reason):
            divergence = ("u", reason_label(u_reason), int(u_iters))
            break

        # Radial return: interpolate UFL expressions onto quadrature storage.
        interpolate_quadrature(P["voigt_from_tensor"](P["new_sig"]), P["sig"])
        interpolate_quadrature(P["dp_"], P["dp_stored"])
        interpolate_quadrature(P["voigt_from_tensor"](P["deps_pl_"]), P["deps_pl_stored"])
        P["p"].x.array[:]      += P["dp_stored"].x.array;      P["p"].x.scatter_forward()
        P["eps_pl"].x.array[:] += P["deps_pl_stored"].x.array; P["eps_pl"].x.scatter_forward()

        # Solve z.
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
    return u_res, z_res, stag, divergence


def run_ductile(
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
    reaction_form: Callable | None = None,
    log_path: str = "output_ductile.txt",
    amr_label: str = "AMR",
    after_rebuild: Callable | None = None,
):
    """Adaptive-dt staggered driver for the ductile case.

    Maintains snapshots of the plastic state (p_old, eps_pl_old) so we can
    revert them on dt cuts.
    """
    msh  = P["msh"]
    comm = msh.comm; rank = comm.rank
    eps  = P["eps"]; h0 = P["h0"]; h_min = P["h_min"]

    dt_first = T_total / steps
    dt_floor = dt_first / 10.0
    stepsize = dt_first
    z_trip   = 10.0 * tol_stag

    open_log(log_path,
             "step  time  dt  stag_iters  u_res  z_res  min_z  max_p  disp  Fy",
             comm)

    t, step = stepsize, 1
    while t - 1e-15 < T_total and step <= steps + 200:
        # Snapshot for possible revert — allocated fresh because AMR may
        # change the problem dimensions.
        saved_u   = fem.Function(P["V"])
        saved_z   = fem.Function(P["Y"])
        saved_zlb = fem.Function(P["Y"])
        saved_u.x.array[:]   = P["u"].x.array;    saved_u.x.scatter_forward()
        saved_z.x.array[:]   = P["z"].x.array;    saved_z.x.scatter_forward()
        saved_zlb.x.array[:] = P["z_lb"].x.array; saved_zlb.x.scatter_forward()
        P["p_old"].x.array[:]      = P["p"].x.array;      P["p_old"].x.scatter_forward()
        P["eps_pl_old"].x.array[:] = P["eps_pl"].x.array; P["eps_pl_old"].x.scatter_forward()

        # Irreversibility: SNES-VI upper bound tracks current z everywhere
        # (mask z <= 1).  Matches fea_codes/ductile_amr.py — damage
        # progression is gated by the plastic-strain-driven degradation
        # ``z^(2(p/0.12)^2)``, so the v_th=0.05 threshold from the brittle
        # case does not apply here.
        mask = P["z"].x.array <= 1.0
        P["z_ub"].x.array[mask] = P["z"].x.array[mask]; P["z_ub"].x.scatter_forward()

        P["disp_const"].value = t * max_disp

        if rank == 0:
            print(f"\n┌── Step {step:<5d}  │  t = {t:.6e}  │  Δt = {stepsize:.2e}  "
                  f"│  {msh.topology.index_map(msh.topology.dim).size_local:>7,} cells")

        u_res, z_res, stag, divergence = _stagger_ductile(
            P, comm, tol_stag=tol_stag, max_stag=max_stag)

        # SNES divergence (any inner solve) → halve and revert.  Mirror of
        # quasistatic.run_quasistatic — same rationale: surfaces silent
        # active-set blocking + line-search failures.
        if divergence is not None:
            which, reason, iters = divergence
            if stepsize > dt_floor:
                if rank == 0:
                    print(f"    [solver-status] {which}-solve diverged "
                          f"({reason}, {iters} iters) — halving Δt and reverting")
                P["u"].x.array[:]    = saved_u.x.array;   P["u"].x.scatter_forward()
                P["z"].x.array[:]    = saved_z.x.array;   P["z"].x.scatter_forward()
                P["z_lb"].x.array[:] = saved_zlb.x.array; P["z_lb"].x.scatter_forward()
                P["p"].x.array[:]      = P["p_old"].x.array;      P["p"].x.scatter_forward()
                P["eps_pl"].x.array[:] = P["eps_pl_old"].x.array; P["eps_pl"].x.scatter_forward()
                stepsize = max(stepsize / 2.0, dt_floor)
                continue
            if rank == 0:
                print(f"    [solver-status] {which}-solve diverged "
                      f"({reason}) at Δt floor — accepting; check log")

        elif z_res > z_trip and stepsize > dt_floor:
            if rank == 0:
                print(f"    ⚠  z_res = {z_res:.3e} > {z_trip:.1e} — halving Δt and reverting")
            P["u"].x.array[:]    = saved_u.x.array;   P["u"].x.scatter_forward()
            P["z"].x.array[:]    = saved_z.x.array;   P["z"].x.scatter_forward()
            P["z_lb"].x.array[:] = saved_zlb.x.array; P["z_lb"].x.scatter_forward()
            P["p"].x.array[:]      = P["p_old"].x.array;      P["p"].x.scatter_forward()
            P["eps_pl"].x.array[:] = P["eps_pl_old"].x.array; P["eps_pl"].x.scatter_forward()
            stepsize = max(stepsize / 2.0, dt_floor)
            continue

        min_z = comm.allreduce(P["z"].x.array.min(), op=MPI.MIN)
        max_p = comm.allreduce(P["p"].x.array.max(), op=MPI.MAX)
        Fy = 0.0
        if reaction_form is not None:
            Fy = comm.allreduce(fem.assemble_scalar(reaction_form(P)), op=MPI.SUM)

        if rank == 0:
            print(f"    ✓  u_res = {u_res:.3e}   z_res = {z_res:.3e}   "
                  f"min(z) = {min_z:.3f}   max(p) = {max_p:.3f}   stag = {stag}")
        write_log_line(log_path,
                       f"{step} {t:.6e} {stepsize:.6e} {stag} {u_res:.6e} {z_res:.6e} "
                       f"{min_z:.6f} {max_p:.6f} {t * max_disp:.6e} {Fy:.6e}",
                       comm)

        if on_output is not None:
            on_output(P, t, step, stepsize,
                      {"min_z": min_z, "max_p": max_p, "Fy": Fy,
                       "u_res": u_res, "z_res": z_res})

        # End-of-step AMR with quadrature-point plastic-state transfer
        # (matches fea_codes/ductile_amr.py — Ortiz & Quigley 1991
        # single-transfer scheme).  Direct QP-to-QP barycentric mapping
        # for (p, eps_pl) lives in ``transfer_plastic_state_qp``;
        # nodal fields (u, z, z_lb, z_ub, z_trial) are interpolated by
        # ``try_amr`` itself; stress is recomputed from the constitutive
        # law for consistency.
        def _after_rebuild(P_new):
            transfer_plastic_state_qp(P_old_ref["P"], P_new)
            # Recompute stress consistent with transferred state.
            interpolate_quadrature(P_new["voigt_from_tensor"](P_new["new_sig"]),
                                    P_new["sig"])

        P_old_ref = {"P": P}  # captured for the after_rebuild closure
        # Chain the user's after_rebuild (re-installs ramp proxies) AFTER the
        # solver's internal one (transfers plastic state).
        def _chained_rebuild(Pn, _user_cb=after_rebuild):
            _after_rebuild(Pn)
            if _user_cb is not None:
                _user_cb(Pn)
        msh, P, did = try_amr(msh, msh_coarse, P,
                              eps=eps, h0=h0, h_min=h_min,
                              build_problem=build_problem,
                              label=amr_label,
                              after_rebuild=_chained_rebuild)

        step += 1
        t = min(t + stepsize, T_total + 1e-14)
        if t >= T_total - 1e-14:
            break

    return P
