"""Adaptive mesh refinement helpers — fixed parameters.

Per user spec, this module hardcodes the AMR rules so the callers cannot
make a bad choice:

    refine_radius      = 3 * eps
    pf_flag_threshold  = -0.1
    coarsen            = False                (refinement only)
    h_threshold_factor = 1.5   (a cell is 'too coarse' iff h > 1.5 * h_min)

Multi-level bisection: each pass halves the edge length inside the active
zone; the number of levels is `ceil(log2(h0 / h_min))` (= 3 when h_min = h0/8).

The `try_amr` entry point plugs into any problem builder that exposes the
usual dict keys: V, Y, u, z, z_lb, z_ub, z_trial, problem_u, problem_z,
indicator_expr, cell_h.

See `amr.try_amr_with_transfers` for the ductile / dynamic variants that
need extra fields (plastic state, velocities) transferred.
"""

from __future__ import annotations
import sys
import time
from typing import Callable, Iterable, Dict, Any, Optional

import numpy as np
from dolfinx import fem, mesh
from dolfinx.fem import create_interpolation_data
from dolfinx.mesh import compute_midpoints, refine, RefinementOption
from mpi4py import MPI
from scipy import spatial

from .transfer import transfer_function

# --- Fixed AMR parameters (user-mandated) ---------------------------------- #
PF_FLAG_THRESHOLD    = -0.1
REFINE_RADIUS_FACTOR = 3.0      # refine_radius = 3 * eps
H_THRESHOLD_FACTOR   = 1.5      # cell too coarse iff h > 1.5 * h_min
INDICATOR_ACTIVE     = 0.5      # cells with indicator > 0.5 are active


# -------------------------------------------------------------------------- #
# Low-level edge / midpoint helpers
# -------------------------------------------------------------------------- #
def compute_edge_lengths(msh):
    """Vectorised edge-length computation. Returns (lengths, midpoints)."""
    tdim = msh.topology.dim
    msh.topology.create_connectivity(1, 0)
    msh.topology.create_connectivity(1, tdim)
    msh.topology.create_connectivity(tdim, 0)

    num_edges = msh.topology.index_map(1).size_local
    edge_indices = np.arange(num_edges, dtype=np.int32)
    midpts = compute_midpoints(msh, 1, edge_indices)

    e2v, c2v = msh.topology.connectivity(1, 0), msh.topology.connectivity(tdim, 0)
    gdofmap, x = msh.geometry.dofmap, msh.geometry.x

    total_verts = msh.topology.index_map(0).size_local + msh.topology.index_map(0).num_ghosts
    total_cells = msh.topology.index_map(tdim).size_local + msh.topology.index_map(tdim).num_ghosts
    v2g = np.empty(total_verts, dtype=np.int32)

    try:
        c2v_arr, c2v_off = c2v.array, c2v.offsets
        vpc = c2v_off[1] - c2v_off[0]
        v2g[c2v_arr[:total_cells * vpc]] = gdofmap[:total_cells].ravel()[:total_cells * vpc]
    except (AttributeError, TypeError):
        all_tverts = np.concatenate([c2v.links(c) for c in range(total_cells)])
        v2g[all_tverts] = np.concatenate([gdofmap[c] for c in range(total_cells)])

    try:
        e2v_flat = e2v.array[:num_edges * 2].reshape(num_edges, 2)
    except (AttributeError, TypeError):
        e2v_flat = np.array([e2v.links(e) for e in range(num_edges)])

    diff = x[v2g[e2v_flat[:, 0]]] - x[v2g[e2v_flat[:, 1]]]
    return np.sqrt(np.einsum('ij,ij->i', diff, diff)), midpts


def extract_active_zone_points(msh, indicator_values, threshold=INDICATOR_ACTIVE):
    """All-gathered midpoints of cells where indicator > threshold."""
    tdim = msh.topology.dim
    n    = msh.topology.index_map(tdim).size_local
    trigger = np.where(indicator_values[:n] > threshold)[0]
    local_pts = (compute_midpoints(msh, tdim, trigger.astype(np.int32))
                 if trigger.size > 0
                 else np.empty((0, 3), dtype=np.float64))
    non_empty = [p for p in msh.comm.allgather(local_pts) if p.shape[0] > 0]
    return np.vstack(non_empty) if non_empty else None


def drain_pending_messages(comm):
    """Drain unmatched MPI messages to prevent MPI_Comm_free aborts."""
    while True:
        status = MPI.Status()
        if not comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
            break
        count = status.Get_count(MPI.BYTE)
        comm.Recv(bytearray(count if count != MPI.UNDEFINED else 0),
                  source=status.Get_source(), tag=status.Get_tag())


# -------------------------------------------------------------------------- #
# Multi-level refinement
# -------------------------------------------------------------------------- #
def refine_from_coarse(msh_coarse, active_points, *, eps, h_min, h0,
                       partitioner=None):
    """Multi-level bisection refinement near `active_points`.

    refine_radius = 3 * eps   (fixed).
    """
    refine_radius = REFINE_RADIUS_FACTOR * eps
    n_levels = max(1, int(np.ceil(np.log2(h0 / h_min))))
    tree = spatial.cKDTree(active_points)
    kept_refs, msh = [], msh_coarse

    for level in range(n_levels):
        h_target = h0 / (2 ** (level + 1))
        edge_lengths, edge_midpoints = compute_edge_lengths(msh)
        long_idx = np.where(edge_lengths > h_target * 1.05)[0]

        if long_idx.size > 0:
            dists, _ = tree.query(edge_midpoints[long_idx], distance_upper_bound=refine_radius)
            edges_to_refine = long_idx[np.isfinite(dists)].astype(np.int32)
        else:
            edges_to_refine = np.array([], dtype=np.int32)

        global_count = msh.comm.allreduce(int(edges_to_refine.size), op=MPI.SUM)
        if global_count == 0:
            break

        if msh.comm.rank == 0:
            print(f"        ↳ Level {level+1}/{n_levels}: "
                  f"refining {global_count:,} edges  (h_target = {h_target:.4e})")
            sys.stdout.flush()

        msh.comm.Set_errhandler(MPI.ERRORS_RETURN)
        if partitioner is None:
            msh_new, pc, pe = refine(msh, edges_to_refine, option=RefinementOption.parent_cell)
        else:
            msh_new, pc, pe = refine(msh, edges_to_refine, partitioner=partitioner,
                                     option=RefinementOption.parent_cell)
        MPI.COMM_WORLD.Barrier()
        drain_pending_messages(msh.comm)
        drain_pending_messages(msh_new.comm)
        msh.comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
        msh_new.comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)

        kept_refs.append((msh, pc, pe))
        msh = msh_new

    MPI.COMM_WORLD.Barrier()
    del kept_refs
    return msh


# -------------------------------------------------------------------------- #
# Public driver — generic case (linear elasticity / dynamic).
# -------------------------------------------------------------------------- #
def try_amr(msh, msh_coarse, P: Dict[str, Any], *,
            eps: float, h0: float, h_min: float,
            build_problem: Callable,
            label: str = "AMR",
            extra_V_fields: Iterable[str] = (),
            extra_Y_fields: Iterable[str] = ("z", "z_lb", "z_ub", "z_trial"),
            partitioner=None,
            after_rebuild: Optional[Callable[[Dict[str, Any]], None]] = None):
    """Check AMR trigger; refine if needed; return (msh, P, did_refine).

    `build_problem(msh_new)` must reconstruct a fresh problem dict with the
    same keys as `P`. All fields named in `extra_V_fields` / `extra_Y_fields`
    are transferred from the old to the new problem via non-matching
    interpolation.
    """
    amr_start = time.time()
    comm      = msh.comm
    rank      = comm.rank

    # --- Evaluate indicator and cell sizes -------------------------------- #
    # Two evaluation paths:
    #   * `indicator_expr`        — cheap Expression-based interpolation
    #   * `indicator_proj_fn(P)`  — fallback L2 projection, required when
    #                               the indicator references a quadrature
    #                               function (ductile case).
    if "indicator_proj_fn" in P and P["indicator_proj_fn"] is not None:
        indicator_proj = P["indicator_proj_fn"](P)
    else:
        indicator_proj = fem.Function(P["Yv"])
        indicator_proj.interpolate(P["indicator_expr"])
    tdim    = msh.topology.dim
    n_local = msh.topology.index_map(tdim).size_local
    ind_vals = indicator_proj.x.array[:n_local]
    h_vals   = P["cell_h"].x.array[:n_local]

    # --- Trigger: flagged AND too coarse ---------------------------------- #
    h_threshold = H_THRESHOLD_FACTOR * h_min
    flagged     = ind_vals > INDICATOR_ACTIVE
    need_refine = flagged & (h_vals > h_threshold)

    n_coarse  = comm.allreduce(int(need_refine.sum()), op=MPI.SUM)
    n_flagged = comm.allreduce(int(flagged.sum()),     op=MPI.SUM)

    if n_coarse == 0:
        if rank == 0:
            msg = ("No flagged cells — mesh unchanged"
                   if n_flagged == 0
                   else f"{n_flagged:,} flagged cells, all fine enough — skipping")
            print(f"    [{label}]  {msg}")
            sys.stdout.flush()
        return msh, P, False

    if rank == 0:
        h_max = comm.allreduce(float(h_vals[need_refine].max()) if need_refine.any() else 0.0, op=MPI.MAX)
        print(f"    [{label}]  {n_coarse:,} coarse flagged cells "
              f"(h_max = {h_max:.4e} > {h_threshold:.4e}) — refining ...")
        sys.stdout.flush()

    # --- Build refined mesh from the original coarse one ------------------ #
    active_points = extract_active_zone_points(msh, indicator_proj.x.array[:])
    msh_new = refine_from_coarse(msh_coarse, active_points,
                                 eps=eps, h_min=h_min, h0=h0,
                                 partitioner=partitioner)
    msh_new.name = getattr(msh, "name", "mesh")

    P_old = P
    P_new = build_problem(msh_new)

    # --- Transfer fields -------------------------------------------------- #
    nc = msh_new.topology.index_map(msh_new.topology.dim).size_local
    cells = np.arange(nc, dtype=np.int32)

    V_fields = ("u", *extra_V_fields)
    Y_fields = tuple(extra_Y_fields)

    if V_fields:
        interp_V = create_interpolation_data(P_new["V"], P_old["u"].function_space,
                                             cells, padding=1e-8)
        for name in V_fields:
            if name in P_old and name in P_new:
                f = transfer_function(P_old[name], P_new["V"],
                                      cells=cells, interp_data=interp_V)
                P_new[name].x.array[:] = f.x.array
                P_new[name].x.scatter_forward()

    if Y_fields:
        interp_Y = create_interpolation_data(P_new["Y"], P_old["z"].function_space,
                                             cells, padding=1e-8)
        for name in Y_fields:
            if name in P_old and name in P_new:
                f = transfer_function(P_old[name], P_new["Y"],
                                      cells=cells, interp_data=interp_Y)
                P_new[name].x.array[:] = f.x.array
                P_new[name].x.scatter_forward()

    # --- Re-apply bounds on z solver ------------------------------------- #
    if "problem_z" in P_new and P_new.get("z_lb") is not None and P_new.get("z_ub") is not None:
        P_new["problem_z"].solver.setVariableBounds(P_new["z_lb"].x.petsc_vec,
                                                    P_new["z_ub"].x.petsc_vec)

    # --- Destroy old solvers to release PETSc objects -------------------- #
    for s in ("problem_u", "problem_z"):
        if s in P_old:
            P_old[s].destroy()

    if after_rebuild is not None:
        after_rebuild(P_new)

    MPI.COMM_WORLD.Barrier()
    drain_pending_messages(comm)
    MPI.COMM_WORLD.Barrier()
    del P_old

    if rank == 0:
        print(f"    [{label}]  Refined → {nc:,} cells  ({time.time() - amr_start:.2f} s)")
        sys.stdout.flush()
    return msh_new, P_new, True
