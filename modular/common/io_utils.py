"""Small I/O helpers: banners, mesh summary, log files."""

from __future__ import annotations
import os
import sys
from itertools import combinations

import numpy as np
from mpi4py import MPI


def print_banner(title: str, comm=None, width: int = 70):
    if comm is not None and comm.rank != 0:
        return
    print()
    print("╔" + "═" * width + "╗")
    print("║" + f"{title:^{width}}" + "║")
    print("╚" + "═" * width + "╝")
    sys.stdout.flush()


def print_mesh_info(msh, comm, output_dir=None, label: str = "INITIAL MESH"):
    """Print mesh statistics — runs in parallel, prints on rank 0."""
    tdim = msh.topology.dim
    gdim = msh.geometry.dim
    for d in range(tdim + 1):
        msh.topology.create_entities(d)

    n_cells_local  = msh.topology.index_map(tdim).size_local
    n_facets_local = msh.topology.index_map(tdim - 1).size_local
    n_verts_local  = msh.topology.index_map(0).size_local

    n_cells  = comm.allreduce(n_cells_local,  op=MPI.SUM)
    n_facets = comm.allreduce(n_facets_local, op=MPI.SUM)
    n_verts  = comm.allreduce(n_verts_local,  op=MPI.SUM)

    coords, dofmap = msh.geometry.x, msh.geometry.dofmap
    h_local = []
    for c in range(n_cells_local):
        pts = coords[dofmap[c]]
        h_local.append(max(np.linalg.norm(pts[i] - pts[j])
                           for i, j in combinations(range(len(pts)), 2)))
    h_local = np.array(h_local) if h_local else np.array([0.0])
    h_min_m = comm.allreduce(float(h_local.min()), op=MPI.MIN)
    h_max_m = comm.allreduce(float(h_local.max()), op=MPI.MAX)
    h_avg_m = comm.allreduce(float(h_local.sum()), op=MPI.SUM) / max(n_cells, 1)

    if comm.rank == 0:
        w = 70
        print()
        print("╔" + "═" * w + "╗")
        print("║" + f"{label:^{w}}" + "║")
        print("╠" + "═" * w + "╣")
        print(f"║  Cell type: {msh.topology.cell_type.name:<20}  dim(geom)={gdim}  dim(topo)={tdim}{'':<{w - 60}}║")
        print(f"║  #vertices = {n_verts:>10,}   #facets = {n_facets:>10,}   #cells = {n_cells:>10,}{'':<{w - 62}}║")
        print(f"║  h: min = {h_min_m:.3e}   max = {h_max_m:.3e}   avg = {h_avg_m:.3e}{'':<{w - 60}}║")
        print("╚" + "═" * w + "╝")
        sys.stdout.flush()


def open_log(path: str, header: str, comm):
    """Rank-0 creates the log with a header. Safe to call before the main loop."""
    if comm.rank == 0:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(header.rstrip() + "\n")


def write_log_line(path: str, line: str, comm):
    if comm.rank == 0:
        with open(path, "a") as f:
            f.write(line.rstrip() + "\n")
