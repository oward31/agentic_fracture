#!/usr/bin/env python3
"""Ductile phase-field fracture, 3D — extruded dogbone."""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mpi4py import MPI

from modular.materials              import load_material
from modular.meshes                 import make_dogbone_3d
from modular.common                 import print_banner, print_mesh_info
from modular.problems               import make_ductile_3d_builder
from modular.post                   import XDMFWriter
from modular.solvers                import run_ductile


def main():
    comm = MPI.COMM_WORLD

    mat = load_material("Al_ductile_3D")
    eps, h0, h_min = mat["eps"], mat["h0"], mat["h_min"]
    print_banner(f"Ductile 3D — {mat['name']}", comm)

    W, L, R, thk = 18.0, 50.0, 2.5, 4.0
    msh_coarse, markers, geom = make_dogbone_3d(
        W=W, L=L, R=R, thickness=thk, h0=h0, comm=comm)
    msh_coarse.name = "dogbone_3d"
    print_mesh_info(msh_coarse, comm)

    build_problem = make_ductile_3d_builder(
        mat=mat, markers_spec=markers, geom=geom,
        eps=eps, h0=h0, h_min=h_min,
    )
    P = build_problem(msh_coarse)

    writer = XDMFWriter("paraview_ductile_3d", include_plastic=True)
    def on_out(P, t, step, dt, extras):
        writer.write(P, t, step)

    run_ductile(
        P, build_problem, msh_coarse,
        T_total=1.0, steps=200, max_stag=20, tol_stag=1.0e-7,
        max_disp=2.0,
        on_output=on_out,
        log_path="output_ductile_3d.txt",
    )


if __name__ == "__main__":
    main()
