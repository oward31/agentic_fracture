#!/usr/bin/env python3
"""Finite elasticity (Lopez-Pamies), 3D — extruded notched plate."""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mpi4py import MPI

from modular.materials              import load_material
from modular.meshes                 import make_notched_plate_3d
from modular.common                 import print_banner, print_mesh_info
from modular.problems               import make_finite_elastic_3d_builder
from modular.post                   import XDMFWriter
from modular.solvers                import run_finite_elasticity


def main():
    comm = MPI.COMM_WORLD

    mat = load_material("Rubber_LopezPamies_3D")
    eps, h0, h_min = mat["eps"], mat["h0"], mat["h_min"]
    print_banner(f"Finite elastic 3D — {mat['name']}", comm)

    W, L, thk = 20.0, 20.0, 2.0
    msh_coarse, markers, geom = make_notched_plate_3d(
        W=W, L=L, ac=W/2.0, cw=0.05, thickness=thk, h0=h0, comm=comm)
    msh_coarse.name = "finite_plate_3d"
    print_mesh_info(msh_coarse, comm)

    build_problem = make_finite_elastic_3d_builder(
        mat=mat, markers_spec=markers, geom=geom,
        eps=eps, h0=h0, h_min=h_min, maxdisp=5.0,
    )
    P = build_problem(msh_coarse)

    writer = XDMFWriter("paraview_finite_3d", include_stress=False)
    def on_out(P, t, step, dt, extras):
        if step % 5 == 0:
            writer.write(P, t, step)

    run_finite_elasticity(
        P, build_problem, msh_coarse,
        T_end=1.0, Totalsteps=250,
        on_output=on_out,
        log_path="output_finite_3d.txt",
    )


if __name__ == "__main__":
    main()
