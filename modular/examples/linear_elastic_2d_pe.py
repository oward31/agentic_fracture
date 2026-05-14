#!/usr/bin/env python3
"""Linear elasticity, 2D plane strain — bench-style three-point bending.

Run:
    wsl
    python modular/examples/linear_elastic_2d_pe.py
or
    mpirun -n 4 python modular/examples/linear_elastic_2d_pe.py
"""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mpi4py import MPI

from modular.materials              import load_material
from modular.meshes                 import make_notched_plate_2d
from modular.common                 import print_banner, print_mesh_info
from modular.problems               import make_linear_elastic_2d_pe_builder
from modular.post                   import XDMFWriter, reaction_form_from_sigma_2d
from modular.solvers                import run_quasistatic


def main():
    comm = MPI.COMM_WORLD

    # ---- Material + derived length scales ------------------------------- #
    mat = load_material("Steel_bench_2D_PE")
    eps, h0, h_min = mat["eps"], mat["h0"], mat["h_min"]
    print_banner(f"Linear elastic 2D-PE — {mat['name']}  "
                 f"eps={eps:.3e}  h0={h0:.3e}  h_min={h_min:.3e}", comm)

    # ---- Mesh ---------------------------------------------------------- #
    W, L = 1.0, 1.0
    ac, cw = W / 2.0, 0.001
    msh_coarse, markers, geom = make_notched_plate_2d(
        W=W, L=L, ac=ac, cw=cw, h0=h0, comm=comm)
    msh_coarse.name = "bench"
    print_mesh_info(msh_coarse, comm)

    # ---- Problem builder factory --------------------------------------- #
    build_problem = make_linear_elastic_2d_pe_builder(
        mat=mat, markers_spec=markers, geom=geom,
        eps=eps, h0=h0, h_min=h_min,
    )
    P = build_problem(msh_coarse)

    # ---- Post-processing ----------------------------------------------- #
    writer = XDMFWriter("paraview_linear_2d_pe")
    def on_out(P, t, step, dt, extras):
        writer.write(P, t, step)

    # ---- Solve --------------------------------------------------------- #
    run_quasistatic(
        P, build_problem, msh_coarse,
        T_total=1.0, steps=200, max_stag=20, tol_stag=1.0e-7,
        max_disp=0.006,
        on_output=on_out,
        reaction_form=reaction_form_from_sigma_2d("top", component=1),
        log_path="output_linear_2d_pe.txt",
    )


if __name__ == "__main__":
    main()
