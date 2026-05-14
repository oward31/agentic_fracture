#!/usr/bin/env python3
"""Linear elasticity, 2D plane stress — displacement-controlled notched plate."""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mpi4py import MPI

from modular.materials              import load_material
from modular.meshes                 import make_notched_plate_2d
from modular.common                 import print_banner, print_mesh_info
from modular.problems               import make_linear_elastic_2d_ps_builder
from modular.post                   import XDMFWriter, reaction_form_from_sigma_2d
from modular.solvers                import run_quasistatic


def main():
    comm = MPI.COMM_WORLD

    # Use the dyn-branch material without ρ (plane-stress).
    mat = load_material("Glass_dyn_2D_PS")
    mat = dict(mat); mat.pop("rho", None)       # rho not needed here
    eps, h0, h_min = mat["eps"], mat["h0"], mat["h_min"]
    print_banner(f"Linear elastic 2D-PS — {mat['name']}", comm)

    W, L = 1.0, 1.0
    msh_coarse, markers, geom = make_notched_plate_2d(
        W=W, L=L, ac=W/2, cw=0.001, h0=h0, comm=comm)
    msh_coarse.name = "plate_ps"
    print_mesh_info(msh_coarse, comm)

    build_problem = make_linear_elastic_2d_ps_builder(
        mat=mat, markers_spec=markers, geom=geom,
        eps=eps, h0=h0, h_min=h_min,
    )
    P = build_problem(msh_coarse)

    writer = XDMFWriter("paraview_linear_2d_ps")
    def on_out(P, t, step, dt, extras):
        writer.write(P, t, step)

    run_quasistatic(
        P, build_problem, msh_coarse,
        T_total=1.0, steps=200, max_stag=20, tol_stag=1.0e-7,
        max_disp=0.006,
        on_output=on_out,
        reaction_form=reaction_form_from_sigma_2d("top", component=1),
        log_path="output_linear_2d_ps.txt",
    )


if __name__ == "__main__":
    main()
