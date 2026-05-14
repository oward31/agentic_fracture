#!/usr/bin/env python3
"""Dynamic phase-field fracture, 2D plane stress — traction branching test."""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mpi4py import MPI

from modular.materials              import load_material
from modular.meshes                 import make_notched_plate_2d
from modular.common                 import print_banner, print_mesh_info
from modular.problems               import make_dynamic_2d_builder
from modular.post                   import XDMFWriter
from modular.solvers                import run_dynamic, dynamic_time_step


def main():
    comm = MPI.COMM_WORLD

    mat = load_material("Glass_dyn_2D_PS")
    eps, h0, h_min = mat["eps"], mat["h0"], mat["h_min"]
    print_banner(f"Dynamic 2D-PS — {mat['name']}", comm)

    W, L = 100.0, 40.0
    msh_coarse, markers, geom = make_notched_plate_2d(
        W=W, L=L, ac=W/2, cw=0.1, h0=h0, comm=comm)
    msh_coarse.name = "dyn_plate"
    print_mesh_info(msh_coarse, comm)

    # Characteristic length = plate height → dt = 0.1·L/c_R.
    dt = dynamic_time_step(mat, L_char=L, safety=0.1)
    if comm.rank == 0:
        print(f"Δt (Rayleigh) = {dt:.3e} s")

    build_problem = make_dynamic_2d_builder(
        mat=mat, markers_spec=markers, geom=geom,
        eps=eps, h0=h0, h_min=h_min, dt=dt,
    )
    P = build_problem(msh_coarse)

    writer = XDMFWriter("paraview_dynamic_2d_ps")
    def on_out(P, t, step, dtv, extras):
        writer.write(P, t, step)

    # Linear ramp for 10 dt then hold.
    p_max = 2.0
    def ramp(t):
        return p_max * min(1.0, t / (10.0 * dt))

    run_dynamic(
        P, build_problem, msh_coarse,
        T_total=80.0e-6, dt=dt,
        max_stag=20, tol_stag=1.0e-7,
        pressure_ramp=ramp,
        on_output=on_out,
        log_path="output_dynamic_2d_ps.txt",
    )


if __name__ == "__main__":
    main()
