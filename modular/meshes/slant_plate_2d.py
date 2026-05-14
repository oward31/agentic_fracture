"""Slant-cracked plate (2D plane stress), inclined V-notch at angle theta.

Same topology as the horizontal notched plate, but the pre-crack tip is at
(xtip, ytip) = c0·(cos θ, sin θ), so the crack runs from the left face to an
off-axis point inside the domain.
"""

from __future__ import annotations
from typing import Tuple, List, Callable, Dict, Any

import numpy as np
import gmsh
from dolfinx.io import gmshio
from dolfinx.cpp.mesh import GhostMode
from dolfinx.mesh import create_cell_partitioner
from mpi4py import MPI


def make_slant_plate_2d(
    W: float,
    H: float,
    c0: float,
    theta: float,
    cw: float,
    h0: float,
    h_global: float = 0.25,
    comm=MPI.COMM_WORLD,
    rank: int = 0,
) -> Tuple[Any, List[Tuple[int, str, Callable]], Dict[str, float]]:
    xtip, ytip = c0 * np.cos(theta), c0 * np.sin(theta)
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Verbosity", 1)

    if comm.rank == 0:
        gmsh.model.add("slant_2d")
        g = gmsh.model.geo
        g.addPoint(0.0, H / 2,   0.0, h_global, 1)
        g.addPoint(W,   H / 2,   0.0, h_global, 2)
        g.addPoint(W,  -H / 2,   0.0, h_global, 3)
        g.addPoint(0.0, -H / 2,  0.0, h_global, 4)
        g.addPoint(0.0, -cw / 2, 0.0, h_global, 5)   # lower crack mouth
        g.addPoint(0.0,  cw / 2, 0.0, h_global, 6)   # upper crack mouth
        g.addPoint(xtip, ytip,   0.0, h_global, 7)   # crack tip

        g.addLine(1, 2, 1)
        g.addLine(2, 3, 2)
        g.addLine(3, 4, 3)
        g.addLine(4, 5, 4)
        g.addLine(5, 7, 5)     # lower crack face
        g.addLine(7, 6, 6)     # upper crack face
        g.addLine(6, 1, 7)
        g.addCurveLoop([1, 2, 3, 4, 5, 6, 7], 10)
        g.addPlaneSurface([10], 11)
        g.synchronize()
        gmsh.model.addPhysicalGroup(2, [11], tag=100)

        # Box field refines the crack-tip neighbourhood.
        bb = gmsh.model.mesh.field.add("Box")
        pad = 2.0 * h0
        gmsh.model.mesh.field.setNumber(bb, "VIn",   h0)
        gmsh.model.mesh.field.setNumber(bb, "VOut",  h_global)
        gmsh.model.mesh.field.setNumber(bb, "XMin",  0.0)
        gmsh.model.mesh.field.setNumber(bb, "XMax",  xtip + 5 * h0)
        gmsh.model.mesh.field.setNumber(bb, "YMin", -pad)
        gmsh.model.mesh.field.setNumber(bb, "YMax",  ytip + pad)
        gmsh.model.mesh.field.setAsBackgroundMesh(bb)
        gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay
        gmsh.model.mesh.generate(2)

    partitioner = create_cell_partitioner(GhostMode.shared_facet)
    msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2,
                                      partitioner=partitioner)
    gmsh.clear()

    tol = 1e-5

    def left(x):   return np.isclose(x[0], 0.0, atol=tol)
    def right(x):  return np.isclose(x[0], W,   atol=tol)
    def top(x):    return np.isclose(x[1],  H / 2, atol=tol)
    def bottom(x): return np.isclose(x[1], -H / 2, atol=tol)

    markers_spec = [
        (1, "bottom", bottom),
        (2, "top",    top),
        (3, "left",   left),
        (4, "right",  right),
    ]
    geom = {"W": W, "H": H, "c0": c0, "theta": theta, "cw": cw,
            "xtip": xtip, "ytip": ytip, "h0": h0}
    return msh, markers_spec, geom
