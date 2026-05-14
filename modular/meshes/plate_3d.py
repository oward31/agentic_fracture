"""Extruded cracked plate (3D).

2D cracked plate in the x-y plane, extruded in z by `thickness`.
Returns tags 1..6 = right, left, top, bottom, zmin, zmax.
"""

from __future__ import annotations
from typing import Tuple, List, Callable, Dict, Any

import numpy as np
import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI


def make_notched_plate_3d(
    W: float,
    L: float,
    ac: float,
    cw: float,
    thickness: float,
    h0: float,
    comm=MPI.COMM_WORLD,
    rank: int = 0,
) -> Tuple[Any, List[Tuple[int, str, Callable]], Dict[str, float]]:
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Verbosity", 1)

    if comm.rank == 0:
        gmsh.model.add("plate_3d")
        g = gmsh.model.geo
        g.addPoint(0.0, L / 2,   0.0, h0, 1)
        g.addPoint(W,   L / 2,   0.0, h0, 2)
        g.addPoint(W,  -L / 2,   0.0, h0, 3)
        g.addPoint(0.0, -L / 2,  0.0, h0, 4)
        g.addPoint(0.0,  cw / 2, 0.0, h0, 5)
        g.addPoint(0.0, -cw / 2, 0.0, h0, 6)
        g.addPoint(ac,   0.0,    0.0, h0, 7)
        g.addLine(1, 2, 1)
        g.addLine(2, 3, 2)
        g.addLine(3, 4, 3)
        g.addLine(4, 6, 4)
        g.addLine(5, 1, 5)
        g.addLine(5, 7, 6)
        g.addLine(6, 7, 7)
        g.addCurveLoop([1, 2, 3, 4, 7, -6, 5], 10)
        g.addPlaneSurface([10], 11)
        g.synchronize()
        extruded = g.extrude([(2, 11)], 0, 0, thickness)
        g.synchronize()
        # Physical group on the 3D volume so gmshio can find cells.
        vol_tags = [e[1] for e in extruded if e[0] == 3]
        gmsh.model.addPhysicalGroup(3, vol_tags, tag=100)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h0)
        gmsh.model.mesh.generate(3)

    msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=3)
    gmsh.clear()

    tol = 1e-8

    def left(x):   return np.isclose(x[0], 0.0, atol=tol)
    def right(x):  return np.isclose(x[0], W,   atol=W * 1e-6 + tol)
    def top(x):    return np.isclose(x[1],  L / 2, atol=L * 1e-6 + tol)
    def bottom(x): return np.isclose(x[1], -L / 2, atol=L * 1e-6 + tol)
    def zmin(x):   return np.isclose(x[2], 0.0,       atol=tol)
    def zmax(x):   return np.isclose(x[2], thickness, atol=thickness * 1e-6 + tol)

    markers_spec = [
        (1, "right",  right),
        (2, "left",   left),
        (3, "top",    top),
        (4, "bottom", bottom),
        (5, "zmin",   zmin),
        (6, "zmax",   zmax),
    ]
    geom = {"W": W, "L": L, "ac": ac, "cw": cw, "thickness": thickness, "h0": h0}
    return msh, markers_spec, geom
