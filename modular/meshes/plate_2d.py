"""Cracked rectangular plate (2D).

Domain: [0, W] × [-L/2, L/2] with a horizontal pre-crack of length ac from
the left face at y = 0, opened by a crack-mouth width cw.

Returns
-------
msh             : dolfinx.mesh.Mesh
markers_spec    : [(tag, name, locator_fn), ...]
                  "right" = 1, "left" = 2, "top" = 3, "bottom" = 4
geom            : dict with W, L, ac, cw for downstream consumers
"""

from __future__ import annotations
from typing import Tuple, List, Callable, Dict, Any

import numpy as np
import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI


def make_notched_plate_2d(
    W: float,
    L: float,
    ac: float,
    cw: float,
    h0: float,
    comm=MPI.COMM_WORLD,
    rank: int = 0,
) -> Tuple[Any, List[Tuple[int, str, Callable]], Dict[str, float]]:
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Verbosity", 1)

    if comm.rank == 0:
        gmsh.model.add("plate_2d")
        g = gmsh.model.geo
        # 4 corners + 2 crack mouth + 1 crack tip.
        g.addPoint(0.0, L / 2,   0.0, h0, 1)
        g.addPoint(W,   L / 2,   0.0, h0, 2)
        g.addPoint(W,  -L / 2,   0.0, h0, 3)
        g.addPoint(0.0, -L / 2,  0.0, h0, 4)
        g.addPoint(0.0,  cw / 2, 0.0, h0, 5)
        g.addPoint(0.0, -cw / 2, 0.0, h0, 6)
        g.addPoint(ac,   0.0,    0.0, h0, 7)
        g.addLine(1, 2, 1)              # top
        g.addLine(2, 3, 2)              # right
        g.addLine(3, 4, 3)              # bottom
        g.addLine(4, 6, 4)              # left-below-crack
        g.addLine(5, 1, 5)              # left-above-crack
        g.addLine(5, 7, 6)              # upper crack face
        g.addLine(6, 7, 7)              # lower crack face
        g.addCurveLoop([1, 2, 3, 4, 7, -6, 5], 10)
        g.addPlaneSurface([10], 11)
        g.synchronize()
        gmsh.model.addPhysicalGroup(2, [11], tag=100)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h0)
        gmsh.model.mesh.generate(2)

    msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
    gmsh.clear()

    tol = 1e-8

    def left(x):   return np.isclose(x[0], 0.0, atol=tol)
    def right(x):  return np.isclose(x[0], W,   atol=W * 1e-6 + tol)
    def top(x):    return np.isclose(x[1],  L / 2, atol=L * 1e-6 + tol)
    def bottom(x): return np.isclose(x[1], -L / 2, atol=L * 1e-6 + tol)

    markers_spec = [
        (1, "right",  right),
        (2, "left",   left),
        (3, "top",    top),
        (4, "bottom", bottom),
    ]
    geom = {"W": W, "L": L, "ac": ac, "cw": cw, "h0": h0}
    return msh, markers_spec, geom
