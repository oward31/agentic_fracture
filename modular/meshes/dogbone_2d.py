"""Ductile dogbone specimen (2D plane strain).

Rectangular coupon [0, W] × [0, L] with two circular notches cut symmetrically
from the left and right faces at y = L/2. The original ductile code used a
diagonal band of finer mesh — we keep the same pattern.
"""

from __future__ import annotations
from typing import Tuple, List, Callable, Dict, Any

import numpy as np
import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI


def make_dogbone_2d(
    W: float,
    L: float,
    R: float,
    h0: float,
    h_fine: float | None = None,
    comm=MPI.COMM_WORLD,
    rank: int = 0,
) -> Tuple[Any, List[Tuple[int, str, Callable]], Dict[str, float]]:
    if h_fine is None:
        h_fine = 0.5 * h0

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Verbosity", 1)

    if comm.rank == 0:
        gmsh.model.add("dogbone_2d")
        og = gmsh.model.occ
        # Base rectangle.
        rect = og.addRectangle(0.0, 0.0, 0.0, W, L)
        # Two notches centred at (0, L/2) and (W, L/2).
        nl = og.addDisk(0.0, L / 2.0, 0.0, R, R)
        nr = og.addDisk(W,    L / 2.0, 0.0, R, R)
        og.cut([(2, rect)], [(2, nl), (2, nr)])
        og.synchronize()

        surfaces = gmsh.model.occ.getEntities(dim=2)
        gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], 100)

        # Background size: use a Box field for a finer strip near the
        # notch mid-plane (y = L/2).  Simpler than a MathEval ternary and
        # portable across GMSH versions.
        f1 = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f1, "VIn",  h_fine)
        gmsh.model.mesh.field.setNumber(f1, "VOut", h0)
        gmsh.model.mesh.field.setNumber(f1, "XMin", -1)
        gmsh.model.mesh.field.setNumber(f1, "XMax",  W + 1)
        gmsh.model.mesh.field.setNumber(f1, "YMin",  L / 2.0 - 5.0)
        gmsh.model.mesh.field.setNumber(f1, "YMax",  L / 2.0 + 5.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(f1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        gmsh.model.mesh.generate(2)

    msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
    gmsh.clear()

    tol = 1e-5

    def bottom(x): return np.isclose(x[1], 0.0, atol=tol)
    def top(x):    return np.isclose(x[1], L,   atol=tol)

    markers_spec = [(1, "bottom", bottom), (2, "top", top)]
    geom = {"W": W, "L": L, "R": R, "h0": h0, "h_fine": h_fine}
    return msh, markers_spec, geom
