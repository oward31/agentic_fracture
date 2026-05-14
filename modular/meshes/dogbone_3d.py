"""3D dogbone specimen obtained by extruding the 2D dogbone in z."""

from __future__ import annotations
from typing import Tuple, List, Callable, Dict, Any

import numpy as np
import gmsh
from dolfinx.io import gmshio
from mpi4py import MPI


def make_dogbone_3d(
    W: float,
    L: float,
    R: float,
    thickness: float,
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
        gmsh.model.add("dogbone_3d")
        og = gmsh.model.occ
        rect = og.addRectangle(0.0, 0.0, 0.0, W, L)
        nl = og.addDisk(0.0, L / 2.0, 0.0, R, R)
        nr = og.addDisk(W,    L / 2.0, 0.0, R, R)
        og.cut([(2, rect)], [(2, nl), (2, nr)])
        og.synchronize()
        surfaces = og.getEntities(dim=2)
        extruded = og.extrude(surfaces, 0.0, 0.0, thickness)
        og.synchronize()
        vol_tags = [e[1] for e in extruded if e[0] == 3]
        gmsh.model.addPhysicalGroup(3, vol_tags, tag=100)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h0)
        gmsh.model.mesh.generate(3)

    msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=3)
    gmsh.clear()

    tol = 1e-5

    def bottom(x): return np.isclose(x[1], 0.0, atol=tol)
    def top(x):    return np.isclose(x[1], L,   atol=tol)
    def zmin(x):   return np.isclose(x[2], 0.0,       atol=tol)
    def zmax(x):   return np.isclose(x[2], thickness, atol=tol)

    markers_spec = [
        (1, "bottom", bottom),
        (2, "top",    top),
        (3, "zmin",   zmin),
        (4, "zmax",   zmax),
    ]
    geom = {"W": W, "L": L, "R": R, "thickness": thickness,
            "h0": h0, "h_fine": h_fine}
    return msh, markers_spec, geom
