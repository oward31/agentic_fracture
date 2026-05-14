"""Non-matching interpolation — used when transferring fields after AMR."""

from __future__ import annotations
import numpy as np
from dolfinx import fem
from dolfinx.fem import create_interpolation_data


def non_matching_interp_data(V_new, V_old, cells=None, padding: float = 1e-8):
    """Pre-build the interpolation data once per pair of function spaces."""
    if cells is None:
        msh = V_new.mesh
        n = msh.topology.index_map(msh.topology.dim).size_local
        cells = np.arange(n, dtype=np.int32)
    return create_interpolation_data(V_new, V_old, cells, padding=padding), cells


def transfer_function(f_old, V_new, cells=None, interp_data=None,
                      padding: float = 1e-8) -> fem.Function:
    """Non-matching interpolation of `f_old` into the function space `V_new`."""
    f_new = fem.Function(V_new)
    if cells is None or interp_data is None:
        interp_data, cells = non_matching_interp_data(V_new, f_old.function_space,
                                                      cells=cells, padding=padding)
    f_new.interpolate_nonmatching(f_old, cells, interpolation_data=interp_data)
    f_new.x.scatter_forward()
    return f_new
