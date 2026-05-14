"""Cached residual norms — avoid re-JITing forms in the staggered loop."""

from __future__ import annotations
import numpy as np
from dolfinx import fem
from mpi4py import MPI


def norm_L2_cached(comm, form) -> float:
    """Global L2 norm from a pre-compiled form."""
    return float(np.sqrt(comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)))
