"""Mesh generators.

Each module exposes `generate(...)` → (msh, markers_spec) where
`markers_spec` is a list of `(tag_id, name, locator_fn)` triples. The
locator is a numpy-vectorised `x -> bool` used by
`dolfinx.mesh.locate_entities_boundary`.

The consumer (problem builder) turns `markers_spec` into a `meshtags`
object and into sub-space DOF lists.
"""

from .plate_2d import make_notched_plate_2d
from .plate_3d import make_notched_plate_3d
from .slant_plate_2d import make_slant_plate_2d
from .dogbone_2d import make_dogbone_2d
from .dogbone_3d import make_dogbone_3d
