"""Knowledge base for the 9 modular problem variants — the RAG primitive.

Retrieval is metadata-filtered (not embedding-similarity) because the search
space is finite and perfectly labelled.  The Strategist simply looks for the
unique row whose metadata matches (dimension, plane_type, constitutive, mode).
Embedding-based fallback exists for "none of the above" queries and for
natural-language variant descriptions from the user.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import MODULAR


# ---------------------------------------------------------------------------
# Variant catalog — canonical facts about every modular skeleton.
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    variant: str                                         # builder key name
    dim: int                                             # 2 or 3
    plane: str                                           # plane_strain / plane_stress / na
    constitutive: str                                    # linear_elastic / j2_plasticity / lopez_pamies
    loading: str                                         # quasistatic / dynamic
    mesh_builder: str
    solver: str
    example_script: str                                  # modular/examples/*.py
    default_material: str                                # materials.json key
    summary: str                                         # short natural-language description


CATALOG: List[Variant] = [
    Variant("linear_elastic_2d_pe", 2, "plane_strain", "linear_elastic", "quasistatic",
            "make_notched_plate_2d", "run_quasistatic",
            "examples/linear_elastic_2d_pe.py", "Steel_bench_2D_PE",
            "Linear elastic phase-field fracture in 2D plane strain on a notched plate."),
    Variant("linear_elastic_2d_ps", 2, "plane_stress", "linear_elastic", "quasistatic",
            "make_notched_plate_2d", "run_quasistatic",
            "examples/linear_elastic_2d_ps.py", "Steel_bench_2D_PE",
            "Linear elastic phase-field fracture in 2D plane stress on a notched plate."),
    Variant("linear_elastic_3d", 3, "na", "linear_elastic", "quasistatic",
            "make_notched_plate_3d", "run_quasistatic",
            "examples/linear_elastic_3d.py", "Ceramic_surfing_3D",
            "Linear elastic phase-field fracture in 3D."),
    Variant("dynamic_2d", 2, "plane_stress", "linear_elastic", "dynamic",
            "make_notched_plate_2d", "run_dynamic",
            "examples/dynamic_2d_ps.py", "Glass_dyn_2D_PS",
            "Dynamic (HHT-alpha) phase-field brittle fracture, 2D plane stress."),
    Variant("ductile_2d_pe", 2, "plane_strain", "j2_plasticity", "quasistatic",
            "make_dogbone_2d", "run_ductile",
            "examples/ductile_2d_pe.py", "Al_ductile_2D_PE",
            "J2 plasticity + phase-field fracture in 2D plane strain (dogbone)."),
    Variant("ductile_3d", 3, "na", "j2_plasticity", "quasistatic",
            "make_dogbone_3d", "run_ductile",
            "examples/ductile_3d.py", "Al_ductile_3D",
            "J2 plasticity + phase-field fracture in 3D (extruded dogbone)."),
    Variant("finite_elastic_2d_ps", 2, "plane_stress", "lopez_pamies", "quasistatic",
            "make_slant_plate_2d", "run_finite_elasticity",
            "examples/finite_elastic_2d_ps.py", "Rubber_LopezPamies_2D_PS",
            "Finite-strain Lopez-Pamies rubber phase-field fracture, 2D plane stress."),
    Variant("finite_elastic_2d_pe", 2, "plane_strain", "lopez_pamies", "quasistatic",
            "make_notched_plate_2d", "run_finite_elasticity",
            "examples/finite_elastic_2d_pe.py", "Rubber_LopezPamies_2D_PE",
            "Finite-strain Lopez-Pamies rubber phase-field fracture, 2D plane strain."),
    Variant("finite_elastic_3d", 3, "na", "lopez_pamies", "quasistatic",
            "make_notched_plate_3d", "run_finite_elasticity",
            "examples/finite_elastic_3d.py", "Rubber_LopezPamies_3D",
            "Finite-strain Lopez-Pamies rubber phase-field fracture, 3D."),
]


# ---------------------------------------------------------------------------
# Metadata-first retrieval
# ---------------------------------------------------------------------------
def retrieve(dim: int,
             plane: str,
             constitutive: str,
             loading: str) -> Optional[Variant]:
    """Exact metadata match; returns None if nothing fits."""
    for v in CATALOG:
        if (v.dim == dim
            and v.constitutive == constitutive
            and v.loading == loading
            and (v.plane == plane or v.plane == "na")):
            return v
    return None


def list_variants() -> List[Dict]:
    return [v.__dict__ for v in CATALOG]


# ---------------------------------------------------------------------------
# Materials — thin wrapper around materials.json for the Architect.
# ---------------------------------------------------------------------------
def list_builtin_materials() -> Dict[str, Dict]:
    with open(MODULAR / "materials" / "materials.json") as fh:
        db = json.load(fh)
    return {k: v for k, v in db.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Source-snippet fetcher — gives the Synthesizer exact code to crib from.
# ---------------------------------------------------------------------------
def example_source(variant: str, max_chars: int = 4000) -> str:
    for v in CATALOG:
        if v.variant == variant:
            path = MODULAR / v.example_script
            txt = path.read_text(encoding="utf-8")
            return txt[:max_chars]
    return ""
