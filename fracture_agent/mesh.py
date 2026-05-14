"""Mesh utilities.

Two jobs:
1. ``plan_mesh_size(mat, dim, trial_cells)`` — implement the user's rescaling
   rule:   start with h0 = 2·eps;  if n_cells < target_min, rescale
           ``eps_new = (n_cells / target_min)**dim * eps_old`` and rebuild.
2. ``custom_gmsh_snippet(desc, bcs_regions)`` — use the LLM to produce a
   self-contained `make_custom_gmsh(...)` builder that returns
   ``(msh, markers_spec, geom)`` in the same format as the modular/meshes/
   modules, so the existing problem builders accept it unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import FAST_MODEL
from .llm import llm


@dataclass
class MeshSizePlan:
    eps: float
    h0: float
    h_min: float
    n_cells_expected: int
    n_rescales: int = 0
    notes: str = ""


def rescale_eps(old_eps: float, n_cells: int, target_min: int, dim: int) -> float:
    """User's rescale rule:

        eps_new = (n_cells / target_min) ** (1 / dim) * eps_old

    Physically: n_cells ~ 1/h^dim ~ 1/eps^dim, so to raise the cell count
    from n -> target_min we scale eps by the dim-th root of the ratio.
    """
    if n_cells >= target_min or n_cells <= 0:
        return old_eps
    return old_eps * (n_cells / target_min) ** (1.0 / dim)


def initial_plan(mat_eps: float, dim: int) -> MeshSizePlan:
    """h0 = 2·eps (same convention as modular/materials/loader.py)."""
    eps = float(mat_eps)
    h0  = 2.0 * eps
    return MeshSizePlan(eps=eps, h0=h0, h_min=h0 / 8.0, n_cells_expected=0,
                        notes="initial: h0 = 2·eps from material loader")


# ---------------------------------------------------------------------------
# Custom gmsh generation — free-form geometries.
# ---------------------------------------------------------------------------
CUSTOM_MESH_SYSTEM = """You are a FEniCSx + gmsh mesh-generation expert.

You receive:
  * a geometry description (may include dimensions, notches, holes, L-shape,
    dogbone, cylinder, etc.);
  * a list of named boundary regions the caller needs, each with a free-text
    description such as "y == 0 (bottom edge)", "x == 0 and y >= L/2 (upper
    half of left edge)", "inner horizontal edge of the L".

Emit a SELF-CONTAINED Python module with a single public function:

    def make_custom_gmsh(h0: float, comm=MPI.COMM_WORLD, rank: int = 0):
        ...

REQUIREMENTS

1. Use `gmsh` (already installed).  `if not gmsh.isInitialized(): gmsh.initialize()`.
2. Build the geometry ONLY on rank 0 ( `if comm.rank == 0:` ) exactly like
   modular/meshes/plate_2d.py does.  After building, call `gmsh.model.mesh.generate(dim)`.
3. Distribute via `msh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=<2 or 3>)`.
4. Use `gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h0)` and
   `CharacteristicLengthMax` with the same h0.
5. Add ONE physical group for the full body of dimension = gdim (any tag).
6. Return `(msh, markers_spec, geom)` where:
     * `msh` — dolfinx Mesh
     * `markers_spec` — list of `(tag:int, name:str, locator_fn)` tuples, one
       per REQUESTED region, in the same order the caller listed them.
       **Use the REQUESTED names verbatim** as the `name:str` field — do not
       rename, "clean up", capitalise, or transliterate them (e.g. if the
       caller asks for `bottom_edge`, the markers_spec entry MUST be
       `(<tag>, "bottom_edge", <locator>)`, NOT `"bottom-edge"` /
       `"BottomEdge"` / `"bottom edge"` / `"bottom"`).  The driver script
       looks the names up by exact string match, so any drift is a runtime
       error.  This applies even if the names contain underscores or are
       not what you would have chosen — the caller has good reasons for
       its choice.  Mirror them in `gmsh.model.setPhysicalName(...)` calls
       too if you use any.
       `locator_fn(x)` is a numpy-vectorised boolean predicate: x is a (gdim,
       N) array; return a length-N bool array.
     * `geom` — dict with the named dimensions you used (W, L, R, etc.).
7. Use `gmsh.clear()` at the very end (after the distribute call).  **Do
   NOT call `gmsh.finalize()`** — the caller may rebuild the mesh several
   times (AMR, mesh-rescale loop) and finalising would break subsequent
   builds.

GEOMETRY TIPS

  * Use the OCC kernel (`gmsh.model.occ`) for curves / circles / holes /
    fillets.  Synchronize with `gmsh.model.occ.synchronize()` before mesh
    generation.
  * L-shape / non-convex polygons: build the vertex chain with
    `gmsh.model.geo.addPoint` + `addLine` + `addCurveLoop` + `addPlaneSurface`
    (this mirrors plate_2d.py exactly).
  * Annulus / O-ring / washer: build two concentric disks with OCC
    (``addDisk``) and use ``gmsh.model.occ.cut`` to subtract the inner from
    the outer — the result is the annular surface.  For a 3D O-ring use
    ``addTorus``.
  * Cruciform / plus / cross specimen: build as the boolean union of two
    perpendicular rectangles (``gmsh.model.occ.addRectangle`` +
    ``fuse``), centred at the origin.
  * Disc / Brazilian / cylinder cross-section: ``gmsh.model.occ.addDisk``.
    For the Brazilian test, ensure the two contact points (top and
    bottom) are marked — they are isolated vertices on the circle, so
    locators should use a wide-enough tolerance to catch the nearest
    boundary nodes.
  * Dogbone: radiused transition from grip to gauge is best via a
    polygonal chain with closely-spaced points on the arc, or via
    ``addBSpline`` / ``addCircleArc`` through three points.  If all you
    want is a tensile coupon, build a straight-sided dogbone with four
    straight segments and two small arcs at the shoulders.
  * Three-point-bend beam: plain rectangle is sufficient — mark the two
    support points (vertices on the bottom edge) and the loading point
    (vertex on the top edge) as distinct regions with point-locators.

FEATURES TO LOOK FOR IN THE DESCRIPTION (read carefully, do NOT skip any)

  Before you write the gmsh code, scan the geometry description for these
  features and IMPLEMENT every one.  Do not silently drop them.

  * Pre-crack / notch entering the domain from an edge.  Phrases:
      "crack from <edge> to <location>", "pre-crack", "notch of length a",
      "edge crack", "slit".
    Implementation: split the affected edge into two segments that meet at
    the crack mouth with a tiny opening ``cw`` (≈ 1e-3 × characteristic
    length or smaller).  From the crack mouth draw two line segments to the
    crack tip inside the domain — the two segments lie on the SAME y (or x)
    but are separated by the tiny cw gap.  Include the tip as a distinct
    point.  This matches the pattern in modular/meshes/plate_2d.py.
    Example vocab → geometry:
      "crack from the top center edge to the center (length L/2)":
         - Split the top edge at x = ±cw/2 around x=0.
         - Two horizontal lines from those split points run DOWN to the
           crack tip at (0, y_mid).
         - Curve loop walks the boundary + DOWN the right crack face + across
           to the left crack face + UP back to the top.
      "edge crack of length a from the left edge at y=0":
         - Mirror of plate_2d.py: split the left edge at y = ±cw/2.
         - Two lines RIGHT to the tip at (a, 0).
  * Hole / cutout.  Use OCC boolean subtract (``gmsh.model.occ.cut``).
  * Fillet / chamfer.  Use OCC ``fillet`` / ``chamfer``.
  * Inclusion / second material region.  Subdivide the surface with lines
    meeting inside the body; add separate physical groups.
  * Symmetric / non-symmetric splits of an edge for BCs.  Insert a split
    point on that edge and expose two separate line segments.

LOCATOR PREDICATES
  * Use `np.isclose(x[0], V, atol=tol)` with `tol = 1e-6 * L + 1e-8` for
    coordinate comparisons.
  * For half-edges use logical AND: `np.isclose(x[0], 0.0, atol=tol) &
    (x[1] >= yCut - tol)`.
  * For curved edges use the parametric / normal-distance form.
  * NEVER use a tolerance of 0 and NEVER compare floats with `==`.

IMPORTS (exactly these, nothing more)
    import numpy as np
    import gmsh
    from dolfinx.io import gmshio
    from mpi4py import MPI

Respond with ONLY valid Python code — no backticks, no commentary.
"""


def custom_gmsh_module(description: str,
                       regions: List[Dict[str, str]],
                       dim: int = 2) -> str:
    """Ask the LLM to write a `make_custom_gmsh` module for the requested shape.

    ``regions`` is a list of dicts with keys ``name`` and ``description`` —
    both provided by the Architect.
    """
    region_block = "\n".join(
        f"  - {r['name']}: {r.get('description', '') or '(no description)'}"
        for r in regions
    ) or "  (none — caller expects no named boundary regions)"

    # Snippet RAG (B1, Index 2): surface up to 3 closest existing mesh
    # builders / locator patterns from ``modular/meshes/`` so the LLM
    # composes from verified code instead of inventing.  Best-effort —
    # never block the mesh build on RAG hiccups.
    rag_block = ""
    try:
        from .rag import retrieve_snippets
        retr = retrieve_snippets(description or "rectangular plate",
                                  k=5, auto_build=False)
        retr = [r for r in retr
                if "meshes" in (r.metadata.get("path") or "").replace("\\", "/")
                or r.metadata.get("name", "").startswith("make_")]
        if retr:
            rag_block = (
                "\n--- existing modular/meshes/ patterns "
                "(reference only — compose, do not authoritatively copy) ---\n")
            for r in retr[:3]:
                rag_block += (
                    f"\n# {r.id}  (similarity {r.score:.3f})\n"
                    f"{r.text[:1500]}\n")
    except Exception:
        rag_block = ""

    prompt = (
        f"Dimension: {dim}D\n"
        f"\nRequired boundary regions (emit markers_spec in this order):\n"
        f"{region_block}\n"
        f"\nGeometry description:\n{description}\n"
        + rag_block
        + "\nWrite the full module now."
    )
    from .telemetry import llm_agent
    with llm_agent("mesh_llm"):
        code = llm().complete(CUSTOM_MESH_SYSTEM, prompt,
                              temperature=0.1,
                              max_output_tokens=32768,
                              thinking_budget=-1,
                              model=FAST_MODEL)
    # Strip accidental code fences.
    s = code.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        s = s.strip()
    return s


def save_custom_mesh_module(session_dir: Path,
                            description: str,
                            regions: List[Dict[str, str]],
                            dim: int) -> Path:
    """Generate and write the mesh module, return its on-disk path."""
    code = custom_gmsh_module(description, regions, dim)
    out = session_dir / "custom_mesh.py"
    out.write_text(code, encoding="utf-8")
    return out
