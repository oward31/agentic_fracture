"""Region-name canonicalisation and validation.

The agent has THREE places that must agree on boundary-region names:

  * Architect's CanonicalSpec (``geometry.custom_regions[].name`` and
    ``bcs.{fixed,loading}[].region``).
  * The mesh: either a built-in builder in ``modular/meshes/`` (fixed names)
    or an LLM-written ``custom_mesh.py`` (LLM-chosen names).
  * The generated driver script (``_t(name)`` lookup against ``markers_spec``).

If those three diverge, the driver crashes at runtime with KeyError.  This
module provides:

* ``slugify(name)``         — turn any free-text name into a code-safe slug.
* ``BUILTIN_MESH_REGIONS``  — the canonical names each built-in builder
                              exposes.
* ``extract_marker_names``  — AST-parse a generated ``custom_mesh.py`` and
                              return the names actually used in
                              ``markers_spec``.
* ``enforce_marker_names``  — rewrite a generated ``custom_mesh.py`` so its
                              ``markers_spec`` names match an expected list
                              (1-to-1 by position).
* ``validate_bc_regions``   — check every BC region in a CanonicalSpec
                              exists in the mesh's exposed names; return a
                              structured report.

All functions are pure (no I/O) so they're easy to unit-test.
"""
from __future__ import annotations
import ast
import re
from typing import Iterable, List, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Slug-ify free-text names into code-safe identifiers.
# ---------------------------------------------------------------------------
_SLUG_NON_WORD = re.compile(r"[^a-zA-Z0-9_]+")
_SLUG_REPEATED = re.compile(r"_+")


def slugify(name: str) -> str:
    """Return a lower-case ``[a-z0-9_]`` slug of ``name``.

    Spaces, dashes and other punctuation collapse to a single underscore;
    repeated underscores collapse; leading/trailing underscores are stripped.

    >>> slugify("bottom edge")
    'bottom_edge'
    >>> slugify("top half of left edge")
    'top_half_of_left_edge'
    >>> slugify("Notched_PLATE-2D")
    'notched_plate_2d'
    >>> slugify("  ___weird  --  thing___")
    'weird_thing'
    >>> slugify("")
    ''
    """
    if not name:
        return ""
    s = _SLUG_NON_WORD.sub("_", str(name))
    s = _SLUG_REPEATED.sub("_", s)
    return s.strip("_").lower()


# ---------------------------------------------------------------------------
# 2. The contract for built-in meshes — what names they expose.
#    Keep in lockstep with modular/meshes/*.py.
# ---------------------------------------------------------------------------
BUILTIN_MESH_REGIONS: dict[str, set[str]] = {
    "make_notched_plate_2d": {"top", "bottom", "left", "right"},
    "make_notched_plate_3d": {"top", "bottom", "left", "right", "zmin", "zmax"},
    "make_slant_plate_2d":   {"top", "bottom", "left", "right"},
    "make_dogbone_2d":       {"top", "bottom"},
    "make_dogbone_3d":       {"top", "bottom", "zmin", "zmax"},
}


def builtin_regions(mesh_builder: str) -> set[str]:
    """Names exposed by a built-in mesh builder, or empty for custom."""
    return BUILTIN_MESH_REGIONS.get(mesh_builder, set())


# ---------------------------------------------------------------------------
# 3. Parse a generated ``custom_mesh.py`` to discover marker names.
# ---------------------------------------------------------------------------
def extract_marker_names(code: str) -> List[str]:
    """Return the string names from ``markers_spec = [(tag, name, fn), ...]``
    in source order.  Returns ``[]`` if the file does not parse, has no
    ``markers_spec`` assignment, or the entries are not parseable.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
        if not any(t.id == "markers_spec" for t in targets):
            continue
        if not isinstance(node.value, ast.List):
            return []
        names: List[str] = []
        for elt in node.value.elts:
            if not isinstance(elt, (ast.Tuple, ast.List)):
                return []
            if len(elt.elts) < 2:
                return []
            name_node = elt.elts[1]
            if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
                names.append(name_node.value)
            else:
                # Non-literal name — give up; caller will warn.
                return []
        return names
    return []


# ---------------------------------------------------------------------------
# 4. Rewrite the generated module to use expected names.
# ---------------------------------------------------------------------------
def enforce_marker_names(code: str,
                          expected_names: List[str]) -> Tuple[str, List[str]]:
    """If ``code``'s ``markers_spec`` names differ from ``expected_names``
    (1-to-1 by position), rewrite the file to use ``expected_names``.

    Replacement is *textual* on the string-literal names so all references
    (markers_spec entry, gmsh setPhysicalName, comments) stay in sync.

    Returns ``(new_code, warnings)``.

    Notes:
      * If the count of marker entries differs from ``len(expected_names)``,
        we don't rewrite — caller should re-prompt the LLM.
      * If ``markers_spec`` cannot be parsed, we don't rewrite either.
    """
    warnings: List[str] = []
    actual = extract_marker_names(code)
    if not actual:
        warnings.append("could not parse `markers_spec` from custom_mesh.py — "
                        "names not validated")
        return code, warnings
    if len(actual) != len(expected_names):
        warnings.append(
            f"custom_mesh.py emitted {len(actual)} marker entries "
            f"({actual!r}); architect expected {len(expected_names)} "
            f"({expected_names!r}) — names not validated, region-lookup may fail")
        return code, warnings
    if actual == expected_names:
        return code, warnings  # no drift — happy path
    out = code
    seen_renames: List[Tuple[str, str]] = []
    for old, new in zip(actual, expected_names):
        if old == new:
            continue
        # Only replace quoted occurrences so we don't accidentally mangle
        # variable identifiers that happen to share the name.
        out = out.replace(f'"{old}"', f'"{new}"')
        out = out.replace(f"'{old}'", f"'{new}'")
        seen_renames.append((old, new))
    if seen_renames:
        renames = ", ".join(f"'{o}' -> '{n}'" for o, n in seen_renames)
        warnings.append(f"renamed mesh markers to match architect: {renames}")
    return out, warnings


# ---------------------------------------------------------------------------
# 5. Edge geometry — which axis a region lives on, and which sign points
#    "outward" (away from the body interior).
# ---------------------------------------------------------------------------
# axis index → 0 (x), 1 (y), 2 (z); outward sign → ±1.
# For half-edge / quarter-edge names the ENTIRE edge determines the axis:
#   - "top_left_half"          → "top"  edge   → y, +1
#   - "top_half_of_left_edge"  → "left" edge   → x, -1
# These two contradict if you just look at tokens, so we resolve in two
# passes: (1) prefer ``<edge>_edge`` substrings (e.g. ``left_edge``) since
# they unambiguously name the edge being subdivided; (2) otherwise pick
# the FIRST edge token in the slug.
_EDGE_AXIS_OUTWARD: dict[str, Tuple[int, int]] = {
    "left":   (0, -1),
    "right":  (0, +1),
    "bottom": (1, -1),
    "top":    (1, +1),
    "front":  (2, -1),
    "back":   (2, +1),
    "zmin":   (2, -1),
    "zmax":   (2, +1),
    "ymin":   (1, -1),
    "ymax":   (1, +1),
    "xmin":   (0, -1),
    "xmax":   (0, +1),
}


_EQ_PAT = re.compile(r"\b([xyz])\s*={1,3}\s*(-?\d+(?:\.\d+)?)")


def edge_from_description(desc: str,
                           dimensions: dict | None = None) -> Tuple[int, int] | None:
    """Parse an architect-emitted region description like ``'x == 0 AND y >= 10'``
    to figure out which rectangular edge the region lies on.

    Returns ``(axis, outward_sign)`` where axis is 0/1/2 for x/y/z and the
    outward sign is the direction pointing away from the body interior.
    Returns ``None`` if the description doesn't unambiguously fix one
    coordinate to a min or max of the bounding box.

    >>> edge_from_description("x == 0 AND y >= 10", {"W": 10.0, "L": 20.0})
    (0, -1)
    >>> edge_from_description("y == 25", {"W": 25.0, "L": 25.0})
    (1, 1)
    >>> edge_from_description("y >= 10", {}) is None
    True
    """
    if not desc:
        return None
    d = desc.lower()
    matches = _EQ_PAT.findall(d)
    if not matches:
        return None
    dims = dimensions or {}
    # Heuristic max-coord lookup: try common synonyms.
    max_x = float(dims.get("W") or dims.get("width") or dims.get("L_outer")
                  or dims.get("Lx") or 0.0)
    max_y = float(dims.get("L") or dims.get("H") or dims.get("height")
                  or dims.get("Ly") or dims.get("L_outer") or 0.0)
    max_z = float(dims.get("thickness") or dims.get("D") or dims.get("depth")
                  or dims.get("Lz") or 0.0)
    axis_max = {0: max_x, 1: max_y, 2: max_z}
    for var, val_str in matches:
        axis = {"x": 0, "y": 1, "z": 2}[var]
        try:
            val = float(val_str)
        except ValueError:
            continue
        if abs(val) < 1e-6:
            return (axis, -1)
        cap = axis_max.get(axis, 0.0)
        if cap > 0 and abs(val - cap) < max(1e-3, 0.05 * cap):
            return (axis, +1)
    return None


def edge_axis_outward_for_region(region_name: str,
                                  description: str = "",
                                  dimensions: dict | None = None
                                  ) -> Tuple[int, int] | None:
    """Resolve a region's edge axis & outward direction.

    Tries (in order):
      1. ``edge_from_description`` — the most reliable for custom regions
         since the architect emits explicit coordinate equations.
      2. ``edge_axis_outward_sign`` — falls back to slug-name parsing,
         which handles built-in mesh names (``left`` / ``right`` /
         ``top`` / ``bottom``) and many half-edge slugs.
    """
    info = edge_from_description(description, dimensions or {})
    if info is not None:
        return info
    return edge_axis_outward_sign(region_name)


def edge_axis_outward_sign(region: str) -> Tuple[int, int] | None:
    """Return ``(axis, outward_sign)`` for a slug region name, or ``None``
    if the region doesn't unambiguously sit on a single rectangular edge.

    >>> edge_axis_outward_sign("top")
    (1, 1)
    >>> edge_axis_outward_sign("left")
    (0, -1)
    >>> edge_axis_outward_sign("top_half_of_left_edge")
    (0, -1)
    >>> edge_axis_outward_sign("top_left_half")
    (1, 1)
    >>> edge_axis_outward_sign("rigid_body_pin") is None
    True
    """
    if not region:
        return None
    s = region.lower()
    # Direct match.
    if s in _EDGE_AXIS_OUTWARD:
        return _EDGE_AXIS_OUTWARD[s]
    # Pass 1: explicit "<edge>_edge" or "<edge>edge" substring.
    for edge in ("left", "right", "top", "bottom", "front", "back"):
        if (f"{edge}_edge" in s) or s.endswith(f"_{edge}edge"):
            return _EDGE_AXIS_OUTWARD[edge]
    # Pass 2: first edge token in the slug.
    for tok in s.split("_"):
        if tok in _EDGE_AXIS_OUTWARD:
            return _EDGE_AXIS_OUTWARD[tok]
    return None


# ---------------------------------------------------------------------------
# 6. End-to-end region validation.
# ---------------------------------------------------------------------------
def collect_referenced_regions(spec) -> Set[str]:
    """Names referenced anywhere in a CanonicalSpec's BCs.

    Accepts the live Pydantic spec object — duck-typed to avoid an import
    cycle with ``fracture_agent.schema``.
    """
    used: Set[str] = set()
    for f in getattr(spec.bcs, "fixed", []) or []:
        if getattr(f, "region", None):
            used.add(f.region)
    for l in getattr(spec.bcs, "loading", []) or []:
        if getattr(l, "region", None):
            used.add(l.region)
    return used


def validate_bc_regions(spec, available_names: Iterable[str]) -> Tuple[bool, List[str]]:
    """Return ``(ok, issues)`` where ``ok`` is True iff every BC-referenced
    region is in ``available_names``.  ``issues`` is a list of human-readable
    descriptions of each missing reference, plus the available set.
    """
    used = collect_referenced_regions(spec)
    avail = set(available_names)
    missing = sorted(used - avail)
    if not missing:
        return True, []
    issues = [f"BC region {m!r} is not exposed by the mesh "
              f"(available: {sorted(avail)})"
              for m in missing]
    return False, issues
