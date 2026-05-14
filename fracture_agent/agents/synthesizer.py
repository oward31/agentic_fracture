"""Synthesizer — render the runnable Python script from (spec, action).

We lean on templates.render_script for the 9 known variants; for custom
geometries we additionally have mesh.save_custom_mesh_module write a
`custom_mesh.py` sibling file that the rendered script imports.

Region-name contract (added 2026-04):
    Before any mesh code is generated, every region name in the spec
    (``custom_regions``, ``bcs.fixed[].region``, ``bcs.loading[].region``) is
    slugified to a code-safe form.  For custom meshes the LLM is then asked
    to emit ``markers_spec`` names verbatim; we post-validate by AST-parsing
    the generated module and rewriting any drifted names.  For built-in
    meshes we validate the references against the known exposed names.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from ..events import DECISION, ERROR, emit
from ..knowledge import Variant, CATALOG
from ..mesh import MeshSizePlan, save_custom_mesh_module
from ..region_names import (BUILTIN_MESH_REGIONS, enforce_marker_names,
                            extract_marker_names, slugify, validate_bc_regions)
from ..schema import Action, CanonicalSpec
from ..templates import render_script


def _variant_obj(variant_name: str) -> Variant:
    for v in CATALOG:
        if v.variant == variant_name:
            return v
    raise KeyError(variant_name)


def _slugify_spec_regions(spec: CanonicalSpec) -> List[Tuple[str, str]]:
    """Rewrite every region name in ``spec`` to its slug form, in place.

    Returns the list of ``(before, after)`` mappings that actually changed
    (useful for emit/logging).
    """
    renames: List[Tuple[str, str]] = []

    def _do(get_name, set_name):
        old = get_name()
        if not old:
            return
        new = slugify(old)
        if new and new != old:
            set_name(new)
            renames.append((old, new))

    for r in spec.geometry.custom_regions:
        _do(lambda r=r: r.name, lambda v, r=r: setattr(r, "name", v))
    for f in spec.bcs.fixed:
        _do(lambda f=f: f.region, lambda v, f=f: setattr(f, "region", v))
    for l in spec.bcs.loading:
        _do(lambda l=l: l.region, lambda v, l=l: setattr(l, "region", v))
    return renames


def _expected_custom_regions(spec: CanonicalSpec) -> List[dict]:
    """Region whitelist (name + description) for the custom-mesh LLM.

    Order: declared ``custom_regions`` first, then any fresh regions
    appearing only in BCs, then the canonical fall-back.  Names are already
    slugified by ``_slugify_spec_regions``.
    """
    regions: list[dict] = []
    seen: set[str] = set()
    for r in spec.geometry.custom_regions:
        if r.name and r.name not in seen:
            regions.append({"name": r.name,
                            "description": r.description or ""})
            seen.add(r.name)
    for f in spec.bcs.fixed:
        if f.region and f.region not in seen:
            regions.append({"name": f.region, "description": ""})
            seen.add(f.region)
    for l in spec.bcs.loading:
        if l.region and l.region not in seen:
            regions.append({"name": l.region, "description": ""})
            seen.add(l.region)
    if not regions:
        regions = [{"name": n, "description": ""}
                   for n in ("bottom", "top", "left", "right")]
    return regions


def _validate_custom_mesh_module(mesh_path: Path,
                                  expected_names: List[str]) -> List[str]:
    """Read the generated ``custom_mesh.py``, AST-parse the markers_spec
    names, rewrite drift, and write the corrected file back.  Returns a list
    of human-readable warnings.
    """
    code = mesh_path.read_text(encoding="utf-8")
    new_code, warnings = enforce_marker_names(code, expected_names)
    if new_code != code:
        mesh_path.write_text(new_code, encoding="utf-8")
    return warnings


def synthesizer(spec: CanonicalSpec,
                action: Action,
                mesh_plan: MeshSizePlan,
                session_dir: Path) -> Path:
    """Return the path to the generated driver script."""

    # 1. Slugify region names so ALL downstream consumers (mesh-LLM,
    #    template's BC block, _t() lookup) agree on a single code-safe key.
    renames = _slugify_spec_regions(spec)
    if renames:
        emit(DECISION,
             "region names slugified: " +
             ", ".join(f"{o!r}->{n!r}" for o, n in renames))

    # 2. For custom meshes: write the LLM module and enforce verbatim names.
    if action.mesh_builder == "make_custom_gmsh":
        regions = _expected_custom_regions(spec)
        dim = 2 if spec.dimension.value == "2D" else 3
        mesh_path = save_custom_mesh_module(
            session_dir,
            description=spec.geometry.custom_description or "",
            regions=regions, dim=dim)
        # Auto-rewrite if the LLM drifted from the expected names.
        warnings = _validate_custom_mesh_module(
            mesh_path, [r["name"] for r in regions])
        for w in warnings:
            emit(DECISION, f"custom_mesh.py: {w}")

    # 3. End-to-end region validation: every BC region must be in the
    #    mesh's exposed names.  For built-ins: BUILTIN_MESH_REGIONS.  For
    #    custom: parse the just-written file.
    available: set[str]
    if action.mesh_builder == "make_custom_gmsh":
        mesh_path = session_dir / "custom_mesh.py"
        if mesh_path.exists():
            available = set(extract_marker_names(
                mesh_path.read_text(encoding="utf-8")))
        else:
            available = set()
    else:
        available = set(BUILTIN_MESH_REGIONS.get(action.mesh_builder, set()))

    if available:
        ok, issues = validate_bc_regions(spec, available)
        if not ok:
            for i in issues:
                emit(ERROR, f"region-validation: {i}")
            raise ValueError(
                "BC references regions the mesh does not expose.  "
                + " | ".join(issues))

    variant = _variant_obj(action.variant)
    return render_script(spec, action, mesh_plan, session_dir, variant)
