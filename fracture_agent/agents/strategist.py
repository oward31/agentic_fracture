"""Strategist — picks the concrete modular variant (the 'Action').

Metadata retrieval is done locally (no LLM needed) because the 9 variants
form a small decision tree.  The LLM is consulted only to (a) explain the
chosen variant in the rationale and (b) pick among ambiguous cases (e.g.
finite_elastic_2d_pe vs _ps for a user who said "rubber").
"""
from __future__ import annotations
from typing import Optional

from ..knowledge import CATALOG, Variant, retrieve
from ..llm import llm
from ..schema import Action, CanonicalSpec, Dimension, Loading


def _metadata_pick(spec: CanonicalSpec) -> Optional[Variant]:
    dim = 2 if spec.dimension == Dimension.two else 3
    mode = spec.loading.mode.value
    return retrieve(dim=dim, plane=spec.plane_type.value,
                    constitutive=spec.constitutive.value, loading=mode)


def _nearest_variant(spec: CanonicalSpec) -> tuple[Optional[Variant], str]:
    """Graceful fallback when the exact (dim, plane, const, mode) tuple has
    no entry in CATALOG.  We try cheap substitutions first.  Returns
    (variant, fallback_note) — variant may be None if even the fallbacks
    fail, in which case caller should raise.
    """
    dim = 2 if spec.dimension == Dimension.two else 3
    mode = spec.loading.mode.value
    const = spec.constitutive.value
    plane = spec.plane_type.value

    # Substitution priority:
    # 1. Flip plane_stress <-> plane_strain.
    # 2. Drop to linear_elastic (lose plasticity / hyperelasticity).
    # 3. 2D <-> 3D swap.
    alt_planes = ("plane_stress", "plane_strain", "na")
    for alt_plane in (plane,) + alt_planes:
        v = retrieve(dim=dim, plane=alt_plane, constitutive=const, loading=mode)
        if v is not None:
            note = "" if alt_plane == plane else (
                f"Note: switched plane type {plane} -> {alt_plane} "
                f"because no variant exists for the requested combination.")
            return v, note
    # Drop to linear_elastic.
    for alt_plane in alt_planes:
        v = retrieve(dim=dim, plane=alt_plane,
                     constitutive="linear_elastic", loading=mode)
        if v is not None:
            return v, (f"Note: downgraded constitutive {const} -> linear_elastic "
                       f"(no variant exists for {const} in {dim}D {alt_plane}).")
    # 2D↔3D swap.
    other_dim = 3 if dim == 2 else 2
    for alt_plane in alt_planes:
        v = retrieve(dim=other_dim, plane=alt_plane,
                     constitutive="linear_elastic", loading=mode)
        if v is not None:
            return v, (f"Note: switched dimension {dim}D -> {other_dim}D "
                       f"and downgraded to linear_elastic.")
    return None, ""


_BUILTIN_KIND_TO_BUILDER = {
    "plate_3d":        "make_notched_plate_3d",
    "slant_plate":     "make_slant_plate_2d",
    "dogbone_2d":      "make_dogbone_2d",
    "dogbone_3d":      "make_dogbone_3d",
}


# Per-variant whitelist of mesh builders the modular problem builder is
# *known* to consume.  Any other builder is incompatible (the modular
# builder reads keys like ``geom["xtip"]`` / ``geom["ytip"]`` only present
# in slant_plate, etc.).  When the strategist's first pick is outside
# this set we coerce to ``make_custom_gmsh`` so the mesh-LLM can build a
# domain that supplies a generic ``geom`` dict.
_VARIANT_COMPATIBLE_MESHES = {
    "linear_elastic_2d_pe": {"make_notched_plate_2d", "make_custom_gmsh"},
    "linear_elastic_2d_ps": {"make_notched_plate_2d", "make_custom_gmsh"},
    "linear_elastic_3d":    {"make_notched_plate_3d", "make_custom_gmsh"},
    "dynamic_2d":           {"make_notched_plate_2d", "make_custom_gmsh"},
    "ductile_2d_pe":        {"make_dogbone_2d",       "make_custom_gmsh"},
    "ductile_3d":           {"make_dogbone_3d",       "make_custom_gmsh"},
    "finite_elastic_2d_ps": {"make_slant_plate_2d",   "make_custom_gmsh"},
    "finite_elastic_2d_pe": {"make_notched_plate_2d", "make_custom_gmsh"},
    "finite_elastic_3d":    {"make_notched_plate_3d", "make_custom_gmsh"},
}


def strategist(spec: CanonicalSpec) -> Action:
    hit = _metadata_pick(spec)
    fallback_note = ""
    if hit is None:
        original_plane = spec.plane_type.value
        original_const = spec.constitutive.value
        original_dim = spec.dimension.value
        hit, fallback_note = _nearest_variant(spec)
        if hit is None:
            raise ValueError(
                f"No modular variant matches spec: dim={spec.dimension} "
                f"plane={spec.plane_type} const={spec.constitutive} "
                f"loading={spec.loading.mode}.  Supported combos are listed in "
                f"fracture_agent/knowledge.py::CATALOG.")
        # Mutate spec to reflect the chosen alt so downstream bookkeeping is
        # consistent.
        from ..schema import Constitutive, Dimension, PlaneType
        spec.plane_type = PlaneType(hit.plane) if hit.plane != "na" else PlaneType.na
        spec.constitutive = Constitutive(hit.constitutive)
        spec.dimension = Dimension.two if hit.dim == 2 else Dimension.three
        # Surface every silent swap as an assumption so the user sees it in
        # the UI / CLI / state.json.  Buried-in-rationale messages get lost
        # in a long run; these get printed as the architect's banner.
        if original_plane != spec.plane_type.value:
            spec.assumptions.append(
                f"Plane type silently swapped {original_plane} -> "
                f"{spec.plane_type.value} (no modular variant exists for "
                f"the requested combination).  Re-run with explicit "
                f"plane_type if this is wrong.")
        if original_const != spec.constitutive.value:
            spec.assumptions.append(
                f"Constitutive silently downgraded {original_const} -> "
                f"{spec.constitutive.value} (no modular variant exists "
                f"for the requested combination).")
        if original_dim != spec.dimension.value:
            spec.assumptions.append(
                f"Dimension silently switched {original_dim} -> "
                f"{spec.dimension.value} (no modular variant exists for "
                f"the requested combination).")

    # Pick the mesh builder from the geometry kind.
    kind = spec.geometry.kind
    if kind in ("custom", "rectangle"):
        mesh_builder = "make_custom_gmsh"
    elif kind == "notched_plate":
        mesh_builder = ("make_notched_plate_2d" if hit.dim == 2
                        else "make_notched_plate_3d")
    elif kind in _BUILTIN_KIND_TO_BUILDER:
        mesh_builder = _BUILTIN_KIND_TO_BUILDER[kind]
    else:
        mesh_builder = hit.mesh_builder

    # Catalog material × variant cross-check.  Catalog names of the form
    # ``*_2D_PE`` / ``*_2D_PS`` / ``*_3D`` encode the dimensionality+plane
    # they were calibrated for.  Applying a 2D-PE catalog to a 3D run (or
    # vice versa) silently produces tuned-for-wrong-state values.  When
    # the embedded tag doesn't match the chosen variant we strip the
    # catalog reference and let the material_helper fall back to a fresh
    # handbook lookup with the correct dimensionality.
    cat = (spec.material.catalog_name or "").strip()
    if cat:
        cat_l = cat.lower()
        is_2d_pe = "2d_pe" in cat_l or "_2d_plane_strain" in cat_l
        is_2d_ps = "2d_ps" in cat_l or "_2d_plane_stress" in cat_l
        is_3d    = "_3d" in cat_l and not (is_2d_pe or is_2d_ps)
        v_l = hit.variant.lower()
        v_is_3d    = "3d" in v_l
        v_is_2d_pe = "2d_pe" in v_l
        v_is_2d_ps = "2d_ps" in v_l
        mismatch = ((is_3d and not v_is_3d)
                    or (is_2d_pe and (v_is_3d or v_is_2d_ps))
                    or (is_2d_ps and (v_is_3d or v_is_2d_pe)))
        if mismatch:
            from ..events import DECISION, emit
            emit(DECISION,
                 f"Catalog material {cat!r} is calibrated for a different "
                 f"dim/plane than variant {hit.variant!r}; stripping the "
                 f"catalog reference so the material lookup re-resolves "
                 f"with the right dimensionality.")
            spec.material.catalog_name = None
            # Keep ``display_name`` so handbook lookup has something to
            # chew on (e.g. "Steel" extracted from "Steel_bench_2D_PE").
            if not (spec.material.display_name or "").strip():
                family = cat.split("_", 1)[0]
                spec.material.display_name = family

    # Variant × mesh-builder compatibility check.  E.g. finite_elastic_2d_ps
    # reads ``geom["xtip"]`` from the slant_plate's geom dict; if the user
    # described a "notched plate" the kind→builder rule above would pick
    # make_notched_plate_2d, which doesn't supply those keys → runtime
    # KeyError inside the modular builder.  Coerce to make_custom_gmsh
    # (the most permissive) and surface the decision.
    compat = _VARIANT_COMPATIBLE_MESHES.get(hit.variant, {mesh_builder})
    if mesh_builder not in compat:
        from ..events import DECISION, emit
        emit(DECISION,
             f"Variant×mesh incompatibility: variant {hit.variant!r} "
             f"does not accept {mesh_builder!r} (only {sorted(compat)}); "
             f"coercing to make_custom_gmsh + kind=custom so the mesh-LLM "
             f"emits a fully populated geom dict.")
        mesh_builder = "make_custom_gmsh"
        # Keep the user's described geometry as the description for the
        # mesh-LLM; flip kind so synthesizer takes the custom path.
        if kind not in ("custom", "rectangle"):
            spec.geometry.kind = "custom"
            kind = "custom"

    # For `rectangle`, the Synthesizer routes through the custom-mesh LLM; we
    # enrich the Architect's custom_description so the LLM has enough to work
    # with even when it was left terse.
    if kind == "rectangle" and not spec.geometry.custom_description:
        dims = spec.geometry.dimensions
        w = dims.get("W", dims.get("width", 1.0))
        l = dims.get("L", dims.get("height", dims.get("length", 1.0)))
        spec.geometry.custom_description = (
            f"Plain rectangular domain {w} mm wide by {l} mm tall, centered at "
            f"the origin (domain = [-{w/2}, {w/2}] x [-{l/2}, {l/2}]). "
            f"No pre-crack.  Use a uniform unstructured triangular mesh.")
        if not spec.geometry.custom_regions:
            from ..schema import RegionSpec
            tol_L = l / 2.0
            tol_W = w / 2.0
            spec.geometry.custom_regions = [
                RegionSpec(name="bottom",
                           description=f"y == -{tol_L} (bottom edge)"),
                RegionSpec(name="top",
                           description=f"y == +{tol_L} (top edge)"),
                RegionSpec(name="left",
                           description=f"x == -{tol_W} (left edge)"),
                RegionSpec(name="right",
                           description=f"x == +{tol_W} (right edge)"),
            ]

    rationale = (f"Picked {hit.variant} because dimension={spec.dimension.value}, "
                 f"constitutive={spec.constitutive.value}, "
                 f"plane={spec.plane_type.value}, loading={spec.loading.mode.value}. "
                 f"Mesh: {mesh_builder} (geometry kind={kind}).  Fracture "
                 f"{'enabled' if spec.fracture_enabled else 'disabled'}.")
    if fallback_note:
        rationale += "  " + fallback_note

    # ------------------------------------------------------------------ #
    # Skeleton-RAG advisory (B1): retrieve the closest example file in
    # ``modular/examples/`` and surface the σ score so the user / paper
    # ablation can reason about retrieval quality.  The deterministic
    # metadata pick remains authoritative — RAG is informational here,
    # ablatable on the dedicated ablation switch.
    try:
        from ..rag import retrieve_skeleton, sigma_gate
        from ..events import DECISION, emit
        from ..telemetry import llm_agent
        with llm_agent("strategist_rag"):
            retr = retrieve_skeleton(_skeleton_query(spec), k=3,
                                       auto_build=False)
        top, _ = sigma_gate(retr, threshold=0.85)
        if retr:
            top0 = retr[0]
            emit(DECISION,
                 f"RAG/skeleton: top match '{top0.id}' σ={top0.score:.3f} "
                 + ("(passes 0.85 gate)" if top else "(below 0.85 gate)"))
            if top0.metadata.get("variant") and top0.metadata["variant"] != hit.variant:
                emit(DECISION,
                     f"RAG/skeleton: closest example '{top0.id}' suggests "
                     f"variant '{top0.metadata['variant']}' but Strategist "
                     f"picked '{hit.variant}' on metadata — keeping the "
                     f"metadata pick as authoritative.")
    except Exception:
        # RAG is best-effort augmentation; never block the deterministic
        # variant pick on its failures.
        pass

    return Action(variant=hit.variant,
                  mesh_builder=mesh_builder,
                  solver=hit.solver,
                  rationale=rationale)


def _skeleton_query(spec: CanonicalSpec) -> str:
    """Distil a CanonicalSpec into a short retrieval query — emphasise
    the dimensions reviewers expect (kinematics, constitutive, mode)."""
    g = spec.geometry
    parts = [
        f"{spec.dimension.value} {spec.plane_type.value}",
        spec.constitutive.value,
        spec.loading.mode.value,
        f"geometry: {g.kind}",
    ]
    if g.dimensions:
        parts.append("dims " + " ".join(f"{k}={v}" for k, v in g.dimensions.items()))
    if g.pre_crack_segments:
        parts.append(f"{len(g.pre_crack_segments)} pre-crack segment(s)")
    src_summary = spec.source_summary or ""
    if src_summary:
        parts.append(src_summary)
    return " | ".join(parts)
