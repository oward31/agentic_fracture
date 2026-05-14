"""Session-slug generation.

A simulation's folder is named ``<stamp>_<material>_<shape>_<loading>`` — a
human-readable identifier that sorts chronologically and tells you at a
glance what the run is about (e.g. ``20260424_141530_graphite_custom_sympull``).
"""
from __future__ import annotations
import re
import time
from typing import Optional

from .schema import CanonicalSpec


def _slugify(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace to underscores."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w]+", "_", s)
    return s.strip("_")


def _material_slug(spec: CanonicalSpec) -> str:
    m = spec.material
    if m.display_name:
        return _slugify(m.display_name)
    if m.catalog_name:
        # Steel_bench_2D_PE → steel
        return _slugify(m.catalog_name.split("_", 1)[0])
    return "unknown"


# Shape keyword → canonical slug.  Keys are regex patterns matched against
# the lowercased description; the first match wins.  Patterns are ordered by
# specificity (multi-word first).  Word-boundary asserts stop false positives
# like "ente**ring**" → ring, "fo**rming**" → ring, etc.
_SHAPE_PATTERNS = (
    (r"\b(three[-\s_]point[-\s_]bend|3pb|3[-\s_]point[-\s_]bend)\b", "three_point_bend"),
    (r"\bplate[-\s_]with[-\s_]hole\b",                                "plate_with_hole"),
    (r"\bcompact[-\s_]tension|\bct\s+specimen\b",                     "compact_tension"),
    (r"\b(l[-\s_]?shape[d]?|l[-\s_]?panel|el[-\s_]?shape)\b",         "l_shape"),
    (r"\bbrazilian\b",                                                 "brazilian"),
    (r"\b(annulus|o[-\s_]?ring|washer)\b",                             "ring"),
    (r"\bdogbone|\bdog[-\s_]bone\b",                                  "dogbone"),
    (r"\bcruciform|\bcross[-\s_]?specimen\b",                         "cruciform"),
    (r"\b(disc|disk)\b",                                               "disc"),
    (r"\bcylinder\b",                                                  "cylinder"),
    (r"\b(ring|hoop)\b",                                               "ring"),
    (r"\bcircle|\bcircular\b",                                        "circle"),
    (r"\b(rectangular|rectangle|bar|coupon)\b",                       "rectangle"),
    (r"\bsquare\b",                                                    "square"),
    (r"\b(beam|notched[-\s_]beam)\b",                                  "beam"),
)


def _shape_slug(spec: CanonicalSpec) -> str:
    kind = spec.geometry.kind or "geom"
    # For custom geometries, try to pull a one-word hint from the
    # description (the first few nouns often summarise the shape).
    if kind == "custom" and spec.geometry.custom_description:
        text = spec.geometry.custom_description.lower()
        for pattern, slug in _SHAPE_PATTERNS:
            if re.search(pattern, text):
                return slug
    return _slugify(kind)


def _loading_slug(spec: CanonicalSpec) -> str:
    """Classify the loading using edge geometry, not just magnitude sign.

    A negative magnitude on the *left* edge x-axis is *outward* (tension),
    not compression: the slug must consult ``edge_axis_outward_sign`` to
    interpret the sign correctly.  Loadings whose component axis is
    parallel to the loaded edge are classified as ``shear`` (e.g. top
    edge pulled in x).
    """
    cases = spec.bcs.loading
    if not cases:
        return "static"
    # Lazy import — avoid a circular dep at module load time.
    from .region_names import edge_axis_outward_sign

    n_outward = n_inward = n_shear = n_unknown = 0
    edge_axes_seen: set[int] = set()
    for c in cases:
        if c.magnitude is None or abs(c.magnitude) < 1e-12:
            continue
        info = edge_axis_outward_sign(c.region)
        comp = c.component if c.component is not None else 1
        if info is None:
            # Region not recognisable as a canonical edge — fall back to
            # raw sign (best effort; may misclassify rotated geometries).
            (n_outward if c.magnitude > 0 else n_inward).__index__  # noqa
            if c.magnitude > 0:
                n_outward += 1
            else:
                n_inward += 1
            n_unknown += 1
            continue
        edge_axis, outward_sign = info
        edge_axes_seen.add(edge_axis)
        if comp != edge_axis:
            # Tangential to the edge → shear.
            n_shear += 1
        else:
            net = c.magnitude * outward_sign
            if net > 0:
                n_outward += 1
            else:
                n_inward += 1

    # Symmetric opposite-direction pulls on different edges → sympull.
    if n_outward >= 2 and n_inward == 0 and n_shear == 0 and len(edge_axes_seen) >= 1:
        return "sympull"
    if n_shear >= 1 and (n_outward + n_inward) == 0:
        return "shear"
    if n_shear >= 1:
        return "mixed"
    if n_outward > 0 and n_inward == 0:
        return "tension"
    if n_inward > 0 and n_outward == 0:
        return "compression"
    if n_outward == 0 and n_inward == 0:
        return "load"
    return "mixed"


def session_slug(spec: CanonicalSpec, stamp: Optional[str] = None) -> str:
    """Return a filesystem-safe slug describing this simulation.

    Shape: ``YYYYMMDD_HHMMSS_<material>_<shape>_<loading>``.
    """
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    mat   = _material_slug(spec)   or "mat"
    shape = _shape_slug(spec)       or "geom"
    load  = _loading_slug(spec)     or "load"
    return f"{stamp}_{mat}_{shape}_{load}"
