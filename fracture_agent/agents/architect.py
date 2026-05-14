"""Architect — merges fragments into a validated CanonicalSpec.

This agent is the main failure point of the pipeline — if it gets the spec
wrong, everything downstream runs correctly on the wrong problem.  So we:

  * feed the LLM the raw user prompt *directly* (not just receptionist
    fragments) so numeric dimensions and material names survive;
  * pass a Gemini ``responseSchema`` derived from the Pydantic model;
  * post-process output strictly — strip hallucinated catalog names,
    extract missing dimensions from free text, force custom_regions when
    kind="custom";
  * keep the system prompt short and example-heavy.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from ..config import FAST_MODEL, PRIMARY_MODEL
from ..knowledge import list_builtin_materials
from ..llm import llm, pydantic_to_gemini_schema
from ..schema import CanonicalSpec


class _Envelope(BaseModel):
    spec: CanonicalSpec
    open_questions: List[str] = Field(default_factory=list)


SYS = """You are the Architect.  Convert the user's problem description to a
CanonicalSpec JSON.  Be bold, inferential, silent on defaults.

RETURN SHAPE — return this exact wrapper (no other keys):
{"spec": {...CanonicalSpec...}, "open_questions": []}

SPEC FIELDS:
  fracture_enabled: bool                (default true)
  dimension:        "2D" | "3D"          (default "2D")
  plane_type:       "plane_strain" | "plane_stress" | "na"
  constitutive:     "linear_elastic" | "j2_plasticity" | "lopez_pamies"
  material:  {catalog_name, display_name, E, nu, Gc, ...}
  geometry:  {kind, dimensions:{W,L,...}, custom_description, custom_regions:[{name,description},...]}
  bcs:       {fixed:[{region,components}], loading:[{region,component,control,magnitude}]}
  loading:   {mode:"quasistatic"|"dynamic", steps:100, T_total:1.0}
  source_summary: "<one sentence>"

KIND:
  rectangle      - plain bar / coupon / window / plate, no crack
  notched_plate  - rectangular plate with a crack from an edge (SENT-like)
  dogbone_2d/3d  - tensile test coupon (use 3d if user says 3D)
  slant_plate    - rubber slant pokerchip
  plate_3d       - 3D slab
  custom         - anything else (L-shape, disc, annulus, cruciform, ring,
                   hole, T, beam-with-supports, etc.)

CONSTITUTIVE (default linear_elastic):
  metals / ceramics / rocks / glass / concrete / paper / wood  -> linear_elastic
  rubber / silicone / elastomer / O-ring / PDMS                -> lopez_pamies
  user explicitly asks for yield/plastic/ductile               -> j2_plasticity

PLANE:
  thin (sheet, foil, window, paper, film) -> plane_stress
  bulk / thick                            -> plane_strain
  3D                                       -> "na"

MODE:
  impact / hit / struck / projectile / drop / shock / wave / fast -> dynamic
  else                                                             -> quasistatic

CATALOG (set catalog_name to an exact match when the first word of the
catalog key equals the user's material word, otherwise leave null):
  {CATALOG_LIST}

BCS: components [] = all clamped, [0]=ux, [1]=uy, [2]=uz. component 0/1/2 in loading is the motion axis.
  pulled  -> fix one end, displace the opposite end outward
  squeezed/compressed -> fix one, push the other inward (negative magnitude)
  bent/3PB -> two supports at the ends (full pin + y-pin), load at midspan
  hit / impacted on a window / plate -> fix ALL 4 outer edges (the frame),
    displace a small central region.  NOT a 3PB problem.
  biaxial / cruciform -> two pairs of opposite-sign loads on orthogonal
    arms.  ALSO include a single fixed region (center pin) to kill rigid
    body motion.  You must not leave bcs.fixed empty.
  pulled by X mm each on opposite edges -> two cases, same component, signs +X and -X

CUSTOM rules:
  - custom_description MUST quote the user's shape and numbers verbatim.
  - custom_regions MUST list every region referenced in bcs, each with a
    one-line predicate description (e.g. "y == y_min", "x == x_min AND y >= y_mid").
  - Do NOT add a pre-crack unless the user said crack/notch/slit/flaw.

PRE-CRACK SEGMENTS (critical for propagation):
  Whenever the user mentions a pre-crack, notch, slit, or flaw, ALSO
  populate ``geometry.pre_crack_segments``.  The generated driver uses it
  to initialise the phase field to z=0 along the cracks (damaged-notch
  BC) — without this the solver sees an intact specimen and will rarely
  nucleate a crack below the nucleation threshold.

  EXACT SHAPE — a list of SEGMENTS, each a PAIR of POINTS, each point [x, y]:

      pre_crack_segments: [
        [ [x1, y1], [x2, y2] ],     <-- segment 1 (one pair of endpoints)
        [ [x3, y3], [x4, y4] ],     <-- segment 2 (if another crack)
        ...
      ]

  THREE levels of nesting.  Never flatten to [[x1,y1],[x2,y2]]; that's
  only ONE segment, not a list of segments — wrap it: [[[x1,y1],[x2,y2]]].

  Coordinates in mm, same frame as custom_description / custom_regions.
  Default to lower-left origin (x=0,y=0 at bottom-left) unless the user
  describes a centred domain.

  Examples:
    "10 mm square, horizontal crack from left edge mid-height to centre":
      pre_crack_segments = [ [[0, 5], [5, 5]] ]
    "200 mm square, two 25 mm horizontal cracks: one from left at y=102.5,
     one from right at y=97.5":
      pre_crack_segments = [
        [[0, 102.5], [25, 102.5]],
        [[175, 97.5], [200, 97.5]]
      ]
    "80x104 mm plate, 20 mm inclined notch at centre, 45 deg"
     (centre (40,52), half-length 10, cos45=sin45≈0.7071):
      pre_crack_segments = [ [[32.93, 44.93], [47.07, 59.07]] ]

UNITS (strict — ALWAYS convert to this system before storing any number):

  Canonical system:  mm, N, MPa, s, tonne   (consistent: 1 tonne·mm/s^2 = 1 N)

  Length:
      m    -> mm   (x 1000)
      cm   -> mm   (x 10)
      um   -> mm   (x 0.001)
      inch -> mm   (x 25.4)
      ft   -> mm   (x 304.8)
  Force:
      kN   -> N    (x 1000)
      MN   -> N    (x 1e6)
      lbf  -> N    (x 4.448)
  Stress / modulus / strength  (E, nu has no units; all others):
      Pa   -> MPa  (x 1e-6)
      kPa  -> MPa  (x 1e-3)
      GPa  -> MPa  (x 1000)
      psi  -> MPa  (x 6.895e-3)
      ksi  -> MPa  (x 6.895)
  Fracture energy (Gc):    target = N/mm  (same as kJ/m^2)
      N/m  = J/m^2     -> N/mm  (x 0.001)
      kN/m = kJ/m^2    -> N/mm  (x 1)
      MJ/m^2           -> N/mm  (x 1000)
      erg/cm^2         -> N/mm  (x 1e-7)
  Density (only for dynamic problems):   target = tonne/mm^3
      kg/m^3           -> tonne/mm^3  (x 1e-12)
      g/cm^3           -> tonne/mm^3  (x 1e-9)
      lb/in^3          -> tonne/mm^3  (x 2.7680e-8)
  Time:
      ms   -> s    (x 0.001)
      us   -> s    (x 1e-6)
  Velocity (dynamic):
      m/s  -> mm/s (x 1000)
  Displacement magnitude: same as length above.

EXAMPLES
  "E = 210 GPa"          -> E: 210000
  "E = 2.07e5 MPa"       -> E: 207000
  "Gc = 2.7 kJ/m^2"      -> Gc: 2.7
  "Gc = 2700 N/m"        -> Gc: 2.7
  "sts = 2 GPa"          -> sigma_ts: 2000
  "density = 2500 kg/m^3" -> rho: 2.5e-9
  "pulled by 1 mm"       -> magnitude: 1.0
  "2 cm displacement"    -> magnitude: 20.0
  "applied 5 kN"         -> magnitude: 5000.0, control = "force"
  "50 m/s impact"        -> for a traction BC, magnitude computed from
                            impedance; leave open_question if unclear.

Any field whose unit you cannot disambiguate must be left null and added
to open_questions rather than guessed.

MATERIAL ALIASES
  Users write short property names; map them to the canonical MaterialSpec
  field names when filling the JSON:
    sts, sigma_t, tensile_strength  -> sigma_ts
    scs, sigma_c, compressive_strength  -> sigma_cs
    shs, sigma_h, hydrostatic_strength -> sigma_hs
    sy, yield, sigma_yield, sigma_y  -> sigma_y0
    density, rho_0                   -> rho
    youngs_modulus, young_modulus    -> E
    poisson, poisson_ratio           -> nu
    fracture_toughness, toughness_gc -> Gc

Keep the response short.  Produce the JSON block only, no commentary.
"""


def _build_response_schema() -> Dict[str, Any]:
    return pydantic_to_gemini_schema(_Envelope)


def architect(fragments_history: List[Dict[str, Any]],
              clarifications: List[Dict[str, str]]) -> Tuple[CanonicalSpec,
                                                              List[str]]:
    mats = list_builtin_materials()
    system = SYS.replace("{CATALOG_LIST}", ", ".join(mats.keys()))

    # Pull the user's raw text front-and-centre for the Architect — the
    # receptionist fragments are auxiliary.
    raw_user_text = _extract_raw_user_text(fragments_history)
    user_msg_parts: List[str] = []
    user_msg_parts.append("USER PROMPT (verbatim — read this carefully):\n"
                          f'"""{raw_user_text}"""')
    user_msg_parts.append("\n\nRECEPTIONIST FRAGMENTS (for reference only):\n"
                          + json.dumps(fragments_history, indent=2))
    if clarifications:
        user_msg_parts.append("\n\nUSER ANSWERS TO PRIOR QUESTIONS:\n"
                              + json.dumps(clarifications, indent=2))
    user = "\n".join(user_msg_parts)

    # We intentionally DO NOT pass responseSchema here.  Gemini 2.5 Pro stubs
    # complex nested-Pydantic schemas — it returns literal "string"/123/1.23
    # placeholder values instead of real content.  Pydantic below catches any
    # shape errors and the post-processing phase recovers from the common
    # ones (missing dims, missing custom_regions, bad catalog matches).
    # Flash is the default for the Architect: this is pure structured
    # extraction, not reasoning, and Pro's occasional thinking-loop
    # deadlocks on combinatorial BC prompts are a serious liability here.
    # We still fall back to Pro if Flash is rate-limited — the keys cycle
    # independently per model on the free tier.
    try:
        reply = llm().complete_json(
            system, user,
            temperature=0.1, max_output_tokens=16384,
            thinking_budget=-1, model=FAST_MODEL)
    except Exception as e:
        print(f"[architect] {FAST_MODEL} failed ({type(e).__name__}); "
              f"retrying on {PRIMARY_MODEL}.")
        reply = llm().complete_json(
            system, user,
            temperature=0.1, max_output_tokens=65536,
            thinking_budget=-1, model=PRIMARY_MODEL)

    # Stub detection: if the response looks like a Gemini placeholder-fill
    # (display="string" / E=1.23 / magnitude=1.23 etc.) we retry with a
    # remonstration prompt.
    if _looks_like_stub(reply):
        remonstrance = (user + "\n\n"
                        "PREVIOUS OUTPUT was rejected — it contained placeholder "
                        "values like 'string' and 1.23.  Re-read the user prompt "
                        "literally and fill every field with real content.")
        reply = llm().complete_json(
            system, remonstrance,
            temperature=0.0, max_output_tokens=32768)

    # Accept either envelope form or bare spec.
    if "spec" in reply and isinstance(reply["spec"], dict):
        reply_spec = reply["spec"]
        open_q = list(reply.get("open_questions", []) or [])
    else:
        for wrap in ("problem_spec", "canonical_spec", "CanonicalSpec", "data"):
            if wrap in reply and isinstance(reply[wrap], dict) and len(reply) <= 2:
                reply = reply[wrap]
                break
        open_q = reply.pop("open_questions", []) or []
        reply_spec = reply
    _reshape(reply_spec)
    _sanitise(reply_spec)
    spec = CanonicalSpec(**reply_spec)

    # ---- Post-processing — defend against common LLM failures -------- #
    _enforce_catalog_match(spec, mats)
    _fallback_material_when_missing(spec, mats)
    _enrich_catalog_from_display_name(spec, mats)
    _backfill_dimensions(spec, raw_user_text)
    _filter_geometry_dimensions(spec)
    _canonicalise_region_names(spec)
    _coerce_to_custom_if_nonstandard_regions(spec, raw_user_text)
    _ensure_custom_fields(spec, raw_user_text)
    _preserve_raw_prompt_in_description(spec, raw_user_text)
    _ensure_rigid_body_constraint(spec, raw_user_text)
    _apply_directional_loading_signs(spec, raw_user_text)
    _fill_default_magnitudes(spec)
    _detect_multi_stage_loading(spec, raw_user_text)
    _downgrade_unrequested_plasticity(spec, raw_user_text)
    _sanity_loading(spec, open_q)
    _unit_sanity_warnings(spec)
    open_q = _strip_resolvable_material_clarifications(spec, open_q, mats)
    return spec, list(open_q)


# ---------------------------------------------------------------------------
# Constitutive defense — Flash habitually picks j2_plasticity for any prompt
# that mentions a metal + the word "tension".  Most users saying "copper bar
# tension" want a *linear-elastic* phase-field run, not a plastic one — they'd
# say "yield" or "plastic" if they meant ductile.  Downgrade silently and log
# the assumption so the user can override.
# ---------------------------------------------------------------------------
_PLASTICITY_KEYWORDS_RE = re.compile(
    r"\b("
    r"plastic(ity)?|"
    r"yield(ing)?|"
    r"ductile|j2|"
    r"hardening|"
    r"flow\s+stress|"
    r"strain[-\s]*hardening"
    r")\b",
    re.IGNORECASE)
_RUBBER_KEYWORDS_RE = re.compile(
    r"\b("
    r"rubber|elastomer|silicone|pdms|neo[-\s]*hookean|"
    r"hyper[-\s]*elastic|finite[-\s]*strain|ogden|mooney|"
    r"o[-\s]*ring"
    r")\b",
    re.IGNORECASE)


def _downgrade_unrequested_plasticity(spec: CanonicalSpec,
                                       raw_user_text: str) -> None:
    """If the architect picked j2_plasticity but the user prompt mentions
    no plastic / yield / ductile keyword, silently downgrade to
    linear_elastic and note the assumption.  Symmetric guard for
    lopez_pamies — finite-elasticity is only chosen when rubber-like
    terms appear."""
    from ..events import DECISION, emit
    from ..schema import Constitutive

    if not raw_user_text:
        return
    has_plastic_kw = bool(_PLASTICITY_KEYWORDS_RE.search(raw_user_text))
    has_rubber_kw = bool(_RUBBER_KEYWORDS_RE.search(raw_user_text))

    if (spec.constitutive == Constitutive.j2_plasticity and not has_plastic_kw):
        spec.constitutive = Constitutive.linear_elastic
        # Reset implied material problem_type so material_helper picks the
        # right handbook lookup later.
        if spec.material.problem_type == "ductile":
            spec.material.problem_type = "linear_elasticity"
        emit(DECISION,
             "Constitutive downgraded j2_plasticity -> linear_elastic: "
             "user prompt did not mention plastic / yield / ductile.  "
             "Add those words explicitly if you wanted plasticity.")
        spec.assumptions.append(
            "Constitutive downgraded j2_plasticity -> linear_elastic "
            "because the user prompt did not mention plastic / yield / "
            "ductile / hardening.")

    if (spec.constitutive == Constitutive.lopez_pamies and not has_rubber_kw):
        spec.constitutive = Constitutive.linear_elastic
        if spec.material.problem_type == "finite_elasticity":
            spec.material.problem_type = "linear_elasticity"
        emit(DECISION,
             "Constitutive downgraded lopez_pamies -> linear_elastic: "
             "user prompt did not mention rubber / elastomer / hyper-elastic.")
        spec.assumptions.append(
            "Constitutive downgraded lopez_pamies -> linear_elastic "
            "because the user prompt did not mention rubber / elastomer / "
            "hyper-elastic / Neo-Hookean.")


def _unit_sanity_warnings(spec: CanonicalSpec) -> None:
    """Post-hoc sanity check for the most common unit-conversion mistakes.
    Emits DECISION events so the user sees any suspect value in the UI
    *before* the simulation runs with bad numbers.  Does not modify the
    spec — the Architect should have converted already."""
    from ..events import DECISION, emit
    m = spec.material
    # Each tuple: (field_name, value, min-plausible, max-plausible, unit-hint)
    checks = [
        ("E (Young's modulus)", m.E, 0.1, 1.5e6,
         "MPa; if > 1e8 you may have stored Pa instead of MPa, "
         "if < 1 you may have stored GPa numerically"),
        ("nu (Poisson's ratio)", m.nu, 0.0, 0.499,
         "dimensionless; must be between 0 and 0.5"),
        ("Gc (fracture energy)", m.Gc, 1e-6, 1e4,
         "N/mm (= kJ/m^2); if > 1e4 you probably stored N/m not N/mm"),
        ("sigma_ts (tensile strength)", m.sigma_ts, 0.01, 1e5,
         "MPa"),
        ("sigma_cs (compressive strength)", m.sigma_cs, 0.01, 1e5,
         "MPa"),
        ("rho (density)", m.rho, 1e-13, 1e-4,
         "tonne/mm^3; typical metals ~8e-9, water 1e-9, "
         "if ~1000 you stored kg/m^3 (divide by 1e12)"),
    ]
    for name, val, lo, hi, hint in checks:
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v < lo or v > hi:
            emit(DECISION,
                 f"[WARN] {name} = {v:g} looks out of range — expected {hint}. "
                 f"Proceeding, but double-check the number.")

    # Dimensions: any single dimension > 1e4 (10 m) is suspect for bench-scale
    # specimens; likely stored m rather than mm.
    for k, v in spec.geometry.dimensions.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 1e4:
            emit(DECISION,
                 f"[WARN] geometry.dimensions[{k}] = {fv:g} mm is very large — "
                 f"did the user give meters? (expected mm)")
        if 0 < fv < 1e-3:
            emit(DECISION,
                 f"[WARN] geometry.dimensions[{k}] = {fv:g} mm is sub-micron — "
                 f"unit mistake? (expected mm)")


_RESERVED_DISPLAY_NAMES = {
    # Constitutive keywords — not real material names.
    "linear_elastic", "linear elastic", "linear_elastic_solid",
    "linear elastic solid", "elastic", "elastic solid", "isotropic",
    "j2_plasticity", "j2 plasticity", "plastic", "ductile",
    "lopez_pamies", "lopez pamies", "finite_elasticity",
    "finite elasticity", "hyperelastic", "rubber-like",
    "unknown", "unspecified", "none", "null", "generic",
    "generic linear elastic", "", "material", "generic material",
}


def _is_reserved_display(name: str) -> bool:
    n = (name or "").lower().strip()
    if not n:
        return True
    if n in _RESERVED_DISPLAY_NAMES:
        return True
    # heuristic: names containing 'elastic', 'plastic', 'generic' with no
    # actual material word are almost always constitutive labels leaking.
    bad = ("elastic solid", "plastic solid", "generic", "solid isotropic")
    if any(b in n for b in bad):
        return True
    return False


def _fallback_material_when_missing(spec: CanonicalSpec,
                                      mats: Dict[str, Any]) -> None:
    """If the user didn't name a material, don't let the LLM leak a
    constitutive-type string into display_name ('Linear Elastic') — fall
    back to a sensible catalog entry based on what the Architect picked."""
    m = spec.material
    if _is_reserved_display(m.display_name or ""):
        # Pick default catalog by constitutive + plane + dim.
        want_type = _match_problem_type(spec)
        for cat_name, cat_entry in mats.items():
            if cat_entry.get("problem_type") != want_type:
                continue
            tag = cat_name.lower()
            dim_want = 3 if spec.dimension.value == "3D" else 2
            if (dim_want == 3 and "3d" in tag) or (
                    dim_want == 2 and ("2d" in tag or "3d" not in tag)):
                m.catalog_name = cat_name
                m.display_name = cat_name.split("_", 1)[0]
                return
        # Last-resort: first compatible catalog entry.
        for cat_name, cat_entry in mats.items():
            if cat_entry.get("problem_type") == want_type:
                m.catalog_name = cat_name
                m.display_name = cat_name.split("_", 1)[0]
                return


def _enrich_catalog_from_display_name(spec: CanonicalSpec,
                                        mats: Dict[str, Any]) -> None:
    """Inverse of ``_enforce_catalog_match``: when the LLM returned
    ``catalog_name = null`` but the user's display_name obviously maps to a
    built-in catalog entry (e.g. "glass" -> Glass_dyn_2D_PS, "rubber" ->
    Rubber_LopezPamies_*), set it.

    CRITICAL: we only set catalog_name when the user DID NOT also supply
    numeric properties.  If they did, the catalog would silently override
    their values at runtime because the generated script calls
    ``load_material('<catalog_name>')`` instead of reading from their
    material.json.
    """
    m = spec.material
    if m.catalog_name:
        return
    # Any user-supplied numeric property disables catalog-auto-set.
    numeric_fields = ("E", "nu", "Gc", "sigma_ts", "sigma_cs", "rho",
                       "sigma_y0", "H_hardening", "mu1", "mu2",
                       "alpha1", "alpha2", "kappa", "sigma_hs")
    user_gave_numbers = any(getattr(m, f, None) is not None
                             for f in numeric_fields)
    if user_gave_numbers:
        return
    word = (m.display_name or "").lower().strip()
    if not word:
        return
    ptype_guess = _match_problem_type(spec)

    best = None
    for cat_name, cat_entry in mats.items():
        family = cat_name.split("_", 1)[0].lower()
        if not family:
            continue
        if family not in word and word not in family:
            continue
        if cat_entry.get("problem_type") != ptype_guess:
            continue
        # Prefer an exact-dimension match if there are multiple candidates
        # (e.g. Rubber_LopezPamies_2D_PE vs _3D).
        dim_want = 3 if spec.dimension.value == "3D" else 2
        tag = cat_name.lower()
        if (dim_want == 3 and "3d" in tag) or \
           (dim_want == 2 and ("2d" in tag or "3d" not in tag)):
            best = cat_name
            break
        if best is None:
            best = cat_name
    if best is not None:
        spec.material.catalog_name = best


def _match_problem_type(spec: CanonicalSpec) -> str:
    if spec.constitutive.value == "j2_plasticity":
        return "ductile"
    if spec.constitutive.value == "lopez_pamies":
        return "finite_elasticity"
    if spec.loading.mode.value == "dynamic":
        return "dynamic_linear_elasticity"
    return "linear_elasticity"


# ---------------------------------------------------------------------------
# Region-name canonicalisation — nudge the Architect's invented region names
# onto the canonical ones that built-in meshes emit, so we can keep using the
# prebuilt mesh instead of coercing to custom.
# ---------------------------------------------------------------------------
_CANON_MAP = {
    # dogbone
    "grip_1": "grip_left", "grip1": "grip_left",
    "top_grip": "grip_right", "bottom_grip": "grip_left",
    "grip_2": "grip_right", "grip2": "grip_right",
    "grip_a": "grip_left", "grip_b": "grip_right",
    # plate / rectangle aliases
    "bottom_edge": "bottom", "top_edge": "top",
    "left_edge":   "left",   "right_edge":  "right",
    "bot": "bottom", "tp": "top",
    "lhs": "left", "rhs": "right",
    # axis-style naming
    "y_min": "bottom", "ymin": "bottom", "ymin_edge": "bottom",
    "y_max": "top",    "ymax": "top",    "ymax_edge": "top",
    "x_min": "left",   "xmin": "left",   "xmin_edge": "left",
    "x_max": "right",  "xmax": "right",  "xmax_edge": "right",
    "z_min": "front",  "z_max": "back",
    # 3D
    "front_face": "front", "back_face": "back",
    "front_edge": "front", "back_edge": "back",
}


_AXIS_VAL_RE = re.compile(r"^([xyz])_(\d+(?:\.\d+)?)$")


def _canonicalise_region_names(spec: CanonicalSpec) -> None:
    """Rename region references to canonical mesh-builder names.

    Two-stage rewrite:
      1. **Slugify** — turn any free-text name into a code-safe slug
         (``bottom edge`` -> ``bottom_edge``, ``Top Half of Left Edge`` ->
         ``top_half_of_left_edge``).  This is the form the mesh LLM will
         emit verbatim and the driver will look up.
      2. **Alias map** — collapse common slug aliases onto the canonical
         names exposed by built-in meshes (``bottom_edge`` -> ``bottom``,
         ``y_min`` -> ``bottom``).  Custom-mesh names without a known alias
         pass through unchanged.
      3. **Axis-value match** — when the architect emits coordinate-style
         names like ``y_0`` or ``x_25`` (one slugified from a predicate
         like ``"y == 0"``), fold them onto the canonical edge whose
         coordinate matches the geometry's bounding box.

    Applied to BCs AND ``custom_regions`` so the mesh LLM and the BCs
    always agree on a single name string.
    """
    from ..region_names import slugify

    dims = spec.geometry.dimensions or {}
    # Pull the most plausible upper-bound coordinate per axis.  We don't
    # require a perfect dimension key — common synonyms are accepted.
    max_x = float(dims.get("W") or dims.get("Lx") or dims.get("width") or 0.0)
    max_y = float(dims.get("L") or dims.get("H") or dims.get("Ly") or
                   dims.get("height") or 0.0)
    max_z = float(dims.get("thickness") or dims.get("D") or dims.get("Lz") or
                   dims.get("depth") or 0.0)
    axis_max = {"x": max_x, "y": max_y, "z": max_z}
    axis_min_name = {"x": "left", "y": "bottom", "z": "front"}
    axis_max_name = {"x": "right", "y": "top", "z": "back"}

    def canon(name: str) -> str:
        if not name:
            return name
        s = slugify(name) or name.lower().strip()
        if s in _CANON_MAP:
            return _CANON_MAP[s]
        # ``y_0`` style — value at coordinate origin.
        m = _AXIS_VAL_RE.match(s)
        if m:
            axis, val_str = m.group(1), m.group(2)
            try:
                val = float(val_str)
            except ValueError:
                return s
            if abs(val) < 1e-6:
                return axis_min_name[axis]
            cap = axis_max.get(axis, 0.0)
            if cap > 0 and abs(val - cap) < max(1e-3, 0.01 * cap):
                return axis_max_name[axis]
        return s

    for f in spec.bcs.fixed:
        f.region = canon(f.region)
    for l in spec.bcs.loading:
        l.region = canon(l.region)
    # For custom kinds, keep the mesh-LLM's region list in sync with the
    # renamed BCs.  For built-in kinds, custom_regions is ignored anyway.
    for r in spec.geometry.custom_regions:
        r.name = canon(r.name)


# ---------------------------------------------------------------------------
# Directional loading — fix the SIGN of displacement magnitudes when the
# user said "outward / apart / outside / away" but the LLM picked the
# inward sign.  This is a conservative *sign-only* correction: we never
# touch the magnitude or the component, only ensure the sign matches the
# user's stated direction.
# ---------------------------------------------------------------------------
_OUTWARD_VERB_PAT = re.compile(
    r"\b("
    r"pull(?:ed|ing)?\s+(?:apart|out(?:side|ward[s]?)?|away)|"
    r"pull(?:ed|ing)?\s+(?:them\s+)?out(?:side|ward[s]?)?|"
    r"displaced?\s+out(?:ward[s]?|side)?|"
    r"moved?\s+out(?:ward[s]?|side|\s+from)?|"
    r"opened?(?:\s+up)?|"
    r"separat(?:ed|ing)|"
    r"part(?:ed|ing)|"
    r"spread(?:ing|\s+apart)?"
    r")\b",
    re.IGNORECASE)

_INWARD_VERB_PAT = re.compile(
    r"\b("
    r"compress(?:ed|ing)?|"
    r"squeez(?:ed|ing)?|"
    r"push(?:ed|ing)?\s+(?:in|inward[s]?|together)|"
    r"pinch(?:ed|ing)?|"
    r"clos(?:ed|ing)?\s+(?:in|together)"
    r")\b",
    re.IGNORECASE)


def _apply_directional_loading_signs(spec: CanonicalSpec,
                                      raw_user_text: str) -> None:
    """If the user's prompt says "pulled apart / pulled outward / pulled
    outside / pulled away" (or symmetric inward verbs), make sure each
    displacement loading on a recognisable rectangular edge has the sign
    the user described.

    Conservative: only flips a sign that disagrees with the user's stated
    direction; leaves the absolute magnitude and the chosen component
    alone.  Skips loading cases on regions we can't classify (interior
    pins, unrecognised half-edges).
    """
    if not raw_user_text or not spec.bcs.loading:
        return
    has_outward = bool(_OUTWARD_VERB_PAT.search(raw_user_text))
    has_inward = bool(_INWARD_VERB_PAT.search(raw_user_text))
    if not (has_outward or has_inward):
        return
    # If both, the prompt probably mixes phases (e.g. "compressed then
    # pulled apart") — too ambiguous for a global flip; bail.
    if has_outward and has_inward:
        return

    from ..events import DECISION, emit
    from ..region_names import edge_axis_outward_for_region

    target_sign = +1 if has_outward else -1  # outward = +1·outward_axis_sign
    desc_by_name = {r.name: (r.description or "")
                    for r in spec.geometry.custom_regions}
    dims = spec.geometry.dimensions or {}
    for lc in spec.bcs.loading:
        if lc.control.value != "displacement":
            continue
        if lc.magnitude is None:
            continue
        info = edge_axis_outward_for_region(
            lc.region, desc_by_name.get(lc.region, ""), dims)
        if info is None:
            continue
        edge_axis, outward_sign = info
        # If the architect picked a component on a different axis, the user
        # is probably describing a tangential pull (e.g. shear) — leave
        # alone.
        if lc.component is not None and lc.component != edge_axis:
            continue
        # Coerce component to the edge axis if not set.
        if lc.component is None:
            lc.component = edge_axis
        wanted = target_sign * outward_sign
        cur_sign = 1 if lc.magnitude >= 0 else -1
        if cur_sign != wanted:
            old = lc.magnitude
            lc.magnitude = wanted * abs(lc.magnitude)
            verb = "outward" if has_outward else "inward"
            emit(DECISION,
                 f"loading on {lc.region!r}: flipped magnitude "
                 f"{old:+g} -> {lc.magnitude:+g} to match user's "
                 f"{verb} verb")


def _ensure_rigid_body_constraint(spec: CanonicalSpec,
                                    raw_user_text: str) -> None:
    """Every FEA run must suppress all rigid-body modes (translations and,
    in 2D, rotation).  We check **per axis** which components are pinned
    by the existing fixed BCs and add a synthetic pin only for the axes
    that aren't already covered.

    Examples:
      * Roller-only setup (``bottom u_y=0`` + ``top u_y=-disp``): axis 0
        (x) is unconstrained → add a single-point pin fixing only u_x.
      * No fixed BCs at all: add a single-point full clamp.
      * Bottom fully clamped: every axis is covered → do nothing.

    The pin region is a single mesh vertex named ``rigid_body_pin``; the
    mesh LLM is told to match exactly ONE vertex, not an edge.
    """
    dim = 3 if spec.dimension.value == "3D" else 2
    all_axes = list(range(dim))

    def axes_pinned_by_fixed_bcs() -> set[int]:
        """Which u-components are pinned by ``bcs.fixed``?  Empty
        ``components`` means full clamp (all axes)."""
        out: set[int] = set()
        for f in spec.bcs.fixed:
            if not f.components:
                out.update(all_axes)
            else:
                for c in f.components:
                    if c in (0, 1, 2):
                        out.add(c)
        return out

    pinned = axes_pinned_by_fixed_bcs()
    missing = [a for a in all_axes if a not in pinned]
    if not missing:
        return

    from ..events import DECISION, emit
    from ..schema import FixedBC, RegionSpec
    # Only one synthetic pin region — its components list carries the
    # missing axes.  Empty list means full clamp (default for "no fixed
    # BCs at all").
    full_clamp = (len(missing) == dim)
    pin_components: List[int] = [] if full_clamp else list(missing)
    pin = FixedBC(region="rigid_body_pin", components=pin_components)
    spec.bcs.fixed.append(pin)
    emit(DECISION,
         f"rigid-body suppressor: added pin on axes {missing} "
         f"(component list {pin_components or 'full-clamp'})")
    spec.assumptions.append(
        f"Rigid-body-motion suppressor added: pin axes {missing} "
        f"(user-specified BCs left {dim - len(pinned)} axis/axes "
        f"unconstrained)")
    # Declare the pin region for the custom-mesh LLM.  Built-in meshes
    # would need this too in principle, but they're rectangular and the
    # rare case of "all BCs are rollers on a built-in plate" hasn't
    # surfaced in practice; if it does, the next step is to coerce to
    # custom (which `_coerce_to_custom_if_nonstandard_regions` does
    # automatically once an out-of-canon region is referenced).
    existing = {r.name for r in spec.geometry.custom_regions}
    if "rigid_body_pin" not in existing:
        spec.geometry.custom_regions.append(RegionSpec(
            name="rigid_body_pin",
            description=("a single vertex at the geometric centre of the "
                         "domain (or any convenient interior point that "
                         "does NOT overlap a loaded boundary).  This is a "
                         "rigid-body-motion suppressor — match exactly ONE "
                         "mesh vertex, not an edge.")))
    # If we added a pin to a built-in geometry, force the coercion to
    # custom — built-in meshes can't expose a "rigid_body_pin" region.
    if spec.geometry.kind in _BUILTIN_REGION_NAMES:
        _coerce_to_custom_if_nonstandard_regions(spec, raw_user_text)


def _preserve_raw_prompt_in_description(spec: CanonicalSpec,
                                         raw_user_text: str) -> None:
    """For kind=custom, guarantee the raw user prompt is embedded verbatim
    in ``custom_description`` so the downstream mesh LLM sees implicit
    geometry features (cracks, notches, holes, slits, fillets) the Architect
    may have summarised away."""
    if spec.geometry.kind != "custom":
        return
    desc = (spec.geometry.custom_description or "").strip()
    if raw_user_text and raw_user_text not in desc:
        if desc:
            spec.geometry.custom_description = (
                desc + "\n\nOriginal user prompt (authoritative):\n"
                + raw_user_text)
        else:
            spec.geometry.custom_description = raw_user_text


# Canonical region names emitted by each built-in mesh.  The single source
# of truth is ``fracture_agent.region_names.BUILTIN_MESH_REGIONS`` (which mirrors
# the ``markers_spec`` lists in ``modular/meshes/*.py``); we mirror it
# here with the architect's ``geometry.kind`` keys.  Keep the two tables
# in lockstep — drift causes silent ValueError at synthesis time.
def _build_builtin_region_names() -> dict:
    from ..region_names import BUILTIN_MESH_REGIONS
    kind_to_builder = {
        "notched_plate":    "make_notched_plate_2d",
        "plate_3d":         "make_notched_plate_3d",
        "slant_plate":      "make_slant_plate_2d",
        "dogbone_2d":       "make_dogbone_2d",
        "dogbone_3d":       "make_dogbone_3d",
    }
    return {kind: set(BUILTIN_MESH_REGIONS.get(builder, set()))
            for kind, builder in kind_to_builder.items()}


_BUILTIN_REGION_NAMES = _build_builtin_region_names()


# Per-kind whitelist of dimension keys the corresponding mesh builder
# actually consumes.  The architect's LLM occasionally drops the user's
# displacement / load value into ``dimensions[L]`` (or similar) — we
# strip any extra keys so they don't propagate downstream.
_KIND_DIMENSION_KEYS = {
    "notched_plate":    {"W", "L", "ac", "cw"},
    "plate_3d":         {"W", "L", "H", "ac", "cw", "thickness"},
    "slant_plate":      {"W", "H", "L", "c0", "theta", "cw"},
    "dogbone_2d":       {"W", "L", "R", "h_fine"},
    "dogbone_3d":       {"W", "L", "R", "thickness", "h_fine"},
    # custom / rectangle: keep all keys — the mesh-LLM may use any of them.
}


def _filter_geometry_dimensions(spec: CanonicalSpec) -> None:
    """Drop dimension keys that don't belong to the chosen geometry kind.

    For built-in kinds we whitelist; for ``custom`` / ``rectangle`` we
    keep everything (the mesh-LLM consumes the description directly so
    extra keys are harmless and sometimes informative).  Negative or
    non-finite values are also stripped — they're symptoms of LLM noise
    rather than user intent.
    """
    kind = spec.geometry.kind
    dims = spec.geometry.dimensions or {}
    if not dims:
        return
    allowed = _KIND_DIMENSION_KEYS.get(kind)
    out = {}
    dropped = []
    for k, v in dims.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            dropped.append((k, v))
            continue
        if not (fv > 0):
            dropped.append((k, v))
            continue
        if allowed is not None and k not in allowed:
            dropped.append((k, v))
            continue
        out[k] = fv
    if dropped:
        from ..events import DECISION, emit
        emit(DECISION,
             f"geometry.dimensions: filtered out keys "
             f"{[k for k,_ in dropped]} (kind={kind})")
    spec.geometry.dimensions = out


_NOTCHED_PLATE_KINDS = {"notched_plate", "plate_3d"}


def _notched_plate_supports_segments(spec: CanonicalSpec) -> bool:
    """Return True iff the spec's ``pre_crack_segments`` are representable
    by ``make_notched_plate_2d/3d``: at most one segment, horizontal, with
    one endpoint on the left face (x ≈ 0) at mid-height (y ≈ L/2 in the
    user frame).  Multi-crack / off-centre / vertical / interior cracks
    must drop to a custom mesh."""
    segs = spec.geometry.pre_crack_segments or []
    if not segs:
        return True   # nothing to violate
    if len(segs) > 1:
        return False
    s = segs[0]
    if not (len(s) == 2 and len(s[0]) >= 2 and len(s[1]) >= 2):
        return False
    (x1, y1), (x2, y2) = (s[0][0], s[0][1]), (s[1][0], s[1][1])
    try:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError):
        return False
    dims = spec.geometry.dimensions or {}
    W = float(dims.get("W", 0.0)) or 1.0
    L = float(dims.get("L", 0.0)) or 1.0
    tol = 1e-2 * max(W, L)
    if abs(y1 - y2) > tol:
        return False                                # not horizontal
    y_mid = L / 2.0
    if abs((y1 + y2) / 2.0 - y_mid) > tol:
        return False                                # not at mid-height
    if not (abs(x1) <= tol or abs(x2) <= tol):
        return False                                # not from the left face
    return True


def _coerce_to_custom_if_nonstandard_regions(spec: CanonicalSpec,
                                               raw_user_text: str = "") -> None:
    """If the Architect picked a built-in mesh kind but the BC regions don't
    match that mesh's canonical region names, OR the pre-crack pattern can't
    be expressed by the built-in mesh, flip to ``custom`` and let the mesh
    LLM build the predicates the Architect actually meant.

    This is the single most common source of downstream failure: the LLM
    invents plausible region names (``left_edge_upper_half``) while also
    picking a built-in mesh that only exposes (``top``, ``bottom``,
    ``left``, ``right``); or the user describes two staggered cracks but
    the built-in only carries a single horizontal edge crack.  Rather than
    reject the spec, we reconcile by elevating the geometry to custom —
    the user's intent (the names + the cracks) is preserved.
    """
    kind = spec.geometry.kind
    if kind not in _BUILTIN_REGION_NAMES:
        return
    coerce = False

    canonical = _BUILTIN_REGION_NAMES[kind]
    referenced = {f.region for f in spec.bcs.fixed}
    referenced |= {l.region for l in spec.bcs.loading}
    if referenced and not referenced.issubset(canonical):
        coerce = True

    if (not coerce) and (kind in _NOTCHED_PLATE_KINDS):
        if not _notched_plate_supports_segments(spec):
            coerce = True

    if not coerce:
        return

    # Out-of-canon names OR cracks the built-in can't carry → switch to custom.
    spec.geometry.kind = "custom"
    # IMPORTANT: when the Architect originally chose a built-in kind (e.g.
    # notched_plate) it may have implied geometry features (a pre-crack, a
    # notch, a hole) that are now lost when we drop to "custom".  We put the
    # raw user prompt into custom_description so the downstream mesh LLM
    # still sees every feature the user asked for.
    if raw_user_text:
        prior = (spec.geometry.custom_description or "").strip()
        spec.geometry.custom_description = (
            f"{raw_user_text}" + (f"\n\nNote: {prior}" if prior else ""))


# ---------------------------------------------------------------------------
def _extract_raw_user_text(fragments_history: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for turn in fragments_history:
        for inp in turn.get("inputs", []):
            if isinstance(inp, (tuple, list)) and len(inp) == 2 and inp[0] == "text":
                out.append(str(inp[1]))
            elif isinstance(inp, dict) and inp.get("kind") == "text":
                out.append(str(inp.get("payload", "")))
    return "\n".join(out).strip()


def _normalise_pre_crack_segments(pcs) -> List[List[List[float]]]:
    """Accept a wide variety of shapes the LLM might emit for pre_crack_segments
    and normalise to the canonical nested list form.

    Canonical form:
        [
          [[x1, y1], [x2, y2]],      # segment 0
          [[x3, y3], [x4, y4]],      # segment 1
          ...
        ]

    Tolerated inputs:
      1. Canonical form (pass-through)
      2. A single segment, no outer wrapper:
           [[0, 5], [5, 5]]   ->  [[[0, 5], [5, 5]]]
      3. List of dicts with start/end keys:
           [{"start": [0, 5], "end": [5, 5]}]
           ->  [[[0, 5], [5, 5]]]
      4. List of dicts with x1, y1, x2, y2 keys:
           [{"x1": 0, "y1": 5, "x2": 5, "y2": 5}]
      5. Flat quadruples:
           [0, 5, 5, 5]  ->  [[[0, 5], [5, 5]]]
           [[0, 5, 5, 5]]  ->  [[[0, 5], [5, 5]]]
      6. 3D points: strip to 2D (take first two components).
    """
    if not isinstance(pcs, list) or not pcs:
        return []

    def _pt(obj):
        if isinstance(obj, dict):
            for keys in (("x", "y"), ("X", "Y")):
                if all(k in obj for k in keys):
                    return [float(obj[keys[0]]), float(obj[keys[1]])]
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            try:
                return [float(obj[0]), float(obj[1])]
            except (TypeError, ValueError):
                return None
        return None

    def _seg(obj):
        # Dict with start/end
        if isinstance(obj, dict):
            if "start" in obj and "end" in obj:
                p1, p2 = _pt(obj["start"]), _pt(obj["end"])
                return [p1, p2] if p1 and p2 else None
            if all(k in obj for k in ("x1", "y1", "x2", "y2")):
                return [[float(obj["x1"]), float(obj["y1"])],
                        [float(obj["x2"]), float(obj["y2"])]]
            return None
        # List-like
        if isinstance(obj, (list, tuple)):
            if len(obj) == 2:
                p1, p2 = _pt(obj[0]), _pt(obj[1])
                if p1 and p2:
                    return [p1, p2]
            if len(obj) == 4:        # flat quadruple [x1, y1, x2, y2]
                try:
                    return [[float(obj[0]), float(obj[1])],
                            [float(obj[2]), float(obj[3])]]
                except (TypeError, ValueError):
                    return None
        return None

    # Case: pcs itself is a flat quadruple.
    if len(pcs) == 4 and all(isinstance(v, (int, float)) for v in pcs):
        s = _seg(pcs)
        return [s] if s else []

    # Case: pcs looks like a single segment [[x1,y1],[x2,y2]] (no outer wrap).
    if (len(pcs) == 2
        and all(isinstance(v, (list, tuple)) and len(v) >= 2
                and all(isinstance(x, (int, float)) for x in v[:2])
                for v in pcs)):
        s = _seg(pcs)
        return [s] if s else []

    # Canonical / mixed case: pcs is a list of segments.
    out: List[List[List[float]]] = []
    for item in pcs:
        s = _seg(item)
        if s is not None:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
def _reshape(spec: Dict[str, Any]) -> None:
    """Unflatten common LLM output variations into the nested CanonicalSpec
    shape Pydantic expects.  In-place; idempotent.

    Handles:
      * geometry fields at the top level (kind, dimensions, custom_*) being
        moved under spec['geometry'];
      * ``constitutive`` being wrapped as `{"kind": "<enum>"}` or
        `{"type": "<enum>"}`;
      * ``dimension`` being an int (2 / 3) instead of the enum string.
    """
    # 1. Geometry promotion: if top-level has geometry keys, move them.
    GEO_KEYS = ("kind", "dimensions", "custom_description", "custom_regions")
    flat_geom = {k: spec[k] for k in GEO_KEYS if k in spec}
    if flat_geom and "geometry" not in spec:
        spec["geometry"] = flat_geom
        for k in flat_geom: spec.pop(k, None)
    # Also handle when geometry exists partially and the rest is flat.
    elif flat_geom and "geometry" in spec:
        g = spec["geometry"]
        for k, v in flat_geom.items():
            g.setdefault(k, v)
            spec.pop(k, None)

    # 2. BC promotion.
    if "bcs" not in spec and any(k in spec for k in ("fixed", "loading", "free_regions")):
        spec["bcs"] = {}
        for k in ("fixed", "loading", "free_regions"):
            if k in spec:
                spec["bcs"][k] = spec.pop(k)

    # 3. Enum un-wrapping.
    for enum_field in ("constitutive", "dimension", "plane_type"):
        v = spec.get(enum_field)
        if isinstance(v, dict):
            # Try well-known keys first.
            for k in ("kind", "type", "value", "name"):
                if k in v and isinstance(v[k], (str, int)):
                    spec[enum_field] = v[k]
                    break
        elif isinstance(v, int) and enum_field == "dimension":
            spec[enum_field] = f"{v}D"

    # 4. Loading mode promotion.
    loading = spec.get("loading")
    if isinstance(loading, list):
        # LLM put the cases under top-level `loading`; move them into bcs.
        spec.setdefault("bcs", {})
        spec["bcs"].setdefault("loading", [])
        if not spec["bcs"]["loading"]:
            spec["bcs"]["loading"] = loading
        spec["loading"] = {"mode": "quasistatic", "steps": 100, "T_total": 1.0}


# ---------------------------------------------------------------------------
def _sanitise(reply: Dict[str, Any]) -> None:
    geom = reply.get("geometry") or {}
    dims = geom.get("dimensions") or {}
    geom["dimensions"] = {k: float(v) for k, v in dims.items() if v is not None}
    cr = geom.get("custom_regions") or []
    fixed_cr = []
    for r in cr:
        if isinstance(r, str):
            fixed_cr.append({"name": r, "description": ""})
        elif isinstance(r, dict):
            fixed_cr.append({"name": r.get("name", ""),
                             "description": r.get("description", "") or ""})
    geom["custom_regions"] = fixed_cr
    reply["geometry"] = geom
    bcs = reply.get("bcs") or {}
    bcs.setdefault("fixed", [])
    bcs.setdefault("loading", [])
    bcs.setdefault("free_regions", [])
    for f in bcs["fixed"]:
        if "components" not in f or f["components"] is None:
            f["components"] = []
    for l in bcs["loading"]:
        if "component" in l and l["component"] is not None:
            try:
                l["component"] = int(l["component"])
            except (TypeError, ValueError):
                l["component"] = 1
        # Coerce magnitude to float; sometimes the LLM leaks the clarification
        # answer ('accept', 'default') or a unit string ('1 mm') into this
        # field.  Extract the first numeric token if possible, else 0 (a
        # rigid_body_pin already protects us).
        mag = l.get("magnitude")
        if mag is None:
            l["magnitude"] = 0.0
        elif isinstance(mag, str):
            import re as _re
            hit = _re.search(r"-?\d+(?:\.\d+)?", mag)
            l["magnitude"] = float(hit.group(0)) if hit else 0.0
        else:
            try:
                l["magnitude"] = float(mag)
            except (TypeError, ValueError):
                l["magnitude"] = 0.0
        # Validate ``control``: schema only accepts displacement | force |
        # traction.  The LLM occasionally emits ``velocity`` / ``acceleration``
        # for dynamic prompts — coerce to ``traction`` (the most general)
        # and let downstream resolve magnitude-units.  Unknown strings get
        # the same treatment.
        ctrl = (l.get("control") or "").strip().lower()
        VALID_CTRL = {"displacement", "force", "traction"}
        if ctrl not in VALID_CTRL:
            COERCE = {
                "velocity":      "traction",
                "acceleration":  "traction",
                "pressure":      "traction",
                "stress":        "traction",
                "load":          "force",
            }
            l["control"] = COERCE.get(ctrl, "displacement")
        else:
            l["control"] = ctrl
    reply["bcs"] = bcs

    # pre_crack_segments: List[List[List[float]]] — a list of segments,
    # each segment is [[x1, y1], [x2, y2]].  LLMs often flatten a level,
    # send dicts, or emit 3D points — normalise all of them.  An LLM-emitted
    # ``None`` is the same as "no segments" — strip rather than crash.
    geom = reply.get("geometry") or {}
    pcs = geom.get("pre_crack_segments")
    if pcs is None:
        # Drop the None key so Pydantic uses the default_factory (empty list).
        geom.pop("pre_crack_segments", None)
    else:
        geom["pre_crack_segments"] = _normalise_pre_crack_segments(pcs)
    # Same defensive strip for ``custom_regions`` and ``dimensions``.
    if geom.get("custom_regions") is None:
        geom.pop("custom_regions", None)
    if geom.get("dimensions") is None:
        geom.pop("dimensions", None)
    reply["geometry"] = geom

    # LoadingSpec: drop explicit Nones so Pydantic's defaults (steps=100,
    # T_total=1.0, mode=quasistatic) apply.  The LLM occasionally returns
    # ``loading: {steps: null, T_total: null}`` for prompts that don't say
    # how many steps to run.
    loading_spec = reply.get("loading")
    if isinstance(loading_spec, dict):
        for k in ("steps", "T_total", "mode"):
            if loading_spec.get(k) is None:
                loading_spec.pop(k, None)
        reply["loading"] = loading_spec
    # F13 — same defensive None-strip for the other top-level scalars/objects
    # that have schema defaults.  An extreme amateur prompt like "plate" can
    # cause the LLM to fill every field with ``null``; without this, Pydantic
    # rejects the spec entirely instead of falling back to defaults.
    if reply.get("loading") is None:
        reply.pop("loading", None)
    for k in ("plane_type", "constitutive", "dimension"):
        if k in reply and reply[k] is None:
            reply.pop(k)
    if reply.get("mesh") is None:
        reply.pop("mesh", None)
    if reply.get("bcs") is None:
        reply.pop("bcs", None)

    # Material: ensure sub-object is present, and alias common short-form
    # keys to their canonical MaterialSpec field names.
    if "material" not in reply or reply["material"] is None:
        reply["material"] = {}
    mat = reply["material"]
    _MAT_ALIASES = {
        "sts": "sigma_ts", "sigma_t": "sigma_ts",
        "tensile_strength": "sigma_ts", "ult_tensile": "sigma_ts",
        "scs": "sigma_cs", "sigma_c": "sigma_cs",
        "compressive_strength": "sigma_cs",
        "shs": "sigma_hs", "sigma_h": "sigma_hs",
        "hydrostatic_strength": "sigma_hs",
        "sy": "sigma_y0", "yield": "sigma_y0",
        "sigma_y": "sigma_y0", "sigma_yield": "sigma_y0",
        "density": "rho", "rho_0": "rho",
        "youngs_modulus": "E", "young_modulus": "E",
        "poisson": "nu", "poisson_ratio": "nu",
        "fracture_toughness": "Gc", "toughness_gc": "Gc",
    }
    for alias, canonical in list(_MAT_ALIASES.items()):
        if alias in mat and canonical not in mat:
            mat[canonical] = mat.pop(alias)
        elif alias in mat and mat.get(canonical) is None:
            mat[canonical] = mat.pop(alias)
    reply["material"] = mat


# ---------------------------------------------------------------------------
def _enforce_catalog_match(spec: CanonicalSpec,
                            mats: Dict[str, Any]) -> None:
    """Strip catalog_name when it doesn't actually correspond to what the
    user said.  The LLM has a habit of reaching for the nearest catalog entry
    even when the user named a material that isn't in the database.

    Also strip when the user supplied explicit numeric properties — the
    runtime script `load_material('<catalog_name>')` would otherwise ignore
    them and silently run with catalog values."""
    m = spec.material
    if not m.catalog_name:
        return
    if m.catalog_name not in mats:
        m.catalog_name = None
        return
    # If the user gave numbers, honour those — NEVER a catalog.
    numeric_fields = ("E", "nu", "Gc", "sigma_ts", "sigma_cs", "rho",
                       "sigma_y0", "H_hardening", "mu1", "mu2",
                       "alpha1", "alpha2", "kappa", "sigma_hs")
    if any(getattr(m, f, None) is not None for f in numeric_fields):
        m.catalog_name = None
        return
    user_word = (m.display_name or "").lower().strip()
    if not user_word:
        return
    catalog_family = m.catalog_name.split("_", 1)[0].lower()      # "steel", "ceramic", etc.
    # A match is accepted only if the catalog family name appears in the user
    # word.  "steel" -> "Steel_bench_..." ok.  "graphite" -> "Steel_..." not
    # ok because "steel" doesn't appear in "graphite".
    if catalog_family not in user_word and user_word not in catalog_family:
        spec.material.catalog_name = None


# ---------------------------------------------------------------------------
_NUM_WITH_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\b", re.I)
_NUM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def _backfill_dimensions(spec: CanonicalSpec, raw_text: str) -> None:
    """When the LLM emits empty `dimensions`, harvest numbers with length
    units from the user prompt (mm/cm/m).  We collect every number with a
    unit suffix, dedupe, and store the first two as W and L.  This is
    intentionally coarse: its only job is to keep mesh generation from
    starving if the LLM forgot to transcribe dimensions."""
    if spec.geometry.dimensions:
        return
    hits = [(float(n), u.lower()) for n, u in _NUM_WITH_UNIT_RE.findall(raw_text)]
    if not hits:
        return
    # Normalise everything to mm.
    mm: List[float] = []
    for v, u in hits:
        if u == "mm": mm.append(v)
        elif u == "cm": mm.append(v * 10.0)
        elif u == "m":  mm.append(v * 1000.0)
    # Keep the order but dedupe nearby duplicates.
    unique: List[float] = []
    for v in mm:
        if not unique or abs(v - unique[-1]) > 1e-9:
            unique.append(v)
    if len(unique) >= 2:
        spec.geometry.dimensions = {"W": unique[0], "L": unique[1]}
    elif len(unique) == 1:
        spec.geometry.dimensions = {"W": unique[0], "L": unique[0]}


# ---------------------------------------------------------------------------
def _ensure_custom_fields(spec: CanonicalSpec, raw_text: str) -> None:
    """Guarantee that kind='custom' has description + regions with non-empty
    descriptions.  Also drop any custom_region name the BCs don't reference
    — leaving orphan regions in the spec can cause the mesh LLM to emit
    duplicate markers (two tags on the same facets), which makes
    ``dolfinx.meshtags`` behave unpredictably."""
    if spec.geometry.kind not in ("custom",):
        return
    if not spec.geometry.custom_description:
        spec.geometry.custom_description = raw_text

    # Collect the names the BCs actually use.
    referenced = {f.region for f in spec.bcs.fixed}
    referenced |= {l.region for l in spec.bcs.loading}
    referenced.discard(None)
    referenced.discard("")

    # Keep only regions the BCs reference, preserving user-provided order
    # and descriptions.  Add any BC-referenced region that the Architect
    # forgot to declare.
    existing = {r.name: r for r in spec.geometry.custom_regions
                if r.name in referenced}
    from ..schema import RegionSpec
    for name in referenced:
        if name not in existing:
            existing[name] = RegionSpec(name=name, description="")
    for r in existing.values():
        if not r.description:
            r.description = _describe_region_from_name(r.name)
    spec.geometry.custom_regions = list(existing.values())


def _describe_region_from_name(name: str) -> str:
    """Generate a best-effort predicate hint from a region name.
    E.g.  "left_upper_half"  → "x == x_min AND y >= y_mid (upper half of left edge)".
    This is intentionally heuristic — the mesh LLM still has the full user
    prompt as context and will disambiguate."""
    n = name.lower()
    parts: List[str] = []
    if "bottom" in n:      parts.append("y == y_min")
    if "top" in n:         parts.append("y == y_max")
    if "left" in n:        parts.append("x == x_min")
    if "right" in n:       parts.append("x == x_max")
    if "front" in n:       parts.append("z == z_min")
    if "back" in n:        parts.append("z == z_max")
    if "upper" in n or "top_half" in n or "upper_half" in n:
        parts.append("y >= (y_min + y_max) / 2")
    if "lower" in n or "bottom_half" in n or "lower_half" in n:
        parts.append("y <= (y_min + y_max) / 2")
    if "inner" in n:       parts.append("interior edge of the domain")
    if "outer" in n:       parts.append("outer edge of the domain")
    return " AND ".join(parts) or f"region named '{name}' (infer from geometry)"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# F16 — Multi-stage loading ("first hold X, then ramp Y") detection.
#
# Schema gives every LoadingCase a (t_start, t_end) pseudo-time window
# defaulting to (0, 1) — i.e. linear ramp over the whole simulation, the
# legacy single-stage behaviour.  When the user prompt contains explicit
# sequencing language we partition the time domain instead.  Conservative:
# only fires when (a) sequencing keywords are present, (b) there are 2+
# displacement loadings, and (c) NONE of them already has a non-default
# window (so we don't override the LLM if it was smart enough to set it).
# ---------------------------------------------------------------------------
_SEQUENCE_PAT = re.compile(
    r"\b("
    r"first(\s+(shift|move|push|pulled|displace|pre[-\s]?\w+))?|"
    r"phase\s*[12345]|"
    r"stage\s*[12345]|"
    r"then\s+(pull|ramp|apply|increase|displace|push|move)|"
    r"after\s+(the|this|that|holding|the\s+pre)|"
    r"subsequent(ly)?|"
    r"held\s+constant|"
    r"pre[-\s]?(shear|load|stress|strain|disp(lacement)?)|"
    r"while\s+(holding|maintaining)"
    r")\b",
    re.IGNORECASE,
)


def _detect_multi_stage_loading(spec: CanonicalSpec, raw_user_text: str) -> None:
    """If the raw prompt sequences loading actions ("first held, then
    pulled") and the architect emitted multiple displacement loadings, set
    each case's (t_start, t_end) so the templated RampProxy emits the
    right piecewise schedule.  No-op when fewer than 2 loadings or when
    the LLM already populated non-default windows.
    """
    if not raw_user_text:
        return
    disp = [lc for lc in spec.bcs.loading
            if lc.control.value == "displacement"]
    if len(disp) < 2:
        return
    # If the LLM already set non-default windows on any case, trust it.
    if any((lc.t_start != 0.0 or lc.t_end != 1.0) for lc in disp):
        return
    if not _SEQUENCE_PAT.search(raw_user_text):
        return

    n = len(disp)
    # Equal time slices.  Each "earlier" stage holds at its peak after its
    # own window ends — that's exactly the piecewise behaviour the
    # RampProxy implements (value = magnitude for t > t_end).
    from ..events import DECISION, emit
    notes = []
    for i, lc in enumerate(disp):
        t0 = i / n
        t1 = (i + 1) / n
        lc.t_start = round(t0, 6)
        lc.t_end   = round(t1, 6)
        notes.append(f"loading[{lc.region}/comp{lc.component}]: "
                     f"window [{t0:.2f}, {t1:.2f}]")
    emit(DECISION,
         "Multi-stage loading detected — partitioning pseudo-time: "
         + "; ".join(notes))
    spec.assumptions.append(
        f"Sequencing language detected; loadings split into {n} equal "
        "pseudo-time stages (each held at peak after its own window).")


def _fill_default_magnitudes(spec: CanonicalSpec) -> None:
    """For displacement loadings the user described qualitatively (no
    explicit number — common for prompts like ``"uniaxial tensile test of
    a 50 mm bar"``), fill in a sensible default scaled to the geometry:
    **1% of the largest in-plane dimension**.  This keeps strains in a
    realistic range (~1%) and lets fracture nucleate at a normal load
    level instead of either zero (no deformation) or a hard-coded constant
    that may be enormous on small specimens or invisible on large ones.

    Only fires when the magnitude is missing/zero AND the architect's
    direction-sign post-processor hasn't already injected a sign — we
    preserve sign and only set the absolute value.
    """
    dims = spec.geometry.dimensions or {}
    sizes = []
    for v in dims.values():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            sizes.append(fv)
    # Two regimes:
    #   * ``sizes`` non-empty → 1% of the largest dimension.
    #   * ``sizes`` empty (geometry dimensions unknown) → conservative
    #     literal 0.1 mm (1% of a typical 10 mm bench specimen).  The
    #     architect emits a DECISION warning either way.
    if sizes:
        default_disp = 0.01 * max(sizes)
        rationale = f"1% of max dimension {max(sizes):g} mm"
    else:
        default_disp = 0.1
        rationale = "fallback default 0.1 mm (no dimensions provided)"
    from ..events import DECISION, emit
    # Traction and force defaults are scaled to material strength when known
    # (~0.5·sigma_ts as a "comfortably above first-yield" probe load).
    # If sigma_ts is unknown we fall back to a literal 1 MPa / 1 N.
    sigma_ts = float(spec.material.sigma_ts) if spec.material.sigma_ts else 0.0
    default_traction = 0.5 * sigma_ts if sigma_ts > 0 else 1.0   # MPa
    default_force    = 1.0                                       # N
    for lc in spec.bcs.loading:
        ctrl = lc.control.value
        mag = lc.magnitude
        if mag is None or abs(float(mag)) < 1e-12:
            sign = 1 if (mag is None or mag >= 0) else -1
            if ctrl == "displacement":
                new_mag = sign * default_disp
                why = rationale
            elif ctrl == "traction":
                new_mag = sign * default_traction
                why = (f"0.5·sigma_ts = {default_traction:g} MPa"
                       if sigma_ts > 0
                       else "fallback default 1 MPa (no sigma_ts available)")
            elif ctrl == "force":
                new_mag = sign * default_force
                why = "fallback default 1 N"
            else:
                continue
            lc.magnitude = new_mag
            emit(DECISION,
                 f"loading on {lc.region!r}: no magnitude given, defaulted "
                 f"to {lc.magnitude:+g} ({ctrl}; {why})")
            spec.assumptions.append(
                f"Loading magnitude on {lc.region!r} defaulted to "
                f"{lc.magnitude:+g} ({ctrl}; {why})")


# ---------------------------------------------------------------------------
# Strip material-property clarification questions when the material name is
# resolvable (catalog hit or cache/handbook lookup will populate values).
# Otherwise, the architect loops up to 3 rounds asking the user for E, nu,
# Gc on materials like "graphite" that the material_helper would resolve
# automatically.
# ---------------------------------------------------------------------------
_MATERIAL_PROP_QUESTION_PAT = re.compile(
    r"\b("
    r"young'?s?\s+modulus|elastic\s+modulus|\bE\b\s*\(|"
    r"poisson|nu\b|"
    r"fracture\s+energy|fracture\s+toughness|\bGc\b|G_?c\b|"
    r"tensile\s+strength|sigma[_\s]?ts|"
    r"compressive\s+strength|sigma[_\s]?cs|"
    r"yield(\s+stress|\s+strength)?|sigma[_\s]?y0?|"
    r"hardening|H[_\s]?hardening|n[_\s]?hardening|"
    r"density|\brho\b|"
    r"shear\s+modulus|bulk\s+modulus|kappa|"
    r"mu1|mu2|alpha1|alpha2|eta1|eta2|"
    r"hydrostatic\s+strength|sigma[_\s]?hs"
    r")\b",
    re.IGNORECASE)


def _strip_resolvable_material_clarifications(
        spec: CanonicalSpec,
        open_q: List[str],
        mats: Dict[str, Any]) -> List[str]:
    """Drop any open question about a material property when the material
    will be resolved automatically — i.e. when ``display_name`` (or
    ``catalog_name``) is set and the user hasn't already provided numeric
    overrides.  ``material_helper.resolve_material`` handles catalog hits,
    on-disk cache, and Gemini handbook lookup, so re-asking the user is
    redundant churn.
    """
    if not open_q:
        return open_q
    m = spec.material
    has_name = bool((m.display_name or "").strip()
                    or (m.catalog_name or "").strip())
    if not has_name:
        return open_q
    numeric_fields = ("E", "nu", "Gc", "sigma_ts", "sigma_cs", "rho",
                      "sigma_y0", "H_hardening", "mu1", "mu2",
                      "alpha1", "alpha2", "kappa", "sigma_hs")
    user_already_gave_numbers = any(
        getattr(m, f, None) is not None for f in numeric_fields)
    if user_already_gave_numbers:
        return open_q   # the user gave enough; whatever's left is real
    kept: List[str] = []
    dropped: List[str] = []
    for q in open_q:
        if _MATERIAL_PROP_QUESTION_PAT.search(q):
            dropped.append(q)
        else:
            kept.append(q)
    if dropped:
        from ..events import DECISION, emit
        emit(DECISION,
             f"material '{m.display_name or m.catalog_name}' will be "
             f"resolved by handbook lookup; suppressed "
             f"{len(dropped)} architect clarification(s): "
             + "; ".join(d.strip() for d in dropped[:3])
             + (" ..." if len(dropped) > 3 else ""))
    return kept


def _sanity_loading(spec: CanonicalSpec, open_q: List[str]) -> None:
    """If loading list is empty, flag a question.  If any magnitude is 0 or
    missing, flag a question.  (``_fill_default_magnitudes`` should have
    populated reasonable defaults before this runs, so this only fires for
    structural problems with the spec.)"""
    if not spec.bcs.loading:
        open_q.append("No loading applied — where and how much should the "
                      "specimen be pulled / pushed?")
        return
    for lc in spec.bcs.loading:
        if lc.magnitude in (0.0, None):
            open_q.append(
                f"Loading on region {lc.region!r} has magnitude 0 — "
                f"what value should it be?")


# ---------------------------------------------------------------------------
# Stub-response detection — Gemini sometimes fills responseSchema with
# type-level placeholders when it can't figure out the real content.
# ---------------------------------------------------------------------------
_STUB_STRINGS = {"string", "STRING", "<string>", "example", "placeholder"}
_STUB_NUMBERS = {123.0, 1.23}


def _looks_like_stub(reply: Dict[str, Any]) -> bool:
    """Heuristic stub-detector for Gemini placeholder output."""
    def walk(node: Any, depth: int = 0) -> int:
        """Return the number of stub-like leaves in the tree."""
        if isinstance(node, dict):
            return sum(walk(v, depth + 1) for v in node.values())
        if isinstance(node, list):
            return sum(walk(v, depth + 1) for v in node)
        if isinstance(node, str) and node.strip() in _STUB_STRINGS:
            return 1
        if isinstance(node, (int, float)) and float(node) in _STUB_NUMBERS:
            return 1
        return 0
    return walk(reply) >= 3
