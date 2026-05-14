"""Material resolution — the 'say steel → ask me' workflow.

Rules:
  1. If the user gives an explicit catalog-name that exists in materials.json,
     use it.
  2. Otherwise, if they provide E, nu, Gc (+ type) we build an ad-hoc entry
     and return a MaterialSpec with catalog_name=None.
  3. If they give only a vague name ("steel"), we first check whether any
     catalog entry reasonably matches; if not, we ask the LLM — which has
     been instructed to source typical engineering handbook values and
     return a fully-specified block.  The user always sees the values and
     can override.
"""
from __future__ import annotations
from typing import Dict, Optional

from ..cache import get_material, put_material
from ..config import FAST_MODEL
from ..knowledge import list_builtin_materials
from ..llm import llm
from ..schema import CanonicalSpec, Constitutive, MaterialSpec


SYS_SEARCH = """You are a materials-database expert for FEA phase-field
fracture modelling.  You have a Google Search tool — USE IT for any
material whose numbers you are not certain about (exotic alloys, polymers,
composites, biological tissues, etc.).

Given a user material description ("steel", "mild steel", "epoxy", "ABS
plastic", "PEEK", "ash wood", "cortical bone", ...) and the problem type
(linear_elasticity / dynamic_linear_elasticity / ductile /
finite_elasticity), return a JSON object with the required properties.

ALWAYS PROVIDE  (units: mm, N, MPa, s)
  * `E` (Young's modulus, MPa) and `nu`
  * `Gc` (critical energy release rate, N/mm)
  * `sigma_ts` (ultimate tensile strength, MPa) and `sigma_cs` (>= 2*sigma_ts)
  * `eta` = 1e-5 (default)

ADDITIONAL FIELDS BY TYPE
  * ductile                      -> sigma_y0, H_hardening,
                                    n_hardening (default 1), sigma_ts_factor 2
  * dynamic_linear_elasticity    -> rho (density, tonne/mm^3; ~8e-9 for metals,
                                    2.5e-9 for glass, 2.7e-9 for aluminium)
  * finite_elasticity (rubber)   -> mu1, mu2, alpha1, alpha2, kappa,
                                    sigma_hs, eta1=1e-3, eta2=1e-5

Use real published handbook values.  When using Google Search, prefer
authoritative sources (MatWeb, ASM handbook, university course notes,
peer-reviewed papers).  Cite the source URL in a `source` field.

Respond with ONLY the JSON object — no markdown fences, no commentary.
"""


def _guess_problem_type(spec: CanonicalSpec) -> str:
    if spec.constitutive == Constitutive.j2_plasticity:
        return "ductile"
    if spec.constitutive == Constitutive.lopez_pamies:
        return "finite_elasticity"
    if spec.loading.mode.value == "dynamic":
        return "dynamic_linear_elasticity"
    return "linear_elasticity"


def resolve_material(spec: CanonicalSpec,
                     user_description: str = "") -> MaterialSpec:
    """Return a MaterialSpec with either a catalog_name or explicit fields.

    The function does NOT ask the user — the orchestrator does that — but it
    returns a completed spec if the user had already supplied values or if a
    catalog match is definitive.  When the user says "steel" with no numbers
    we call the LLM to propose handbook values.
    """
    m = spec.material
    # 1. user gave a catalog name → trust it.
    if m.catalog_name and m.catalog_name in list_builtin_materials():
        return m

    # 2. numbers already present → use them as-is.
    if any(v is not None for v in (m.E, m.mu1)) and m.Gc is not None:
        m.problem_type = m.problem_type or _guess_problem_type(spec)
        return m

    # 3. LLM handbook lookup — grounded with Google Search so exotic and
    # recently-reported materials get real numbers (MatWeb, ASM, papers).
    # We cache on-disk so repeat prompts (steel, copper, ...) are free.
    ptype = _guess_problem_type(spec)
    user_desc = user_description or m.display_name or "(unspecified)"
    cached = get_material(user_desc, ptype)
    if cached:
        reply = cached
    else:
        prompt = (f"User material description: {user_desc}\nProblem type: {ptype}\n"
                  f"Look up authoritative values (MatWeb, handbook, peer-reviewed) "
                  f"and return the JSON block only.")
        try:
            # Flash is fine for structured handbook lookup and much faster;
            # the search grounding is what gives us authoritative numbers.
            text = llm().complete(SYS_SEARCH, prompt,
                                  temperature=0.1,
                                  max_output_tokens=16384,
                                  use_google_search=True,
                                  model=FAST_MODEL)
            from ..llm import _parse_json_loose
            reply = _parse_json_loose(text)
        except Exception:
            reply = llm().complete_json(SYS_SEARCH, prompt,
                                        temperature=0.1, model=FAST_MODEL)
        put_material(user_desc, ptype, reply)

    # n_hardening is typed as int in MaterialSpec but the LLM occasionally
    # returns a Ramberg-Osgood-style fractional exponent (e.g. 0.2).  Round
    # to the nearest int so Pydantic accepts it.
    def _coerce_int(v, default=1):
        if v is None:
            return default
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return default

    fresh = MaterialSpec(
        display_name=m.display_name or user_desc,
        problem_type=ptype,
        units=reply.get("units", "mm, N, MPa, s"),
        E=reply.get("E"), nu=reply.get("nu"), Gc=reply.get("Gc"),
        sigma_ts=reply.get("sigma_ts"), sigma_cs=reply.get("sigma_cs"),
        sigma_y0=reply.get("sigma_y0"),
        H_hardening=reply.get("H_hardening"),
        n_hardening=_coerce_int(reply.get("n_hardening", 1)),
        sigma_ts_factor=reply.get("sigma_ts_factor", 2.0),
        rho=reply.get("rho"),
        mu1=reply.get("mu1"), mu2=reply.get("mu2"),
        alpha1=reply.get("alpha1"), alpha2=reply.get("alpha2"),
        kappa=reply.get("kappa"), sigma_hs=reply.get("sigma_hs"),
        eta=reply.get("eta", 1.0e-5),
        eta1=reply.get("eta1", 1.0e-3),
        eta2=reply.get("eta2", 1.0e-5),
    )
    _backfill_required_fields(fresh, ptype)
    return fresh


def _backfill_required_fields(m: MaterialSpec, ptype: str,
                                 spec=None) -> None:
    """The modular ``loader`` derives ``Wts`` from ``sigma_ts`` and (for
    ductile/dynamic) needs ``sigma_y0``, ``rho`` etc.  When the handbook
    LLM omits one of these — common for less-studied materials —
    ``derive_length_scales`` raises and the orchestrator falls back to
    ``eps=0.25`` (fine, but losing material specificity).  Fill the gaps
    with conservative engineering correlations so the loader succeeds.

    ``spec`` is the parent CanonicalSpec — when provided, every backfill
    note is also appended to ``spec.assumptions`` so the user sees the
    audit trail.  Pass-through is optional (the function still works
    without it; the notes just don't reach spec.assumptions).

    None of these fallbacks attempt to author novel constitutive
    behaviour — they're just defensive numerics derived from values the
    LLM did return.
    """
    from ..events import DECISION, emit
    notes = []

    # E is non-negotiable for any phase-field run — without it nothing
    # downstream can compute eps.  Bail so the orchestrator's eps=0.25
    # fallback fires (and at least the user sees the eps-fallback ERROR).
    if m.E is None:
        return

    # Gc is required for any phase-field run, but the handbook LLM
    # occasionally omits it for ductile materials (it gives sigma_y0
    # instead) or for less-studied materials.  Without a Gc the
    # downstream ``derive_length_scales`` raises and the orchestrator
    # falls back to ``eps=0.25`` (silent loss of material specificity).
    # Use a problem-type-typed default — these are conservative low-end
    # handbook values so a real-but-unlisted material isn't blown out by
    # a wildly wrong number.  The assumption note tells the user.
    _GC_TYPED_DEFAULT = {
        "linear_elasticity":         0.1,    # brittle ceramics / rocks low end
        "dynamic_linear_elasticity": 0.1,
        "ductile":                   50.0,   # mild steel / copper low end
        "finite_elasticity":         1.0,    # natural rubber low end
    }
    if m.Gc is None or m.Gc <= 0:
        Gc_default = _GC_TYPED_DEFAULT.get(ptype, 0.1)
        m.Gc = Gc_default
        notes.append(
            f"Gc was missing from the handbook lookup; defaulted to "
            f"{Gc_default} N/mm (typed low-end for {ptype}).  "
            "Override with an explicit Gc=<value> in the prompt if you "
            "have a better number.")

    # 1) sigma_ts — drives Wts = 0.5*sigma_ts^2/E.  Three fallbacks in
    #    order of preference:
    #    (a) sigma_y0 known          → sigma_ts ≈ sigma_ts_factor * sigma_y0
    #    (b) E and Gc known          → sigma_ts ≈ 0.005·E (0.5% of E,
    #        order-of-magnitude correct for many engineering materials)
    #    (c) E known                 → sigma_ts ≈ 0.001·E (deeply conservative)
    #
    #    Earlier versions used ``sqrt(2·Gc·E)`` which over-estimates by 10×
    #    for tough materials (titanium, copper) — switched to the 0.5%·E
    #    correlation which is more representative of real bench values.
    if m.sigma_ts is None or m.sigma_ts <= 0:
        if m.sigma_y0 and m.sigma_y0 > 0:
            factor = float(m.sigma_ts_factor or 2.0)
            m.sigma_ts = factor * float(m.sigma_y0)
            notes.append(f"sigma_ts inferred from {factor}*sigma_y0 = {m.sigma_ts:g}")
        else:
            # 0.5% of E lands in 100-1500 MPa range for typical metals,
            # 10-50 MPa for polymers/concrete (Gc ranges line up too).
            m.sigma_ts = 0.005 * float(m.E)
            notes.append(f"sigma_ts inferred as 0.5% of E = {m.sigma_ts:g} "
                         f"(handbook lookup did not return tensile strength)")

    # 2) sigma_cs — modular's compressive surface uses 2*sigma_ts at
    #    minimum.  Some catalog entries leave it None.
    if m.sigma_cs is None or m.sigma_cs <= 0:
        m.sigma_cs = 2.0 * float(m.sigma_ts)
        notes.append(f"sigma_cs defaulted to 2*sigma_ts = {m.sigma_cs:g}")

    # 3) ductile — sigma_y0 backfill from sigma_ts/sigma_ts_factor.
    if ptype == "ductile":
        if m.sigma_y0 is None or m.sigma_y0 <= 0:
            factor = float(m.sigma_ts_factor or 2.0)
            m.sigma_y0 = float(m.sigma_ts) / max(factor, 1.0)
            notes.append(f"sigma_y0 inferred from sigma_ts/{factor} = {m.sigma_y0:g}")
        if m.H_hardening is None:
            m.H_hardening = 0.01 * float(m.E)
            notes.append(f"H_hardening defaulted to 1% of E = {m.H_hardening:g}")

    # 4) dynamic — rho fallback (8e-9 tonne/mm^3 typical metal).
    if ptype == "dynamic_linear_elasticity":
        if m.rho is None or m.rho <= 0:
            m.rho = 8.0e-9
            notes.append("rho defaulted to 8e-9 tonne/mm^3 (typical metal)")

    # 5) finite-elasticity — sigma_hs (hydrostatic strength) defaults to
    #    2*sigma_ts.  mu1/mu2/kappa/alpha1/alpha2 are model-specific and
    #    we DON'T invent them — the LLM should always have returned them
    #    for a finite-elasticity prompt.
    if ptype == "finite_elasticity":
        if (m.sigma_hs is None or m.sigma_hs <= 0):
            m.sigma_hs = 2.0 * float(m.sigma_ts)
            notes.append(f"sigma_hs defaulted to 2*sigma_ts = {m.sigma_hs:g}")

    if notes:
        msg = (f"material '{m.display_name or m.catalog_name or '?'}': "
               "backfilled missing fields — " + "; ".join(notes))
        emit(DECISION, msg)
        if spec is not None and hasattr(spec, "assumptions"):
            for n in notes:
                spec.assumptions.append(
                    f"Material '{m.display_name or m.catalog_name or '?'}' "
                    f"backfill: {n}")
