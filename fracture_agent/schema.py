"""Canonical problem spec — the single source of truth the agents pass around.

Pydantic is preferred (runtime validation, cheap JSON round-trip).  The fields
are deliberately shallow and named after the vocabulary used inside `modular/`
so that downstream code-generation is a direct mapping.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Dimension(str, Enum):
    two = "2D"
    three = "3D"


class PlaneType(str, Enum):
    plane_strain = "plane_strain"
    plane_stress = "plane_stress"
    na = "na"                                            # 3D or not applicable


class Constitutive(str, Enum):
    linear_elastic = "linear_elastic"
    j2_plasticity  = "j2_plasticity"                     # ductile
    lopez_pamies   = "lopez_pamies"                      # finite elasticity (rubber)


class Loading(str, Enum):
    quasistatic = "quasistatic"
    dynamic     = "dynamic"


class Control(str, Enum):
    displacement = "displacement"
    force        = "force"
    traction     = "traction"                            # pressure / dynamic


# ---------------------------------------------------------------------------
# Sub-specs
# ---------------------------------------------------------------------------
class MaterialSpec(BaseModel):
    """Either a name from materials.json, or a fully user-supplied block.

    If `catalog_name` is set, it will be loaded via `load_material`.  Otherwise
    the remaining fields are used to build an ad-hoc entry on the fly (the
    agent writes a temporary JSON entry into the runs/ directory).
    """
    catalog_name: Optional[str] = None                   # e.g. "Steel_bench_2D_PE"
    display_name: Optional[str] = None                   # user-provided (e.g. "Steel")
    problem_type: Optional[Literal[
        "linear_elasticity", "dynamic_linear_elasticity",
        "ductile", "finite_elasticity"]] = None
    units: str = "mm, N, MPa, s"
    # Primary properties (linear + ductile + dynamic)
    E: Optional[float] = None
    nu: Optional[float] = None
    Gc: Optional[float] = None
    sigma_ts: Optional[float] = None
    sigma_cs: Optional[float] = None
    rho: Optional[float] = None                          # dynamic
    # Ductile
    sigma_y0: Optional[float] = None
    H_hardening: Optional[float] = None
    n_hardening: Optional[int] = 1
    sigma_ts_factor: Optional[float] = 2.0
    # Finite elasticity
    mu1: Optional[float] = None
    mu2: Optional[float] = None
    alpha1: Optional[float] = None
    alpha2: Optional[float] = None
    kappa: Optional[float] = None
    sigma_hs: Optional[float] = None
    eta:  Optional[float] = 1.0e-5
    eta1: Optional[float] = 1.0e-3
    eta2: Optional[float] = 1.0e-5


class RegionSpec(BaseModel):
    """A named boundary region the mesh builder must produce.

    For built-in meshes the name typically matches one of the canonical
    labels ("top", "bottom", "left", "right", "front", "back").  For
    *custom* geometries the Architect invents the name and writes a free-text
    ``description`` telling the custom-mesh LLM how to locate it — e.g.
    "upper half of the left edge" or "the inner horizontal edge of the L".
    """
    name: str
    description: str = ""


class FixedBC(BaseModel):
    """u = 0 on ``region`` for the listed components.

    Components: 0 = x, 1 = y, 2 = z.  Empty list means "pin all components"
    (useful shorthand for full-clamp).
    """
    region: str
    components: List[int] = Field(default_factory=list)


class LoadingCase(BaseModel):
    """One applied displacement / force / traction schedule.

    Multi-stage loading: ``(t_start, t_end)`` define the active window in
    pseudo-time t∈[0,1].  Outside the window the value is 0 before
    ``t_start`` and held at ``magnitude`` after ``t_end``.  Defaults
    ``(0.0, 1.0)`` recover the legacy "linear ramp 0 → magnitude over the
    whole simulation" behaviour, so existing specs are unchanged.

    Examples:
      * Single-stage tension: ``t_start=0, t_end=1``  (default)
      * Pre-shear held + then-tensile: case A has ``t_start=0, t_end=0.3``
        (ramp + hold from t=0.3 onwards); case B has ``t_start=0.3,
        t_end=1.0`` (waits for case A to settle, then ramps).
    """
    region: str
    component: Optional[int] = 1        # 0/1/2; None ⇒ vector BC (needs vec value)
    control: Control = Control.displacement
    # For displacement control we ramp from 0 → `magnitude` linearly.
    # For force/traction control we interpret `magnitude` directly.
    magnitude: float = 0.0              # signed; +ve = outward in the chosen axis
    t_start: float = 0.0                # ramp begins at this pseudo-time (in [0,1])
    t_end:   float = 1.0                # ramp ends; magnitude held after


class GeometrySpec(BaseModel):
    """What shape to mesh.

    ``kind="custom"`` is the generic escape hatch.  When used, the Architect
    MUST populate ``custom_regions`` with every region it references from
    BCs or loading cases — the custom-mesh LLM uses these to generate
    locator predicates.
    """
    kind: Literal[
        "notched_plate", "plate_3d", "slant_plate",
        "dogbone_2d", "dogbone_3d",
        "rectangle",        # plain 2D rectangle, no pre-crack (custom route)
        "custom",
    ] = "notched_plate"
    # Named dimensions — meaning depends on kind.
    dimensions: Dict[str, float] = Field(default_factory=dict)
    # Custom-gmsh description (natural language — the mesh agent renders it).
    custom_description: Optional[str] = None
    # Regions the custom mesh builder must expose (name + description).
    # Ignored for built-in mesh kinds.
    custom_regions: List[RegionSpec] = Field(default_factory=list)
    # Pre-crack segments: list of [[x_start, y_start], [x_end, y_end]] pairs
    # in mm.  When non-empty, the generated driver initialises the phase
    # field to z = 0 along these segments (damaged-notch BC), matching the
    # standard pre-cracked-specimen setup in the variational phase-field
    # fracture literature.  Without this, the geometric slit alone gives
    # nucleation-from-fresh behaviour, which rarely propagates below the
    # nucleation threshold.
    pre_crack_segments: List[List[List[float]]] = Field(default_factory=list)


class LoadingSpec(BaseModel):
    """Timing / mode of the loading schedule.  The *where* and *what* live
    in ``BCSpec.loading`` (one LoadingCase per applied action).
    """
    mode: Loading = Loading.quasistatic
    steps: int = 100                # default time-step count; solver
                                    # halves/reverts dt adaptively when
                                    # z_res exceeds 10*tol_stag.
    T_total: float = 1.0


class BCSpec(BaseModel):
    """Dirichlet BC schedule.

    ``fixed`` pins listed components on each region.  ``loading`` specifies
    where to apply displacement / force / traction.  ``free_regions`` is
    informational only (no constraint emitted).
    """
    fixed: List[FixedBC] = Field(default_factory=list)
    loading: List[LoadingCase] = Field(default_factory=list)
    free_regions: List[str] = Field(default_factory=list)

    # ---- legacy accessors for older code paths -------------------------- #
    @property
    def fixed_regions(self) -> List[str]:
        return [f.region for f in self.fixed]

    @property
    def loaded_region(self) -> str:
        return self.loading[0].region if self.loading else "top"


class MeshStrategy(BaseModel):
    """User's rule:
      * start with a uniform mesh at h0 = 2*eps (no box/tip refinement);
      * while n_cells < target_min_cells, apply
            new_eps = (n_cells / target_min_cells) ** (1 / dim) * old_eps
        and rebuild.  Iterates until the target is met or the safety cap
        (max_rescales) is reached — necessary when the material-derived
        eps is initially larger than the specimen itself (very tough
        materials in a small coupon).
    """
    target_min_cells: int = 5000
    max_rescales: int = 6


# ---------------------------------------------------------------------------
# Canonical spec — Architect output
# ---------------------------------------------------------------------------
class CanonicalSpec(BaseModel):
    """All information needed to synthesise and run a problem."""
    # Fracture toggle — if False we force Gc → ∞ so the phase field stays at 1.
    fracture_enabled: bool = True

    dimension: Dimension = Dimension.two
    plane_type: PlaneType = PlaneType.plane_strain       # 2D only

    constitutive: Constitutive = Constitutive.linear_elastic
    material: MaterialSpec
    geometry: GeometrySpec
    bcs: BCSpec = Field(default_factory=BCSpec)
    loading: LoadingSpec = Field(default_factory=LoadingSpec)
    mesh: MeshStrategy = Field(default_factory=MeshStrategy)

    # Expert overrides
    eps_override: Optional[float] = None
    h0_override: Optional[float] = None
    max_stag: int = 20
    tol_stag: float = 1.0e-7

    # Bookkeeping
    source_summary: str = ""                             # 1-2 sentence paraphrase
    # Assumption log — when the architect invents a dimension, region, BC,
    # magnitude, or material value because the user prompt was under-
    # specified, it appends one entry here.  The orchestrator surfaces the
    # log to the user (CLI prints, UI banner) so amateur prompts get an
    # auditable "the agent guessed X" trail.  Reviewers expect this for
    # under-specified evaluation tiers (Tier-2 / Tier-3 in the agent_plan
    # benchmark suite).
    assumptions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategist output — variant decision
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """Concrete template + module decision the Synthesizer will realise."""
    variant: Literal[
        "linear_elastic_2d_pe", "linear_elastic_2d_ps", "linear_elastic_3d",
        "dynamic_2d", "ductile_2d_pe", "ductile_3d",
        "finite_elastic_2d_pe", "finite_elastic_2d_ps", "finite_elastic_3d",
    ]
    mesh_builder: Literal[
        "make_notched_plate_2d", "make_notched_plate_3d",
        "make_slant_plate_2d", "make_dogbone_2d", "make_dogbone_3d",
        "make_custom_gmsh",
    ]
    solver: Literal["run_quasistatic", "run_dynamic",
                    "run_finite_elasticity", "run_ductile"]
    rationale: str = ""


# ---------------------------------------------------------------------------
# Result metrics — Advisor output
# ---------------------------------------------------------------------------
class ResultSummary(BaseModel):
    n_steps: int = 0
    final_time: float = 0.0
    final_disp: float = 0.0
    final_reaction: float = 0.0
    peak_reaction: float = 0.0
    peak_reaction_disp: float = 0.0
    min_z: float = 1.0
    cracked: bool = False
    crack_initiated_step: Optional[int] = None
    max_von_mises_MPa: Optional[float] = None
    diverged: bool = False
    message: str = ""
