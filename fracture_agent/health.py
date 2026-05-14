"""Composite physics-admissibility reward for a finished simulation run.

Computed entirely from data we ALREADY persist:
    * ``output_*.txt``       — per-step log: step, time, dt, stag_iters,
                                u_res, z_res, min_z, disp, Fy
    * ``ExecutionRecord``    — returncode, wall_time_s, diverged, n_cells
    * ``MeshSizePlan``       — eps, h0, h_min

No XDMF readback, no field-level integration, no LLM authoring of physics.
Field-level checks (energy balance ∫G_c dΓ vs Π_ext - Ψ, pointwise
irreversibility) are parked: implementing them well needs a clean h5py
helper inside ``modular/post/`` and per-rank aggregation that's out of
scope for v1.

Reward decomposition (100 pts total) follows the 5-component split from
``papers/agent_plan.md`` §8 and ATHENA (arXiv 2512.03476) — adapted to
fracture and to log-only inputs:

    INTEGRITY        25 pts   exit-0, no NaN, reached T_total
    ADMISSIBILITY    30 pts   min_z monotone, residuals bounded, stagger health
    ACCURACY         25 pts   reaction-trajectory shape, Δt-floor not hit
    MESH-INDEP       10 pts   h_min ≤ ε/4 static audit
    EFFICIENCY       10 pts   wall-time per accepted step

Each component returns its own breakdown so the Reviser can target the
specific failure mode (saturated stagger → tighten tol; Δt floor hit →
reduce max_disp; etc.).
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tunable thresholds.  Conservative defaults; the values that matter for
# accept/revise routing are exposed in ``HealthReport.SCORE_*``.
# ---------------------------------------------------------------------------
ACCEPT_THRESHOLD = 85          # >= → archive as accepted
WARN_THRESHOLD   = 60          # >=60 and <85 → accept with warnings
                                # < WARN → revise via Reflect-Revise

# Per-component caps (must sum to 100).
W_INTEGRITY     = 25
W_ADMISSIBILITY = 30
W_ACCURACY      = 25
W_MESH_INDEP    = 10
W_EFFICIENCY    = 10
assert W_INTEGRITY + W_ADMISSIBILITY + W_ACCURACY + W_MESH_INDEP + W_EFFICIENCY == 100


class ComponentScore(BaseModel):
    """One component of the 100-pt reward."""
    name: str
    points: float            # awarded (≤ max_points)
    max_points: float
    notes: List[str] = Field(default_factory=list)


class HealthReport(BaseModel):
    """The 100-pt composite + per-component diagnostics.

    ``verdict`` is one of {"accept", "warn", "revise"} based on
    ``ACCEPT_THRESHOLD`` / ``WARN_THRESHOLD``.

    ``flags`` is a structured list of detected issues (e.g.
    ``"stagger_saturated"``, ``"dt_floor_hit"``) the Reviser can target
    without parsing free-text notes.
    """
    integrity:        ComponentScore
    admissibility:    ComponentScore
    accuracy:         ComponentScore
    mesh_independence: ComponentScore
    efficiency:       ComponentScore

    total: float = 0.0
    verdict: str = "revise"
    flags: List[str] = Field(default_factory=list)

    # Useful raw aggregates for the LLM verdict prompt.
    n_steps: int = 0
    pct_target_time: float = 0.0
    pct_steps_dt_floor: float = 0.0
    pct_steps_stag_max: float = 0.0
    min_z_violations: int = 0
    h_eps_ratio: float = float("nan")     # h_min / eps; want ≤ 0.25
    sec_per_step: float = 0.0


# ---------------------------------------------------------------------------
# Compute helpers — each returns (ComponentScore, list_of_flags).
# ---------------------------------------------------------------------------
def _has_nan_inf(cols: Dict[str, List[float]]) -> bool:
    for vs in cols.values():
        for v in vs:
            try:
                if math.isnan(v) or math.isinf(v):
                    return True
            except TypeError:
                continue
    return False


def _component_integrity(cols: Dict[str, List[float]],
                          *,
                          returncode: int,
                          T_total: float) -> Tuple[ComponentScore, List[str], float]:
    notes: List[str] = []
    flags: List[str] = []
    pts = 0.0

    if returncode == 0:
        pts += 10
        notes.append("returncode == 0")
    else:
        notes.append(f"returncode == {returncode}")
        flags.append("nonzero_rc")

    if cols and not _has_nan_inf(cols):
        pts += 5
        notes.append("no NaN/Inf in log")
    else:
        notes.append("NaN or Inf detected" if cols else "no log data")
        flags.append("nan_in_log" if cols else "empty_log")

    pct_target = 0.0
    if cols.get("time"):
        last_t = cols["time"][-1]
        pct_target = min(1.0, last_t / max(T_total, 1e-12))
        if pct_target >= 0.99:
            pts += 10
            notes.append(f"reached t={last_t:.3e}/{T_total:.3e}")
        elif pct_target >= 0.5:
            pts += 10 * pct_target
            notes.append(f"reached only {pct_target * 100:.1f}% of T_total")
            flags.append("partial_run")
        else:
            notes.append(f"reached only {pct_target * 100:.1f}% of T_total")
            flags.append("early_termination")
    else:
        notes.append("no time column in log")
        flags.append("empty_log")

    return (ComponentScore(name="integrity", points=pts, max_points=W_INTEGRITY,
                            notes=notes), flags, pct_target)


def _component_admissibility(cols: Dict[str, List[float]],
                              *, max_stag: int) -> Tuple[ComponentScore, List[str], Dict[str, float]]:
    notes: List[str] = []
    flags: List[str] = []
    pts = 0.0
    aux = {"min_z_violations": 0, "pct_steps_stag_max": 0.0}

    n = len(cols.get("step", []))
    if n == 0:
        return (ComponentScore(name="admissibility", points=0, max_points=W_ADMISSIBILITY,
                                notes=["no log data"]),
                ["empty_log"], aux)

    # 1) min_z monotone non-increasing (12 pts).  Damage shouldn't reverse.
    mz = cols["min_z"]
    violations = sum(1 for i in range(1, n) if mz[i] > mz[i - 1] + 1e-9)
    aux["min_z_violations"] = violations
    if violations == 0:
        pts += 12
        notes.append("min_z monotone non-increasing")
    else:
        # Lose proportional credit; floor at 0.
        pts += max(0.0, 12.0 * (1.0 - violations / max(1, n - 1)))
        notes.append(f"min_z increased {violations} times — possible "
                     "irreversibility issue")
        flags.append("irreversibility_violation")

    # 2) Residuals bounded (8 pts).  Use a sliding window: u_res, z_res
    # never exceed 100x their median over the previous 5 steps.
    def _resid_blowup(values: List[float], k: int = 5, mult: float = 100.0) -> int:
        bad = 0
        for i in range(k, len(values)):
            window = sorted(values[i - k:i])
            med = window[k // 2]
            if med > 0 and values[i] > med * mult:
                bad += 1
        return bad
    blow_u = _resid_blowup(cols["u_res"])
    blow_z = _resid_blowup(cols["z_res"])
    if blow_u == 0 and blow_z == 0:
        pts += 8
        notes.append("residuals bounded")
    else:
        pts += max(0.0, 8.0 * (1.0 - (blow_u + blow_z) / (2 * max(n, 1))))
        notes.append(f"residual spikes: u={blow_u}, z={blow_z}")
        flags.append("residual_blowup")

    # 3) Stagger health (10 pts).  Fraction of steps that hit max_stag.
    saturated = sum(1 for s in cols["stag_iters"] if int(s) >= max_stag)
    frac_sat = saturated / n
    aux["pct_steps_stag_max"] = frac_sat
    if frac_sat < 0.10:
        pts += 10
        notes.append(f"stagger healthy ({saturated}/{n} steps maxed)")
    elif frac_sat < 0.50:
        pts += 10 * (1.0 - (frac_sat - 0.10) / 0.40)
        notes.append(f"stagger saturating: {saturated}/{n} ({frac_sat * 100:.0f}%) "
                     f"hit max_stag={max_stag}")
        flags.append("stagger_warning")
    else:
        notes.append(f"stagger saturated: {saturated}/{n} ({frac_sat * 100:.0f}%) "
                     f"hit max_stag={max_stag}")
        flags.append("stagger_saturated")

    return (ComponentScore(name="admissibility", points=pts, max_points=W_ADMISSIBILITY,
                            notes=notes), flags, aux)


def _component_accuracy(cols: Dict[str, List[float]],
                         *, fracture_enabled: bool, dt_first: float
                         ) -> Tuple[ComponentScore, List[str], float]:
    notes: List[str] = []
    flags: List[str] = []
    pts = 0.0
    pct_dt_floor = 0.0

    n = len(cols.get("step", []))
    if n == 0:
        return (ComponentScore(name="accuracy", points=0, max_points=W_ACCURACY,
                                notes=["no log data"]), ["empty_log"], 0.0)

    # F12 — silent-solver detection.  When the staggered solver returns
    # z_res ≈ 0 for EVERY step AND no damage formed, something stopped the
    # phase-field from evolving (active-set blocking, line-search
    # collapse, or the formulation simply seeing no driving force).  The
    # modular code now halves Δt on PETSc-reported divergence; this flag
    # catches the residual case where SNES converged in 0 iterations with
    # no field motion — it looks "clean" but the solver did literally
    # nothing.  Intentional under-load runs (z_res tiny but min_z<1) are
    # NOT flagged.
    if fracture_enabled:
        z_res_all_tiny = all(zr < 1e-12 for zr in cols.get("z_res", []))
        min_z_overall  = min(cols.get("min_z", [1.0]) or [1.0])
        if z_res_all_tiny and min_z_overall >= 0.99 and n > 0:
            flags.append("no_z_evolution")
            notes.append(f"phase-field NEVER moved (z_res<1e-12 on all "
                         f"{n} steps, min_z={min_z_overall:.3f}) — solver "
                         "may have been silently inert")

    # 1) Reaction-trajectory shape (15 pts).  We expect a single dominant
    # peak (peak Fy at some interior step), with the trajectory rising
    # then falling.  Penalise the two pathologies:
    #   (a) peak at the LAST step → didn't crack yet; load still rising
    #   (b) peak at step 0 or trajectory all zero → never loaded
    Fy = [abs(f) for f in cols["Fy"]]
    peak_idx = max(range(n), key=lambda i: Fy[i])
    peak_F = Fy[peak_idx]
    if peak_F < 1e-12:
        notes.append("Fy never rose above zero — load not transmitted")
        flags.append("no_load_path")
    elif peak_idx == 0:
        notes.append("peak Fy at step 0 — degenerate trajectory")
        flags.append("degenerate_trajectory")
    elif peak_idx >= n - 1:
        # Peak at last step: only OK if fracture is disabled (still loading).
        if fracture_enabled:
            pts += 7   # half credit — loaded but not fully cracked
            notes.append(f"peak Fy at last step ({peak_idx}) — may not have cracked")
            flags.append("peak_at_end")
        else:
            pts += 15
            notes.append("loading curve monotonic (fracture disabled)")
    else:
        # Peak interior — bonus if drop after peak indicates softening.
        post_peak_drop = peak_F - Fy[-1]
        rel_drop = post_peak_drop / max(peak_F, 1e-12)
        if rel_drop > 0.10:
            pts += 15
            notes.append(f"single peak at step {peak_idx}, "
                         f"{rel_drop * 100:.0f}% softening drop")
        else:
            pts += 10
            notes.append(f"peak at step {peak_idx} but small post-peak drop")
            flags.append("weak_softening")

    # 2) Δt floor not hit (10 pts).  Solver halves dt and reverts when
    # z_res > 10*tol; floor = dt_first/10.
    if dt_first > 0:
        floor = dt_first / 10.0
        floor_hits = sum(1 for d in cols["dt"] if d <= floor * 1.001)
        pct_dt_floor = floor_hits / n
        if pct_dt_floor < 0.05:
            pts += 10
            notes.append("Δt floor essentially not hit")
        elif pct_dt_floor < 0.30:
            pts += 10 * (1.0 - (pct_dt_floor - 0.05) / 0.25)
            notes.append(f"Δt floor hit on {floor_hits}/{n} steps "
                         f"({pct_dt_floor * 100:.0f}%)")
            flags.append("dt_floor_warning")
        else:
            notes.append(f"Δt floor hit on {floor_hits}/{n} steps "
                         f"({pct_dt_floor * 100:.0f}%) — solver struggled")
            flags.append("dt_floor_saturated")
    else:
        # Couldn't determine floor — give partial credit.
        pts += 5
        notes.append("dt_first unknown; cannot audit floor hits")

    return (ComponentScore(name="accuracy", points=pts, max_points=W_ACCURACY,
                            notes=notes), flags, pct_dt_floor)


def _component_mesh_independence(eps: float, h_min: float
                                  ) -> Tuple[ComponentScore, List[str], float]:
    notes: List[str] = []
    flags: List[str] = []
    pts = 0.0
    ratio = float("nan")
    if eps and eps > 0:
        ratio = h_min / eps
        if ratio <= 0.25 + 1e-9:
            pts += W_MESH_INDEP
            notes.append(f"h_min/ε = {ratio:.3f} ≤ 0.25 ✓")
        elif ratio <= 0.50:
            pts += W_MESH_INDEP * (1.0 - (ratio - 0.25) / 0.25)
            notes.append(f"h_min/ε = {ratio:.3f} > 0.25 — mesh borderline")
            flags.append("mesh_borderline")
        else:
            notes.append(f"h_min/ε = {ratio:.3f} > 0.50 — mesh too coarse")
            flags.append("mesh_too_coarse")
    else:
        notes.append("eps unknown; cannot audit mesh ratio")
    return (ComponentScore(name="mesh_independence", points=pts,
                            max_points=W_MESH_INDEP, notes=notes), flags, ratio)


def _component_efficiency(cols: Dict[str, List[float]],
                           *, wall_time_s: float
                           ) -> Tuple[ComponentScore, List[str], float]:
    notes: List[str] = []
    flags: List[str] = []
    pts = 0.0
    sec_per_step = 0.0
    n = len(cols.get("step", []))
    if n > 0 and wall_time_s > 0:
        sec_per_step = wall_time_s / n
        # Full points if ≤ 5 s/step (typical 2D linear elastic).
        # Half points at 30 s/step.  Zero past 120 s/step.
        if sec_per_step <= 5.0:
            pts += W_EFFICIENCY
        elif sec_per_step <= 30.0:
            pts += W_EFFICIENCY * (1.0 - (sec_per_step - 5.0) / 50.0)
        elif sec_per_step <= 120.0:
            pts += max(0.0, W_EFFICIENCY * 0.5 * (1.0 - (sec_per_step - 30.0) / 90.0))
        notes.append(f"{sec_per_step:.2f} s/step ({n} steps in {wall_time_s:.1f}s)")
        if sec_per_step > 30:
            flags.append("slow_run")
    else:
        notes.append("insufficient data to compute efficiency")
    return (ComponentScore(name="efficiency", points=pts, max_points=W_EFFICIENCY,
                            notes=notes), flags, sec_per_step)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def compute_health(cols: Dict[str, List[float]],
                    *,
                    returncode: int,
                    wall_time_s: float,
                    fracture_enabled: bool,
                    T_total: float,
                    dt_first: float,
                    eps: float,
                    h_min: float,
                    max_stag: int) -> HealthReport:
    """Build the 100-pt composite reward from log columns + run metadata.

    Pure function — no I/O, no LLM, no XDMF.  Order of arguments matches
    the order they are produced in ``orchestrator.advise``.
    """
    integ, f1, pct_target = _component_integrity(
        cols, returncode=returncode, T_total=T_total)
    admiss, f2, aux = _component_admissibility(cols, max_stag=max_stag)
    accur, f3, pct_dt_floor = _component_accuracy(
        cols, fracture_enabled=fracture_enabled, dt_first=dt_first)
    mesh,  f4, ratio = _component_mesh_independence(eps=eps, h_min=h_min)
    effic, f5, sec_per_step = _component_efficiency(
        cols, wall_time_s=wall_time_s)

    total = integ.points + admiss.points + accur.points + mesh.points + effic.points
    all_flags = list(f1 + f2 + f3 + f4 + f5)
    # Hard-trigger flags ALWAYS produce verdict="revise" regardless of the
    # composite score, because they signal something the Reviser can act on
    # cheaply (tighter tolerance, more steps).  ``no_z_evolution`` is the
    # canonical example: a fracture run can score 92/100 yet have z stuck
    # at 1.0 — the run is "clean" but the agent should still try the
    # cheaper interventions before declaring done.  The user explicitly
    # said no-crack with a correct setup is fine; the Reviser handles that
    # by escalating to ``infeasible`` after both interventions fail.
    HARD_REVISE_FLAGS = {"no_z_evolution", "empty_log", "no_load_path"}
    if HARD_REVISE_FLAGS & set(all_flags):
        verdict = "revise"
    elif total >= ACCEPT_THRESHOLD:
        verdict = "accept"
    elif total >= WARN_THRESHOLD:
        verdict = "warn"
    else:
        verdict = "revise"

    return HealthReport(
        integrity=integ, admissibility=admiss, accuracy=accur,
        mesh_independence=mesh, efficiency=effic,
        total=round(total, 2), verdict=verdict,
        flags=all_flags,
        n_steps=len(cols.get("step", [])),
        pct_target_time=round(pct_target, 4),
        pct_steps_dt_floor=round(pct_dt_floor, 4),
        pct_steps_stag_max=round(aux.get("pct_steps_stag_max", 0.0), 4),
        min_z_violations=int(aux.get("min_z_violations", 0)),
        h_eps_ratio=round(ratio, 4) if ratio == ratio else float("nan"),
        sec_per_step=round(sec_per_step, 3),
    )
