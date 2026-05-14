"""Reviser agent — Reflect-Revise outer loop.

When the Physics-Advisor's ``HealthReport.verdict == "revise"``, route the
problem back through a structured action that changes only existing knobs.
The Reviser **never** edits modular physics; it only:

  * tweaks scalars in ``CanonicalSpec`` (steps, max_disp, tol_stag, max_stag)
  * flips spec.eps_override
  * requests a re-Strategist with a preference hint (e.g. plane stress →
    plane strain)
  * declares the run infeasible with a reason

This deterministic rule-based pass mirrors ATHENA's contextual-bandit
revision (arXiv 2512.03476) and MCP-SIM's Plan-Act-Reflect-Revise loop
(npj AI 2025) but with bounded actions — no LLM authoring of weak forms.

A2 is precedent: Foam-Agent v2 + Reviewer adds +3.6 pp on top of
hierarchical RAG; MCP-SIM converges in ≤ 5 P-A-R-R cycles vs. 12-16
without it.  The bounded-action design keeps every cycle auditable and
reversible.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from ..events import DECISION, emit
from ..health import HealthReport
from ..schema import Action, CanonicalSpec


class RevisionAction(BaseModel):
    """The Reviser's structured output."""
    action: str = "accept"            # accept | revise_spec | revise_action | infeasible
    spec_patches: dict = Field(default_factory=dict)   # field_path -> new_value
    strategist_hint: Optional[str] = None              # plain-English hint for next pick
    reason: str = ""
    flags_acted_on: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule table — ordered by priority.  Each rule looks for a flag pattern
# and emits a small, reversible spec patch.  At most one rule fires per
# cycle to keep diagnosis attributable.
# ---------------------------------------------------------------------------
def _propose_revision(spec: CanonicalSpec,
                       action: Action,
                       health: HealthReport,
                       prior_revisions: List["RevisionAction"]) -> RevisionAction:
    flags = set(health.flags)
    prior_count = len(prior_revisions)
    prior_flags = {r for rev in prior_revisions for r in rev.flags_acted_on}

    # 1. Hard infeasibility — if we've revised twice and the new attempt
    # still produces the same critical issue, declare infeasible.
    critical = {"empty_log", "nonzero_rc", "nan_in_log", "no_load_path",
                 "early_termination"}
    if prior_count >= 2 and (flags & critical) & prior_flags:
        return RevisionAction(
            action="infeasible",
            reason=(f"Persistent critical issues across {prior_count + 1} attempts: "
                    f"{sorted((flags & critical) & prior_flags)}.  "
                    "Likely a spec-level mismatch the agent can't recover from "
                    "automatically — surface to the user."),
            flags_acted_on=sorted(flags))

    # 2. Solver bailed early or returned non-zero — usually due to over-aggressive
    # loading on a brittle spec.  Halve max_disp on every displacement load.
    if {"early_termination", "nonzero_rc"} & flags:
        if "max_disp_halved" in prior_flags:
            # Already tried — if still failing, escalate.
            return RevisionAction(
                action="infeasible",
                reason="Already halved max_disp; further reduction unlikely "
                       "to help.  Surface for user inspection.",
                flags_acted_on=sorted(flags))
        patches = {}
        for i, lc in enumerate(spec.bcs.loading):
            if lc.control.value == "displacement" and lc.magnitude is not None:
                patches[f"bcs.loading[{i}].magnitude"] = lc.magnitude * 0.5
        if patches:
            return RevisionAction(
                action="revise_spec", spec_patches=patches,
                reason=("Solver terminated before reaching T_total — load was "
                        "likely too aggressive.  Halving displacement magnitudes."),
                flags_acted_on=["max_disp_halved"])

    # 3. Δt floor saturated — solver is fighting timestepping.  Increase
    # ``steps`` so dt_first shrinks proportionally, giving more headroom.
    if "dt_floor_saturated" in flags and "steps_doubled" not in prior_flags:
        new_steps = min(spec.loading.steps * 2, 800)
        return RevisionAction(
            action="revise_spec",
            spec_patches={"loading.steps": new_steps},
            reason=(f"Δt floor hit on {health.pct_steps_dt_floor * 100:.0f}% "
                    f"of steps — doubling step count "
                    f"({spec.loading.steps} → {new_steps}) so the "
                    f"adaptive Δt has room to manoeuvre."),
            flags_acted_on=["steps_doubled"])

    # 3b. F12 — phase-field never moved.  Two interventions in order:
    #   (a) tighten tol_stag ×0.01 so even tiny SNES motions register and
    #       the modular driver's adaptive-Δt halving has a chance to fire
    #   (b) double the step count so dt_first is smaller from the start
    # The solver may also be reporting hidden divergence; the modular
    # patches in this PR halve Δt on diverged-reason events automatically,
    # so by the time we get here the issue is more likely a too-coarse
    # initial Δt than a hard solver failure.
    if "no_z_evolution" in flags:
        if "tol_stag_tightened" not in prior_flags:
            new_tol = max(spec.tol_stag * 1e-2, 1e-12)
            return RevisionAction(
                action="revise_spec",
                spec_patches={"tol_stag": new_tol},
                reason=(f"Phase-field never evolved — tightening tol_stag "
                        f"({spec.tol_stag:.0e} → {new_tol:.0e}) so the "
                        f"adaptive-Δt trip threshold catches smaller z_res "
                        f"spikes around the bifurcation."),
                flags_acted_on=["tol_stag_tightened"])
        if "steps_doubled" not in prior_flags:
            new_steps = min(spec.loading.steps * 2, 800)
            return RevisionAction(
                action="revise_spec",
                spec_patches={"loading.steps": new_steps},
                reason=(f"Phase-field still not evolving after tighter tol — "
                        f"doubling step count ({spec.loading.steps} → "
                        f"{new_steps}) so the load increment per step is "
                        f"finer near the elastic-to-damaged bifurcation."),
                flags_acted_on=["steps_doubled"])
        return RevisionAction(
            action="infeasible",
            reason=("Phase-field still inert after tightening tol_stag and "
                    "doubling steps.  Either the load level is below the "
                    "material's strength surface (no damage expected — "
                    "this is fine), or the solver is genuinely stuck on a "
                    "metastable intact branch.  Surface for user inspection."),
            flags_acted_on=sorted(flags))

    # 4. Stagger saturated — relax tol_stag and / or raise max_stag.
    if "stagger_saturated" in flags and "stagger_relaxed" not in prior_flags:
        return RevisionAction(
            action="revise_spec",
            spec_patches={
                "max_stag": min(spec.max_stag * 2, 80),
                "tol_stag": min(spec.tol_stag * 10.0, 1e-4),
            },
            reason=(f"{health.pct_steps_stag_max * 100:.0f}% of steps hit "
                    f"max_stag={spec.max_stag} — relaxing tol_stag "
                    f"{spec.tol_stag:.0e}→{min(spec.tol_stag * 10, 1e-4):.0e} "
                    f"and doubling max_stag."),
            flags_acted_on=["stagger_relaxed"])

    # 5. Mesh too coarse for ε — shrink eps so h_min/ε ratio drops; the
    # mesh-rescale loop will then refine to hit target_min_cells.
    if ("mesh_too_coarse" in flags or "mesh_borderline" in flags) \
            and "eps_shrunk" not in prior_flags:
        cur_eps = float(spec.eps_override) if spec.eps_override else 0.0
        if cur_eps > 0:
            new_eps = cur_eps * 0.5
            return RevisionAction(
                action="revise_spec",
                spec_patches={"eps_override": new_eps},
                reason=(f"h_min/ε = {health.h_eps_ratio:.3f} — shrinking eps "
                        f"({cur_eps:.3e} → {new_eps:.3e}) so the rescale loop "
                        f"refines toward h_min/ε ≤ 0.25."),
                flags_acted_on=["eps_shrunk"])

    # 6. Residuals blowing up — usually means solver is unstable; bail.
    if "residual_blowup" in flags:
        # Same playbook as early_termination, but only if we haven't already
        # halved.
        if "max_disp_halved" not in prior_flags:
            patches = {}
            for i, lc in enumerate(spec.bcs.loading):
                if lc.control.value == "displacement" and lc.magnitude is not None:
                    patches[f"bcs.loading[{i}].magnitude"] = lc.magnitude * 0.5
            if patches:
                return RevisionAction(
                    action="revise_spec", spec_patches=patches,
                    reason="Residual blow-up — halving displacement magnitudes.",
                    flags_acted_on=["max_disp_halved"])

    # 7. Cracked late ("peak_at_end") — extend the loading window so a
    # post-peak softening branch becomes visible.
    if "peak_at_end" in flags and "max_disp_extended" not in prior_flags:
        patches = {}
        for i, lc in enumerate(spec.bcs.loading):
            if lc.control.value == "displacement" and lc.magnitude is not None:
                patches[f"bcs.loading[{i}].magnitude"] = lc.magnitude * 1.5
        if patches:
            return RevisionAction(
                action="revise_spec", spec_patches=patches,
                reason=("Reaction peak at the last step — extending displacement "
                        "by 50% so post-peak softening can be observed."),
                flags_acted_on=["max_disp_extended"])

    # 8. No matching rule — fall back to "warn" (accept with notes).
    return RevisionAction(
        action="accept",
        reason="No revision rule matched; accepting with the warnings above.",
        flags_acted_on=sorted(flags))


# ---------------------------------------------------------------------------
# Apply a RevisionAction to the spec.  Patches are simple dotted-path
# assignments — no nested LLM logic.
# ---------------------------------------------------------------------------
def _apply_patches(spec: CanonicalSpec, patches: dict) -> List[str]:
    """Mutate ``spec`` in place; return human-readable change log."""
    log: List[str] = []
    for path, new_val in patches.items():
        parts = path.split(".")
        # Resolve everything except the last segment.
        cur = spec
        for seg in parts[:-1]:
            if "[" in seg and seg.endswith("]"):
                name, idx = seg[:-1].split("[", 1)
                cur = getattr(cur, name)[int(idx)]
            else:
                cur = getattr(cur, seg)
        last = parts[-1]
        old_val = getattr(cur, last, None)
        try:
            setattr(cur, last, new_val)
            log.append(f"{path}: {old_val!r} → {new_val!r}")
        except Exception as e:
            log.append(f"{path}: failed to set ({type(e).__name__}: {e})")
    return log


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def reviser(spec: CanonicalSpec,
             action: Action,
             health: HealthReport,
             prior_revisions: Optional[List[RevisionAction]] = None
             ) -> Tuple[RevisionAction, List[str]]:
    """Decide a revision action and (if applicable) mutate ``spec``.

    Returns ``(RevisionAction, change_log)``.  ``change_log`` lists the
    field-level patches actually applied so the orchestrator can persist
    them in ``state.json``.
    """
    prior_revisions = prior_revisions or []
    rev = _propose_revision(spec, action, health, prior_revisions)

    if rev.action == "revise_spec" and rev.spec_patches:
        log = _apply_patches(spec, rev.spec_patches)
    else:
        log = []

    emit(DECISION,
         f"Reviser: {rev.action} ({rev.reason})"
         + (f"  patches: {log}" if log else ""))
    return rev, log
