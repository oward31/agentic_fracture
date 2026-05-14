"""End-to-end pipeline: turn raw user input into a finished run + Q&A loop.

State-machine, no framework.  Each transition reads/mutates SessionState and
is fully resumable by reloading the JSON on disk.

    RECEIVE -> ARCHITECT -> (clarify?) -> STRATEGIST -> MATERIAL -> MESH -> SYNTH
        -> INSPECT -> RUN -> (debug-loop) -> ADVISE -> Q&A
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from .agents import (advisor, architect, debugger, inspector, receptionist,
                     resolve_material, strategist, synthesizer)
from .agents.advisor import advise_summary, answer_question, physics_advisor
from .agents.reviser import RevisionAction, reviser
from .config import MAX_DEBUG_ITERS, MAX_MESH_ITERS, WSL_MPI_DEFAULT
from .events import (CONSOLE, DECISION, ERROR, METADATA, RESULT, STATUS,
                     emit)
from .executor import run_script
from .knowledge import list_builtin_materials
from .llm import AllKeysExhausted
from .mesh import initial_plan, rescale_eps
from .schema import Action, CanonicalSpec, ResultSummary
from .state import ExecutionRecord, SessionState
from .telemetry import (add_solver_wall_time, bump, llm_agent,
                         set_active as set_active_telemetry)
from .templates import render_script


def attach_telemetry(state: SessionState) -> None:
    """Wire ``state.telemetry`` as the active sink so subsequent LLM calls
    are recorded against this session.  Idempotent — safe to call from
    ``main.py`` or the UI server before any phase runs."""
    set_active_telemetry(state.telemetry)


def detach_telemetry() -> None:
    """Clear the active telemetry sink (e.g. between independent sessions
    in the same process, like an ablation harness)."""
    set_active_telemetry(None)


Clarifier = Callable[[List[str]], List[str]]
"""Callback the orchestrator uses to ask the user clarifying questions.
Receives a list of questions, must return a list of answers (same length)."""


# ---------------------------------------------------------------------------
# Phase 1 — conceptualisation: raw inputs -> validated canonical spec.
# ---------------------------------------------------------------------------
def conceptualise(state: SessionState,
                  user_inputs: List[Tuple[str, Any]],
                  ask_user: Clarifier) -> CanonicalSpec:
    """Receptionist -> Architect -> (loop clarification) -> validated CanonicalSpec."""
    attach_telemetry(state)
    emit(STATUS, "Parsing problem description...")
    with llm_agent("receptionist"):
        frag = receptionist(user_inputs)
    state.raw_inputs.append({"inputs": [(k, str(v)) for k, v in user_inputs],
                             "fragments": frag})
    state.save()

    rounds = 0
    while True:
        rounds += 1
        bump("architect_rounds")
        emit(STATUS, f"Architect building canonical spec (round {rounds})...")
        with llm_agent("architect"):
            spec, questions = architect(state.raw_inputs, state.clarifications)
        if not questions or rounds >= 3:
            state.spec = spec
            state.save()
            emit(DECISION,
                 f"{spec.dimension.value} {spec.plane_type.value}  |  "
                 f"{spec.constitutive.value}  |  "
                 f"{spec.loading.mode.value}  |  "
                 f"fracture {'on' if spec.fracture_enabled else 'off'}")
            emit(DECISION,
                 f"Geometry: {spec.geometry.kind} {dict(spec.geometry.dimensions)}")
            regs = [f.region for f in spec.bcs.fixed] + [l.region for l in spec.bcs.loading]
            emit(DECISION, f"BC regions: fixed={[f.region for f in spec.bcs.fixed]}, "
                           f"loaded={[(l.region, l.component, l.magnitude) for l in spec.bcs.loading]}")
            # Surface the assumption log so the user knows which fields
            # were guessed (vs. user-specified).
            if spec.assumptions:
                emit(DECISION,
                     f"Architect made {len(spec.assumptions)} assumption(s) — "
                     f"see spec.assumptions for the full list.")
                print("\n--- The architect made these assumptions ---")
                for a in spec.assumptions:
                    print(f"   * {a}")
            return spec
        emit(STATUS, f"Clarifying {len(questions)} field(s)")
        print("\n--- The architect needs clarification --- ")
        answers = ask_user(questions)
        for q, a in zip(questions, answers):
            state.clarifications.append({"q": q, "a": a})
        state.save()


# ---------------------------------------------------------------------------
# Phase 2 — material resolution.
# ---------------------------------------------------------------------------
def fill_material(spec: CanonicalSpec, ask_user: Clarifier) -> CanonicalSpec:
    """Make sure `spec.material` has either a catalog_name or numeric fields.

    If the user gave only a vague name like "steel" we call the material
    helper to propose handbook values, then display them and give the user a
    chance to accept or override.
    """
    from .agents.material_helper import (_backfill_required_fields,
                                          _guess_problem_type)

    m = spec.material
    if m.catalog_name and m.catalog_name in list_builtin_materials():
        emit(DECISION, f"Material: catalog '{m.catalog_name}' (built-in)")
        return spec
    if m.E is not None or m.mu1 is not None:
        # User-supplied (or LLM-extracted) material values may still be
        # incomplete: missing sigma_ts → loader's Wts derivation crashes.
        # Run the backfill defensively here too — same pass that
        # ``resolve_material`` runs on handbook responses.
        ptype = m.problem_type or _guess_problem_type(spec)
        m.problem_type = m.problem_type or ptype
        _backfill_required_fields(m, ptype=ptype, spec=spec)
        emit(DECISION, f"Material: user-supplied ({m.display_name!r})")
        return spec

    emit(STATUS, f"Looking up handbook properties for {m.display_name!r}...")
    with llm_agent("material_helper"):
        proposed = resolve_material(spec, user_description=m.display_name or "")
    emit(DECISION,
         f"Material: {proposed.display_name!r} -> E={proposed.E}, "
         f"nu={proposed.nu}, Gc={proposed.Gc}")
    print("\n[material] Proposed handbook values for "
          f"{proposed.display_name!r}:")
    for k, v in proposed.model_dump().items():
        if v is not None and k not in ("catalog_name",):
            print(f"   {k:20s} = {v}")
    ans = ask_user(
        ["Accept these values? (yes) — or type overrides as key=value "
         "space-separated (E=210000 nu=0.3 ...)"]
    )[0].strip()
    if ans.lower() not in ("", "y", "yes", "ok"):
        for tok in ans.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                if hasattr(proposed, k):
                    try:
                        setattr(proposed, k, float(v))
                    except ValueError:
                        setattr(proposed, k, v)
    spec.material = proposed
    return spec


# ---------------------------------------------------------------------------
# Phase 3 — strategist + mesh sizing + synthesiser (with rescale loop).
# ---------------------------------------------------------------------------
def plan_and_build(state: SessionState,
                   spec: CanonicalSpec,
                   *,
                   probe_mesh: bool = True) -> Tuple[Action, Path]:
    """Strategist picks variant, template renders script; no execution yet.

    ``probe_mesh`` controls the WSL-side mesh-cell probe used by the
    eps-rescale loop.  Set to False from `--no-execute` callers that want
    a purely offline render with no WSL/conda dependency."""
    emit(STATUS, "Strategist picking modular variant...")
    action = strategist(spec)
    state.action = action
    emit(DECISION, f"Variant: {action.variant} (solver={action.solver}, "
                    f"mesh_builder={action.mesh_builder})")

    # Default initial eps from a material-loader dry run.  We import that
    # logic here so we don't reimplement it.
    from modular.materials.loader import (derive_length_scales,
                                            _augment_linear,
                                            _augment_finite_elasticity,
                                            load_material as _load_mat)  # type: ignore

    mat_dict = spec.material.model_dump()
    mat_dict = {k: v for k, v in mat_dict.items() if v is not None}
    # When the spec uses a catalog_name (the most common path) the explicit
    # numeric fields (E, Gc, sigma_ts, ...) are intentionally None — they
    # live in modular/materials/materials.json and are loaded at runtime.
    # But our eps derivation needs the actual numbers: load them here so we
    # don't fall back to the eps=0.25 default for every catalog material.
    if mat_dict.get("catalog_name"):
        try:
            cat_vals = _load_mat(mat_dict["catalog_name"])
            for k, v in cat_vals.items():
                if k not in mat_dict or mat_dict[k] is None:
                    mat_dict[k] = v
        except Exception as e:
            emit(ERROR,
                 f"Could not load catalog '{mat_dict['catalog_name']}': "
                 f"{type(e).__name__}: {e}")
    # The modular loader requires `problem_type` — if it's missing from the
    # spec.material, derive it from constitutive + mode.
    if "problem_type" not in mat_dict or not mat_dict["problem_type"]:
        if spec.constitutive.value == "j2_plasticity":
            mat_dict["problem_type"] = "ductile"
        elif spec.constitutive.value == "lopez_pamies":
            mat_dict["problem_type"] = "finite_elasticity"
        elif spec.loading.mode.value == "dynamic":
            mat_dict["problem_type"] = "dynamic_linear_elasticity"
        else:
            mat_dict["problem_type"] = "linear_elasticity"
    try:
        if spec.constitutive.value == "lopez_pamies":
            _augment_finite_elasticity(mat_dict)
        else:
            _augment_linear(mat_dict)
        mat_dict.update(derive_length_scales(mat_dict))
        eps0 = float(mat_dict["eps"])
    except Exception as e:
        # Fallback — flag it loudly rather than silently sitting on 0.25.
        emit(ERROR, f"Could not derive eps from material (fell back to 0.25): "
                    f"{type(e).__name__}: {e}")
        eps0 = 0.25

    if spec.eps_override is not None:
        eps0 = float(spec.eps_override)

    # No geometry-based eps cap — the only adjustment is the single
    # post-run rescale in ``rescale_and_rerun`` (below), which applies the
    # user-mandated rule   new_eps = (n_cells/5000)**dim * old_eps   when
    # the initial uniform mesh has fewer than 5000 cells.
    plan = initial_plan(eps0, dim=(2 if spec.dimension.value == "2D" else 3))
    state.mesh_plan = plan.__dict__.copy()
    emit(DECISION,
         f"Mesh plan: eps={plan.eps:.3e}, h0={plan.h0:.3e}, h_min={plan.h_min:.3e}")

    emit(STATUS, "Synthesising DOLFINx driver script...")
    with llm_agent("synthesizer"):
        script = synthesizer(spec, action, plan, state.dir)
    state.generated_scripts.append(str(script))
    _custom_mesh = script.parent / "custom_mesh.py"
    if _custom_mesh.exists():
        import time as _time
        state.generated_meshes.append({
            "ts":      _time.strftime("%Y-%m-%d %H:%M:%S"),
            "path":    str(_custom_mesh),
            "trigger": "synthesizer_initial",
        })
        emit(DECISION, "Custom gmsh module generated for this geometry")
    state.save()
    emit(DECISION, f"Driver written to {script.name}")

    # ---- Probe the mesh BEFORE the expensive solve so we rescale eps
    # up front (user's rule) rather than wasting a full run.
    # ``--no-execute`` callers (offline review, no WSL) skip this.
    if probe_mesh:
        script = _probe_and_maybe_rescale(state, spec, action, script, plan)
    else:
        emit(DECISION,
             "Mesh probe skipped (probe_mesh=False); eps not auto-rescaled. "
             "Run without --no-execute to apply the (n_cells/target)^(1/dim) rule.")
    return action, script


MAX_PROBE_DEBUG_ITERS = 3
"""Per-probe-attempt budget for inviting the multi-file Debugger to fix a
mesh-build error before retrying the probe.  Bounded separately from
``MAX_DEBUG_ITERS`` (the main-solve debug budget) because probe failures
are usually mesh-LLM bugs, and 3 attempts is enough to fix the common
gmsh API gotchas (binary fuse/cut, missing ry on addDisk, missing
synchronize, etc.) without burning into the main-solve budget."""


def _probe_and_maybe_rescale(state: SessionState, spec: CanonicalSpec,
                              action: Action, script: Path,
                              plan) -> Path:
    """Run the generated driver with --mesh-only so it exits right after
    printing n_cells.  If n_cells < target_min, apply the user's rule
    (eps_new = (n/target)**(1/dim) * eps_old) and regenerate the driver.

    If the probe itself fails (mesh-build crash → n_cells is None), we
    invoke the multi-file Debugger to patch the offending file (almost
    always ``custom_mesh.py``) and retry the probe up to
    ``MAX_PROBE_DEBUG_ITERS`` times *per rescale attempt*.  Without
    this, mesh-LLM bugs would silently bypass the rescale rule and the
    main solve would run on whatever coarse mesh the unrescaled eps
    produces.

    Returns the path of the final driver ready for the full solve."""
    import time as _time
    from .agents.debugger import patched_mesh_path_if_any

    dim = 2 if spec.dimension.value == "2D" else 3
    target_min = spec.mesh.target_min_cells
    dim_key = "2D" if dim == 2 else "3D"

    from .knowledge import CATALOG
    variant = next(v for v in CATALOG if v.variant == action.variant)

    for probe_attempt in range(spec.mesh.max_rescales + 1):
        # ---- Inner debug-retry loop: if the probe crashes (n_cells is
        # None), invite the Debugger to fix the file the traceback points
        # to, then retry the probe.  Bounded by MAX_PROBE_DEBUG_ITERS so
        # a chronically broken mesh module doesn't hang the orchestrator.
        n_cells: Optional[int] = None
        probe = None
        for debug_iter in range(MAX_PROBE_DEBUG_ITERS + 1):
            emit(STATUS, f"Probing mesh cell count (--mesh-only, attempt "
                         f"{probe_attempt + 1}"
                         + (f", debug-retry #{debug_iter}" if debug_iter else "")
                         + ")...")
            probe = run_script(script, nprocs=1,
                               echo=False,
                               on_line=lambda ln: emit(CONSOLE, ln.rstrip()),
                               extra_args=["--mesh-only"])
            n_cells = probe.n_cells
            if n_cells is not None:
                break
            # Probe crashed.  If we still have debug budget, ask the
            # Debugger to patch (multi-file aware — will fix the mesh
            # module when the traceback points there).  Same audit-trail
            # bookkeeping as run_with_debug.
            if debug_iter >= MAX_PROBE_DEBUG_ITERS:
                emit(DECISION,
                     f"Mesh probe: exhausted {MAX_PROBE_DEBUG_ITERS} "
                     f"debug-retries without recovering n_cells; "
                     f"proceeding without rescale.")
                return script
            emit(STATUS, "Mesh probe failed to report n_cells (mesh build "
                         "likely crashed) — invoking Debugger.")
            bump("debugger_attempts")
            _mesh_p = patched_mesh_path_if_any(script)
            _mesh_bytes_before = (_mesh_p.read_bytes()
                                   if _mesh_p and _mesh_p.exists() else None)
            with llm_agent("debugger"):
                script = debugger(script, probe.tail or "")
            state.generated_scripts.append(str(script))
            # Content-compare (not mtime) so the audit trail is reliable
            # even when fs mtime resolution is coarse.
            if (_mesh_p and _mesh_bytes_before is not None
                    and _mesh_p.exists()
                    and _mesh_p.read_bytes() != _mesh_bytes_before):
                state.generated_meshes.append({
                    "ts":      _time.strftime("%Y-%m-%d %H:%M:%S"),
                    "path":    str(_mesh_p),
                    "trigger": f"probe_debug_attempt={debug_iter + 1}",
                })
            state.save()

        if n_cells is None:
            # Defensive: should be unreachable (loop above either breaks
            # with a real n_cells or returns).  Bail without rescaling.
            emit(DECISION,
                 "Mesh probe did not report n_cells — proceeding without rescale.")
            return script
        if n_cells >= target_min:
            emit(DECISION,
                 f"Mesh probe: {n_cells} >= {target_min} cells ({dim_key}). "
                 f"Target met after {probe_attempt} rescale(s).")
            return script
        if probe_attempt >= spec.mesh.max_rescales:
            emit(DECISION,
                 f"Mesh probe: {n_cells} < {target_min} after "
                 f"{spec.mesh.max_rescales} rescales — budget exhausted, "
                 f"proceeding with current mesh. (Consider increasing "
                 f"mesh.max_rescales or shrinking the specimen / raising Gc.)")
            return script

        # Apply the rule:   eps_new = (n/target)**(1/dim) * eps_old
        new_eps = rescale_eps(old_eps=plan.eps, n_cells=n_cells,
                              target_min=target_min, dim=dim)
        if new_eps >= plan.eps * 0.999:
            emit(DECISION,
                 f"Mesh probe: rule produced a no-op (eps change <0.1%); "
                 f"proceeding with current mesh.")
            return script
        factor = new_eps / plan.eps
        emit(DECISION,
             f"Rescale #{probe_attempt + 1}: n_cells={n_cells} < {target_min}; "
             f"({n_cells}/{target_min})^(1/{dim}) = {factor:.4f}; "
             f"eps {plan.eps:.4e} -> {new_eps:.4e}.")
        plan.eps    = new_eps
        plan.h0     = 2.0 * new_eps
        plan.h_min  = plan.h0 / 8.0
        plan.n_cells_expected = n_cells
        plan.n_rescales = probe_attempt + 1
        plan.notes = f"rescale #{probe_attempt + 1} per user rule"
        state.mesh_plan = plan.__dict__.copy()
        spec.eps_override = new_eps
        bump("mesh_rescales")
        state.save()
        script = render_script(spec, action, plan, state.dir, variant)
        # Same path on rescale — dedupe so state.generated_scripts stays
        # informative (one entry per unique driver written).
        if not state.generated_scripts or state.generated_scripts[-1] != str(script):
            state.generated_scripts.append(str(script))
    return script


# ---------------------------------------------------------------------------
# Phase 4 — run with debug loop.
# ---------------------------------------------------------------------------
def run_with_debug(state: SessionState,
                   script: Path,
                   *,
                   nprocs: int = WSL_MPI_DEFAULT,
                   max_debug_iters: int = MAX_DEBUG_ITERS) -> ExecutionRecord:
    """Execute, and on failure hand the traceback to the Debugger for a patch.

    The mesh-size rescale (user's rule) is applied in a separate earlier pass
    — see ``rescale_and_rerun``.
    """
    current = script
    last_error_sig: Optional[str] = None
    last_error_tail: str = ""
    for attempt in range(max_debug_iters + 1):
        with llm_agent("inspector"):
            ok, issues = inspector(current)
        if not ok:
            emit(DECISION, f"Inspector flagged issues: {issues}")
        emit(STATUS, f"Launching {current.name} in WSL (attempt {attempt + 1})")
        result = run_script(current, nprocs=nprocs,
                            echo=True, on_line=lambda ln: emit(CONSOLE, ln.rstrip()))
        add_solver_wall_time(result.wall_time_s)
        rec = ExecutionRecord(
            script_path=str(current),
            returncode=result.returncode,
            wall_time_s=result.wall_time_s,
            stdout_tail=result.tail,
            diverged=result.diverged,
            error_signature=result.error_signature,
            n_cells=result.n_cells,
        )
        state.runs.append(rec)
        state.save()

        if result.returncode == 0 and not result.diverged:
            emit(DECISION, f"Run succeeded in {result.wall_time_s:.1f}s")
            # If a Debugger patch saved this run, archive (error -> fix)
            # into the Error-Fix RAG so future runs can retrieve it.
            if last_error_sig and attempt > 0:
                try:
                    from .rag.index import record_error_fix
                    record_error_fix(
                        error_signature=(last_error_sig
                                          or last_error_tail[-300:]),
                        offending_snippet=last_error_tail[-1500:],
                        applied_fix=current.read_text(encoding="utf-8")[:1500],
                        outcome="success")
                except Exception:
                    pass
            return rec
        if attempt == max_debug_iters:
            emit(ERROR, f"Exhausted {max_debug_iters} debug attempts; last rc={result.returncode}")
            return rec
        emit(STATUS, f"Uh oh - rc={result.returncode}, "
                     f"diverged={result.diverged}.  Asking debugger to patch.")
        bump("debugger_attempts")
        last_error_sig = result.error_signature or "rc=" + str(result.returncode)
        last_error_tail = result.tail
        # Snapshot custom_mesh.py CONTENT before the Debugger fires; if
        # it changes, log the touch into ``state.generated_meshes`` so
        # the audit trail shows which iteration of the mesh was in
        # effect at each retry.  Content-compare (not mtime) so the
        # signal is reliable even when fs mtime resolution is coarse
        # (Windows: 100ns claimed but often 1s in practice).  Mesh
        # edits land *in place* — no per-attempt suffix — so this is
        # the only audit hook we have.
        import time as _time
        from .agents.debugger import patched_mesh_path_if_any
        _mesh_p = patched_mesh_path_if_any(current)
        _mesh_bytes_before = (_mesh_p.read_bytes()
                               if _mesh_p and _mesh_p.exists() else None)
        with llm_agent("debugger"):
            current = debugger(current, result.tail)
        state.generated_scripts.append(str(current))
        if _mesh_p and _mesh_bytes_before is not None and _mesh_p.exists():
            if _mesh_p.read_bytes() != _mesh_bytes_before:
                state.generated_meshes.append({
                    "ts":      _time.strftime("%Y-%m-%d %H:%M:%S"),
                    "path":    str(_mesh_p),
                    "trigger": f"debugger_attempt={attempt + 1}",
                })
        state.save()
    return rec  # unreachable


def mesh_size_audit(state: SessionState,
                    action: Action) -> Optional[int]:
    """Return the n_cells_global captured by the executor during the last
    run.  Much more reliable than grepping the rolling stdout tail, which
    gets evicted during 100+-step runs."""
    if not state.runs:
        return None
    return state.runs[-1].n_cells


def rescale_and_rerun(state: SessionState,
                      spec: CanonicalSpec,
                      action: Action,
                      script: Path) -> Tuple[Path, ExecutionRecord]:
    """Run the full solve.  Any mesh-size rescaling has already been applied
    up front by ``_probe_and_maybe_rescale`` in :func:`plan_and_build`, so
    this just executes the finalised driver once (with the debug loop for
    error recovery)."""
    rec = run_with_debug(state, script)
    return script, rec


# ---------------------------------------------------------------------------
# Phase 4.5 — Reflect-Revise outer loop (A1).
#
# Wraps {plan_and_build, rescale_and_rerun, physics_advisor, reviser} so a
# low-health verdict triggers up to ``MAX_REFLECT_REVISE`` re-plans with a
# patched spec.  Mirrors MCP-SIM (npj AI 2025) Plan-Act-Reflect-Revise
# but with bounded actions and no LLM authoring of physics — only
# scalar tweaks the Reviser's deterministic rule table emits.
# ---------------------------------------------------------------------------
MAX_REFLECT_REVISE = 2


def run_with_reflect_revise(state: SessionState,
                             spec: CanonicalSpec,
                             action: Action,
                             script: Path,
                             *,
                             nprocs: int = WSL_MPI_DEFAULT,
                             max_outer: int = MAX_REFLECT_REVISE
                             ) -> Tuple[Path, "ExecutionRecord", Any]:
    """Run, advise, possibly revise & re-run; return final (script, rec, health).

    Health-driven decision loop:
      * verdict == "accept" or "warn"  → exit (warnings reported to user)
      * verdict == "revise"            → Reviser proposes a patch; if patch
                                          applies, regenerate the driver
                                          and re-run.  Bounded by max_outer.
      * verdict == "infeasible" (from Reviser) → exit; Advisor surfaces it.
    """
    from .agents.reviser import RevisionAction, reviser
    from .agents.advisor import physics_advisor
    from .templates import render_script
    from .knowledge import CATALOG

    prior_revisions: List[RevisionAction] = []

    for outer in range(max_outer + 1):
        if outer > 0:
            bump("reflect_revise_cycles")
            emit(STATUS, f"Reflect-Revise cycle #{outer}: regenerating script "
                          f"with revised spec")
            # Re-render with the patched spec.  Mesh-rescale loop already
            # ran once on the original spec; respect the user-set
            # ``eps_override`` and rebuild ``MeshSizePlan`` from it.
            from .mesh import MeshSizePlan
            mp_dict = state.mesh_plan or {}
            plan = MeshSizePlan(
                eps=float(mp_dict.get("eps") or 0.0),
                h0=float(mp_dict.get("h0") or 0.0),
                h_min=float(mp_dict.get("h_min") or 0.0),
                n_cells_expected=int(mp_dict.get("n_cells_expected") or 0),
                n_rescales=int(mp_dict.get("n_rescales") or 0),
                notes=str(mp_dict.get("notes") or ""),
            )
            variant = next(v for v in CATALOG if v.variant == action.variant)
            with llm_agent("synthesizer"):
                script = render_script(spec, action, plan, state.dir, variant)
            state.generated_scripts.append(str(script))
            state.save()

        # Run + debug-loop on the current driver.
        _, rec = rescale_and_rerun(state, spec, action, script)

        # Score the run.
        log_name = f"output_{spec.geometry.kind}_{action.variant}.txt"
        log_path = script.parent / log_name
        health = physics_advisor(
            spec, log_path,
            returncode=rec.returncode,
            wall_time_s=rec.wall_time_s,
            mesh_plan=state.mesh_plan or {},
        )
        emit(DECISION,
             f"[outer {outer}] Health={health.total:.1f}/100 "
             f"({health.verdict}); flags={sorted(set(health.flags))}")

        if health.verdict in ("accept", "warn"):
            state.health_report = health
            state.save()
            return script, rec, health

        # verdict == "revise": ask the Reviser for a patch.
        with llm_agent("reviser"):
            rev, change_log = reviser(spec, action, health, prior_revisions)
        prior_revisions.append(rev)
        state.revision_history.append({
            "outer": outer,
            "health_total": health.total,
            "flags": sorted(set(health.flags)),
            "action": rev.action,
            "reason": rev.reason,
            "patches": rev.spec_patches,
            "change_log": change_log,
        })
        state.save()

        if rev.action in ("infeasible", "accept"):
            # Reviser declined to revise — exit with the current health.
            state.health_report = health
            state.save()
            return script, rec, health
        if outer == max_outer:
            emit(ERROR, f"Reflect-Revise budget exhausted ({max_outer} cycles); "
                         "accepting the last attempt.")
            state.health_report = health
            state.save()
            return script, rec, health
        # else: loop with the patched spec and re-render.

    # Unreachable — every branch returns or escalates above.
    return script, rec, health


# ---------------------------------------------------------------------------
# Phase 5 — advise + interactive Q&A.
# ---------------------------------------------------------------------------
def advise(state: SessionState, spec: CanonicalSpec,
           last_script: Path, last_rec: ExecutionRecord) -> ResultSummary:
    emit(STATUS, "Advisor analysing log + results...")
    log_name = f"output_{spec.geometry.kind}_{state.action.variant}.txt"  # type: ignore
    log_path = last_script.parent / log_name
    summary = advisor(spec, log_path, diverged=last_rec.diverged)

    # Composite physics-admissibility reward (100-pt, ATHENA-style;
    # see fracture_agent/health.py for the breakdown).  Computed deterministically
    # from log + mesh_plan + execution metadata — no LLM, no XDMF.
    health = physics_advisor(
        spec, log_path,
        returncode=last_rec.returncode,
        wall_time_s=last_rec.wall_time_s,
        mesh_plan=state.mesh_plan or {},
    )
    state.final_result = summary
    state.health_report = health
    state.save()
    emit(DECISION,
         f"Health score: {health.total:.1f}/100 [{health.verdict}]  "
         f"(integ {health.integrity.points:.1f}/{health.integrity.max_points}, "
         f"admiss {health.admissibility.points:.1f}/{health.admissibility.max_points}, "
         f"accur {health.accuracy.points:.1f}/{health.accuracy.max_points}, "
         f"mesh {health.mesh_independence.points:.1f}/{health.mesh_independence.max_points}, "
         f"effic {health.efficiency.points:.1f}/{health.efficiency.max_points})")
    if health.flags:
        emit(DECISION, f"Health flags: {sorted(set(health.flags))}")

    try:
        with llm_agent("advisor"):
            verdict = advise_summary(spec, summary)
    except AllKeysExhausted:
        verdict = "(Advisor summary unavailable — LLM keys exhausted)"
    except Exception as e:
        # Any other LLM failure (MAX_TOKENS, transient 5xx, JSON error) —
        # fall back to the deterministic summary.message so the UI still
        # has SOMETHING to show and image rendering still runs.
        verdict = (summary.message or "Run completed; LLM verdict unavailable.")
        emit(ERROR, f"Advisor LLM summary failed ({type(e).__name__}); "
                    f"using computed summary instead.")
    emit(RESULT, verdict)
    emit(METADATA, json.dumps(summary.model_dump(), default=str))
    # Print verdict + health table.  ``_safe_print`` falls back to an
    # ASCII transliteration if the console can't encode the unicode glyphs
    # in HealthReport notes (Windows cp1252).  Belt-and-braces with the
    # ``main.py`` stdout reconfigure.
    def _safe_print(s: str) -> None:
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", "replace").decode("ascii"))

    _safe_print("\n=== Verdict ===")
    _safe_print(verdict)
    _safe_print(f"\n=== Health: {health.total:.1f}/100 [{health.verdict}] ===")
    for c in (health.integrity, health.admissibility, health.accuracy,
              health.mesh_independence, health.efficiency):
        _safe_print(f"  {c.name:18s} {c.points:5.1f} / {c.max_points:>3.0f}   "
                    + "; ".join(c.notes))
    if health.flags:
        _safe_print(f"  flags: {sorted(set(health.flags))}")
    _safe_print("\n=== Metrics ===")
    for k, v in summary.model_dump().items():
        _safe_print(f"  {k:25s} = {v}")
    return summary


def qa(state: SessionState, spec: CanonicalSpec, script: Path) -> None:
    """Interactive Q&A about the run.  Empty line exits."""
    log_name = f"output_{spec.geometry.kind}_{state.action.variant}.txt"  # type: ignore
    log_path = script.parent / log_name
    summary = state.final_result
    if summary is None:
        return
    print("\nType a question about the result (empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            return
        if not q:
            return
        try:
            with llm_agent("advisor"):
                ans = answer_question(spec, summary, log_path, q,
                                      history=state.conversation)
        except AllKeysExhausted as e:
            print(f"[advisor] {e}")
            return
        state.conversation.append({"q": q, "a": ans})
        state.save()
        print(ans + "\n")
