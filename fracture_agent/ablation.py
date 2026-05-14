"""Ablation harness — B1 / B2 / B3 / Full.

The four canonical configurations every reviewer expects (MCP-SIM Fig. 2B,
ALL-FEM 2-agent vs multi-agent cross, Foam-Agent v2 ablation):

    Level 1  (B1)  — one-shot Gemini Pro with the modular CATALOG glued
                     into the prompt; no Architect, no Strategist, no
                     Synthesizer, no Inspector, no Debugger.  This is the
                     "what does a frontier LLM do without help?" baseline.

    Level 2  (B2)  — B1 + the Inspector AST check (legacy-import scan,
                     module presence, solver-call presence).  Catches the
                     ``from dolfin import *`` failure mode that bites
                     every existing FEniCS-LLM agent.  No Debugger.

    Level 3  (B3)  — full Architect → Strategist → Synthesizer → Inspector
                     → Executor → Debugger inner loop.  No Reflect-Revise
                     outer loop.  This is what fracture_agent was BEFORE A1.

    Level 4  (B4)  — full fracture_agent: B3 + Reflect-Revise outer loop with
                     Physics-Advisor composite reward (A1).

The harness reuses the existing orchestrator phases at higher levels and
provides a new ``b1_one_shot`` for Level 1.  All four paths share the
same telemetry hooks (A4) so the cost-table aggregator can compare
tokens / cost / wall-clock per ablation.

Hard constraints:
    * No level authors physics — even B1 is asked to compose ``modular.*``
      symbols; if it deviates the Inspector (B2+) or runtime catches it.
    * Each level returns the same shape ``(script_path, ExecutionRecord,
      HealthReport | None)`` so the benchmark runner can tabulate.
"""
from __future__ import annotations
from enum import IntEnum
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import FAST_MODEL, PRIMARY_MODEL, WSL_MPI_DEFAULT
from .events import DECISION, ERROR, STATUS, emit
from .health import HealthReport
from .knowledge import CATALOG, list_builtin_materials
from .llm import llm
from .schema import CanonicalSpec
from .state import ExecutionRecord, SessionState
from .telemetry import bump, llm_agent


class AblationLevel(IntEnum):
    """The four canonical ablation configurations."""
    B1_ONE_SHOT      = 1     # raw Gemini Pro w/ catalog glued in
    B2_INSPECTOR     = 2     # + AST inspection
    B3_FULL_INNER    = 3     # + Architect/Strategist/Synthesizer/Debugger
    B4_FULL_PFAGENT  = 4     # + Reflect-Revise outer loop (A1)


# ---------------------------------------------------------------------------
# B1 — one-shot prompt.  Frontier LLM with the modular API in context.
# ---------------------------------------------------------------------------
B1_SYSTEM = """You are a single-shot DOLFINx phase-field fracture script
generator.  You receive a user prompt describing a fracture problem and
emit ONE complete, runnable Python script that drives the user's verified
``modular`` toolbox.

You MUST:
  1. Compose existing ``modular.*`` symbols.  Do NOT author new UFL forms,
     constitutive laws, or solver schemes.
  2. Pick exactly one variant from the modular CATALOG (listed below).
  3. Pick exactly one mesh builder.
  4. Use the standard load_material / build_problem / run_quasistatic
     (or run_dynamic / run_ductile / run_finite_elasticity) idiom.
  5. Emit one ``[mesh-audit] n_cells_global = N`` print after mesh build.
  6. Honor any user-specified material values, geometry dimensions, BCs.

You MUST NOT:
  * use ``from dolfin import …`` (legacy FEniCS — DOLFINx only)
  * define your own UFL forms via TestFunction/TrialFunction
  * implement constitutive logic in the script

The script will be written verbatim to disk and executed in WSL with
the conda env ``fenicsx`` (DOLFINx 0.9 + petsc4py + gmsh).

Return ONLY Python source, no commentary, no code fences.
"""


def _build_b1_user_prompt(prompt: str) -> str:
    """Assemble the catalog summary + materials list + user prompt."""
    cat_lines = [
        "MODULAR CATALOG (pick exactly one variant):"
    ]
    for v in CATALOG:
        cat_lines.append(
            f"  - {v.variant}: dim={v.dim}, plane={v.plane}, "
            f"constitutive={v.constitutive}, loading={v.loading}, "
            f"solver={v.solver}, mesh_builder={v.mesh_builder}, "
            f"example={v.example_script}, default_material={v.default_material}")
        cat_lines.append(f"      {v.summary}")
    materials = sorted(list_builtin_materials().keys())
    mat_block = "BUILTIN MATERIALS: " + ", ".join(materials)

    api_block = """\
MODULAR API (only these symbols are allowed besides stdlib + numpy + mpi4py + dolfinx + petsc4py):
    from modular.materials import load_material
    from modular.common    import print_banner, print_mesh_info
    from modular.problems  import (make_linear_elastic_2d_pe_builder,
                                    make_linear_elastic_2d_ps_builder,
                                    make_linear_elastic_3d_builder,
                                    make_dynamic_2d_builder,
                                    make_ductile_2d_pe_builder,
                                    make_ductile_3d_builder,
                                    make_finite_elastic_2d_pe_builder,
                                    make_finite_elastic_2d_ps_builder,
                                    make_finite_elastic_3d_builder)
    from modular.meshes    import (make_notched_plate_2d, make_notched_plate_3d,
                                    make_slant_plate_2d,
                                    make_dogbone_2d, make_dogbone_3d)
    from modular.post      import (XDMFWriter,
                                    reaction_form_from_sigma_2d,
                                    reaction_form_from_sigma_3d)
    from modular.solvers   import (run_quasistatic, run_dynamic,
                                    run_ductile, run_finite_elasticity)

The script must walk up 3 directories to import ``modular``:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    sys.path.insert(0, _REPO)

For custom geometries write a sibling ``custom_mesh.py`` exposing
``make_custom_gmsh(h0, comm, rank)`` returning ``(msh, markers_spec, geom)``
in modular format and import it via ``from custom_mesh import ...``.
"""
    return "\n\n".join([
        api_block, "\n".join(cat_lines), mat_block,
        f"USER PROMPT:\n{prompt}",
        "Emit the complete Python module now."])


def b1_one_shot(state: SessionState,
                 prompt: str,
                 *,
                 model: str = PRIMARY_MODEL) -> Path:
    """Level-1 ablation: ask the LLM to write the whole script in one go.

    Writes the script to ``state.dir / "run_b1.py"`` and returns its path.
    No Architect, no Strategist, no Synthesizer template.
    """
    emit(STATUS, "[B1] One-shot LLM script generation...")
    user_prompt = _build_b1_user_prompt(prompt)
    with llm_agent("b1_one_shot"):
        text = llm().complete(B1_SYSTEM, user_prompt,
                               temperature=0.1,
                               max_output_tokens=32768,
                               thinking_budget=-1,
                               model=model)
    s = text.strip()
    # Strip accidental code fences.
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rsplit("```", 1)[0]
        s = s.strip()
    out = state.dir / "run_b1.py"
    out.write_text(s, encoding="utf-8")
    state.generated_scripts.append(str(out))
    state.save()
    emit(DECISION, f"[B1] script written: {out.name} ({len(s)} chars)")
    return out


# ---------------------------------------------------------------------------
# Per-level entry point.  Each level returns (script_path, exec_record,
# health_report) so the benchmark runner can tabulate uniformly.
# ---------------------------------------------------------------------------
def run_at_level(state: SessionState,
                  user_inputs: List[Tuple[str, Any]],
                  *,
                  level: AblationLevel,
                  ask_user=lambda qs: [""] * len(qs),
                  nprocs: int = WSL_MPI_DEFAULT,
                  execute: bool = True
                  ) -> Tuple[Path, Optional[ExecutionRecord], Optional[HealthReport]]:
    """Run the pipeline at the given ablation level.

    Levels 1 and 2 (one-shot) bypass the architect/strategist/synthesizer.
    Levels 3 and 4 use the full conceptualise → plan_and_build path.
    Level 4 wraps the run in the Reflect-Revise outer loop (A1).
    """
    from .agents.advisor import physics_advisor
    from .agents.inspector import inspector
    from .executor import run_script
    from .orchestrator import (attach_telemetry, conceptualise, fill_material,
                                plan_and_build, run_with_debug,
                                run_with_reflect_revise)

    attach_telemetry(state)

    # ----- Level 1 / 2 — one-shot generation ------------------------- #
    if level in (AblationLevel.B1_ONE_SHOT, AblationLevel.B2_INSPECTOR):
        # Combine all text inputs into a single prompt; B1 has no clarifier.
        prompt = "\n".join(str(v) for k, v in user_inputs if k == "text")
        script = b1_one_shot(state, prompt)

        # Level 2: inspector pre-flight.  Stop on failure (no debugger).
        if level == AblationLevel.B2_INSPECTOR:
            with llm_agent("inspector"):
                ok, issues = inspector(script)
            if not ok:
                emit(ERROR, f"[B2] Inspector rejected B1 output: {issues}")
                if execute:
                    # Record a synthetic "didn't run" execution record so the
                    # benchmark runner sees the failure.
                    rec = ExecutionRecord(
                        script_path=str(script), returncode=-1,
                        wall_time_s=0.0, stdout_tail=str(issues),
                        diverged=False, error_signature="inspector_rejected")
                    state.runs.append(rec)
                    state.save()
                    return script, rec, None
                return script, None, None

        if not execute:
            return script, None, None

        # Run the one-shot script.
        emit(STATUS, f"[L{level}] launching one-shot script in WSL...")
        result = run_script(script, nprocs=nprocs, echo=True)
        rec = ExecutionRecord(
            script_path=str(script),
            returncode=result.returncode,
            wall_time_s=result.wall_time_s,
            stdout_tail=result.tail,
            diverged=result.diverged,
            error_signature=result.error_signature,
            n_cells=result.n_cells,
        )
        state.runs.append(rec)
        state.save()
        return script, rec, None

    # ----- Level 3 / 4 — full pipeline ------------------------------- #
    spec = conceptualise(state, user_inputs, ask_user)
    spec = fill_material(spec, ask_user)
    state.spec = spec
    state.save()
    state.rename_to_slug()
    action, script = plan_and_build(state, spec)

    if not execute:
        return script, None, None

    if level == AblationLevel.B3_FULL_INNER:
        rec = run_with_debug(state, script, nprocs=nprocs)
        # Compute health for reporting parity with B4 — but DON'T loop.
        log_name = f"output_{spec.geometry.kind}_{action.variant}.txt"
        log_path = script.parent / log_name
        health = physics_advisor(
            spec, log_path,
            returncode=rec.returncode,
            wall_time_s=rec.wall_time_s,
            mesh_plan=state.mesh_plan or {})
        state.health_report = health
        state.save()
        return script, rec, health

    # B4 — full Reflect-Revise.
    script, rec, health = run_with_reflect_revise(
        state, spec, action, script, nprocs=nprocs)
    return script, rec, health
