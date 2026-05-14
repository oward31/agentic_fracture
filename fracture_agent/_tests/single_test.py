"""Run just one prompt and optionally execute it in WSL briefly."""
from __future__ import annotations
import sys
from pathlib import Path

from fracture_agent.agents.inspector import inspector
from fracture_agent.orchestrator import conceptualise, fill_material, plan_and_build
from fracture_agent.state import SessionState


def ask(questions):
    return ["accept defaults"] * len(questions)


def main(prompt: str, execute: bool = False, steps_override: int | None = None):
    state = SessionState.new()
    print(f"Session: {state.session_id}")
    spec = conceptualise(state, [("text", prompt)], ask)
    spec = fill_material(spec, ask)
    if steps_override:
        spec.loading.steps = steps_override
    state.spec = spec; state.save()
    state.rename_to_slug()
    action, script = plan_and_build(state, spec)
    ok, issues = inspector(script)
    print(f"\n[action] {action.variant}  |  mesh={action.mesh_builder}")
    print(f"[spec]  kind={spec.geometry.kind}  dims={spec.geometry.dimensions}")
    print(f"[bcs]   fixed={[(f.region, f.components) for f in spec.bcs.fixed]}")
    print(f"        loading={[(l.region, l.component, l.magnitude) for l in spec.bcs.loading]}")
    print(f"[mesh-plan] eps={state.mesh_plan['eps']:.3e}  h0={state.mesh_plan['h0']:.3e}")
    print(f"[inspector] {'PASS' if ok else issues}")
    print(f"[script]    {script}")
    if execute:
        from fracture_agent.executor import run_script
        print("\n[executing in WSL — 60 s timeout]")
        rec = run_script(script, nprocs=1, timeout_s=120, echo=True)
        print(f"\n[run] rc={rec.returncode}  wall={rec.wall_time_s:.1f}s  "
              f"diverged={rec.diverged}")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "50mm by 10mm rectangular copper bar, plane stress, uniaxial tension"
    execute = "--exec" in sys.argv
    main(prompt, execute=execute, steps_override=3)
