"""Dry-run harness: feed three prompts through conceptualise -> plan_and_build
and print the spec, action, and rendered script for review.  No WSL execution.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from fracture_agent.agents.inspector import inspector
from fracture_agent.orchestrator import conceptualise, fill_material, plan_and_build
from fracture_agent.state import SessionState


PROMPTS = [
    ("case1_copper_bar",
     "uniaxial tensile test of a rectangular bar 50 mm by 10 mm of copper plane stress"),
    ("case2_graphite_symmetric",
     "a 10mm wide by 20 mm tall graphite sheet plane stress with a crack from top "
     "center edge to center, with bottom fixed and top half of left and right edges "
     "being pulled part until they are displaced by 1mm each"),
    ("case3_L_shape_alumina",
     "an L shaped domain of alumina with outer lengths of 15mm, inner of 10 mm. "
     "bottom edge fixed and the top half of left edge being pulled outside by 2mm"),
]


def silent_asker(questions):
    # Auto-accept all proposed defaults so we see what the agent decides alone.
    return ["accept defaults"] * len(questions)


def run_one(label: str, prompt: str) -> dict:
    print(f"\n{'='*80}\n  {label}\n  prompt: {prompt}\n{'='*80}")
    state = SessionState.new()
    report = {"label": label, "session": state.session_id, "ok": False,
              "spec": None, "action": None, "issues": []}
    try:
        spec = conceptualise(state, [("text", prompt)], silent_asker)
        spec = fill_material(spec, silent_asker)
        state.spec = spec; state.save()
        action, script = plan_and_build(state, spec)
        ok, issues = inspector(script)
        report.update({
            "ok": ok, "issues": issues,
            "spec": spec.model_dump(),
            "action": action.model_dump(),
            "script": str(script),
        })
        print(f"\n[spec summary] kind={spec.geometry.kind} "
              f"dims={spec.geometry.dimensions}")
        print(f"[bcs] fixed={[(f.region, f.components) for f in spec.bcs.fixed]} "
              f"loading={[(l.region, l.component, l.control.value, l.magnitude) for l in spec.bcs.loading]}")
        print(f"[custom_regions] {[(r.name, r.description) for r in spec.geometry.custom_regions]}")
        print(f"[action] variant={action.variant} mesh={action.mesh_builder}")
        print(f"[material] catalog={spec.material.catalog_name} "
              f"display={spec.material.display_name} E={spec.material.E} "
              f"nu={spec.material.nu} Gc={spec.material.Gc}")
        print(f"[inspector] {'PASS' if ok else issues}")
        print(f"[script] {script}")
    except Exception as e:
        import traceback
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
    return report


if __name__ == "__main__":
    out = []
    for label, prompt in PROMPTS:
        out.append(run_one(label, prompt))
    Path(__file__).with_name("test_report.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\nWrote {len(out)} reports -> test_report.json")
