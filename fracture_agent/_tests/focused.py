"""Focused re-test of the two prompts that previously failed with
MAX_TOKENS, plus one representative 'easy' case as a regression check."""
from __future__ import annotations
import json
from pathlib import Path

from fracture_agent.agents.inspector import inspector
from fracture_agent.orchestrator import conceptualise, fill_material, plan_and_build
from fracture_agent.state import SessionState


PROMPTS = [
    ("cruciform_biax",
     "cruciform specimen under equal biaxial tension, 100 mm arm length, "
     "pulled by 1 mm on each arm"),
    ("al_dogbone_3d",
     "3D aluminium alloy dogbone tensile coupon, total length 40 mm, pulled "
     "until yielding and necking"),
    ("simple_regression",
     "steel plate 20 mm x 20 mm with left edge crack, pulled vertically"),
]


def ask(qs):
    # Empty strings tell the Architect "user did not answer — use defaults".
    return [""] * len(qs)


def run(label, prompt):
    state = SessionState.new()
    print(f"\n{'='*70}\n  {label}\n  {prompt}\n{'='*70}")
    try:
        spec = conceptualise(state, [("text", prompt)], ask)
        spec = fill_material(spec, ask)
        state.spec = spec; state.save()
        state.rename_to_slug()
        action, script = plan_and_build(state, spec)
        ok, issues = inspector(script)
        print(f"[kind/const/plane/mode]  {spec.geometry.kind} / {spec.constitutive.value} / "
              f"{spec.plane_type.value} / {spec.loading.mode.value}  fracture={spec.fracture_enabled}")
        print(f"[dims]  {spec.geometry.dimensions}")
        print(f"[mat]   catalog={spec.material.catalog_name}  display={spec.material.display_name}  E={spec.material.E}")
        print(f"[fixed] {[(f.region, f.components) for f in spec.bcs.fixed]}")
        print(f"[load]  {[(l.region, l.component, l.magnitude) for l in spec.bcs.loading]}")
        print(f"[action] variant={action.variant}  mesh={action.mesh_builder}")
        print(f"[slug]  {state.session_id}")
        print(f"[insp]  {'PASS' if ok else issues}")
        return {"label": label, "ok": ok}
    except Exception as e:
        print(f"!! FAILED: {type(e).__name__}: {e}")
        return {"label": label, "ok": False, "error": str(e)[:200]}


if __name__ == "__main__":
    rows = [run(l, p) for l, p in PROMPTS]
    ok = sum(1 for r in rows if r["ok"])
    print(f"\n\n  {ok}/{len(rows)} passed")
    for r in rows:
        print(f"   {'OK' if r['ok'] else 'ERR':3s} {r['label']}")
