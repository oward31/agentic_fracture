"""Diverse prompt battery — test the agent's breadth, not just the three
originals.  Each prompt probes a different axis (vagueness, geometry,
material, physics, BC style).
"""
from __future__ import annotations
import json
from pathlib import Path

from fracture_agent.agents.inspector import inspector
from fracture_agent.orchestrator import conceptualise, fill_material, plan_and_build
from fracture_agent.state import SessionState


PROMPTS = [
    # --- Ultra-vague (the agent must infer nearly everything) ---
    ("vague_paper_tear",
     "a thin paper being torn"),
    ("vague_broken_window",
     "simulate a broken window"),

    # --- Classic benchmarks (specific, well-known problems) ---
    ("concrete_3pb",
     "300mm long concrete beam under three-point bending, simply "
     "supported at the ends and loaded at the midspan"),
    ("brazilian_disc",
     "Brazilian disc test on sandstone, 50 mm diameter, compressed between "
     "two platens until splitting"),

    # --- Exotic materials / custom geometries ---
    ("rubber_oring",
     "a rubber O-ring of 20 mm outer diameter and 15 mm inner diameter "
     "compressed by 10% between two rigid plates"),
    ("cruciform_biax",
     "cruciform specimen under equal biaxial tension, 100 mm arm length, "
     "pulled by 1 mm on each arm"),

    # --- Dynamic / 3D / ductile ---
    ("al_dogbone_3d",
     "3D aluminium alloy dogbone tensile coupon, total length 40 mm, pulled "
     "until yielding and necking"),
    ("glass_impact",
     "a 200 mm square glass window hit in the centre by a fast impactor"),

    # --- Under-specified material with plane-stress hint ---
    ("thin_steel_plate_crack",
     "thin steel plate with a crack"),
]


def ask(qs):
    return [""] * len(qs)   # simulate user not replying -> defaults


def run(label, prompt):
    state = SessionState.new()
    report = {"label": label, "session": state.session_id, "ok": False,
              "error": None, "spec": None, "action": None, "script": None}
    print(f"\n{'='*76}\n  {label}\n  prompt: {prompt}\n{'='*76}")
    try:
        spec = conceptualise(state, [("text", prompt)], ask)
        spec = fill_material(spec, ask)
        state.spec = spec; state.save()
        state.rename_to_slug()
        action, script = plan_and_build(state, spec)
        ok, issues = inspector(script)
        report.update({
            "ok": ok, "issues": issues,
            "spec": spec.model_dump(),
            "action": action.model_dump(),
            "script": str(script),
        })
        print(f"[kind]    {spec.geometry.kind}")
        print(f"[dims]    {spec.geometry.dimensions}")
        print(f"[bcs]     fixed={[(f.region, f.components) for f in spec.bcs.fixed]}")
        print(f"          load ={[(l.region, l.component, l.control.value, l.magnitude) for l in spec.bcs.loading]}")
        print(f"[mat]     catalog={spec.material.catalog_name} "
              f"display={spec.material.display_name!r} "
              f"E={spec.material.E} nu={spec.material.nu} Gc={spec.material.Gc}")
        print(f"[const]   {spec.constitutive.value} / {spec.plane_type.value} / "
              f"{spec.loading.mode.value}  fracture={spec.fracture_enabled}")
        print(f"[action]  variant={action.variant} mesh={action.mesh_builder}")
        print(f"[slug]    {state.session_id}")
        print(f"[inspec]  {'PASS' if ok else issues}")
    except Exception as e:
        import traceback
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"!! FAILED: {report['error']}")
        print(traceback.format_exc()[-400:])
    return report


if __name__ == "__main__":
    rows = [run(l, p) for l, p in PROMPTS]
    out = Path(__file__).with_name("battery_report.json")
    out.write_text(json.dumps(rows, indent=2, default=str))
    ok_count = sum(1 for r in rows if r["ok"])
    print(f"\n\n================ SUMMARY ================")
    print(f"  {ok_count}/{len(rows)} prompts produced a valid script")
    for r in rows:
        status = "OK" if r["ok"] else ("ERR" if r["error"] else "FAIL")
        print(f"  {status:<4} {r['label']}")
    print(f"  report -> {out}")
