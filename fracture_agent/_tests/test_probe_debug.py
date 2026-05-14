"""Tests for the probe → debug → retry → rescale chain in the orchestrator.

Focuses on the non-obvious case:
  * The mesh probe fails (n_cells is None) because custom_mesh.py crashes.
  * Without the new debug-retry, the orchestrator would bail and the
    rescale rule would never fire.
  * With the new debug-retry, _probe_and_maybe_rescale invokes the
    multi-file Debugger, which patches the mesh module, the probe
    re-runs and reports a real n_cells, and the rescale loop proceeds.

The unit test mocks ``run_script`` (no WSL), ``debugger`` (no LLM), and
``render_script`` (no template render).  It exercises just the control
flow inside ``_probe_and_maybe_rescale``.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest

from fracture_agent import orchestrator as orch
from fracture_agent.executor import ExecResult
from fracture_agent.mesh import MeshSizePlan
from fracture_agent.schema import (Action, BCSpec, CanonicalSpec, FixedBC, GeometrySpec,
                             LoadingCase, LoadingSpec, MaterialSpec, MeshStrategy)
from fracture_agent.state import SessionState


# -------------------------------------------------------------------------- #
# Test fixtures
# -------------------------------------------------------------------------- #
def _spec() -> CanonicalSpec:
    return CanonicalSpec(
        material=MaterialSpec(display_name="Test", E=10000, nu=0.2,
                              Gc=0.05, sigma_ts=20, sigma_cs=40),
        geometry=GeometrySpec(kind="custom", dimensions={"W": 1.0, "L": 1.0},
                              custom_description="square 1mm",
                              custom_regions=[]),
        bcs=BCSpec(fixed=[FixedBC(region="bottom", components=[0, 1])],
                   loading=[LoadingCase(region="top", component=1, magnitude=0.1)]),
        loading=LoadingSpec(steps=10, T_total=1.0),
        mesh=MeshStrategy(target_min_cells=5000, max_rescales=4),
    )


def _action() -> Action:
    return Action(variant="linear_elastic_2d_ps",
                  mesh_builder="make_custom_gmsh",
                  solver="run_quasistatic",
                  rationale="test")


def _plan(eps: float = 0.7) -> MeshSizePlan:
    return MeshSizePlan(eps=eps, h0=2 * eps, h_min=eps / 4,
                        n_cells_expected=0, n_rescales=0, notes="initial")


def _exec(rc: int, n_cells: Optional[int], tail: str = "") -> ExecResult:
    return ExecResult(
        returncode=rc, wall_time_s=1.0,
        stdout=tail, stderr="", diverged=False,
        error_signature=None, tail=tail, n_cells=n_cells)


# -------------------------------------------------------------------------- #
# The happy path (probe never fails) — confirms backward-compat
# -------------------------------------------------------------------------- #
def test_probe_no_failure_rescales_until_target_met(tmp_path, monkeypatch):
    """Classic path: probe always reports n_cells, rescale fires until
    target is met.  No Debugger invocation."""
    state = SessionState(session_id="t")
    state.spec = _spec()
    spec, action, plan = state.spec, _action(), _plan(eps=0.5)
    script = tmp_path / "run_linear_elastic_2d_ps.py"
    script.write_text("# initial\n", encoding="utf-8")

    # Probe sequence: 11 -> 5500 (target=5000 met, stop)
    cell_counts = iter([11, 5500])
    debugger_calls = []

    def fake_run_script(s_path, **_kw):
        return _exec(0, next(cell_counts))

    def fake_debugger(s_path, tail):
        debugger_calls.append(s_path)
        return s_path  # never called in this test

    def fake_render(s_spec, s_action, s_plan, s_dir, s_variant):
        # Re-emit the same path; in real life it'd overwrite with new eps.
        return script

    monkeypatch.setattr(orch, "run_script", fake_run_script)
    monkeypatch.setattr(orch, "debugger",   fake_debugger)
    monkeypatch.setattr(orch, "render_script", fake_render)

    out = orch._probe_and_maybe_rescale(state, spec, action, script, plan)
    assert out == script
    assert debugger_calls == []                    # no Debugger needed
    assert plan.n_rescales == 1                    # one rescale fired
    # Rescale formula: eps_new = 0.5 * (11/5000)^(1/2) ≈ 0.0234
    assert plan.eps == pytest.approx(0.5 * (11 / 5000) ** 0.5, rel=1e-3)


# -------------------------------------------------------------------------- #
# The bug-fix case — probe fails, Debugger fixes mesh, probe retries
# -------------------------------------------------------------------------- #
def test_probe_failure_invokes_debugger_then_retries(tmp_path, monkeypatch):
    """Probe crashes (n_cells=None) -> Debugger fires -> probe retries
    on the patched script and reports a real n_cells -> rescale fires."""
    state = SessionState(session_id="t")
    state.spec = _spec()
    spec, action, plan = state.spec, _action(), _plan(eps=0.5)
    script = tmp_path / "run_linear_elastic_2d_ps.py"
    script.write_text("# initial\n", encoding="utf-8")
    mesh = tmp_path / "custom_mesh.py"
    mesh.write_text("# original mesh module\n", encoding="utf-8")

    # Sequence:
    #   (1) probe attempt 1: fails (n_cells=None)
    #   (2) probe attempt 1 retry after Debugger: n_cells=11
    #   (3) probe attempt 2 (post-rescale, on regenerated script): n_cells=5500
    probe_results = iter([
        _exec(1, None, tail='File "/tmp/custom_mesh.py", line 7\nTypeError\n'),
        _exec(0, 11),
        _exec(0, 5500),
    ])
    debugger_calls = []

    def fake_run_script(s_path, **_kw):
        return next(probe_results)

    def fake_debugger(s_path, tail):
        debugger_calls.append((str(s_path), tail))
        # Simulate the multi-file Debugger touching custom_mesh.py
        # (would normally be done by the real Debugger).
        mesh.write_text("# patched mesh\n", encoding="utf-8")
        # And return a new attempt path for the driver.
        new_p = s_path.with_name(s_path.stem + ".attempt1.py")
        new_p.write_text("# patched driver\n", encoding="utf-8")
        return new_p

    def fake_render(s_spec, s_action, s_plan, s_dir, s_variant):
        # In real life this overwrites run_*.py with new eps.  Here we
        # just re-emit a path that the next probe will run.
        return script

    monkeypatch.setattr(orch, "run_script", fake_run_script)
    monkeypatch.setattr(orch, "debugger",   fake_debugger)
    monkeypatch.setattr(orch, "render_script", fake_render)

    out = orch._probe_and_maybe_rescale(state, spec, action, script, plan)
    assert out == script
    # Debugger was invoked exactly once (1 probe failure → 1 debug call).
    assert len(debugger_calls) == 1
    # mesh patch was recorded in the audit trail.
    assert any(m["trigger"].startswith("probe_debug_attempt")
               for m in state.generated_meshes)
    # Rescale fired once (after the second successful probe).
    assert plan.n_rescales == 1
    # The patched mesh content survived.
    assert "patched mesh" in mesh.read_text(encoding="utf-8")


# -------------------------------------------------------------------------- #
# Bounded retry: chronically broken mesh module shouldn't hang
# -------------------------------------------------------------------------- #
def test_probe_failure_exhausts_debug_budget_and_bails(tmp_path, monkeypatch):
    """If the mesh module keeps crashing through MAX_PROBE_DEBUG_ITERS
    rounds, _probe_and_maybe_rescale must bail without rescaling."""
    state = SessionState(session_id="t")
    state.spec = _spec()
    spec, action, plan = state.spec, _action(), _plan(eps=0.5)
    script = tmp_path / "run_linear_elastic_2d_ps.py"
    script.write_text("# initial\n", encoding="utf-8")
    mesh = tmp_path / "custom_mesh.py"
    mesh.write_text("# bug\n", encoding="utf-8")

    n_calls = [0]
    def fake_run_script(s_path, **_kw):
        n_calls[0] += 1
        return _exec(1, None, tail='File "/tmp/custom_mesh.py", line 7\n')

    def fake_debugger(s_path, tail):
        # Simulate a debug attempt that doesn't actually fix the bug.
        return s_path  # same path, no actual patch

    monkeypatch.setattr(orch, "run_script", fake_run_script)
    monkeypatch.setattr(orch, "debugger",   fake_debugger)
    monkeypatch.setattr(orch, "render_script",
                        lambda *a, **k: script)

    out = orch._probe_and_maybe_rescale(state, spec, action, script, plan)
    assert out == script
    # 1 initial probe + MAX_PROBE_DEBUG_ITERS retries = 4 calls before bail.
    assert n_calls[0] == 1 + orch.MAX_PROBE_DEBUG_ITERS
    # No rescale could fire because no probe ever succeeded.
    assert plan.n_rescales == 0


# -------------------------------------------------------------------------- #
# Confirm the constant is sensible (don't accidentally let it explode)
# -------------------------------------------------------------------------- #
def test_max_probe_debug_iters_is_bounded():
    assert 1 <= orch.MAX_PROBE_DEBUG_ITERS <= 10
