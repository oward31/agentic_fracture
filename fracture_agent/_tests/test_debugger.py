"""Unit tests for the multi-file Debugger.

Pure logic tests — no LLM calls.  We exercise:
  * Traceback regex correctly identifies / rejects custom_mesh frames.
  * Multi-file parser splits a header-delimited LLM response.
  * Multi-file parser is robust to absent / malformed input.
  * The end-to-end ``debugger()`` call honors the backward-compat path
    when no custom_mesh.py exists (driver-only).
  * The end-to-end call honors the multi-file path when custom_mesh.py
    exists AND the traceback points into it.
  * Marker-name preservation re-fixes drift from the LLM patch.

The LLM is monkey-patched with a fake that returns canned strings, so
the tests run offline and are fast.
"""
from __future__ import annotations
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the MODULE (not the re-exported function from fracture_agent.agents).
from fracture_agent.agents import debugger as dbg
if not hasattr(dbg, "_MESH_TB_RE"):
    # `from fracture_agent.agents import debugger` resolves to the function, not
    # the module, because fracture_agent/agents/__init__.py re-exports it.  Pull
    # the actual module via importlib to access internals.
    import importlib
    dbg = importlib.import_module("fracture_agent.agents.debugger")


# -------------------------------------------------------------------------- #
# Traceback regex
# -------------------------------------------------------------------------- #
def test_mesh_tb_regex_matches_unix_path():
    tb = (
        'Traceback (most recent call last):\n'
        '  File "/mnt/c/runs/sid_42/run_linear_elastic_2d_ps.py", line 50, in main\n'
        '    msh, m, g = make_custom_gmsh(...)\n'
        '  File "/mnt/c/runs/sid_42/custom_mesh.py", line 112, in make_custom_gmsh\n'
        '    fused = gmsh.model.occ.fuse(parts)\n'
        'TypeError: ...')
    assert dbg._MESH_TB_RE.search(tb)


def test_mesh_tb_regex_matches_windows_path():
    tb = (
        '  File "C:\\Users\\x\\runs\\sid\\custom_mesh.py", line 7, in make_custom_gmsh\n'
        '    return msh\n')
    assert dbg._MESH_TB_RE.search(tb)


def test_mesh_tb_regex_rejects_when_no_custom_mesh():
    tb = (
        '  File "/mnt/c/runs/sid_42/run_linear_elastic_2d_ps.py", line 50, in main\n'
        '    raise RuntimeError("solver did not converge")\n'
        'RuntimeError: solver did not converge\n')
    assert not dbg._MESH_TB_RE.search(tb)


def test_mesh_tb_regex_case_insensitive_safety():
    # Some tracebacks may say "Custom_Mesh.py" via case-shifted filesystem.
    tb = '  File "/runs/sid/Custom_Mesh.py", line 10, in foo'
    assert dbg._MESH_TB_RE.search(tb)


# -------------------------------------------------------------------------- #
# _split_multifile parser
# -------------------------------------------------------------------------- #
def test_split_multifile_two_blocks():
    text = textwrap.dedent("""\
        ### FILE: run_linear_elastic_2d_ps.py
        # driver content
        from mpi4py import MPI

        ### FILE: custom_mesh.py
        # mesh content
        import gmsh
        """)
    parts = dbg._split_multifile(text)
    assert set(parts) == {"driver", "custom_mesh.py"}
    assert "from mpi4py import MPI" in parts["driver"]
    assert "import gmsh" in parts["custom_mesh.py"]


def test_split_multifile_only_mesh_block():
    text = textwrap.dedent("""\
        ### FILE: custom_mesh.py
        import gmsh
        # mesh-only patch
        """)
    parts = dbg._split_multifile(text)
    assert set(parts) == {"custom_mesh.py"}
    assert "mesh-only patch" in parts["custom_mesh.py"]


def test_split_multifile_only_driver_block():
    text = textwrap.dedent("""\
        ### FILE: run_finite_elastic_3d.attempt2.py
        # driver-only patch
        """)
    parts = dbg._split_multifile(text)
    assert set(parts) == {"driver"}
    assert "driver-only patch" in parts["driver"]


def test_split_multifile_no_headers_returns_empty():
    text = "from mpi4py import MPI\n# raw script with no header\n"
    parts = dbg._split_multifile(text)
    assert parts == {}


def test_split_multifile_ignores_unknown_filename():
    text = textwrap.dedent("""\
        ### FILE: extra_helper.py
        # never accepted
        ### FILE: custom_mesh.py
        # ok
        """)
    parts = dbg._split_multifile(text)
    assert "custom_mesh.py" in parts
    assert "extra_helper.py" not in parts
    # The unknown-named block's content also doesn't leak into the mesh.
    assert "never accepted" not in parts["custom_mesh.py"]


# -------------------------------------------------------------------------- #
# _strip_fences
# -------------------------------------------------------------------------- #
def test_strip_fences_removes_triple_backticks():
    raw = "```python\nimport sys\nprint('x')\n```"
    assert dbg._strip_fences(raw) == "import sys\nprint('x')"


def test_strip_fences_passthrough_when_no_fences():
    raw = "import sys\nprint('x')"
    assert dbg._strip_fences(raw) == "import sys\nprint('x')"


# -------------------------------------------------------------------------- #
# patched_mesh_path_if_any
# -------------------------------------------------------------------------- #
def test_patched_mesh_path_if_any_present(tmp_path):
    drv = tmp_path / "run_x.py"; drv.write_text("# driver", encoding="utf-8")
    mesh = tmp_path / "custom_mesh.py"; mesh.write_text("# mesh", encoding="utf-8")
    assert dbg.patched_mesh_path_if_any(drv) == mesh


def test_patched_mesh_path_if_any_absent(tmp_path):
    drv = tmp_path / "run_x.py"; drv.write_text("# driver", encoding="utf-8")
    assert dbg.patched_mesh_path_if_any(drv) is None


# -------------------------------------------------------------------------- #
# End-to-end debugger() — backward-compat (no custom_mesh.py)
# -------------------------------------------------------------------------- #
class _FakeLLM:
    """Minimal stand-in for fracture_agent.llm.llm() that returns a canned string."""
    def __init__(self, reply: str):
        self._reply = reply
        self.last_sys = None
        self.last_user = None
    def complete(self, sys, user, **_kw):
        self.last_sys = sys
        self.last_user = user
        return self._reply
    def embed(self, *_a, **_kw):  # not used by debugger but kept for safety
        return [0.0] * 8


def _patch_llm(reply: str):
    fake = _FakeLLM(reply)
    return patch("fracture_agent.agents.debugger.llm", lambda: fake), fake


def test_debugger_driver_only_no_mesh_file(tmp_path):
    """No custom_mesh.py present -> driver-only path, identical to legacy."""
    drv = tmp_path / "run_linear_elastic_2d_ps.py"
    drv.write_text("# original driver\n", encoding="utf-8")
    fixed_driver = "# patched driver\nimport sys\n"
    p, fake = _patch_llm(fixed_driver)
    with p:
        out = dbg.debugger(drv, "Traceback ...\nValueError: oops\n")
    assert out.parent == tmp_path
    assert out.name == "run_linear_elastic_2d_ps.attempt1.py"
    # _strip_fences calls .strip() on the LLM reply; trailing newlines go.
    assert out.read_text(encoding="utf-8") == fixed_driver.strip()
    # Driver-only system prompt was used.
    assert "header-delimited" not in fake.last_sys.lower()


def test_debugger_driver_only_when_mesh_not_in_traceback(tmp_path):
    """custom_mesh.py exists but traceback doesn't reference it -> driver-only."""
    drv = tmp_path / "run_linear_elastic_2d_ps.py"; drv.write_text("# drv", encoding="utf-8")
    mesh = tmp_path / "custom_mesh.py"; mesh.write_text("# mesh-original", encoding="utf-8")
    fixed = "# patched driver only\n"
    p, fake = _patch_llm(fixed)
    with p:
        out = dbg.debugger(drv, "RuntimeError: solver diverged\n")
    assert out.read_text(encoding="utf-8") == fixed.strip()
    # Mesh is untouched.
    assert mesh.read_text(encoding="utf-8") == "# mesh-original"
    # Driver-only system prompt.
    assert "header-delimited" not in fake.last_sys.lower()


# -------------------------------------------------------------------------- #
# End-to-end debugger() — multi-file path
# -------------------------------------------------------------------------- #
def _mesh_with_marker(name="bottom", tag=2):
    return textwrap.dedent(f"""\
        import gmsh
        from dolfinx.io import gmshio
        from mpi4py import MPI

        def make_custom_gmsh(h0, comm=MPI.COMM_WORLD, rank=0):
            # ... gmsh body ...
            markers_spec = [
                ({tag}, "{name}", lambda x: x[1] < 1e-9),
            ]
            return None, markers_spec, {{"W": 1.0}}
        """)


def test_debugger_multifile_patches_only_mesh(tmp_path):
    """Traceback points into custom_mesh.py; LLM returns ONLY a mesh block."""
    drv = tmp_path / "run_linear_elastic_2d_ps.py"
    drv.write_text("# driver\n", encoding="utf-8")
    mesh = tmp_path / "custom_mesh.py"
    mesh.write_text(_mesh_with_marker("bottom", 2), encoding="utf-8")

    new_mesh = _mesh_with_marker("bottom", 2).replace(
        "# ... gmsh body ...", "# PATCHED MESH BODY")
    reply = f"### FILE: custom_mesh.py\n{new_mesh}\n"

    tb = (
        '  File "/mnt/c/runs/sid/run_linear_elastic_2d_ps.py", line 50, in main\n'
        '  File "/mnt/c/runs/sid/custom_mesh.py", line 7, in make_custom_gmsh\n'
        'TypeError: fuse() missing 1 required positional argument\n')
    p, fake = _patch_llm(reply)
    with p:
        out = dbg.debugger(drv, tb)

    # Driver path returned (always); driver content unchanged from input.
    assert out.name == "run_linear_elastic_2d_ps.attempt1.py"
    assert out.read_text(encoding="utf-8") == "# driver\n"
    # Mesh was overwritten in place.
    assert "PATCHED MESH BODY" in mesh.read_text(encoding="utf-8")
    # Multi-file system prompt was used.
    assert "header-delimited" in fake.last_sys.lower()


def test_debugger_multifile_patches_both(tmp_path):
    drv = tmp_path / "run_linear_elastic_2d_ps.py"
    drv.write_text("# original driver\n", encoding="utf-8")
    mesh = tmp_path / "custom_mesh.py"
    mesh.write_text(_mesh_with_marker("top", 3), encoding="utf-8")

    new_drv = "# patched driver, both touched\n"
    new_mesh = _mesh_with_marker("top", 3).replace(
        "# ... gmsh body ...", "# PATCHED BOTH FILES")
    reply = (f"### FILE: run_linear_elastic_2d_ps.py\n{new_drv}\n"
             f"### FILE: custom_mesh.py\n{new_mesh}\n")

    tb = '  File "/runs/sid/custom_mesh.py", line 7\n'
    p, _ = _patch_llm(reply)
    with p:
        out = dbg.debugger(drv, tb)
    # Driver block content (multi-file split preserves trailing newline
    # via the slice; we strip both sides for a robust compare).
    assert out.read_text(encoding="utf-8").strip() == new_drv.strip()
    assert "PATCHED BOTH FILES" in mesh.read_text(encoding="utf-8")


def test_debugger_multifile_marker_drift_is_corrected(tmp_path):
    """If the LLM renames a marker, enforce_marker_names re-fixes it."""
    drv = tmp_path / "run_x.py"; drv.write_text("# drv\n", encoding="utf-8")
    mesh = tmp_path / "custom_mesh.py"
    # Original marker is "bottom_legs" (a name the BCs would reference).
    mesh.write_text(_mesh_with_marker("bottom_legs", 5), encoding="utf-8")

    # LLM renames marker to "bottom" — drift.
    drifted_mesh = _mesh_with_marker("bottom", 5).replace(
        "# ... gmsh body ...", "# DRIFTED RENAME")
    reply = f"### FILE: custom_mesh.py\n{drifted_mesh}\n"

    tb = '  File "/runs/sid/custom_mesh.py", line 7\n'
    p, _ = _patch_llm(reply)
    with p:
        dbg.debugger(drv, tb)

    # The post-write mesh must STILL have "bottom_legs", not "bottom".
    final = mesh.read_text(encoding="utf-8")
    assert '"bottom_legs"' in final, f"marker name was not preserved: {final}"
    assert '"bottom"' not in final or '"bottom_legs"' in final
    # The body of the patch is preserved.
    assert "DRIFTED RENAME" in final


def test_debugger_multifile_no_headers_falls_back_to_driver(tmp_path):
    """If multi-file mode is active but the LLM forgot the headers, treat
    the whole reply as a driver patch (legacy fallback) and leave the
    mesh untouched."""
    drv = tmp_path / "run_x.py"; drv.write_text("# original\n", encoding="utf-8")
    mesh = tmp_path / "custom_mesh.py"
    mesh_orig = _mesh_with_marker("bottom", 2)
    mesh.write_text(mesh_orig, encoding="utf-8")

    reply = "# raw driver patch, no headers\n"
    tb = '  File "/runs/sid/custom_mesh.py", line 9\n'
    p, _ = _patch_llm(reply)
    with p:
        out = dbg.debugger(drv, tb)
    assert out.read_text(encoding="utf-8") == reply.strip()
    # Mesh untouched.
    assert mesh.read_text(encoding="utf-8") == mesh_orig
