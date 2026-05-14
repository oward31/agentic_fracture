"""Inspector — static-AST + grep-style check on the generated script.

The point is to catch the three classes of error that will waste a full WSL
launch otherwise:
  1.  Legacy `from dolfin import` / `import dolfin` (must be dolfinx only).
  2.  Imports of modular submodules that don't exist (typos).
  3.  A SyntaxError (AST parse failure).
"""
from __future__ import annotations
import ast
from pathlib import Path
from typing import List, Tuple


FORBIDDEN = (
    ("from dolfin import", "legacy FEniCS — must use dolfinx"),
    ("import dolfin\n",    "legacy FEniCS — must use dolfinx"),
    ("from fenics import", "legacy FEniCS — must use dolfinx"),
)


_ALLOWED_MODULAR = {
    "modular.materials", "modular.meshes", "modular.problems",
    "modular.common",    "modular.post",   "modular.solvers",
    "modular.constitutive",
    "modular.materials.loader",
}


def inspector(script_path: Path) -> Tuple[bool, List[str]]:
    """Return (ok, issues)."""
    issues: List[str] = []
    src = script_path.read_text(encoding="utf-8")

    for needle, reason in FORBIDDEN:
        if needle in src:
            issues.append(f"Forbidden pattern '{needle.strip()}' — {reason}")

    # AST parse.
    try:
        tree = ast.parse(src, filename=str(script_path))
    except SyntaxError as e:
        issues.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        return False, issues

    # Import sanity: any `modular.*` must be in the allowed set.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("modular") and mod not in _ALLOWED_MODULAR:
                issues.append(f"Suspect import 'from {mod} import ...' — "
                              f"not a known modular submodule")
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith("modular") and n.name not in _ALLOWED_MODULAR:
                    issues.append(f"Suspect import '{n.name}' — "
                                  f"not a known modular submodule")

    # Must reference at least one run_* solver call.
    if not any(s in src for s in (
            "run_quasistatic", "run_dynamic",
            "run_finite_elasticity", "run_ductile")):
        issues.append("No solver call found (run_quasistatic / run_dynamic / "
                      "run_finite_elasticity / run_ductile)")

    # If there's a sibling custom_mesh.py, AST-check that too — a syntax
    # error there would only show up after a slow WSL launch.
    cm = script_path.with_name("custom_mesh.py")
    if cm.exists():
        try:
            cm_src = cm.read_text(encoding="utf-8")
            ast.parse(cm_src, filename=str(cm))
        except SyntaxError as e:
            issues.append(f"custom_mesh.py SyntaxError at line {e.lineno}: {e.msg}")
        else:
            if "def make_custom_gmsh" not in cm_src:
                issues.append("custom_mesh.py: missing `make_custom_gmsh` function")
            if "markers_spec" not in cm_src:
                issues.append("custom_mesh.py: no markers_spec in return")

    return (not issues), issues
