"""Debugger — propose patched files from a failed run's traceback.

Two modes:

  * **Driver-only** (current behaviour, used when the traceback doesn't
    point at the custom mesh module): the LLM gets the driver script and
    the stdout tail and returns a fixed full replacement script.  The
    file is written to ``run_<variant>.attempt<N>.py`` and the orchestrator
    re-runs it.

  * **Driver + custom_mesh** (new — fires when the traceback contains a
    ``File "...custom_mesh.py"`` frame AND the session has a custom mesh
    module): the LLM additionally sees ``custom_mesh.py``, returns the
    fixed file(s) using ``### FILE: <name>`` headers, and the patched
    mesh module overwrites the original (gmsh code is regenerated in
    place; there's no per-attempt mesh suffix because the driver imports
    ``from custom_mesh import make_custom_gmsh`` by exact name).

For mesh patches the marker names from the pre-patch ``markers_spec`` are
re-enforced on the LLM output via :func:`fracture_agent.region_names.enforce_marker_names`,
so a sloppy LLM rename can't break the BC plumbing the Synthesizer
already validated.

The driver script path is always returned (callers don't need to know
about the optional mesh edit).  Mesh patches are recorded on the
``SessionState`` via ``state.generated_meshes`` for the audit trail.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import FAST_MODEL
from ..events import DECISION, emit
from ..llm import llm
from ..region_names import enforce_marker_names, extract_marker_names


# Match a Python traceback frame pointing into custom_mesh.py.  Tolerates
# any path prefix (Windows / WSL / mounted) and any line number suffix.
_MESH_TB_RE = re.compile(r'File\s+"[^"]*custom_mesh\.py"', re.IGNORECASE)


SYS_DRIVER_ONLY = """You are the Debugger agent.  Given a failing FEniCSx / DOLFINx
phase-field fracture driver script and the combined stdout+stderr tail of its
last run, produce a FIXED full replacement script.

Constraints:
  * Make the smallest possible change that fixes the error.
  * Do NOT introduce legacy FEniCS API calls (`from dolfin import ...`).
  * Preserve the overall structure: material load → mesh → problem builder →
    run_* solver.
  * If the error is a solver non-convergence (`SNES_DIVERGED_*`), consider:
      - halving `max_disp`;
      - increasing `max_stag` (up to 40);
      - reducing `steps` divider to give the solver smaller increments.
  * If the error is a missing import, fix the import path against the
    modular/ package only.
  * Do NOT create new files; only edit the driver provided.
  * Return the ENTIRE patched file, no commentary, no backticks.
"""


SYS_MULTIFILE = """You are the Debugger agent.  Given (a) a failing FEniCSx /
DOLFINx phase-field fracture driver script, (b) the sibling custom_mesh.py
gmsh module that the driver imports, and (c) the combined stdout+stderr
tail of the last run, produce FIXED full replacement(s) for whichever
file(s) contain the bug.

The traceback points into custom_mesh.py for THIS failure — the mesh module
likely contains the bug.  But you may patch either file or both, depending
on what the traceback actually says.

RESPONSE FORMAT — strict.  Use header-delimited blocks:

  ### FILE: run_<variant>.py
  <the full content of the patched driver, OR omit this block entirely>

  ### FILE: custom_mesh.py
  <the full content of the patched mesh module, OR omit this block entirely>

You MUST emit at least one block.  Omit a block ONLY when that file did not
need editing.  Do NOT add commentary, do NOT use backticks, do NOT create
new files — only edit the two files provided.

Constraints:
  * Make the smallest possible change that fixes the error.
  * Do NOT introduce legacy FEniCS API calls (`from dolfin import ...`).
  * In custom_mesh.py: preserve the existing `markers_spec` names verbatim
    — they are looked up by exact string match elsewhere.  If a region
    name MUST change, also update bc_spec_u in the driver to match.
  * In custom_mesh.py: preserve the existing function signature
    `def make_custom_gmsh(h0: float, comm=MPI.COMM_WORLD, rank: int = 0)`
    and its return-tuple shape `(msh, markers_spec, geom)`.
  * In custom_mesh.py: do NOT call `gmsh.finalize()` (the caller may
    rebuild several times for AMR).
  * In the driver: preserve the overall structure: material load → mesh →
    problem builder → run_* solver.
  * Common gmsh API gotchas to check for:
      - `gmsh.model.occ.fuse(objectDimTags, toolDimTags, ...)` is BINARY,
        not N-ary.  To union N entities, do iterative binary fuse.
      - `gmsh.model.occ.cut(objectDimTags, toolDimTags)` similarly.
      - `gmsh.model.occ.synchronize()` MUST be called between OCC build
        and `mesh.generate(...)`.
      - `gmsh.model.occ.extrude([(2, surface)], dx, dy, dz)` returns a
        list mixing dim=2 (top/side faces) and dim=3 (volume) entities;
        pick the dim=3 entry as the volume.
"""


def _split_multifile(text: str) -> dict[str, str]:
    """Parse the LLM's multi-file reply into ``{filename_label: content}``.

    Only ``run_*.py`` (any tail) and ``custom_mesh.py`` are recognised.
    Returns an empty dict if no headers are present (caller falls back to
    treating the entire reply as a driver patch).
    """
    parts: dict[str, str] = {}
    matches = list(re.finditer(r'^###\s*FILE:\s*(\S+?)\s*$', text, re.M))
    if not matches:
        return parts
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip("\r\n")
        # Map any "run_..." filename label to a logical "driver" key.
        if name == "custom_mesh.py":
            parts["custom_mesh.py"] = body
        elif name.startswith("run_") and name.endswith(".py"):
            parts["driver"] = body
    return parts


def _strip_fences(s: str) -> str:
    """Drop leading/trailing ``` fences if Flash slipped them in."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        s = s.strip()
    return s


def debugger(script_path: Path, stdout_tail: str) -> Path:
    """Patch the driver (and optionally custom_mesh.py) and return the new
    driver path.  ``stdout_tail`` is the combined stdout/stderr tail from
    the failed run.

    Backward compatible: when no ``custom_mesh.py`` exists alongside the
    driver, or when the traceback doesn't reference it, behaviour is
    identical to the original driver-only Debugger.
    """
    src = script_path.read_text(encoding="utf-8")

    mesh_path = script_path.parent / "custom_mesh.py"
    mesh_src: Optional[str] = None
    mesh_in_traceback = bool(_MESH_TB_RE.search(stdout_tail or ""))
    if mesh_path.exists() and mesh_in_traceback:
        try:
            mesh_src = mesh_path.read_text(encoding="utf-8")
        except Exception:
            mesh_src = None

    # ------------------------------------------------------------------ #
    # Error-Fix RAG (B1, Index 3): retrieve up to 3 prior fixes that match
    # the current stacktrace.  Best-effort — RAG must not block the patch
    # on a build / API hiccup.
    rag_block = ""
    try:
        from ..rag import retrieve_error_fix
        prior = retrieve_error_fix(stdout_tail[-600:], k=3, auto_build=False)
        prior = [p for p in prior if p.score >= 0.5]
        if prior:
            rag_block = "\n--- prior fixes for similar errors ---\n"
            for p in prior:
                meta = p.metadata or {}
                rag_block += (
                    f"\n[score={p.score:.2f}] {meta.get('error_signature', p.id)}\n"
                    f"   applied_fix:\n   {meta.get('applied_fix', '')[:600]}\n"
                    f"   outcome: {meta.get('outcome', '?')}\n")
    except Exception:
        rag_block = ""

    # Build the prompt.  In multi-file mode we show BOTH files and ask for
    # the strict header-delimited reply; otherwise we keep the existing
    # driver-only prompt.
    if mesh_src is not None:
        emit(DECISION,
             "Debugger: traceback points into custom_mesh.py — "
             "loading both files for editing")
        prompt = (
            f"--- current driver ({script_path.name}) ---\n{src}\n\n"
            f"--- current custom_mesh.py ---\n{mesh_src}\n\n"
            f"--- stdout tail ---\n{stdout_tail[-4000:]}\n"
            + rag_block + "\n"
            f"Return the fixed file(s) now, header-delimited.")
        sys_prompt = SYS_MULTIFILE
    else:
        prompt = (
            f"--- current script ({script_path.name}) ---\n{src}\n\n"
            f"--- stdout tail ---\n{stdout_tail[-4000:]}\n"
            + rag_block + "\n"
            f"Return the fixed script now.")
        sys_prompt = SYS_DRIVER_ONLY

    fixed_raw = llm().complete(sys_prompt, prompt, temperature=0.1,
                               max_output_tokens=16384, model=FAST_MODEL)
    fixed_raw = _strip_fences(fixed_raw)

    # Write the patched files.  Driver always lands in a fresh
    # `<base>.attempt<N>.py`; mesh module overwrites in place (no good
    # attempt-suffix convention because the driver imports it by name).
    base = script_path.with_suffix("")
    n = 1
    while True:
        candidate = script_path.with_name(f"{base.name}.attempt{n}.py")
        if not candidate.exists():
            break
        n += 1

    if mesh_src is not None:
        parts = _split_multifile(fixed_raw)
        new_driver = parts.get("driver", src)              # default: keep current
        new_mesh   = parts.get("custom_mesh.py")           # may be None
        if not parts:
            # LLM ignored the multi-file format and returned a single script
            # — treat it as a driver patch (legacy fallback).  This is
            # safer than dropping the response entirely.
            emit(DECISION,
                 "Debugger: multi-file response not detected; "
                 "treating reply as driver-only patch")
            new_driver = fixed_raw
            new_mesh = None
        candidate.write_text(new_driver, encoding="utf-8")
        if new_mesh is not None:
            # Re-enforce the original markers_spec names so a sloppy
            # rename can't break the BC plumbing (the Synthesizer's
            # validator already approved the original names against
            # spec.bcs).
            try:
                expected = extract_marker_names(mesh_src)
            except Exception:
                expected = []
            if expected:
                fixed_mesh, warnings = enforce_marker_names(new_mesh, expected)
                if fixed_mesh != new_mesh:
                    new_mesh = fixed_mesh
                for w in warnings:
                    emit(DECISION, f"Debugger custom_mesh.py: {w}")
            mesh_path.write_text(new_mesh, encoding="utf-8")
            emit(DECISION,
                 f"Debugger patched {mesh_path.name} (and "
                 f"{candidate.name}); markers preserved")
        else:
            emit(DECISION,
                 f"Debugger left custom_mesh.py unchanged; patched "
                 f"{candidate.name} only")
    else:
        # Driver-only path.  Identical to the original behaviour.
        candidate.write_text(fixed_raw, encoding="utf-8")

    return candidate


def patched_mesh_path_if_any(driver_path: Path) -> Optional[Path]:
    """Helper for orchestrator audit trail: returns the sibling
    ``custom_mesh.py`` path if it exists, else None.

    The orchestrator records this on every Debugger call so
    ``state.generated_meshes`` reflects which session-folder mesh was
    in effect at each retry.  The mesh file is overwritten in place
    rather than versioned, so what we record is just "the mesh existed
    and may have been patched at this attempt."
    """
    p = driver_path.parent / "custom_mesh.py"
    return p if p.exists() else None
