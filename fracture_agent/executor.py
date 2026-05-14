"""Run a generated script inside WSL, stream output, detect divergence.

Windows → WSL hand-off pattern (from modular/README.md):

    cd /mnt/c/Users/adahal8/Downloads/agent_full_2/agent_v2
    wsl
    conda activate fenicsx
    python <script.py>
    # or
    mpirun -n 4 python <script.py>

We run the whole chain in a single ``wsl.exe`` call:

    wsl -- bash -lc 'cd <wsl-path> && source ~/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx && mpirun -n N python <script>'

On Windows 10/11, ``wsl.exe`` is always on PATH.  We capture stdout+stderr
merged so the debugger sees the actual traceback location.
"""
from __future__ import annotations
import os
import re
import shlex
import signal
import subprocess
import sys
from shlex import quote as shlex_quote
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

from .config import EXECUTION_TIMEOUT_S, WSL_CONDA_ENV, WSL_MPI_DEFAULT, repo_root_wsl


# ---------------------------------------------------------------------------
# Divergence / error signatures worth pattern-matching as we stream.
# ---------------------------------------------------------------------------
DIVERGENCE_PATTERNS = [
    re.compile(r"DIVERGED_(LINEAR|LS|FNORM|FUNCTION_COUNT|ITS)", re.I),
    re.compile(r"Newton solver did not converge", re.I),
    re.compile(r"SNES .*diverged", re.I),
    re.compile(r"RuntimeError: .*diverge", re.I),
]
ERROR_PATTERNS = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\s*(Error|Fatal Error|FATAL)"),
    re.compile(r"^\s*(\w+Error:|\w+Exception:)"),
]
# Captures the mesh size the generated script prints right after meshing.
# Needed independently of the rolling log tail, which evicts the line
# during 100+-step runs.
NCELLS_PATTERN = re.compile(r"\[mesh-audit\]\s*n_cells_global\s*=\s*(\d+)")


@dataclass
class ExecResult:
    returncode: int
    wall_time_s: float
    stdout: str
    stderr: str
    diverged: bool
    error_signature: Optional[str] = None
    tail: str = ""                   # last ~50 lines combined
    n_cells: Optional[int] = None    # captured from [mesh-audit] line


def _path_to_wsl(win_path: Path) -> str:
    """Convert a Windows path to its /mnt/<drive>/... WSL equivalent."""
    p = win_path.as_posix()
    if len(p) >= 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def _build_wsl_cmd(script_wsl_path: str, nprocs: int,
                   conda_env: str, run_dir_wsl: str,
                   extra_args: Optional[list] = None) -> list:
    """Compose the single ``wsl.exe -- bash -ilc '...'`` call.

    ``bash -ilc`` forces an interactive login shell so the user's .bashrc
    (where conda init usually lives) is sourced — necessary because many
    distros guard .bashrc with ``case $- in *i*) ;; *) return;; esac`` and
    would otherwise skip conda initialisation in non-interactive mode.

    We additionally probe a few common conda install paths as a hard
    fallback in case the distro has no conda-init at all.
    """
    conda_probe = (
        "if ! command -v conda >/dev/null 2>&1; then "
        " for p in \"$HOME/miniconda3\" \"$HOME/anaconda3\" "
        "          \"$HOME/miniforge3\" \"/opt/conda\" "
        "          \"/usr/local/miniconda3\" \"/usr/local/anaconda3\"; do "
        "   if [ -f \"$p/etc/profile.d/conda.sh\" ]; then "
        "     . \"$p/etc/profile.d/conda.sh\"; break; "
        "   fi; done; "
        "fi; ")
    activate = f"conda activate {conda_env} 2>&1 || true; "
    args_str = ""
    if extra_args:
        args_str = " " + " ".join(shlex_quote(a) for a in extra_args)
    if nprocs > 1:
        run = f"mpirun -n {nprocs} python -u {script_wsl_path}{args_str}"
    else:
        run = f"python -u {script_wsl_path}{args_str}"
    inner = (conda_probe
             + f"cd {run_dir_wsl}; "
             + activate
             + run + " 2>&1")
    # -i for interactive (sources .bashrc), -l for login (.profile), -c for cmd
    return ["wsl.exe", "--", "bash", "-ilc", inner]


# ---------------------------------------------------------------------------
# Streaming runner
# ---------------------------------------------------------------------------
def run_script(script_win_path: Path,
               *,
               nprocs: int = WSL_MPI_DEFAULT,
               conda_env: str = WSL_CONDA_ENV,
               timeout_s: int = EXECUTION_TIMEOUT_S,
               echo: bool = True,
               on_line: Optional[callable] = None,
               extra_args: Optional[list] = None,
               run_dir: Optional[Path] = None) -> ExecResult:
    """Execute ``script_win_path`` inside WSL.

    ``run_dir`` — cwd for the WSL process; defaults to the script's own dir so
    per-session XDMFs and logs land alongside the generated .py.
    """
    # Always resolve to absolute before converting to WSL-style — a
    # relative path would otherwise stack onto the WSL cwd.
    script_win_path = script_win_path.resolve()
    run_dir_abs = (run_dir or script_win_path.parent).resolve() \
        if run_dir else script_win_path.parent
    script_wsl = _path_to_wsl(script_win_path)
    run_dir_wsl = _path_to_wsl(run_dir_abs)
    cmd = _build_wsl_cmd(script_wsl, nprocs, conda_env, run_dir_wsl,
                         extra_args=extra_args)

    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace")

    # Windows cp1252 terminals choke on the modular solver's fancy unicode
    # (┌─┐, ✓, ⚠, …).  Make every echo ascii-safe regardless of the console.
    def _safe_echo(s: str) -> None:
        try:
            sys.stdout.write(s)
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write(s.encode("ascii", "replace").decode("ascii"))
            sys.stdout.flush()

    tail: list = []
    diverged = False
    n_cells: Optional[int] = None
    err_sig: Optional[str] = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if echo:
                _safe_echo(line)
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:
                    pass
            # Capture mesh-audit count as it streams — the rolling tail
            # cap would otherwise drop it during long runs.
            if n_cells is None:
                m = NCELLS_PATTERN.search(line)
                if m:
                    try: n_cells = int(m.group(1))
                    except ValueError: pass
            tail.append(line)
            if len(tail) > 300:
                tail.pop(0)
            for pat in DIVERGENCE_PATTERNS:
                if pat.search(line):
                    diverged = True
                    err_sig = err_sig or pat.pattern
            for pat in ERROR_PATTERNS:
                if pat.search(line):
                    err_sig = err_sig or line.strip()
            if time.monotonic() - start > timeout_s:
                proc.terminate()
                tail.append(f"\n[fracture_agent] TIMEOUT after {timeout_s}s — terminating.\n")
                break
        proc.wait(timeout=30)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass

    wall = time.monotonic() - start
    full = "".join(tail)
    return ExecResult(
        returncode=proc.returncode if proc.returncode is not None else -1,
        wall_time_s=wall,
        stdout=full,
        stderr="",
        diverged=diverged,
        error_signature=err_sig,
        tail="\n".join(tail[-50:]),
        n_cells=n_cells,
    )
