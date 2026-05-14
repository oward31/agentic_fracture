"""Parse post-run artefacts (log + XDMF) into a ResultSummary.

The modular quasistatic/ductile solvers write one line per accepted step to
``output_*.txt`` with the header:

    step  time  dt  stag_iters  u_res  z_res  min_z  disp  Fy

This file alone answers the most common user questions:
  * Did it crack?  → any ``min_z`` below a small threshold.
  * When?          → first step at which min_z < 0.3 (default).
  * Max reaction?  → ``max(Fy)``.
  * Max stress?    → requires reading XDMF; we use ``pyvista`` only on demand
    inside WSL to avoid a hard dependency here.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import ResultSummary


HEADER_RE = re.compile(r"^\s*step\s+time", re.I)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
def parse_log(log_path: Path) -> Dict[str, List[float]]:
    """Return a dict of columns (``step``, ``time``, ``dt``, ``u_res``,
    ``z_res``, ``min_z``, ``disp``, ``Fy``).  Missing file → empty."""
    if not log_path.exists():
        return {}
    cols: Dict[str, List[float]] = {
        "step": [], "time": [], "dt": [], "stag_iters": [],
        "u_res": [], "z_res": [], "min_z": [], "disp": [], "Fy": []}
    with open(log_path, "r") as fh:
        for line in fh:
            if HEADER_RE.match(line):
                continue
            toks = line.strip().split()
            if len(toks) < 9:
                continue
            try:
                cols["step"].append(float(toks[0]))
                cols["time"].append(float(toks[1]))
                cols["dt"].append(float(toks[2]))
                cols["stag_iters"].append(float(toks[3]))
                cols["u_res"].append(float(toks[4]))
                cols["z_res"].append(float(toks[5]))
                cols["min_z"].append(float(toks[6]))
                cols["disp"].append(float(toks[7]))
                cols["Fy"].append(float(toks[8]))
            except ValueError:
                continue
    return cols


def summarise(log_path: Path,
              *,
              crack_threshold: float = 0.3,
              fracture_enabled: bool = True,
              diverged: bool = False) -> ResultSummary:
    cols = parse_log(log_path)
    if not cols or not cols["step"]:
        return ResultSummary(diverged=diverged,
                             message="no log data — run produced no steps")

    n = len(cols["step"])
    final_disp   = cols["disp"][-1]
    final_F      = cols["Fy"][-1]
    peak_F_idx   = max(range(n), key=lambda i: abs(cols["Fy"][i]))
    peak_F       = cols["Fy"][peak_F_idx]
    peak_F_disp  = cols["disp"][peak_F_idx]
    min_z        = min(cols["min_z"])

    cracked = fracture_enabled and (min_z < crack_threshold)
    init_step: Optional[int] = None
    if cracked:
        for i, z in enumerate(cols["min_z"]):
            if z < crack_threshold:
                init_step = int(cols["step"][i])
                break

    msg_parts: List[str] = [f"{n} accepted steps, final disp={final_disp:.3e}"]
    if cracked:
        msg_parts.append(f"cracked at step {init_step} (min z = {min_z:.3f})")
    elif fracture_enabled:
        msg_parts.append(f"no crack (min z = {min_z:.3f})")
    else:
        msg_parts.append("fracture disabled — deformation only")
    if diverged:
        msg_parts.append("solver diverged before reaching target disp")

    return ResultSummary(
        n_steps=n, final_time=cols["time"][-1],
        final_disp=final_disp, final_reaction=final_F,
        peak_reaction=peak_F, peak_reaction_disp=peak_F_disp,
        min_z=min_z, cracked=cracked, crack_initiated_step=init_step,
        diverged=diverged, message="; ".join(msg_parts),
    )


# ---------------------------------------------------------------------------
# XDMF helper — delegated to WSL so it uses the solver's Python env, not ours.
# The caller passes an ExecResult; we extract the last sigma_vm by running a
# tiny one-shot Python snippet through the same wsl infrastructure.
# ---------------------------------------------------------------------------
XDMF_PROBE_TEMPLATE = '''
import sys, glob, os
xdmf_dir = sys.argv[1]
try:
    import pyvista as pv
except ImportError:
    print("PV_UNAVAILABLE"); sys.exit(0)

frames = sorted(glob.glob(os.path.join(xdmf_dir, "step_*.xdmf")))
if not frames:
    print("NO_FRAMES"); sys.exit(0)
last = frames[-1]
try:
    r = pv.XdmfReader(last); m = r.read()
    if hasattr(m, "GetBlock"):
        m = m[0]
    names = list(m.array_names) if hasattr(m, "array_names") else []
    print("ARRAYS=" + ",".join(names))
    for nm in names:
        a = m.get_array(nm)
        if a is None:
            continue
        try:
            import numpy as np
            print(f"STATS {nm} min={np.min(a):.6e} max={np.max(a):.6e}")
        except Exception:
            pass
except Exception as e:
    print(f"READ_FAIL: {e}")
'''
