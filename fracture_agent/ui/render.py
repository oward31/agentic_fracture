"""Post-run visualisation.

Generates three PNGs inside a session directory by reading the XDMF frames
the modular solver wrote.  Runs inside WSL (needs pyvista + matplotlib,
both in the DOLFINx conda env).

    python -m fracture_agent.ui.render <session_dir>

Outputs (all beside state.json):
  * initial_mesh.png  — undeformed triangulation
  * final_damage.png  — last-step displaced config, coloured by phase field z
  * load_disp.png     — reaction force vs displacement (matplotlib)
"""
from __future__ import annotations
import glob
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def _find_frames(session_dir: Path):
    """Return the list of step HDF5 files (companion to the XDMF wrappers).
    We read the HDF5 directly because VTK's XDMF reader segfaults on
    dolfinx XDMF3 output."""
    paraview_dirs = sorted(session_dir.glob("paraview_*"))
    if not paraview_dirs:
        return []
    return sorted(paraview_dirs[0].glob("step_*.h5"))


def _read_frame(path: Path):
    """Return (points2D, triangles, fields_dict) by reading the HDF5
    directly with h5py — no VTK dependency."""
    import h5py
    with h5py.File(path, "r") as f:
        # --- mesh --- #
        geom = np.asarray(f["Mesh/mesh/geometry"])          # (N, 2) or (N, 3)
        topo = np.asarray(f["Mesh/mesh/topology"]).astype(np.int64)  # (M, 3) for tris
        pts = geom[:, :2]
        # --- fields --- #
        fields = {}
        if "Function" in f:
            for fname, grp in f["Function"].items():
                # each group has ONE time-indexed dataset; grab the only one.
                if len(grp) == 0:
                    continue
                ds_key = list(grp.keys())[0]
                arr = np.asarray(grp[ds_key])
                # Fields are (N, 1) for scalars, (N, 3) for vectors — squeeze
                # trailing singleton dims for scalars.
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                fields[fname] = arr
        # XDMFFile uses the function's .name as the group name; the modular
        # phase-field variable is exported as "phasefield".  Alias to "z"
        # for the rest of the renderer.
        if "phasefield" in fields and "z" not in fields:
            fields["z"] = fields["phasefield"]
    return pts, topo, fields


# ---------------------------------------------------------------------------
def _get_fields(fields):
    """Robust field picker — names vary across modular variants.
    (Cannot use `a or b` on numpy arrays — their truth value is ambiguous.)"""
    z = None
    for k in ("z", "phasefield", "d", "damage"):
        if k in fields:
            z = fields[k]; break
    u = None
    for k in ("u", "displacement"):
        if k in fields:
            u = fields[k]; break
    return z, u


def plot_final_mesh(out: Path, pts, tris, fields) -> bool:
    """Final (AMR-refined) mesh in undeformed coordinates — mesh only, no
    field overlay."""
    if tris is None or len(tris) == 0:
        return False
    tri = Triangulation(pts[:, 0], pts[:, 1], tris)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1b1d22")
    ax.set_facecolor("#1b1d22")
    ax.triplot(tri, lw=0.35, color="#8fb8ff")
    ax.set_aspect("equal")
    ax.set_title("Final mesh - undeformed", color="#e7e9ec", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def plot_damage_deformed(out: Path, pts, tris, fields) -> bool:
    """Final deformed configuration, coloured by the phase field.
    Colour map: blue at z=1 (intact), red at z=0 (broken)."""
    if tris is None or len(tris) == 0:
        return False
    z, u = _get_fields(fields)
    if z is None:
        return False
    pts_plot = pts.copy()
    title = "Deformed configuration  (phase field z)"
    if u is not None:
        u2 = np.asarray(u)[:, :2]
        span = max(pts[:, 0].ptp(), pts[:, 1].ptp(), 1.0)
        u_mag = float(np.linalg.norm(u2, axis=1).max() or 1e-12)
        scale = 1.0 if u_mag >= 0.03 * span else (0.08 * span) / u_mag
        pts_plot = pts + scale * u2
        if scale != 1.0:
            title = f"{title}  (displacement x{scale:.0f})"

    tri = Triangulation(pts_plot[:, 0], pts_plot[:, 1], tris)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1b1d22")
    ax.set_facecolor("#1b1d22")
    # RdBu (not reversed): RED at low values (z=0, broken),
    #                      BLUE at high values (z=1, intact).
    tpc = ax.tripcolor(tri, z, cmap="RdBu", vmin=0, vmax=1,
                       shading="gouraud")
    ax.triplot(tri, lw=0.18, color="#111", alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(title, color="#e7e9ec", fontsize=11)
    ax.axis("off")
    cb = fig.colorbar(tpc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("z  (1 = intact, 0 = broken)", color="#e7e9ec")
    cb.ax.tick_params(colors="#e7e9ec")
    cb.outline.set_edgecolor("#444")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def plot_load_disp(out: Path, log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        data = np.loadtxt(log_path, skiprows=1)
    except Exception:
        return False
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 9:
        return False
    disp = data[:, 7]
    Fy = data[:, 8]
    fig, ax = plt.subplots(figsize=(5, 3.6), facecolor="#1b1d22")
    ax.set_facecolor("#1b1d22")
    ax.plot(disp, Fy, "-", color="#8fb8ff", lw=1.6)
    ax.plot(disp, Fy, "o", color="#f7d675", ms=3)
    ax.set_xlabel("displacement", color="#e7e9ec")
    ax.set_ylabel("reaction force", color="#e7e9ec")
    ax.set_title("Load-displacement", color="#e7e9ec", fontsize=11)
    ax.grid(alpha=0.25, color="#555")
    for s in ax.spines.values():
        s.set_color("#555")
    ax.tick_params(colors="#aaa")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
def main(session_dir: str) -> int:
    sd = Path(session_dir)
    frames = _find_frames(sd)
    print(f"[render] found {len(frames)} XDMF frames")
    if not frames:
        return 0

    pts_last, tris_last, fields_last = _read_frame(frames[-1])

    if plot_final_mesh(sd / "final_mesh.png", pts_last, tris_last, fields_last):
        print(f"[render] wrote final_mesh.png")
    if plot_damage_deformed(sd / "final_damage.png", pts_last, tris_last,
                             fields_last):
        print(f"[render] wrote final_damage.png")

    logs = list(sd.glob("output_*.txt"))
    if logs and plot_load_disp(sd / "load_disp.png", logs[0]):
        print(f"[render] wrote load_disp.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
