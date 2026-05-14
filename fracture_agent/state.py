"""Per-session state container — the shared memory every agent reads & writes.

Stored on disk as pretty-printed JSON at runs/<session_id>/state.json so a
crashed session can resume and so the Advisor has a stable object to report on.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import RUNS_DIR
from .health import HealthReport
from .schema import Action, CanonicalSpec, ResultSummary
from .telemetry import SessionTelemetry


class ExecutionRecord(BaseModel):
    script_path: str
    returncode: int
    wall_time_s: float
    stdout_tail: str = ""
    stderr_tail: str = ""
    diverged: bool = False
    error_signature: Optional[str] = None                # e.g. "SNES_DIVERGED_LINEAR_SOLVE"
    n_cells: Optional[int] = None                        # from [mesh-audit] print


class SessionState(BaseModel):
    session_id: str
    raw_inputs: List[Dict[str, Any]] = Field(default_factory=list)      # user turns
    clarifications: List[Dict[str, str]] = Field(default_factory=list)  # Q/A pairs

    spec: Optional[CanonicalSpec] = None
    action: Optional[Action] = None

    mesh_plan: Dict[str, Any] = Field(default_factory=dict)             # eps/h0 etc.
    generated_scripts: List[str] = Field(default_factory=list)          # file paths
    # Mesh-module audit trail.  For built-in geometries this stays empty;
    # for custom geometries the Synthesizer writes the initial path here,
    # and the Debugger may overwrite the file in place when a traceback
    # points into custom_mesh.py — this list records the "(timestamp,
    # action)" of each touch so a reader can tell which iteration of the
    # mesh was in effect.
    generated_meshes: List[Dict[str, str]] = Field(default_factory=list)
    runs: List[ExecutionRecord] = Field(default_factory=list)

    final_result: Optional[ResultSummary] = None
    health_report: Optional[HealthReport] = None        # 100-pt composite (A1)
    revision_history: List[Dict[str, Any]] = Field(default_factory=list)
    conversation: List[Dict[str, str]] = Field(default_factory=list)    # post-run Q&A

    # Per-session telemetry (token counts, cost, wall-clock, iteration counters).
    # The orchestrator sets this as the active sink at session start; the LLM
    # transport in ``llm.py`` records each call.  Counters are bumped at the
    # appropriate phase boundaries.
    telemetry: SessionTelemetry = Field(default_factory=SessionTelemetry)

    @property
    def dir(self) -> Path:
        d = RUNS_DIR / self.session_id
        d.mkdir(exist_ok=True, parents=True)
        return d

    def save(self) -> None:
        # Force UTF-8 — solver stdout frequently contains unicode glyphs
        # (checkmarks, box-drawing) that cp1252 on Windows cannot encode.
        (self.dir / "state.json").write_text(
            self.model_dump_json(indent=2), encoding="utf-8")
        # Per-call CSV next to state.json — easy to slurp into a pandas
        # DataFrame across many sessions for the cost/wall-clock table.
        try:
            self.telemetry.write_cost_csv(self.dir / "cost_table.csv")
        except Exception:
            pass  # never let telemetry break a save

    @classmethod
    def new(cls) -> "SessionState":
        # Placeholder name — renamed to a descriptive slug once the
        # CanonicalSpec is available (see :meth:`rename_to_slug`).
        sid = time.strftime("pending_%Y%m%d_%H%M%S")
        s = cls(session_id=sid)
        s.save()
        return s

    @classmethod
    def load(cls, session_id: str) -> "SessionState":
        data = json.loads(
            (RUNS_DIR / session_id / "state.json").read_text(encoding="utf-8"))
        return cls(**data)

    # ------------------------------------------------------------------ #
    def rename_to_slug(self) -> Path:
        """Move the session folder to a descriptive slug derived from
        ``self.spec``.  Safe to call repeatedly; a no-op if the current name
        already matches.  Collisions get a short numeric suffix."""
        if self.spec is None:
            return self.dir
        from .slug import session_slug
        stamp = self.session_id.split("_", 1)[-1] if "_" in self.session_id \
            else time.strftime("%Y%m%d_%H%M%S")
        if self.session_id.startswith("pending_"):
            stamp = self.session_id[len("pending_"):]
        new_id = session_slug(self.spec, stamp=stamp)
        if new_id == self.session_id:
            return self.dir
        old = RUNS_DIR / self.session_id
        new = RUNS_DIR / new_id
        # Avoid collisions.
        suffix = 0
        while new.exists():
            suffix += 1
            new = RUNS_DIR / f"{new_id}_{suffix:02d}"
        old.rename(new)
        self.session_id = new.name
        self.save()
        return self.dir
