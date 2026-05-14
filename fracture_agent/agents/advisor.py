"""Advisor — user-facing Q&A about a finished run.

The Advisor has two jobs:
  1. Given a fresh run, produce a short plain-English summary of what
     happened ("The bar cracked at step 73 at y=0 ...").
  2. Answer follow-up user questions using the ResultSummary + recent log
     excerpt as grounded context.

We avoid feeding the full XDMF into the LLM — the numbers come from
results.summarise() and the log parser.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import FAST_MODEL
from ..health import HealthReport, compute_health
from ..llm import llm
from ..results import parse_log, summarise
from ..schema import CanonicalSpec, ResultSummary


SUMMARY_SYS = """You are the Advisor agent of an FEA system.  You see a
ResultSummary (numbers, derived metrics) and are asked to produce a short,
plain-English status report (≤ 80 words).

Rules:
  * Lead with whether the specimen cracked (if fracture enabled), and if so
    at which step and under how much displacement.
  * Report peak reaction force and the displacement at which it occurred.
  * Mention divergence if it happened.
  * Do not invent values that are not in the summary.
"""


ANSWER_SYS = """You are the Advisor agent answering questions about a
completed phase-field fracture FEA run.

You are given:
  * The CanonicalSpec describing the problem.
  * The ResultSummary with derived scalars.
  * The last N steps of the quasistatic log (space-separated).

Ground your answer strictly in these data.  If the question asks for
information not recoverable from log + summary (e.g. "where exactly did the
crack initiate"), say so and suggest opening the XDMF in ParaView.

Keep answers under 120 words unless the user explicitly asks for detail.
"""


def advise_summary(spec: CanonicalSpec, summary: ResultSummary) -> str:
    payload = {"spec": spec.model_dump(), "summary": summary.model_dump()}
    # Flash: the verdict is a 1-paragraph structured summary, not a
    # reasoning task.  Avoids Pro's occasional MAX_TOKENS-at-512-thoughts
    # failure mode.  Answer-Q&A below still uses Pro.
    return llm().complete(SUMMARY_SYS, json.dumps(payload, default=str),
                          temperature=0.1, max_output_tokens=4096,
                          model=FAST_MODEL)


def answer_question(spec: CanonicalSpec,
                    summary: ResultSummary,
                    log_path: Path,
                    question: str,
                    history: Optional[List[Dict[str, str]]] = None) -> str:
    cols = parse_log(log_path)
    tail_rows: List[str] = []
    if cols and cols["step"]:
        n = len(cols["step"])
        k = min(n, 20)
        for i in range(n - k, n):
            tail_rows.append(
                f"{int(cols['step'][i])}  {cols['time'][i]:.4e}  "
                f"{cols['disp'][i]:.4e}  {cols['Fy'][i]:.4e}  "
                f"{cols['min_z'][i]:.4f}")
    ctx = {
        "spec": spec.model_dump(),
        "summary": summary.model_dump(),
        "log_tail": "\n".join(tail_rows),
        "history": history or [],
    }
    user = ("Context:\n" + json.dumps(ctx, default=str, indent=2)
            + f"\n\nUser question: {question}")
    return llm().complete(ANSWER_SYS, user,
                          temperature=0.15, max_output_tokens=8192)


def advisor(spec: CanonicalSpec,
            log_path: Path,
            diverged: bool) -> ResultSummary:
    """Compute the summary and also write a plain-English verdict alongside."""
    s = summarise(log_path,
                  fracture_enabled=spec.fracture_enabled,
                  diverged=diverged)
    return s


def physics_advisor(spec: CanonicalSpec,
                     log_path: Path,
                     *,
                     returncode: int,
                     wall_time_s: float,
                     mesh_plan: Dict[str, Any]) -> HealthReport:
    """Compute the 100-pt composite admissibility reward.

    Pure derivation from the per-step log + (returncode, wall_time, mesh_plan)
    — no XDMF readback, no LLM authoring of physics.  Returns a structured
    ``HealthReport`` whose ``verdict`` ∈ {accept, warn, revise} drives the
    Reflect-Revise outer loop in the orchestrator.
    """
    cols = parse_log(log_path)
    eps = float(mesh_plan.get("eps") or 0.0)
    h_min = float(mesh_plan.get("h_min") or 0.0)
    # First step's stepsize is T_total / steps; the orchestrator may have
    # adjusted it via rescaling, but the canonical value is the spec's.
    dt_first = float(spec.loading.T_total) / max(int(spec.loading.steps), 1)
    return compute_health(
        cols,
        returncode=returncode,
        wall_time_s=wall_time_s,
        fracture_enabled=spec.fracture_enabled,
        T_total=float(spec.loading.T_total),
        dt_first=dt_first,
        eps=eps,
        h_min=h_min,
        max_stag=int(spec.max_stag),
    )
