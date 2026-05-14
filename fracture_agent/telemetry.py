"""Per-session telemetry — token counts, latency, cost, iteration counters.

Reviewers at every venue surveyed (CMAME, npj-AI, EML, ALL-FEM Appx C.1,
ChatCFD §Cost) demand per-case cost / wall-clock numbers.  This module
provides:

    * ``LLMCallRecord``      — one row per Gemini API call.
    * ``IterationCounters``  — round counts the orchestrator increments.
    * ``SessionTelemetry``   — the per-session aggregator persisted to
                                ``state.json["telemetry"]`` and exported
                                as ``cost_table.csv`` on save.
    * ``llm_agent``          — context manager: tag every LLM call inside
                                the ``with`` block with an agent name.

Pricing follows the public Gemini rate card (2025-04, USD per 1M tokens);
update ``GEMINI_PRICES`` if rates change.  Embedding token cost is folded
into ``prompt_tokens`` for now (Gemini bills embedding by characters).
"""
from __future__ import annotations
import contextlib
import threading
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens).  Public Gemini rates as of April 2025; conservative.
# ---------------------------------------------------------------------------
GEMINI_PRICES: Dict[str, Dict[str, float]] = {
    "gemini-2.5-pro":   {"input": 1.25,  "output": 5.00},
    "gemini-2.5-flash": {"input": 0.30,  "output": 2.50},
    "gemini-2.5-pro-thinking":  {"input": 1.25, "output": 5.00},
    "gemini-embedding-001":     {"input": 0.05, "output": 0.0},
}


def _price_for(model: str) -> Dict[str, float]:
    """Return the {input, output} dollar rate per 1M tokens for ``model``,
    falling back to Pro pricing if the model isn't in the table."""
    return GEMINI_PRICES.get(model, GEMINI_PRICES["gemini-2.5-pro"])


def _calc_cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> float:
    p = _price_for(model)
    return (prompt_tokens / 1_000_000.0) * p["input"] \
         + (output_tokens / 1_000_000.0) * p["output"]


# ---------------------------------------------------------------------------
# Per-call record.
# ---------------------------------------------------------------------------
class LLMCallRecord(BaseModel):
    """One Gemini API call, captured at the transport layer."""
    ts: float                                       # monotonic timestamp at start
    model: str
    agent: str = "unknown"                          # filled by ``llm_agent`` context
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0                        # Gemini 2.5 reports this separately
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    used_search: bool = False                       # google_search grounding ON?
    finish_reason: str = ""                         # STOP / MAX_TOKENS / SAFETY / ...
    attempt: int = 1                                # 1-indexed retry attempt at _post


# ---------------------------------------------------------------------------
# Iteration counters — reviewer-observed signal that an agent isn't looping.
# ---------------------------------------------------------------------------
class IterationCounters(BaseModel):
    architect_rounds: int = 0           # clarification loop in conceptualise()
    debugger_attempts: int = 0          # script.attempt2.py, attempt3.py, ...
    mesh_rescales: int = 0              # eps rescale rounds
    reflect_revise_cycles: int = 0      # outer loop (Tier-A1)


# ---------------------------------------------------------------------------
# Session aggregate.
# ---------------------------------------------------------------------------
class SessionTelemetry(BaseModel):
    """One row per session — totals + per-agent + per-call detail.

    Aggregates are recomputed cheaply on every save; counters / call list
    are the source of truth.
    """
    calls: List[LLMCallRecord] = Field(default_factory=list)
    iters: IterationCounters = Field(default_factory=IterationCounters)
    wall_time_solver_s: float = 0.0     # sum of ExecutionRecord.wall_time_s
    started_ts: float = Field(default_factory=time.monotonic)

    # ---- aggregations (computed properties) -------------------------- #
    def totals(self) -> Dict[str, Any]:
        n = len(self.calls)
        prompt = sum(c.prompt_tokens for c in self.calls)
        output = sum(c.output_tokens for c in self.calls)
        thinking = sum(c.thinking_tokens for c in self.calls)
        cost = sum(c.cost_usd for c in self.calls)
        latency_ms = sum(c.latency_ms for c in self.calls)
        wall_total_s = (time.monotonic() - self.started_ts)
        return {
            "n_llm_calls": n,
            "prompt_tokens": prompt,
            "output_tokens": output,
            "thinking_tokens": thinking,
            "total_tokens": prompt + output + thinking,
            "cost_usd": round(cost, 6),
            "llm_latency_s": round(latency_ms / 1000.0, 3),
            "wall_time_solver_s": round(self.wall_time_solver_s, 3),
            "wall_time_total_s": round(wall_total_s, 3),
        }

    def by_agent(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for c in self.calls:
            row = out.setdefault(c.agent, {
                "n_calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                "thinking_tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0,
            })
            row["n_calls"] += 1
            row["prompt_tokens"] += c.prompt_tokens
            row["output_tokens"] += c.output_tokens
            row["thinking_tokens"] += c.thinking_tokens
            row["cost_usd"] += c.cost_usd
            row["latency_ms"] += c.latency_ms
        for v in out.values():
            v["cost_usd"] = round(v["cost_usd"], 6)
            v["latency_s"] = round(v.pop("latency_ms") / 1000.0, 3)
        return out

    # ---- mutation helpers -------------------------------------------- #
    def append_call(self, rec: LLMCallRecord) -> None:
        self.calls.append(rec)

    def add_solver_wall_time(self, dt_s: float) -> None:
        self.wall_time_solver_s += float(dt_s)

    # ---- CSV export -------------------------------------------------- #
    def write_cost_csv(self, path) -> None:
        """Emit a per-call CSV next to ``state.json``.  Pure stdlib csv;
        used by the benchmark harness to aggregate across runs."""
        import csv
        from pathlib import Path
        path = Path(path)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "agent", "model", "prompt_tokens", "output_tokens",
                        "thinking_tokens", "cost_usd", "latency_ms",
                        "used_search", "finish_reason", "attempt"])
            for c in self.calls:
                w.writerow([round(c.ts, 3), c.agent, c.model,
                            c.prompt_tokens, c.output_tokens, c.thinking_tokens,
                            round(c.cost_usd, 6), round(c.latency_ms, 1),
                            c.used_search, c.finish_reason, c.attempt])


# ---------------------------------------------------------------------------
# Active-agent context (thread-local) — set by `with llm_agent("name"):`
# blocks so the LLM transport can tag every call without each agent having
# to thread a parameter.
# ---------------------------------------------------------------------------
_local = threading.local()


def _agent_stack() -> List[str]:
    s = getattr(_local, "stack", None)
    if s is None:
        s = []
        _local.stack = s
    return s


def current_agent() -> str:
    s = _agent_stack()
    return s[-1] if s else "unknown"


@contextlib.contextmanager
def llm_agent(name: str):
    """Tag every LLM call inside this block with ``name``.  Stacks: nested
    blocks restore the outer name on exit."""
    s = _agent_stack()
    s.append(name)
    try:
        yield
    finally:
        s.pop()


# ---------------------------------------------------------------------------
# Active-session sink — set by orchestrator at session start; LLM client
# calls ``record_llm_call(...)`` which dispatches here.
# ---------------------------------------------------------------------------
_active_telemetry: Optional[SessionTelemetry] = None


def set_active(t: Optional[SessionTelemetry]) -> None:
    global _active_telemetry
    _active_telemetry = t


def detach_telemetry() -> None:
    """Convenience: same as ``set_active(None)`` — useful for benchmark
    runners that loop over many sessions in one process."""
    set_active(None)


def get_active() -> Optional[SessionTelemetry]:
    return _active_telemetry


def record_llm_call(*,
                     model: str,
                     prompt_tokens: int,
                     output_tokens: int,
                     thinking_tokens: int = 0,
                     latency_ms: float = 0.0,
                     used_search: bool = False,
                     finish_reason: str = "",
                     attempt: int = 1) -> None:
    """Append an ``LLMCallRecord`` to the active session telemetry, if any.
    No-op when no session is active (e.g. unit tests, embed-only scripts)."""
    t = _active_telemetry
    if t is None:
        return
    rec = LLMCallRecord(
        ts=time.monotonic(),
        model=model,
        agent=current_agent(),
        prompt_tokens=int(prompt_tokens),
        output_tokens=int(output_tokens),
        thinking_tokens=int(thinking_tokens),
        latency_ms=float(latency_ms),
        cost_usd=_calc_cost_usd(model, prompt_tokens, output_tokens + thinking_tokens),
        used_search=bool(used_search),
        finish_reason=str(finish_reason),
        attempt=int(attempt),
    )
    t.append_call(rec)


def add_solver_wall_time(dt_s: float) -> None:
    t = _active_telemetry
    if t is not None:
        t.add_solver_wall_time(dt_s)


def bump(counter: str, n: int = 1) -> None:
    """Increment one of ``IterationCounters``' fields by ``n``."""
    t = _active_telemetry
    if t is None:
        return
    if hasattr(t.iters, counter):
        setattr(t.iters, counter, getattr(t.iters, counter) + n)
