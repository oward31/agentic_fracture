"""Process-wide event sink for agent-activity streaming.

The orchestrator and agents call :func:`emit` at notable moments
(''I picked plane stress'', ''running in WSL'', ''crack initiated at step
73'').  A single global sink forwards these events to whatever front-end is
currently active — the CLI prints them, the web UI pushes them into an
SSE queue.  Defaults to a no-op so unit tests aren't chatty.
"""
from __future__ import annotations
from typing import Callable, Optional


# Event categories — the UI styles each differently.
STATUS    = "status"      # ''what I'm doing now'' (Architect running, WSL launching)
DECISION  = "decision"    # ''why I picked X'' (plane strain, catalog material, bc_spec)
CONSOLE   = "console"     # raw WSL / solver output
RESULT    = "result"      # final verdict from Advisor
ERROR     = "error"       # something broke
DONE      = "done"        # pipeline finished
IMAGE     = "image"       # a rendered PNG is ready at <path>
METADATA  = "meta"        # one-shot spec/action summary


_SINK: Optional[Callable[[str, str], None]] = None


def set_sink(fn: Optional[Callable[[str, str], None]]) -> None:
    global _SINK
    _SINK = fn


def emit(category: str, message: str) -> None:
    """Publish one event.  Safe to call with sink unset."""
    if _SINK is not None:
        try:
            _SINK(category, message)
        except Exception:
            pass


def tee(category: str, message: str) -> None:
    """Emit AND print — useful during development."""
    emit(category, message)
    print(f"[{category}] {message}", flush=True)
