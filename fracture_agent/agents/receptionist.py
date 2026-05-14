"""Receptionist — multimodal input → JSON fragment list.

Minimal surface: accepts a list of (type, payload) tuples where type ∈
{"text", "image", "audio"} and payload is either the raw string or an
absolute file path.  Emits a structured 'fragments' dict the Architect then
merges.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..config import FAST_MODEL
from ..llm import llm


SYS = """You are the Receptionist agent of a phase-field fracture FEA system.

Your job is to read one user turn — text, an image (sketch / photograph /
drawing), or a transcribed voice clip — and distil it into a JSON fragment:

{
  "intent": "new_problem" | "clarification" | "result_question" | "other",
  "summary": "<one short paraphrase of what the user is asking>",
  "fragments": {
     "geometry":   "<copied verbatim description of shape/dimensions or null>",
     "material":   "<what the user said about material; e.g. 'steel', 'E=210000'>",
     "loading":    "<boundary conditions / forces / displacements mentioned>",
     "fracture":   "<'enable' / 'disable' / null>",
     "output":     "<what the user wants to know or see>",
     "units":      "<mm/m/N/MPa... or null>"
  }
}

Principles:
  * Do NOT invent values the user did not provide.  Leave fields null.
  * If an image is attached, extract any numeric dimensions you can read and
    put them in the geometry fragment — but flag uncertainty with "(approx)".
  * Keep the summary under 25 words.
Respond with the JSON object only.
"""


def receptionist(parts: List[Tuple[str, Any]]) -> Dict[str, Any]:
    user_text_blobs: List[str] = []
    files: List[Path] = []
    for kind, payload in parts:
        if kind == "text":
            user_text_blobs.append(payload.strip())
        elif kind in ("image", "audio"):
            p = Path(payload)
            if p.exists():
                files.append(p)
            else:
                user_text_blobs.append(f"[missing {kind} at {payload}]")
    merged = "\n".join(user_text_blobs) if user_text_blobs else \
             "(no text — see attached files)"
    schema = {
        "type": "object",
        "properties": {
            "intent":  {"type": "string"},
            "summary": {"type": "string"},
            "fragments": {
                "type": "object",
                "properties": {
                    "geometry": {"type": "string", "nullable": True},
                    "material": {"type": "string", "nullable": True},
                    "loading":  {"type": "string", "nullable": True},
                    "fracture": {"type": "string", "nullable": True},
                    "output":   {"type": "string", "nullable": True},
                    "units":    {"type": "string", "nullable": True},
                },
            },
        },
        "required": ["intent", "summary", "fragments"],
    }
    # ``thinking_budget=0`` disables Gemini 2.5 Flash's reasoning mode
    # entirely — this agent does pure structured extraction (text →
    # one shallow JSON fragment).  In one observed P5 run the receptionist
    # spent 16,084 output tokens (and $0.04) thinking before bailing on
    # MAX_TOKENS; with thinking off it returns a clean 100-token JSON in
    # one shot.  Don't lose this knob.
    return llm().complete_json(SYS, merged, files=files, schema=schema,
                                temperature=0.05, max_output_tokens=2048,
                                thinking_budget=0, model=FAST_MODEL)
