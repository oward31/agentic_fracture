"""Small on-disk caches to avoid repeat LLM calls across sessions.

Currently caches:

* **Material handbook lookups** — keyed by (display_name lowercased,
  problem_type).  Identical prompts return identical handbook values, so
  caching saves a full Gemini-Pro-with-search call (~15-30 s per hit).

Cache lives under ``agent_v2/.fracture_agent_cache/`` so it's outside the source
tree and easy to wipe manually.  Thread-safety is not addressed — the
orchestrator is single-threaded.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import REPO_ROOT


CACHE_DIR = REPO_ROOT / ".fracture_agent_cache"
CACHE_DIR.mkdir(exist_ok=True)
MAT_CACHE_FILE = CACHE_DIR / "materials.json"


def _load() -> Dict[str, Any]:
    if MAT_CACHE_FILE.exists():
        try:
            return json.loads(MAT_CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save(db: Dict[str, Any]) -> None:
    MAT_CACHE_FILE.write_text(json.dumps(db, indent=2, sort_keys=True),
                              encoding="utf-8")


def _key(display_name: str, problem_type: str) -> str:
    return f"{(display_name or '').strip().lower()}|{problem_type or ''}"


def get_material(display_name: str, problem_type: str) -> Optional[Dict[str, Any]]:
    if not display_name:
        return None
    return _load().get(_key(display_name, problem_type))


def put_material(display_name: str, problem_type: str,
                 data: Dict[str, Any]) -> None:
    if not display_name or not data:
        return
    db = _load()
    db[_key(display_name, problem_type)] = data
    _save(db)
