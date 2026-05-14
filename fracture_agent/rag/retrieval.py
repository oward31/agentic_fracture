"""Cosine-similarity retrieval over the cached embedding indices.

Pure numpy — no FAISS for v1.  The corpora are small (≤ 500 chunks total
across the three indices); a single vectorised cosine over a stacked
(N, D) matrix is well under 1 ms per query on commodity hardware.

Index handles are cached at module level so repeated retrievals don't
re-read the JSON.
"""
from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import EMBED_MODEL
from ..llm import llm
from .index import (build_error_fix_index, build_skeleton_index,
                     build_snippet_index, index_path)


# ---------------------------------------------------------------------------
# Lazy index cache: name -> (mtime, items, embedding_matrix)
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]], np.ndarray]] = {}


def _load_index(name: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    p = index_path(name)
    if not p.exists():
        return [], np.zeros((0, 1))
    mtime = p.stat().st_mtime
    if name in _CACHE and _CACHE[name][0] == mtime:
        return _CACHE[name][1], _CACHE[name][2]
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("items") or []
    if not items:
        emb = np.zeros((0, 1))
    else:
        emb = np.array([it["embedding"] for it in items], dtype=np.float32)
        # L2-normalise once so cosine = dot.
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb = emb / norms
    _CACHE[name] = (mtime, items, emb)
    return items, emb


def _embed_query(text: str) -> np.ndarray:
    v = np.array(llm().embed(text, model=EMBED_MODEL), dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _topk(items: List[Dict[str, Any]], emb: np.ndarray,
           query: np.ndarray, k: int) -> List[Dict[str, Any]]:
    if not items or emb.shape[0] == 0:
        return []
    if emb.shape[1] != query.shape[0]:
        # Dimension mismatch (e.g. cached index from an older model).  Skip.
        return []
    sims = emb @ query                 # (N,)
    order = np.argsort(-sims)[:k]
    out = []
    for idx in order:
        it = items[int(idx)].copy()
        it["score"] = float(sims[int(idx)])
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# Public retrieval surface.
# ---------------------------------------------------------------------------
@dataclass
class Retrieval:
    """One retrieval result wrapping the matched item plus its score."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

    @classmethod
    def from_item(cls, item: Dict[str, Any]) -> "Retrieval":
        return cls(id=item["id"], score=float(item.get("score", 0.0)),
                    text=item.get("text", ""),
                    metadata=item.get("metadata", {}) or {})


def retrieve_skeleton(query: str, *, k: int = 3,
                       auto_build: bool = True) -> List[Retrieval]:
    """Top-K matches against the Skeleton index (whole-file granularity).

    If the index doesn't exist yet, ``auto_build=True`` (default) calls
    ``build_skeleton_index()`` first; pass ``auto_build=False`` from the
    UI to avoid the embedding-call latency.
    """
    if auto_build and not index_path("skeleton").exists():
        build_skeleton_index()
    items, emb = _load_index("skeleton")
    q = _embed_query(query)
    return [Retrieval.from_item(it) for it in _topk(items, emb, q, k)]


def retrieve_snippets(query: str, *, k: int = 5,
                       auto_build: bool = True) -> List[Retrieval]:
    """Top-K matches against the Snippet index (function granularity)."""
    if auto_build and not index_path("snippet").exists():
        build_snippet_index(exclude_examples=True)
    items, emb = _load_index("snippet")
    q = _embed_query(query)
    return [Retrieval.from_item(it) for it in _topk(items, emb, q, k)]


def retrieve_error_fix(error_signature: str, *, k: int = 3,
                        auto_build: bool = True) -> List[Retrieval]:
    """Top-K matches against the bootstrapped Error-Fix memory."""
    if auto_build and not index_path("error_fix").exists():
        build_error_fix_index()
    items, emb = _load_index("error_fix")
    if not items:
        return []
    q = _embed_query(error_signature)
    return [Retrieval.from_item(it) for it in _topk(items, emb, q, k)]


def sigma_gate(matches: List[Retrieval],
                threshold: float = 0.85) -> Tuple[Optional[Retrieval], List[Retrieval]]:
    """Return ``(top_match_above_threshold or None, all_matches)``.

    Mirrors FeaGPT's σ ≥ 0.85 knowledge-augmented gate (arXiv 2510.21993,
    §II.B).  When the top match clears the bar, the caller may enter
    skeleton-mutation mode; otherwise the existing fallback path stays
    in charge.

    The threshold itself is a tunable: fracture_agent's eventual paper sweeps
    it from 0.7 to 0.95 and reports the success-rate curve as the
    ablation requested by reviewers (FeaGPT introduced σ but never
    ablated it).
    """
    if not matches:
        return None, []
    top = matches[0]
    return (top if top.score >= threshold else None), matches
