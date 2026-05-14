"""Hierarchical RAG over the user's modular DOLFINx codes.

Three indices, each queried at a different pipeline stage:

    Skeleton  — whole files in ``modular/examples/*.py``.  Metadata-rich
                (variant name, dim, plane, constitutive, mode).  Queried
                by the Strategist to confirm / nudge the variant choice
                with a similarity score.

    Snippet   — function-level chunks of ``modular/``.  Queried by the
                Synthesizer / Debugger to surface concrete call patterns
                (BC-pin, mesh-audit print, ramp-proxy, AMR setup,
                XDMFWriter wiring).

    ErrorFix  — bootstrapped from past Debugger successes.  Queried with
                a stack-trace signature so the Debugger gets prior
                fixes as concrete examples.

The retrieval is *always* over the user's verified modular content; the
LLM never authors physics from these snippets — it only picks which
snippets to compose.

Embedder:  Gemini's ``gemini-embedding-001`` via ``llm().embed()`` — no
new external dependency.  Cached on disk under
``.fracture_agent_cache/rag/<index>.json`` so re-indexing only happens when the
modular tree changes.

Vector store:  numpy cosine similarity.  No FAISS for v1 — the corpus is
small (~10 example files, <500 function chunks).  We can swap in FAISS
later without changing the retriever's external API.
"""
from .index import (build_all_indices, build_error_fix_index,
                     build_skeleton_index, build_snippet_index, index_path)
from .retrieval import (Retrieval, retrieve_error_fix, retrieve_skeleton,
                         retrieve_snippets, sigma_gate)

__all__ = [
    "Retrieval",
    "build_all_indices", "build_skeleton_index", "build_snippet_index",
    "build_error_fix_index", "index_path",
    "retrieve_skeleton", "retrieve_snippets", "retrieve_error_fix",
    "sigma_gate",
]
