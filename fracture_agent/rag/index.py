"""Indexers — turn modular/ source into embedded retrieval indices.

Each index is persisted as a single JSON file under
``.fracture_agent_cache/rag/<index>.json`` with shape::

    {
      "model": "gemini-embedding-001",
      "version": 1,
      "items": [
        {"id": "...", "text": "...", "metadata": {...}, "embedding": [...]},
        ...
      ]
    }

Idempotent: a build that finds the same source mtimes / hashes
short-circuits and returns the cached index.

We keep the AST chunker simple — ``ast.FunctionDef`` and
``ast.AsyncFunctionDef`` (plus module-level `if __name__ ...` blocks) ARE
the natural function-level units of ``modular/``.  No need for the
``astchunk`` dep at v1.
"""
from __future__ import annotations
import ast
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..config import EMBED_MODEL, MODULAR, PKG_ROOT
from ..events import DECISION, emit
from ..llm import llm


CACHE_ROOT = PKG_ROOT.parent / ".fracture_agent_cache" / "rag"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Path helpers.
# ---------------------------------------------------------------------------
def index_path(name: str) -> Path:
    return CACHE_ROOT / f"{name}.json"


def _hash_corpus(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            h.update(p.read_bytes())
            h.update(str(p.relative_to(MODULAR.parent)).encode("utf-8"))
        except Exception:
            continue
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Generic embed-and-cache helper.
# ---------------------------------------------------------------------------
@dataclass
class IndexBuilder:
    name: str
    model: str = EMBED_MODEL
    items: List[Dict[str, Any]] = field(default_factory=list)
    corpus_hash: str = ""

    def add(self, *, id_: str, text: str, metadata: Dict[str, Any]) -> None:
        self.items.append({"id": id_, "text": text, "metadata": metadata,
                            "embedding": None})

    def embed(self, *, retag_only: bool = False) -> None:
        """Embed every item.  If ``retag_only``, only items with a None
        embedding are sent to the API (resume from a partial build)."""
        for i, it in enumerate(self.items):
            if retag_only and it.get("embedding") is not None:
                continue
            it["embedding"] = llm().embed(it["text"], model=self.model)

    def save(self) -> Path:
        out = index_path(self.name)
        payload = {
            "model": self.model,
            "version": 1,
            "corpus_hash": self.corpus_hash,
            "built_at": time.time(),
            "items": self.items,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out


def _maybe_load_cached(name: str, corpus_hash: str) -> Optional[List[Dict[str, Any]]]:
    p = index_path(name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("corpus_hash") != corpus_hash:
        return None
    items = data.get("items") or []
    if not all(isinstance(it.get("embedding"), list) and it["embedding"]
               for it in items):
        return None
    return items


# ---------------------------------------------------------------------------
# Index 1 — Skeleton.  ``modular/examples/*.py``, whole-file granularity.
# Metadata is rich because it drives the σ-gate decision.
# ---------------------------------------------------------------------------
_SKELETON_NAME = "skeleton"


def _skeleton_metadata(path: Path) -> Dict[str, Any]:
    """Heuristic: read the file, look for the variant + builder + solver
    references to tag the skeleton."""
    src = path.read_text(encoding="utf-8")
    md: Dict[str, Any] = {"path": str(path.relative_to(MODULAR.parent)),
                           "stem": path.stem}
    # Variant from the filename — modular/examples/<variant>.py.
    md["variant"] = path.stem
    # Solver call.
    for s in ("run_quasistatic", "run_dynamic",
              "run_finite_elasticity", "run_ductile"):
        if re.search(rf"\b{s}\(", src):
            md["solver"] = s
            break
    # Mesh builder.
    for b in ("make_notched_plate_2d", "make_notched_plate_3d",
              "make_slant_plate_2d", "make_dogbone_2d", "make_dogbone_3d"):
        if re.search(rf"\b{b}\(", src):
            md["mesh_builder"] = b
            break
    # Material catalog name.
    m = re.search(r"load_material\(\s*['\"]([^'\"]+)['\"]", src)
    if m:
        md["material"] = m.group(1)
    # Quick dim guess from the variant name.
    md["dim"] = 3 if "3d" in path.stem else 2
    md["plane"] = ("plane_strain" if path.stem.endswith("pe")
                    else "plane_stress" if path.stem.endswith("ps") else "na")
    md["constitutive"] = (
        "j2_plasticity" if "ductile" in path.stem else
        "lopez_pamies" if "finite" in path.stem else
        "linear_elastic")
    return md


def _skeleton_text(path: Path, md: Dict[str, Any]) -> str:
    """The text we embed: variant metadata header + the file content
    truncated to ~6k chars (Gemini's free-tier embedding limit is 2048
    tokens, well above this)."""
    src = path.read_text(encoding="utf-8")
    head = (
        f"# Variant: {md.get('variant')}\n"
        f"# Solver: {md.get('solver', '?')}\n"
        f"# Mesh builder: {md.get('mesh_builder', '?')}\n"
        f"# Material default: {md.get('material', '?')}\n"
        f"# Dim/plane/constitutive: "
        f"{md.get('dim')}D / {md.get('plane')} / {md.get('constitutive')}\n\n"
    )
    return head + src[:6000]


def build_skeleton_index() -> Path:
    examples_dir = MODULAR / "examples"
    files = sorted(examples_dir.glob("*.py"))
    files = [f for f in files if not f.name.startswith("_")]
    corpus_hash = _hash_corpus(files)

    cached = _maybe_load_cached(_SKELETON_NAME, corpus_hash)
    if cached is not None:
        emit(DECISION, f"RAG: skeleton index up to date ({len(cached)} items)")
        return index_path(_SKELETON_NAME)

    emit(DECISION, f"RAG: building skeleton index ({len(files)} files)...")
    b = IndexBuilder(name=_SKELETON_NAME)
    b.corpus_hash = corpus_hash
    for f in files:
        md = _skeleton_metadata(f)
        b.add(id_=f.stem, text=_skeleton_text(f, md), metadata=md)
    b.embed()
    return b.save()


# ---------------------------------------------------------------------------
# Index 2 — Snippet.  Function-level AST chunks across modular/.
# ---------------------------------------------------------------------------
_SNIPPET_NAME = "snippet"


def _ast_function_chunks(src: str, *, path: Path,
                          max_chars: int = 4000) -> List[Dict[str, Any]]:
    """Extract one chunk per top-level / class-level function, plus
    each ``if __name__ == '__main__'`` block.  Returns a list of
    ``{"id", "text", "metadata"}`` dicts ready for the IndexBuilder."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    rel = path.relative_to(MODULAR.parent)
    out: List[Dict[str, Any]] = []
    src_lines = src.splitlines(keepends=True)

    def _slice(node) -> str:
        a = max(0, getattr(node, "lineno", 1) - 1)
        b = getattr(node, "end_lineno", a + 1)
        snippet = "".join(src_lines[a:b])
        return snippet[:max_chars]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            doc = ast.get_docstring(node) or ""
            chunk = _slice(node)
            out.append({
                "id": f"{rel}:{name}",
                "text": (
                    f"# File: {rel}\n# Function: {name}\n"
                    f"# Docstring: {doc[:300]}\n\n{chunk}"
                ),
                "metadata": {
                    "path": str(rel), "kind": "function", "name": name,
                    "lineno": node.lineno, "end_lineno": node.end_lineno,
                },
            })
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = _slice(sub)
                    qual = f"{node.name}.{sub.name}"
                    doc = ast.get_docstring(sub) or ""
                    out.append({
                        "id": f"{rel}:{qual}",
                        "text": (
                            f"# File: {rel}\n# Method: {qual}\n"
                            f"# Docstring: {doc[:300]}\n\n{chunk}"
                        ),
                        "metadata": {
                            "path": str(rel), "kind": "method",
                            "name": qual,
                            "lineno": sub.lineno, "end_lineno": sub.end_lineno,
                        },
                    })
    return out


def build_snippet_index(*, exclude_examples: bool = False) -> Path:
    """Walk modular/, extract functions, embed them.

    ``exclude_examples`` skips ``modular/examples/`` (those are already in
    the Skeleton index at file granularity).
    """
    files: List[Path] = []
    for p in sorted(MODULAR.rglob("*.py")):
        if p.name.startswith("_"):
            continue
        if exclude_examples and "examples" in p.parts:
            continue
        files.append(p)
    corpus_hash = _hash_corpus(files)

    cached = _maybe_load_cached(_SNIPPET_NAME, corpus_hash)
    if cached is not None:
        emit(DECISION, f"RAG: snippet index up to date ({len(cached)} items)")
        return index_path(_SNIPPET_NAME)

    chunks: List[Dict[str, Any]] = []
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
        except Exception:
            continue
        chunks.extend(_ast_function_chunks(src, path=f))
    emit(DECISION,
         f"RAG: building snippet index ({len(chunks)} chunks "
         f"from {len(files)} files)...")
    b = IndexBuilder(name=_SNIPPET_NAME)
    b.corpus_hash = corpus_hash
    for c in chunks:
        b.add(id_=c["id"], text=c["text"], metadata=c["metadata"])
    b.embed()
    return b.save()


# ---------------------------------------------------------------------------
# Index 3 — ErrorFix.  Self-populating from past Debugger successes.
# ---------------------------------------------------------------------------
_ERRFIX_NAME = "error_fix"


def build_error_fix_index() -> Path:
    """Build / refresh the error-fix memory index.

    Items are populated via ``record_error_fix`` (called from
    ``debugger.py`` on a successful patch).  An empty index is fine —
    Index 3 only earns its keep after the agent has been run a few times.
    """
    raw = CACHE_ROOT / "error_fix_raw.json"
    if not raw.exists():
        # Initialise an empty payload so retrieval doesn't blow up.
        empty = {"model": EMBED_MODEL, "version": 1, "corpus_hash": "",
                  "built_at": time.time(), "items": []}
        index_path(_ERRFIX_NAME).write_text(json.dumps(empty, indent=2),
                                               encoding="utf-8")
        return index_path(_ERRFIX_NAME)
    items = json.loads(raw.read_text(encoding="utf-8"))
    corpus_hash = hashlib.sha256(raw.read_bytes()).hexdigest()[:16]
    cached = _maybe_load_cached(_ERRFIX_NAME, corpus_hash)
    if cached is not None:
        return index_path(_ERRFIX_NAME)
    b = IndexBuilder(name=_ERRFIX_NAME)
    b.corpus_hash = corpus_hash
    for it in items:
        b.add(id_=it["id"],
              text=it["error_signature"] + "\n\n" + it.get("offending_snippet", ""),
              metadata=it)
    b.embed()
    return b.save()


def record_error_fix(error_signature: str,
                      offending_snippet: str,
                      applied_fix: str,
                      outcome: str) -> None:
    """Append one fix record so the next ``build_error_fix_index`` picks
    it up.  Uses a content hash as the id."""
    raw = CACHE_ROOT / "error_fix_raw.json"
    items = []
    if raw.exists():
        try:
            items = json.loads(raw.read_text(encoding="utf-8"))
        except Exception:
            items = []
    rid = hashlib.sha1(
        (error_signature + "\n" + applied_fix).encode("utf-8")).hexdigest()[:12]
    items.append({
        "id": rid,
        "error_signature": error_signature,
        "offending_snippet": offending_snippet[:1500],
        "applied_fix": applied_fix[:1500],
        "outcome": outcome,
        "ts": time.time(),
    })
    raw.write_text(json.dumps(items, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Convenience — build every index.
# ---------------------------------------------------------------------------
def build_all_indices() -> Dict[str, Path]:
    """Build (or refresh) skeleton + snippet + error-fix indices.  Safe
    to re-call: each individual builder short-circuits when its corpus
    hash matches the cached version."""
    return {
        "skeleton":  build_skeleton_index(),
        "snippet":   build_snippet_index(exclude_examples=True),
        "error_fix": build_error_fix_index(),
    }
