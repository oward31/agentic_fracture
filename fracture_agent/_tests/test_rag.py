"""Tests for ``fracture_agent.rag`` — index assembly, AST chunking, σ-gate.

These tests do NOT call the embedder; we test:
  * the AST chunker produces sensible function-level chunks
  * the IndexBuilder serialises payloads correctly
  * sigma_gate returns top match iff above threshold
  * retrieval handles empty / missing indices gracefully

Run with::

    python -m fracture_agent._tests.test_rag
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from fracture_agent.rag.index import (IndexBuilder, _ast_function_chunks, index_path)
from fracture_agent.rag.retrieval import Retrieval, sigma_gate


SAMPLE_SRC = '''\
"""Module docstring."""
import numpy as np


def free_function(a, b):
    """Add two things."""
    return a + b


class Foo:
    """A class."""

    def method_one(self, x):
        """Compute thing."""
        return x * 2

    def method_two(self):
        return 42


def _private_helper():
    return None
'''


class TestASTChunker(unittest.TestCase):
    def test_extracts_functions_and_methods(self):
        with tempfile.TemporaryDirectory() as d:
            from fracture_agent.config import MODULAR
            # Use a temp path that pretends to be inside MODULAR.parent so
            # `relative_to` works.  Easiest: write into MODULAR/_test_tmp.
            # But we don't want to touch the real package — instead, mock
            # via a path inside the agent_v2 root.
            tmp = MODULAR.parent / "_rag_test_tmp.py"
            tmp.write_text(SAMPLE_SRC, encoding="utf-8")
            try:
                chunks = _ast_function_chunks(SAMPLE_SRC, path=tmp)
            finally:
                tmp.unlink(missing_ok=True)
        names = [c["metadata"]["name"] for c in chunks]
        self.assertIn("free_function", names)
        self.assertIn("Foo.method_one", names)
        self.assertIn("Foo.method_two", names)
        self.assertIn("_private_helper", names)

    def test_chunks_carry_path_metadata(self):
        from fracture_agent.config import MODULAR
        tmp = MODULAR.parent / "_rag_test_tmp2.py"
        tmp.write_text(SAMPLE_SRC, encoding="utf-8")
        try:
            chunks = _ast_function_chunks(SAMPLE_SRC, path=tmp)
        finally:
            tmp.unlink(missing_ok=True)
        for c in chunks:
            self.assertIn("path", c["metadata"])
            self.assertIn("kind", c["metadata"])
            self.assertIn("lineno", c["metadata"])

    def test_empty_or_invalid_source(self):
        from fracture_agent.config import MODULAR
        tmp = MODULAR.parent / "_rag_test_tmp3.py"
        tmp.write_text("not = valid : python", encoding="utf-8")
        try:
            chunks = _ast_function_chunks("not = valid : python", path=tmp)
        finally:
            tmp.unlink(missing_ok=True)
        self.assertEqual(chunks, [])


class TestSigmaGate(unittest.TestCase):
    def _r(self, id_, score):
        return Retrieval(id=id_, score=score, text="", metadata={})

    def test_top_above_threshold(self):
        rs = [self._r("a", 0.9), self._r("b", 0.5)]
        top, all_ = sigma_gate(rs, threshold=0.85)
        self.assertIsNotNone(top)
        self.assertEqual(top.id, "a")
        self.assertEqual(len(all_), 2)

    def test_top_below_threshold_returns_none(self):
        rs = [self._r("a", 0.7), self._r("b", 0.5)]
        top, _ = sigma_gate(rs, threshold=0.85)
        self.assertIsNone(top)

    def test_empty_returns_none(self):
        top, all_ = sigma_gate([])
        self.assertIsNone(top)
        self.assertEqual(all_, [])

    def test_threshold_at_exact_score(self):
        rs = [self._r("a", 0.85)]
        top, _ = sigma_gate(rs, threshold=0.85)
        self.assertIsNotNone(top)


class TestIndexBuilderSerialisation(unittest.TestCase):
    def test_save_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            from fracture_agent.rag import index as idx_mod
            old_root = idx_mod.CACHE_ROOT
            try:
                idx_mod.CACHE_ROOT = Path(d)
                b = IndexBuilder(name="x")
                b.corpus_hash = "abc123"
                b.add(id_="i1", text="hello",
                       metadata={"path": "x.py", "name": "f"})
                # Inject a deterministic embedding so we don't call the API
                b.items[0]["embedding"] = [0.1, 0.2, 0.3]
                p = b.save()
                self.assertTrue(p.exists())
                data = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(data["model"], idx_mod.EMBED_MODEL)
                self.assertEqual(data["corpus_hash"], "abc123")
                self.assertEqual(len(data["items"]), 1)
                self.assertEqual(data["items"][0]["embedding"], [0.1, 0.2, 0.3])
            finally:
                idx_mod.CACHE_ROOT = old_root


class TestEmptyRetrieval(unittest.TestCase):
    """Retrieval against a missing / empty index returns []."""
    def test_missing_index_yields_empty(self):
        from fracture_agent.rag.retrieval import _load_index, _CACHE
        from fracture_agent.rag import index as idx_mod
        with tempfile.TemporaryDirectory() as d:
            old_root = idx_mod.CACHE_ROOT
            try:
                idx_mod.CACHE_ROOT = Path(d)
                _CACHE.clear()
                items, emb = _load_index("not_a_real_index_name")
                self.assertEqual(items, [])
                self.assertEqual(emb.shape[0], 0)
            finally:
                idx_mod.CACHE_ROOT = old_root


if __name__ == "__main__":
    unittest.main(verbosity=2)
