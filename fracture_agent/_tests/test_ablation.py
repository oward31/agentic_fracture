"""Smoke tests for the ablation harness (no LLM, no WSL).

The B1 path can't be unit-tested without a real LLM call.  We test:
  * AblationLevel enum is wired correctly
  * The B1 prompt assembler produces non-empty output containing the
    catalog and material list
  * The benchmark runner can list and load the bundled suite JSONs

Run with::

    python -m fracture_agent._tests.test_ablation
"""
from __future__ import annotations
import json
import unittest
from pathlib import Path

from fracture_agent.ablation import (AblationLevel, B1_SYSTEM, _build_b1_user_prompt)
from fracture_agent.benchmark.runner import list_suite, load_problem


class TestAblationLevel(unittest.TestCase):
    def test_levels_present(self):
        self.assertEqual(AblationLevel.B1_ONE_SHOT.value, 1)
        self.assertEqual(AblationLevel.B2_INSPECTOR.value, 2)
        self.assertEqual(AblationLevel.B3_FULL_INNER.value, 3)
        self.assertEqual(AblationLevel.B4_FULL_PFAGENT.value, 4)

    def test_int_to_level(self):
        self.assertEqual(AblationLevel(1), AblationLevel.B1_ONE_SHOT)
        self.assertEqual(AblationLevel(4), AblationLevel.B4_FULL_PFAGENT)


class TestB1PromptBuilder(unittest.TestCase):
    def test_includes_catalog(self):
        out = _build_b1_user_prompt("a steel plate")
        self.assertIn("MODULAR CATALOG", out)
        # Each variant shows up by name
        self.assertIn("linear_elastic_2d_pe", out)
        self.assertIn("ductile_2d_pe", out)
        self.assertIn("finite_elastic_2d_ps", out)

    def test_includes_user_prompt(self):
        out = _build_b1_user_prompt("a steel plate xyz123")
        self.assertIn("xyz123", out)

    def test_includes_api_block(self):
        out = _build_b1_user_prompt("foo")
        self.assertIn("from modular.problems", out)
        self.assertIn("from modular.solvers", out)
        self.assertIn("make_custom_gmsh", out)

    def test_system_prompt_forbids_authoring(self):
        self.assertIn("Do NOT author", B1_SYSTEM)
        self.assertIn("from dolfin", B1_SYSTEM)


class TestSuiteLoader(unittest.TestCase):
    def test_tier1_suite_loads(self):
        files = list_suite("tier1")
        self.assertGreaterEqual(len(files), 1)
        for p in files:
            data = load_problem(p)
            # Required fields
            self.assertIn("id", data)
            self.assertIn("prompts", data)
            self.assertIn("tier1_full", data["prompts"])

    def test_unknown_suite_raises(self):
        with self.assertRaises(FileNotFoundError):
            list_suite("does_not_exist_xyz")

    def test_each_problem_has_three_tiers(self):
        for p in list_suite("tier1"):
            data = load_problem(p)
            self.assertIn("tier1_full", data["prompts"])
            self.assertIn("tier2_partial", data["prompts"])
            self.assertIn("tier3_figure_only", data["prompts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
