"""Tests for ``fracture_agent.telemetry``.

Run with::

    python -m fracture_agent._tests.test_telemetry

(no pytest dep — pure stdlib unittest, mirroring the rest of the test
folder's script-based style).
"""
from __future__ import annotations
import csv
import tempfile
import unittest
from pathlib import Path

from fracture_agent.telemetry import (GEMINI_PRICES, IterationCounters, LLMCallRecord,
                                SessionTelemetry, _calc_cost_usd, bump,
                                current_agent, get_active, llm_agent,
                                record_llm_call, set_active)


class TestPricing(unittest.TestCase):
    def test_pro_price(self):
        # 1M input + 1M output of pro = $1.25 + $5.00 = $6.25
        self.assertAlmostEqual(
            _calc_cost_usd("gemini-2.5-pro", 1_000_000, 1_000_000), 6.25, places=4)

    def test_flash_price(self):
        # 100k input + 50k output of flash = 0.03 + 0.125 = 0.155
        self.assertAlmostEqual(
            _calc_cost_usd("gemini-2.5-flash", 100_000, 50_000), 0.155, places=4)

    def test_unknown_model_falls_back_to_pro(self):
        a = _calc_cost_usd("unknown-model-xyz", 1000, 1000)
        b = _calc_cost_usd("gemini-2.5-pro",    1000, 1000)
        self.assertEqual(a, b)


class TestAgentContext(unittest.TestCase):
    def setUp(self):
        # Clean stack
        from fracture_agent import telemetry as t
        t._local.stack = []

    def test_default_unknown(self):
        self.assertEqual(current_agent(), "unknown")

    def test_single_block(self):
        with llm_agent("architect"):
            self.assertEqual(current_agent(), "architect")
        self.assertEqual(current_agent(), "unknown")

    def test_nested_blocks(self):
        with llm_agent("synthesizer"):
            self.assertEqual(current_agent(), "synthesizer")
            with llm_agent("mesh_llm"):
                self.assertEqual(current_agent(), "mesh_llm")
            self.assertEqual(current_agent(), "synthesizer")
        self.assertEqual(current_agent(), "unknown")


class TestActiveSink(unittest.TestCase):
    def setUp(self):
        set_active(None)

    def tearDown(self):
        set_active(None)

    def test_no_op_without_active(self):
        # Should not raise; just discards.
        record_llm_call(model="gemini-2.5-flash",
                        prompt_tokens=10, output_tokens=20)
        bump("architect_rounds")
        self.assertIsNone(get_active())

    def test_records_with_active(self):
        t = SessionTelemetry()
        set_active(t)
        with llm_agent("architect"):
            record_llm_call(model="gemini-2.5-flash",
                            prompt_tokens=100, output_tokens=200,
                            latency_ms=345.6)
        self.assertEqual(len(t.calls), 1)
        c = t.calls[0]
        self.assertEqual(c.agent, "architect")
        self.assertEqual(c.prompt_tokens, 100)
        self.assertEqual(c.output_tokens, 200)
        self.assertAlmostEqual(c.latency_ms, 345.6, places=3)
        # cost = 100/1M * 0.30 + 200/1M * 2.50 = 0.00003 + 0.0005 = 0.00053
        self.assertAlmostEqual(c.cost_usd, 0.00053, places=6)

    def test_iteration_counters(self):
        t = SessionTelemetry()
        set_active(t)
        bump("architect_rounds", 3)
        bump("debugger_attempts")
        bump("mesh_rescales", 2)
        self.assertEqual(t.iters.architect_rounds, 3)
        self.assertEqual(t.iters.debugger_attempts, 1)
        self.assertEqual(t.iters.mesh_rescales, 2)
        self.assertEqual(t.iters.reflect_revise_cycles, 0)

    def test_unknown_counter_is_no_op(self):
        t = SessionTelemetry()
        set_active(t)
        bump("not_a_real_counter")  # must not raise
        self.assertEqual(t.iters.architect_rounds, 0)


class TestAggregations(unittest.TestCase):
    def _make(self):
        t = SessionTelemetry()
        set_active(t)
        with llm_agent("architect"):
            record_llm_call(model="gemini-2.5-flash",
                            prompt_tokens=1000, output_tokens=500)
            record_llm_call(model="gemini-2.5-flash",
                            prompt_tokens=800, output_tokens=300)
        with llm_agent("advisor"):
            record_llm_call(model="gemini-2.5-pro",
                            prompt_tokens=2000, output_tokens=1500,
                            thinking_tokens=400)
        return t

    def setUp(self):
        set_active(None)

    def tearDown(self):
        set_active(None)

    def test_totals(self):
        t = self._make()
        tot = t.totals()
        self.assertEqual(tot["n_llm_calls"], 3)
        self.assertEqual(tot["prompt_tokens"], 3800)
        self.assertEqual(tot["output_tokens"], 2300)
        self.assertEqual(tot["thinking_tokens"], 400)
        self.assertEqual(tot["total_tokens"], 3800 + 2300 + 400)
        self.assertGreater(tot["cost_usd"], 0)

    def test_by_agent(self):
        t = self._make()
        agg = t.by_agent()
        self.assertIn("architect", agg)
        self.assertIn("advisor", agg)
        self.assertEqual(agg["architect"]["n_calls"], 2)
        self.assertEqual(agg["architect"]["prompt_tokens"], 1800)
        self.assertEqual(agg["advisor"]["n_calls"], 1)


class TestCostCSV(unittest.TestCase):
    def test_writes_well_formed_csv(self):
        t = SessionTelemetry()
        set_active(t)
        with llm_agent("architect"):
            record_llm_call(model="gemini-2.5-flash",
                            prompt_tokens=500, output_tokens=200,
                            latency_ms=120.5, finish_reason="STOP")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cost.csv"
            t.write_cost_csv(p)
            with p.open("r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows[0][0], "ts")  # header
        self.assertEqual(len(rows), 2)       # header + 1 row
        # Row content sanity
        row = rows[1]
        self.assertIn("architect", row)
        self.assertIn("gemini-2.5-flash", row)
        set_active(None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
