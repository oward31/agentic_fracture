"""Tests for ``fracture_agent.health`` and ``fracture_agent.agents.reviser``.

Run with::

    python -m fracture_agent._tests.test_health
"""
from __future__ import annotations
import unittest
from typing import Dict, List

from fracture_agent.health import (ACCEPT_THRESHOLD, WARN_THRESHOLD, HealthReport,
                              compute_health)


def _good_log(n: int = 100, T: float = 1.0,
               cracks_at: int = 60, dt: float = 0.01) -> Dict[str, List[float]]:
    """Synthesize a healthy run: monotone min_z dropping past 0.3 mid-way,
    bounded residuals, sub-max stagger counts, peak Fy in the middle."""
    cols = {k: [] for k in ("step", "time", "dt", "stag_iters",
                              "u_res", "z_res", "min_z", "disp", "Fy")}
    for i in range(n):
        cols["step"].append(float(i + 1))
        cols["time"].append((i + 1) * (T / n))
        cols["dt"].append(dt)
        cols["stag_iters"].append(3.0)              # well below max_stag
        cols["u_res"].append(1e-6)
        cols["z_res"].append(1e-6)
        # min_z: 1.0 → 0.05 once we pass cracks_at, then plateau
        if i < cracks_at:
            cols["min_z"].append(1.0 - 0.7 * (i / max(cracks_at, 1)))
        else:
            cols["min_z"].append(max(0.05, 0.3 - 0.5 * ((i - cracks_at) / n)))
        # disp linear, Fy rises then falls past peak (cracks_at + 5)
        cols["disp"].append((i + 1) * (1.0 / n))
        peak_at = cracks_at + 5
        if i <= peak_at:
            cols["Fy"].append(100.0 * (i + 1) / max(peak_at, 1))
        else:
            cols["Fy"].append(max(20.0, 100.0 - 100.0 * (i - peak_at) / n))
    return cols


class TestHealthHappyPath(unittest.TestCase):
    def test_clean_run_scores_high(self):
        cols = _good_log()
        h = compute_health(
            cols,
            returncode=0, wall_time_s=20.0,
            fracture_enabled=True, T_total=1.0, dt_first=0.01,
            eps=0.5, h_min=0.0625, max_stag=20)
        self.assertGreaterEqual(h.total, ACCEPT_THRESHOLD,
                                 f"Expected ≥{ACCEPT_THRESHOLD}, got {h.total}: "
                                 f"flags={h.flags}")
        self.assertEqual(h.verdict, "accept")
        self.assertEqual(h.min_z_violations, 0)
        # h_min/eps = 0.0625/0.5 = 0.125 ≤ 0.25 → full mesh-indep credit
        self.assertEqual(h.mesh_independence.points,
                         h.mesh_independence.max_points)


class TestHealthFailureModes(unittest.TestCase):
    def test_nonzero_rc(self):
        h = compute_health(_good_log(),
                            returncode=1, wall_time_s=20.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.0625, max_stag=20)
        self.assertIn("nonzero_rc", h.flags)
        # 10 pts gone from integrity; total still high but not full integ.
        self.assertLess(h.integrity.points, h.integrity.max_points)

    def test_partial_run_loses_integrity(self):
        cols = _good_log()
        # Truncate to half — last time becomes ~0.5, T_total stays 1.0.
        for k in cols:
            cols[k] = cols[k][:50]
        h = compute_health(cols,
                            returncode=0, wall_time_s=20.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.0625, max_stag=20)
        self.assertIn("partial_run", h.flags)
        self.assertLess(h.pct_target_time, 1.0)

    def test_min_z_violation_flags(self):
        cols = _good_log()
        # Inject damage healing: mid-run, bump min_z back up.
        cols["min_z"][50] = 0.9
        h = compute_health(cols,
                            returncode=0, wall_time_s=20.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.0625, max_stag=20)
        self.assertIn("irreversibility_violation", h.flags)
        self.assertGreaterEqual(h.min_z_violations, 1)

    def test_stagger_saturated(self):
        cols = _good_log()
        # Make every step hit max_stag.
        cols["stag_iters"] = [20.0] * len(cols["stag_iters"])
        h = compute_health(cols,
                            returncode=0, wall_time_s=20.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.0625, max_stag=20)
        self.assertIn("stagger_saturated", h.flags)

    def test_dt_floor_saturated(self):
        cols = _good_log()
        cols["dt"] = [0.001] * len(cols["dt"])  # exactly at floor 0.01/10
        h = compute_health(cols,
                            returncode=0, wall_time_s=20.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.0625, max_stag=20)
        self.assertIn("dt_floor_saturated", h.flags)
        self.assertGreater(h.pct_steps_dt_floor, 0.30)

    def test_mesh_too_coarse(self):
        h = compute_health(_good_log(),
                            returncode=0, wall_time_s=20.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.5, max_stag=20)   # ratio=1.0
        self.assertIn("mesh_too_coarse", h.flags)
        self.assertEqual(h.mesh_independence.points, 0.0)

    def test_mesh_borderline_partial_credit(self):
        h = compute_health(_good_log(),
                            returncode=0, wall_time_s=20.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.2, max_stag=20)   # ratio=0.4
        self.assertIn("mesh_borderline", h.flags)
        self.assertGreater(h.mesh_independence.points, 0.0)
        self.assertLess(h.mesh_independence.points,
                         h.mesh_independence.max_points)

    def test_empty_log_is_revise(self):
        h = compute_health({}, returncode=1, wall_time_s=0.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.0625, max_stag=20)
        self.assertEqual(h.verdict, "revise")
        self.assertIn("empty_log", h.flags)

    def test_peak_at_end_partial_credit(self):
        cols = _good_log()
        # Make Fy monotonically rising — peak at last step.
        cols["Fy"] = [10.0 * (i + 1) for i in range(len(cols["Fy"]))]
        h = compute_health(cols,
                            returncode=0, wall_time_s=20.0,
                            fracture_enabled=True, T_total=1.0, dt_first=0.01,
                            eps=0.5, h_min=0.0625, max_stag=20)
        self.assertIn("peak_at_end", h.flags)
        # Should still get partial credit on accuracy (fracture enabled).
        self.assertGreater(h.accuracy.points, 0.0)


class TestReviserRules(unittest.TestCase):
    """Smoke tests over the reviser's rule table."""

    def _make_spec(self):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       LoadingCase, GeometrySpec, LoadingSpec,
                                       MaterialSpec)
        return CanonicalSpec(
            material=MaterialSpec(catalog_name="Steel_bench_2D_PE"),
            geometry=GeometrySpec(kind="notched_plate",
                                   dimensions={"W": 1.0, "L": 1.0}),
            bcs=BCSpec(fixed=[FixedBC(region="bottom", components=[0, 1])],
                       loading=[LoadingCase(region="top", component=1,
                                              control="displacement",
                                              magnitude=0.006)]),
            loading=LoadingSpec(steps=100, T_total=1.0),
        )

    def _make_action(self):
        from fracture_agent.schema import Action
        return Action(variant="linear_elastic_2d_pe",
                       mesh_builder="make_notched_plate_2d",
                       solver="run_quasistatic",
                       rationale="test")

    def _make_health(self, flags, total=40):
        from fracture_agent.health import (ComponentScore, HealthReport, W_INTEGRITY,
                                      W_ADMISSIBILITY, W_ACCURACY,
                                      W_MESH_INDEP, W_EFFICIENCY)
        zero = lambda nm, mx: ComponentScore(name=nm, points=0.0, max_points=mx)
        return HealthReport(
            integrity=zero("integrity", W_INTEGRITY),
            admissibility=zero("admissibility", W_ADMISSIBILITY),
            accuracy=zero("accuracy", W_ACCURACY),
            mesh_independence=zero("mesh_independence", W_MESH_INDEP),
            efficiency=zero("efficiency", W_EFFICIENCY),
            total=float(total), verdict="revise", flags=list(flags))

    def test_dt_floor_saturated_doubles_steps(self):
        from fracture_agent.agents.reviser import reviser
        spec = self._make_spec()
        h = self._make_health(["dt_floor_saturated"])
        h.pct_steps_dt_floor = 0.5
        rev, log = reviser(spec, self._make_action(), h)
        self.assertEqual(rev.action, "revise_spec")
        self.assertEqual(spec.loading.steps, 200)

    def test_stagger_saturated_relaxes_tol(self):
        from fracture_agent.agents.reviser import reviser
        spec = self._make_spec()
        old_tol = spec.tol_stag
        h = self._make_health(["stagger_saturated"])
        h.pct_steps_stag_max = 0.6
        rev, log = reviser(spec, self._make_action(), h)
        self.assertEqual(rev.action, "revise_spec")
        self.assertGreater(spec.tol_stag, old_tol)
        self.assertGreater(spec.max_stag, 20)

    def test_early_termination_halves_max_disp(self):
        from fracture_agent.agents.reviser import reviser
        spec = self._make_spec()
        old = spec.bcs.loading[0].magnitude
        h = self._make_health(["early_termination", "nonzero_rc"])
        rev, log = reviser(spec, self._make_action(), h)
        self.assertEqual(rev.action, "revise_spec")
        self.assertAlmostEqual(spec.bcs.loading[0].magnitude, old * 0.5)

    def test_no_match_defaults_to_accept(self):
        from fracture_agent.agents.reviser import reviser
        spec = self._make_spec()
        h = self._make_health(["unknown_flag"])
        rev, log = reviser(spec, self._make_action(), h)
        self.assertEqual(rev.action, "accept")
        self.assertEqual(log, [])

    def test_persistent_critical_after_two_attempts_is_infeasible(self):
        from fracture_agent.agents.reviser import RevisionAction, reviser
        spec = self._make_spec()
        prior = [
            RevisionAction(action="revise_spec",
                            flags_acted_on=["empty_log"]),
            RevisionAction(action="revise_spec",
                            flags_acted_on=["empty_log"]),
        ]
        h = self._make_health(["empty_log"])
        rev, log = reviser(spec, self._make_action(), h, prior_revisions=prior)
        self.assertEqual(rev.action, "infeasible")


if __name__ == "__main__":
    unittest.main(verbosity=2)
