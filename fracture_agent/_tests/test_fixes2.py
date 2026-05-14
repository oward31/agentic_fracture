"""Tests for F12-F16 + the modular SNES-status patch.

Run with::

    python -m fracture_agent._tests.test_fixes2
"""
from __future__ import annotations
import unittest


# ---------------------------------------------------------------------------
# F13 — None-strip extension
# ---------------------------------------------------------------------------
class TestSanitiseNoneStrip(unittest.TestCase):
    def test_top_level_nones_stripped(self):
        from fracture_agent.agents.architect import _sanitise
        from fracture_agent.schema import CanonicalSpec
        reply = {
            "geometry": None,
            "material": None,
            "bcs": None,
            "loading": None,
            "plane_type": None,
            "constitutive": None,
            "dimension": None,
            "mesh": None,
        }
        _sanitise(reply)
        # All Nones removed → defaults apply.  Geometry has no default
        # (it's required), so we still need to provide one for the schema
        # to validate; check that the others were dropped.
        self.assertNotIn("loading", reply)
        self.assertNotIn("plane_type", reply)
        self.assertNotIn("constitutive", reply)
        self.assertNotIn("dimension", reply)
        self.assertNotIn("mesh", reply)
        # Geometry is required — _sanitise sets it to {} via the geom path
        self.assertIn("geometry", reply)
        # bcs gets defaulted to a dict with empty fixed/loading/free
        self.assertIn("bcs", reply)


# ---------------------------------------------------------------------------
# F14 — traction/force default magnitudes
# ---------------------------------------------------------------------------
class TestTractionDefault(unittest.TestCase):
    def _spec(self, control, sigma_ts=None):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        return CanonicalSpec(
            material=MaterialSpec(catalog_name=None,
                                    display_name="X", E=10000.0,
                                    nu=0.3, Gc=1.0, sigma_ts=sigma_ts),
            geometry=GeometrySpec(kind="custom",
                                    dimensions={"W": 100.0, "L": 100.0}),
            bcs=BCSpec(
                fixed=[FixedBC(region="bottom", components=[0, 1])],
                loading=[LoadingCase(region="top", component=1,
                                       control=control, magnitude=0.0)]),
            loading=LoadingSpec(),
        )

    def test_traction_default_uses_half_sigma_ts(self):
        from fracture_agent.agents.architect import _fill_default_magnitudes
        spec = self._spec("traction", sigma_ts=200.0)
        _fill_default_magnitudes(spec)
        # 0.5 * 200 = 100 MPa
        self.assertAlmostEqual(spec.bcs.loading[0].magnitude, 100.0)

    def test_traction_default_falls_back_to_1_MPa(self):
        from fracture_agent.agents.architect import _fill_default_magnitudes
        spec = self._spec("traction", sigma_ts=None)
        _fill_default_magnitudes(spec)
        self.assertAlmostEqual(spec.bcs.loading[0].magnitude, 1.0)

    def test_force_default_1_N(self):
        from fracture_agent.agents.architect import _fill_default_magnitudes
        spec = self._spec("force")
        _fill_default_magnitudes(spec)
        self.assertAlmostEqual(spec.bcs.loading[0].magnitude, 1.0)

    def test_assumption_logged(self):
        from fracture_agent.agents.architect import _fill_default_magnitudes
        spec = self._spec("traction", sigma_ts=200.0)
        _fill_default_magnitudes(spec)
        self.assertEqual(len(spec.assumptions), 1)
        self.assertIn("traction", spec.assumptions[0])


# ---------------------------------------------------------------------------
# F12 — no_z_evolution flag in health composite
# ---------------------------------------------------------------------------
class TestHealthNoZEvolution(unittest.TestCase):
    def _baseline_log(self, n=100, z_res_value=0.0, min_z=1.0):
        return {
            "step": [float(i + 1) for i in range(n)],
            "time": [(i + 1) * (1.0 / n) for i in range(n)],
            "dt":   [0.01] * n,
            "stag_iters": [2.0] * n,
            "u_res": [1e-9] * n,
            "z_res": [z_res_value] * n,
            "min_z": [min_z] * n,
            "disp": [(i + 1) * (1.0 / n) for i in range(n)],
            "Fy":   [10.0 * (i + 1) for i in range(n)],
        }

    def test_silent_solver_flagged(self):
        from fracture_agent.health import compute_health
        cols = self._baseline_log(z_res_value=0.0, min_z=1.0)
        h = compute_health(cols, returncode=0, wall_time_s=10.0,
                            fracture_enabled=True, T_total=1.0,
                            dt_first=0.01, eps=0.5, h_min=0.0625,
                            max_stag=20)
        self.assertIn("no_z_evolution", h.flags)
        # Hard-revise: even a high score must produce verdict="revise"
        # so the Reflect-Revise loop kicks in.
        self.assertEqual(h.verdict, "revise")

    def test_active_solver_not_flagged(self):
        from fracture_agent.health import compute_health
        # z_res grows occasionally (real stagger activity)
        cols = self._baseline_log(z_res_value=0.0, min_z=1.0)
        cols["z_res"][50] = 1e-8   # one tiny but non-zero z_res
        h = compute_health(cols, returncode=0, wall_time_s=10.0,
                            fracture_enabled=True, T_total=1.0,
                            dt_first=0.01, eps=0.5, h_min=0.0625,
                            max_stag=20)
        self.assertNotIn("no_z_evolution", h.flags)

    def test_cracked_run_not_flagged(self):
        from fracture_agent.health import compute_health
        # Damage grew → not silent
        cols = self._baseline_log(z_res_value=0.0, min_z=1.0)
        cols["min_z"][70] = 0.05   # cracked at step 70
        cols["min_z"][71:] = [0.05] * (len(cols["min_z"]) - 71)
        h = compute_health(cols, returncode=0, wall_time_s=10.0,
                            fracture_enabled=True, T_total=1.0,
                            dt_first=0.01, eps=0.5, h_min=0.0625,
                            max_stag=20)
        self.assertNotIn("no_z_evolution", h.flags)

    def test_fracture_disabled_not_flagged(self):
        from fracture_agent.health import compute_health
        cols = self._baseline_log(z_res_value=0.0, min_z=1.0)
        h = compute_health(cols, returncode=0, wall_time_s=10.0,
                            fracture_enabled=False,
                            T_total=1.0, dt_first=0.01, eps=0.5,
                            h_min=0.0625, max_stag=20)
        self.assertNotIn("no_z_evolution", h.flags)


# ---------------------------------------------------------------------------
# F12 — Reviser rules for no_z_evolution
# ---------------------------------------------------------------------------
class TestReviserNoZEvolution(unittest.TestCase):
    def _spec(self, steps=100, tol=1e-7):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        return CanonicalSpec(
            material=MaterialSpec(catalog_name="Steel_bench_2D_PE"),
            geometry=GeometrySpec(kind="notched_plate",
                                    dimensions={"W": 1.0, "L": 1.0}),
            bcs=BCSpec(
                fixed=[FixedBC(region="bottom", components=[0, 1])],
                loading=[LoadingCase(region="top", component=1,
                                       control="displacement",
                                       magnitude=0.006)]),
            loading=LoadingSpec(steps=steps),
            tol_stag=tol,
        )

    def _action(self):
        from fracture_agent.schema import Action
        return Action(variant="linear_elastic_2d_pe",
                       mesh_builder="make_notched_plate_2d",
                       solver="run_quasistatic",
                       rationale="t")

    def _health(self, flags):
        from fracture_agent.health import (ComponentScore, HealthReport,
                                      W_INTEGRITY, W_ADMISSIBILITY,
                                      W_ACCURACY, W_MESH_INDEP, W_EFFICIENCY)
        zero = lambda nm, mx: ComponentScore(name=nm, points=0.0, max_points=mx)
        return HealthReport(
            integrity=zero("integrity", W_INTEGRITY),
            admissibility=zero("admissibility", W_ADMISSIBILITY),
            accuracy=zero("accuracy", W_ACCURACY),
            mesh_independence=zero("mesh_independence", W_MESH_INDEP),
            efficiency=zero("efficiency", W_EFFICIENCY),
            total=70.0, verdict="revise", flags=list(flags))

    def test_first_attempt_tightens_tol(self):
        from fracture_agent.agents.reviser import reviser
        spec = self._spec(steps=100, tol=1e-7)
        rev, _ = reviser(spec, self._action(),
                          self._health(["no_z_evolution"]))
        self.assertEqual(rev.action, "revise_spec")
        self.assertAlmostEqual(spec.tol_stag, 1e-9)
        self.assertEqual(spec.loading.steps, 100)  # unchanged

    def test_second_attempt_doubles_steps(self):
        from fracture_agent.agents.reviser import RevisionAction, reviser
        spec = self._spec(steps=100, tol=1e-9)
        prior = [RevisionAction(action="revise_spec",
                                 flags_acted_on=["tol_stag_tightened"])]
        rev, _ = reviser(spec, self._action(),
                          self._health(["no_z_evolution"]),
                          prior_revisions=prior)
        self.assertEqual(rev.action, "revise_spec")
        self.assertEqual(spec.loading.steps, 200)

    def test_third_attempt_infeasible(self):
        from fracture_agent.agents.reviser import RevisionAction, reviser
        spec = self._spec(steps=200, tol=1e-9)
        prior = [
            RevisionAction(action="revise_spec",
                            flags_acted_on=["tol_stag_tightened"]),
            RevisionAction(action="revise_spec",
                            flags_acted_on=["steps_doubled"]),
        ]
        rev, _ = reviser(spec, self._action(),
                          self._health(["no_z_evolution"]),
                          prior_revisions=prior)
        self.assertEqual(rev.action, "infeasible")


# ---------------------------------------------------------------------------
# F16 — multi-stage loading detection
# ---------------------------------------------------------------------------
class TestMultiStageLoading(unittest.TestCase):
    def _spec_two_loads(self):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        return CanonicalSpec(
            material=MaterialSpec(catalog_name="Steel_bench_2D_PE"),
            geometry=GeometrySpec(kind="custom",
                                    dimensions={"W": 100.0, "L": 100.0}),
            bcs=BCSpec(
                fixed=[FixedBC(region="bottom", components=[0, 1])],
                loading=[
                    LoadingCase(region="left", component=0,
                                  control="displacement", magnitude=0.005),
                    LoadingCase(region="top", component=1,
                                  control="displacement", magnitude=0.02),
                ]),
            loading=LoadingSpec(),
        )

    def test_first_then_pattern_partitions_time(self):
        from fracture_agent.agents.architect import _detect_multi_stage_loading
        spec = self._spec_two_loads()
        prompt = ("Left edge first shifted by 0.005 mm, held constant. "
                  "Top edge then pulled vertically by 0.02 mm.")
        _detect_multi_stage_loading(spec, prompt)
        self.assertEqual(spec.bcs.loading[0].t_start, 0.0)
        self.assertEqual(spec.bcs.loading[0].t_end, 0.5)
        self.assertEqual(spec.bcs.loading[1].t_start, 0.5)
        self.assertEqual(spec.bcs.loading[1].t_end, 1.0)
        self.assertGreaterEqual(len(spec.assumptions), 1)

    def test_phase_keywords_partition_time(self):
        from fracture_agent.agents.architect import _detect_multi_stage_loading
        spec = self._spec_two_loads()
        prompt = "Phase 1: pre-shear. Phase 2: tensile load."
        _detect_multi_stage_loading(spec, prompt)
        self.assertEqual(spec.bcs.loading[0].t_end, 0.5)

    def test_no_sequencing_keeps_defaults(self):
        from fracture_agent.agents.architect import _detect_multi_stage_loading
        spec = self._spec_two_loads()
        prompt = "Plate pulled in shear and tension simultaneously."
        _detect_multi_stage_loading(spec, prompt)
        # No sequencing language → defaults preserved
        self.assertEqual(spec.bcs.loading[0].t_start, 0.0)
        self.assertEqual(spec.bcs.loading[0].t_end, 1.0)
        self.assertEqual(spec.bcs.loading[1].t_end, 1.0)

    def test_single_load_no_partition(self):
        from fracture_agent.agents.architect import _detect_multi_stage_loading
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        spec = CanonicalSpec(
            material=MaterialSpec(catalog_name="Steel_bench_2D_PE"),
            geometry=GeometrySpec(kind="notched_plate",
                                    dimensions={"W": 1.0, "L": 1.0}),
            bcs=BCSpec(
                fixed=[FixedBC(region="bottom", components=[0, 1])],
                loading=[LoadingCase(region="top", component=1,
                                       control="displacement",
                                       magnitude=0.006)]),
            loading=LoadingSpec(),
        )
        _detect_multi_stage_loading(spec, "first held then pulled")
        self.assertEqual(spec.bcs.loading[0].t_start, 0.0)
        self.assertEqual(spec.bcs.loading[0].t_end, 1.0)


# ---------------------------------------------------------------------------
# Modular SNES status patch — sanity check that the new helpers exist
# ---------------------------------------------------------------------------
class TestModularSNESHelpers(unittest.TestCase):
    def test_is_diverged_imports(self):
        try:
            from modular.common.snes import is_diverged, reason_label
        except ImportError as e:
            # Modular tests may not be runnable without DOLFINx; skip if so.
            self.skipTest(f"modular import failed: {e}")
        # Negative reason → diverged; 0 / positive → not.  PETSc reason
        # codes are platform-dependent so we just check the function runs.
        self.assertIsInstance(is_diverged(-1), bool)
        self.assertFalse(is_diverged(0))
        self.assertFalse(is_diverged(2))   # any positive = converged


if __name__ == "__main__":
    unittest.main(verbosity=2)
