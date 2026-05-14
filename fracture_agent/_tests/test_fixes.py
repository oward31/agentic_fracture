"""Tests for the F1-F9 surgical fixes from the prompt-battery diagnostic.

Run with::

    python -m fracture_agent._tests.test_fixes
"""
from __future__ import annotations
import unittest

from fracture_agent.agents.architect import (_BUILTIN_REGION_NAMES,
                                        _filter_geometry_dimensions,
                                        _fill_default_magnitudes,
                                        _ensure_rigid_body_constraint,
                                        _sanitise)
from fracture_agent.region_names import BUILTIN_MESH_REGIONS


# ---------------------------------------------------------------------------
# F1 — dogbone canon table aligned with actual mesh
# ---------------------------------------------------------------------------
class TestDogboneRegions(unittest.TestCase):
    def test_dogbone_2d_only_top_bottom(self):
        # Architect's canon table for dogbone_2d must equal the actual
        # markers_spec from modular/meshes/dogbone_2d.py.
        self.assertEqual(_BUILTIN_REGION_NAMES["dogbone_2d"],
                          BUILTIN_MESH_REGIONS["make_dogbone_2d"])
        self.assertEqual(_BUILTIN_REGION_NAMES["dogbone_2d"],
                          {"top", "bottom"})

    def test_dogbone_3d_top_bottom_zminzmax(self):
        self.assertEqual(_BUILTIN_REGION_NAMES["dogbone_3d"],
                          BUILTIN_MESH_REGIONS["make_dogbone_3d"])
        self.assertEqual(_BUILTIN_REGION_NAMES["dogbone_3d"],
                          {"top", "bottom", "zmin", "zmax"})

    def test_plate_3d_uses_zmin_zmax(self):
        # NOT front/back — those were the old (wrong) names.
        self.assertEqual(_BUILTIN_REGION_NAMES["plate_3d"],
                          {"top", "bottom", "left", "right", "zmin", "zmax"})


# ---------------------------------------------------------------------------
# F8 — geometry dimensions filtered per kind
# ---------------------------------------------------------------------------
class TestFilterGeometryDimensions(unittest.TestCase):
    def _make_spec(self, kind, dims):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, GeometrySpec,
                                       LoadingSpec, MaterialSpec)
        return CanonicalSpec(
            material=MaterialSpec(catalog_name="Steel_bench_2D_PE"),
            geometry=GeometrySpec(kind=kind, dimensions=dims),
            bcs=BCSpec(),
            loading=LoadingSpec(),
        )

    def test_notched_plate_keeps_canonical(self):
        spec = self._make_spec("notched_plate",
                                {"W": 25.0, "L": 25.0, "ac": 12.5,
                                 "load_value": 0.08})
        _filter_geometry_dimensions(spec)
        self.assertEqual(set(spec.geometry.dimensions.keys()),
                          {"W", "L", "ac"})

    def test_brazilian_disc_displacement_dropped(self):
        # P4 in the diagnostic: dimensions={'W': 50, 'L': 0.05} where 0.05
        # was actually the displacement, not a dimension.  For kind=custom
        # we keep everything (no whitelist).
        spec = self._make_spec("custom", {"W": 50.0, "L": 0.05})
        _filter_geometry_dimensions(spec)
        # custom kind: all keys preserved
        self.assertEqual(set(spec.geometry.dimensions.keys()), {"W", "L"})

    def test_dogbone_filters_unrelated(self):
        spec = self._make_spec("dogbone_2d",
                                {"W": 6.0, "L": 30.0, "R": 5.0,
                                 "thickness": 1.0, "ac": 10.0})
        _filter_geometry_dimensions(spec)
        self.assertNotIn("ac", spec.geometry.dimensions)
        # thickness is for dogbone_3d not dogbone_2d
        self.assertNotIn("thickness", spec.geometry.dimensions)
        self.assertEqual(set(spec.geometry.dimensions.keys()),
                          {"W", "L", "R"})

    def test_negative_dimensions_dropped(self):
        spec = self._make_spec("notched_plate",
                                {"W": 10.0, "L": -5.0})
        _filter_geometry_dimensions(spec)
        self.assertNotIn("L", spec.geometry.dimensions)
        self.assertEqual(set(spec.geometry.dimensions.keys()), {"W"})


# ---------------------------------------------------------------------------
# F2 — control enum coercion
# ---------------------------------------------------------------------------
class TestControlCoercion(unittest.TestCase):
    def _reply(self, control):
        return {
            "geometry": {"kind": "notched_plate",
                          "dimensions": {"W": 1.0, "L": 1.0}},
            "bcs": {
                "fixed": [],
                "loading": [{"region": "top", "component": 1,
                              "control": control, "magnitude": 1.0}],
            },
            "material": {},
        }

    def test_velocity_to_traction(self):
        r = self._reply("velocity")
        _sanitise(r)
        self.assertEqual(r["bcs"]["loading"][0]["control"], "traction")

    def test_pressure_to_traction(self):
        r = self._reply("pressure")
        _sanitise(r)
        self.assertEqual(r["bcs"]["loading"][0]["control"], "traction")

    def test_load_to_force(self):
        r = self._reply("load")
        _sanitise(r)
        self.assertEqual(r["bcs"]["loading"][0]["control"], "force")

    def test_unknown_to_displacement(self):
        r = self._reply("oscillating")
        _sanitise(r)
        self.assertEqual(r["bcs"]["loading"][0]["control"], "displacement")

    def test_valid_passes_through(self):
        for ctrl in ("displacement", "force", "traction"):
            r = self._reply(ctrl)
            _sanitise(r)
            self.assertEqual(r["bcs"]["loading"][0]["control"], ctrl)


# ---------------------------------------------------------------------------
# F3 — default magnitude with empty dimensions
# ---------------------------------------------------------------------------
class TestDefaultMagnitudeFallback(unittest.TestCase):
    def _make_spec(self, dims, mag=None):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        return CanonicalSpec(
            material=MaterialSpec(catalog_name="Steel_bench_2D_PE"),
            geometry=GeometrySpec(kind="custom", dimensions=dims),
            bcs=BCSpec(
                fixed=[FixedBC(region="left", components=[0, 1])],
                loading=[LoadingCase(region="right", component=0,
                                       control="displacement",
                                       magnitude=mag or 0.0)]),
            loading=LoadingSpec(),
        )

    def test_empty_dimensions_uses_literal_default(self):
        spec = self._make_spec({})
        _fill_default_magnitudes(spec)
        # magnitude was 0 -> defaulted to 0.1 (literal fallback)
        self.assertAlmostEqual(spec.bcs.loading[0].magnitude, 0.1)

    def test_nonempty_dimensions_uses_1_percent(self):
        spec = self._make_spec({"W": 100.0, "L": 100.0})
        _fill_default_magnitudes(spec)
        self.assertAlmostEqual(spec.bcs.loading[0].magnitude, 1.0)  # 1% of 100

    def test_existing_magnitude_preserved(self):
        spec = self._make_spec({"W": 100.0}, mag=0.5)
        _fill_default_magnitudes(spec)
        self.assertEqual(spec.bcs.loading[0].magnitude, 0.5)


# ---------------------------------------------------------------------------
# F4 — variant × mesh-builder compatibility
# ---------------------------------------------------------------------------
class TestVariantMeshCompat(unittest.TestCase):
    def _make_spec(self, kind, plane, constitutive):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        return CanonicalSpec(
            material=MaterialSpec(catalog_name=None, display_name="Steel",
                                    E=210000.0, nu=0.3, Gc=2.7,
                                    sigma_ts=300.0),
            geometry=GeometrySpec(kind=kind,
                                    dimensions={"W": 100.0, "L": 50.0}),
            bcs=BCSpec(
                fixed=[FixedBC(region="bottom", components=[0, 1])],
                loading=[LoadingCase(region="top", component=1,
                                       control="displacement", magnitude=1.0)]),
            plane_type=plane, constitutive=constitutive,
        )

    def test_finite_elastic_2d_ps_with_notched_plate_coerces_to_custom(self):
        from fracture_agent.agents.strategist import strategist
        spec = self._make_spec("notched_plate", "plane_stress", "lopez_pamies")
        action = strategist(spec)
        self.assertEqual(action.variant, "finite_elastic_2d_ps")
        # The strategist should have detected variant×mesh incompatibility
        # (finite_elastic_2d_ps wants slant_plate) and coerced to custom.
        self.assertEqual(action.mesh_builder, "make_custom_gmsh")
        self.assertEqual(spec.geometry.kind, "custom")

    def test_finite_elastic_2d_pe_with_notched_plate_is_compatible(self):
        from fracture_agent.agents.strategist import strategist
        # finite_elastic_2d_pe legitimately uses make_notched_plate_2d.
        spec = self._make_spec("notched_plate", "plane_strain", "lopez_pamies")
        action = strategist(spec)
        self.assertEqual(action.variant, "finite_elastic_2d_pe")
        self.assertEqual(action.mesh_builder, "make_notched_plate_2d")

    def test_linear_2d_ps_with_notched_plate_compatible(self):
        from fracture_agent.agents.strategist import strategist
        spec = self._make_spec("notched_plate", "plane_stress", "linear_elastic")
        action = strategist(spec)
        self.assertEqual(action.mesh_builder, "make_notched_plate_2d")


# ---------------------------------------------------------------------------
# F5 — catalog material × variant mismatch detection
# ---------------------------------------------------------------------------
class TestCatalogMaterialMismatch(unittest.TestCase):
    def _make_spec(self, catalog_name, plane, constitutive,
                    dimension="2D"):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       Dimension, GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        return CanonicalSpec(
            dimension=Dimension(dimension),
            plane_type=plane, constitutive=constitutive,
            material=MaterialSpec(catalog_name=catalog_name,
                                    display_name=None),
            geometry=GeometrySpec(kind="notched_plate",
                                    dimensions={"W": 5.0, "L": 5.0}),
            bcs=BCSpec(
                fixed=[FixedBC(region="bottom", components=[0, 1])],
                loading=[LoadingCase(region="top", component=1,
                                       control="displacement", magnitude=0.01)]),
        )

    def test_2d_pe_catalog_on_3d_variant_stripped(self):
        from fracture_agent.agents.strategist import strategist
        spec = self._make_spec("Steel_bench_2D_PE", "na",
                                "linear_elastic", dimension="3D")
        action = strategist(spec)
        self.assertEqual(spec.material.catalog_name, None,
                          "3D variant must strip the 2D-PE catalog reference")
        # display_name should be set so handbook lookup has a hint.
        self.assertEqual(spec.material.display_name, "Steel")

    def test_matching_catalog_preserved(self):
        from fracture_agent.agents.strategist import strategist
        spec = self._make_spec("Steel_bench_2D_PE", "plane_strain",
                                "linear_elastic")
        action = strategist(spec)
        self.assertEqual(spec.material.catalog_name, "Steel_bench_2D_PE")


# ---------------------------------------------------------------------------
# F7 — material handbook field backfill
# ---------------------------------------------------------------------------
class TestMaterialBackfill(unittest.TestCase):
    def _ms(self, **kw):
        from fracture_agent.schema import MaterialSpec
        return MaterialSpec(**kw)

    def test_sigma_ts_from_sigma_y0(self):
        from fracture_agent.agents.material_helper import _backfill_required_fields
        m = self._ms(E=70000.0, nu=0.3, Gc=10.0,
                      sigma_y0=200.0, sigma_ts=None,
                      sigma_ts_factor=2.5)
        _backfill_required_fields(m, ptype="ductile")
        self.assertAlmostEqual(m.sigma_ts, 500.0)
        self.assertAlmostEqual(m.sigma_cs, 1000.0)  # 2*sigma_ts

    def test_sigma_ts_inferred_as_0_5_pct_of_E(self):
        # No sigma_y0; sigma_ts ≈ 0.005·E (the conservative correlation
        # that lands in real engineering ranges across metals/polymers).
        from fracture_agent.agents.material_helper import _backfill_required_fields
        m = self._ms(E=210000.0, nu=0.3, Gc=2.7, sigma_ts=None)
        _backfill_required_fields(m, ptype="linear_elasticity")
        self.assertIsNotNone(m.sigma_ts)
        self.assertAlmostEqual(m.sigma_ts, 1050.0)   # 0.5% of 210000

    def test_sigma_ts_writes_to_assumptions(self):
        from fracture_agent.agents.material_helper import _backfill_required_fields
        from fracture_agent.schema import (BCSpec, CanonicalSpec, GeometrySpec,
                                       LoadingSpec, MaterialSpec)
        m = self._ms(E=70000.0, nu=0.3, Gc=10.0, sigma_ts=None)
        spec = CanonicalSpec(material=m,
                              geometry=GeometrySpec(kind="notched_plate",
                                                      dimensions={"W": 1.0,
                                                                  "L": 1.0}),
                              bcs=BCSpec(), loading=LoadingSpec())
        # Use the spec's own material (Pydantic copy semantics)
        _backfill_required_fields(spec.material, ptype="linear_elasticity",
                                    spec=spec)
        self.assertGreaterEqual(len(spec.assumptions), 1)
        self.assertTrue(any("sigma_ts" in a for a in spec.assumptions))

    def test_dynamic_rho_default(self):
        from fracture_agent.agents.material_helper import _backfill_required_fields
        m = self._ms(E=70000.0, nu=0.3, Gc=10.0, sigma_ts=300.0,
                      rho=None)
        _backfill_required_fields(m, ptype="dynamic_linear_elasticity")
        self.assertEqual(m.rho, 8.0e-9)

    def test_finite_elasticity_sigma_hs_default(self):
        from fracture_agent.agents.material_helper import _backfill_required_fields
        m = self._ms(E=10.0, nu=0.45, Gc=0.5, sigma_ts=2.0,
                      sigma_hs=None)
        _backfill_required_fields(m, ptype="finite_elasticity")
        self.assertAlmostEqual(m.sigma_hs, 4.0)

    def test_no_E_or_Gc_skips_backfill(self):
        # When the LLM didn't even return E or Gc, we don't invent them.
        from fracture_agent.agents.material_helper import _backfill_required_fields
        m = self._ms(E=None, nu=None, Gc=None, sigma_ts=None)
        _backfill_required_fields(m, ptype="linear_elasticity")
        self.assertIsNone(m.sigma_ts)


# ---------------------------------------------------------------------------
# F9 — assumptions log on the spec
# ---------------------------------------------------------------------------
class TestAssumptionsLog(unittest.TestCase):
    def test_default_magnitude_appends(self):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        spec = CanonicalSpec(
            material=MaterialSpec(catalog_name="Steel_bench_2D_PE"),
            geometry=GeometrySpec(kind="notched_plate",
                                    dimensions={"W": 100.0, "L": 100.0}),
            bcs=BCSpec(
                fixed=[FixedBC(region="bottom", components=[0, 1])],
                loading=[LoadingCase(region="top", component=1,
                                       control="displacement",
                                       magnitude=0.0)]),
            loading=LoadingSpec(),
        )
        self.assertEqual(spec.assumptions, [])
        _fill_default_magnitudes(spec)
        self.assertEqual(len(spec.assumptions), 1)
        self.assertIn("defaulted", spec.assumptions[0])
        self.assertIn("'top'", spec.assumptions[0])

    def test_rbm_pin_appends(self):
        from fracture_agent.schema import (BCSpec, CanonicalSpec, FixedBC,
                                       GeometrySpec, LoadingCase,
                                       LoadingSpec, MaterialSpec)
        spec = CanonicalSpec(
            material=MaterialSpec(catalog_name="Steel_bench_2D_PE"),
            geometry=GeometrySpec(kind="custom",
                                    dimensions={"W": 100.0, "L": 100.0}),
            bcs=BCSpec(
                # Roller-only y constraints — no x pin → RBM in x.
                fixed=[FixedBC(region="bottom", components=[1])],
                loading=[LoadingCase(region="top", component=1,
                                       control="displacement",
                                       magnitude=-0.1)]),
            loading=LoadingSpec(),
        )
        _ensure_rigid_body_constraint(spec, raw_user_text="")
        self.assertGreaterEqual(len(spec.assumptions), 1)
        self.assertTrue(any("Rigid-body" in a for a in spec.assumptions))


if __name__ == "__main__":
    unittest.main(verbosity=2)
