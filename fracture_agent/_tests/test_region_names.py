"""Tests for ``fracture_agent.region_names``.

Run with::

    python -m fracture_agent._tests.test_region_names

(no pytest dep — pure stdlib unittest, mirroring the rest of the test
folder's script-based style).
"""
from __future__ import annotations
import textwrap
import unittest

from fracture_agent.region_names import (
    BUILTIN_MESH_REGIONS,
    edge_axis_outward_for_region,
    edge_axis_outward_sign,
    edge_from_description,
    enforce_marker_names,
    extract_marker_names,
    slugify,
    validate_bc_regions,
)


class TestSlugify(unittest.TestCase):
    def test_simple_space(self):
        self.assertEqual(slugify("bottom edge"), "bottom_edge")

    def test_phrase(self):
        self.assertEqual(slugify("top half of left edge"),
                         "top_half_of_left_edge")

    def test_mixed_separators(self):
        self.assertEqual(slugify("Notched-PLATE 2D"), "notched_plate_2d")

    def test_collapses_repeated_underscores(self):
        self.assertEqual(slugify("  ___weird  --  thing___"), "weird_thing")

    def test_idempotent(self):
        for name in ("bottom", "top_half_of_left_edge", "y_min"):
            self.assertEqual(slugify(name), name)
            self.assertEqual(slugify(slugify(name)), name)

    def test_empty_input(self):
        self.assertEqual(slugify(""), "")
        self.assertEqual(slugify(None or ""), "")


class TestExtractMarkerNames(unittest.TestCase):
    def test_simple_module(self):
        code = textwrap.dedent("""
            import numpy as np

            def make_custom_gmsh(h0, comm=None, rank=0):
                pass

            markers_spec = [
                (1, "bottom", lambda x: True),
                (2, "top",    lambda x: True),
            ]
        """)
        self.assertEqual(extract_marker_names(code), ["bottom", "top"])

    def test_with_variable_tags(self):
        code = textwrap.dedent("""
            bottom_tag = 1
            markers_spec = [
                (bottom_tag, "bottom_edge", lambda x: True),
                (2,          "top_edge",    lambda x: True),
            ]
        """)
        self.assertEqual(extract_marker_names(code),
                         ["bottom_edge", "top_edge"])

    def test_missing_assignment(self):
        code = "x = 1\n"
        self.assertEqual(extract_marker_names(code), [])

    def test_syntax_error_returns_empty(self):
        self.assertEqual(extract_marker_names("def : broken("), [])


class TestEnforceMarkerNames(unittest.TestCase):
    def test_no_drift_returns_unchanged(self):
        code = (
            'markers_spec = [\n'
            '    (1, "bottom", lambda x: True),\n'
            '    (2, "top",    lambda x: True),\n'
            ']\n'
        )
        new, warnings = enforce_marker_names(code, ["bottom", "top"])
        self.assertEqual(new, code)
        self.assertEqual(warnings, [])

    def test_renames_drifted_markers(self):
        code = (
            'gmsh.model.setPhysicalName(1, t1, "bottom_edge")\n'
            'markers_spec = [\n'
            '    (1, "bottom_edge", lambda x: True),\n'
            '    (2, "top_edge",    lambda x: True),\n'
            ']\n'
        )
        new, warnings = enforce_marker_names(
            code, ["bottom edge", "top edge"])
        self.assertIn('"bottom edge"', new)
        self.assertIn('"top edge"', new)
        self.assertNotIn('"bottom_edge"', new)
        self.assertNotIn('"top_edge"', new)
        self.assertTrue(any("renamed mesh markers" in w for w in warnings))

    def test_count_mismatch_does_not_rewrite(self):
        code = 'markers_spec = [(1, "a", lambda x: True)]\n'
        new, warnings = enforce_marker_names(code, ["a", "b"])
        self.assertEqual(new, code)
        self.assertTrue(any("emitted 1 marker" in w for w in warnings))

    def test_unparseable_does_not_rewrite(self):
        code = "x = 1\n"
        new, warnings = enforce_marker_names(code, ["bottom"])
        self.assertEqual(new, code)
        self.assertTrue(any("could not parse" in w for w in warnings))

    def test_partial_rename(self):
        # Only one of two names drifted — warnings should mention only that.
        code = (
            'markers_spec = [\n'
            '    (1, "bottom",     lambda x: True),\n'
            '    (2, "top_drift",  lambda x: True),\n'
            ']\n'
        )
        new, warnings = enforce_marker_names(code, ["bottom", "top"])
        self.assertIn('"top"', new)
        self.assertNotIn('"top_drift"', new)
        renames = [w for w in warnings if "renamed" in w]
        self.assertEqual(len(renames), 1)
        self.assertIn("'top_drift' -> 'top'", renames[0])


class TestValidateBcRegions(unittest.TestCase):
    """validate_bc_regions duck-types over CanonicalSpec; we mock with
    simple namespaces."""

    @staticmethod
    def _spec(fixed_regions, loading_regions):
        from types import SimpleNamespace
        return SimpleNamespace(
            bcs=SimpleNamespace(
                fixed=[SimpleNamespace(region=r) for r in fixed_regions],
                loading=[SimpleNamespace(region=r) for r in loading_regions],
            )
        )

    def test_all_present(self):
        spec = self._spec(["bottom"], ["top"])
        ok, issues = validate_bc_regions(spec, {"bottom", "top", "left"})
        self.assertTrue(ok)
        self.assertEqual(issues, [])

    def test_missing_one(self):
        spec = self._spec(["bottom"], ["top_half"])
        ok, issues = validate_bc_regions(spec, {"bottom", "top"})
        self.assertFalse(ok)
        self.assertEqual(len(issues), 1)
        self.assertIn("'top_half'", issues[0])

    def test_missing_multiple(self):
        spec = self._spec(["a"], ["b", "c"])
        ok, issues = validate_bc_regions(spec, {"d"})
        self.assertFalse(ok)
        self.assertEqual(sorted(i.split("'")[1] for i in issues),
                         ["a", "b", "c"])


class TestEdgeAxisOutwardSign(unittest.TestCase):
    def test_canonical_edges(self):
        self.assertEqual(edge_axis_outward_sign("left"),   (0, -1))
        self.assertEqual(edge_axis_outward_sign("right"),  (0, +1))
        self.assertEqual(edge_axis_outward_sign("bottom"), (1, -1))
        self.assertEqual(edge_axis_outward_sign("top"),    (1, +1))

    def test_axis_aliases(self):
        self.assertEqual(edge_axis_outward_sign("xmin"), (0, -1))
        self.assertEqual(edge_axis_outward_sign("ymax"), (1, +1))
        self.assertEqual(edge_axis_outward_sign("zmin"), (2, -1))

    def test_top_half_of_left_edge(self):
        # The "_left_edge" substring should pin axis = x.
        self.assertEqual(edge_axis_outward_sign("top_half_of_left_edge"),
                         (0, -1))

    def test_top_left_half_falls_back_to_first_token(self):
        # Without an explicit "_edge" suffix, the first token wins.
        self.assertEqual(edge_axis_outward_sign("top_left_half"),
                         (1, +1))

    def test_unknown_returns_none(self):
        self.assertIsNone(edge_axis_outward_sign("rigid_body_pin"))
        self.assertIsNone(edge_axis_outward_sign(""))
        self.assertIsNone(edge_axis_outward_sign("inner_corner"))


class TestEdgeFromDescription(unittest.TestCase):
    def test_x_zero(self):
        self.assertEqual(
            edge_from_description("x == 0 AND y >= 10",
                                   {"W": 10.0, "L": 20.0}),
            (0, -1))

    def test_y_max(self):
        self.assertEqual(
            edge_from_description("y == 25", {"W": 25.0, "L": 25.0}),
            (1, +1))

    def test_x_max(self):
        self.assertEqual(
            edge_from_description("x == 200", {"W": 200.0, "L": 200.0}),
            (0, +1))

    def test_no_dimensions_only_zero(self):
        # Without dims we can still recognise =0 (definitely a min-edge).
        self.assertEqual(edge_from_description("y == 0", {}), (1, -1))

    def test_inequality_only_returns_none(self):
        self.assertIsNone(edge_from_description("y >= 10", {}))

    def test_empty_returns_none(self):
        self.assertIsNone(edge_from_description("", {"W": 1.0}))


class TestEdgeAxisOutwardForRegion(unittest.TestCase):
    def test_description_overrides_name(self):
        # Name "top_left_half" alone says top-edge (y), but the description
        # says x==0 — left edge — and that should win.
        self.assertEqual(
            edge_axis_outward_for_region("top_left_half",
                                          "x == 0 AND y >= 10",
                                          {"W": 10.0, "L": 20.0}),
            (0, -1))

    def test_falls_back_to_name_without_description(self):
        self.assertEqual(
            edge_axis_outward_for_region("top", "", {}),
            (1, +1))

    def test_unknown_everywhere(self):
        self.assertIsNone(
            edge_axis_outward_for_region("center_pin", "", {}))


class TestBuiltinMeshRegions(unittest.TestCase):
    """Sanity check that the BUILTIN_MESH_REGIONS table doesn't drift away
    from the modular/meshes/ files.  Hardcoded expectations — if a built-in
    mesh changes its exposed names, this test breaks fast."""

    EXPECTED = {
        "make_notched_plate_2d": {"top", "bottom", "left", "right"},
        "make_notched_plate_3d":
            {"top", "bottom", "left", "right", "zmin", "zmax"},
        "make_slant_plate_2d":   {"top", "bottom", "left", "right"},
        "make_dogbone_2d":       {"top", "bottom"},
        "make_dogbone_3d":       {"top", "bottom", "zmin", "zmax"},
    }

    def test_matches(self):
        self.assertEqual(BUILTIN_MESH_REGIONS, self.EXPECTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
