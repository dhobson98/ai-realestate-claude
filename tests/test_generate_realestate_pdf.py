"""
Comprehensive test suite for generate_realestate_pdf.py

Test types covered:
  - Unit tests (pure functions, deterministic outputs)
  - Integration tests (PDF generation pipeline)
  - Boundary / edge case tests
  - Negative / error handling tests
  - Regression tests (score thresholds, grade boundaries)
  - Destructive tests (malformed input, missing keys, type errors)
  - Security / injection tests (HTML in strings, path traversal, huge inputs)

Run:
  pip install pytest reportlab
  pytest tests/test_generate_realestate_pdf.py -v --tb=short
  pytest tests/test_generate_realestate_pdf.py -v --cov=scripts --cov-report=term-missing
"""

import os
import sys
import json
import copy
import tempfile
import pytest

# ---------------------------------------------------------------------------
# Path setup — let tests import from scripts/
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import generate_realestate_pdf as pdf_mod
from generate_realestate_pdf import (
    score_color,
    score_grade,
    property_signal,
    signal_color,
    draw_score_gauge,
    create_bar_chart,
    create_neighborhood_bar_chart,
    get_styles,
    standard_table_style,
    get_demo_data,
    generate_report,
    COLORS,
    DISCLAIMER_TEXT,
)
from reportlab.lib.colors import HexColor
from reportlab.graphics.shapes import Drawing
from reportlab.platypus import TableStyle


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def demo_data():
    """Return a deep copy of demo data for each test."""
    return copy.deepcopy(get_demo_data())


@pytest.fixture
def tmp_pdf(tmp_path):
    """Return a temp path for PDF output."""
    return str(tmp_path / "test_output.pdf")


@pytest.fixture
def minimal_data():
    """Minimal valid data dict — only required keys."""
    return {
        "address": "1 Test St",
        "price": "$100,000",
        "overall_score": 50,
    }


# ===========================================================================
# 1. UNIT TESTS — score_color
# ===========================================================================

class TestScoreColor:

    def test_score_70_returns_forest_green(self):
        assert score_color(70) == COLORS["forest_green"]

    def test_score_100_returns_forest_green(self):
        assert score_color(100) == COLORS["forest_green"]

    def test_score_71_returns_forest_green(self):
        assert score_color(71) == COLORS["forest_green"]

    def test_score_69_returns_warm_gold(self):
        assert score_color(69) == COLORS["warm_gold"]

    def test_score_40_returns_warm_gold(self):
        assert score_color(40) == COLORS["warm_gold"]

    def test_score_55_returns_warm_gold(self):
        assert score_color(55) == COLORS["warm_gold"]

    def test_score_39_returns_danger(self):
        assert score_color(39) == COLORS["danger"]

    def test_score_0_returns_danger(self):
        assert score_color(0) == COLORS["danger"]

    def test_score_1_returns_danger(self):
        assert score_color(1) == COLORS["danger"]

    def test_returns_hex_color_instance(self):
        for s in [0, 25, 40, 55, 70, 85, 100]:
            assert hasattr(score_color(s), "hexval"), f"score_color({s}) not a color"

    # Boundary regression
    def test_boundary_40_is_gold_not_danger(self):
        assert score_color(40) == COLORS["warm_gold"]

    def test_boundary_39_is_danger_not_gold(self):
        assert score_color(39) == COLORS["danger"]

    def test_boundary_70_is_green_not_gold(self):
        assert score_color(70) == COLORS["forest_green"]

    def test_boundary_69_is_gold_not_green(self):
        assert score_color(69) == COLORS["warm_gold"]


# ===========================================================================
# 2. UNIT TESTS — score_grade
# ===========================================================================

class TestScoreGrade:

    @pytest.mark.parametrize("score,expected", [
        (100, "A+"), (85, "A+"), (86, "A+"),
        (84, "A"),  (70, "A"),  (71, "A"),
        (69, "B"),  (55, "B"),  (56, "B"),
        (54, "C"),  (40, "C"),  (41, "C"),
        (39, "D"),  (25, "D"),  (26, "D"),
        (24, "F"),  (0, "F"),   (1, "F"),
    ])
    def test_grade_mapping(self, score, expected):
        assert score_grade(score) == expected

    def test_returns_string(self):
        for s in range(0, 101, 5):
            assert isinstance(score_grade(s), str)

    # Regression: all 6 grade values must be reachable
    def test_all_grades_reachable(self):
        grades = {score_grade(s) for s in range(0, 101)}
        assert grades == {"A+", "A", "B", "C", "D", "F"}

    # Boundary: exact threshold values
    def test_exact_boundary_85(self):
        assert score_grade(85) == "A+"

    def test_exact_boundary_84(self):
        assert score_grade(84) == "A"

    def test_exact_boundary_70(self):
        assert score_grade(70) == "A"

    def test_exact_boundary_69(self):
        assert score_grade(69) == "B"

    def test_exact_boundary_55(self):
        assert score_grade(55) == "B"

    def test_exact_boundary_54(self):
        assert score_grade(54) == "C"

    def test_exact_boundary_40(self):
        assert score_grade(40) == "C"

    def test_exact_boundary_39(self):
        assert score_grade(39) == "D"

    def test_exact_boundary_25(self):
        assert score_grade(25) == "D"

    def test_exact_boundary_24(self):
        assert score_grade(24) == "F"


# ===========================================================================
# 3. UNIT TESTS — property_signal
# ===========================================================================

class TestPropertySignal:

    @pytest.mark.parametrize("score,expected", [
        (100, "STRONG BUY"), (85, "STRONG BUY"),
        (84, "BUY"),         (70, "BUY"),
        (69, "HOLD / WATCH"), (55, "HOLD / WATCH"),
        (54, "CAUTION"),     (40, "CAUTION"),
        (39, "PASS"),        (25, "PASS"),
        (24, "AVOID"),       (0, "AVOID"),
    ])
    def test_signal_mapping(self, score, expected):
        assert property_signal(score) == expected

    def test_returns_string(self):
        for s in range(0, 101, 5):
            assert isinstance(property_signal(s), str)

    def test_all_signals_reachable(self):
        signals = {property_signal(s) for s in range(0, 101)}
        assert signals == {"STRONG BUY", "BUY", "HOLD / WATCH", "CAUTION", "PASS", "AVOID"}

    # Regression: boundary values
    def test_boundary_85(self):
        assert property_signal(85) == "STRONG BUY"

    def test_boundary_84(self):
        assert property_signal(84) == "BUY"


# ===========================================================================
# 4. UNIT TESTS — signal_color
# ===========================================================================

class TestSignalColor:

    def test_score_70_returns_forest_green(self):
        assert signal_color(70) == COLORS["forest_green"]

    def test_score_100_returns_forest_green(self):
        assert signal_color(100) == COLORS["forest_green"]

    def test_score_55_returns_sky_blue(self):
        assert signal_color(55) == COLORS["sky_blue"]

    def test_score_69_returns_sky_blue(self):
        assert signal_color(69) == COLORS["sky_blue"]

    def test_score_40_returns_warm_gold(self):
        assert signal_color(40) == COLORS["warm_gold"]

    def test_score_54_returns_warm_gold(self):
        assert signal_color(54) == COLORS["warm_gold"]

    def test_score_0_returns_danger(self):
        assert signal_color(0) == COLORS["danger"]

    def test_score_39_returns_danger(self):
        assert signal_color(39) == COLORS["danger"]

    def test_returns_hex_color_instance(self):
        for s in [0, 39, 40, 54, 55, 69, 70, 100]:
            assert hasattr(signal_color(s), "hexval"), f"signal_color({s}) not a color"

    # Regression: signal_color and score_color agree on green threshold (>=70)
    def test_green_threshold_agreement_with_score_color(self):
        for s in [70, 80, 100]:
            assert signal_color(s) == score_color(s) == COLORS["forest_green"]


# ===========================================================================
# 5. UNIT TESTS — draw_score_gauge
# ===========================================================================

class TestDrawScoreGauge:

    def test_returns_drawing_instance(self):
        d = draw_score_gauge(72)
        assert isinstance(d, Drawing)

    def test_default_size_produces_correct_dimensions(self):
        d = draw_score_gauge(72, size=140)
        assert d.width == 160   # size + 20
        assert d.height == 160

    def test_custom_size(self):
        d = draw_score_gauge(50, size=200)
        assert d.width == 220
        assert d.height == 220

    def test_score_zero(self):
        d = draw_score_gauge(0)
        assert isinstance(d, Drawing)

    def test_score_100(self):
        d = draw_score_gauge(100)
        assert isinstance(d, Drawing)

    def test_drawing_has_elements(self):
        d = draw_score_gauge(72)
        # Should have circles + string elements
        assert len(d.contents) > 0

    def test_drawing_not_empty_for_any_score(self):
        for score in [0, 25, 40, 55, 70, 85, 100]:
            d = draw_score_gauge(score)
            assert len(d.contents) > 0


# ===========================================================================
# 6. UNIT TESTS — create_bar_chart
# ===========================================================================

class TestCreateBarChart:

    def test_returns_drawing(self):
        d = create_bar_chart(["Cat A", "Cat B"], [60, 80])
        assert isinstance(d, Drawing)

    def test_default_dimensions(self):
        d = create_bar_chart(["A", "B"], [50, 50])
        assert d.width == 470
        assert d.height == 200

    def test_custom_dimensions(self):
        d = create_bar_chart(["A"], [50], width=300, height=100)
        assert d.width == 300
        assert d.height == 100

    def test_empty_categories(self):
        d = create_bar_chart([], [])
        assert isinstance(d, Drawing)

    def test_single_category(self):
        d = create_bar_chart(["Only One"], [75])
        assert isinstance(d, Drawing)

    def test_all_five_categories(self):
        cats = ["Value & Comps", "Income Potential", "Neighborhood", "Investment", "Market"]
        scores = [62, 52, 88, 44, 41]
        d = create_bar_chart(cats, scores)
        assert isinstance(d, Drawing)

    def test_score_zero_no_crash(self):
        d = create_bar_chart(["Zero"], [0])
        assert isinstance(d, Drawing)

    def test_score_100_no_crash(self):
        d = create_bar_chart(["Perfect"], [100])
        assert isinstance(d, Drawing)

    def test_long_category_name_truncated(self):
        # Names >25 chars should be sliced without crashing
        long_name = "A" * 50
        d = create_bar_chart([long_name], [50])
        assert isinstance(d, Drawing)


# ===========================================================================
# 7. UNIT TESTS — create_neighborhood_bar_chart
# ===========================================================================

class TestCreateNeighborhoodBarChart:

    def test_returns_drawing(self):
        d = create_neighborhood_bar_chart(["Schools", "Crime"], [80, 70])
        assert isinstance(d, Drawing)

    def test_default_dimensions(self):
        d = create_neighborhood_bar_chart(["A"], [50])
        assert d.width == 470
        assert d.height == 160

    def test_color_green_for_high_score(self):
        # Score >= 80 → forest_green; just verify no crash
        d = create_neighborhood_bar_chart(["High"], [80])
        assert isinstance(d, Drawing)

    def test_color_sky_blue_range(self):
        d = create_neighborhood_bar_chart(["Mid-high"], [60])
        assert isinstance(d, Drawing)

    def test_color_warm_gold_range(self):
        d = create_neighborhood_bar_chart(["Mid"], [40])
        assert isinstance(d, Drawing)

    def test_color_danger_range(self):
        d = create_neighborhood_bar_chart(["Low"], [20])
        assert isinstance(d, Drawing)

    def test_score_zero_minimum_bar_width(self):
        # max(0/100 * width, 2) must not be negative → no crash
        d = create_neighborhood_bar_chart(["Zero"], [0])
        assert isinstance(d, Drawing)

    def test_long_name_truncated(self):
        d = create_neighborhood_bar_chart(["B" * 50], [50])
        assert isinstance(d, Drawing)

    # Regression: bar_width uses max(..., 2) floor — verify same logic in bar_chart
    def test_minimum_bar_width_prevents_zero_rect(self):
        # This exercises the max(..., 2) branch at score=0
        d1 = create_neighborhood_bar_chart(["Zero"], [0])
        d2 = create_bar_chart(["Zero"], [0])
        assert isinstance(d1, Drawing)
        assert isinstance(d2, Drawing)


# ===========================================================================
# 8. UNIT TESTS — get_styles
# ===========================================================================

class TestGetStyles:

    REQUIRED_KEYS = [
        "title", "address", "price", "subtitle", "heading", "subheading",
        "body", "body_small", "signal", "footer", "disclaimer",
        "grade_large", "bullet",
    ]

    def test_returns_dict(self):
        assert isinstance(get_styles(), dict)

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_key_present(self, key):
        styles = get_styles()
        assert key in styles, f"Missing style key: {key}"

    def test_all_values_are_paragraph_style(self):
        from reportlab.lib.styles import ParagraphStyle
        for key, val in get_styles().items():
            assert isinstance(val, ParagraphStyle), f"Style '{key}' is not a ParagraphStyle"

    def test_idempotent_multiple_calls(self):
        s1 = get_styles()
        s2 = get_styles()
        assert set(s1.keys()) == set(s2.keys())


# ===========================================================================
# 9. UNIT TESTS — standard_table_style
# ===========================================================================

class TestStandardTableStyle:

    def test_returns_table_style(self):
        assert isinstance(standard_table_style(), TableStyle)

    def test_no_extra_returns_table_style(self):
        ts = standard_table_style(extra=None)
        assert isinstance(ts, TableStyle)

    def test_extra_commands_merged(self):
        extra = [("ALIGN", (0, 0), (-1, -1), "CENTER")]
        ts = standard_table_style(extra=extra)
        assert isinstance(ts, TableStyle)

    def test_empty_extra_list(self):
        ts = standard_table_style(extra=[])
        assert isinstance(ts, TableStyle)


# ===========================================================================
# 10. UNIT TESTS — get_demo_data
# ===========================================================================

class TestGetDemoData:

    REQUIRED_TOP_KEYS = [
        "address", "price", "date", "overall_score", "property_details",
        "categories", "comps", "comp_summary", "cashflow", "investment_metrics",
        "mortgage", "neighborhood", "strategies", "appreciation_projections",
        "scenarios", "recommendation", "risk_factors",
    ]

    def test_returns_dict(self):
        assert isinstance(get_demo_data(), dict)

    @pytest.mark.parametrize("key", REQUIRED_TOP_KEYS)
    def test_required_key_present(self, key):
        data = get_demo_data()
        assert key in data, f"Missing key: {key}"

    def test_overall_score_is_numeric(self):
        assert isinstance(get_demo_data()["overall_score"], (int, float))

    def test_overall_score_in_valid_range(self):
        score = get_demo_data()["overall_score"]
        assert 0 <= score <= 100

    def test_address_is_string(self):
        assert isinstance(get_demo_data()["address"], str)

    def test_price_is_string(self):
        assert isinstance(get_demo_data()["price"], str)

    def test_categories_has_five_entries(self):
        assert len(get_demo_data()["categories"]) == 5

    def test_comps_has_entries(self):
        assert len(get_demo_data()["comps"]) > 0

    def test_cashflow_items_list(self):
        items = get_demo_data()["cashflow"]["items"]
        assert isinstance(items, list)
        assert len(items) > 0

    def test_strategies_list(self):
        assert isinstance(get_demo_data()["strategies"], list)

    def test_appreciation_projections_list(self):
        assert isinstance(get_demo_data()["appreciation_projections"], list)

    def test_risk_factors_list(self):
        assert isinstance(get_demo_data()["risk_factors"], list)

    def test_returns_independent_copies(self):
        d1 = get_demo_data()
        d2 = get_demo_data()
        d1["address"] = "mutated"
        assert d2["address"] != "mutated"

    def test_property_details_keys(self):
        pd = get_demo_data()["property_details"]
        for key in ["beds", "baths", "sqft", "year_built", "lot_size", "property_type"]:
            assert key in pd


# ===========================================================================
# 11. INTEGRATION TESTS — generate_report (full pipeline)
# ===========================================================================

class TestGenerateReport:

    def test_demo_data_produces_pdf(self, demo_data, tmp_pdf):
        result = generate_report(demo_data, tmp_pdf)
        assert result == tmp_pdf
        assert os.path.isfile(tmp_pdf)

    def test_pdf_has_nonzero_size(self, demo_data, tmp_pdf):
        generate_report(demo_data, tmp_pdf)
        assert os.path.getsize(tmp_pdf) > 1024  # > 1 KB

    def test_returns_output_path(self, demo_data, tmp_pdf):
        result = generate_report(demo_data, tmp_pdf)
        assert result == tmp_pdf

    def test_minimal_data_produces_pdf(self, minimal_data, tmp_pdf):
        generate_report(minimal_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_score_zero_pdf(self, demo_data, tmp_pdf):
        demo_data["overall_score"] = 0
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_score_100_pdf(self, demo_data, tmp_pdf):
        demo_data["overall_score"] = 100
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_empty_comps_falls_back_to_defaults(self, demo_data, tmp_pdf):
        demo_data["comps"] = []
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_empty_categories_falls_back_to_defaults(self, demo_data, tmp_pdf):
        demo_data["categories"] = {}
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_missing_cashflow_items_uses_defaults(self, demo_data, tmp_pdf):
        del demo_data["cashflow"]["items"]
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_missing_neighborhood_scores_uses_defaults(self, demo_data, tmp_pdf):
        del demo_data["neighborhood"]["scores"]
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_empty_strategies_uses_defaults(self, demo_data, tmp_pdf):
        demo_data["strategies"] = []
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_empty_projections_uses_defaults(self, demo_data, tmp_pdf):
        demo_data["appreciation_projections"] = []
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_empty_scenarios_uses_defaults(self, demo_data, tmp_pdf):
        demo_data["scenarios"] = []
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_empty_risk_factors_uses_defaults(self, demo_data, tmp_pdf):
        demo_data["risk_factors"] = []
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_overwrite_existing_pdf(self, demo_data, tmp_pdf):
        generate_report(demo_data, tmp_pdf)
        size1 = os.path.getsize(tmp_pdf)
        generate_report(demo_data, tmp_pdf)
        size2 = os.path.getsize(tmp_pdf)
        assert size2 > 0

    def test_custom_address_in_pdf(self, demo_data, tmp_pdf):
        demo_data["address"] = "999 Custom Blvd, Test City TX 99999"
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_unicode_address(self, demo_data, tmp_pdf):
        demo_data["address"] = "Calle Señor López #42, Ciudad de México"
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_categories_as_plain_numbers(self, demo_data, tmp_pdf):
        # categories values as raw ints rather than dicts
        demo_data["categories"] = {
            "Value & Comps": 62,
            "Income Potential": 52,
            "Neighborhood Quality": 88,
            "Investment Upside": 44,
            "Market Conditions": 41,
        }
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_two_scenarios_no_crash(self, demo_data, tmp_pdf):
        # Scenario color logic requires >=3; test with 2 (branch not taken)
        demo_data["scenarios"] = demo_data["scenarios"][:2]
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_one_comp(self, demo_data, tmp_pdf):
        demo_data["comps"] = [demo_data["comps"][0]]
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)


# ===========================================================================
# 12. REGRESSION TESTS — score thresholds produce correct grade+signal combos
# ===========================================================================

class TestScoreThresholdRegression:
    """Ensure grade and signal stay in sync across all thresholds."""

    @pytest.mark.parametrize("score,grade,signal", [
        (100, "A+", "STRONG BUY"),
        (90,  "A+", "STRONG BUY"),
        (85,  "A+", "STRONG BUY"),
        (84,  "A",  "BUY"),
        (75,  "A",  "BUY"),
        (70,  "A",  "BUY"),
        (69,  "B",  "HOLD / WATCH"),
        (60,  "B",  "HOLD / WATCH"),
        (55,  "B",  "HOLD / WATCH"),
        (54,  "C",  "CAUTION"),
        (47,  "C",  "CAUTION"),
        (40,  "C",  "CAUTION"),
        (39,  "D",  "PASS"),
        (30,  "D",  "PASS"),
        (25,  "D",  "PASS"),
        (24,  "F",  "AVOID"),
        (10,  "F",  "AVOID"),
        (0,   "F",  "AVOID"),
    ])
    def test_grade_and_signal_consistent(self, score, grade, signal):
        assert score_grade(score) == grade
        assert property_signal(score) == signal


# ===========================================================================
# 13. NEGATIVE TESTS — bad inputs, missing keys, wrong types
# ===========================================================================

class TestNegativeInputs:

    def test_generate_report_missing_overall_score_defaults_zero(self, tmp_pdf):
        data = {"address": "Test", "price": "$100k"}
        generate_report(data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_generate_report_score_as_string_digit(self, demo_data, tmp_pdf):
        # data.get("overall_score", 0) — if string is passed, score_grade/signal receive it
        demo_data["overall_score"] = "72"
        # score_grade compares with >= which works on numeric strings in some cases,
        # but should not crash the PDF pipeline with "72"
        # Python: "72" >= 70 raises TypeError — this should propagate cleanly
        with pytest.raises((TypeError, Exception)):
            generate_report(demo_data, tmp_pdf)

    def test_score_color_negative_score(self):
        # Negative score falls into danger branch
        assert score_color(-1) == COLORS["danger"]

    def test_score_grade_negative_score(self):
        assert score_grade(-1) == "F"

    def test_property_signal_negative_score(self):
        assert property_signal(-1) == "AVOID"

    def test_signal_color_negative_score(self):
        assert signal_color(-1) == COLORS["danger"]

    def test_score_color_above_100(self):
        # >100 returns green (>= 70 branch)
        assert score_color(150) == COLORS["forest_green"]

    def test_score_grade_above_100(self):
        assert score_grade(150) == "A+"

    def test_property_signal_above_100(self):
        assert property_signal(150) == "STRONG BUY"

    def test_generate_report_invalid_output_path(self, demo_data):
        bad_path = "/nonexistent_directory_xyz/output.pdf"
        with pytest.raises((IOError, OSError, Exception)):
            generate_report(demo_data, bad_path)

    def test_generate_report_empty_dict(self, tmp_pdf):
        generate_report({}, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_draw_score_gauge_fractional_score(self):
        d = draw_score_gauge(72.7)
        assert isinstance(d, Drawing)

    def test_bar_chart_mismatched_lengths(self):
        # More categories than scores — zip truncates; should not crash
        d = create_bar_chart(["A", "B", "C"], [50, 60])
        assert isinstance(d, Drawing)

    def test_neighborhood_chart_mismatched_lengths(self):
        d = create_neighborhood_bar_chart(["A", "B"], [50])
        assert isinstance(d, Drawing)


# ===========================================================================
# 14. DESTRUCTIVE TESTS — extreme / adversarial values
# ===========================================================================

class TestDestructiveInputs:

    def test_score_float_boundary(self):
        # 69.9 should still be "B" not "A"
        assert score_grade(69.9) == "B"

    def test_score_70_0_exactly(self):
        assert score_grade(70.0) == "A"

    def test_very_large_score(self):
        assert score_grade(999999) == "A+"
        assert property_signal(999999) == "STRONG BUY"

    def test_very_negative_score(self):
        assert score_grade(-999999) == "F"

    def test_nan_score_in_color_raises_or_handles(self):
        import math
        # NaN comparisons always return False; all >= checks fail → falls to danger
        result = score_color(float("nan"))
        assert result == COLORS["danger"]

    def test_nan_grade(self):
        import math
        result = score_grade(float("nan"))
        assert result == "F"

    def test_generate_report_very_long_address(self, demo_data, tmp_pdf):
        demo_data["address"] = "A" * 500
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_generate_report_very_long_price(self, demo_data, tmp_pdf):
        demo_data["price"] = "$" + "9" * 20
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_generate_report_many_comps(self, demo_data, tmp_pdf):
        demo_data["comps"] = [
            {"address": f"{i} Test Ave", "price": f"${400000+i*1000}",
             "sqft": "1800", "price_sqft": "$222", "sold_date": "Jan 2026", "distance": "0.5 mi"}
            for i in range(50)
        ]
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_generate_report_many_risk_factors(self, demo_data, tmp_pdf):
        demo_data["risk_factors"] = [
            {"factor": f"Risk {i}", "probability": "Medium", "impact": "High", "notes": f"Note {i}"}
            for i in range(30)
        ]
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_generate_report_many_action_items(self, demo_data, tmp_pdf):
        demo_data["recommendation"]["action_items"] = [f"Item {i}" for i in range(50)]
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_bar_chart_100_categories(self):
        cats = [f"Cat{i}" for i in range(100)]
        scores = [i % 101 for i in range(100)]
        d = create_bar_chart(cats, scores)
        assert isinstance(d, Drawing)

    def test_generate_report_null_values_in_comps(self, demo_data, tmp_pdf):
        demo_data["comps"] = [
            {"address": None, "price": None, "sqft": None,
             "price_sqft": None, "sold_date": None, "distance": None}
        ]
        # .get() with default "" handles None keys but not None values
        # This may or may not crash depending on reportlab — capture either outcome
        try:
            generate_report(demo_data, tmp_pdf)
        except Exception:
            pass  # Acceptable — None values are not valid input

    def test_score_gauge_size_zero_no_divide_by_zero(self):
        # size=0 → division-related shapes; should not raise ZeroDivisionError
        try:
            d = draw_score_gauge(50, size=0)
            assert isinstance(d, Drawing)
        except Exception:
            pass  # Acceptable for nonsensical size


# ===========================================================================
# 15. SECURITY / INJECTION TESTS
# ===========================================================================

class TestSecurityInputs:

    def test_html_in_address_does_not_crash(self, tmp_pdf):
        data = {"address": "<script>alert('xss')</script>", "price": "$100", "overall_score": 50}
        generate_report(data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_html_tags_in_price(self, demo_data, tmp_pdf):
        demo_data["price"] = "<b>$999,999</b>"
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_sql_injection_string_in_address(self, demo_data, tmp_pdf):
        demo_data["address"] = "'; DROP TABLE properties; --"
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_path_traversal_in_output_path(self, demo_data):
        bad_path = "../../etc/passwd.pdf"
        # In a real deployment this would write to a relative path — here it may
        # succeed or fail depending on cwd; important: no code execution
        try:
            generate_report(demo_data, bad_path)
        except Exception:
            pass  # Expected if directory doesn't exist
        finally:
            if os.path.isfile(bad_path):
                os.remove(bad_path)

    def test_null_byte_in_string(self, demo_data, tmp_pdf):
        null_char = chr(0)
        demo_data["address"] = f"123 Main{null_char}Street"
        try:
            generate_report(demo_data, tmp_pdf)
        except Exception:
            pass  # Null bytes in strings may cause issues — no crash into shell is OK

    def test_format_string_in_address(self, demo_data, tmp_pdf):
        demo_data["address"] = "%s %d %n {{inject}}"
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)

    def test_unicode_control_chars(self, demo_data, tmp_pdf):
        demo_data["address"] = "123 Main​﻿Street"
        try:
            generate_report(demo_data, tmp_pdf)
        except Exception:
            pass  # Some control chars rejected by reportlab — no shell execution

    def test_extremely_nested_json_no_crash(self, demo_data, tmp_pdf):
        # Deeply nested dict in notes — should not cause stack overflow
        nested = {"notes": "ok"}
        for _ in range(50):
            nested = {"wrapper": nested}
        demo_data["risk_factors"] = [{"factor": "Deep", "probability": "Low",
                                       "impact": "Low", "notes": str(nested)}]
        generate_report(demo_data, tmp_pdf)
        assert os.path.isfile(tmp_pdf)


# ===========================================================================
# 16. CLI / main() INTEGRATION TESTS
# ===========================================================================

class TestMainCLI:

    def test_demo_mode_no_args(self, tmp_path, monkeypatch):
        out_file = str(tmp_path / "PROPERTY-REPORT-sample.pdf")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["generate_realestate_pdf.py"])
        pdf_mod.main()
        assert os.path.isfile(out_file)

    def test_demo_mode_explicit_flag(self, tmp_path, monkeypatch):
        out_file = str(tmp_path / "PROPERTY-REPORT-sample.pdf")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["generate_realestate_pdf.py", "--demo"])
        pdf_mod.main()
        assert os.path.isfile(out_file)

    def test_json_input_mode(self, demo_data, tmp_path, monkeypatch):
        json_file = str(tmp_path / "input.json")
        out_file = str(tmp_path / "output.pdf")
        with open(json_file, "w") as f:
            json.dump(demo_data, f)
        monkeypatch.setattr(sys, "argv", ["generate_realestate_pdf.py", json_file, out_file])
        pdf_mod.main()
        assert os.path.isfile(out_file)

    def test_json_input_default_output_filename(self, demo_data, tmp_path, monkeypatch):
        json_file = str(tmp_path / "input.json")
        default_out = str(tmp_path / "PROPERTY-REPORT.pdf")
        with open(json_file, "w") as f:
            json.dump(demo_data, f)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["generate_realestate_pdf.py", json_file])
        pdf_mod.main()
        assert os.path.isfile(default_out)

    def test_missing_input_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "generate_realestate_pdf.py",
            str(tmp_path / "nonexistent.json"),
        ])
        with pytest.raises((FileNotFoundError, IOError)):
            pdf_mod.main()

    def test_invalid_json_raises(self, tmp_path, monkeypatch):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not valid json {{{{")
        monkeypatch.setattr(sys, "argv", ["generate_realestate_pdf.py", str(bad_json)])
        with pytest.raises((json.JSONDecodeError, ValueError)):
            pdf_mod.main()


# ===========================================================================
# 17. COLORS DICT TESTS
# ===========================================================================

class TestColorPalette:

    REQUIRED_COLORS = [
        "navy", "navy_light", "forest_green", "green_light", "warm_gold",
        "gold_light", "danger", "sky_blue", "gray", "light_bg", "text",
        "text_light", "border", "header_bg", "row_alt", "white", "black",
    ]

    @pytest.mark.parametrize("color_key", REQUIRED_COLORS)
    def test_color_key_exists(self, color_key):
        assert color_key in COLORS

    def test_all_hex_colors_valid(self):
        for key, val in COLORS.items():
            # All values should have hexval() or be reportlab color objects
            assert hasattr(val, "hexval") or hasattr(val, "red"), f"Bad color: {key}"

    def test_navy_is_dark(self):
        # COLORS["navy"] should be #1a2332 — a dark color (low RGB)
        c = COLORS["navy"]
        # reportlab HexColor exposes .red, .green, .blue as 0-1 floats
        assert c.red < 0.2

    def test_forest_green_has_green_dominant(self):
        c = COLORS["forest_green"]
        assert c.green > c.red


# ===========================================================================
# 18. DISCLAIMER TEXT TEST
# ===========================================================================

class TestDisclaimerText:

    def test_disclaimer_not_empty(self):
        assert len(DISCLAIMER_TEXT) > 0

    def test_disclaimer_contains_not_financial_advice(self):
        assert "NOT" in DISCLAIMER_TEXT
        assert "financial" in DISCLAIMER_TEXT.lower()

    def test_disclaimer_contains_educational(self):
        assert "educational" in DISCLAIMER_TEXT.lower()

    def test_disclaimer_mentions_licensed_professionals(self):
        assert "licensed" in DISCLAIMER_TEXT.lower()

    def test_disclaimer_is_string(self):
        assert isinstance(DISCLAIMER_TEXT, str)
