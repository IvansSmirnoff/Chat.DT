"""Tests for the shared IFC property-name canonicaliser."""

import pytest

from src.utils.property_names import (
    RESERVED_NAMES,
    assign_unique_property_name,
    canonical_property_name,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        # The case that motivated the fix.
        ("Ramp Max-Slope (1/x)", "Ramp_Max_Slope_1_x"),
        # Parens around units.
        ("Heat Transfer Coefficient (U)", "Heat_Transfer_Coefficient_U"),
        ("Thermal Resistance (R)", "Thermal_Resistance_R"),
        # Already canonical -- must round-trip unchanged.
        ("GlobalId", "GlobalId"),
        ("Ramp_Max_Slope_1_x", "Ramp_Max_Slope_1_x"),
        # Pure dots / dashes / mixed.
        ("foo.bar-baz", "foo_bar_baz"),
        # Many runs of separators collapse.
        ("a   b---c...d", "a_b_c_d"),
        # Leading/trailing junk gets stripped.
        ("__x__", "x"),
        ("(weird)", "weird"),
    ],
)
def test_canonical_property_name(raw, expected):
    assert canonical_property_name(raw) == expected


def test_canonical_property_name_is_idempotent():
    samples = [
        "Ramp Max-Slope (1/x)",
        "Heat Transfer Coefficient (U)",
        "weirdName",
        "_leading_and_trailing_",
    ]
    for s in samples:
        once = canonical_property_name(s)
        twice = canonical_property_name(once)
        assert once == twice


@pytest.mark.parametrize("reserved", sorted(RESERVED_NAMES))
def test_reserved_names_get_prefixed(reserved):
    assert canonical_property_name(reserved) == f"ifc_{reserved}"
    # Case-insensitive: "ID" must also be prefixed.
    assert canonical_property_name(reserved.upper()).lower() == f"ifc_{reserved}"


def test_assign_unique_property_name_no_collision():
    assert assign_unique_property_name("Foo Bar", set()) == "Foo_Bar"


def test_assign_unique_property_name_with_collisions():
    existing = {"Foo_Bar"}
    first = assign_unique_property_name("Foo Bar", existing)
    assert first == "Foo_Bar_1"

    existing.add(first)
    second = assign_unique_property_name("Foo Bar", existing)
    assert second == "Foo_Bar_2"


def test_assign_unique_property_name_does_not_mutate_existing():
    existing = {"Foo_Bar"}
    snapshot = set(existing)
    assign_unique_property_name("Foo Bar", existing)
    assert existing == snapshot


def test_loader_and_scanner_agree_for_known_drift_cases():
    """The two sides used to disagree. Now they must produce the same string."""
    from src.constraints.schema_scanner import canonical_property_name as scanner_fn
    from src.etl.loader import sanitize_property_name as loader_fn

    cases = [
        "Ramp Max-Slope (1/x)",
        "Heat Transfer Coefficient (U)",
        "Thermal Resistance (R)",
        "FrameOffset (External)",
    ]
    for raw in cases:
        # Scanner result.
        scanner_out = scanner_fn(raw)
        # Loader result with no collisions -- must match the canonical form.
        loader_out = loader_fn(raw, set())
        assert scanner_out == loader_out, (raw, scanner_out, loader_out)
