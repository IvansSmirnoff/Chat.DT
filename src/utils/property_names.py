"""Single source of truth for IFC -> Neo4j property-name canonicalisation.

Both the ETL loader (write side) and the schema scanner (read side) used to
ship slightly different sanitisation rules, which produced different names for
the same IFC property. The graph then carried one form (e.g.
``Ramp_Max_Slope__1_x_``) while the bundle vocabulary, system prompt, and
constrained-decoding regex all advertised the other (``Ramp_Max_Slope_1_x``),
silently breaking generated queries that referenced the property.

This module exposes the canonical rule and a collision-aware variant for the
loader. Keep these two functions as the only places that decide a property
name.
"""

from typing import Set

# Neo4j keywords / reserved names that we prefix with ``ifc_`` to avoid clashing
# with Cypher syntax when used unbacktick'd.
RESERVED_NAMES = {"id", "labels", "type", "start", "end"}


def canonical_property_name(name: str) -> str:
    """Map a raw IFC property name to its canonical Neo4j key.

    Steps:
      1. Replace spaces / dots / dashes with underscores.
      2. Replace any remaining non-alphanumeric character (other than ``_``)
         with an underscore.
      3. Collapse runs of consecutive underscores to one.
      4. Strip leading and trailing underscores.
      5. Prefix Neo4j reserved names with ``ifc_``.

    Idempotent: ``canonical_property_name(canonical_property_name(x)) ==
    canonical_property_name(x)``.
    """
    sanitized = name.replace(" ", "_").replace(".", "_").replace("-", "_")
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)

    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    sanitized = sanitized.strip("_")

    if sanitized.lower() in RESERVED_NAMES:
        sanitized = f"ifc_{sanitized}"

    return sanitized


def assign_unique_property_name(name: str, existing: Set[str]) -> str:
    """Loader-side helper: canonicalise then resolve collisions.

    If the canonical form is already in ``existing``, append ``_1``, ``_2``, ...
    until a free key is found. Returns the chosen key but does **not** mutate
    ``existing`` -- the caller decides when to mark it taken.
    """
    base = canonical_property_name(name)
    if base not in existing:
        return base

    counter = 1
    while True:
        candidate = f"{base}_{counter}"
        if candidate not in existing:
            return candidate
        counter += 1
