"""Schema-derived answer to "which IFC direct attributes can be a node property?".

The ETL used to read exactly four IFC attributes by hand::

    properties = {"GlobalId": ..., "Name": ..., "Description": ..., "ObjectType": ...}

Everything else an IFC entity declares -- ``IfcSpace.LongName``,
``IfcDoor.OverallWidth``, ``IfcBuildingStorey.Elevation``, ``PredefinedType``,
``Tag`` -- was never looked at. Property sets could not compensate:
``ifcopenshell.util.element.get_psets`` returns property/quantity sets only, and
direct attributes never appear there. The graph therefore spoke the exporter's
property-set vocabulary while an IDS speaks the IFC schema's, and nothing
reconciled the two.

This module derives the answer from the ifcopenshell schema declaration at
runtime instead, so there is no hand-maintained list to fall behind. It is
imported by **both** the ETL loader (write side) and
``src.constraints.schema_scanner`` (read side); keep it the only place that
decides whether an attribute is storable, the same way
``src.utils.property_names`` is the only place that decides what a property is
called.

The rule: an attribute is storable when its declared type bottoms out in a
*simple* EXPRESS type (string / real / integer / boolean / logical / binary),
an enumeration, or an aggregate of those. Attributes typed as entities are
**not** storable -- they are relationship territory (``OwnerHistory``,
``ObjectPlacement``, ``Representation``, ``RelatingStructure``, ...) and are
either materialised as edges or deliberately out of scope. Nothing is listed by
name, so a new IFC version or an unusual leaf class needs no code change.
"""

from functools import lru_cache
from typing import Any, Tuple

from ifcopenshell import ifcopenshell_wrapper

# Classification vocabulary returned by :func:`classify_attribute`.
SIMPLE = "simple"
ENTITY = "entity"
AGGREGATE_SIMPLE = "aggregate-simple"
AGGREGATE_ENTITY = "aggregate-entity"
UNKNOWN = "unknown"

#: Classifications whose values can be stored on a Neo4j node.
STORABLE_KINDS = frozenset({SIMPLE, AGGREGATE_SIMPLE})

# Guard against a pathological schema cycle while unwrapping named types.
_MAX_UNWRAP_DEPTH = 32


def classify_attribute(attr_type: Any, _depth: int = 0) -> str:
    """Classify an EXPRESS attribute type as storable, entity, or aggregate.

    ``attr_type`` is what ``declaration.all_attributes()[i].type_of_attribute()``
    returns. The wrappers nest -- a measure such as ``IfcPositiveLengthMeasure``
    unwraps ``named_type -> type_declaration -> named_type -> type_declaration
    -> simple_type`` before reaching ``'real'`` -- so this recurses rather than
    testing the outermost class only.

    Returns one of :data:`SIMPLE`, :data:`ENTITY`, :data:`AGGREGATE_SIMPLE`,
    :data:`AGGREGATE_ENTITY`, :data:`UNKNOWN`.
    """
    if attr_type is None or _depth > _MAX_UNWRAP_DEPTH:
        return UNKNOWN

    kind = type(attr_type).__name__

    if kind == "named_type":
        return classify_attribute(attr_type.declared_type(), _depth + 1)

    if kind == "entity":
        return ENTITY

    # Enumerations arrive as plain strings from ifcopenshell.
    if kind == "enumeration_type":
        return SIMPLE

    if kind == "simple_type":
        return SIMPLE

    if kind == "type_declaration":
        return classify_attribute(attr_type.declared_type(), _depth + 1)

    if kind == "aggregation_type":
        inner = classify_attribute(attr_type.type_of_element(), _depth + 1)
        if inner == SIMPLE:
            return AGGREGATE_SIMPLE
        if inner in (ENTITY, AGGREGATE_ENTITY):
            return AGGREGATE_ENTITY
        # An aggregate of aggregates of simples is still not a Neo4j list.
        return AGGREGATE_ENTITY if inner == AGGREGATE_SIMPLE else UNKNOWN

    if kind == "select_type":
        # A select is storable only when every member is. IfcValue-style selects
        # (all measure types) qualify; anything admitting an entity does not.
        try:
            members = attr_type.select_list()
        except Exception:
            return UNKNOWN
        if not members:
            return UNKNOWN
        for member in members:
            if classify_attribute(member, _depth + 1) != SIMPLE:
                return ENTITY
        return SIMPLE

    return UNKNOWN


@lru_cache(maxsize=None)
def _storable_attributes(schema_name: str, ifc_class: str) -> Tuple[str, ...]:
    schema = ifcopenshell_wrapper.schema_by_name(schema_name)
    declaration = schema.declaration_by_name(ifc_class)

    # declaration_by_name is typed as the `declaration` base class; the entity
    # subclass is what actually carries all_attributes()/derived().
    attributes = declaration.all_attributes()  # type: ignore[attr-defined]
    try:
        derived = declaration.derived()  # type: ignore[attr-defined]
    except Exception:
        derived = ()

    names = []
    for index, attribute in enumerate(attributes):
        # DERIVE'd attributes are computed by the schema, not authored; reading
        # them can raise or fabricate a value that is not in the file.
        if index < len(derived) and derived[index]:
            continue
        if classify_attribute(attribute.type_of_attribute()) in STORABLE_KINDS:
            names.append(attribute.name())

    return tuple(names)


def storable_attributes(schema_name: str, ifc_class: str) -> Tuple[str, ...]:
    """Names of ``ifc_class``'s direct attributes that can be node properties.

    ``schema_name`` is the IFC schema identifier (``"IFC2X3"``, ``"IFC4"``) --
    ``ifcopenshell.file.schema`` or ``schema.name()``. Results are cached, so
    this is cheap to call once per element.

    Returns an empty tuple for a class the schema does not know, rather than
    raising: a loader should degrade to property sets, not abort the run.
    """
    try:
        return _storable_attributes(schema_name, ifc_class)
    except Exception:
        return ()


def attribute_kinds(schema_name: str, ifc_class: str) -> dict:
    """``{attribute_name: classification}`` for every attribute of a class.

    Diagnostic helper -- used by the loader's run report to show *why* an
    attribute was skipped, so a gap is visible instead of silent.
    """
    try:
        schema = ifcopenshell_wrapper.schema_by_name(schema_name)
        declaration = schema.declaration_by_name(ifc_class)
    except Exception:
        return {}
    return {
        attribute.name(): classify_attribute(attribute.type_of_attribute())
        for attribute in declaration.all_attributes()  # type: ignore[attr-defined]
    }
