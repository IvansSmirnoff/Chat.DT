"""IFC class hierarchy, read from the ifcopenshell schema declarations.

Graph nodes carry their leaf class plus every supertype as labels, so a node
count alone cannot always say which label is the leaf: when a class has a single
subclass in the model (every opening an ``IfcOpeningElement``, every wall an
``IfcWallStandardCase``) the labels sit on exactly the same nodes. The schema
can. Nothing here is listed by name.
"""

from functools import lru_cache
from typing import FrozenSet, Tuple


@lru_cache(maxsize=None)
def supertypes(ifc_class: str) -> Tuple[FrozenSet[str], ...]:
    """Supertype names of ``ifc_class``, one set per IFC schema that declares it.

    Per schema because the hierarchy moved between versions: IFC4X3 puts
    ``IfcWall`` under ``IfcBuiltElement``, the earlier schemas under
    ``IfcBuildingElement``. Empty for a name no schema declares (a synthetic or
    internal label), or without ifcopenshell.
    """
    try:
        from ifcopenshell import ifcopenshell_wrapper
    except ImportError:
        return ()
    chains = []
    for schema_name in ifcopenshell_wrapper.schema_names():
        try:
            decl = ifcopenshell_wrapper.schema_by_name(schema_name).declaration_by_name(ifc_class)
        except Exception:  # noqa: BLE001 -- not declared in this schema
            continue
        names = set()
        decl = decl.supertype() if hasattr(decl, "supertype") else None
        while decl is not None:
            names.add(decl.name())
            decl = decl.supertype() if hasattr(decl, "supertype") else None
        chains.append(frozenset(names))
    return tuple(chains)


def is_subtype(ifc_class: str, ancestor: str) -> bool:
    """True if ``ifc_class`` is ``ancestor`` or inherits from it in some IFC schema."""
    return ifc_class == ancestor or any(ancestor in chain for chain in supertypes(ifc_class))
