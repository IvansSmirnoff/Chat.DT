#!/usr/bin/env python3
"""Analyze IFC file to understand its structure."""

import ifcopenshell
import ifcopenshell.util.element

ifc = ifcopenshell.open('data/model.ifc')

print('=== IFC SCHEMA ===')
print(f'Schema: {ifc.schema}')

print('\n=== ENTITY COUNTS ===')
entity_types = {}
for element in ifc.by_type('IfcRoot'):
    etype = element.is_a()
    entity_types[etype] = entity_types.get(etype, 0) + 1

for etype, count in sorted(entity_types.items(), key=lambda x: -x[1])[:25]:
    print(f'{etype}: {count}')

print('\n=== RELATIONSHIPS ===')
print(f'IfcRelContainedInSpatialStructure: {len(ifc.by_type("IfcRelContainedInSpatialStructure"))}')
print(f'IfcRelAggregates: {len(ifc.by_type("IfcRelAggregates"))}')
print(f'IfcRelAssociatesMaterial: {len(ifc.by_type("IfcRelAssociatesMaterial"))}')

print('\n=== BUILDING STOREYS ===')
for storey in ifc.by_type('IfcBuildingStorey'):
    print(f'  Name: "{storey.Name}"')

print('\n=== SAMPLE PROPERTIES BY ENTITY ===')
for etype in ['IfcWall', 'IfcDoor', 'IfcColumn', 'IfcSpace', 'IfcWindow']:
    elements = ifc.by_type(etype)
    if elements:
        elem = elements[0]
        psets = ifcopenshell.util.element.get_psets(elem, psets_only=False)
        print(f'\n{etype} ({len(elements)} total):')
        all_props = {}
        for pset_name, props in psets.items():
            if isinstance(props, dict):
                for k, v in props.items():
                    if v is not None and k != 'id':
                        all_props[k] = v
        for k, v in sorted(all_props.items())[:15]:
            print(f'  {k}: {repr(v)[:50]}')

print('\n=== MATERIAL ASSOCIATIONS ===')
mat_rels = ifc.by_type("IfcRelAssociatesMaterial")
if mat_rels:
    for rel in mat_rels[:5]:
        material = rel.RelatingMaterial
        mat_name = None
        if material.is_a("IfcMaterial"):
            mat_name = material.Name
        elif material.is_a("IfcMaterialLayerSetUsage"):
            ls = material.ForLayerSet
            if ls and ls.MaterialLayers:
                mat_name = ls.MaterialLayers[0].Material.Name if ls.MaterialLayers[0].Material else None
        elif material.is_a("IfcMaterialLayerSet"):
            if material.MaterialLayers:
                mat_name = material.MaterialLayers[0].Material.Name if material.MaterialLayers[0].Material else None
        
        related_types = set(obj.is_a() for obj in rel.RelatedObjects[:5])
        print(f'  Material: "{mat_name}" -> {related_types}')
else:
    print('  No material associations found')
