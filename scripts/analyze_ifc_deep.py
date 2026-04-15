#!/usr/bin/env python3
"""Deep analysis of IFC file for specific properties."""

import ifcopenshell
import ifcopenshell.util.element

ifc = ifcopenshell.open('data/model.ifc')

print('=== CHECKING FOR FIRE RATING ===')
for etype in ['IfcDoor', 'IfcWall', 'IfcWindow']:
    elements = ifc.by_type(etype)
    fire_rating_count = 0
    sample_rating = None
    for elem in elements:
        psets = ifcopenshell.util.element.get_psets(elem, psets_only=False)
        for pset_name, props in psets.items():
            if isinstance(props, dict):
                for k, v in props.items():
                    if 'fire' in k.lower() or 'rating' in k.lower():
                        if v is not None:
                            fire_rating_count += 1
                            sample_rating = (k, v)
                            break
    print(f'{etype}: {fire_rating_count}/{len(elements)} have fire-related properties')
    if sample_rating:
        print(f'  Sample: {sample_rating}')

print('\n=== DETAILED MATERIAL CHECK ===')
mat_rels = ifc.by_type("IfcRelAssociatesMaterial")
material_to_elements = {}
for rel in mat_rels:
    material = rel.RelatingMaterial
    mat_name = "Unknown"
    
    if material.is_a("IfcMaterial"):
        mat_name = material.Name if material.Name else "Unnamed"
    elif material.is_a("IfcMaterialLayerSetUsage"):
        ls = material.ForLayerSet
        if ls and ls.MaterialLayers:
            layers = []
            for layer in ls.MaterialLayers:
                if layer.Material:
                    layers.append(layer.Material.Name)
            mat_name = " + ".join(layers) if layers else "Unnamed LayerSet"
    elif material.is_a("IfcMaterialLayerSet"):
        if material.MaterialLayers:
            layers = []
            for layer in material.MaterialLayers:
                if layer.Material:
                    layers.append(layer.Material.Name)
            mat_name = " + ".join(layers) if layers else "Unnamed LayerSet"
    elif material.is_a("IfcMaterialConstituentSet"):
        if hasattr(material, 'MaterialConstituents') and material.MaterialConstituents:
            constituents = []
            for const in material.MaterialConstituents:
                if const.Material:
                    constituents.append(const.Material.Name)
            mat_name = " + ".join(constituents) if constituents else "Unnamed ConstituentSet"
    
    for obj in rel.RelatedObjects:
        etype = obj.is_a()
        if mat_name not in material_to_elements:
            material_to_elements[mat_name] = {}
        if etype not in material_to_elements[mat_name]:
            material_to_elements[mat_name][etype] = 0
        material_to_elements[mat_name][etype] += 1

print('Materials found:')
for mat_name, etypes in sorted(material_to_elements.items()):
    print(f'\n  "{mat_name}":')
    for etype, count in sorted(etypes.items()):
        print(f'    {etype}: {count}')

print('\n=== COLUMN HEIGHTS (Length property) ===')
columns = ifc.by_type('IfcColumn')
heights = []
for col in columns[:10]:
    psets = ifcopenshell.util.element.get_psets(col, psets_only=False)
    for pset_name, props in psets.items():
        if isinstance(props, dict):
            if 'Length' in props:
                heights.append(props['Length'])
                break
print(f'Sample column heights (mm): {heights}')
print(f'In meters: {[h/1000 for h in heights]}')

print('\n=== WALL WIDTHS ===')
walls = ifc.by_type('IfcWall')
widths = []
for wall in walls[:10]:
    psets = ifcopenshell.util.element.get_psets(wall, psets_only=False)
    for pset_name, props in psets.items():
        if isinstance(props, dict):
            if 'Width' in props:
                widths.append(props['Width'])
                break
print(f'Sample wall widths (mm): {widths}')
print(f'In meters: {[w/1000 for w in widths]}')

print('\n=== SPACE AREAS ===')
spaces = ifc.by_type('IfcSpace')
areas = []
for space in spaces[:10]:
    psets = ifcopenshell.util.element.get_psets(space, psets_only=False)
    for pset_name, props in psets.items():
        if isinstance(props, dict):
            if 'GrossFloorArea' in props:
                areas.append(props['GrossFloorArea'])
                break
print(f'Sample space GrossFloorArea: {areas}')
