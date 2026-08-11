#!/usr/bin/env python3
"""Build the AES North Carolina bank map book using a 3-mile radius."""

from __future__ import annotations

import sys
import types
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent


def load_transformed_module(name: str, path: Path, replacements: list[tuple[str, str]]):
    source = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in source:
            print(f"Warning: transformation text not found in {path.name}: {old!r}", flush=True)
        source = source.replace(old, new)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = None
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


nc = load_transformed_module(
    "nc_bank_map",
    TOOLS_DIR / "nc_bank_map.py",
    [
        ("RADIUS_MILES = 1.5", "RADIUS_MILES = 3.0"),
        ('TODAY_LABEL = "August 10, 2026"', 'TODAY_LABEL = "August 11, 2026"'),
        ("AES_NC_Stores_Nearby_Banks_Detailed_Map.pdf", "AES_NC_Stores_Nearby_Banks_Detailed_Map_3mi.pdf"),
        ("AES_NC_Stores_Nearby_Banks_1.5mi.csv", "AES_NC_Stores_Nearby_Banks_3mi.csv"),
        ("within 1.5 miles", "within 3 miles"),
        ("1.5-mile radius", "3-mile radius"),
        ("the red dashed ring is 1.5 miles", "the red dashed ring is 3 miles"),
        ("radius_miles = 1.5", "radius_miles = 3.0"),
    ],
)

tv = load_transformed_module(
    "build_with_tiger_vector",
    TOOLS_DIR / "build_with_tiger_vector.py",
    [
        ("LOCAL_HALF_HEIGHT_MILES = 1.88", "LOCAL_HALF_HEIGHT_MILES = 3.42"),
    ],
)


def load_roads_fixed(bbox):
    layers = {
        "primary": 0,
        "secondary": 1,
        "local": 2,
        "rail": 3,
    }
    output = {}
    for kind, layer in layers.items():
        try:
            data = tv.query_geojson(tv.ROAD_BASE, layer, bbox, "*")
            output[kind] = data["features"]
        except Exception as exc:
            if kind == "rail":
                print(f"Rail layer skipped: {exc}", flush=True)
                output[kind] = []
            else:
                raise
    return output


tv.load_roads = load_roads_fixed
nc.create_store_map = tv.create_store_map_vector
nc.create_overview_map = tv.create_overview_vector

if __name__ == "__main__":
    nc.main()
