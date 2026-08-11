#!/usr/bin/env python3
"""Compatibility wrapper for the TIGERweb street-map build."""

from __future__ import annotations

import nc_bank_map as nc
import build_with_tiger_vector as tv


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
