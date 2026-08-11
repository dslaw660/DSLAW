#!/usr/bin/env python3
"""Build the NC bank map book with a detailed, public-domain TIGERweb street basemap."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.patheffects as path_effects
import numpy as np

import nc_bank_map as nc

ROAD_BASE = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Transportation_LargeScale/MapServer"
HYDRO_BASE = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Hydro/MapServer"
STATE_BASE = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer"

LOCAL_DISPLAY_ASPECT = 7.35 / 6.98
LOCAL_HALF_HEIGHT_MILES = 1.88


def local_bbox(lat: float, lon: float, half_height_miles: float = LOCAL_HALF_HEIGHT_MILES,
               aspect: float = LOCAL_DISPLAY_ASPECT) -> tuple[float, float, float, float]:
    half_width_miles = half_height_miles * aspect
    dlat = half_height_miles / 69.0
    dlon = half_width_miles / (69.0 * max(0.2, math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def query_geojson(base: str, layer: int, bbox: tuple[float, float, float, float],
                  out_fields: str = "*") -> dict[str, Any]:
    west, south, east, north = bbox
    params = {
        "where": "1=1",
        "geometry": f"{west:.8f},{south:.8f},{east:.8f},{north:.8f}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "geometryPrecision": "7",
        "resultRecordCount": "100000",
        "f": "geojson",
    }
    data = nc.request_get(f"{base}/{layer}/query", params=params, timeout=75, attempts=4).json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"TIGERweb query error at layer {layer}: {data['error']}")
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        raise RuntimeError(f"Unexpected TIGERweb response at layer {layer}")
    return data


def line_parts(geometry: dict[str, Any] | None) -> list[list[tuple[float, float]]]:
    if not geometry:
        return []
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "LineString":
        return [[(float(x), float(y)) for x, y, *_ in coords]]
    if kind == "MultiLineString":
        return [[(float(x), float(y)) for x, y, *_ in part] for part in coords]
    return []


def polygon_parts(geometry: dict[str, Any] | None) -> list[list[tuple[float, float]]]:
    if not geometry:
        return []
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        return [[(float(x), float(y)) for x, y, *_ in ring] for ring in coords[:1]]
    if kind == "MultiPolygon":
        return [[(float(x), float(y)) for x, y, *_ in polygon[0]] for polygon in coords if polygon]
    return []


def feature_name(feature: dict[str, Any]) -> str:
    p = feature.get("properties") or {}
    for key in ("NAME", "BASENAME", "FULLNAME"):
        value = str(p.get(key) or "").strip()
        if value and value.lower() not in {"none", "null"}:
            return value
    return ""


def physical_length(points: list[tuple[float, float]], lat0: float) -> float:
    if len(points) < 2:
        return 0.0
    coslat = math.cos(math.radians(lat0))
    return sum(math.hypot((x2 - x1) * coslat, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def midpoint_and_angle(points: list[tuple[float, float]], lat0: float) -> tuple[float, float, float] | None:
    if len(points) < 2:
        return None
    coslat = math.cos(math.radians(lat0))
    lengths = [math.hypot((x2 - x1) * coslat, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:])]
    total = sum(lengths)
    if total <= 0:
        return None
    target = total / 2
    run = 0.0
    for i, seglen in enumerate(lengths):
        if run + seglen >= target:
            frac = (target - run) / max(seglen, 1e-12)
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            x = x1 + frac * (x2 - x1)
            y = y1 + frac * (y2 - y1)
            dx = (x2 - x1) * coslat
            dy = y2 - y1
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            return x, y, angle
        run += seglen
    return None


def draw_hydro(ax: Any, bbox: tuple[float, float, float, float]) -> None:
    try:
        areas = query_geojson(HYDRO_BASE, 1, bbox, "NAME")
        for feature in areas["features"]:
            for ring in polygon_parts(feature.get("geometry")):
                if len(ring) >= 3:
                    xs, ys = zip(*ring)
                    ax.fill(xs, ys, color="#DCECF7", edgecolor="#B8D6E8", linewidth=0.35, zorder=0.8)
    except Exception as exc:
        print(f"Hydro polygon query skipped: {exc}")
    try:
        lines = query_geojson(HYDRO_BASE, 0, bbox, "NAME")
        for feature in lines["features"]:
            for points in line_parts(feature.get("geometry")):
                if len(points) >= 2:
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, color="#9EC6DF", linewidth=0.65, zorder=1.0)
    except Exception as exc:
        print(f"Hydro line query skipped: {exc}")


def load_roads(bbox: tuple[float, float, float, float]) -> dict[str, list[dict[str, Any]]]:
    layers = {
        "primary": 0,
        "secondary": 1,
        "local": 2,
        "rail": 3,
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for kind, layer in layers.items():
        data = query_geojson(ROAD_BASE, layer, bbox, "NAME,BASENAME,RTTYP")
        output[kind] = data["features"]
    return output


def draw_roads(ax: Any, roads: dict[str, list[dict[str, Any]]], lat0: float,
               bbox: tuple[float, float, float, float]) -> None:
    styles = {
        "local": ("#C4C7CB", 0.72, 2.0, None),
        "rail": ("#777C82", 0.85, 2.2, (0, (4, 3))),
        "secondary": ("#D6A352", 1.45, 2.8, None),
        "primary": ("#BF5B39", 2.25, 3.2, None),
    }
    label_candidates: dict[tuple[str, str], tuple[float, list[tuple[float, float]]]] = {}

    for kind in ("local", "rail", "secondary", "primary"):
        color, width, zorder, linestyle = styles[kind]
        for feature in roads.get(kind, []):
            name = feature_name(feature)
            for points in line_parts(feature.get("geometry")):
                if len(points) < 2:
                    continue
                xs, ys = zip(*points)
                ax.plot(xs, ys, color=color, linewidth=width, linestyle=linestyle or "solid",
                        solid_capstyle="round", zorder=zorder)
                if name and kind != "rail":
                    length = physical_length(points, lat0)
                    key = (kind, name.casefold())
                    if key not in label_candidates or length > label_candidates[key][0]:
                        label_candidates[key] = (length, points)

    west, south, east, north = bbox
    physical_width = (east - west) * math.cos(math.radians(lat0))
    physical_height = north - south
    placed: list[tuple[float, float]] = []
    priorities = {"primary": 0, "secondary": 1, "local": 2}
    candidates = []
    for (kind, _), (length, points) in label_candidates.items():
        name = feature_name({"properties": {"NAME": ""}})  # placeholder to satisfy type checker
        candidates.append((priorities[kind], -length, kind, points))

    # Rebuild while retaining the original display name.
    candidates = []
    for (kind, folded), (length, points) in label_candidates.items():
        display = ""
        for feature in roads.get(kind, []):
            candidate = feature_name(feature)
            if candidate.casefold() == folded:
                display = candidate
                break
        if display:
            candidates.append((priorities[kind], -length, kind, display, points))
    candidates.sort()

    max_labels = {"primary": 14, "secondary": 22, "local": 34}
    used = defaultdict(int)
    for _priority, neg_length, kind, name, points in candidates:
        if used[kind] >= max_labels[kind]:
            continue
        length = -neg_length
        min_length = {"primary": 0.0015, "secondary": 0.0011, "local": 0.0015}[kind]
        if length < min_length:
            continue
        placement = midpoint_and_angle(points, lat0)
        if placement is None:
            continue
        x, y, angle = placement
        nx = (x - west) * math.cos(math.radians(lat0)) / max(physical_width, 1e-9)
        ny = (y - south) / max(physical_height, 1e-9)
        min_sep = 0.060 if kind == "local" else 0.052
        if any(math.hypot(nx - px, ny - py) < min_sep for px, py in placed):
            continue
        placed.append((nx, ny))
        used[kind] += 1
        fontsize = {"local": 4.6, "secondary": 5.35, "primary": 5.8}[kind]
        color = {"local": "#545A61", "secondary": "#7B5427", "primary": "#74331F"}[kind]
        text = ax.text(x, y, name, fontsize=fontsize, color=color, ha="center", va="center",
                       rotation=angle, rotation_mode="anchor", zorder=6.0, clip_on=True)
        text.set_path_effects([path_effects.withStroke(linewidth=1.8, foreground="white", alpha=0.95)])


def create_store_map_vector(store: nc.Store, store_matches: list[nc.Match], out_path: Path):
    assert store.lat is not None and store.lon is not None
    bbox = local_bbox(store.lat, store.lon)
    print(f"  Loading TIGERweb streets for store {store.store}", flush=True)
    roads = load_roads(bbox)

    fig, ax = nc.plt.subplots(figsize=(7.35, 6.98), dpi=195)
    ax.set_facecolor("#F7F5ED")
    draw_hydro(ax, bbox)
    draw_roads(ax, roads, store.lat, bbox)

    circle_x, circle_y = nc.circle_points(store.lat, store.lon, nc.RADIUS_MILES)
    ax.fill(circle_x, circle_y, color="#B42318", alpha=0.055, zorder=3.4)
    ax.plot(circle_x, circle_y, color="#B42318", linewidth=2.0, linestyle=(0, (5, 3)), zorder=7)

    for match in store_matches:
        nc.add_number_marker(ax, match.branch.lon, match.branch.lat, match.number, "#1769AA", size=175)

    ax.scatter([store.lon], [store.lat], s=300, c=["#B42318"], marker="*",
               edgecolors="white", linewidths=1.9, zorder=12)
    ax.annotate(f"AES {store.store}", (store.lon, store.lat), xytext=(9, 9), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.4, weight="bold", color="#7A1111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B42318", alpha=0.96), zorder=13)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#BFC9D6")
        spine.set_linewidth(0.8)

    ax.annotate("N", xy=(0.955, 0.945), xycoords="axes fraction", ha="center", va="center",
                fontsize=10, weight="bold", color="#152238", zorder=20)
    ax.annotate("", xy=(0.955, 0.925), xytext=(0.955, 0.855), xycoords="axes fraction",
                arrowprops=dict(facecolor="#152238", edgecolor="#152238", width=2.5, headwidth=9), zorder=20)

    scale_miles = 0.5
    scale_dlon = scale_miles / (69.0 * math.cos(math.radians(store.lat)))
    x0 = bbox[0] + (bbox[2] - bbox[0]) * 0.06
    y0 = bbox[1] + (bbox[3] - bbox[1]) * 0.055
    ax.plot([x0, x0 + scale_dlon], [y0, y0], color="#152238", linewidth=4,
            solid_capstyle="butt", zorder=20)
    tick = (bbox[3] - bbox[1]) * 0.008
    ax.plot([x0, x0], [y0 - tick, y0 + tick], color="#152238", linewidth=1.2, zorder=20)
    ax.plot([x0 + scale_dlon, x0 + scale_dlon], [y0 - tick, y0 + tick],
            color="#152238", linewidth=1.2, zorder=20)
    ax.text(x0 + scale_dlon / 2, y0 + (bbox[3] - bbox[1]) * 0.015, "0.5 mi",
            ha="center", va="bottom", fontsize=7, color="#152238",
            bbox=dict(fc="white", ec="none", alpha=0.86, pad=1.2), zorder=20)

    ax.text(0.995, 0.006, "Roads and hydrography: U.S. Census Bureau TIGERweb (2025)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.1, color="#354052",
            bbox=dict(fc="white", ec="none", alpha=0.80, pad=1.1), zorder=25)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=195, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    nc.plt.close(fig)
    return (*bbox, "U.S. Census Bureau TIGERweb transportation and hydrography")


def query_nc_state() -> dict[str, Any]:
    params = {
        "where": "STUSAB='NC'",
        "outFields": "STUSAB,NAME,BASENAME",
        "returnGeometry": "true",
        "outSR": "4326",
        "geometryPrecision": "5",
        "f": "geojson",
    }
    data = nc.request_get(f"{STATE_BASE}/0/query", params=params, timeout=75, attempts=4).json()
    if data.get("error") or not data.get("features"):
        raise RuntimeError(f"Could not retrieve NC state outline: {data.get('error')}")
    return data


def create_overview_vector(stores: list[nc.Store], out_path: Path) -> str:
    state = query_nc_state()
    fig, ax = nc.plt.subplots(figsize=(10.1, 4.35), dpi=195)
    ax.set_facecolor("#EEF4F8")
    for feature in state["features"]:
        for ring in polygon_parts(feature.get("geometry")):
            if len(ring) >= 3:
                xs, ys = zip(*ring)
                ax.fill(xs, ys, color="#F4F0E4", edgecolor="#7D8791", linewidth=1.0, zorder=1)
    for store in stores:
        assert store.lat is not None and store.lon is not None
        ax.scatter([store.lon], [store.lat], s=90, c=["#B42318"], edgecolors="white", linewidths=1.1, zorder=8)
        ax.annotate(store.store, (store.lon, store.lat), xytext=(4, 3), textcoords="offset points",
                    fontsize=6.6, weight="bold", color="#7A1111",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.88), zorder=9)
    ax.set_xlim(-84.45, -75.25)
    ax.set_ylim(33.72, 36.72)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_edgecolor("#BFC9D6")
        spine.set_linewidth(0.8)
    ax.text(-79.05, 35.15, "NORTH CAROLINA", fontsize=18, weight="bold", color="#B4BAC1",
            ha="center", va="center", zorder=2)
    ax.annotate("N", xy=(0.968, 0.91), xycoords="axes fraction", ha="center", va="center", fontsize=10, weight="bold")
    ax.annotate("", xy=(0.968, 0.89), xytext=(0.968, 0.76), xycoords="axes fraction",
                arrowprops=dict(facecolor="#152238", edgecolor="#152238", width=2.5, headwidth=9))
    ax.text(0.995, 0.008, "State boundary: U.S. Census Bureau TIGERweb (2025)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.2, color="#354052",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=195, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    nc.plt.close(fig)
    return "U.S. Census Bureau TIGERweb state boundary"


nc.create_store_map = create_store_map_vector
nc.create_overview_map = create_overview_vector

if __name__ == "__main__":
    nc.main()
