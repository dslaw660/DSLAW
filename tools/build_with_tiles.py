#!/usr/bin/env python3
"""Run the NC bank map build with stitched raster street-map tiles."""

from __future__ import annotations

import io
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import nc_bank_map as nc

LOCAL_DISPLAY_ASPECT = 7.35 / 6.98
LOCAL_HALF_HEIGHT_MILES = 1.85
TILE_SIZE = 256
CACHE_LOCK = threading.Lock()
TILE_CACHE: dict[tuple[str, int, int, int], Image.Image] = {}


def world_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = max(-85.05112878, min(85.05112878, lat))
    scale = TILE_SIZE * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return x, y


def map_bbox_fixed(lat: float, lon: float, half_height_miles: float = LOCAL_HALF_HEIGHT_MILES,
                   aspect: float = LOCAL_DISPLAY_ASPECT) -> tuple[float, float, float, float]:
    half_width_miles = half_height_miles * aspect
    dlat = half_height_miles / 69.0
    dlon = half_width_miles / (69.0 * max(0.2, math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def tile_providers(overview: bool) -> list[tuple[str, str, str]]:
    service = "World_Topo_Map" if overview else "World_Street_Map"
    label = "Esri World Topographic Map (raster tiles)" if overview else "Esri World Street Map (raster tiles)"
    return [
        (label, f"https://server.arcgisonline.com/ArcGIS/rest/services/{service}/MapServer/tile/{{z}}/{{y}}/{{x}}", "esri"),
        (label, f"https://services.arcgisonline.com/ArcGIS/rest/services/{service}/MapServer/tile/{{z}}/{{y}}/{{x}}", "esri"),
        ("OpenStreetMap standard tiles", "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "osm"),
    ]


def fetch_tile(provider_label: str, template: str, zoom: int, x: int, y: int) -> Image.Image:
    n = 2 ** zoom
    x = x % n
    if y < 0 or y >= n:
        return Image.new("RGB", (TILE_SIZE, TILE_SIZE), "white")
    cache_key = (template, zoom, x, y)
    with CACHE_LOCK:
        cached = TILE_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()
    url = template.format(z=zoom, x=x, y=y)
    response = nc.request_get(url, timeout=35, attempts=3, expected="image")
    with Image.open(io.BytesIO(response.content)) as src:
        tile = src.convert("RGB")
    if tile.size != (TILE_SIZE, TILE_SIZE):
        tile = tile.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
    with CACHE_LOCK:
        TILE_CACHE[cache_key] = tile.copy()
    return tile


def build_from_provider(bbox: tuple[float, float, float, float], width: int, height: int,
                        zoom: int, provider_label: str, template: str) -> Image.Image:
    west, south, east, north = bbox
    left_px, top_px = world_pixel(west, north, zoom)
    right_px, bottom_px = world_pixel(east, south, zoom)
    x0 = math.floor(left_px / TILE_SIZE)
    x1 = math.floor((right_px - 1e-6) / TILE_SIZE)
    y0 = math.floor(top_px / TILE_SIZE)
    y1 = math.floor((bottom_px - 1e-6) / TILE_SIZE)
    coords = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    if len(coords) > 120:
        raise RuntimeError(f"Refusing oversized tile request: {len(coords)} tiles at zoom {zoom}")

    mosaic = Image.new("RGB", ((x1 - x0 + 1) * TILE_SIZE, (y1 - y0 + 1) * TILE_SIZE), "white")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_tile, provider_label, template, zoom, x, y): (x, y) for x, y in coords}
        for future in as_completed(futures):
            x, y = futures[future]
            tile = future.result()
            mosaic.paste(tile, ((x - x0) * TILE_SIZE, (y - y0) * TILE_SIZE))

    crop = (
        max(0, int(round(left_px - x0 * TILE_SIZE))),
        max(0, int(round(top_px - y0 * TILE_SIZE))),
        min(mosaic.width, int(round(right_px - x0 * TILE_SIZE))),
        min(mosaic.height, int(round(bottom_px - y0 * TILE_SIZE))),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise RuntimeError(f"Invalid tile crop bounds: {crop}")
    image = mosaic.crop(crop)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def image_has_detail(image: Image.Image) -> bool:
    arr = np.asarray(image.convert("RGB").resize((250, 200)))
    # A blank service image has very little edge energy and very few unique colors.
    gray = arr.mean(axis=2)
    edge_energy = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
    unique = len(np.unique(arr.reshape(-1, 3), axis=0))
    return edge_energy > 2.0 and unique > 300


def download_basemap_tiles(bbox: tuple[float, float, float, float], out_path: Path,
                           *, overview: bool = False, width: int = 1800, height: int = 1500) -> str:
    zoom = 8 if overview else 15
    errors: list[str] = []
    for provider_label, template, _kind in tile_providers(overview):
        try:
            image = build_from_provider(bbox, width, height, zoom, provider_label, template)
            if not image_has_detail(image):
                raise RuntimeError("provider returned a visually blank or low-detail image")
            image.save(out_path, format="PNG", optimize=True)
            return provider_label
        except Exception as exc:
            errors.append(f"{provider_label}: {exc}")
    raise RuntimeError("All basemap tile providers failed: " + " | ".join(errors))


def create_store_map_fixed(store: nc.Store, store_matches: list[nc.Match], out_path: Path):
    assert store.lat is not None and store.lon is not None
    bbox = map_bbox_fixed(store.lat, store.lon)
    base_path = out_path.with_name(out_path.stem + "_base.png")
    basemap_source = download_basemap_tiles(bbox, base_path, width=1650, height=1570)
    img = np.asarray(Image.open(base_path).convert("RGB"))

    fig, ax = nc.plt.subplots(figsize=(7.35, 6.98), dpi=190)
    ax.imshow(img, extent=bbox, origin="upper", zorder=0, interpolation="bilinear", aspect="auto")
    ax.set_aspect("auto")
    circle_x, circle_y = nc.circle_points(store.lat, store.lon, nc.RADIUS_MILES)
    ax.fill(circle_x, circle_y, color="#B42318", alpha=0.08, zorder=2)
    ax.plot(circle_x, circle_y, color="#B42318", linewidth=2.0, linestyle=(0, (5, 3)), zorder=5)

    for match in store_matches:
        nc.add_number_marker(ax, match.branch.lon, match.branch.lat, match.number, "#1769AA")

    ax.scatter([store.lon], [store.lat], s=280, c=["#B42318"], marker="*",
               edgecolors="white", linewidths=1.8, zorder=11)
    ax.annotate(f"AES {store.store}", (store.lon, store.lat), xytext=(9, 9), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, weight="bold", color="#7A1111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B42318", alpha=0.94), zorder=12)

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
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
            bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.2), zorder=20)

    credit = "Basemap © Esri" if basemap_source.startswith("Esri") else "© OpenStreetMap contributors"
    ax.text(0.995, 0.006, credit, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.3,
            color="#354052", bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.1), zorder=25)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=190, bbox_inches="tight", pad_inches=0.01)
    nc.plt.close(fig)
    base_path.unlink(missing_ok=True)
    return (*bbox, basemap_source)


nc.download_basemap = download_basemap_tiles
nc.create_store_map = create_store_map_fixed

if __name__ == "__main__":
    nc.main()
