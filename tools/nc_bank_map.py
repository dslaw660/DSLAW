#!/usr/bin/env python3
"""Build a static PDF map book for selected AES North Carolina stores and nearby banks."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
MAP_DIR = OUTPUT_DIR / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

RADIUS_MILES = 1.5
EARTH_RADIUS_MILES = 3958.7613
FULL_SERVICE_CODES = {"11", "12"}
USER_AGENT = "AES-Restaurant-Group-NC-Bank-Map/1.0 (business map exhibit)"
TODAY_LABEL = "August 10, 2026"

FDIC_API = "https://banks.data.fdic.gov/api/locations"
ARCGIS_BANK_QUERY = "https://gis.vta.org/gis/rest/services/Insured_Banks_and_Credit_Unions/MapServer/0/query"
ARCGIS_GEOCODER = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
CENSUS_GEOCODER = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
ESRI_STREET_EXPORTS = [
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/export",
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/export",
]
ESRI_TOPO_EXPORTS = [
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/export",
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/export",
]

STORES_RAW = [
    ("8026", "1130 West 15th Street", "Washington", "NC", "27889"),
    ("9010", "8105 Richlands Hwy", "Richlands", "NC", "28574"),
    ("8007", "408 S. Marine Blvd.", "Jacksonville", "NC", "28540"),
    ("1889", "3209 Taylorsville Road", "Statesville", "NC", "28625"),
    ("6425", "1808 NC 67 Highway", "Jonesville", "NC", "28642"),
    ("6408", "1319 Bridford Parkway", "Greensboro", "NC", "27407"),
    ("6151", "6308 Capital Blvd.", "Raleigh", "NC", "27616"),
    ("6445", "4173 W Vernon Avenue", "Kinston", "NC", "28504"),
    ("8116", "1901 E Cone Blvd.", "Greensboro", "NC", "27405"),
    ("149", "3802 South Holden Rd.", "Greensboro", "NC", "27406"),
    ("8084", "1314 Mebane Oaks Road", "Mebane", "NC", "27302"),
]


@dataclass
class Store:
    store: str
    address: str
    city: str
    state: str
    zip: str
    lat: float | None = None
    lon: float | None = None
    geocode_label: str = ""
    geocode_score: float | None = None
    geocode_source: str = ""

    @property
    def full_address(self) -> str:
        return f"{self.address}, {self.city}, {self.state} {self.zip}"


@dataclass
class Branch:
    name: str
    office: str
    address: str
    city: str
    state: str
    zip: str
    lat: float
    lon: float
    run_date: str = ""
    service_code: str = ""
    service_desc: str = ""
    source: str = ""

    @property
    def full_address(self) -> str:
        parts = [self.address, self.city, self.state, self.zip]
        return ", ".join(p for p in parts if p)


@dataclass
class Match:
    store: Store
    branch: Branch
    distance: float
    number: int = 0


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})


def request_get(url: str, *, params: dict[str, Any] | None = None, timeout: int = 40,
                attempts: int = 4, expected: str | None = None) -> requests.Response:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = SESSION.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            if expected and expected not in response.headers.get("content-type", "").lower():
                raise RuntimeError(f"Unexpected content type {response.headers.get('content-type')} from {url}")
            return response
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {attempts} attempts: {url}: {last}")


def clean_zip(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)
    digits = re.sub(r"\D", "", text)
    return digits[:5].zfill(5) if digits else text


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def normalize_date(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    try:
        n = float(text)
        if n > 100000000000:
            return datetime.fromtimestamp(n / 1000, tz=timezone.utc).date().isoformat()
        if n > 1000000000:
            return datetime.fromtimestamp(n, tz=timezone.utc).date().isoformat()
    except ValueError:
        pass
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def geocode_arcgis(address: str) -> tuple[float, float, str, float] | None:
    params = {
        "SingleLine": address,
        "f": "json",
        "outFields": "Match_addr,Addr_type,Postal,City,Region",
        "maxLocations": "5",
        "countryCode": "USA",
    }
    data = request_get(ARCGIS_GEOCODER, params=params, timeout=35).json()
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    best = max(candidates, key=lambda c: float(c.get("score") or 0))
    loc = best.get("location") or {}
    lat, lon = to_float(loc.get("y")), to_float(loc.get("x"))
    if lat is None or lon is None:
        return None
    return lat, lon, str(best.get("address") or address), float(best.get("score") or 0)


def geocode_census(address: str) -> tuple[float, float, str, float] | None:
    params = {"address": address, "benchmark": "Public_AR_Current", "format": "json"}
    data = request_get(CENSUS_GEOCODER, params=params, timeout=35).json()
    matches = (((data.get("result") or {}).get("addressMatches")) or [])
    if not matches:
        return None
    best = matches[0]
    coords = best.get("coordinates") or {}
    lat, lon = to_float(coords.get("y")), to_float(coords.get("x"))
    if lat is None or lon is None:
        return None
    return lat, lon, str(best.get("matchedAddress") or address), 100.0


def geocode_stores(stores: list[Store]) -> None:
    for index, store in enumerate(stores, start=1):
        print(f"Geocoding store {index}/{len(stores)}: {store.full_address}", flush=True)
        result = None
        try:
            result = geocode_arcgis(store.full_address)
            if result:
                store.geocode_source = "ArcGIS World Geocoder"
        except Exception as exc:
            print(f"  ArcGIS geocode failed: {exc}", file=sys.stderr)
        if not result:
            try:
                result = geocode_census(store.full_address)
                if result:
                    store.geocode_source = "U.S. Census Geocoder"
            except Exception as exc:
                print(f"  Census geocode failed: {exc}", file=sys.stderr)
        if not result:
            raise RuntimeError(f"Could not geocode store {store.store}: {store.full_address}")
        store.lat, store.lon, store.geocode_label, store.geocode_score = result
        print(f"  -> {store.lat:.7f}, {store.lon:.7f}; score {store.geocode_score:.1f}", flush=True)
        time.sleep(0.12)


def normalize_branch_record(raw: dict[str, Any], source: str) -> Branch | None:
    r = raw.get("data") or raw.get("attributes") or raw
    geometry = raw.get("geometry") or {}
    lat = to_float(r.get("LATITUDE") if "LATITUDE" in r else geometry.get("y"))
    lon = to_float(r.get("LONGITUDE") if "LONGITUDE" in r else geometry.get("x"))
    if lat is None or lon is None:
        lat = to_float(geometry.get("y"))
        lon = to_float(geometry.get("x"))
    state = str(r.get("STALP") or r.get("state") or "").strip().upper()
    name = str(r.get("NAME") or r.get("name") or "").strip()
    if not name or state != "NC" or lat is None or lon is None:
        return None
    return Branch(
        name=name,
        office=str(r.get("OFFNAME") or r.get("office") or "").strip(),
        address=str(r.get("ADDRESS") or r.get("address") or "").strip(),
        city=str(r.get("CITY") or r.get("city") or "").strip(),
        state=state,
        zip=clean_zip(r.get("ZIP") or r.get("zip")),
        lat=lat,
        lon=lon,
        run_date=normalize_date(r.get("RUNDATE") or r.get("runDate")),
        service_code=str(r.get("SERVTYPE") or r.get("serviceCode") or "").replace(".0", "").strip(),
        service_desc=str(r.get("SERVTYPE_DESC") or r.get("serviceDesc") or "").strip(),
        source=source,
    )


def load_fdic_branches() -> list[Branch]:
    fields = ",".join([
        "NAME", "OFFNAME", "ADDRESS", "CITY", "STALP", "ZIP", "LATITUDE",
        "LONGITUDE", "RUNDATE", "SERVTYPE", "SERVTYPE_DESC", "UNINUM",
    ])
    params = {
        "filters": "STALP:NC",
        "fields": fields,
        "sort_by": "NAME",
        "sort_order": "ASC",
        "limit": "10000",
        "offset": "0",
        "format": "json",
        "download": "false",
    }
    data = request_get(FDIC_API, params=params, timeout=60).json()
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise RuntimeError("FDIC response did not contain a data array")
    branches = [b for row in rows if (b := normalize_branch_record(row, "FDIC BankFind Suite Locations API"))]
    if not branches:
        raise RuntimeError("FDIC response contained no usable North Carolina coordinates")
    print(f"Loaded {len(branches)} usable North Carolina branches from FDIC", flush=True)
    return branches


def query_arcgis_near_store(store: Store) -> list[Branch]:
    assert store.lat is not None and store.lon is not None
    fields = ",".join([
        "NAME", "OFFNAME", "ADDRESS", "CITY", "STALP", "ZIP", "LATITUDE",
        "LONGITUDE", "RUNDATE", "SERVTYPE", "SERVTYPE_DESC",
    ])
    params = {
        "where": "STALP='NC' AND SERVTYPE IS NOT NULL",
        "geometry": f"{store.lon},{store.lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "distance": str(RADIUS_MILES * 1609.344),
        "units": "esriSRUnit_Meter",
        "outFields": fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
        "resultRecordCount": "2000",
    }
    data = request_get(ARCGIS_BANK_QUERY, params=params, timeout=45).json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    rows = data.get("features") or []
    return [b for row in rows if (b := normalize_branch_record(row, "ArcGIS FDIC branch-layer fallback"))]


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(min(1, math.sqrt(a)))


def physical_full_service(branch: Branch) -> bool:
    code = branch.service_code.strip()
    if code in FULL_SERVICE_CODES:
        return True
    desc = branch.service_desc.lower()
    return "full service" in desc and not any(term in desc for term in ("cyber", "mobile", "home", "phone"))


def deduplicate_branches(branches: Iterable[Branch]) -> list[Branch]:
    best: dict[tuple[str, str, str, str], Branch] = {}
    for b in branches:
        key = (
            re.sub(r"\W+", "", b.name.lower()),
            re.sub(r"\W+", "", b.address.lower()),
            re.sub(r"\W+", "", b.city.lower()),
            b.zip,
        )
        if key not in best:
            best[key] = b
    return list(best.values())


def build_matches(stores: list[Store], all_branches: list[Branch] | None) -> tuple[dict[str, list[Match]], str]:
    matches: dict[str, list[Match]] = {}
    source_label = ""
    for store in stores:
        candidates = all_branches
        if candidates is None:
            candidates = query_arcgis_near_store(store)
        source_label = source_label or (candidates[0].source if candidates else "ArcGIS FDIC branch-layer fallback")
        filtered = deduplicate_branches(b for b in candidates if physical_full_service(b))
        store_matches: list[Match] = []
        assert store.lat is not None and store.lon is not None
        for branch in filtered:
            distance = haversine_miles(store.lat, store.lon, branch.lat, branch.lon)
            if distance <= RADIUS_MILES + 0.0002:
                store_matches.append(Match(store=store, branch=branch, distance=distance))
        store_matches.sort(key=lambda m: (m.distance, m.branch.name.lower(), m.branch.address.lower()))
        for number, match in enumerate(store_matches, start=1):
            match.number = number
        matches[store.store] = store_matches
        print(f"Store {store.store}: {len(store_matches)} full-service branch(es) within {RADIUS_MILES} miles", flush=True)
    return matches, source_label


def circle_points(lat: float, lon: float, radius_miles: float, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * math.pi, n)
    angular = radius_miles / EARTH_RADIUS_MILES
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lats, lons = [], []
    for bearing in angles:
        lat2 = math.asin(math.sin(lat1) * math.cos(angular) + math.cos(lat1) * math.sin(angular) * math.cos(bearing))
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular) * math.cos(lat1),
            math.cos(angular) - math.sin(lat1) * math.sin(lat2),
        )
        lats.append(math.degrees(lat2))
        lons.append(math.degrees(lon2))
    return np.asarray(lons), np.asarray(lats)


def map_bbox(lat: float, lon: float, half_height_miles: float = 2.05,
             aspect: float = 1.18) -> tuple[float, float, float, float]:
    half_width_miles = half_height_miles * aspect
    dlat = half_height_miles / 69.0
    dlon = half_width_miles / (69.0 * max(0.2, math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def download_basemap(bbox: tuple[float, float, float, float], out_path: Path,
                     *, overview: bool = False, width: int = 1800, height: int = 1500) -> str:
    endpoints = ESRI_TOPO_EXPORTS if overview else ESRI_STREET_EXPORTS
    params = {
        "bbox": ",".join(f"{v:.8f}" for v in bbox),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width},{height}",
        "format": "png32",
        "transparent": "false",
        "dpi": "150",
        "f": "image",
    }
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            response = request_get(endpoint, params=params, timeout=70, expected="image", attempts=3)
            out_path.write_bytes(response.content)
            with Image.open(out_path) as test:
                test.verify()
            return "Esri World Topographic Map" if overview else "Esri World Street Map"
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("Could not download basemap: " + " | ".join(errors))


def add_number_marker(ax: Any, x: float, y: float, number: int, color: str, size: int = 155) -> None:
    ax.scatter([x], [y], s=size, c=[color], edgecolors="white", linewidths=1.5, zorder=8)
    ax.annotate(str(number), (x, y), xytext=(0, -0.5), textcoords="offset points",
                ha="center", va="center", color="white", fontsize=7.4, weight="bold", zorder=9)


def create_store_map(store: Store, store_matches: list[Match], out_path: Path) -> tuple[float, float, float, float, str]:
    assert store.lat is not None and store.lon is not None
    bbox = map_bbox(store.lat, store.lon)
    base_path = out_path.with_name(out_path.stem + "_base.png")
    basemap_source = download_basemap(bbox, base_path, width=1770, height=1500)
    img = np.asarray(Image.open(base_path).convert("RGB"))

    fig, ax = plt.subplots(figsize=(8.2, 6.95), dpi=190)
    ax.imshow(img, extent=bbox, origin="upper", zorder=0, interpolation="bilinear")
    circle_x, circle_y = circle_points(store.lat, store.lon, RADIUS_MILES)
    ax.fill(circle_x, circle_y, color="#B42318", alpha=0.08, zorder=2)
    ax.plot(circle_x, circle_y, color="#B42318", linewidth=2.0, linestyle=(0, (5, 3)), zorder=5)

    for match in store_matches:
        add_number_marker(ax, match.branch.lon, match.branch.lat, match.number, "#1769AA")

    ax.scatter([store.lon], [store.lat], s=280, c=["#B42318"], marker="*",
               edgecolors="white", linewidths=1.8, zorder=11)
    ax.annotate(f"AES {store.store}", (store.lon, store.lat), xytext=(9, 9), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, weight="bold", color="#7A1111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B42318", alpha=0.92), zorder=12)

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
    ax.plot([x0, x0 + scale_dlon], [y0, y0], color="#152238", linewidth=4, solid_capstyle="butt", zorder=20)
    ax.plot([x0, x0], [y0 - (bbox[3]-bbox[1])*0.008, y0 + (bbox[3]-bbox[1])*0.008], color="#152238", linewidth=1.2, zorder=20)
    ax.plot([x0 + scale_dlon, x0 + scale_dlon], [y0 - (bbox[3]-bbox[1])*0.008, y0 + (bbox[3]-bbox[1])*0.008], color="#152238", linewidth=1.2, zorder=20)
    ax.text(x0 + scale_dlon / 2, y0 + (bbox[3]-bbox[1])*0.015, "0.5 mi", ha="center", va="bottom",
            fontsize=7, color="#152238", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2), zorder=20)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=190, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    base_path.unlink(missing_ok=True)
    return (*bbox, basemap_source)


def create_overview_map(stores: list[Store], out_path: Path) -> str:
    bbox = (-84.45, 33.72, -75.25, 36.72)
    base_path = out_path.with_name(out_path.stem + "_base.png")
    basemap_source = download_basemap(bbox, base_path, overview=True, width=2100, height=900)
    img = np.asarray(Image.open(base_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(10.1, 4.35), dpi=190)
    ax.imshow(img, extent=bbox, origin="upper", zorder=0)
    for store in stores:
        assert store.lat is not None and store.lon is not None
        ax.scatter([store.lon], [store.lat], s=85, c=["#B42318"], edgecolors="white", linewidths=1.1, zorder=8)
        ax.annotate(store.store, (store.lon, store.lat), xytext=(4, 3), textcoords="offset points",
                    fontsize=6.6, weight="bold", color="#7A1111",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.82), zorder=9)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#BFC9D6")
        spine.set_linewidth(0.8)
    ax.annotate("N", xy=(0.968, 0.91), xycoords="axes fraction", ha="center", va="center", fontsize=10, weight="bold")
    ax.annotate("", xy=(0.968, 0.89), xytext=(0.968, 0.76), xycoords="axes fraction",
                arrowprops=dict(facecolor="#152238", edgecolor="#152238", width=2.5, headwidth=9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=190, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    base_path.unlink(missing_ok=True)
    return basemap_source


NAVY = colors.HexColor("#152238")
BLUE = colors.HexColor("#1769AA")
RED = colors.HexColor("#B42318")
MUTED = colors.HexColor("#5E6B7C")
LIGHT = colors.HexColor("#F4F7FB")
LINE = colors.HexColor("#D7DEE8")
WHITE = colors.white
PAGE_W, PAGE_H = landscape(letter)


def fit_text(c: canvas.Canvas, text: str, font: str, max_size: float, min_size: float, width: float) -> float:
    size = max_size
    while size > min_size and stringWidth(text, font, size) > width:
        size -= 0.25
    return size


def draw_header(c: canvas.Canvas, title: str, subtitle: str = "") -> None:
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 0.78 * inch, PAGE_W, 0.78 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    size = fit_text(c, title, "Helvetica-Bold", 17, 11, PAGE_W - 0.8 * inch)
    c.setFont("Helvetica-Bold", size)
    c.drawString(0.4 * inch, PAGE_H - 0.38 * inch, title)
    if subtitle:
        c.setFont("Helvetica", 8.2)
        c.setFillColor(colors.HexColor("#D9E2EF"))
        c.drawString(0.4 * inch, PAGE_H - 0.62 * inch, subtitle)


def draw_footer(c: canvas.Canvas, page_no: int, text: str) -> None:
    c.setStrokeColor(LINE)
    c.line(0.4 * inch, 0.31 * inch, PAGE_W - 0.4 * inch, 0.31 * inch)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 6.4)
    c.drawString(0.4 * inch, 0.15 * inch, text[:205])
    c.drawRightString(PAGE_W - 0.4 * inch, 0.15 * inch, f"Page {page_no}")


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y_top: float, width: float,
                 font: str = "Helvetica", size: float = 7.0, leading: float | None = None,
                 color: colors.Color = NAVY, max_lines: int | None = None) -> float:
    leading = leading or size * 1.22
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = word if not current else current + " " + word
        if stringWidth(test, font, size) <= width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and stringWidth(last + "...", font, size) > width:
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."
    c.setFont(font, size)
    c.setFillColor(color)
    y = y_top
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def branch_display_name(branch: Branch) -> str:
    if branch.office and branch.office.lower() not in {branch.name.lower(), "main office"}:
        return f"{branch.name} - {branch.office}"
    return branch.name


def draw_branch_block(c: canvas.Canvas, match: Match, x: float, y_top: float, width: float,
                      compact: bool = False) -> float:
    marker_r = 0.115 * inch if not compact else 0.105 * inch
    c.setFillColor(BLUE)
    c.circle(x + marker_r, y_top - marker_r, marker_r, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 6.8 if not compact else 6.2)
    c.drawCentredString(x + marker_r, y_top - marker_r - 2.0, str(match.number))

    tx = x + marker_r * 2 + 0.08 * inch
    tw = width - (tx - x)
    name_size = 7.3 if not compact else 6.7
    y = draw_wrapped(c, branch_display_name(match.branch), tx, y_top - 0.02 * inch, tw,
                     font="Helvetica-Bold", size=name_size, leading=name_size * 1.12,
                     color=NAVY, max_lines=2)
    addr_size = 6.6 if not compact else 6.1
    y = draw_wrapped(c, match.branch.full_address, tx, y - 0.01 * inch, tw,
                     size=addr_size, leading=addr_size * 1.13, color=MUTED, max_lines=2)
    c.setFont("Helvetica-Bold", 6.6 if not compact else 6.1)
    c.setFillColor(RED)
    c.drawString(tx, y - 0.01 * inch, f"{match.distance:.2f} mi straight-line")
    y -= (0.18 if not compact else 0.15) * inch
    c.setStrokeColor(LINE)
    c.line(x, y + 0.03 * inch, x + width, y + 0.03 * inch)
    return y


def create_pdf(stores: list[Store], matches: dict[str, list[Match]], map_paths: dict[str, Path],
               overview_path: Path, pdf_path: Path, source_label: str, bank_run_date: str,
               basemap_source: str) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=landscape(letter), pageCompression=1)
    c.setTitle("AES North Carolina Stores and Nearby Banks - Detailed Map Book")
    c.setAuthor("AES Restaurant Group")
    page_no = 1

    draw_header(c, "AES North Carolina Stores and Nearby Banks",
                "Eleven requested stores; full-service FDIC-insured bank branches within 1.5 miles")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 21)
    c.drawString(0.48 * inch, PAGE_H - 1.26 * inch, "Detailed Map Book")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(0.49 * inch, PAGE_H - 1.52 * inch, f"Prepared {TODAY_LABEL}")
    c.drawImage(str(overview_path), 0.48 * inch, 2.15 * inch, width=10.04 * inch, height=4.36 * inch,
                preserveAspectRatio=False, mask="auto")

    all_matches = [m for values in matches.values() for m in values]
    unique_branches = {
        (m.branch.name.lower(), m.branch.address.lower(), m.branch.city.lower(), m.branch.zip)
        for m in all_matches
    }
    box_y = 1.07 * inch
    metrics = [
        ("11", "AES stores mapped"),
        (str(len(unique_branches)), "unique bank branches"),
        (str(len(all_matches)), "store-branch matches"),
    ]
    for idx, (value, label) in enumerate(metrics):
        x = 0.49 * inch + idx * 2.0 * inch
        c.setFillColor(LIGHT)
        c.roundRect(x, box_y, 1.78 * inch, 0.76 * inch, 6, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(x + 0.89 * inch, box_y + 0.39 * inch, value)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 6.8)
        c.drawCentredString(x + 0.89 * inch, box_y + 0.17 * inch, label.upper())

    note_x = 6.7 * inch
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(note_x, box_y + 0.62 * inch, "Method")
    method = (
        "Branch source: FDIC BankFind Suite Locations API; ArcGIS FDIC branch layer used only as a fallback. "
        "Included offices are full-service physical branches (service codes 11 and 12). Distances are geodesic "
        "straight-line distances from the geocoded store point; the red dashed ring is 1.5 miles."
    )
    draw_wrapped(c, method, note_x, box_y + 0.46 * inch, 3.8 * inch, size=6.8, leading=8.0, color=MUTED)
    footer_source = f"Bank data: {source_label}" + (f"; data run date {bank_run_date}" if bank_run_date else "")
    draw_footer(c, page_no, footer_source + f"; basemap: {basemap_source}.")
    c.showPage()
    page_no += 1

    for store in stores:
        store_matches = matches[store.store]
        draw_header(c, f"Store {store.store} - {store.city}, North Carolina",
                    f"{store.full_address} | {len(store_matches)} qualifying branch(es) within 1.5 miles")
        map_x, map_y, map_w, map_h = 0.35 * inch, 0.55 * inch, 7.35 * inch, 6.98 * inch
        c.setStrokeColor(LINE)
        c.roundRect(map_x - 0.02 * inch, map_y - 0.02 * inch, map_w + 0.04 * inch, map_h + 0.04 * inch,
                    5, fill=0, stroke=1)
        c.drawImage(str(map_paths[store.store]), map_x, map_y, width=map_w, height=map_h,
                    preserveAspectRatio=False, mask="auto")

        panel_x, panel_y, panel_w, panel_h = 7.88 * inch, 0.55 * inch, 2.77 * inch, 6.98 * inch
        c.setFillColor(LIGHT)
        c.roundRect(panel_x, panel_y, panel_w, panel_h, 7, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(panel_x + 0.15 * inch, panel_y + panel_h - 0.28 * inch, "Bank branches")
        c.setFont("Helvetica", 6.8)
        c.setFillColor(MUTED)
        c.drawString(panel_x + 0.15 * inch, panel_y + panel_h - 0.48 * inch, "Numbered markers correspond to the map")

        shown = store_matches[:11]
        y = panel_y + panel_h - 0.71 * inch
        if not shown:
            c.setFillColor(MUTED)
            c.setFont("Helvetica-Oblique", 8)
            draw_wrapped(c, "No qualifying full-service FDIC-insured bank branch was found within the 1.5-mile radius.",
                         panel_x + 0.15 * inch, y, panel_w - 0.3 * inch, font="Helvetica-Oblique",
                         size=8, leading=10, color=MUTED)
        else:
            compact = len(shown) >= 8
            for match in shown:
                y = draw_branch_block(c, match, panel_x + 0.13 * inch, y, panel_w - 0.26 * inch, compact=compact)
                if y < panel_y + 0.18 * inch:
                    break
        if len(store_matches) > len(shown):
            c.setFillColor(RED)
            c.setFont("Helvetica-Bold", 6.6)
            c.drawRightString(panel_x + panel_w - 0.15 * inch, panel_y + 0.12 * inch,
                              f"{len(store_matches)-len(shown)} additional branch(es) on next page")

        footer = f"Bank data: {source_label}"
        if bank_run_date:
            footer += f"; run date {bank_run_date}"
        footer += f"; basemap: {basemap_source}; 1.5-mile radius is straight-line, not driving distance."
        draw_footer(c, page_no, footer)
        c.showPage()
        page_no += 1

        remaining = store_matches[11:]
        while remaining:
            chunk, remaining = remaining[:24], remaining[24:]
            draw_header(c, f"Store {store.store} - Additional Bank Branches",
                        f"Continuation list for {store.full_address}")
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 7)
            c.drawString(0.45 * inch, PAGE_H - 1.05 * inch,
                         "All numbered markers are shown on the preceding detailed map page.")
            col_w = 4.85 * inch
            for col in range(2):
                x = 0.5 * inch + col * 5.25 * inch
                y = PAGE_H - 1.34 * inch
                for match in chunk[col * 12:(col + 1) * 12]:
                    y = draw_branch_block(c, match, x, y, col_w, compact=False)
            draw_footer(c, page_no, footer)
            c.showPage()
            page_no += 1

    c.save()


def write_outputs(stores: list[Store], matches: dict[str, list[Match]], source_label: str,
                  bank_run_date: str, basemap_source: str) -> None:
    csv_path = OUTPUT_DIR / "AES_NC_Stores_Nearby_Banks_1.5mi.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Store #", "Store Address", "Store City", "Store State", "Store ZIP",
            "Store Latitude", "Store Longitude", "Map Marker", "Bank", "Branch",
            "Bank Address", "Bank City", "Bank State", "Bank ZIP", "Distance Miles",
            "Service Type", "Service Description", "Bank Data Run Date", "Source",
        ])
        for store in stores:
            values = matches[store.store]
            if not values:
                writer.writerow([
                    store.store, store.address, store.city, store.state, store.zip,
                    f"{store.lat:.7f}", f"{store.lon:.7f}", "", "", "", "", "", "", "", "",
                    "", "", bank_run_date, source_label,
                ])
            for match in values:
                b = match.branch
                writer.writerow([
                    store.store, store.address, store.city, store.state, store.zip,
                    f"{store.lat:.7f}", f"{store.lon:.7f}", match.number, b.name, b.office,
                    b.address, b.city, b.state, b.zip, f"{match.distance:.3f}",
                    b.service_code, b.service_desc, b.run_date or bank_run_date, b.source,
                ])

    store_json = []
    for store in stores:
        row = asdict(store)
        row["full_address"] = store.full_address
        row["matches"] = [
            {"number": m.number, "distance_miles": round(m.distance, 4), "branch": asdict(m.branch)}
            for m in matches[store.store]
        ]
        store_json.append(row)
    (OUTPUT_DIR / "map_data.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "radius_miles": RADIUS_MILES,
        "bank_source": source_label,
        "bank_run_date": bank_run_date,
        "basemap_source": basemap_source,
        "stores": store_json,
    }, indent=2), encoding="utf-8")


def main() -> None:
    stores = [Store(*row) for row in STORES_RAW]
    geocode_stores(stores)

    all_branches: list[Branch] | None = None
    source_label = ""
    try:
        all_branches = load_fdic_branches()
        source_label = "FDIC BankFind Suite Locations API"
    except Exception as exc:
        print(f"FDIC API unavailable or unusable; using ArcGIS fallback per store: {exc}", file=sys.stderr, flush=True)

    matches, fallback_source = build_matches(stores, all_branches)
    source_label = source_label or fallback_source or "ArcGIS FDIC branch-layer fallback"
    run_dates = sorted({m.branch.run_date for values in matches.values() for m in values if m.branch.run_date})
    bank_run_date = run_dates[-1] if run_dates else ""

    overview_path = MAP_DIR / "00_NC_overview.png"
    basemap_source = create_overview_map(stores, overview_path)
    map_paths: dict[str, Path] = {}
    for idx, store in enumerate(stores, start=1):
        out_path = MAP_DIR / f"{idx:02d}_store_{store.store}.png"
        print(f"Rendering detailed map {idx}/{len(stores)} for store {store.store}", flush=True)
        *_, local_source = create_store_map(store, matches[store.store], out_path)
        basemap_source = local_source or basemap_source
        map_paths[store.store] = out_path

    pdf_path = OUTPUT_DIR / "AES_NC_Stores_Nearby_Banks_Detailed_Map.pdf"
    create_pdf(stores, matches, map_paths, overview_path, pdf_path, source_label, bank_run_date, basemap_source)
    write_outputs(stores, matches, source_label, bank_run_date, basemap_source)

    if pdf_path.stat().st_size < 100_000:
        raise RuntimeError(f"Generated PDF looks too small: {pdf_path.stat().st_size} bytes")
    if len(list(MAP_DIR.glob("*.png"))) < len(stores) + 1:
        raise RuntimeError("Not all map images were created")
    print(f"Created {pdf_path} ({pdf_path.stat().st_size:,} bytes)", flush=True)
    print(f"Bank source: {source_label}; run date: {bank_run_date or 'not supplied'}", flush=True)


if __name__ == "__main__":
    main()
