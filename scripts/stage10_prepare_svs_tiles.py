#!/usr/bin/env python3
"""Prepare tile coordinate files for downloaded TCGA SVS slides."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


TARGET_MPP = 0.50
TILE_SIZE = 256
MIN_TISSUE_OCCUPANCY = 0.70
MAX_TILES = 8000
MIN_TILES = 200


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def safe_component(value: Any) -> str:
    text = str(value or "").strip()
    keep = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text]
    return "".join(keep)[:180] or "unknown"


def slide_key(slide_id: str) -> str:
    return hashlib.sha1(str(slide_id).encode("utf-8")).hexdigest()[:16]


def find_downloaded_svs(root: Path, download_dir: Path, file_id: str, filename: str) -> Path | None:
    candidates = [
        download_dir / file_id / filename,
        download_dir / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    matches = list(download_dir.rglob(filename)) if download_dir.exists() else []
    matches = [path for path in matches if path.is_file() and path.stat().st_size > 0]
    if matches:
        return sorted(matches, key=lambda p: p.stat().st_size, reverse=True)[0]
    root_matches = list((root / "data/raw/gdc").rglob(filename))
    root_matches = [path for path in root_matches if path.is_file() and path.stat().st_size > 0]
    if root_matches:
        return sorted(root_matches, key=lambda p: p.stat().st_size, reverse=True)[0]
    return None


def objective_mpp(slide: Any) -> float | None:
    props = slide.properties
    for key in ["openslide.mpp-x", "aperio.MPP"]:
        value = props.get(key)
        if value:
            try:
                return float(value)
            except Exception:
                pass
    objective = props.get("openslide.objective-power") or props.get("aperio.AppMag")
    try:
        obj = float(objective)
    except Exception:
        return None
    if obj > 0:
        return 10.0 / obj
    return None


def make_tissue_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    gray = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    return (gray < 235.0) & ((saturation > 8.0) | (gray < 220.0))


def integral_image(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)


def rect_sum(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def choose_thumbnail(slide: Any, max_dim: int = 2048) -> Image.Image:
    width, height = slide.dimensions
    scale = min(max_dim / max(width, height), 1.0)
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return slide.get_thumbnail(size).convert("RGB")


def candidate_tiles(slide: Any, mask: np.ndarray, target_mpp: float, max_tiles: int) -> tuple[np.ndarray, dict[str, Any]]:
    width, height = slide.dimensions
    native_mpp = objective_mpp(slide) or target_mpp
    tile_native_px = max(1, int(round(TILE_SIZE * target_mpp / native_mpp)))
    mask_h, mask_w = mask.shape
    ii = integral_image(mask)
    coords: list[tuple[int, int]] = []
    occupancies: list[float] = []
    for y in range(0, max(height - tile_native_px + 1, 1), tile_native_px):
        y1 = min(y + tile_native_px, height)
        my0 = min(max(int(y / height * mask_h), 0), mask_h - 1)
        my1 = min(max(int(y1 / height * mask_h), my0 + 1), mask_h)
        for x in range(0, max(width - tile_native_px + 1, 1), tile_native_px):
            x1 = min(x + tile_native_px, width)
            mx0 = min(max(int(x / width * mask_w), 0), mask_w - 1)
            mx1 = min(max(int(x1 / width * mask_w), mx0 + 1), mask_w)
            area = max(1, (my1 - my0) * (mx1 - mx0))
            occ = rect_sum(ii, my0, mx0, my1, mx1) / area
            if occ >= MIN_TISSUE_OCCUPANCY:
                coords.append((x, y))
                occupancies.append(float(occ))
    if not coords:
        return np.zeros((0, 3), dtype=np.float32), {
            "native_mpp": native_mpp,
            "tile_native_px": tile_native_px,
            "candidate_tiles": 0,
            "selected_tiles": 0,
            "mean_selected_tissue_occupancy": 0.0,
        }
    coords_arr = np.asarray(coords, dtype=np.int64)
    occ_arr = np.asarray(occupancies, dtype=np.float32)
    if max_tiles > 0 and len(coords_arr) > max_tiles:
        idx = np.linspace(0, len(coords_arr) - 1, max_tiles).round().astype(np.int64)
        coords_arr = coords_arr[idx]
        occ_arr = occ_arr[idx]
    meta = {
        "native_mpp": float(native_mpp),
        "target_mpp": float(target_mpp),
        "tile_native_px": int(tile_native_px),
        "candidate_tiles": int(len(coords)),
        "selected_tiles": int(len(coords_arr)),
        "mean_selected_tissue_occupancy": float(occ_arr.mean()) if len(occ_arr) else 0.0,
    }
    return np.column_stack([coords_arr, occ_arr]).astype(np.float32), meta


def draw_thumbnail(thumbnail: Image.Image, coords: np.ndarray, slide_dims: tuple[int, int], out_path: Path) -> None:
    image = thumbnail.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = slide_dims
    tw, th = image.size
    step = max(1, len(coords) // 1000)
    for row in coords[::step]:
        x = int(float(row[0]) / width * tw)
        y = int(float(row[1]) / height * th)
        draw.rectangle((x - 1, y - 1, x + 1, y + 1), outline=(220, 30, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=90)


def process_row(row: dict[str, Any], root: str, download_dir: str, max_tiles: int, min_tiles: int) -> dict[str, Any]:
    import openslide

    project_root = Path(root)
    dl_dir = project_root / download_dir
    cohort = str(row["cohort"])
    patient_id = str(row["patient_id"])
    file_id = str(row["id"])
    filename = str(row["filename"])
    key = slide_key(file_id)
    result: dict[str, Any] = {
        "cohort": cohort,
        "patient_id": patient_id,
        "slide_id": file_id,
        "slide_candidate_key": key,
        "filename": filename,
        "gdc_file_id": file_id,
        "gdc_md5": row.get("md5", ""),
        "gdc_size": row.get("size", ""),
        "sample_type": row.get("sample_type", row.get("sample_types", "")),
        "data_type": row.get("data_type", ""),
        "tcga_wsi_selection_role": row.get("tcga_wsi_selection_role", ""),
        "tcga_wsi_selection_rule": row.get("tcga_wsi_selection_rule", ""),
        "qc_method": "OpenSlide thumbnail tissue-mask proxy; HistoQC not run for Stage 10 SVS pilot",
        "histoqc_status": "not_run_openslide_thumbnail_proxy",
        "status": "FAILED",
    }
    try:
        svs_path = find_downloaded_svs(project_root, dl_dir, file_id, filename)
        if svs_path is None:
            raise FileNotFoundError(f"downloaded SVS not found for {file_id}/{filename} under {dl_dir}")
        slide = openslide.OpenSlide(str(svs_path))
        slide_width, slide_height = slide.dimensions
        level_count = int(getattr(slide, "level_count", 0))
        thumb = choose_thumbnail(slide)
        mask = make_tissue_mask(np.asarray(thumb))
        coords, meta = candidate_tiles(slide, mask, TARGET_MPP, max_tiles)
        status = "PASS" if int(meta["selected_tiles"]) >= min_tiles else "FAILED"
        out_base = project_root / "features/tcga_wsi_tiles" / cohort / patient_id / key
        h5_path = out_base / "tile_coords.h5"
        thumb_path = project_root / "features/tcga_wsi_thumbnails" / cohort / patient_id / f"{key}.jpg"
        out_base.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("coords_x_y_tissue", data=coords, compression="gzip")
            handle.attrs["cohort"] = cohort
            handle.attrs["patient_id"] = patient_id
            handle.attrs["slide_id"] = file_id
            handle.attrs["slide_candidate_key"] = key
            handle.attrs["slide_format"] = "SVS"
            handle.attrs["svs_file"] = str(svs_path.relative_to(project_root))
            handle.attrs["filename"] = filename
            handle.attrs["total_cols"] = int(slide_width)
            handle.attrs["total_rows"] = int(slide_height)
            handle.attrs["openslide_level_count"] = int(level_count)
            for name, value in meta.items():
                handle.attrs[name] = value
        draw_thumbnail(thumb, coords, (slide_width, slide_height), thumb_path)
        slide.close()
        result.update(
            {
                "status": status,
                "slide_path": str(svs_path.relative_to(project_root)),
                "tile_coords_path": str(h5_path.relative_to(project_root)),
                "thumbnail_qc_path": str(thumb_path.relative_to(project_root)),
                "slide_width": int(slide_width),
                "slide_height": int(slide_height),
                "level_mpp": meta["native_mpp"],
                "target_mpp": meta["target_mpp"],
                "tile_native_px": meta["tile_native_px"],
                "candidate_tiles": meta["candidate_tiles"],
                "selected_tiles": meta["selected_tiles"],
                "mean_selected_tissue_occupancy": round(meta["mean_selected_tissue_occupancy"], 4),
                "openslide_level_count": int(level_count),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", default="manifests/gdc/tcga_wsi_transfer/tcga_wsi_pilot_slide_manifest.tsv")
    parser.add_argument("--download-dir", default="data/raw/gdc/tcga_wsi_transfer/pilot")
    parser.add_argument("--output-prefix", default="data/processed/stage10/tcga_wsi_pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tiles-per-slide", type=int, default=MAX_TILES)
    parser.add_argument("--min-tiles-per-slide", type=int, default=MIN_TILES)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest_path = root / args.manifest
    out_prefix = root / args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str).fillna("")
    if manifest.empty:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    rows = manifest.to_dict("records")
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(process_row, row, str(root), args.download_dir, args.max_tiles_per_slide, args.min_tiles_per_slide)
            for row in rows
        ]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
    results = sorted(results, key=lambda r: (r.get("cohort", ""), r.get("patient_id", ""), r.get("slide_id", "")))
    out_table = Path(str(out_prefix) + "_tile_qc.tsv")
    pd.DataFrame(results).to_csv(out_table, sep="\t", index=False)
    pass_rows = [row for row in results if row.get("status") == "PASS"]
    summary = {
        "built_at": now_iso(),
        "status": "PASS" if len(pass_rows) == len(results) else "PARTIAL",
        "manifest": str(manifest_path.relative_to(root)),
        "download_dir": args.download_dir,
        "slides_requested": int(len(results)),
        "pass_slides": int(len(pass_rows)),
        "failed_slides": int(len(results) - len(pass_rows)),
        "total_selected_tiles": int(sum(int(float(row.get("selected_tiles", 0) or 0)) for row in pass_rows)),
        "max_tiles_per_slide": int(args.max_tiles_per_slide),
        "min_tiles_per_slide": int(args.min_tiles_per_slide),
        "output_table": str(out_table.relative_to(root)),
    }
    Path(str(out_prefix) + "_tile_prep_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
