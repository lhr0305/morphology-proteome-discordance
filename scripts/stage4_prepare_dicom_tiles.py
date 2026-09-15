#!/usr/bin/env python3
"""Prepare Stage 4 DICOM WSI QC proxies and tile coordinate files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw


RANDOM_SEED = 20260609
TARGET_MPP = 0.50
TILE_SIZE = 256
MIN_TISSUE_OCCUPANCY = 0.70
MAX_TILES = 8000
MIN_TILES = 500


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pixel_spacing_um(ds: pydicom.Dataset) -> float | None:
    try:
        spacing = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        return float(spacing[0]) * 1000.0
    except Exception:
        try:
            return float(ds.PixelSpacing[0]) * 1000.0
        except Exception:
            return None


def image_type(ds: pydicom.Dataset) -> str:
    return "|".join(str(x) for x in getattr(ds, "ImageType", []))


def slide_key(slide_id: str) -> str:
    return hashlib.sha1(slide_id.encode("utf-8")).hexdigest()[:16]


def inspect_series(series_path: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(series_path.glob("*.dcm")):
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        rows.append(
            {
                "path": path,
                "file_name": path.name,
                "frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
                "rows": int(getattr(ds, "Rows", 0) or 0),
                "cols": int(getattr(ds, "Columns", 0) or 0),
                "total_rows": int(getattr(ds, "TotalPixelMatrixRows", getattr(ds, "Rows", 0)) or 0),
                "total_cols": int(getattr(ds, "TotalPixelMatrixColumns", getattr(ds, "Columns", 0)) or 0),
                "image_type": image_type(ds),
                "mpp": pixel_spacing_um(ds),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def choose_volume_level(levels: list[dict[str, Any]]) -> dict[str, Any] | None:
    volume = [row for row in levels if "VOLUME" in row["image_type"] and row.get("mpp")]
    if not volume:
        return None
    # Prefer levels that can derive a 0.5 um/px tile by integer downsampling.
    def score(row: dict[str, Any]) -> tuple[float, float, int]:
        mpp = float(row["mpp"])
        ratio = TARGET_MPP / mpp
        nearest_int = max(1, round(ratio))
        derived_mpp = mpp * nearest_int
        return (abs(math.log2(derived_mpp / TARGET_MPP)), abs(math.log2(mpp / TARGET_MPP)), int(row["frames"]))
    return sorted(volume, key=score)[0]


def choose_overview(levels: list[dict[str, Any]], chosen: dict[str, Any]) -> dict[str, Any] | None:
    overview = [row for row in levels if "OVERVIEW" in row["image_type"] or "THUMBNAIL" in row["image_type"]]
    if overview:
        return sorted(overview, key=lambda row: row["rows"] * row["cols"], reverse=True)[0]
    resampled = [row for row in levels if "VOLUME" in row["image_type"] and row["path"] != chosen["path"]]
    if resampled:
        return sorted(resampled, key=lambda row: row["rows"] * row["cols"] * row["frames"])[0]
    return chosen


def read_rgb(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return arr[:, :, :3].astype(np.uint8)


def make_tissue_mask(overview: np.ndarray) -> np.ndarray:
    rgb = overview.astype(np.float32)
    gray = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    # White background in H&E has high brightness and low saturation.
    return (gray < 235.0) & ((saturation > 8.0) | (gray < 220.0))


def integral_image(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)


def rect_sum(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def candidate_tiles(chosen: dict[str, Any], overview: np.ndarray, tissue_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    native_mpp = float(chosen["mpp"])
    downsample_factor = max(1, int(round(TARGET_MPP / native_mpp)))
    derived_mpp = native_mpp * downsample_factor
    tile_native_px = int(round(TILE_SIZE * TARGET_MPP / native_mpp))
    total_rows = int(chosen["total_rows"])
    total_cols = int(chosen["total_cols"])
    ov_rows, ov_cols = tissue_mask.shape
    ii = integral_image(tissue_mask)
    coords = []
    occupancies = []
    for y in range(0, max(total_rows - tile_native_px + 1, 1), tile_native_px):
        y1_native = min(y + tile_native_px, total_rows)
        oy0 = int(y / total_rows * ov_rows)
        oy1 = max(oy0 + 1, int(y1_native / total_rows * ov_rows))
        oy0 = min(max(oy0, 0), ov_rows - 1)
        oy1 = min(max(oy1, oy0 + 1), ov_rows)
        for x in range(0, max(total_cols - tile_native_px + 1, 1), tile_native_px):
            x1_native = min(x + tile_native_px, total_cols)
            ox0 = int(x / total_cols * ov_cols)
            ox1 = max(ox0 + 1, int(x1_native / total_cols * ov_cols))
            ox0 = min(max(ox0, 0), ov_cols - 1)
            ox1 = min(max(ox1, ox0 + 1), ov_cols)
            area = max(1, (oy1 - oy0) * (ox1 - ox0))
            occ = rect_sum(ii, oy0, ox0, oy1, ox1) / area
            if occ >= MIN_TISSUE_OCCUPANCY:
                coords.append((x, y))
                occupancies.append(occ)
    coords_arr = np.array(coords, dtype=np.int64) if coords else np.zeros((0, 2), dtype=np.int64)
    occ_arr = np.array(occupancies, dtype=np.float32) if occupancies else np.zeros((0,), dtype=np.float32)
    if len(coords_arr) > MAX_TILES:
        idx = np.linspace(0, len(coords_arr) - 1, MAX_TILES).round().astype(int)
        coords_arr = coords_arr[idx]
        occ_arr = occ_arr[idx]
    meta = {
        "native_mpp": native_mpp,
        "target_mpp": TARGET_MPP,
        "downsample_factor": downsample_factor,
        "derived_mpp": derived_mpp,
        "tile_native_px": tile_native_px,
        "candidate_tiles": int(len(coords)),
        "selected_tiles": int(len(coords_arr)),
        "mean_selected_tissue_occupancy": float(occ_arr.mean()) if len(occ_arr) else 0.0,
    }
    return np.column_stack([coords_arr, occ_arr]) if len(coords_arr) else np.zeros((0, 3), dtype=np.float32), meta


def draw_thumbnail(overview: np.ndarray, coords: np.ndarray, chosen: dict[str, Any], out_path: Path) -> None:
    image = Image.fromarray(overview).convert("RGB")
    draw = ImageDraw.Draw(image)
    ov_w, ov_h = image.size
    total_rows = int(chosen["total_rows"])
    total_cols = int(chosen["total_cols"])
    for row in coords[:: max(1, len(coords) // 1000)]:
        x = int(float(row[0]) / total_cols * ov_w)
        y = int(float(row[1]) / total_rows * ov_h)
        draw.rectangle((x - 1, y - 1, x + 1, y + 1), outline=(220, 30, 30))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((1600, 1600))
    image.save(out_path, quality=90)


def process_slide(row: dict[str, str], root: Path, candidate_outputs: bool = False) -> dict[str, Any]:
    cohort = row["cohort"]
    patient_id = row["patient_id"]
    slide_id = row["slide_id"]
    series_path = root / row["slide_path"]
    candidate_key = slide_key(slide_id)
    if candidate_outputs:
        out_base = root / "features/tiles_candidates" / cohort / patient_id
        h5_path = out_base / f"{candidate_key}.tile_coords.h5"
        thumb_path = root / "features/thumbnails_candidates" / cohort / patient_id / f"{candidate_key}.jpg"
    else:
        out_base = root / "features/tiles" / cohort / patient_id
        h5_path = out_base / "tile_coords.h5"
        thumb_path = root / "features/thumbnails" / cohort / f"{patient_id}.jpg"
    result: dict[str, Any] = {
        "cohort": cohort,
        "patient_id": patient_id,
        "slide_id": slide_id,
        "slide_candidate_key": candidate_key,
        "slide_path": row["slide_path"],
        "status": "FAILED",
        "qc_method": "DICOM overview tissue-mask proxy; HistoQC not run",
        "histoqc_status": "not_run_histoqc_unavailable_for_current_dicom_toolchain",
        "is_representative_stage2": row.get("is_representative_stage2", ""),
        "representative_rank_stage2": row.get("representative_rank_stage2", ""),
    }
    try:
        levels = inspect_series(series_path)
        chosen = choose_volume_level(levels)
        if chosen is None:
            raise RuntimeError("no VOLUME level with pixel spacing")
        overview_level = choose_overview(levels, chosen)
        if overview_level is None:
            raise RuntimeError("no overview/thumbnail/volume level available")
        overview = read_rgb(Path(overview_level["path"]))
        tissue_mask = make_tissue_mask(overview)
        coords, meta = candidate_tiles(chosen, overview, tissue_mask)
        status = "PASS" if meta["selected_tiles"] >= MIN_TILES else "FAILED"
        out_base.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("coords_x_y_tissue", data=coords, compression="gzip")
            handle.attrs["cohort"] = cohort
            handle.attrs["patient_id"] = patient_id
            handle.attrs["slide_id"] = slide_id
            handle.attrs["dicom_level_file"] = str(Path(chosen["path"]).relative_to(root))
            handle.attrs["overview_file"] = str(Path(overview_level["path"]).relative_to(root))
            for key, value in meta.items():
                handle.attrs[key] = value
        draw_thumbnail(overview, coords, chosen, thumb_path)
        result.update(
            {
                "status": status,
                "level_file": str(Path(chosen["path"]).relative_to(root)),
                "overview_file": str(Path(overview_level["path"]).relative_to(root)),
                "tile_coords_path": str(h5_path.relative_to(root)),
                "thumbnail_qc_path": str(thumb_path.relative_to(root)),
                "level_mpp": meta["native_mpp"],
                "target_mpp": meta["target_mpp"],
                "derived_mpp": meta["derived_mpp"],
                "downsample_factor": meta["downsample_factor"],
                "tile_native_px": meta["tile_native_px"],
                "candidate_tiles": meta["candidate_tiles"],
                "selected_tiles": meta["selected_tiles"],
                "mean_selected_tissue_occupancy": round(meta["mean_selected_tissue_occupancy"], 4),
                "dicom_files": len(levels),
                "level_frames": chosen["frames"],
                "level_total_rows": chosen["total_rows"],
                "level_total_cols": chosen["total_cols"],
                "overview_rows": overview.shape[0],
                "overview_cols": overview.shape[1],
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def stage2_rank_value(row: dict[str, Any]) -> int:
    try:
        return int(row.get("representative_rank_stage2", "") or 999999)
    except Exception:
        return 999999


def select_stage4_representatives(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    by_patient: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_patient.setdefault((str(row.get("cohort", "")), str(row.get("patient_id", ""))), []).append(row)
    for key in sorted(by_patient):
        rows = by_patient[key]

        def sort_key(row: dict[str, Any]) -> tuple[int, int, float, int, str]:
            status_score = 1 if row.get("status") == "PASS" else 0
            tiles = int(float(row.get("selected_tiles", 0) or 0))
            occ = float(row.get("mean_selected_tissue_occupancy", 0) or 0)
            return (-status_score, -tiles, -occ, stage2_rank_value(row), str(row.get("slide_id", "")))

        best = sorted(rows, key=sort_key)[0].copy()
        pass_count = sum(1 for row in rows if row.get("status") == "PASS")
        best["candidate_slide_count_stage4"] = len(rows)
        best["candidate_pass_count_stage4"] = pass_count
        best["selected_representative_stage4"] = "True"
        best["stage4_selection_reason"] = (
            "best_PASS_by_selected_tiles_then_tissue_occupancy_then_stage2_rank"
            if best.get("status") == "PASS"
            else "no_candidate_reached_min_tile_gate_best_available_recorded_as_FAILED"
        )
        selected.append(best)
    return selected


def publish_canonical_outputs(root: Path, selected_rows: list[dict[str, Any]]) -> None:
    for row in selected_rows:
        if row.get("status") != "PASS":
            continue
        cohort = str(row["cohort"])
        patient_id = str(row["patient_id"])
        src_h5 = root / str(row["tile_coords_path"])
        src_thumb = root / str(row["thumbnail_qc_path"])
        dst_h5 = root / "features/tiles" / cohort / patient_id / "tile_coords.h5"
        dst_thumb = root / "features/thumbnails" / cohort / f"{patient_id}.jpg"
        dst_h5.parent.mkdir(parents=True, exist_ok=True)
        dst_thumb.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_h5, dst_h5)
        shutil.copy2(src_thumb, dst_thumb)
        with h5py.File(dst_h5, "a") as handle:
            handle.attrs["canonical_stage4_representative"] = True
            handle.attrs["canonical_source_tile_coords_path"] = str(src_h5.relative_to(root))
        row["candidate_tile_coords_path"] = row["tile_coords_path"]
        row["candidate_thumbnail_qc_path"] = row["thumbnail_qc_path"]
        row["tile_coords_path"] = str(dst_h5.relative_to(root))
        row["thumbnail_qc_path"] = str(dst_thumb.relative_to(root))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--set-name", default="set_triplet")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("STAGE4_WORKERS", "8")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all-candidates", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    master = pd.read_csv(root / "data/processed/stage2/master_case_table.tsv", sep="\t", dtype=str).fillna("")
    analysis = pd.read_csv(root / "data/processed/stage2/analysis_sets.tsv", sep="\t", dtype=str).fillna("")
    selected = analysis[analysis["set_name"] == args.set_name][["cohort", "patient_id"]].drop_duplicates()
    if args.all_candidates:
        multi = pd.read_csv(root / "data/processed/stage2/multi_slide_sensitivity_table.tsv", sep="\t", dtype=str).fillna("")
        work = selected.merge(multi, on=["cohort", "patient_id"], how="left").sort_values(
            ["cohort", "patient_id", "representative_rank_stage2", "slide_id"]
        )
    else:
        work = selected.merge(master, on=["cohort", "patient_id"], how="left").sort_values(["cohort", "patient_id"])
    if args.limit:
        work = work.head(args.limit)
    rows = work.to_dict("records")
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process_slide, row, root, args.all_candidates) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results = sorted(results, key=lambda row: (row.get("cohort", ""), row.get("patient_id", "")))
    out_dir = root / "data/processed/stage4"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.all_candidates:
        write_tsv(out_dir / "slide_qc_all_candidates.tsv", results)
        final_results = select_stage4_representatives(results)
        publish_canonical_outputs(root, final_results)
    else:
        final_results = results
    write_tsv(out_dir / "slide_qc_table.tsv", final_results)
    summary = {
        "set_name": args.set_name,
        "workers": args.workers,
        "all_candidates": args.all_candidates,
        "slides_requested": len(rows),
        "slides_processed": len(results),
        "candidate_pass_slides": sum(1 for row in results if row.get("status") == "PASS"),
        "candidate_failed_slides": sum(1 for row in results if row.get("status") != "PASS"),
        "final_representative_slides": len(final_results),
        "pass_slides": sum(1 for row in final_results if row.get("status") == "PASS"),
        "failed_slides": sum(1 for row in final_results if row.get("status") != "PASS"),
        "total_selected_tiles": int(sum(int(row.get("selected_tiles", 0) or 0) for row in final_results)),
        "min_tiles_per_slide": MIN_TILES,
        "max_tiles_per_slide": MAX_TILES,
        "target_mpp": TARGET_MPP,
        "tile_size": TILE_SIZE,
        "method_note": "DICOM overview tissue-mask proxy; HistoQC package was attempted but not used as final QC because current IDC DICOM route and dependency constraints prevented a valid HistoQC run.",
    }
    (out_dir / "stage4_tile_prep_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["failed_slides"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
