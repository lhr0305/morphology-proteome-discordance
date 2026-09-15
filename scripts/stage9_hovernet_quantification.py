#!/usr/bin/env python3
"""Quantify representative Stage 9 tiles with TIAToolbox HoVer-Net."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


PANNNUKE_TYPE_NAMES = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Non-Neoplastic Epithelial",
}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"patient_id": str, "slide_candidate_key": str})


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def polygon_area(points: np.ndarray) -> float:
    if points.size == 0:
        return float("nan")
    valid = np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 1] >= 0)
    pts = points[valid].astype(np.float64, copy=False)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def class_fraction(counts: Counter[int], class_id: int, total: int) -> float:
    if total <= 0:
        return float("nan")
    return float(counts.get(class_id, 0) / total)


def load_type_dict(model_name: str) -> dict[int, str]:
    try:
        from tiatoolbox.models.architecture import PRETRAINED_INFO

        info = PRETRAINED_INFO.get(model_name, {})
        raw = info.get("architecture", {}).get("kwargs", {}).get("nuc_type_dict", None)
        if raw:
            return {int(k): str(v) for k, v in raw.items()}
    except Exception:
        pass
    return PANNNUKE_TYPE_NAMES.copy()


def summarize_tile(
    catalog_row: pd.Series,
    local_index: int,
    outputs: dict[str, Any],
    type_dict: dict[int, str],
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_path = root / str(catalog_row["image_path"])
    with Image.open(image_path) as img:
        tile_width, tile_height = img.size

    boxes = np.asarray(outputs["box"][local_index])
    centroids = np.asarray(outputs["centroid"][local_index])
    probs = np.asarray(outputs["prob"][local_index])
    nuc_types = np.asarray(outputs["type"][local_index], dtype=np.int64)
    contours = np.asarray(outputs["contours"][local_index])
    predictions = outputs.get("predictions")
    if predictions is not None:
        pred = np.asarray(predictions[local_index])
        pred_height, pred_width = int(pred.shape[-2]), int(pred.shape[-1])
    else:
        pred_height, pred_width = tile_height, tile_width

    n_nuclei = int(len(nuc_types))
    counts = Counter(int(x) for x in nuc_types.tolist())
    effective_area_px = max(1, pred_height * pred_width)
    tile_area_px = max(1, tile_height * tile_width)
    tissue_area_px = max(1.0, tile_area_px * safe_float(catalog_row.get("tissue_occupancy", np.nan)))
    tumor_like = counts.get(1, 0) + counts.get(5, 0)
    stromal_like = counts.get(3, 0)
    immune_like = counts.get(2, 0)
    stromal_denominator = tumor_like + stromal_like

    tile_row = catalog_row.to_dict()
    tile_row.update(
        {
            "hovernet_model": "hovernet_fast-pannuke",
            "hovernet_nuclei_total": n_nuclei,
            "hovernet_output_height_px": pred_height,
            "hovernet_output_width_px": pred_width,
            "hovernet_tile_height_px": tile_height,
            "hovernet_tile_width_px": tile_width,
            "hovernet_effective_area_px": effective_area_px,
            "hovernet_nuclei_density_per_10k_effective_px": float(n_nuclei * 10000.0 / effective_area_px),
            "hovernet_nuclei_density_per_10k_tile_px": float(n_nuclei * 10000.0 / tile_area_px),
            "hovernet_nuclei_density_per_10k_tissue_px": float(n_nuclei * 10000.0 / tissue_area_px),
            "hovernet_neoplastic_count": int(counts.get(1, 0)),
            "hovernet_inflammatory_count": int(counts.get(2, 0)),
            "hovernet_connective_count": int(counts.get(3, 0)),
            "hovernet_dead_count": int(counts.get(4, 0)),
            "hovernet_non_neoplastic_epithelial_count": int(counts.get(5, 0)),
            "hovernet_neoplastic_fraction": class_fraction(counts, 1, n_nuclei),
            "hovernet_inflammatory_fraction": class_fraction(counts, 2, n_nuclei),
            "hovernet_connective_fraction": class_fraction(counts, 3, n_nuclei),
            "hovernet_dead_fraction": class_fraction(counts, 4, n_nuclei),
            "hovernet_non_neoplastic_epithelial_fraction": class_fraction(counts, 5, n_nuclei),
            "hovernet_stromal_richness_proxy": float(stromal_like / stromal_denominator)
            if stromal_denominator > 0
            else float("nan"),
            "hovernet_immune_richness_proxy": float(immune_like / n_nuclei) if n_nuclei > 0 else float("nan"),
            "hovernet_tumor_like_epithelial_fraction": float(tumor_like / n_nuclei) if n_nuclei > 0 else float("nan"),
        }
    )

    instance_rows: list[dict[str, Any]] = []
    for idx in range(n_nuclei):
        box = boxes[idx].astype(float) if boxes.size else np.full(4, np.nan)
        centroid = centroids[idx].astype(float) if centroids.size else np.full(2, np.nan)
        type_id = int(nuc_types[idx])
        instance_rows.append(
            {
                "export_type": catalog_row.get("export_type"),
                "motif_cluster": catalog_row.get("motif_cluster"),
                "cohort": catalog_row.get("cohort"),
                "patient_id": catalog_row.get("patient_id"),
                "slide_candidate_key": catalog_row.get("slide_candidate_key"),
                "tile_index": catalog_row.get("tile_index"),
                "image_path": catalog_row.get("image_path"),
                "nucleus_index": idx,
                "type_id": type_id,
                "type_name": type_dict.get(type_id, f"type_{type_id}"),
                "probability": float(probs[idx]) if len(probs) > idx else float("nan"),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "box_x1": float(box[0]),
                "box_y1": float(box[1]),
                "box_x2": float(box[2]),
                "box_y2": float(box[3]),
                "box_area_px": float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])),
                "contour_area_px": polygon_area(contours[idx]) if len(contours) > idx else float("nan"),
            }
        )
    return tile_row, instance_rows


def group_summary(tile_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in tile_df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=False))
        total_nuclei = int(sub["hovernet_nuclei_total"].sum())
        row.update(
            {
                "tile_rows": int(len(sub)),
                "patients": int(sub["patient_id"].nunique()),
                "slides": int(sub[["cohort", "patient_id", "slide_candidate_key"]].drop_duplicates().shape[0]),
                "nuclei_total": total_nuclei,
                "mean_nuclei_density_per_10k_effective_px": float(
                    sub["hovernet_nuclei_density_per_10k_effective_px"].mean()
                ),
                "median_nuclei_density_per_10k_effective_px": float(
                    sub["hovernet_nuclei_density_per_10k_effective_px"].median()
                ),
                "weighted_neoplastic_fraction": float(sub["hovernet_neoplastic_count"].sum() / total_nuclei)
                if total_nuclei
                else float("nan"),
                "weighted_inflammatory_fraction": float(sub["hovernet_inflammatory_count"].sum() / total_nuclei)
                if total_nuclei
                else float("nan"),
                "weighted_connective_fraction": float(sub["hovernet_connective_count"].sum() / total_nuclei)
                if total_nuclei
                else float("nan"),
                "weighted_dead_fraction": float(sub["hovernet_dead_count"].sum() / total_nuclei)
                if total_nuclei
                else float("nan"),
                "weighted_non_neoplastic_epithelial_fraction": float(
                    sub["hovernet_non_neoplastic_epithelial_count"].sum() / total_nuclei
                )
                if total_nuclei
                else float("nan"),
                "mean_stromal_richness_proxy": float(sub["hovernet_stromal_richness_proxy"].mean()),
                "mean_immune_richness_proxy": float(sub["hovernet_immune_richness_proxy"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--catalog", default="results/spatial/representative_tile_catalog.tsv")
    parser.add_argument("--output-dir", default="results/spatial")
    parser.add_argument("--output-prefix", default="hovernet")
    parser.add_argument("--model", default="hovernet_fast-pannuke")
    parser.add_argument("--weights", default="env/model_cache/tiatoolbox_weights/hovernet_fast-pannuke.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out_dir = root / args.output_dir
    qc_dir = root / "logs" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    weights = (root / args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"HoVer-Net weights not found: {weights}")
    catalog_all = read_tsv(root / args.catalog)
    image_status_counts = {
        str(k): int(v)
        for k, v in catalog_all["image_status"].fillna("NA").value_counts(dropna=False).to_dict().items()
    }
    catalog = catalog_all[catalog_all["image_status"].fillna("") == "PASS"].copy()
    if args.limit and args.limit > 0:
        catalog = catalog.head(args.limit).copy()
    catalog = catalog.reset_index(drop=True)
    if catalog.empty:
        raise RuntimeError("No PASS tile images available for HoVer-Net quantification.")

    tile_out = out_dir / f"{args.output_prefix}_tile_nuclei_quantification.tsv"
    inst_out = out_dir / f"{args.output_prefix}_nucleus_instance_table.tsv.gz"
    motif_out = out_dir / f"{args.output_prefix}_motif_cell_summary.tsv"
    cohort_motif_out = out_dir / f"{args.output_prefix}_motif_cell_summary_by_cohort.tsv"
    summary_json = out_dir / f"{args.output_prefix}_run_summary.json"
    qc_out = qc_dir / f"stage9_{args.output_prefix}_qc.json"
    for path in [tile_out, inst_out, motif_out, cohort_motif_out, summary_json, qc_out]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output without --overwrite: {path}")

    from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
    import torch
    import tiatoolbox
    import cv2
    import skimage

    type_dict = load_type_dict(args.model)
    segmentor = NucleusInstanceSegmentor(
        model=args.model,
        weights=str(weights),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        verbose=True,
    )

    tile_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    images = [str(root / path) for path in catalog["image_path"].tolist()]
    for start in range(0, len(images), args.chunk_size):
        end = min(start + args.chunk_size, len(images))
        sub_images = images[start:end]
        try:
            outputs = segmentor.run(
                sub_images,
                patch_mode=True,
                output_type="dict",
                overwrite=True,
                return_predictions=(False,),
                return_probabilities=False,
            )
            for local_idx, row_idx in enumerate(range(start, end)):
                tile_row, inst = summarize_tile(catalog.iloc[row_idx], local_idx, outputs, type_dict, root)
                tile_rows.append(tile_row)
                instance_rows.extend(inst)
        except Exception as exc:
            if len(sub_images) == 1:
                failed.append({"image_path": catalog.iloc[start]["image_path"], "error": repr(exc)})
                continue
            for row_idx in range(start, end):
                try:
                    outputs = segmentor.run(
                        [images[row_idx]],
                        patch_mode=True,
                        output_type="dict",
                        overwrite=True,
                        return_predictions=(False,),
                        return_probabilities=False,
                    )
                    tile_row, inst = summarize_tile(catalog.iloc[row_idx], 0, outputs, type_dict, root)
                    tile_rows.append(tile_row)
                    instance_rows.extend(inst)
                except Exception as image_exc:
                    failed.append({"image_path": catalog.iloc[row_idx]["image_path"], "error": repr(image_exc)})

    tile_df = pd.DataFrame(tile_rows)
    instance_df = pd.DataFrame(instance_rows)
    tile_df.to_csv(tile_out, sep="\t", index=False)
    instance_df.to_csv(inst_out, sep="\t", index=False, compression="gzip")

    motif_summary = group_summary(tile_df, ["export_type", "motif_cluster"])
    motif_summary.to_csv(motif_out, sep="\t", index=False)
    cohort_motif_summary = group_summary(tile_df, ["cohort", "export_type", "motif_cluster"])
    cohort_motif_summary.to_csv(cohort_motif_out, sep="\t", index=False)

    warn_count = 0
    notes = [
        "HoVer-Net quantification used TIAToolbox NucleusInstanceSegmentor with local pretrained hovernet_fast-pannuke weights.",
        "Density is reported per model effective output pixels because patch-mode HoVer-Net emits a cropped central output region.",
        "Stromal-richness proxy is connective nuclei divided by connective plus tumor-like epithelial nuclei.",
    ]
    skipped_by_quality_gate = int(len(catalog_all) - len(catalog))
    if skipped_by_quality_gate:
        warn_count += 1
        notes.append(
            f"{skipped_by_quality_gate} representative tile images were excluded by the pre-existing image_status quality gate."
        )
    if failed:
        warn_count += 1
        notes.append(f"{len(failed)} tile images failed and were excluded from summaries.")
    if len(tile_df) != len(catalog):
        warn_count += 1
    if int(tile_df["hovernet_nuclei_total"].sum()) <= 0:
        warn_count += 1
        notes.append("No nuclei were detected across all processed tiles.")

    qc = {
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS" if not failed and len(tile_df) == len(catalog) and int(tile_df["hovernet_nuclei_total"].sum()) > 0 else "PARTIAL",
        "fail_count": 0,
        "warn_count": warn_count,
        "model": args.model,
        "weights_path": str(weights),
        "weights_size_bytes": int(weights.stat().st_size),
        "weights_sha256": sha256_file(weights),
        "type_dict": type_dict,
        "catalog_path": str((root / args.catalog).resolve()),
        "catalog_total_rows": int(len(catalog_all)),
        "catalog_image_status_counts": image_status_counts,
        "catalog_nonpass_rows_skipped": skipped_by_quality_gate,
        "catalog_pass_rows_requested": int(len(catalog)),
        "tiles_processed": int(len(tile_df)),
        "tiles_failed": int(len(failed)),
        "nuclei_total": int(tile_df["hovernet_nuclei_total"].sum()) if len(tile_df) else 0,
        "tile_output": str(tile_out),
        "instance_output": str(inst_out),
        "motif_summary_output": str(motif_out),
        "cohort_motif_summary_output": str(cohort_motif_out),
        "failed_tiles": failed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tiatoolbox": getattr(tiatoolbox, "__version__", ""),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cv2": cv2.__version__,
            "skimage": skimage.__version__,
        },
        "notes": notes,
    }
    summary_json.write_text(json.dumps(qc, indent=2, ensure_ascii=False) + "\n")
    qc_out.write_text(json.dumps(qc, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: qc[k] for k in ["status", "warn_count", "tiles_processed", "tiles_failed", "nuclei_total"]}))
    if qc["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
