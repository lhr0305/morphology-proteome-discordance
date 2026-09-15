#!/usr/bin/env python3
"""Prepare representative TCGA WSI manifests for Stage 10 transfer.

The full GDC slide manifests contain multiple slides per case. Stage 10 WSI
transfer should use a deterministic representative primary-tumor slide first,
then optionally expand to all slides if resources allow.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


GDC_CLIENT_COLUMNS = ["id", "filename", "md5", "size", "state"]


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype("int64")
    df["cohort"] = path.name.replace("_slide_images_manifest.tsv", "")
    df["patient_id"] = df["case_submitter_ids"].astype(str)
    df["sample_type"] = df["sample_types"].astype(str)
    df["slide_strategy"] = df.get("experimental_strategy", "").astype(str)
    df["is_primary_tumor"] = df["sample_type"].str.contains("Primary Tumor", case=False, na=False)
    df["is_diagnostic"] = df["slide_strategy"].str.contains("Diagnostic", case=False, na=False) | df["filename"].str.contains("-DX", case=False, na=False)
    df["is_tissue_slide"] = df["slide_strategy"].str.contains("Tissue", case=False, na=False) | df["filename"].str.contains("-TS", case=False, na=False)
    return df


def select_representatives(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    sort_cols = ["patient_id", "is_primary_tumor", "is_diagnostic", "size", "filename"]
    work = work.sort_values(sort_cols, ascending=[True, False, False, False, True])
    rep = work.groupby("patient_id", as_index=False, sort=True).head(1).copy()
    rep["tcga_wsi_selection_role"] = "representative_primary_tumor_slide"
    rep["tcga_wsi_selection_rule"] = "primary_tumor_first__diagnostic_slide_first__largest_size__filename_tiebreak"
    return rep


def select_pilot(representatives: pd.DataFrame, per_cohort: int, min_size_mb: float, max_size_gb: float) -> pd.DataFrame:
    if per_cohort <= 0:
        return representatives.iloc[0:0].copy()
    rows = []
    min_size = int(min_size_mb * 1024**2)
    max_size = int(max_size_gb * 1024**3) if max_size_gb > 0 else None
    for cohort, sub in representatives.groupby("cohort", sort=True):
        work = sub[sub["size"] >= min_size].copy()
        if max_size is not None:
            work = work[work["size"] <= max_size].copy()
        if work.empty:
            work = sub.copy()
        work = work.sort_values(["size", "patient_id", "filename"], ascending=[True, True, True]).head(per_cohort)
        rows.append(work)
    pilot = pd.concat(rows, ignore_index=True) if rows else representatives.iloc[0:0].copy()
    pilot["tcga_wsi_selection_role"] = "pilot_representative_primary_tumor_slide"
    pilot["tcga_wsi_selection_rule"] = (
        "smallest_representative_primary_tumor_slide_with_size_bounds_for_pipeline_pilot; "
        "not the final production representative ranking"
    )
    return pilot


def write_gdc_manifest(df: pd.DataFrame, path: Path) -> None:
    out = df[GDC_CLIENT_COLUMNS].copy()
    out.to_csv(path, sep="\t", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest-dir", default="manifests/gdc")
    parser.add_argument("--output-dir", default="manifests/gdc/tcga_wsi_transfer")
    parser.add_argument("--pilot-per-cohort", type=int, default=1)
    parser.add_argument("--pilot-min-size-mb", type=float, default=50.0)
    parser.add_argument("--pilot-max-size-gb", type=float, default=0.8)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest_dir = root / args.manifest_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = sorted(manifest_dir.glob("TCGA-*_slide_images_manifest.tsv"))
    if not manifests:
        raise RuntimeError(f"No TCGA slide manifests found in {manifest_dir}")
    all_slides = pd.concat([read_manifest(path) for path in manifests], ignore_index=True)
    reps = pd.concat([select_representatives(sub) for _, sub in all_slides.groupby("cohort", sort=True)], ignore_index=True)
    pilot = select_pilot(reps, args.pilot_per_cohort, args.pilot_min_size_mb, args.pilot_max_size_gb)

    rep_path = output_dir / "tcga_wsi_representative_slide_manifest.tsv"
    pilot_path = output_dir / "tcga_wsi_pilot_slide_manifest.tsv"
    pilot_gdc_path = output_dir / "tcga_wsi_pilot_gdc_manifest.tsv"
    full_gdc_path = output_dir / "tcga_wsi_representative_gdc_manifest.tsv"
    reps.to_csv(rep_path, sep="\t", index=False)
    pilot.to_csv(pilot_path, sep="\t", index=False)
    write_gdc_manifest(pilot, pilot_gdc_path)
    write_gdc_manifest(reps, full_gdc_path)

    cohort_summary = []
    for cohort, sub in all_slides.groupby("cohort", sort=True):
        rep_sub = reps[reps["cohort"].eq(cohort)]
        pilot_sub = pilot[pilot["cohort"].eq(cohort)]
        cohort_summary.append(
            {
                "cohort": cohort,
                "all_slide_rows": int(len(sub)),
                "case_count": int(sub["patient_id"].nunique()),
                "all_slide_size_tib": float(sub["size"].sum() / 1024**4),
                "representative_rows": int(len(rep_sub)),
                "representative_size_gb": float(rep_sub["size"].sum() / 1024**3),
                "pilot_rows": int(len(pilot_sub)),
                "pilot_size_gb": float(pilot_sub["size"].sum() / 1024**3),
            }
        )
    summary = {
        "built_at": now_iso(),
        "status": "PASS",
        "all_slide_rows": int(len(all_slides)),
        "all_slide_size_tib": float(all_slides["size"].sum() / 1024**4),
        "representative_rows": int(len(reps)),
        "representative_size_gb": float(reps["size"].sum() / 1024**3),
        "pilot_rows": int(len(pilot)),
        "pilot_size_gb": float(pilot["size"].sum() / 1024**3),
        "cohort_summary": cohort_summary,
        "outputs": {
            "representative_manifest": str(rep_path.relative_to(root)),
            "representative_gdc_manifest": str(full_gdc_path.relative_to(root)),
            "pilot_manifest": str(pilot_path.relative_to(root)),
            "pilot_gdc_manifest": str(pilot_gdc_path.relative_to(root)),
        },
    }
    (output_dir / "tcga_wsi_manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
