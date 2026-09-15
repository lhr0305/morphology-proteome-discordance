#!/usr/bin/env python3
"""Aggregate all-candidate slide embeddings into patient-level multi-slide features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

META = ["cohort", "patient_id"]


def read_slide_mean(path: Path) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as h5:
        if "embeddings" not in h5:
            raise ValueError(f"missing embeddings dataset: {path}")
        arr = h5["embeddings"][:].astype(np.float32)
        model_repo = str(h5.attrs.get("model_repo", ""))
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"bad embedding shape {arr.shape}: {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite embeddings: {path}")
    return arr.mean(axis=0, dtype=np.float64).astype(np.float32), model_repo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--encoder-slug", required=True)
    parser.add_argument("--aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--expected-dim", type=int, default=1536)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    out_prefix = Path(args.output_prefix)
    if not out_prefix.is_absolute():
        out_prefix = root / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t", dtype={"patient_id": str}).fillna("")
    manifest = manifest[manifest["status"].isin(["PASS", "SKIPPED_EXISTS"])].copy()
    if manifest.empty:
        raise RuntimeError("No PASS/SKIPPED_EXISTS rows in manifest")

    slide_rows = []
    slide_vectors = []
    for row in manifest.to_dict("records"):
        emb_path = root / str(row["embedding_path"])
        vec, model_repo = read_slide_mean(emb_path)
        if vec.shape[0] != args.expected_dim:
            raise ValueError(f"embedding dim mismatch {vec.shape[0]} != {args.expected_dim}: {emb_path}")
        slide_rows.append(
            {
                "cohort": row.get("cohort", ""),
                "patient_id": row.get("patient_id", ""),
                "slide_candidate_key": row.get("slide_candidate_key", ""),
                "slide_id": row.get("slide_id", ""),
                "tile_count": int(float(row.get("tile_count", 0) or 0)),
                "embedding_path": row.get("embedding_path", ""),
                "model_repo": model_repo,
            }
        )
        slide_vectors.append(vec)
    slide_matrix = np.vstack(slide_vectors).astype(np.float32)
    emb_cols = [f"emb_{i:04d}" for i in range(slide_matrix.shape[1])]
    slide_df = pd.concat([pd.DataFrame(slide_rows), pd.DataFrame(slide_matrix, columns=emb_cols)], axis=1)
    slide_out = out_prefix.with_name(out_prefix.name + "_slide_mean_embeddings.tsv")
    patient_out = out_prefix.with_name(out_prefix.name + f"_patient_multislide_{args.aggregation}_embeddings.tsv")
    summary_out = out_prefix.with_name(out_prefix.name + "_aggregation_summary.json")
    slide_df.to_csv(slide_out, sep="\t", index=False)

    patient_records = []
    for (cohort, patient_id), group in slide_df.groupby(META, sort=True):
        values = group[emb_cols].to_numpy(dtype=np.float32)
        if args.aggregation == "median":
            agg = np.median(values, axis=0)
        else:
            agg = values.mean(axis=0)
        model_repos = sorted(str(value) for value in group["model_repo"].dropna().unique() if str(value))
        if len(model_repos) != 1:
            raise ValueError(f"expected one model_repo for {cohort}/{patient_id}, observed {model_repos}")
        rec = {
            "cohort": cohort,
            "patient_id": patient_id,
            "encoder_slug": args.encoder_slug,
            "model_repo": model_repos[0],
            "embedding_summary": f"multi_slide_{args.aggregation}_of_slide_means",
            "slide_count": int(len(group)),
            "tile_count_sum": int(group["tile_count"].sum()),
        }
        rec.update({col: float(val) for col, val in zip(emb_cols, agg)})
        patient_records.append(rec)
    patient_df = pd.DataFrame(patient_records)
    patient_df.to_csv(patient_out, sep="\t", index=False)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "manifest": str(manifest_path.relative_to(root) if manifest_path.is_relative_to(root) else manifest_path),
        "encoder_slug": args.encoder_slug,
        "model_repo_values": sorted(slide_df["model_repo"].dropna().unique().tolist()),
        "aggregation": args.aggregation,
        "slide_rows": int(len(slide_df)),
        "patient_rows": int(len(patient_df)),
        "embedding_dim": int(len(emb_cols)),
        "slide_count_min": int(patient_df["slide_count"].min()),
        "slide_count_median": float(patient_df["slide_count"].median()),
        "slide_count_max": int(patient_df["slide_count"].max()),
        "outputs": {
            "slide_mean_embeddings": str(slide_out.relative_to(root)),
            "patient_embeddings": str(patient_out.relative_to(root)),
            "summary": str(summary_out.relative_to(root)),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
