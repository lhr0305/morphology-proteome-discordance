#!/usr/bin/env python3
"""Full-retraining label-permutation null for Stage 5 prediction outputs."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning

from stage5_train_morphology_omics import (
    RANDOM_SEED,
    crossfit_predict,
    encoder_role_from_metadata,
    feature_columns,
    load_embedding_features,
    pearson_vec,
    single_metadata_value,
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None
    return value


def align_target(x_meta: pd.DataFrame, target_path: Path) -> tuple[np.ndarray, list[str]]:
    target = pd.read_csv(target_path, sep="\t", dtype={"patient_id": str})
    target_cols = [col for col in target.columns if col not in {"cohort", "patient_id"}]
    aligned = x_meta.merge(target, on=["cohort", "patient_id"], how="left", validate="one_to_one")
    if len(aligned) != len(x_meta):
        raise RuntimeError(f"Target alignment changed row count for {target_path}: {len(aligned)} vs {len(x_meta)}")
    missing = aligned[target_cols].isna().any(axis=1)
    if missing.any():
        missing_ids = aligned.loc[missing, "patient_id"].tolist()
        raise RuntimeError(f"Target alignment has missing values for {target_path}: {missing_ids[:10]}")
    return aligned[target_cols].to_numpy(dtype=np.float32), target_cols


def parse_filter(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_target_pca_sets(value: str | None, target_specs: list[tuple[str, Path]]) -> set[str]:
    if value is None:
        return {target_set for target_set, _ in target_specs}
    return {item.strip() for item in value.split(",") if item.strip()}


def summarize_full_null(true_summary: pd.DataFrame, null_df: pd.DataFrame) -> pd.DataFrame:
    null_summary = (
        null_df.groupby(["cohort", "target_set"])
        .agg(
            full_null_mean_pearson_mean=("mean_pearson", "mean"),
            full_null_mean_pearson_sd=("mean_pearson", "std"),
            full_null_mean_pearson_p95=("mean_pearson", lambda s: float(np.percentile(s, 95))),
        )
        .reset_index()
    )
    out = true_summary.merge(null_summary, on=["cohort", "target_set"], how="left")
    empirical = []
    for row in out.itertuples(index=False):
        vals = null_df[(null_df["cohort"] == row.cohort) & (null_df["target_set"] == row.target_set)]["mean_pearson"].to_numpy(dtype=float)
        empirical.append(float((1 + np.sum(vals >= row.mean_pearson)) / (len(vals) + 1)))
    out["true_minus_full_null_mean_pearson"] = out["mean_pearson"] - out["full_null_mean_pearson_mean"]
    out["empirical_p_full_retrain_mean_pearson"] = empirical
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--embedding-table", required=True)
    parser.add_argument("--encoder-slug", required=True)
    parser.add_argument("--phospho-target-matrix", default="data/processed/stage3/kinase_ptm_activity.tsv")
    parser.add_argument("--phospho-target-set", default="phospho_kinase_ptm")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--target-pca", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--target-pca-target-sets",
        default=None,
        help="Comma-separated target sets to transform with training-fold PCA. Defaults to all target sets for backward compatibility.",
    )
    parser.add_argument("--target-pca-max-components", type=int, default=30)
    parser.add_argument("--target-pca-variance", type=float, default=0.80)
    parser.add_argument("--embedding-summary", choices=["mean", "mean_std"], default="mean")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--include-cohorts", default=None, help="Comma-separated cohort filter, e.g. CPTAC-PDAC")
    parser.add_argument("--include-target-sets", default=None, help="Comma-separated target_set filter")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing *_permutation_null.tsv file")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write the null TSV after this many new rows")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=LinAlgWarning)

    root = Path(args.project_root).resolve()
    pred_dir = root / args.prediction_dir
    if args.out_prefix:
        out_prefix = root / args.out_prefix
    else:
        out_prefix = pred_dir / "stage5_full_retrain_null"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    null_path = Path(f"{out_prefix}_permutation_null.tsv")
    summary_path = Path(f"{out_prefix}_summary_with_null.tsv")
    json_path = Path(f"{out_prefix}_validation.json")

    emb = pd.read_csv(root / args.embedding_table, sep="\t", dtype={"patient_id": str})
    x_meta = emb[["cohort", "patient_id"]].copy()
    x, emb_cols, embedding_input_table = load_embedding_features(root, emb, args.embedding_summary, root / "data/processed/stage5", args.embedding_table)
    model_repo = single_metadata_value(emb, "model_repo")
    encoder_role = encoder_role_from_metadata(model_repo, args.encoder_slug)

    target_specs = [
        ("protein_pathway", root / "data/processed/stage3/protein_pathway_matrix.tsv"),
        (args.phospho_target_set, root / args.phospho_target_matrix),
    ]
    include_targets = parse_filter(args.include_target_sets)
    if include_targets is not None:
        target_specs = [spec for spec in target_specs if spec[0] in include_targets]
    if not target_specs:
        raise RuntimeError("No target sets left after --include-target-sets filtering")
    target_pca_target_sets = parse_target_pca_sets(args.target_pca_target_sets, target_specs)

    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if args.resume and null_path.exists():
        existing = pd.read_csv(null_path, sep="\t")
        rows.extend(existing.to_dict(orient="records"))
        completed = {
            (str(row.cohort), str(row.target_set), int(row.permutation))
            for row in existing[["cohort", "target_set", "permutation"]].itertuples(index=False)
        }
        print(f"{datetime.now().isoformat()} resume loaded {len(existing)} existing null rows from {null_path}", flush=True)

    include_cohorts = parse_filter(args.include_cohorts)
    new_rows_since_checkpoint = 0
    for target_idx, (target_set, target_path) in enumerate(target_specs):
        y_full, target_cols = align_target(x_meta, target_path)
        cohorts = sorted(x_meta["cohort"].unique())
        if include_cohorts is not None:
            cohorts = [cohort for cohort in cohorts if cohort in include_cohorts]
        if not cohorts:
            raise RuntimeError("No cohorts left after --include-cohorts filtering")
        for cohort_idx, cohort in enumerate(cohorts):
            idx = np.flatnonzero(x_meta["cohort"].to_numpy() == cohort)
            x_sub = x[idx]
            y_sub = y_full[idx]
            patient_ids = x_meta.iloc[idx]["patient_id"].to_numpy(dtype=str)
            for perm in range(args.permutations):
                key = (cohort, target_set, perm)
                if key in completed:
                    continue
                rng = np.random.default_rng(args.seed + 900000 + target_idx * 100000 + cohort_idx * 1000 + perm)
                shuffled = rng.permutation(y_sub.shape[0])
                y_perm = y_sub[shuffled]
                pred_perm, _, _ = crossfit_predict(
                    x_sub,
                    y_perm,
                    patient_ids,
                    cohort=cohort,
                    target_set=target_set,
                    folds=args.folds,
                    repeats=args.repeats,
                    seed=args.seed,
                    target_pca=args.target_pca,
                    target_pca_target_sets=target_pca_target_sets,
                    target_pca_max_components=args.target_pca_max_components,
                    target_pca_variance=args.target_pca_variance,
                )
                pear = pearson_vec(y_sub, pred_perm)
                rows.append(
                    {
                        "cohort": cohort,
                        "target_set": target_set,
                        "permutation": perm,
                        "null_method": "full_retrain_label_shuffle",
                        "n_features": int(len(target_cols)),
                        "n_samples": int(len(idx)),
                        "mean_pearson": float(np.nanmean(pear)),
                        "median_pearson": float(np.nanmedian(pear)),
                        "p90_pearson": float(np.nanpercentile(pear, 90)),
                        "frac_pearson_gt0": float(np.nanmean(pear > 0)),
                    }
                )
                completed.add(key)
                new_rows_since_checkpoint += 1
                if new_rows_since_checkpoint >= max(1, args.checkpoint_every):
                    pd.DataFrame(rows).to_csv(null_path, sep="\t", index=False)
                    new_rows_since_checkpoint = 0
                if (perm + 1) % 10 == 0 or perm == 0:
                    print(f"{datetime.now().isoformat()} {cohort} {target_set} permutation {perm + 1}/{args.permutations}", flush=True)

    null_df = pd.DataFrame(rows)
    if null_df.empty:
        raise RuntimeError("No permutation rows were generated")
    null_df.to_csv(null_path, sep="\t", index=False)
    true_summary = pd.read_csv(pred_dir / "prediction_summary.tsv", sep="\t")
    combos = null_df[["cohort", "target_set"]].drop_duplicates()
    true_summary = true_summary.merge(combos, on=["cohort", "target_set"], how="inner")
    summary = summarize_full_null(true_summary, null_df)
    fail_rows = summary[
        ~(
            (summary["mean_pearson"] > summary["full_null_mean_pearson_p95"])
            & (summary["empirical_p_full_retrain_mean_pearson"] <= 0.05)
        )
    ].copy()
    status = "PASS" if fail_rows.empty else "FAIL"

    summary.to_csv(summary_path, sep="\t", index=False)
    validation = {
        "validated_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "fail_count": int(len(fail_rows)),
        "null_method": "full_retrain_label_shuffle",
        "prediction_dir": args.prediction_dir,
        "embedding_table": embedding_input_table,
        "encoder_slug": args.encoder_slug,
        "encoder_role": encoder_role,
        "folds": args.folds,
        "repeats": args.repeats,
        "permutations": args.permutations,
        "include_cohorts": sorted(include_cohorts) if include_cohorts is not None else None,
        "include_target_sets": sorted(include_targets) if include_targets is not None else None,
        "completed_rows": int(len(null_df)),
        "target_sets": [target_set for target_set, _ in target_specs],
        "target_pca_target_sets": sorted(target_pca_target_sets) if args.target_pca else [],
        "outputs": {
            "full_retrain_null": str(null_path.relative_to(root)),
            "summary_with_full_retrain_null": str(summary_path.relative_to(root)),
            "validation_json": str(json_path.relative_to(root)),
        },
        "failed_rows": fail_rows.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(to_jsonable(validation), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(to_jsonable({"status": status, "fail_count": len(fail_rows), "validation_json": str(json_path)}), ensure_ascii=False), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
