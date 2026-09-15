#!/usr/bin/env python3
"""Stage 5 cross-fitted morphology-to-omics prediction models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 20260609
ALPHAS = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)


def feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...] | None = None) -> list[str]:
    excluded = {"cohort", "patient_id", "tile_count", "embedding_path", "model_repo", "encoder_slug", "encoder_role", "embedding_dim"}
    cols = [col for col in df.columns if col not in excluded]
    if prefixes:
        cols = [col for col in cols if col.startswith(prefixes)]
    return cols


def encoder_role_from_metadata(model_repo: str, encoder_slug: str) -> str:
    if model_repo == "bioptimus/H-optimus-0" or encoder_slug == "bioptimus_H_optimus_0":
        return "proposal_default_H_Optimus_0_pathology_encoder"
    if model_repo == "prov-gigapath/prov-gigapath" or encoder_slug == "prov_gigapath_prov_gigapath":
        return "proposal_reproducibility_Prov_GigaPath_pathology_encoder"
    return "open_fallback_pathology_encoder_not_H_Optimus_or_Prov_GigaPath"


def single_metadata_value(df: pd.DataFrame, column: str, default: str = "") -> str:
    if column not in df.columns:
        return default
    values = sorted(str(value) for value in df[column].dropna().unique() if str(value))
    if not values:
        return default
    if len(values) > 1:
        raise RuntimeError(f"Expected one {column} value, observed {values}")
    return values[0]


def load_embedding_features(root: Path, emb: pd.DataFrame, mode: str, out_dir: Path, embedding_input_table: str) -> tuple[np.ndarray, list[str], str]:
    emb_cols = feature_columns(emb, prefixes=("emb_",))
    mean_matrix = emb[emb_cols].to_numpy(dtype=np.float32)
    if mode == "mean":
        return mean_matrix, emb_cols, embedding_input_table
    if mode != "mean_std":
        raise ValueError(f"Unsupported embedding summary mode: {mode}")

    import h5py

    encoder_slug = single_metadata_value(emb, "encoder_slug", Path(embedding_input_table).stem.replace("patient_mean_embeddings_", ""))
    std_rows = []
    for row in emb.itertuples(index=False):
        path = root / getattr(row, "embedding_path")
        with h5py.File(path, "r") as handle:
            arr = handle["embeddings"][:].astype(np.float32)
        std_rows.append(arr.std(axis=0))
    std_matrix = np.vstack(std_rows).astype(np.float32)
    std_cols = [f"embstd_{idx:04d}" for idx in range(std_matrix.shape[1])]
    feature_matrix = np.concatenate([mean_matrix, std_matrix], axis=1)
    feature_cols = emb_cols + std_cols
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"patient_embedding_features_{encoder_slug}_mean_std.tsv"
    meta_cols = [col for col in ["cohort", "patient_id", "tile_count", "embedding_path", "model_repo", "encoder_slug"] if col in emb.columns]
    feature_df = pd.concat(
        [
            emb[meta_cols].reset_index(drop=True),
            pd.DataFrame(feature_matrix, columns=feature_cols),
        ],
        axis=1,
    )
    feature_df.to_csv(output_path, sep="\t", index=False)
    return feature_matrix, feature_cols, output_path.relative_to(root).as_posix()


def pearson_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt**2).sum(axis=0) * (yp**2).sum(axis=0))
    out = np.full(y_true.shape[1], np.nan, dtype=float)
    ok = denom > 0
    out[ok] = (yt[:, ok] * yp[:, ok]).sum(axis=0) / denom[ok]
    return out


def spearman_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.apply_along_axis(rankdata, 0, y_true)
    yp = np.apply_along_axis(rankdata, 0, y_pred)
    return pearson_vec(yt, yp)


def crossfit_predict(
    x: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    cohort: str,
    target_set: str,
    folds: int,
    repeats: int,
    seed: int,
    target_pca: bool,
    target_pca_target_sets: set[str],
    target_pca_max_components: int,
    target_pca_variance: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    pred_sum = np.zeros_like(y, dtype=np.float64)
    pred_count = np.zeros(y.shape[0], dtype=np.int32)
    fold_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x)):
            n_components = max(1, min(256, x.shape[1], len(train_idx) - 1))
            inner_splits = max(2, min(3, len(train_idx) // 10))
            model = Pipeline(
                steps=[
                    ("x_scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components, random_state=seed + repeat)),
                    (
                        "ridge",
                        RidgeCV(
                            alphas=ALPHAS,
                            cv=KFold(n_splits=inner_splits, shuffle=True, random_state=seed + repeat + 1000),
                            scoring="neg_mean_squared_error",
                        ),
                    ),
                ]
            )
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y[train_idx])
            target_components = y_train.shape[1]
            target_variance_retained = 1.0
            use_target_pca = target_pca and target_set in target_pca_target_sets
            if use_target_pca:
                max_target_components = max(1, min(target_pca_max_components, y_train.shape[1], len(train_idx) - 1))
                pilot_pca = PCA(n_components=max_target_components, random_state=seed + repeat + 2000)
                pilot_pca.fit(y_train)
                cumulative = np.cumsum(pilot_pca.explained_variance_ratio_)
                target_components = int(min(max_target_components, np.searchsorted(cumulative, target_pca_variance) + 1))
                y_pca = PCA(n_components=target_components, random_state=seed + repeat + 3000)
                y_model_train = y_pca.fit_transform(y_train)
                target_variance_retained = float(np.sum(y_pca.explained_variance_ratio_))
                model.fit(x[train_idx], y_model_train)
                pred_model = model.predict(x[test_idx])
                if pred_model.ndim == 1:
                    pred_model = pred_model.reshape(-1, 1)
                pred_scaled = y_pca.inverse_transform(pred_model)
            else:
                model.fit(x[train_idx], y_train)
                pred_scaled = model.predict(x[test_idx])
            pred = y_scaler.inverse_transform(pred_scaled)
            pred_sum[test_idx] += pred
            pred_count[test_idx] += 1
            ridge = model.named_steps["ridge"]
            fold_rows.append(
                {
                    "cohort": cohort,
                    "target_set": target_set,
                    "repeat": repeat,
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "x_pca_components": int(n_components),
                    "inner_cv_splits": int(inner_splits),
                    "alpha": float(ridge.alpha_),
                    "target_transform": "training_fold_pca" if use_target_pca else "direct_scaled_targets",
                    "target_pca_components": int(target_components),
                    "target_pca_variance_retained": float(target_variance_retained),
                }
            )
            for patient_id in patient_ids[test_idx]:
                assignment_rows.append(
                    {
                        "cohort": cohort,
                        "patient_id": patient_id,
                        "target_set": target_set,
                        "repeat": repeat,
                        "fold": fold_idx,
                    }
                )
    if not np.all(pred_count == repeats):
        raise RuntimeError(f"OOF prediction count mismatch for {cohort}/{target_set}: {np.unique(pred_count)}")
    return pred_sum / pred_count[:, None], pd.DataFrame(fold_rows), pd.DataFrame(assignment_rows)


def feature_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    cohort: str,
    target_set: str,
) -> pd.DataFrame:
    pear = pearson_vec(y_true, y_pred)
    spear = spearman_vec(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    return pd.DataFrame(
        {
            "cohort": cohort,
            "target_set": target_set,
            "feature": feature_names,
            "n_samples": int(y_true.shape[0]),
            "pearson": pear,
            "spearman": spear,
            "mse": mse,
            "r2": r2,
        }
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, target_set), sub in metrics.groupby(["cohort", "target_set"]):
        pear = sub["pearson"].to_numpy(dtype=float)
        spear = sub["spearman"].to_numpy(dtype=float)
        rows.append(
            {
                "cohort": cohort,
                "target_set": target_set,
                "n_features": int(len(sub)),
                "n_samples": int(sub["n_samples"].iloc[0]),
                "mean_pearson": float(np.nanmean(pear)),
                "median_pearson": float(np.nanmedian(pear)),
                "p90_pearson": float(np.nanpercentile(pear, 90)),
                "frac_pearson_gt0": float(np.nanmean(pear > 0)),
                "mean_spearman": float(np.nanmean(spear)),
                "median_spearman": float(np.nanmedian(spear)),
                "mean_r2": float(np.nanmean(sub["r2"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def permutation_null(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cohort: str,
    target_set: str,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for perm in range(permutations):
        idx = rng.permutation(y_true.shape[0])
        pear = pearson_vec(y_true[idx], y_pred)
        rows.append(
            {
                "cohort": cohort,
                "target_set": target_set,
                "permutation": perm,
                "null_method": "posthoc_label_shuffle_on_oof_predictions",
                "mean_pearson": float(np.nanmean(pear)),
                "median_pearson": float(np.nanmedian(pear)),
                "p90_pearson": float(np.nanpercentile(pear, 90)),
                "frac_pearson_gt0": float(np.nanmean(pear > 0)),
            }
        )
    return pd.DataFrame(rows)


def write_prediction(path: Path, meta: pd.DataFrame, feature_names: list[str], matrix: np.ndarray) -> None:
    values = pd.DataFrame(matrix, columns=feature_names)
    out = pd.concat([meta[["cohort", "patient_id"]].reset_index(drop=True), values], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/prediction")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--target-pca", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--target-pca-target-sets",
        default="protein_pathway",
        help=(
            "Comma-separated target sets that receive training-fold target PCA. "
            "Default keeps protein pathway PCA while predicting phospho activity vectors directly, "
            "matching the Stage 5 proposal wording."
        ),
    )
    parser.add_argument("--target-pca-max-components", type=int, default=30)
    parser.add_argument("--target-pca-variance", type=float, default=0.80)
    parser.add_argument("--embedding-summary", choices=["mean", "mean_std"], default="mean")
    parser.add_argument("--embedding-table", default="data/processed/stage4/patient_mean_embeddings_owkin_phikon_v2.tsv")
    parser.add_argument("--encoder-slug", default="owkin_phikon_v2")
    parser.add_argument("--phospho-target-matrix", default="data/processed/stage3/kinase_ptm_activity.tsv")
    parser.add_argument("--phospho-target-set", default="phospho_kinase_ptm")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stage5_dir = root / "data/processed/stage5"
    stage4_dir = root / "data/processed/stage4"
    stage3_dir = root / "data/processed/stage3"

    embedding_table = root / args.embedding_table
    target_pca_target_sets = {item.strip() for item in args.target_pca_target_sets.split(",") if item.strip()}
    emb = pd.read_csv(embedding_table, sep="\t", dtype={"patient_id": str})
    x_meta = emb[["cohort", "patient_id"]].copy()
    expected_counts = {"CPTAC-COAD": 97, "CPTAC-PDAC": 139}
    cohort_counts = x_meta["cohort"].value_counts().sort_index().to_dict()
    if cohort_counts != expected_counts:
        raise RuntimeError(f"Stage 5 requires exactly {expected_counts}; observed {cohort_counts}")
    excluded_patients = {"C3L-04072", "C3L-04853"}
    present_excluded = sorted(excluded_patients & set(x_meta["patient_id"]))
    if present_excluded:
        raise RuntimeError(f"Stage 5 pathology exclusions present in embedding table: {present_excluded}")
    if "encoder_slug" in emb.columns and set(emb["encoder_slug"].dropna().unique()) != {args.encoder_slug}:
        raise RuntimeError(f"Unexpected encoder_slug values: {sorted(emb['encoder_slug'].dropna().unique())}")
    model_repo = single_metadata_value(emb, "model_repo")
    encoder_role = encoder_role_from_metadata(model_repo, args.encoder_slug)
    x, emb_cols, embedding_input_table = load_embedding_features(root, emb, args.embedding_summary, stage5_dir, args.embedding_table)

    target_specs = [
        {
            "target_set": "protein_pathway",
            "input": stage3_dir / "protein_pathway_matrix.tsv",
            "predicted": out_dir / "predicted_protein_pathway_oof.tsv",
            "observed": out_dir / "observed_protein_pathway_aligned.tsv",
            "compat": out_dir / "predicted_protein_oof.tsv",
        },
        {
            "target_set": args.phospho_target_set,
            "input": root / args.phospho_target_matrix,
            "predicted": out_dir / "predicted_phospho_kinase_ptm_oof.tsv",
            "observed": out_dir / "observed_phospho_kinase_ptm_aligned.tsv",
            "compat": out_dir / "predicted_phospho_oof.tsv",
        },
    ]

    all_metrics = []
    all_folds = []
    all_assignments = []
    all_null = []
    for spec in target_specs:
        target = pd.read_csv(spec["input"], sep="\t", dtype={"patient_id": str})
        target_cols = [col for col in target.columns if col not in {"cohort", "patient_id"}]
        aligned = x_meta.merge(target, on=["cohort", "patient_id"], how="left", validate="one_to_one")
        if len(aligned) != len(x_meta):
            raise RuntimeError(f"Target alignment changed row count for {spec['target_set']}: {len(aligned)} vs {len(x_meta)}")
        missing_target = aligned[target_cols].isna().any(axis=1)
        if missing_target.any():
            missing_ids = aligned.loc[missing_target, "patient_id"].tolist()
            raise RuntimeError(f"Target alignment has missing values for {spec['target_set']}: {missing_ids[:10]}")
        y_full = aligned[target_cols].to_numpy(dtype=np.float32)
        pred_full = np.zeros_like(y_full, dtype=np.float64)
        write_prediction(spec["observed"], aligned[["cohort", "patient_id"]], target_cols, y_full)
        for cohort in sorted(aligned["cohort"].unique()):
            idx = np.flatnonzero(aligned["cohort"].to_numpy() == cohort)
            pred, folds_df, assignments_df = crossfit_predict(
                x[idx],
                y_full[idx],
                aligned.iloc[idx]["patient_id"].to_numpy(dtype=str),
                cohort=cohort,
                target_set=spec["target_set"],
                folds=args.folds,
                repeats=args.repeats,
                seed=args.seed,
                target_pca=args.target_pca,
                target_pca_target_sets=target_pca_target_sets,
                target_pca_max_components=args.target_pca_max_components,
                target_pca_variance=args.target_pca_variance,
            )
            pred_full[idx] = pred
            all_folds.append(folds_df)
            all_assignments.append(assignments_df)
            all_metrics.append(feature_metrics(y_full[idx], pred, target_cols, cohort, spec["target_set"]))
            all_null.append(permutation_null(y_full[idx], pred, cohort, spec["target_set"], args.permutations, args.seed + len(all_null) * 10000))
        write_prediction(spec["predicted"], aligned[["cohort", "patient_id"]], target_cols, pred_full)
        write_prediction(spec["compat"], aligned[["cohort", "patient_id"]], target_cols, pred_full)

    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize_metrics(metrics)
    folds = pd.concat(all_folds, ignore_index=True)
    assignments = pd.concat(all_assignments, ignore_index=True)
    null = pd.concat(all_null, ignore_index=True)
    metrics.to_csv(out_dir / "prediction_feature_metrics.tsv", sep="\t", index=False)
    metrics.to_csv(out_dir / "prediction_metrics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "prediction_summary.tsv", sep="\t", index=False)
    folds.to_csv(out_dir / "stage5_cv_folds.tsv", sep="\t", index=False)
    assignments.to_csv(out_dir / "stage5_oof_assignments.tsv", sep="\t", index=False)
    null.to_csv(out_dir / "prediction_permutation_null.tsv", sep="\t", index=False)

    null_summary = null.groupby(["cohort", "target_set"]).agg(
        null_mean_pearson_mean=("mean_pearson", "mean"),
        null_mean_pearson_sd=("mean_pearson", "std"),
        null_mean_pearson_p95=("mean_pearson", lambda s: float(np.percentile(s, 95))),
    ).reset_index()
    merged_summary = summary.merge(null_summary, on=["cohort", "target_set"], how="left")
    merged_summary["true_minus_null_mean_pearson"] = merged_summary["mean_pearson"] - merged_summary["null_mean_pearson_mean"]
    p_values = []
    for row in merged_summary.itertuples(index=False):
        null_values = null[(null["cohort"] == row.cohort) & (null["target_set"] == row.target_set)]["mean_pearson"].to_numpy(dtype=float)
        p_values.append(float((1 + np.sum(null_values >= row.mean_pearson)) / (len(null_values) + 1)))
    merged_summary["empirical_p_mean_pearson"] = p_values
    merged_summary.to_csv(out_dir / "prediction_summary_with_null.tsv", sep="\t", index=False)

    skipped_targets = pd.DataFrame(
        [
            {
                "target_set": "rna_pathway",
                "status": "SKIPPED",
                "reason": "No matched CPTAC COAD/PDAC RNA pathway matrix is available in Stage 3; TCGA RNA pathway matrix is external validation material for later stages.",
            }
        ]
    )
    skipped_targets.to_csv(out_dir / "prediction_skipped_targets.tsv", sep="\t", index=False)

    build_summary = {
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "encoder_slug": args.encoder_slug,
        "model_repo": model_repo,
        "encoder_role": encoder_role,
        "samples": int(len(x_meta)),
        "cohort_counts": cohort_counts,
        "excluded_pathology_patients": sorted(excluded_patients),
        "embedding_features": int(len(emb_cols)),
        "embedding_dim": int(len([col for col in emb.columns if col.startswith("emb_")])),
        "embedding_summary": args.embedding_summary,
        "embedding_input_table": embedding_input_table,
        "target_sets": [spec["target_set"] for spec in target_specs],
        "target_sources": {spec["target_set"]: spec["input"].relative_to(root).as_posix() for spec in target_specs},
        "skipped_target_sets": skipped_targets.to_dict(orient="records"),
        "folds": args.folds,
        "repeats": args.repeats,
        "permutations": args.permutations,
        "permutation_null_method": "posthoc_label_shuffle_on_oof_predictions",
        "target_transform": "mixed_by_target_set" if args.target_pca and target_pca_target_sets else "direct_scaled_targets",
        "target_pca_target_sets": sorted(target_pca_target_sets) if args.target_pca else [],
        "target_pca_max_components": args.target_pca_max_components,
        "target_pca_variance": args.target_pca_variance,
        "alpha_grid": ALPHAS.tolist(),
        "outputs": {
            "prediction_feature_metrics": (out_dir / "prediction_feature_metrics.tsv").relative_to(root).as_posix(),
            "prediction_summary": (out_dir / "prediction_summary.tsv").relative_to(root).as_posix(),
            "prediction_summary_with_null": (out_dir / "prediction_summary_with_null.tsv").relative_to(root).as_posix(),
            "stage5_cv_folds": (out_dir / "stage5_cv_folds.tsv").relative_to(root).as_posix(),
            "stage5_oof_assignments": (out_dir / "stage5_oof_assignments.tsv").relative_to(root).as_posix(),
            "prediction_permutation_null": (out_dir / "prediction_permutation_null.tsv").relative_to(root).as_posix(),
            "prediction_skipped_targets": (out_dir / "prediction_skipped_targets.tsv").relative_to(root).as_posix(),
        },
    }
    (out_dir / "stage5_prediction_build_summary.json").write_text(
        json.dumps(build_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(build_summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
