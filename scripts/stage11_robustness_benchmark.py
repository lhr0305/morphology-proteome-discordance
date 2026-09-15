#!/usr/bin/env python3
"""Stage 11 robustness, null-model, fusion benchmark, and artifact checks."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 20260609
META_COLUMNS = {"cohort", "patient_id"}


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"patient_id": str})


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def finite_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=1)
    if not np.isfinite(sd) or sd <= 1e-10:
        sd = 1.0
    return (arr - mean) / sd


def cohort_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        z = np.full(len(out), np.nan, dtype=float)
        for cohort in sorted(out["cohort"].dropna().unique()):
            idx = out["cohort"].to_numpy() == cohort
            z[idx] = zscore(pd.to_numeric(out.loc[idx, col], errors="coerce").to_numpy(dtype=float))
        out[f"{col}_z"] = z
    return out


def bh_adjust(p_values: list[float] | np.ndarray) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return out.tolist()
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[ok] = restored
    return out.tolist()


def safe_metric_row(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auc": np.nan, "average_precision": np.nan, "brier": np.nan}
    return {
        "auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }


def choose_cv_splits(y: np.ndarray, requested: int) -> int:
    counts = np.bincount(y.astype(int), minlength=2)
    return int(max(2, min(requested, counts.min())))


def logistic_coef(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[keep]
    y = y[keep]
    if len(y) == 0 or len(np.unique(y.astype(int))) < 2:
        return {"status": "SKIP", "n": int(len(y)), "reason": "less_than_two_classes"}
    design = np.column_stack([np.ones(len(y)), x])
    if min(int(y.sum()), int(len(y) - y.sum())) < design.shape[1] + 1:
        return {"status": "SKIP", "n": int(len(y)), "reason": "insufficient_events_per_parameter"}

    def nll(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(-np.sum(y * eta - np.logaddexp(0.0, eta)) + 0.5e-6 * np.sum(beta[1:] ** 2))

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = design @ beta
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        g = design.T @ (prob - y)
        g[1:] += 1e-6 * beta[1:]
        return g

    start = np.zeros(design.shape[1], dtype=float)
    rate = np.clip(y.mean(), 1e-5, 1 - 1e-5)
    start[0] = math.log(rate / (1 - rate))
    res = optimize.minimize(nll, start, jac=grad, method="BFGS", options={"maxiter": 500})
    beta = np.asarray(res.x, dtype=float)
    eta = design @ beta
    prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    w = prob * (1 - prob)
    hessian = design.T @ (design * w[:, None])
    hessian[1:, 1:] += 1e-6 * np.eye(design.shape[1] - 1)
    cov = np.linalg.pinv(hessian)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    return {
        "status": "PASS" if res.success else "WARN",
        "n": int(len(y)),
        "events": int(y.sum()),
        "nonevents": int(len(y) - y.sum()),
        "coef": float(beta[1]) if len(beta) > 1 else np.nan,
        "se": float(se[1]) if len(se) > 1 else np.nan,
        "z": float(z[1]) if len(z) > 1 else np.nan,
        "p": float(p[1]) if len(p) > 1 else np.nan,
        "or": float(np.exp(beta[1])) if len(beta) > 1 else np.nan,
    }


def fit_cv_benchmark(
    model_name: str,
    feature_df: pd.DataFrame,
    clinical: pd.DataFrame,
    feature_mode: str,
    input_path: str,
    max_pcs: int,
    folds: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = feature_df.merge(
        clinical[["cohort", "patient_id", "advanced_at_presentation"]],
        on=["cohort", "patient_id"],
        how="inner",
        validate="one_to_one",
    )
    feat_cols = feature_columns(feature_df)
    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for cohort in sorted(merged["cohort"].dropna().unique()):
        sub = merged[merged["cohort"] == cohort].copy().reset_index(drop=True)
        x = sub[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["advanced_at_presentation"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
        sub = sub.loc[keep].reset_index(drop=True)
        x = x[keep]
        y = y[keep].astype(int)
        if len(y) == 0 or len(np.unique(y)) < 2:
            metric_rows.append(
                {
                    "benchmark_model": model_name,
                    "cohort": cohort,
                    "status": "SKIP",
                    "skip_reason": "less_than_two_classes_or_no_data",
                    "n": int(len(y)),
                    "events": int(y.sum()) if len(y) else 0,
                    "nonevents": int(len(y) - y.sum()) if len(y) else 0,
                    "n_features": len(feat_cols),
                    "feature_mode": feature_mode,
                    "input_path": input_path,
                }
            )
            continue
        splits = choose_cv_splits(y, folds)
        pred_sum = np.zeros(len(y), dtype=float)
        pred_count = np.zeros(len(y), dtype=int)
        for repeat in range(repeats):
            cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED + repeat)
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y)):
                if feature_mode == "pca_logit":
                    n_pca = max(1, min(max_pcs, x.shape[1], len(train_idx) - 1))
                    pipe = Pipeline(
                        steps=[
                            ("scale", StandardScaler()),
                            ("pca", PCA(n_components=n_pca, random_state=RANDOM_SEED + repeat + fold)),
                            (
                                "logit",
                                LogisticRegression(
                                    penalty="l2",
                                    C=1.0,
                                    solver="lbfgs",
                                    max_iter=2000,
                                    class_weight="balanced",
                                    random_state=RANDOM_SEED + repeat + fold,
                                ),
                            ),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[
                            ("scale", StandardScaler()),
                            (
                                "logit",
                                LogisticRegression(
                                    penalty="l2",
                                    C=1.0,
                                    solver="lbfgs",
                                    max_iter=2000,
                                    class_weight="balanced",
                                    random_state=RANDOM_SEED + repeat + fold,
                                ),
                            ),
                        ]
                    )
                pipe.fit(x[train_idx], y[train_idx])
                pred_sum[test_idx] += pipe.predict_proba(x[test_idx])[:, 1]
                pred_count[test_idx] += 1
        if not np.all(pred_count == repeats):
            raise RuntimeError(f"{model_name} {cohort} OOF predictions incomplete")
        prob = pred_sum / pred_count
        metrics = safe_metric_row(y, prob)
        metric_rows.append(
            {
                "benchmark_model": model_name,
                "cohort": cohort,
                "status": "PASS",
                "skip_reason": "",
                "n": int(len(y)),
                "events": int(y.sum()),
                "nonevents": int(len(y) - y.sum()),
                "n_features": len(feat_cols),
                "feature_mode": feature_mode,
                "cv_folds": int(splits),
                "cv_repeats": int(repeats),
                "auc": metrics["auc"],
                "average_precision": metrics["average_precision"],
                "brier": metrics["brier"],
                "input_path": input_path,
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "benchmark_model": model_name,
                    "cohort": sub["cohort"],
                    "patient_id": sub["patient_id"],
                    "advanced_at_presentation": y,
                    "predicted_probability": prob,
                }
            )
        )
    pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if not pred.empty and pred["advanced_at_presentation"].nunique() > 1:
        pooled = safe_metric_row(pred["advanced_at_presentation"].to_numpy(dtype=int), pred["predicted_probability"].to_numpy(dtype=float))
        metric_rows.append(
            {
                "benchmark_model": model_name,
                "cohort": "pooled",
                "status": "PASS",
                "skip_reason": "",
                "n": int(len(pred)),
                "events": int(pred["advanced_at_presentation"].sum()),
                "nonevents": int(len(pred) - pred["advanced_at_presentation"].sum()),
                "n_features": len(feat_cols),
                "feature_mode": feature_mode,
                "cv_folds": int(folds),
                "cv_repeats": int(repeats),
                "auc": pooled["auc"],
                "average_precision": pooled["average_precision"],
                "brier": pooled["brier"],
                "input_path": input_path,
            }
        )
    return pd.DataFrame(metric_rows), pred


def summarize_prediction_dir(root: Path, rel_dir: str, label: str) -> pd.DataFrame:
    path = root / rel_dir / "prediction_summary_with_null.tsv"
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "analysis": label,
                    "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                    "reason": f"missing {path.relative_to(root)}",
                }
            ]
        )
    df = read_tsv(path)
    df.insert(0, "analysis", label)
    df.insert(1, "prediction_dir", rel_dir)
    df["passes_posthoc_null_gate"] = (df["mean_pearson"] > df["null_mean_pearson_p95"]) & (
        df["empirical_p_mean_pearson"] <= 0.05
    )
    df["status"] = np.where(df["passes_posthoc_null_gate"], "PASS", "WARN")
    return df


def build_pdc000109_label_free_repeat(root: Path, out_dir: Path) -> dict[str, Any]:
    """Use downloaded PDC000109 Protein_Assembly files for a COAD label-free sanity check."""
    summary_path = root / "data/raw/pdc/PDC000109/Protein_Assembly/CPTAC2_Prospective_Colon_VU_Proteome.summary.tsv"
    sample_path = root / "data/raw/pdc/PDC000109/Protein_Assembly/CPTAC2_Prospective_Colon_VU_Proteome.sample.txt"
    biospecimen_path = root / "manifests/pdc/PDC000109_biospecimen_metadata.tsv"
    primary_path = root / "data/processed/stage3/protein_matrix.tsv"
    matrix_out = out_dir / "pdc000109_label_free_spectral_count_matrix.tsv"
    concordance_out = out_dir / "pdc000109_label_free_technical_repeat.tsv"
    summary_out = out_dir / "pdc000109_label_free_technical_repeat_summary.tsv"

    required = [summary_path, sample_path, biospecimen_path, primary_path]
    missing = [str(p.relative_to(root)) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        pd.DataFrame([{"status": "SKIPPED_INPUT_NOT_AVAILABLE", "missing": ";".join(missing)}]).to_csv(
            summary_out, sep="\t", index=False
        )
        return {
            "category": "platform_robustness",
            "analysis": "PDC000109_label_free_COAD_technical_repeat",
            "cohort": "CPTAC-COAD",
            "target_set": "label_free_proteome",
            "status": "SKIPPED_INPUT_NOT_AVAILABLE",
            "primary_metric": "required_file_count",
            "primary_value": len(required) - len(missing),
            "null_reference": len(required),
            "empirical_p": np.nan,
            "direction": "",
            "input_path": "data/raw/pdc/PDC000109/Protein_Assembly; manifests/pdc/PDC000109_biospecimen_metadata.tsv",
            "note": f"PDC000109 Protein_Assembly inputs missing: {missing}",
        }

    summary = pd.read_csv(summary_path, sep="\t")
    sample_map = pd.read_csv(sample_path, sep="\t", dtype=str)
    biospecimen = pd.read_csv(biospecimen_path, sep="\t", dtype=str)
    sample_id_col = sample_map.columns[2]
    map_df = sample_map.merge(
        biospecimen[["aliquot_submitter_id", "case_submitter_id", "sample_type"]],
        left_on=sample_id_col,
        right_on="aliquot_submitter_id",
        how="left",
        validate="many_to_one",
    )
    sample_to_patient = dict(zip(map_df["AnalyticalSample"], map_df["case_submitter_id"]))
    sample_to_type = dict(zip(map_df["AnalyticalSample"], map_df["sample_type"]))
    spectral_cols = [
        c
        for c in summary.columns
        if c.endswith(" Spectral Counts")
        and "_CPTAC_COprospective_Proteome_VU_" in c
        and c.rsplit(" Spectral Counts", 1)[0] in sample_to_patient
    ]
    if not spectral_cols:
        pd.DataFrame([{"status": "SKIPPED_INPUT_NOT_AVAILABLE", "reason": "no mapped spectral-count sample columns"}]).to_csv(
            summary_out, sep="\t", index=False
        )
        return {
            "category": "platform_robustness",
            "analysis": "PDC000109_label_free_COAD_technical_repeat",
            "cohort": "CPTAC-COAD",
            "target_set": "label_free_proteome",
            "status": "SKIPPED_INPUT_NOT_AVAILABLE",
            "primary_metric": "mapped_sample_count",
            "primary_value": 0,
            "null_reference": np.nan,
            "empirical_p": np.nan,
            "direction": "",
            "input_path": str(summary_path.relative_to(root)),
            "note": "PDC000109 summary table did not expose mapped spectral-count sample columns.",
        }

    mat = summary[["Gene", *spectral_cols]].copy()
    mat = mat[mat["Gene"].notna() & (mat["Gene"].astype(str).str.len() > 0)]
    for col in spectral_cols:
        mat[col] = pd.to_numeric(mat[col], errors="coerce").fillna(0.0)
    mat = mat.groupby("Gene", as_index=True)[spectral_cols].sum()
    mat = np.log1p(mat)
    mat.columns = [c.rsplit(" Spectral Counts", 1)[0] for c in mat.columns]
    long = mat.T
    long.index.name = "analytical_sample"
    lf = long.reset_index()
    lf["patient_id"] = lf["analytical_sample"].map(sample_to_patient)
    lf["sample_type"] = lf["analytical_sample"].map(sample_to_type)
    lf = lf[lf["patient_id"].notna() & lf["sample_type"].astype(str).str.contains("Primary Tumor", case=False, na=False)]
    lf = lf.drop(columns=["sample_type"])
    gene_cols = [c for c in lf.columns if c not in {"analytical_sample", "patient_id"}]
    lf = lf.groupby("patient_id", as_index=False)[gene_cols].mean()
    lf.insert(0, "cohort", "CPTAC-COAD")
    lf.to_csv(matrix_out, sep="\t", index=False)

    primary = read_tsv(primary_path)
    primary = primary[primary["cohort"] == "CPTAC-COAD"].copy()
    merged = lf.merge(primary, on=["cohort", "patient_id"], suffixes=("_pdc000109", "_primary"), how="inner")
    shared_genes = sorted(set(gene_cols) & set(feature_columns(primary)))
    patient_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        x = pd.to_numeric(row[[f"{g}_pdc000109" for g in shared_genes]], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(row[[f"{g}_primary" for g in shared_genes]], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() >= 30 and np.nanstd(x[keep]) > 0 and np.nanstd(y[keep]) > 0:
            rho, p = stats.spearmanr(x[keep], y[keep])
        else:
            rho, p = np.nan, np.nan
        patient_rows.append(
            {
                "cohort": row["cohort"],
                "patient_id": row["patient_id"],
                "n_shared_genes": int(keep.sum()),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(p) if np.isfinite(p) else np.nan,
            }
        )
    concordance = pd.DataFrame(patient_rows)
    concordance.to_csv(concordance_out, sep="\t", index=False)
    valid = concordance["spearman_rho"].dropna()
    positives = int((valid > 0).sum())
    binom_p = stats.binomtest(positives, len(valid), 0.5, alternative="greater").pvalue if len(valid) else np.nan
    median_rho = float(valid.median()) if len(valid) else np.nan
    summary_row = {
        "status": "PASS" if len(valid) >= 20 and np.isfinite(median_rho) and median_rho > 0 and np.isfinite(binom_p) and binom_p <= 0.05 else "WARN",
        "input_summary": str(summary_path.relative_to(root)),
        "input_sample_map": str(sample_path.relative_to(root)),
        "input_biospecimen": str(biospecimen_path.relative_to(root)),
        "matrix_output": str(matrix_out.relative_to(root)),
        "concordance_output": str(concordance_out.relative_to(root)),
        "label_free_patients": int(len(lf)),
        "matched_primary_patients": int(len(valid)),
        "label_free_genes": int(len(gene_cols)),
        "shared_genes_with_primary": int(len(shared_genes)),
        "median_patient_spearman_rho": median_rho,
        "positive_patient_count": positives,
        "binomial_p_positive_rho": float(binom_p) if np.isfinite(binom_p) else np.nan,
    }
    pd.DataFrame([summary_row]).to_csv(summary_out, sep="\t", index=False)
    return {
        "category": "platform_robustness",
        "analysis": "PDC000109_label_free_COAD_technical_repeat",
        "cohort": "CPTAC-COAD",
        "target_set": "label_free_proteome_spectral_count",
        "status": summary_row["status"],
        "primary_metric": "median_patient_spearman_rho",
        "primary_value": median_rho,
        "null_reference": 0.0,
        "empirical_p": summary_row["binomial_p_positive_rho"],
        "direction": "positive" if np.isfinite(median_rho) and median_rho > 0 else "non_positive",
        "input_path": f"{summary_path.relative_to(root)}; {sample_path.relative_to(root)}; {biospecimen_path.relative_to(root)}",
        "note": (
            f"Downloaded PDC000109 Protein_Assembly label-free files; patient-level log1p spectral-count matrix has "
            f"{summary_row['label_free_patients']} tumor patients and {summary_row['label_free_genes']} genes; "
            f"{summary_row['matched_primary_patients']} patients and {summary_row['shared_genes_with_primary']} genes overlap the primary COAD protein matrix."
        ),
    }


def build_macenko_preprocessing_robustness(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Compare the no-stain-normalization representative baseline with Macenko embeddings."""
    baseline_rel = "results/prediction_prov_gigapath_exact_phospho_direct_strict_mean"
    macenko_rel = "results/prediction_prov_gigapath_macenko_repr_exact_phospho_direct_sensitivity"
    baseline_path = root / baseline_rel / "prediction_summary_with_null.tsv"
    macenko_path = root / macenko_rel / "prediction_summary_with_null.tsv"
    manifest_path = root / "data/processed/stage4/embedding_manifest_prov_gigapath_prov_gigapath_macenko_repr.tsv"
    summary_path = root / "data/processed/stage4/embedding_summary_prov_gigapath_prov_gigapath_macenko_repr.json"
    out_path = out_dir / "macenko_preprocessing_sensitivity.tsv"

    if not macenko_path.exists() or macenko_path.stat().st_size == 0:
        skipped = pd.DataFrame(
            [
                {
                    "cohort": "CPTAC-COAD+CPTAC-PDAC",
                    "target_set": "pathology_embedding",
                    "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                    "skip_reason": f"missing {macenko_path.relative_to(root)}",
                    "baseline_mean_pearson": np.nan,
                    "macenko_mean_pearson": np.nan,
                    "delta_mean_pearson": np.nan,
                    "macenko_empirical_p_mean_pearson": np.nan,
                    "macenko_null_mean_pearson_p95": np.nan,
                    "manifest_pass_rows": np.nan,
                    "manifest_total_rows": np.nan,
                }
            ]
        )
        skipped.to_csv(out_path, sep="\t", index=False)
        return [
            {
                "category": "preprocessing_robustness",
                "analysis": "no_stain_norm_vs_Macenko",
                "cohort": "CPTAC-COAD+CPTAC-PDAC",
                "target_set": "pathology_embedding",
                "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                "primary_metric": "macenko_prediction_summary_available",
                "primary_value": 0.0,
                "null_reference": np.nan,
                "empirical_p": np.nan,
                "direction": "",
                "input_path": str(macenko_path.relative_to(root)),
                "note": "Macenko-normalized Stage 5 prediction summary is not available yet.",
            }
        ]

    macenko = read_tsv(macenko_path)
    baseline = read_tsv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    if not baseline.empty:
        baseline = baseline.rename(
            columns={
                "mean_pearson": "baseline_mean_pearson",
                "empirical_p_mean_pearson": "baseline_empirical_p_mean_pearson",
                "null_mean_pearson_p95": "baseline_null_mean_pearson_p95",
            }
        )
        merged = macenko.merge(
            baseline[
                [
                    "cohort",
                    "target_set",
                    "baseline_mean_pearson",
                    "baseline_empirical_p_mean_pearson",
                    "baseline_null_mean_pearson_p95",
                ]
            ],
            on=["cohort", "target_set"],
            how="left",
            validate="one_to_one",
        )
    else:
        merged = macenko.copy()
        merged["baseline_mean_pearson"] = np.nan
        merged["baseline_empirical_p_mean_pearson"] = np.nan
        merged["baseline_null_mean_pearson_p95"] = np.nan

    manifest_rows = np.nan
    manifest_pass = np.nan
    macenko_fit_counts = ""
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        manifest = read_tsv(manifest_path)
        manifest_rows = int(len(manifest))
        manifest_pass = int((manifest.get("status", pd.Series(dtype=str)).astype(str) == "PASS").sum())
        if "macenko_fit_status" in manifest.columns:
            macenko_fit_counts = ";".join(
                f"{k}:{v}" for k, v in sorted(manifest["macenko_fit_status"].fillna("").astype(str).value_counts().to_dict().items())
            )

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        mean_pearson = float(row.get("mean_pearson", np.nan))
        null_p95 = float(row.get("null_mean_pearson_p95", np.nan))
        empirical_p = float(row.get("empirical_p_mean_pearson", np.nan))
        passed_null = bool(np.isfinite(mean_pearson) and np.isfinite(null_p95) and mean_pearson > null_p95 and empirical_p <= 0.05)
        status = "PASS" if passed_null and mean_pearson > 0 else "WARN"
        delta = mean_pearson - float(row.get("baseline_mean_pearson", np.nan))
        detail = {
            "cohort": row.get("cohort", ""),
            "target_set": row.get("target_set", ""),
            "status": status,
            "skip_reason": "",
            "baseline_mean_pearson": row.get("baseline_mean_pearson", np.nan),
            "baseline_empirical_p_mean_pearson": row.get("baseline_empirical_p_mean_pearson", np.nan),
            "baseline_null_mean_pearson_p95": row.get("baseline_null_mean_pearson_p95", np.nan),
            "macenko_mean_pearson": mean_pearson,
            "macenko_median_pearson": row.get("median_pearson", np.nan),
            "macenko_empirical_p_mean_pearson": empirical_p,
            "macenko_null_mean_pearson_p95": null_p95,
            "delta_mean_pearson": delta,
            "manifest_pass_rows": manifest_pass,
            "manifest_total_rows": manifest_rows,
            "macenko_fit_status_counts": macenko_fit_counts,
            "manifest_path": str(manifest_path.relative_to(root)) if manifest_path.exists() else "",
            "summary_path": str(summary_path.relative_to(root)) if summary_path.exists() else "",
            "prediction_summary_path": str(macenko_path.relative_to(root)),
        }
        detail_rows.append(detail)
        summary_rows.append(
            {
                "category": "preprocessing_robustness",
                "analysis": "no_stain_norm_vs_Macenko",
                "cohort": row.get("cohort", ""),
                "target_set": row.get("target_set", ""),
                "status": status,
                "primary_metric": "macenko_mean_pearson",
                "primary_value": mean_pearson,
                "null_reference": null_p95,
                "empirical_p": empirical_p,
                "direction": "positive" if mean_pearson > 0 else "non_positive",
                "input_path": f"{macenko_path.relative_to(root)}; {baseline_path.relative_to(root)}",
                "note": (
                    "Macenko-normalized representative-slide Prov-GigaPath Stage 5 sensitivity. "
                    f"Delta versus no-stain representative baseline={delta:.4g}; "
                    f"manifest PASS rows={manifest_pass}/{manifest_rows}; Macenko fit counts={macenko_fit_counts or 'not_recorded'}."
                ),
            }
        )
    pd.DataFrame(detail_rows).to_csv(out_path, sep="\t", index=False)
    return summary_rows


def _read_pdc_protein_quant_for_selection(root: Path, study_id: str, selected_columns: list[str]) -> pd.DataFrame:
    path = root / "data/raw/pdc_quant_api" / study_id / f"{study_id}_unshared_log2_ratio.quantDataMatrix.tsv"
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    keep_cols = [header[0], *[c for c in selected_columns if c in header]]
    if len(keep_cols) <= 1:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", usecols=keep_cols)


def build_nat_sanity_check(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Project PDC NAT protein quantitation onto a tumor hidden-aggressive protein-centroid proxy."""
    detail_out = out_dir / "nat_sanity_feature_scores.tsv"
    summary_out = out_dir / "nat_sanity_summary.tsv"
    selection_path = root / "data/processed/stage2/pdc_sample_selection.tsv"
    hidden_path = root / "results/clinical/hidden_aggressive_groups.tsv"
    protein_path = root / "data/processed/stage3/protein_matrix.tsv"
    required = [selection_path, hidden_path, protein_path]
    missing = [str(p.relative_to(root)) for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        summary = pd.DataFrame(
            [
                {
                    "cohort": "CPTAC-COAD+CPTAC-PDAC",
                    "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                    "skip_reason": "missing required inputs: " + ";".join(missing),
                    "nat_n": 0,
                    "nat_hidden_like_n": 0,
                    "nat_hidden_like_fraction": np.nan,
                    "tumor_threshold_quantile": 0.75,
                    "n_features": 0,
                }
            ]
        )
        summary.to_csv(summary_out, sep="\t", index=False)
        pd.DataFrame(columns=["cohort", "patient_id", "sample_type_group", "hidden_aggressive_protein_centroid_score_z"]).to_csv(
            detail_out, sep="\t", index=False
        )
        return [
            {
                "category": "NAT_sanity_check",
                "analysis": "normal_adjacent_tissue_hidden_aggressive_protein_centroid_proxy",
                "cohort": "CPTAC-COAD+CPTAC-PDAC",
                "target_set": "protein_quant_hidden_aggressive_proxy",
                "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                "primary_metric": "nat_scored_rows",
                "primary_value": 0,
                "null_reference": np.nan,
                "empirical_p": np.nan,
                "direction": "",
                "input_path": ";".join(missing),
                "note": "Required inputs for NAT protein-centroid sanity check were not available.",
            }
        ]

    selection = read_tsv(selection_path)
    hidden = read_tsv(hidden_path)
    protein = read_tsv(protein_path)
    hidden = hidden[hidden.get("cutoff", pd.Series(dtype=str)).astype(str) == "top25"].copy()
    hidden = hidden[["cohort", "patient_id", "hidden_aggressive_group", "hidden_aggressive_indicator"]].drop_duplicates()
    protein_feature_set = set(feature_columns(protein))
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for cohort in ["CPTAC-COAD", "CPTAC-PDAC"]:
        sel = selection[
            (selection["cohort"] == cohort)
            & (selection["modality"].astype(str) == "proteome")
            & (selection["sample_type"].astype(str).str.contains("Primary Tumor|Normal", case=False, regex=True, na=False))
        ].copy()
        if sel.empty:
            summary_rows.append(
                {
                    "cohort": cohort,
                    "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                    "skip_reason": "no primary-tumor/NAT proteome rows in pdc_sample_selection",
                    "nat_n": 0,
                    "nat_hidden_like_n": 0,
                    "nat_hidden_like_fraction": np.nan,
                    "tumor_n": 0,
                    "hidden_tumor_n": 0,
                    "tumor_threshold_quantile": 0.75,
                    "tumor_score_threshold": np.nan,
                    "n_features": 0,
                    "mannwhitney_nat_vs_hidden_tumor_p": np.nan,
                    "input_path": str(selection_path.relative_to(root)),
                }
            )
            continue
        study_ids = sorted(sel["pdc_study_id"].dropna().astype(str).unique().tolist())
        if not study_ids:
            continue
        study_id = study_ids[0]
        quant = _read_pdc_protein_quant_for_selection(root, study_id, sel["matrix_column"].dropna().astype(str).tolist())
        if quant.empty:
            summary_rows.append(
                {
                    "cohort": cohort,
                    "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                    "skip_reason": f"no selected columns found in {study_id} quantDataMatrix",
                    "nat_n": 0,
                    "nat_hidden_like_n": 0,
                    "nat_hidden_like_fraction": np.nan,
                    "tumor_n": 0,
                    "hidden_tumor_n": 0,
                    "tumor_threshold_quantile": 0.75,
                    "tumor_score_threshold": np.nan,
                    "n_features": 0,
                    "mannwhitney_nat_vs_hidden_tumor_p": np.nan,
                    "input_path": f"{selection_path.relative_to(root)}; data/raw/pdc_quant_api/{study_id}",
                }
            )
            continue

        gene_col = quant.columns[0]
        quant = quant[quant[gene_col].notna()].copy()
        quant[gene_col] = quant[gene_col].astype(str)
        shared_genes = sorted(set(quant[gene_col]) & protein_feature_set)
        if len(shared_genes) < 100:
            status = "WARN"
        else:
            status = "PASS"
        quant = quant[quant[gene_col].isin(shared_genes)].drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
        sample_meta = sel[sel["matrix_column"].isin(quant.columns)].copy()
        expr = quant[sample_meta["matrix_column"].tolist()].T
        expr.index.name = "matrix_column"
        expr = expr.apply(pd.to_numeric, errors="coerce")
        meta = sample_meta.set_index("matrix_column")[["cohort", "patient_id", "sample_type"]]
        sample_df = meta.join(expr, how="inner").reset_index(drop=False)
        sample_df["sample_type_group"] = np.where(
            sample_df["sample_type"].astype(str).str.contains("Normal", case=False, na=False), "NAT", "Primary Tumor"
        )
        feature_cols = [c for c in shared_genes if c in sample_df.columns]
        sample_df = sample_df.groupby(["cohort", "patient_id", "sample_type_group"], as_index=False)[feature_cols].mean()
        tumor = sample_df[sample_df["sample_type_group"] == "Primary Tumor"].merge(hidden, on=["cohort", "patient_id"], how="inner")
        nat = sample_df[sample_df["sample_type_group"] == "NAT"].copy()
        feature_cols = [c for c in feature_cols if tumor[c].notna().mean() >= 0.70]
        if len(tumor) < 20 or len(nat) == 0 or len(feature_cols) < 100 or tumor["hidden_aggressive_indicator"].nunique() < 2:
            summary_rows.append(
                {
                    "cohort": cohort,
                    "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                    "skip_reason": "insufficient tumor/NAT rows, shared features, or hidden-aggressive label variation",
                    "nat_n": int(len(nat)),
                    "nat_hidden_like_n": 0,
                    "nat_hidden_like_fraction": np.nan,
                    "tumor_n": int(len(tumor)),
                    "hidden_tumor_n": int(pd.to_numeric(tumor.get("hidden_aggressive_indicator", pd.Series(dtype=float)), errors="coerce").sum())
                    if len(tumor)
                    else 0,
                    "tumor_threshold_quantile": 0.75,
                    "tumor_score_threshold": np.nan,
                    "n_features": len(feature_cols),
                    "mannwhitney_nat_vs_hidden_tumor_p": np.nan,
                    "input_path": f"{selection_path.relative_to(root)}; data/raw/pdc_quant_api/{study_id}/{study_id}_unshared_log2_ratio.quantDataMatrix.tsv",
                }
            )
            continue

        tumor_x = tumor[feature_cols].apply(pd.to_numeric, errors="coerce")
        means = tumor_x.mean(axis=0, skipna=True)
        sds = tumor_x.std(axis=0, skipna=True, ddof=1).replace(0, np.nan)
        tumor_z = (tumor_x - means) / sds
        labels = pd.to_numeric(tumor["hidden_aggressive_indicator"], errors="coerce").fillna(0).astype(int)
        hidden_mean = tumor_z.loc[labels == 1].mean(axis=0, skipna=True)
        background_mean = tumor_z.loc[labels == 0].mean(axis=0, skipna=True)
        weights = (hidden_mean - background_mean).replace([np.inf, -np.inf], np.nan).dropna()
        weights = weights.reindex(weights.abs().sort_values(ascending=False).head(min(300, len(weights))).index)
        norm = float(np.sqrt(np.nansum(np.square(weights.to_numpy(dtype=float)))))
        if not np.isfinite(norm) or norm <= 1e-12:
            norm = 1.0
        weights = weights / norm

        def score(frame: pd.DataFrame) -> np.ndarray:
            x = frame[weights.index.tolist()].apply(pd.to_numeric, errors="coerce")
            z = (x - means[weights.index]) / sds[weights.index]
            return np.nansum(z.to_numpy(dtype=float) * weights.to_numpy(dtype=float)[None, :], axis=1)

        tumor_scores = score(tumor)
        nat_scores = score(nat)
        score_mean = float(np.nanmean(tumor_scores))
        score_sd = float(np.nanstd(tumor_scores, ddof=1))
        if not np.isfinite(score_sd) or score_sd <= 1e-12:
            score_sd = 1.0
        tumor_score_z = (tumor_scores - score_mean) / score_sd
        nat_score_z = (nat_scores - score_mean) / score_sd
        threshold = float(np.nanquantile(tumor_score_z, 0.75))
        tumor_detail = tumor[["cohort", "patient_id", "sample_type_group", "hidden_aggressive_group", "hidden_aggressive_indicator"]].copy()
        tumor_detail["hidden_aggressive_protein_centroid_score_z"] = tumor_score_z
        tumor_detail["tumor_top_quartile_threshold"] = threshold
        tumor_detail["hidden_like_by_protein_centroid"] = tumor_detail["hidden_aggressive_protein_centroid_score_z"] >= threshold
        nat_detail = nat[["cohort", "patient_id", "sample_type_group"]].copy()
        nat_detail["hidden_aggressive_group"] = "NAT"
        nat_detail["hidden_aggressive_indicator"] = np.nan
        nat_detail["hidden_aggressive_protein_centroid_score_z"] = nat_score_z
        nat_detail["tumor_top_quartile_threshold"] = threshold
        nat_detail["hidden_like_by_protein_centroid"] = nat_detail["hidden_aggressive_protein_centroid_score_z"] >= threshold
        detail = pd.concat([tumor_detail, nat_detail], ignore_index=True)
        detail["n_features"] = len(weights)
        detail["study_id"] = study_id
        detail_frames.append(detail)

        nat_high = int(nat_detail["hidden_like_by_protein_centroid"].sum())
        nat_fraction = nat_high / len(nat_detail) if len(nat_detail) else np.nan
        hidden_scores = tumor_detail.loc[labels.to_numpy() == 1, "hidden_aggressive_protein_centroid_score_z"].to_numpy(dtype=float)
        mwu_p = (
            stats.mannwhitneyu(nat_score_z, hidden_scores, alternative="less").pvalue
            if len(nat_score_z) > 0 and len(hidden_scores) > 0
            else np.nan
        )
        summary_rows.append(
            {
                "cohort": cohort,
                "status": "PASS" if status == "PASS" and np.isfinite(nat_fraction) and nat_fraction <= 0.25 else "WARN",
                "skip_reason": "",
                "nat_n": int(len(nat_detail)),
                "nat_hidden_like_n": nat_high,
                "nat_hidden_like_fraction": float(nat_fraction),
                "tumor_n": int(len(tumor_detail)),
                "hidden_tumor_n": int(labels.sum()),
                "tumor_threshold_quantile": 0.75,
                "tumor_score_threshold": threshold,
                "n_features": int(len(weights)),
                "mannwhitney_nat_vs_hidden_tumor_p": float(mwu_p) if np.isfinite(mwu_p) else np.nan,
                "input_path": f"{selection_path.relative_to(root)}; data/raw/pdc_quant_api/{study_id}/{study_id}_unshared_log2_ratio.quantDataMatrix.tsv; {hidden_path.relative_to(root)}",
            }
        )

    detail_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not detail_all.empty:
        nat_all = detail_all[detail_all["sample_type_group"] == "NAT"]
        pooled_fraction = float(nat_all["hidden_like_by_protein_centroid"].mean()) if len(nat_all) else np.nan
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "cohort": "pooled",
                            "status": "PASS" if np.isfinite(pooled_fraction) and pooled_fraction <= 0.25 else "WARN",
                            "skip_reason": "",
                            "nat_n": int(len(nat_all)),
                            "nat_hidden_like_n": int(nat_all["hidden_like_by_protein_centroid"].sum()) if len(nat_all) else 0,
                            "nat_hidden_like_fraction": pooled_fraction,
                            "tumor_n": int((detail_all["sample_type_group"] == "Primary Tumor").sum()),
                            "hidden_tumor_n": int(pd.to_numeric(detail_all["hidden_aggressive_indicator"], errors="coerce").fillna(0).sum()),
                            "tumor_threshold_quantile": 0.75,
                            "tumor_score_threshold": np.nan,
                            "n_features": int(detail_all["n_features"].max()),
                            "mannwhitney_nat_vs_hidden_tumor_p": np.nan,
                            "input_path": f"{selection_path.relative_to(root)}; data/raw/pdc_quant_api; {hidden_path.relative_to(root)}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    detail_all.to_csv(detail_out, sep="\t", index=False)
    summary.to_csv(summary_out, sep="\t", index=False)

    if summary.empty:
        return [
            {
                "category": "NAT_sanity_check",
                "analysis": "normal_adjacent_tissue_hidden_aggressive_protein_centroid_proxy",
                "cohort": "CPTAC-COAD+CPTAC-PDAC",
                "target_set": "protein_quant_hidden_aggressive_proxy",
                "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                "primary_metric": "nat_scored_rows",
                "primary_value": 0,
                "null_reference": np.nan,
                "empirical_p": np.nan,
                "direction": "",
                "input_path": f"{selection_path.relative_to(root)}; {hidden_path.relative_to(root)}",
                "note": "No evaluable NAT protein-centroid proxy rows were produced.",
            }
        ]

    out_rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        out_rows.append(
            {
                "category": "NAT_sanity_check",
                "analysis": "normal_adjacent_tissue_hidden_aggressive_protein_centroid_proxy",
                "cohort": row.get("cohort", ""),
                "target_set": "protein_quant_hidden_aggressive_proxy",
                "status": row.get("status", "WARN"),
                "primary_metric": "nat_hidden_like_fraction",
                "primary_value": row.get("nat_hidden_like_fraction", np.nan),
                "null_reference": 0.25,
                "empirical_p": row.get("mannwhitney_nat_vs_hidden_tumor_p", np.nan),
                "direction": "not_excess_hidden_like"
                if row.get("nat_hidden_like_fraction", np.nan) <= 0.25
                else "excess_hidden_like",
                "input_path": row.get("input_path", ""),
                "note": (
                    "PDC NAT protein quantitation projected onto a tumor hidden-aggressive protein-centroid proxy. "
                    "This is a NAT omics sanity check, not a full morphology-proteome residual or Stage 8 quadrant assignment."
                ),
            }
        )
    return out_rows


def build_robustness_summary(root: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction_specs = [
        ("encoder_robustness", "H_Optimus_0_strict_mean", "results/prediction_h_optimus_strict_mean"),
        ("encoder_robustness", "Prov_GigaPath_strict_mean", "results/prediction_prov_gigapath_strict_mean"),
        (
            "encoder_robustness",
            "Prov_GigaPath_exact_multislide_median_final",
            "results/prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity",
        ),
        ("slide_selection_robustness", "Prov_GigaPath_representative_strict_mean_exact", "results/prediction_prov_gigapath_exact_phospho_direct_strict_mean"),
        (
            "slide_selection_robustness",
            "Prov_GigaPath_all_slides_median_exact_final",
            "results/prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity",
        ),
    ]
    for category, analysis, rel_dir in prediction_specs:
        df = summarize_prediction_dir(root, rel_dir, analysis)
        for _, row in df.iterrows():
            rows.append(
                {
                    "category": category,
                    "analysis": analysis,
                    "cohort": row.get("cohort", ""),
                    "target_set": row.get("target_set", ""),
                    "status": row.get("status", "WARN"),
                    "primary_metric": "mean_pearson",
                    "primary_value": row.get("mean_pearson", np.nan),
                    "null_reference": row.get("null_mean_pearson_p95", np.nan),
                    "empirical_p": row.get("empirical_p_mean_pearson", np.nan),
                    "direction": "positive" if row.get("mean_pearson", np.nan) > 0 else "non_positive",
                    "input_path": rel_dir,
                    "note": "Stage 5 prediction-summary reuse; PASS requires mean_pearson above posthoc null p95 and empirical p<=0.05.",
                }
            )

    rows.append(build_pdc000109_label_free_repeat(root, out_dir))
    rows.extend(build_macenko_preprocessing_robustness(root, out_dir))
    rows.extend(build_nat_sanity_check(root, out_dir))
    return pd.DataFrame(rows)


def build_artifact_regression(root: Path) -> pd.DataFrame:
    risk = read_tsv(root / "results/clinical/discordance_risk.tsv")
    qc = read_tsv(root / "data/processed/stage4/slide_qc_table.tsv")
    merged = risk.merge(qc, on=["cohort", "patient_id"], how="inner", suffixes=("", "_slide"))
    numeric_cols = [
        "candidate_tiles",
        "selected_tiles",
        "mean_selected_tissue_occupancy",
        "level_mpp",
        "derived_mpp",
        "downsample_factor",
        "tile_native_px",
        "dicom_files",
        "level_frames",
        "level_total_rows",
        "level_total_cols",
        "overview_rows",
        "overview_cols",
        "candidate_slide_count_stage4",
        "candidate_pass_count_stage4",
    ]
    rows = []
    for col in numeric_cols:
        if col not in merged.columns:
            continue
        x = finite_float(merged[col])
        y = finite_float(merged["discordance_risk_z"])
        keep = x.notna() & y.notna()
        if keep.sum() < 10 or x[keep].nunique() < 2:
            rows.append(
                {
                    "metric": col,
                    "status": "SKIP",
                    "skip_reason": "insufficient_variation_or_n",
                    "n": int(keep.sum()),
                    "spearman_rho": np.nan,
                    "p": np.nan,
                }
            )
            continue
        rho, p = stats.spearmanr(x[keep].to_numpy(dtype=float), y[keep].to_numpy(dtype=float))
        rows.append(
            {
                "metric": col,
                "status": "PASS",
                "skip_reason": "",
                "n": int(keep.sum()),
                "spearman_rho": float(rho),
                "p": float(p),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = bh_adjust(out["p"].tolist())
        out["flag_large_correlation"] = (out["spearman_rho"].abs() >= 0.35) & (out["q"] <= 0.10)
        out.loc[out["flag_large_correlation"].fillna(False), "status"] = "WARN"
    histoqc_status = sorted(qc["histoqc_status"].dropna().astype(str).unique().tolist()) if "histoqc_status" in qc.columns else []
    out["histoqc_status_values"] = ";".join(histoqc_status)
    out["note"] = "Spearman correlation between Stage 8 discordance_risk_z and Stage 4 representative-slide QC metrics."
    return out


def random_pathway_null(root: Path, permutations: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    matrix = read_tsv(root / "results/discordance/discordance_pathway_matrix.tsv")
    clinical = read_tsv(root / "data/processed/stage3/clinical_endpoints.tsv")
    risk = read_tsv(root / "results/clinical/discordance_risk.tsv")
    loadings = read_tsv(root / "results/archetypes/archetype_loadings.tsv")
    merged = risk[["cohort", "patient_id", "discordance_risk_z"]].merge(
        clinical[["cohort", "patient_id", "advanced_at_presentation"]],
        on=["cohort", "patient_id"],
        how="inner",
        validate="one_to_one",
    )
    base = matrix.merge(merged[["cohort", "patient_id"]], on=["cohort", "patient_id"], how="inner", validate="one_to_one")
    feature_cols = feature_columns(base)
    x_all = base[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_all = x_all.apply(lambda s: zscore(s.to_numpy(dtype=float)), axis=0, result_type="broadcast")
    x_all.columns = feature_cols

    top = loadings[loadings["archetype"].isin(["A1", "A2"])].copy()
    top = top[top["loading_rank"] <= 100]
    top["matrix_col"] = top["target_set"].astype(str) + "__" + top["feature"].astype(str)
    top = top[top["matrix_col"].isin(feature_cols)]
    set_counts = top.groupby(["archetype", "target_set", "direction"]).size().reset_index(name="n")
    observed = logistic_coef(merged["discordance_risk_z"].to_numpy(dtype=float), merged["advanced_at_presentation"].to_numpy(dtype=float))
    null_coefs = []
    pools = {
        target: [c for c in feature_cols if c.startswith(f"{target}__")]
        for target in sorted(top["target_set"].dropna().unique().tolist())
    }
    for _ in range(permutations):
        score = np.zeros(len(base), dtype=float)
        for _, row in set_counts.iterrows():
            pool = pools.get(row["target_set"], feature_cols)
            n = int(min(row["n"], len(pool)))
            if n <= 0:
                continue
            cols = rng.choice(pool, size=n, replace=False)
            sign = -1.0 if str(row["direction"]).startswith("negative") else 1.0
            archetype_sign = 1.0 if row["archetype"] == "A2" else -1.0
            score += archetype_sign * sign * x_all.loc[:, cols].mean(axis=1).to_numpy(dtype=float)
        score = zscore(score)
        fit = logistic_coef(score, merged["advanced_at_presentation"].to_numpy(dtype=float))
        if fit.get("status") in {"PASS", "WARN"} and np.isfinite(fit.get("coef", np.nan)):
            null_coefs.append(float(fit["coef"]))
    null = np.asarray(null_coefs, dtype=float)
    coef = float(observed.get("coef", np.nan))
    empirical = (1.0 + float(np.sum(null >= coef))) / (len(null) + 1.0) if len(null) and np.isfinite(coef) else np.nan
    return pd.DataFrame(
        [
            {
                "null_model": "random_pathway_sets_matched_by_archetype_target_direction_size",
                "cohort": "pooled",
                "target_set": "discordance_pathway_matrix",
                "status": "PASS" if np.isfinite(empirical) and empirical <= 0.10 else "WARN",
                "n": observed.get("n", np.nan),
                "events": observed.get("events", np.nan),
                "observed_coef": coef,
                "observed_or": observed.get("or", np.nan),
                "observed_p": observed.get("p", np.nan),
                "null_iterations": int(len(null)),
                "null_coef_mean": float(np.nanmean(null)) if len(null) else np.nan,
                "null_coef_sd": float(np.nanstd(null, ddof=1)) if len(null) > 1 else np.nan,
                "null_coef_p95": float(np.nanpercentile(null, 95)) if len(null) else np.nan,
                "empirical_p": empirical,
                "input_path": "results/discordance/discordance_pathway_matrix.tsv; results/archetypes/archetype_loadings.tsv",
                "note": "Random feature-set null matched to top-100 A1/A2 loading counts by target set and discordance direction; compared with Stage 8 discordance_risk_z advanced association.",
            }
        ]
    )


def build_null_results(root: Path, random_permutations: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final_summary = read_tsv(root / "results/prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity/prediction_summary_with_null.tsv")
    for _, row in final_summary.iterrows():
        passed = bool((row["mean_pearson"] > row["null_mean_pearson_p95"]) and (row["empirical_p_mean_pearson"] <= 0.05))
        rows.append(
            {
                "null_model": "posthoc_label_shuffle_on_oof_predictions",
                "cohort": row["cohort"],
                "target_set": row["target_set"],
                "status": "PASS" if passed else "WARN",
                "n": row["n_samples"],
                "events": np.nan,
                "observed_coef": row["mean_pearson"],
                "observed_or": np.nan,
                "observed_p": np.nan,
                "null_iterations": 100,
                "null_coef_mean": row["null_mean_pearson_mean"],
                "null_coef_sd": row["null_mean_pearson_sd"],
                "null_coef_p95": row["null_mean_pearson_p95"],
                "empirical_p": row["empirical_p_mean_pearson"],
                "input_path": "results/prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity/prediction_summary_with_null.tsv",
                "note": "Final Stage 5 candidate posthoc label-shuffle null.",
            }
        )
    final_full_candidates = sorted(
        (root / "logs/qc").glob("stage5_final_candidate_full_retrain_null_100_omp8_*_summary_with_null.tsv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    full_path = final_full_candidates[0] if final_full_candidates else root / "logs/qc/stage5_prov_gigapath_full_retrain_null_100_omp8_summary_with_null.tsv"
    if full_path.exists():
        full = read_tsv(full_path)
        is_final_candidate = full_path.name.startswith("stage5_final_candidate_full_retrain_null_100_omp8_")
        for _, row in full.iterrows():
            passed = bool((row["mean_pearson"] > row["full_null_mean_pearson_p95"]) and (row["empirical_p_full_retrain_mean_pearson"] <= 0.05))
            rows.append(
                {
                    "null_model": "full_retrain_patient_label_permutation_final_exact_multislide_median_candidate"
                    if is_final_candidate
                    else "full_retrain_patient_label_permutation_existing_strict_mean_candidate",
                    "cohort": row["cohort"],
                    "target_set": row["target_set"],
                    "status": "PASS" if passed else "WARN",
                    "n": row["n_samples"],
                    "events": np.nan,
                    "observed_coef": row["mean_pearson"],
                    "observed_or": np.nan,
                    "observed_p": np.nan,
                    "null_iterations": 100,
                    "null_coef_mean": row["full_null_mean_pearson_mean"],
                    "null_coef_sd": row["full_null_mean_pearson_sd"],
                    "null_coef_p95": row["full_null_mean_pearson_p95"],
                    "empirical_p": row["empirical_p_full_retrain_mean_pearson"],
                    "input_path": str(full_path.relative_to(root)),
                    "note": "Final exact multi-slide median Prov-GigaPath candidate full-retrain label-permutation null."
                    if is_final_candidate
                    else "Existing full-retrain null was run on older strict-mean Prov-GigaPath candidate, not the final exact multislide median candidate.",
                }
            )
    else:
        rows.append(
            {
                "null_model": "full_retrain_patient_label_permutation",
                "cohort": "CPTAC-COAD+CPTAC-PDAC",
                "target_set": "protein_pathway+phospho",
                "status": "SKIPPED_INPUT_NOT_AVAILABLE",
                "note": "No full-retrain null output present.",
            }
        )
    rows_df = pd.DataFrame(rows)
    return pd.concat([rows_df, random_pathway_null(root, random_permutations)], ignore_index=True, sort=False)


def build_fusion_benchmark(root: Path, max_pcs: int, folds: int, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = read_tsv(root / "data/processed/stage3/clinical_endpoints.tsv")
    morph = read_tsv(root / "results/clinical/morphology_risk_oof.tsv")[["cohort", "patient_id", "morph_risk_z"]]
    discordance = read_tsv(root / "results/clinical/discordance_risk.tsv")[["cohort", "patient_id", "discordance_risk_z"]]
    pred_dir = root / "results/prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity"
    protein = read_tsv(pred_dir / "observed_protein_pathway_aligned.tsv")
    phospho = read_tsv(pred_dir / "observed_phospho_kinase_ptm_aligned.tsv")
    early = morph.merge(protein, on=["cohort", "patient_id"], how="inner", validate="one_to_one").merge(
        phospho, on=["cohort", "patient_id"], how="inner", validate="one_to_one", suffixes=("", "_phospho")
    )
    morph_disc = morph.merge(discordance, on=["cohort", "patient_id"], how="inner", validate="one_to_one")
    specs = [
        ("morphology_only", morph, "low_dim_logit", "results/clinical/morphology_risk_oof.tsv"),
        (
            "proteome_only",
            protein,
            "pca_logit",
            "results/prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity/observed_protein_pathway_aligned.tsv",
        ),
        (
            "phospho_only",
            phospho,
            "pca_logit",
            "results/prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity/observed_phospho_kinase_ptm_aligned.tsv",
        ),
        ("simple_early_fusion", early, "pca_logit", "morph_risk_z + observed protein pathway + observed exact phospho activity"),
        (
            "morphology_plus_discordance",
            morph_disc,
            "low_dim_logit",
            "results/clinical/morphology_risk_oof.tsv; results/clinical/discordance_risk.tsv",
        ),
    ]
    metrics = []
    predictions = []
    for name, df, mode, input_path in specs:
        m, p = fit_cv_benchmark(name, df, clinical, mode, input_path, max_pcs, folds, repeats)
        metrics.append(m)
        predictions.append(p)
    out = pd.concat(metrics, ignore_index=True)
    pred = pd.concat([p for p in predictions if not p.empty], ignore_index=True)
    baseline = out[out["benchmark_model"] == "morphology_only"][["cohort", "auc"]].rename(columns={"auc": "morphology_only_auc"})
    out = out.merge(baseline, on="cohort", how="left")
    out["delta_auc_vs_morphology_only"] = out["auc"] - out["morphology_only_auc"]
    out["biological_interpretability_note"] = np.where(
        out["benchmark_model"] == "morphology_plus_discordance",
        "Uses a single Stage 8 discordance-risk axis tied to Stage 7 archetypes; interpretable hidden-risk proxy.",
        np.where(
            out["benchmark_model"].isin(["proteome_only", "phospho_only", "simple_early_fusion"]),
            "Predictive comparator; high-dimensional omics/fusion features are less directly interpretable than the discordance-risk axis.",
            "Morphology-only OOF risk baseline from Stage 8.",
        ),
    )
    return out, pred


def write_qc(root: Path, out_dir: Path, qc_dir: Path, outputs: dict[str, Path]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, details: dict[str, Any] | None = None, severity: str = "FAIL") -> None:
        checks.append({"check": name, "status": "PASS" if ok else severity, "details": details or {}})

    for name, path in outputs.items():
        check(f"output_exists::{name}", path.exists() and path.stat().st_size > 0, {"path": str(path.relative_to(root)), "size": path.stat().st_size if path.exists() else 0})

    robustness = read_tsv(outputs["robustness_summary.tsv"])
    nulls = read_tsv(outputs["null_model_results.tsv"])
    fusion = read_tsv(outputs["fusion_benchmark.tsv"])
    artifact = read_tsv(outputs["artifact_regression.tsv"])
    required_models = {"morphology_only", "proteome_only", "phospho_only", "simple_early_fusion", "morphology_plus_discordance"}
    check("fusion_required_models_present", required_models <= set(fusion["benchmark_model"]), {"models": sorted(fusion["benchmark_model"].unique().tolist())})
    check("robustness_key_categories_present", {"encoder_robustness", "platform_robustness", "slide_selection_robustness", "preprocessing_robustness", "NAT_sanity_check"} <= set(robustness["category"]), {"categories": sorted(robustness["category"].unique().tolist())})
    check("null_models_present", {"posthoc_label_shuffle_on_oof_predictions", "random_pathway_sets_matched_by_archetype_target_direction_size"} <= set(nulls["null_model"]), {"null_models": sorted(nulls["null_model"].unique().tolist())})
    check("artifact_regression_ran", len(artifact) > 0, {"rows": len(artifact)})
    check(
        "fusion_morphology_plus_discordance_interpretability_recorded",
        bool((fusion["benchmark_model"] == "morphology_plus_discordance").any()),
        {},
    )
    warn_sources = []
    for label, df in [("robustness", robustness), ("null", nulls), ("fusion", fusion), ("artifact", artifact)]:
        status_col = "status"
        if status_col in df.columns:
            counts = df[status_col].value_counts(dropna=False).to_dict()
            warn_sources.append({"table": label, "status_counts": counts})
    for item in warn_sources:
        if any(k in item["status_counts"] for k in ["WARN", "SKIP", "SKIPPED_INPUT_NOT_AVAILABLE"]):
            check(f"non_blocking_warnings::{item['table']}", False, item, severity="WARN")

    fail_count = sum(1 for row in checks if row["status"] == "FAIL")
    warn_count = sum(1 for row in checks if row["status"] == "WARN")
    qc = {
        "stage": "Stage 11",
        "built_at": now_iso(),
        "status": "PASS" if fail_count == 0 else "FAIL",
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checks": checks,
    }
    (qc_dir / "stage11_robustness_qc.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(checks).to_csv(qc_dir / "stage11_robustness_qc_summary.tsv", sep="\t", index=False)
    return qc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/robustness")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-pcs", type=int, default=10)
    parser.add_argument("--random-pathway-permutations", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / args.output_dir
    qc_dir = root / "logs/qc"
    ensure_dir(out_dir)
    ensure_dir(qc_dir)

    robustness = build_robustness_summary(root, out_dir)
    artifact = build_artifact_regression(root)
    artifact_status = "WARN" if bool(artifact.get("flag_large_correlation", pd.Series(dtype=bool)).fillna(False).any()) else "PASS"
    robustness = pd.concat(
        [
            robustness,
            pd.DataFrame(
                [
                    {
                        "category": "artifact_regression",
                        "analysis": "discordance_risk_vs_slide_qc_metrics",
                        "cohort": "CPTAC-COAD+CPTAC-PDAC",
                        "target_set": "discordance_risk_z",
                        "status": artifact_status,
                        "primary_metric": "max_abs_spearman_rho",
                        "primary_value": float(artifact["spearman_rho"].abs().max(skipna=True)),
                        "null_reference": 0.35,
                        "empirical_p": np.nan,
                        "direction": "",
                        "input_path": "results/clinical/discordance_risk.tsv; data/processed/stage4/slide_qc_table.tsv",
                        "note": "WARN if any slide-QC metric has |rho|>=0.35 and BH q<=0.10.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    nulls = build_null_results(root, args.random_pathway_permutations)
    fusion, predictions = build_fusion_benchmark(root, args.max_pcs, args.folds, args.repeats)

    outputs = {
        "robustness_summary.tsv": out_dir / "robustness_summary.tsv",
        "pdc000109_label_free_spectral_count_matrix.tsv": out_dir / "pdc000109_label_free_spectral_count_matrix.tsv",
        "pdc000109_label_free_technical_repeat.tsv": out_dir / "pdc000109_label_free_technical_repeat.tsv",
        "pdc000109_label_free_technical_repeat_summary.tsv": out_dir / "pdc000109_label_free_technical_repeat_summary.tsv",
        "macenko_preprocessing_sensitivity.tsv": out_dir / "macenko_preprocessing_sensitivity.tsv",
        "nat_sanity_summary.tsv": out_dir / "nat_sanity_summary.tsv",
        "nat_sanity_feature_scores.tsv": out_dir / "nat_sanity_feature_scores.tsv",
        "null_model_results.tsv": out_dir / "null_model_results.tsv",
        "fusion_benchmark.tsv": out_dir / "fusion_benchmark.tsv",
        "fusion_benchmark_oof_predictions.tsv": out_dir / "fusion_benchmark_oof_predictions.tsv",
        "artifact_regression.tsv": out_dir / "artifact_regression.tsv",
        "stage11_build_summary.json": out_dir / "stage11_build_summary.json",
    }
    robustness.to_csv(outputs["robustness_summary.tsv"], sep="\t", index=False)
    nulls.to_csv(outputs["null_model_results.tsv"], sep="\t", index=False)
    fusion.to_csv(outputs["fusion_benchmark.tsv"], sep="\t", index=False)
    predictions.to_csv(outputs["fusion_benchmark_oof_predictions.tsv"], sep="\t", index=False)
    artifact.to_csv(outputs["artifact_regression.tsv"], sep="\t", index=False)

    non_executed_branches = {
        "TCGA_WSI_transfer": "not recomputed in Stage 11; Stage 10 final audit and production outputs are carried forward",
    }
    executed_branches = {
        "PDC000109_label_free": "executed from downloaded Protein_Assembly summary spectral counts because GraphQL quantDataMatrix was unavailable",
        "Macenko_preprocessing": "executed when results/prediction_prov_gigapath_macenko_repr_exact_phospho_direct_sensitivity/prediction_summary_with_null.tsv is present; see macenko_preprocessing_sensitivity.tsv",
        "NAT_sanity_check": "executed as a PDC NAT protein-quant hidden-aggressive centroid proxy; see nat_sanity_summary.tsv and nat_sanity_feature_scores.tsv",
    }
    if bool(
        (
            (robustness["category"] == "preprocessing_robustness")
            & (robustness["status"] == "SKIPPED_INPUT_NOT_AVAILABLE")
        ).any()
    ):
        non_executed_branches["Macenko_preprocessing"] = "Macenko-normalized Stage 5 prediction summary was unavailable at run time"
        executed_branches.pop("Macenko_preprocessing", None)
    if bool(
        (
            (robustness["category"] == "NAT_sanity_check")
            & (robustness["status"] == "SKIPPED_INPUT_NOT_AVAILABLE")
        ).all()
    ):
        non_executed_branches["NAT_sanity_check"] = "PDC NAT protein-centroid proxy inputs were unavailable or not evaluable at run time"
        executed_branches.pop("NAT_sanity_check", None)

    build_summary = {
        "built_at": now_iso(),
        "method": "reuse_validated_stage5_stage8_stage10_outputs_plus_stage11_cpu_benchmarks",
        "folds": args.folds,
        "repeats": args.repeats,
        "max_pcs": args.max_pcs,
        "random_pathway_permutations": args.random_pathway_permutations,
        "outputs": {k: str(v.relative_to(root)) for k, v in outputs.items()},
        "executed_branches": executed_branches,
        "non_executed_branches": non_executed_branches,
    }
    outputs["stage11_build_summary.json"].write_text(json.dumps(build_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    qc = write_qc(root, out_dir, qc_dir, outputs)
    build_summary["qc"] = {"status": qc["status"], "fail_count": qc["fail_count"], "warn_count": qc["warn_count"]}
    outputs["stage11_build_summary.json"].write_text(json.dumps(build_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": qc["status"], "fail_count": qc["fail_count"], "warn_count": qc["warn_count"]}, ensure_ascii=False))
    return 0 if qc["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
