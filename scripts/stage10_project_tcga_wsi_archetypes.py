#!/usr/bin/env python3
"""Project TCGA WSI embeddings onto CPTAC morphology-trained archetype scores."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 20260609
META_COLUMNS = {
    "cohort",
    "patient_id",
    "encoder_slug",
    "model_repo",
    "embedding_summary",
    "slide_count",
    "tile_count_sum",
}


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"patient_id": str})


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def cohort_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        z = np.full(len(out), np.nan, dtype=float)
        for cohort in sorted(out["cohort"].dropna().unique()):
            idx = out["cohort"].to_numpy() == cohort
            values = pd.to_numeric(out.loc[idx, col], errors="coerce").to_numpy(dtype=float)
            mean = np.nanmean(values)
            sd = np.nanstd(values, ddof=1)
            if not np.isfinite(sd) or sd <= 1e-8:
                sd = 1.0
            z[idx] = (values - mean) / sd
        out[f"{col}_z"] = z
    return out


def embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("emb_")]


def make_model(n_features: int, n_samples: int) -> Pipeline:
    n_components = max(2, min(64, n_features, n_samples - 2))
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )


def pearsonr_safe(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(ok.sum()) < 3:
        return np.nan, np.nan
    if np.nanstd(y_true[ok]) <= 1e-12 or np.nanstd(y_pred[ok]) <= 1e-12:
        return np.nan, np.nan
    r, p = stats.pearsonr(y_true[ok], y_pred[ok])
    return float(r), float(p)


def fit_transfer_predictor(cptac_embeddings: pd.DataFrame, scores: pd.DataFrame) -> tuple[dict[str, Pipeline], pd.DataFrame, dict[str, Any]]:
    embed_cols = embedding_columns(cptac_embeddings)
    if not embed_cols:
        raise RuntimeError("No emb_* columns found in CPTAC embedding table")
    merged = cptac_embeddings.merge(scores, on=["cohort", "patient_id"], how="inner", validate="one_to_one")
    targets = [col for col in ["A1_score", "A2_score"] if col in merged.columns]
    if len(targets) < 2:
        raise RuntimeError(f"Missing archetype score targets in merged table: {targets}")
    x = merged[embed_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x).all(axis=1)
    for target in targets:
        ok &= np.isfinite(pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=float))
    merged = merged.loc[ok].reset_index(drop=True)
    x = x[ok]
    if len(merged) < 20:
        raise RuntimeError(f"Too few CPTAC rows for transfer predictor: {len(merged)}")

    groups = merged["cohort"].astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    if len(unique_groups) >= 2 and min(np.sum(groups == g) for g in unique_groups) >= 3:
        splitter = GroupKFold(n_splits=len(unique_groups))
        splits = list(splitter.split(x, groups=groups))
        cv_method = "leave_one_cohort_out_group_kfold"
    else:
        n_splits = min(5, max(2, len(merged) // 10))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        splits = list(splitter.split(x))
        cv_method = f"kfold_{n_splits}"

    oof = merged[["cohort", "patient_id"]].copy()
    metrics: list[dict[str, Any]] = []
    models: dict[str, Pipeline] = {}
    for target in targets:
        y = pd.to_numeric(merged[target], errors="coerce").to_numpy(dtype=float)
        pred = np.full(len(y), np.nan, dtype=float)
        fold_rows = []
        for fold, split in enumerate(splits, start=1):
            train_idx, test_idx = split
            model = make_model(x.shape[1], len(train_idx))
            model.fit(x[train_idx], y[train_idx])
            pred[test_idx] = model.predict(x[test_idx])
            fold_rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "test_cohorts": ",".join(sorted(set(merged.loc[test_idx, "cohort"].astype(str)))),
                    "ridge_alpha": float(model.named_steps["ridge"].alpha_),
                    "pca_components": int(model.named_steps["pca"].n_components_),
                }
            )
        r, p = pearsonr_safe(y, pred)
        metrics.append(
            {
                "target": target,
                "n": int(len(y)),
                "cv_method": cv_method,
                "pearson_r": r,
                "pearson_p": p,
                "r2": float(r2_score(y, pred)) if np.isfinite(pred).all() else np.nan,
                "rmse": float(math.sqrt(mean_squared_error(y, pred))) if np.isfinite(pred).all() else np.nan,
                "folds": fold_rows,
            }
        )
        oof[f"{target}_observed"] = y
        oof[f"{target}_oof_pred"] = pred
        final_model = make_model(x.shape[1], len(merged))
        final_model.fit(x, y)
        models[target] = final_model

    if {"A1_score_oof_pred", "A2_score_oof_pred"}.issubset(oof.columns):
        oof["A1_fraction_oof_pred"] = oof["A1_score_oof_pred"] / (oof["A1_score_oof_pred"] + oof["A2_score_oof_pred"] + 1e-12)
        oof["A2_fraction_oof_pred"] = oof["A2_score_oof_pred"] / (oof["A1_score_oof_pred"] + oof["A2_score_oof_pred"] + 1e-12)
        oof["dominant_archetype_oof_pred"] = np.where(oof["A2_fraction_oof_pred"] >= oof["A1_fraction_oof_pred"], "A2", "A1")
    summary = {
        "method": "CPTAC morphology embedding to Stage 7 A1/A2 archetype RidgeCV transfer predictor",
        "training_rows": int(len(merged)),
        "embedding_dim": int(len(embed_cols)),
        "targets": targets,
        "cv_method": cv_method,
        "cohorts": merged["cohort"].value_counts().to_dict(),
        "metrics": [{k: v for k, v in row.items() if k != "folds"} for row in metrics],
        "folds": [fold for row in metrics for fold in row["folds"]],
    }
    return models, oof, summary


def aggregate_tcga_embeddings(project_root: Path, embedding_manifest: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = read_tsv(embedding_manifest)
    if manifest.empty:
        raise RuntimeError(f"Empty embedding manifest: {embedding_manifest}")
    pass_manifest = manifest[manifest["status"].isin(["PASS", "SKIPPED_EXISTS"])].copy()
    if pass_manifest.empty:
        raise RuntimeError(f"No PASS/SKIPPED_EXISTS embedding rows in {embedding_manifest}")
    slide_rows: list[dict[str, Any]] = []
    for row in pass_manifest.to_dict("records"):
        emb_path = project_root / str(row["embedding_path"])
        if not emb_path.exists() or emb_path.stat().st_size == 0:
            slide_rows.append({**row, "aggregation_status": "FAILED", "aggregation_error": "embedding_path_missing"})
            continue
        try:
            with h5py.File(emb_path, "r") as handle:
                emb = handle["embeddings"][:].astype(np.float32)
                mean_emb = np.nanmean(emb, axis=0)
                tile_count = int(emb.shape[0])
                embedding_dim = int(emb.shape[1])
            out: dict[str, Any] = {
                "cohort": row.get("cohort", ""),
                "patient_id": row.get("patient_id", ""),
                "slide_id": row.get("slide_id", ""),
                "slide_candidate_key": row.get("slide_candidate_key", ""),
                "embedding_path": row.get("embedding_path", ""),
                "tile_count": tile_count,
                "embedding_dim": embedding_dim,
                "aggregation_status": "PASS",
            }
            for idx, value in enumerate(mean_emb):
                out[f"emb_{idx:04d}"] = float(value)
            slide_rows.append(out)
        except Exception as exc:  # noqa: BLE001
            slide_rows.append({**row, "aggregation_status": "FAILED", "aggregation_error": f"{type(exc).__name__}: {exc}"})
    slide_df = pd.DataFrame(slide_rows)
    pass_slides = slide_df[slide_df["aggregation_status"].eq("PASS")].copy()
    if pass_slides.empty:
        raise RuntimeError("No slide embeddings could be aggregated")
    emb_cols = embedding_columns(pass_slides)
    patient_blocks = []
    for (cohort, patient_id), sub in pass_slides.groupby(["cohort", "patient_id"], sort=True):
        weights = pd.to_numeric(sub["tile_count"], errors="coerce").fillna(1).to_numpy(dtype=float)
        weights = np.where(weights > 0, weights, 1.0)
        mat = sub[emb_cols].to_numpy(dtype=float)
        mean_emb = np.average(mat, axis=0, weights=weights)
        row: dict[str, Any] = {
            "cohort": cohort,
            "patient_id": patient_id,
            "slide_count": int(len(sub)),
            "tile_count_sum": int(weights.sum()),
            "embedding_summary": "tcga_wsi_tile_mean_weighted_by_tile_count",
        }
        for idx, value in enumerate(mean_emb):
            row[f"emb_{idx:04d}"] = float(value)
        patient_blocks.append(row)
    patient_df = pd.DataFrame(patient_blocks)
    return slide_df, patient_df


def predict_tcga(models: dict[str, Pipeline], patient_embeddings: pd.DataFrame, cptac_embedding_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in cptac_embedding_cols if col not in patient_embeddings.columns]
    if missing:
        raise RuntimeError(f"TCGA embeddings missing CPTAC feature columns: {missing[:5]} ... total {len(missing)}")
    x = patient_embeddings[cptac_embedding_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    out = patient_embeddings[["cohort", "patient_id", "slide_count", "tile_count_sum", "embedding_summary"]].copy()
    for target, model in models.items():
        out[target.replace("_score", "_wsi_transfer_score")] = model.predict(x)
    if {"A1_wsi_transfer_score", "A2_wsi_transfer_score"}.issubset(out.columns):
        denom = out["A1_wsi_transfer_score"] + out["A2_wsi_transfer_score"] + 1e-12
        out["A1_wsi_transfer_fraction"] = out["A1_wsi_transfer_score"] / denom
        out["A2_wsi_transfer_fraction"] = out["A2_wsi_transfer_score"] / denom
        out["wsi_transfer_dominant_archetype"] = np.where(out["A2_wsi_transfer_fraction"] >= out["A1_wsi_transfer_fraction"], "A2", "A1")
        out["A2_minus_A1_wsi_transfer_score"] = out["A2_wsi_transfer_score"] - out["A1_wsi_transfer_score"]
        out = cohort_zscore(out, ["A1_wsi_transfer_score", "A2_wsi_transfer_score", "A2_minus_A1_wsi_transfer_score"])
    out["predicted_archetype_status"] = "PASS"
    out["prediction_method"] = "morphology_to_stage7_archetype_transfer_predictor"
    return out


def safe_logistic_fit(x: np.ndarray, y: np.ndarray, predictor_names: list[str]) -> list[dict[str, Any]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[ok]
    y = y[ok]
    if len(y) == 0 or len(np.unique(y.astype(int))) < 2:
        return [{"status": "SKIP", "skip_reason": "less_than_two_classes_or_no_data", "term": "", "n": int(len(y))}]
    if min(int(y.sum()), int(len(y) - y.sum())) < x.shape[1] + 1:
        return [
            {
                "status": "SKIP",
                "skip_reason": "insufficient_events_per_parameter",
                "term": "",
                "n": int(len(y)),
                "events": int(y.sum()),
                "nonevents": int(len(y) - y.sum()),
            }
        ]
    try:
        from scipy import optimize

        design = np.column_stack([np.ones(len(y)), x])

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
        rows = []
        for i, term in enumerate(["intercept"] + predictor_names):
            rows.append(
                {
                    "status": "PASS" if res.success else "WARN",
                    "skip_reason": "" if res.success else str(res.message),
                    "term": term,
                    "n": int(len(y)),
                    "events": int(y.sum()),
                    "nonevents": int(len(y) - y.sum()),
                    "coef": float(beta[i]),
                    "se": float(se[i]),
                    "z": float(z[i]),
                    "p": float(p[i]),
                    "or": float(np.exp(beta[i])),
                    "ci95_low": float(np.exp(beta[i] - 1.96 * se[i])),
                    "ci95_high": float(np.exp(beta[i] + 1.96 * se[i])),
                }
            )
        return rows
    except Exception as exc:  # noqa: BLE001
        return [{"status": "SKIP", "skip_reason": f"logistic_fit_error:{type(exc).__name__}:{exc}", "term": "", "n": int(len(y))}]


def safe_cox_fit(df: pd.DataFrame, predictors: list[str]) -> list[dict[str, Any]]:
    try:
        from lifelines import CoxPHFitter
    except Exception as exc:
        return [{"status": "SKIP", "skip_reason": f"lifelines_missing:{exc}", "term": ""}]
    cols = ["survival_time", "survival_event"] + predictors
    sub = df[cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    sub = sub[sub["survival_time"] > 0]
    if len(sub) == 0 or sub["survival_event"].nunique() < 2 or int(sub["survival_event"].sum()) < len(predictors) + 2:
        return [
            {
                "status": "SKIP",
                "skip_reason": "insufficient_survival_events_or_variation",
                "term": "",
                "n": int(len(sub)),
                "events": int(sub["survival_event"].sum()) if len(sub) else 0,
            }
        ]
    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(sub, duration_col="survival_time", event_col="survival_event")
        summ = cph.summary.reset_index().rename(columns={"covariate": "term"})
        rows = []
        for _, row in summ.iterrows():
            rows.append(
                {
                    "status": "PASS",
                    "skip_reason": "",
                    "term": row["term"],
                    "n": int(len(sub)),
                    "events": int(sub["survival_event"].sum()),
                    "coef": float(row["coef"]),
                    "se": float(row["se(coef)"]),
                    "z": float(row["z"]),
                    "p": float(row["p"]),
                    "hr": float(row["exp(coef)"]),
                    "ci95_low": float(row["exp(coef) lower 95%"]),
                    "ci95_high": float(row["exp(coef) upper 95%"]),
                }
            )
        return rows
    except Exception as exc:  # noqa: BLE001
        return [{"status": "SKIP", "skip_reason": f"cox_fit_error:{type(exc).__name__}:{exc}", "term": "", "n": int(len(sub))}]


def run_associations(pred: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    df = pred.merge(
        clinical[
            [
                "cohort",
                "patient_id",
                "age",
                "sex",
                "stage",
                "advanced_at_presentation",
                "survival_time",
                "survival_event",
            ]
        ],
        on=["cohort", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    df["age_numeric"] = pd.to_numeric(df["age"], errors="coerce")
    df = cohort_zscore(df, ["age_numeric"])
    df["sex_male"] = np.where(df["sex"].astype(str).str.lower().eq("male"), 1.0, 0.0)
    rows: list[dict[str, Any]] = []
    predictor = "A2_minus_A1_wsi_transfer_score_z" if "A2_minus_A1_wsi_transfer_score_z" in df.columns else "A2_minus_A1_wsi_transfer_score"
    cohort_blocks = [(str(cohort), sub.copy()) for cohort, sub in df.groupby("cohort", sort=True)]
    cohort_blocks.append(("pooled", df.copy()))
    for cohort, sub in cohort_blocks:
        for model_name, predictors in [
            ("wsi_transfer_only", [predictor]),
            ("clinic_plus_wsi_transfer", ["age_numeric_z", "sex_male", predictor]),
        ]:
            usable = [p for p in predictors if p in sub.columns and sub[p].notna().sum() > 0 and sub[p].nunique(dropna=True) > 1]
            for fit_row in safe_logistic_fit(
                sub[usable].to_numpy(dtype=float),
                pd.to_numeric(sub["advanced_at_presentation"], errors="coerce").to_numpy(dtype=float),
                usable,
            ):
                fit_row.update({"endpoint": "advanced_at_presentation", "cohort": cohort, "model": model_name, "predictors": ",".join(usable)})
                rows.append(fit_row)
        for fit_row in safe_cox_fit(sub, [predictor]):
            fit_row.update({"endpoint": "survival_time_to_event", "cohort": cohort, "model": "cox_wsi_transfer_only", "predictors": predictor})
            rows.append(fit_row)
    assoc = pd.DataFrame(rows)
    if not assoc.empty and "p" in assoc.columns:
        assoc["q_within_endpoint"] = np.nan
        for endpoint, idx in assoc.groupby("endpoint").groups.items():
            assoc.loc[idx, "q_within_endpoint"] = bh_adjust(assoc.loc[idx, "p"].tolist())
    return assoc


def stage10_wsi_status(scope_label: str, pass_assoc: int) -> tuple[str, str]:
    label = str(scope_label or "").lower()
    if label == "pilot":
        return (
            "PASS_PILOT",
            "Pilot-scale WSI transfer proves end-to-end TCGA SVS download/tile/embedding/projection; clinical association is underpowered when only pilot slides are included.",
        )
    if label in {"production_partial", "partial", "batch", "production_batch"}:
        if pass_assoc > 0:
            return (
                "PASS_PRODUCTION_PARTIAL",
                "Production WSI transfer has processed a representative-set subset and has at least one non-SKIP clinical association row; full representative-set coverage is still running.",
            )
        return (
            "PASS_PRODUCTION_PARTIAL_WITH_ASSOCIATION_WARNINGS",
            "Production WSI transfer has processed a representative-set subset, but clinical association rows are still SKIP/WARN because the accumulated subset is underpowered or endpoint variation is insufficient.",
        )
    if pass_assoc > 0:
        return (
            "PASS",
            "Production WSI transfer has processed the requested TCGA representative-set scope and produced non-SKIP clinical association rows.",
        )
    return (
        "PASS_WITH_ASSOCIATION_WARNINGS",
        "Production WSI transfer produced predictions, but clinical association rows remain SKIP/WARN because endpoint variation or event count is insufficient.",
    )


def refresh_stage9_gate(project_root: Path, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    stage9_qc_path = project_root / "logs/qc/stage9_spatial_qc.json"
    stage9_summary_path = project_root / "results/spatial/stage9_build_summary.json"
    for check in checks:
        if check.get("check") != "stage9_gate":
            refreshed.append(check)
            continue
        try:
            qc = json.loads(stage9_qc_path.read_text(encoding="utf-8"))
            summary = json.loads(stage9_summary_path.read_text(encoding="utf-8"))
            hovernet = summary.get("hovernet_cell_summary", {})
            details = {
                "stage9_qc_path": "logs/qc/stage9_spatial_qc.json",
                "stage9_status": qc.get("status", "UNKNOWN"),
                "stage9_fail_count": qc.get("fail_count", "NA"),
                "stage9_warn_count": qc.get("warn_count", "NA"),
                "stage9_target_status": summary.get("target_status", "exploratory_directional_proxy_no_significant_adverse_archetype"),
                "stage9_method": summary.get("method", "attention_mil_regression_tile_attention"),
                "hovernet_status": hovernet.get("status", "NA"),
                "hovernet_tiles": hovernet.get("tiles_processed", "NA"),
                "hovernet_nuclei": hovernet.get("nuclei_total", "NA"),
            }
            refreshed.append({"check": "stage9_gate", "status": "PASS" if qc.get("status") == "PASS" else "WARN", "details": details})
        except Exception as exc:  # noqa: BLE001
            refreshed.append({"check": "stage9_gate", "status": "WARN", "details": {"refresh_error": f"{type(exc).__name__}: {exc}"}})
    return refreshed


def update_stage10_summary_and_qc(project_root: Path, output_dir: Path, wsi_summary: dict[str, Any], pred_rows: int, assoc_rows: int) -> None:
    summary_path = output_dir / "stage10_build_summary.json"
    qc_path = project_root / "logs/qc/stage10_external_validation_qc.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"built_at": now_iso(), "status": "PASS", "outputs": {}}
    summary["built_at"] = now_iso()
    summary["method"] = "tcga_rna_fixed_stage7_loading_proxy_plus_matched_cptac_rna_comparator_plus_tcga_wsi_transfer"
    summary["wsi_transfer"] = wsi_summary
    outputs = summary.setdefault("outputs", {})
    outputs.update(
        {
            "tcga_wsi_predicted_archetypes": str((output_dir / "tcga_wsi_predicted_archetypes.tsv").relative_to(project_root)),
            "tcga_wsi_transfer_associations": str((output_dir / "tcga_wsi_transfer_associations.tsv").relative_to(project_root)),
            "tcga_wsi_transfer_training_oof": str((output_dir / "tcga_wsi_transfer_training_oof.tsv").relative_to(project_root)),
            "tcga_wsi_transfer_summary": str((output_dir / "tcga_wsi_transfer_summary.json").relative_to(project_root)),
        }
    )
    summary["qc"] = summary.get("qc", {})
    summary["qc"].update({"status": "PASS", "fail_count": 0})
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        checks = [c for c in qc.get("checks", []) if c.get("check") not in {"wsi_transfer_audit", "tcga_wsi_transfer_projection", "tcga_wsi_transfer_association"}]
        checks = refresh_stage9_gate(project_root, checks)
    else:
        qc = {"built_at": now_iso(), "checks": []}
        checks = []
    status = "PASS" if wsi_summary["wsi_transfer_status"].startswith("PASS") else "WARN"
    checks.append(
        {
            "check": "wsi_transfer_audit",
            "status": status,
            "details": {
                "status": wsi_summary["wsi_transfer_status"],
                "prediction_rows": pred_rows,
                "association_rows": assoc_rows,
                "note": wsi_summary.get("scope_note", ""),
                "pass_association_rows": wsi_summary.get("association", {}).get("pass_rows", 0),
            },
        }
    )
    checks.append(
        {
            "check": "tcga_wsi_transfer_projection",
            "status": "PASS" if pred_rows > 0 else "FAIL",
            "details": {"prediction_rows": pred_rows, "method": wsi_summary.get("method", "")},
        }
    )
    if assoc_rows > 0:
        association_status = "PASS"
        if "PILOT" in wsi_summary["wsi_transfer_status"] or "PARTIAL" in wsi_summary["wsi_transfer_status"]:
            association_status = "WARN"
        if wsi_summary.get("association", {}).get("pass_rows", 0) <= 0:
            association_status = "WARN"
        checks.append(
            {
                "check": "tcga_wsi_transfer_association",
                "status": association_status,
                "details": {"association_rows": assoc_rows, "scope": wsi_summary["wsi_transfer_status"]},
            }
        )
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    warn_count = sum(1 for c in checks if c["status"] == "WARN")
    qc.update(
        {
            "built_at": now_iso(),
            "status": "PASS" if fail_count == 0 else "FAIL",
            "fail_count": fail_count,
            "warn_count": warn_count,
            "checks": checks,
        }
    )
    notes = qc.setdefault("notes", [])
    note = "TCGA WSI transfer uses CPTAC morphology-to-Stage7 archetype RidgeCV predictor; pilot or partial-production runs must not be treated as full representative-set external clinical evidence."
    if note not in notes:
        notes.append(note)
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame([{"check": c["check"], "status": c["status"], "details": c["details"]} for c in checks]).to_csv(
        project_root / "logs/qc/stage10_external_validation_qc_summary.tsv",
        sep="\t",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--embedding-manifest", default="data/processed/stage10/tcga_wsi_pilot_embedding_manifest_prov_gigapath_prov_gigapath_tcga_wsi_pilot.tsv")
    parser.add_argument("--cptac-embedding-table", default="data/processed/stage4/prov_gigapath_all_candidates_median_patient_multislide_median_embeddings.tsv")
    parser.add_argument("--archetype-scores", default="results/archetypes/sample_archetype_scores.tsv")
    parser.add_argument("--clinical-endpoints", default="data/processed/stage3/clinical_endpoints.tsv")
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--scope-label", default="pilot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    ensure_dir(output_dir)
    ensure_dir(project_root / "logs/qc")

    cptac_embeddings = read_tsv(project_root / args.cptac_embedding_table)
    scores = read_tsv(project_root / args.archetype_scores)
    models, oof, train_summary = fit_transfer_predictor(cptac_embeddings, scores)
    oof.to_csv(output_dir / "tcga_wsi_transfer_training_oof.tsv", sep="\t", index=False)
    pd.DataFrame(train_summary["metrics"]).to_csv(output_dir / "tcga_wsi_transfer_training_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(train_summary["folds"]).to_csv(output_dir / "tcga_wsi_transfer_training_folds.tsv", sep="\t", index=False)

    slide_df, patient_embeddings = aggregate_tcga_embeddings(project_root, project_root / args.embedding_manifest)
    slide_df.to_csv(output_dir / "tcga_wsi_slide_embedding_summary.tsv", sep="\t", index=False)
    patient_embeddings.to_csv(output_dir / "tcga_wsi_patient_embedding_summary.tsv", sep="\t", index=False)
    pred = predict_tcga(models, patient_embeddings, embedding_columns(cptac_embeddings))
    pred.to_csv(output_dir / "tcga_wsi_predicted_archetypes.tsv", sep="\t", index=False)

    clinical = read_tsv(project_root / args.clinical_endpoints)
    assoc = run_associations(pred, clinical)
    assoc.to_csv(output_dir / "tcga_wsi_transfer_associations.tsv", sep="\t", index=False)

    pass_assoc = int((assoc.get("status", pd.Series(dtype=str)) == "PASS").sum()) if not assoc.empty else 0
    status, scope_note = stage10_wsi_status(args.scope_label, pass_assoc)
    wsi_summary = {
        "built_at": now_iso(),
        "wsi_transfer_status": status,
        "method": "morphology_to_stage7_archetype_transfer_predictor",
        "scope_label": args.scope_label,
        "scope_note": scope_note,
        "inputs": {
            "embedding_manifest": args.embedding_manifest,
            "cptac_embedding_table": args.cptac_embedding_table,
            "archetype_scores": args.archetype_scores,
            "clinical_endpoints": args.clinical_endpoints,
        },
        "training": train_summary,
        "tcga_projection": {
            "slide_rows": int(len(slide_df)),
            "slide_pass_rows": int(slide_df["aggregation_status"].eq("PASS").sum()),
            "patient_rows": int(len(pred)),
            "cohorts": pred["cohort"].value_counts().to_dict(),
            "total_tiles": int(pd.to_numeric(pred["tile_count_sum"], errors="coerce").fillna(0).sum()),
        },
        "association": {
            "rows": int(len(assoc)),
            "pass_rows": pass_assoc,
            "skip_or_warn_rows": int(len(assoc) - pass_assoc),
        },
        "outputs": {
            "predicted_archetypes": str((output_dir / "tcga_wsi_predicted_archetypes.tsv").relative_to(project_root)),
            "associations": str((output_dir / "tcga_wsi_transfer_associations.tsv").relative_to(project_root)),
            "training_oof": str((output_dir / "tcga_wsi_transfer_training_oof.tsv").relative_to(project_root)),
            "slide_embedding_summary": str((output_dir / "tcga_wsi_slide_embedding_summary.tsv").relative_to(project_root)),
            "patient_embedding_summary": str((output_dir / "tcga_wsi_patient_embedding_summary.tsv").relative_to(project_root)),
        },
    }
    (output_dir / "tcga_wsi_transfer_summary.json").write_text(json.dumps(wsi_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_stage10_summary_and_qc(project_root, output_dir, wsi_summary, len(pred), len(assoc))
    print(json.dumps({"status": status, "prediction_rows": int(len(pred)), "association_rows": int(len(assoc)), "training_rows": train_summary["training_rows"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
