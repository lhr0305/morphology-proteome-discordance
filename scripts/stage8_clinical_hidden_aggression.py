#!/usr/bin/env python3
"""Stage 8 clinical modeling for hidden aggressive discordance archetypes."""

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

try:
    from lifelines import CoxPHFitter
except Exception:  # pragma: no cover - optional dependency checked at runtime
    CoxPHFitter = None


RANDOM_SEED = 20260609
META_COLUMNS = {
    "cohort",
    "patient_id",
    "encoder_slug",
    "model_repo",
    "embedding_summary",
    "slide_count",
    "tile_count_sum",
    "tile_count",
    "embedding_path",
    "encoder_role",
    "embedding_dim",
}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"patient_id": str})


def feature_columns(df: pd.DataFrame, prefix: str = "emb_") -> list[str]:
    return [col for col in df.columns if col.startswith(prefix) and col not in META_COLUMNS]


def finite_or_raise(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains non-finite values")


def norm_cdf(x: float) -> float:
    return float(stats.norm.cdf(x))


def safe_logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def logistic_calibration(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    keep = np.isfinite(y) & np.isfinite(prob)
    y = y[keep]
    prob = prob[keep]
    if len(y) == 0 or len(np.unique(y.astype(int))) < 2 or np.nanstd(prob) <= 1e-10:
        return {
            "calibration_status": "SKIP",
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
            "calibration_reason": "insufficient_classes_or_prediction_variation",
        }
    x = safe_logit(prob)
    design = np.column_stack([np.ones(len(y)), x])

    def nll(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(-np.sum(y * eta - np.logaddexp(0.0, eta)))

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = design @ beta
        fitted = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        return design.T @ (fitted - y)

    res = optimize.minimize(nll, np.array([0.0, 1.0]), jac=grad, method="BFGS", options={"maxiter": 500})
    status = "PASS" if res.success else "WARN"
    return {
        "calibration_status": status,
        "calibration_intercept": float(res.x[0]),
        "calibration_slope": float(res.x[1]),
        "calibration_reason": "" if res.success else str(res.message),
    }


def bh_adjust(p_values: list[float]) -> list[float]:
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
        values = np.zeros(len(out), dtype=float)
        for cohort in sorted(out["cohort"].dropna().unique()):
            idx = out["cohort"].to_numpy() == cohort
            block = out.loc[idx, col].to_numpy(dtype=float)
            mean = np.nanmean(block)
            sd = np.nanstd(block, ddof=1)
            if not np.isfinite(sd) or sd <= 1e-8:
                sd = 1.0
            values[idx] = (block - mean) / sd
        out[f"{col}_z"] = values
    return out


def choose_cv_splits(y: np.ndarray, max_splits: int) -> int:
    counts = np.bincount(y.astype(int), minlength=2)
    return int(max(2, min(max_splits, counts.min())))


def fit_morphology_oof(emb: pd.DataFrame, clinical: pd.DataFrame, folds: int, repeats: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    emb_cols = feature_columns(emb)
    if not emb_cols:
        raise RuntimeError("No embedding columns found")
    rows: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for cohort in sorted(emb["cohort"].unique()):
        sub = emb[emb["cohort"] == cohort].merge(
            clinical[["cohort", "patient_id", "advanced_at_presentation"]],
            on=["cohort", "patient_id"],
            how="left",
        )
        trainable = sub.dropna(subset=["advanced_at_presentation"]).reset_index(drop=True)
        y = trainable["advanced_at_presentation"].astype(int).to_numpy()
        x = trainable[emb_cols].to_numpy(dtype=float)
        finite_or_raise(f"{cohort} morphology embeddings", x)
        if len(np.unique(y)) < 2:
            raise RuntimeError(f"{cohort} advanced endpoint has <2 classes")
        splits = choose_cv_splits(y, folds)
        pred_sum = np.zeros(len(trainable), dtype=float)
        pred_count = np.zeros(len(trainable), dtype=int)
        fold_rows: list[dict[str, Any]] = []
        for repeat in range(repeats):
            cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED + repeat)
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y)):
                n_pca = max(1, min(128, x.shape[1], len(train_idx) - 1))
                pipe = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
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
                pipe.fit(x[train_idx], y[train_idx])
                prob = pipe.predict_proba(x[test_idx])[:, 1]
                pred_sum[test_idx] += prob
                pred_count[test_idx] += 1
                fold_rows.append(
                    {
                        "cohort": cohort,
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "x_pca_components": int(n_pca),
                    }
                )
        if not np.all(pred_count == repeats):
            raise RuntimeError(f"{cohort} OOF predictions incomplete")
        trainable["morph_risk"] = pred_sum / pred_count
        all_sub = sub[["cohort", "patient_id", "advanced_at_presentation"]].copy()
        all_sub = all_sub.merge(trainable[["cohort", "patient_id", "morph_risk"]], on=["cohort", "patient_id"], how="left")
        rows.append(all_sub)
        y_pred = trainable["morph_risk"].to_numpy(dtype=float)
        metrics.append(
            {
                "cohort": cohort,
                "n": int(len(y)),
                "events": int(y.sum()),
                "nonevents": int((1 - y).sum()),
                "folds": int(splits),
                "repeats": int(repeats),
                "auc": float(roc_auc_score(y, y_pred)),
                "average_precision": float(average_precision_score(y, y_pred)),
                "brier": float(brier_score_loss(y, y_pred)),
                "mean_pred_event": float(y_pred[y == 1].mean()),
                "mean_pred_nonevent": float(y_pred[y == 0].mean()),
            }
        )
    out = pd.concat(rows, ignore_index=True)
    out = cohort_zscore(out, ["morph_risk"])
    return out, {"embedding_features": len(emb_cols), "fold_metrics": metrics, "folds": folds, "repeats": repeats}


def logistic_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[keep]
    y = y[keep]
    if len(y) == 0 or len(np.unique(y.astype(int))) < 2:
        return {"status": "SKIP", "reason": "less_than_two_classes", "n": int(len(y))}
    design = np.column_stack([np.ones(len(y)), x])
    p = design.shape[1]
    events = int(y.sum())
    nonevents = int(len(y) - events)
    if min(events, nonevents) < p + 1:
        return {"status": "SKIP", "reason": "insufficient_events_per_parameter", "n": int(len(y)), "events": events, "parameters": p}

    def nll(beta: np.ndarray) -> float:
        eta = design @ beta
        ll = np.sum(y * eta - np.logaddexp(0.0, eta))
        penalty = 0.5 * ridge * float(np.sum(beta[1:] ** 2))
        return float(-ll + penalty)

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = design @ beta
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        g = design.T @ (prob - y)
        g[1:] += ridge * beta[1:]
        return g

    start = np.zeros(p, dtype=float)
    base_rate = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    start[0] = math.log(base_rate / (1.0 - base_rate))
    res = optimize.minimize(nll, start, jac=grad, method="BFGS", options={"maxiter": 500})
    beta = np.asarray(res.x, dtype=float)
    eta = design @ beta
    prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    w = prob * (1.0 - prob)
    hessian = design.T @ (design * w[:, None])
    hessian[1:, 1:] += ridge * np.eye(p - 1)
    cov = np.linalg.pinv(hessian)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z)))
    calibration = logistic_calibration(y, prob)
    variables = max(1, p - 1)
    out: dict[str, Any] = {
        "status": "PASS" if res.success else "WARN",
        "optimizer_success": bool(res.success),
        "optimizer_message": str(res.message),
        "n": int(len(y)),
        "events": events,
        "nonevents": nonevents,
        "parameters": int(p),
        "variables": int(variables),
        "epv_min_class_per_variable": float(min(events, nonevents) / variables),
        "auc": float(roc_auc_score(y.astype(int), prob)),
        "average_precision": float(average_precision_score(y.astype(int), prob)),
        "brier": float(brier_score_loss(y.astype(int), prob)),
        "observed_event_rate": float(y.mean()),
        "predicted_event_rate": float(prob.mean()),
        "expected_events": float(prob.sum()),
        "observed_events": float(y.sum()),
        "e_over_o": float(prob.sum() / max(float(y.sum()), 1e-8)),
        "log_likelihood": float(-nll(beta)),
        "coef": beta.tolist(),
        "se": se.tolist(),
        "z": z.tolist(),
        "p": pvals.tolist(),
        "or": np.exp(beta).tolist(),
        "ci95_low": np.exp(beta - 1.96 * se).tolist(),
        "ci95_high": np.exp(beta + 1.96 * se).tolist(),
    }
    out.update(calibration)
    return out


def run_logistic_models(df: pd.DataFrame) -> pd.DataFrame:
    models: list[dict[str, Any]] = []
    endpoint_specs = [("advanced_at_presentation", "advanced disease")]
    recurrence_binary = df["recurrence"].where(df["recurrence"].astype(str).str.lower() == "yes")
    df = df.copy()
    df["recurrence_yes"] = np.where(recurrence_binary.notna(), 1.0, np.nan)
    endpoint_specs.append(("recurrence_yes", "recurrence yes only; Not Reported treated as missing"))
    model_specs = [
        ("clinic_only", ["age_z", "sex_male"]),
        ("clinic_plus_morph", ["age_z", "sex_male", "morph_risk_z"]),
        ("clinic_plus_morph_plus_discordance", ["age_z", "sex_male", "morph_risk_z", "discordance_risk_z"]),
        ("discordance_only", ["discordance_risk_z"]),
        ("morph_only", ["morph_risk_z"]),
        ("hidden_aggressive_group", ["hidden_aggressive_indicator"]),
    ]
    for endpoint, endpoint_note in endpoint_specs:
        for cohort in sorted(df["cohort"].unique()):
            sub = df[df["cohort"] == cohort].copy()
            for model_name, predictors in model_specs:
                available = [col for col in predictors if col in sub.columns]
                if len(available) != len(predictors):
                    status = {"status": "SKIP", "reason": "missing_predictor"}
                else:
                    usable = [col for col in available if sub[col].notna().sum() > 0 and sub[col].nunique(dropna=True) > 1]
                    if not usable and model_name.startswith("clinic"):
                        status = {"status": "SKIP", "reason": "no_nonconstant_clinical_predictor", "n": int(sub[endpoint].notna().sum())}
                    else:
                        status = logistic_fit(sub[usable].to_numpy(dtype=float), sub[endpoint].to_numpy(dtype=float))
                row: dict[str, Any] = {
                    "endpoint": endpoint,
                    "endpoint_note": endpoint_note,
                    "cohort": cohort,
                    "model": model_name,
                    "predictors": ",".join(available),
                    "used_predictors": ",".join(usable) if len(available) == len(predictors) else "",
                    "status": status.get("status", "NA"),
                    "skip_reason": status.get("reason", ""),
                    "n": status.get("n", np.nan),
                    "events": status.get("events", np.nan),
                    "nonevents": status.get("nonevents", np.nan),
                    "parameters": status.get("parameters", np.nan),
                    "variables": status.get("variables", np.nan),
                    "epv_min_class_per_variable": status.get("epv_min_class_per_variable", np.nan),
                    "auc": status.get("auc", np.nan),
                    "average_precision": status.get("average_precision", np.nan),
                    "brier": status.get("brier", np.nan),
                    "observed_event_rate": status.get("observed_event_rate", np.nan),
                    "predicted_event_rate": status.get("predicted_event_rate", np.nan),
                    "expected_events": status.get("expected_events", np.nan),
                    "observed_events": status.get("observed_events", np.nan),
                    "e_over_o": status.get("e_over_o", np.nan),
                    "calibration_status": status.get("calibration_status", ""),
                    "calibration_intercept": status.get("calibration_intercept", np.nan),
                    "calibration_slope": status.get("calibration_slope", np.nan),
                    "calibration_reason": status.get("calibration_reason", ""),
                    "optimizer_success": status.get("optimizer_success", np.nan),
                }
                if status.get("status") in {"PASS", "WARN"}:
                    for idx, term in enumerate(["intercept"] + usable):
                        model_row = row.copy()
                        model_row.update(
                            {
                                "term": term,
                                "coef": status["coef"][idx],
                                "se": status["se"][idx],
                                "z": status["z"][idx],
                                "p": status["p"][idx],
                                "or": status["or"][idx],
                                "ci95_low": status["ci95_low"][idx],
                                "ci95_high": status["ci95_high"][idx],
                            }
                        )
                        models.append(model_row)
                else:
                    row.update({"term": "", "coef": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "or": np.nan, "ci95_low": np.nan, "ci95_high": np.nan})
                    models.append(row)
    out = pd.DataFrame(models)
    out["q_within_endpoint"] = np.nan
    for endpoint, idx in out.groupby("endpoint").groups.items():
        pvals = out.loc[idx, "p"].astype(float).tolist()
        out.loc[idx, "q_within_endpoint"] = bh_adjust(pvals)
    return out


def fixed_random_meta(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in effects.dropna(subset=["coef", "se"]).groupby(["endpoint", "model", "term"]):
        endpoint, model, term = keys
        sub = sub[(sub["se"] > 0) & np.isfinite(sub["coef"])]
        if len(sub) == 0:
            continue
        yi = sub["coef"].to_numpy(dtype=float)
        sei = sub["se"].to_numpy(dtype=float)
        vi = sei**2
        wi = 1.0 / vi
        fixed = float(np.sum(wi * yi) / np.sum(wi))
        fixed_se = float(math.sqrt(1.0 / np.sum(wi)))
        q = float(np.sum(wi * (yi - fixed) ** 2))
        df_q = max(0, len(yi) - 1)
        c = float(np.sum(wi) - np.sum(wi**2) / np.sum(wi))
        tau2 = max(0.0, (q - df_q) / c) if c > 0 else 0.0
        wr = 1.0 / (vi + tau2)
        random = float(np.sum(wr * yi) / np.sum(wr))
        random_se = float(math.sqrt(1.0 / np.sum(wr)))
        for method, est, se in [("fixed_effect_inverse_variance", fixed, fixed_se), ("random_effects_dersimonian_laird", random, random_se)]:
            z = est / se if se > 0 else np.nan
            p = 2.0 * (1.0 - norm_cdf(abs(z))) if np.isfinite(z) else np.nan
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": model,
                    "term": term,
                    "method": method,
                    "k_cohorts": int(len(yi)),
                    "coef": est,
                    "se": se,
                    "z": z,
                    "p": p,
                    "or": float(math.exp(est)),
                    "ci95_low": float(math.exp(est - 1.96 * se)),
                    "ci95_high": float(math.exp(est + 1.96 * se)),
                    "q_heterogeneity": q,
                    "q_df": df_q,
                    "tau2": tau2,
                    "i2": float(max(0.0, (q - df_q) / q) * 100.0) if q > 0 else 0.0,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_within_endpoint"] = np.nan
        for endpoint, idx in out.groupby("endpoint").groups.items():
            out.loc[idx, "q_within_endpoint"] = bh_adjust(out.loc[idx, "p"].astype(float).tolist())
    return out


def build_model_diagnostics(models: pd.DataFrame) -> pd.DataFrame:
    if models.empty:
        return pd.DataFrame()
    keys = [
        "endpoint",
        "cohort",
        "model",
        "status",
        "skip_reason",
        "n",
        "events",
        "nonevents",
        "parameters",
        "variables",
        "epv_min_class_per_variable",
        "auc",
        "average_precision",
        "brier",
        "observed_event_rate",
        "predicted_event_rate",
        "expected_events",
        "observed_events",
        "e_over_o",
        "calibration_status",
        "calibration_intercept",
        "calibration_slope",
        "calibration_reason",
    ]
    diagnostics = models[[col for col in keys if col in models.columns]].drop_duplicates().copy()
    diagnostics["diagnostic_status"] = "PASS"
    diagnostics.loc[
        (diagnostics["endpoint"] != "survival_time_to_event")
        & (diagnostics["status"].isin(["PASS", "WARN"]))
        & (pd.to_numeric(diagnostics["epv_min_class_per_variable"], errors="coerce") < 10),
        "diagnostic_status",
    ] = "WARN"
    diagnostics.loc[
        (diagnostics["endpoint"] != "survival_time_to_event")
        & (diagnostics["status"].isin(["PASS", "WARN"]))
        & (diagnostics["calibration_status"].astype(str) != "PASS"),
        "diagnostic_status",
    ] = "WARN"
    diagnostics.loc[
        (diagnostics["endpoint"] == "survival_time_to_event")
        & (diagnostics["status"].isin(["PASS", "WARN"]))
        & (pd.to_numeric(diagnostics["epv_min_class_per_variable"], errors="coerce") < 10),
        "diagnostic_status",
    ] = "WARN"
    diagnostics.loc[diagnostics["status"].astype(str).eq("SKIP"), "diagnostic_status"] = "SKIP"
    return diagnostics


def bootstrap_logistic_effects(df: pd.DataFrame, n_bootstrap: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED + 8000)
    target_specs = [
        ("advanced_at_presentation", "discordance_only", ["discordance_risk_z"], "discordance_risk_z"),
        ("advanced_at_presentation", "clinic_plus_morph_plus_discordance", ["age_z", "sex_male", "morph_risk_z", "discordance_risk_z"], "discordance_risk_z"),
        ("advanced_at_presentation", "hidden_aggressive_group", ["hidden_aggressive_indicator"], "hidden_aggressive_indicator"),
    ]
    rows: list[dict[str, Any]] = []
    for endpoint, model_name, predictors, term in target_specs:
        for cohort in sorted(df["cohort"].dropna().unique()):
            sub0 = df[df["cohort"] == cohort].copy()
            usable = [col for col in predictors if col in sub0.columns and sub0[col].notna().sum() > 0 and sub0[col].nunique(dropna=True) > 1]
            if term not in usable:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "cohort": cohort,
                        "model": model_name,
                        "term": term,
                        "status": "SKIP",
                        "skip_reason": "target_term_unavailable_or_constant",
                        "n_bootstrap_requested": n_bootstrap,
                        "n_success": 0,
                        "coef_median": np.nan,
                        "coef_ci95_low": np.nan,
                        "coef_ci95_high": np.nan,
                        "or_median": np.nan,
                        "or_ci95_low": np.nan,
                        "or_ci95_high": np.nan,
                    }
                )
                continue
            data = sub0.dropna(subset=[endpoint] + usable).reset_index(drop=True)
            y_all = data[endpoint].astype(int).to_numpy() if endpoint in data.columns else np.array([])
            if len(data) == 0 or len(np.unique(y_all)) < 2:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "cohort": cohort,
                        "model": model_name,
                        "term": term,
                        "status": "SKIP",
                        "skip_reason": "less_than_two_classes",
                        "n_bootstrap_requested": n_bootstrap,
                        "n_success": 0,
                        "coef_median": np.nan,
                        "coef_ci95_low": np.nan,
                        "coef_ci95_high": np.nan,
                        "or_median": np.nan,
                        "or_ci95_low": np.nan,
                        "or_ci95_high": np.nan,
                    }
                )
                continue
            coefs: list[float] = []
            term_index = 1 + usable.index(term)
            for _ in range(n_bootstrap):
                sample_idx = rng.integers(0, len(data), len(data))
                sampled = data.iloc[sample_idx]
                y = sampled[endpoint].astype(int).to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                fit = logistic_fit(sampled[usable].to_numpy(dtype=float), y)
                if fit.get("status") in {"PASS", "WARN"} and len(fit.get("coef", [])) > term_index:
                    coef = float(fit["coef"][term_index])
                    if np.isfinite(coef):
                        coefs.append(coef)
            if len(coefs) < max(30, int(0.2 * n_bootstrap)):
                rows.append(
                    {
                        "endpoint": endpoint,
                        "cohort": cohort,
                        "model": model_name,
                        "term": term,
                        "status": "WARN",
                        "skip_reason": "too_few_successful_bootstrap_fits",
                        "n_bootstrap_requested": n_bootstrap,
                        "n_success": len(coefs),
                        "coef_median": np.nan,
                        "coef_ci95_low": np.nan,
                        "coef_ci95_high": np.nan,
                        "or_median": np.nan,
                        "or_ci95_low": np.nan,
                        "or_ci95_high": np.nan,
                    }
                )
                continue
            arr = np.asarray(coefs, dtype=float)
            rows.append(
                {
                    "endpoint": endpoint,
                    "cohort": cohort,
                    "model": model_name,
                    "term": term,
                    "status": "PASS",
                    "skip_reason": "",
                    "n_bootstrap_requested": n_bootstrap,
                    "n_success": int(len(arr)),
                    "coef_median": float(np.median(arr)),
                    "coef_ci95_low": float(np.quantile(arr, 0.025)),
                    "coef_ci95_high": float(np.quantile(arr, 0.975)),
                    "or_median": float(np.exp(np.median(arr))),
                    "or_ci95_low": float(np.exp(np.quantile(arr, 0.025))),
                    "or_ci95_high": float(np.exp(np.quantile(arr, 0.975))),
                }
            )
    out = pd.DataFrame(rows)
    summary = {
        "n_bootstrap": n_bootstrap,
        "rows": int(len(out)),
        "pass_rows": int((out["status"] == "PASS").sum()) if not out.empty else 0,
        "warn_rows": int((out["status"] == "WARN").sum()) if not out.empty else 0,
        "skip_rows": int((out["status"] == "SKIP").sum()) if not out.empty else 0,
    }
    return out, summary


def run_cox_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if CoxPHFitter is None:
        return pd.DataFrame(
            [
                {
                    "endpoint": "survival_time_to_event",
                    "endpoint_note": "Cox skipped because lifelines is not installed",
                    "cohort": "combined",
                    "model": "cox_secondary",
                    "predictors": "clinic+morph_risk+discordance_risk",
                    "status": "SKIP",
                    "skip_reason": "lifelines_missing",
                    "n": int(df[["survival_time", "survival_event"]].dropna().shape[0]),
                    "events": int(pd.to_numeric(df["survival_event"], errors="coerce").fillna(0).sum()),
                    "nonevents": np.nan,
                    "parameters": np.nan,
                    "variables": np.nan,
                    "epv_min_class_per_variable": np.nan,
                    "auc": np.nan,
                    "average_precision": np.nan,
                    "brier": np.nan,
                    "observed_event_rate": np.nan,
                    "predicted_event_rate": np.nan,
                    "expected_events": np.nan,
                    "observed_events": np.nan,
                    "e_over_o": np.nan,
                    "calibration_status": "",
                    "calibration_intercept": np.nan,
                    "calibration_slope": np.nan,
                    "calibration_reason": "",
                    "optimizer_success": np.nan,
                    "term": "",
                    "coef": np.nan,
                    "se": np.nan,
                    "z": np.nan,
                    "p": np.nan,
                    "or": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "q_within_endpoint": np.nan,
                }
            ]
        )
    model_specs = [
        ("cox_morph_only", ["morph_risk_z"]),
        ("cox_discordance_only", ["discordance_risk_z"]),
        ("cox_morph_plus_discordance", ["morph_risk_z", "discordance_risk_z"]),
        ("cox_hidden_aggressive_group", ["hidden_aggressive_indicator"]),
    ]
    for cohort in sorted(df["cohort"].unique()):
        sub0 = df[df["cohort"] == cohort].copy()
        sub0["survival_time"] = pd.to_numeric(sub0["survival_time"], errors="coerce")
        sub0["survival_event"] = pd.to_numeric(sub0["survival_event"], errors="coerce")
        for model_name, predictors in model_specs:
            sub = sub0[["survival_time", "survival_event"] + predictors].replace([np.inf, -np.inf], np.nan).dropna()
            n = int(len(sub))
            events = int(sub["survival_event"].sum()) if n else 0
            usable = [col for col in predictors if sub[col].nunique(dropna=True) > 1]
            if n == 0 or events < max(10, 5 * max(1, len(usable))) or not usable:
                rows.append(
                    {
                        "endpoint": "survival_time_to_event",
                        "endpoint_note": "Cox secondary survival model",
                        "cohort": cohort,
                        "model": model_name,
                        "predictors": ",".join(predictors),
                        "used_predictors": ",".join(usable),
                        "status": "SKIP",
                        "skip_reason": "insufficient_survival_records_events_or_predictor_variation",
                        "n": n,
                        "events": events,
                        "nonevents": n - events,
                        "parameters": len(usable),
                        "variables": len(usable),
                        "epv_min_class_per_variable": float(events / max(len(usable), 1)) if usable else np.nan,
                        "auc": np.nan,
                        "average_precision": np.nan,
                        "brier": np.nan,
                        "observed_event_rate": np.nan,
                        "predicted_event_rate": np.nan,
                        "expected_events": np.nan,
                        "observed_events": np.nan,
                        "e_over_o": np.nan,
                        "calibration_status": "",
                        "calibration_intercept": np.nan,
                        "calibration_slope": np.nan,
                        "calibration_reason": "",
                        "optimizer_success": np.nan,
                        "term": "",
                        "coef": np.nan,
                        "se": np.nan,
                        "z": np.nan,
                        "p": np.nan,
                        "or": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "q_within_endpoint": np.nan,
                    }
                )
                continue
            cph = CoxPHFitter(penalizer=0.01)
            fit_status = "PASS"
            skip_reason = ""
            try:
                cph.fit(sub[["survival_time", "survival_event"] + usable], duration_col="survival_time", event_col="survival_event", show_progress=False)
                csum = cph.summary.reset_index(names="term")
            except Exception as exc:
                fit_status = "WARN"
                skip_reason = f"cox_fit_failed:{type(exc).__name__}:{exc}"
                csum = pd.DataFrame()
            if csum.empty:
                rows.append(
                    {
                        "endpoint": "survival_time_to_event",
                        "endpoint_note": "Cox secondary survival model",
                        "cohort": cohort,
                        "model": model_name,
                        "predictors": ",".join(predictors),
                        "used_predictors": ",".join(usable),
                        "status": fit_status,
                        "skip_reason": skip_reason,
                        "n": n,
                        "events": events,
                        "nonevents": n - events,
                        "parameters": len(usable),
                        "variables": len(usable),
                        "epv_min_class_per_variable": float(events / max(len(usable), 1)) if usable else np.nan,
                        "auc": np.nan,
                        "average_precision": np.nan,
                        "brier": np.nan,
                        "observed_event_rate": np.nan,
                        "predicted_event_rate": np.nan,
                        "expected_events": np.nan,
                        "observed_events": np.nan,
                        "e_over_o": np.nan,
                        "calibration_status": "",
                        "calibration_intercept": np.nan,
                        "calibration_slope": np.nan,
                        "calibration_reason": "",
                        "optimizer_success": fit_status == "PASS",
                        "term": "",
                        "coef": np.nan,
                        "se": np.nan,
                        "z": np.nan,
                        "p": np.nan,
                        "or": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "q_within_endpoint": np.nan,
                    }
                )
                continue
            for rec in csum.to_dict(orient="records"):
                rows.append(
                    {
                        "endpoint": "survival_time_to_event",
                        "endpoint_note": "Cox secondary survival model",
                        "cohort": cohort,
                        "model": model_name,
                        "predictors": ",".join(predictors),
                        "used_predictors": ",".join(usable),
                        "status": fit_status,
                        "skip_reason": skip_reason,
                        "n": n,
                        "events": events,
                        "nonevents": n - events,
                        "parameters": len(usable),
                        "variables": len(usable),
                        "epv_min_class_per_variable": float(events / max(len(usable), 1)) if usable else np.nan,
                        "auc": np.nan,
                        "average_precision": np.nan,
                        "brier": np.nan,
                        "observed_event_rate": np.nan,
                        "predicted_event_rate": np.nan,
                        "expected_events": np.nan,
                        "observed_events": np.nan,
                        "e_over_o": np.nan,
                        "calibration_status": "",
                        "calibration_intercept": np.nan,
                        "calibration_slope": np.nan,
                        "calibration_reason": "",
                        "optimizer_success": fit_status == "PASS",
                        "term": rec["term"],
                        "coef": rec["coef"],
                        "se": rec["se(coef)"],
                        "z": rec["z"],
                        "p": rec["p"],
                        "or": rec["exp(coef)"],
                        "ci95_low": rec["exp(coef) lower 95%"],
                        "ci95_high": rec["exp(coef) upper 95%"],
                        "q_within_endpoint": np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_within_endpoint"] = bh_adjust(out["p"].astype(float).tolist())
    return out


def archetype_direction_table(df: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    archetype_cols = [col for col in df.columns if col.endswith("_score") and col.startswith("A")]
    for col in archetype_cols:
        for cohort, sub in df.dropna(subset=["advanced_at_presentation", col]).groupby("cohort"):
            y = sub["advanced_at_presentation"].astype(int).to_numpy()
            x = sub[col].to_numpy(dtype=float)
            if len(np.unique(y)) < 2:
                continue
            r, p = stats.pointbiserialr(y, x)
            rows.append(
                {
                    "archetype_score": col,
                    "archetype": col.replace("_score", ""),
                    "cohort": cohort,
                    "n": int(len(y)),
                    "events": int(y.sum()),
                    "point_biserial_r": float(r),
                    "p": float(p),
                    "mean_event": float(x[y == 1].mean()),
                    "mean_nonevent": float(x[y == 0].mean()),
                }
            )
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab, {
            "status": "no_archetype_testable",
            "strict_adverse": [],
            "strict_protective": [],
            "directional_adverse": [],
            "directional_protective": [],
            "risk_archetypes": [],
            "protective_archetypes": [],
            "risk_status": "no_archetype_testable",
            "warning": "No archetype had enough advanced endpoint data for direction testing.",
        }
    meta = []
    for archetype, sub in tab.groupby("archetype"):
        vals = sub["point_biserial_r"].to_numpy(dtype=float)
        pvals = sub["p"].to_numpy(dtype=float)
        meta.append(
            {
                "archetype": archetype,
                "mean_r": float(np.mean(vals)),
                "n_cohorts": int(len(vals)),
                "min_p": float(np.nanmin(pvals)) if np.isfinite(pvals).any() else np.nan,
                "all_positive": bool(np.all(vals > 0)),
                "all_negative": bool(np.all(vals < 0)),
                "any_significant_positive": bool(np.any((vals > 0) & (pvals < alpha))),
                "any_significant_negative": bool(np.any((vals < 0) & (pvals < alpha))),
            }
        )
    meta_df = pd.DataFrame(meta)
    strict_adverse = meta_df[meta_df["any_significant_positive"]]["archetype"].tolist()
    strict_protective = meta_df[meta_df["any_significant_negative"]]["archetype"].tolist()
    directional_adverse = meta_df[meta_df["mean_r"] > 0]["archetype"].tolist()
    directional_protective = meta_df[meta_df["mean_r"] < 0]["archetype"].tolist()
    if strict_adverse:
        risk_archetypes = strict_adverse
        risk_status = "strict_significant_adverse_archetypes"
        warning = ""
    elif directional_adverse:
        risk_archetypes = directional_adverse
        risk_status = "exploratory_directional_proxy_no_significant_adverse_archetype"
        warning = (
            "No archetype is significantly positively associated with advanced_at_presentation at alpha=0.05; "
            "discordance_risk uses non-significant directional orientation only."
        )
    else:
        risk_archetypes = []
        risk_status = "no_positive_advanced_association_archetype"
        warning = "No archetype had positive advanced endpoint association; discordance_risk is undefined."
    protective = strict_protective if strict_protective else directional_protective
    if strict_protective:
        protective_status = "strict_significant_protective_archetypes"
    elif directional_protective:
        protective_status = "exploratory_directional_proxy_no_significant_protective_archetype"
    else:
        protective_status = "no_negative_advanced_association_archetype"
    tab = tab.merge(meta_df, on="archetype", how="left", suffixes=("", "_meta"))
    return tab, {
        "status": risk_status,
        "strict_adverse": strict_adverse,
        "strict_protective": strict_protective,
        "directional_adverse": directional_adverse,
        "directional_protective": directional_protective,
        "risk_archetypes": risk_archetypes,
        "protective_archetypes": protective,
        "protective_status": protective_status,
        "warning": warning,
        "alpha": alpha,
    }


def build_stage8(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / "logs" / "qc").mkdir(parents=True, exist_ok=True)

    stage7_summary = json.loads((root / args.stage7_summary).read_text())
    if stage7_summary.get("status") != "PASS":
        raise RuntimeError(f"Stage 7 summary is not PASS: {args.stage7_summary}")

    emb = read_table(root / args.embedding_table)
    clinical = read_table(root / args.clinical_endpoints)
    arch = read_table(root / args.archetype_scores)
    disc = read_table(root / args.discordance_scores)

    morph, morph_meta = fit_morphology_oof(emb, clinical, args.folds, args.repeats)
    morph.to_csv(out_dir / "morphology_risk_oof.tsv", sep="\t", index=False)

    morph_signal = morph[["cohort", "patient_id", "morph_risk", "morph_risk_z"]].copy()
    df = arch.merge(morph_signal, on=["cohort", "patient_id"], how="left")
    keep_cols = [
        "cohort",
        "patient_id",
        "age",
        "sex",
        "stage",
        "N_stage",
        "M_stage",
        "grade",
        "recurrence",
        "survival_time",
        "survival_event",
        "advanced_at_presentation",
        "advanced_endpoint_source",
    ]
    df = df.merge(clinical[[col for col in keep_cols if col in clinical.columns]], on=["cohort", "patient_id"], how="left")
    df = df.merge(disc, on=["cohort", "patient_id"], how="left")

    archetype_tests, direction_meta = archetype_direction_table(df)
    direction_status = direction_meta["status"]
    adverse = direction_meta["risk_archetypes"]
    protective = direction_meta["protective_archetypes"]
    if not archetype_tests.empty:
        archetype_tests.to_csv(out_dir / "archetype_advanced_association.tsv", sep="\t", index=False)

    score_cols = [f"{a}_score" for a in adverse + protective if f"{a}_score" in df.columns]
    df = cohort_zscore(df, [col for col in score_cols if col in df.columns])
    if adverse:
        adverse_z = [f"{a}_score_z" for a in adverse if f"{a}_score_z" in df.columns]
        adverse_value = df[adverse_z].mean(axis=1)
    else:
        adverse_z = []
        adverse_value = pd.Series(np.nan, index=df.index)
    if protective:
        protective_z = [f"{a}_score_z" for a in protective if f"{a}_score_z" in df.columns]
        protective_value = df[protective_z].mean(axis=1)
    else:
        protective_z = []
        protective_value = pd.Series(0.0, index=df.index)
    df["discordance_risk"] = adverse_value - protective_value
    if not adverse:
        df["discordance_risk_status"] = "no_positive_advanced_association_archetype"
    else:
        df["discordance_risk_status"] = direction_status
    df = cohort_zscore(df, ["discordance_risk"])

    discordance_cols = [
        "cohort",
        "patient_id",
        "discordance_risk",
        "discordance_risk_z",
        "discordance_risk_status",
        "dominant_archetype",
        "dominant_archetype_fraction",
    ] + [col for col in arch.columns if col.startswith("A")]
    df[[col for col in discordance_cols if col in df.columns]].to_csv(out_dir / "discordance_risk.tsv", sep="\t", index=False)

    df["age_numeric"] = pd.to_numeric(df["age"], errors="coerce")
    df = cohort_zscore(df, ["age_numeric"])
    df["age_z"] = df["age_numeric_z"]
    df["sex_male"] = np.where(df["sex"].astype(str).str.lower() == "male", 1.0, np.where(df["sex"].astype(str).str.lower() == "female", 0.0, np.nan))

    all_group_frames: list[pd.DataFrame] = []
    for cutoff_name, quantile in [("top25", 0.75), ("top33", 2.0 / 3.0), ("top40", 0.60)]:
        sub = df.copy()
        sub["cutoff"] = cutoff_name
        sub["discordance_high_quantile"] = quantile
        sub["morph_group"] = "missing_morph"
        sub["discordance_group"] = "missing_discordance"
        sub["hidden_aggressive_group"] = "unassigned"
        for cohort in sorted(sub["cohort"].unique()):
            idx = sub["cohort"] == cohort
            morph_cut = sub.loc[idx, "morph_risk"].median(skipna=True)
            disc_cut = sub.loc[idx, "discordance_risk"].quantile(quantile)
            sub.loc[idx & (sub["morph_risk"] <= morph_cut), "morph_group"] = "morphology_low"
            sub.loc[idx & (sub["morph_risk"] > morph_cut), "morph_group"] = "morphology_high"
            sub.loc[idx & (sub["discordance_risk"] <= disc_cut), "discordance_group"] = "discordance_low"
            sub.loc[idx & (sub["discordance_risk"] > disc_cut), "discordance_group"] = "discordance_high"
        sub.loc[(sub["morph_group"] == "morphology_low") & (sub["discordance_group"] == "discordance_low"), "hidden_aggressive_group"] = "true_low_risk"
        sub.loc[(sub["morph_group"] == "morphology_low") & (sub["discordance_group"] == "discordance_high"), "hidden_aggressive_group"] = "hidden_aggressive"
        sub.loc[(sub["morph_group"] == "morphology_high") & (sub["discordance_group"] == "discordance_low"), "hidden_aggressive_group"] = "visual_high_molecular_quiet"
        sub.loc[(sub["morph_group"] == "morphology_high") & (sub["discordance_group"] == "discordance_high"), "hidden_aggressive_group"] = "overt_high_risk"
        all_group_frames.append(sub)
    groups = pd.concat(all_group_frames, ignore_index=True)
    groups["hidden_aggressive_indicator"] = (groups["hidden_aggressive_group"] == "hidden_aggressive").astype(float)
    groups.to_csv(out_dir / "hidden_aggressive_groups.tsv", sep="\t", index=False)

    default_groups = groups[groups["cutoff"] == "top33"].copy()
    models = run_logistic_models(default_groups)
    cox_models = run_cox_models(default_groups)
    cox_note = "cox_modeled_with_lifelines" if not cox_models.empty and (cox_models["status"] == "PASS").any() else "cox_skipped_or_failed"
    models = pd.concat([models, cox_models], ignore_index=True)
    models.to_csv(out_dir / "clinical_association_models.tsv", sep="\t", index=False)

    model_diagnostics = build_model_diagnostics(models)
    model_diagnostics.to_csv(out_dir / "model_diagnostics.tsv", sep="\t", index=False)

    bootstrap_effects, bootstrap_summary = bootstrap_logistic_effects(default_groups, args.bootstrap)
    bootstrap_effects.to_csv(out_dir / "bootstrap_effects.tsv", sep="\t", index=False)

    meta = fixed_random_meta(models[(models["status"].isin(["PASS", "WARN"])) & (models["term"] != "intercept")])
    meta.to_csv(out_dir / "meta_analysis_results.tsv", sep="\t", index=False)

    hidden_summary = (
        groups.groupby(["cutoff", "cohort", "hidden_aggressive_group"], dropna=False)
        .agg(
            n=("patient_id", "count"),
            advanced_nonmissing=("advanced_at_presentation", lambda x: int(x.notna().sum())),
            advanced_events=("advanced_at_presentation", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).sum())),
            mean_morph_risk=("morph_risk", "mean"),
            mean_discordance_risk=("discordance_risk", "mean"),
        )
        .reset_index()
    )
    hidden_summary.to_csv(out_dir / "hidden_aggressive_group_summary.tsv", sep="\t", index=False)

    significant_primary = models[
        (models["endpoint"] == "advanced_at_presentation")
        & (models["status"].isin(["PASS", "WARN"]))
        & (models["term"].isin(["discordance_risk_z", "hidden_aggressive_indicator"]))
        & (pd.to_numeric(models["p"], errors="coerce") < 0.05)
    ]
    logistic_diag_for_summary = model_diagnostics[
        (model_diagnostics["endpoint"] != "survival_time_to_event")
        & (model_diagnostics["status"].isin(["PASS", "WARN"]))
    ]
    fitted_diag_for_summary = model_diagnostics[model_diagnostics["status"].isin(["PASS", "WARN"])]
    cox_pass_cohorts = sorted(models.loc[(models["endpoint"] == "survival_time_to_event") & (models["status"] == "PASS"), "cohort"].dropna().astype(str).unique())
    cox_pass_count = int(((models["endpoint"] == "survival_time_to_event") & (models["status"] == "PASS")).sum())
    epv_for_summary = pd.to_numeric(fitted_diag_for_summary["epv_min_class_per_variable"], errors="coerce") if not fitted_diag_for_summary.empty else pd.Series(dtype=float)
    stage8_diagnostic_summary = pd.DataFrame(
        [
            {
                "diagnostic": "strict_adverse_archetypes",
                "status": "PASS" if direction_meta["strict_adverse"] else "WARN",
                "value": ",".join(direction_meta["strict_adverse"]),
                "note": "proposal strict adverse requires significant positive advanced endpoint association",
            },
            {
                "diagnostic": "directional_proxy_archetypes",
                "status": "WARN" if direction_meta["status"].startswith("exploratory") else "PASS",
                "value": ",".join(adverse),
                "note": direction_meta["warning"],
            },
            {
                "diagnostic": "logistic_calibration",
                "status": "PASS" if not logistic_diag_for_summary.empty and logistic_diag_for_summary["calibration_status"].astype(str).eq("PASS").all() else "WARN",
                "value": str(int((model_diagnostics.get("calibration_status", pd.Series(dtype=str)) == "PASS").sum())) if not model_diagnostics.empty else "0",
                "note": "calibration intercept/slope recorded in model_diagnostics.tsv",
            },
            {
                "diagnostic": "epv",
                "status": "PASS" if len(epv_for_summary.dropna()) > 0 and (epv_for_summary.dropna() >= 10).all() else "WARN",
                "value": str(float(epv_for_summary.min())) if len(epv_for_summary.dropna()) else "NA",
                "note": "minimum class-count per variable for logistic; events per variable for Cox",
            },
            {
                "diagnostic": "bootstrap_ci",
                "status": "PASS" if bootstrap_summary["pass_rows"] > 0 else "WARN",
                "value": json.dumps(bootstrap_summary, ensure_ascii=False),
                "note": "bootstrap CI generated for core advanced endpoint terms",
            },
            {
                "diagnostic": "primary_clinical_significance",
                "status": "PASS" if len(significant_primary) > 0 else "WARN",
                "value": str(int(len(significant_primary))),
                "note": "discordance and hidden-aggressive primary terms are exploratory if no p<0.05",
            },
            {
                "diagnostic": "survival_scope",
                "status": "WARN" if cox_pass_count > 0 and len(cox_pass_cohorts) == 1 else "PASS",
                "value": ",".join(cox_pass_cohorts),
                "note": "Cox is PDAC-only if only one cohort has usable survival events",
            },
        ]
    )
    stage8_diagnostic_summary.to_csv(out_dir / "stage8_diagnostic_summary.tsv", sep="\t", index=False)

    checks: list[dict[str, Any]] = []
    def add_check(name: str, status: str, details: dict[str, Any] | None = None) -> None:
        checks.append({"check": name, "status": status, "details": details or {}})

    add_check("stage7_summary_pass", "PASS", {"status": stage7_summary.get("status"), "selected_rank": stage7_summary.get("selected_rank")})
    add_check("morphology_risk_rows", "PASS" if len(morph) == len(arch) else "FAIL", {"rows": len(morph), "expected": len(arch)})
    add_check("default_group_rows", "PASS" if len(default_groups) == len(arch) else "FAIL", {"rows": len(default_groups), "expected": len(arch)})
    add_check("advanced_nonmissing", "PASS" if int(default_groups["advanced_at_presentation"].notna().sum()) >= 0.7 * len(default_groups) else "WARN", {"nonmissing": int(default_groups["advanced_at_presentation"].notna().sum()), "rows": len(default_groups)})
    add_check("strict_adverse_archetype_defined", "PASS" if direction_meta["strict_adverse"] else "WARN", direction_meta)
    add_check("discordance_risk_defined", "PASS" if adverse and not direction_status.startswith("exploratory") else "WARN", {"adverse_archetypes": adverse, "protective_archetypes": protective, "status": direction_status, "warning": direction_meta["warning"]})
    for cutoff in ["top25", "top33", "top40"]:
        counts = groups[groups["cutoff"] == cutoff]["hidden_aggressive_group"].value_counts(dropna=False).to_dict()
        add_check(f"group_counts::{cutoff}", "PASS" if "hidden_aggressive" in counts else "WARN", counts)
    for path in [
        "morphology_risk_oof.tsv",
        "discordance_risk.tsv",
        "hidden_aggressive_groups.tsv",
        "clinical_association_models.tsv",
        "meta_analysis_results.tsv",
        "model_diagnostics.tsv",
        "bootstrap_effects.tsv",
        "stage8_diagnostic_summary.tsv",
    ]:
        p = out_dir / path
        add_check(f"output_exists::{path}", "PASS" if p.exists() and p.stat().st_size > 0 else "FAIL", {"path": str(p.relative_to(root)), "size": p.stat().st_size if p.exists() else 0})

    if not model_diagnostics.empty:
        logistic_diag = model_diagnostics[(model_diagnostics["endpoint"] != "survival_time_to_event") & (model_diagnostics["status"].isin(["PASS", "WARN"]))]
        cox_diag = model_diagnostics[(model_diagnostics["endpoint"] == "survival_time_to_event") & (model_diagnostics["status"].isin(["PASS", "WARN"]))]
        add_check(
            "logistic_calibration_recorded",
            "PASS" if not logistic_diag.empty and logistic_diag["calibration_status"].astype(str).eq("PASS").all() else "WARN",
            {
                "evaluated_models": int(len(logistic_diag)),
                "pass_calibration": int(logistic_diag["calibration_status"].astype(str).eq("PASS").sum()) if not logistic_diag.empty else 0,
            },
        )
        epv_values = pd.to_numeric(logistic_diag["epv_min_class_per_variable"], errors="coerce")
        add_check(
            "logistic_epv_recorded",
            "PASS" if len(epv_values.dropna()) > 0 and float(epv_values.min()) >= 10.0 else "WARN",
            {"min_epv": float(epv_values.min()) if len(epv_values.dropna()) else np.nan},
        )
        cox_epv = pd.to_numeric(cox_diag["epv_min_class_per_variable"], errors="coerce")
        add_check(
            "cox_event_counts_recorded",
            "PASS" if len(cox_epv.dropna()) > 0 else "WARN",
            {"cox_models": int(len(cox_diag)), "min_event_per_variable": float(cox_epv.min()) if len(cox_epv.dropna()) else np.nan},
        )
    add_check("bootstrap_ci_recorded", "PASS" if bootstrap_summary["pass_rows"] > 0 else "WARN", bootstrap_summary)

    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    warn_count = sum(1 for c in checks if c["status"] == "WARN")
    summary: dict[str, Any] = {
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS" if fail_count == 0 else "FAIL",
        "fail_count": fail_count,
        "warn_count": warn_count,
        "stage7_summary": args.stage7_summary,
        "inputs": {
            "embedding_table": args.embedding_table,
            "clinical_endpoints": args.clinical_endpoints,
            "archetype_scores": args.archetype_scores,
            "discordance_scores": args.discordance_scores,
        },
        "morphology_model": morph_meta,
        "archetype_direction_status": direction_status,
        "archetype_direction_meta": direction_meta,
        "strict_adverse_archetypes": direction_meta["strict_adverse"],
        "strict_protective_archetypes": direction_meta["strict_protective"],
        "directional_proxy_adverse_archetypes": adverse,
        "directional_proxy_protective_archetypes": protective,
        "default_cutoff": "top33",
        "survival_status": cox_note,
        "bootstrap_summary": bootstrap_summary,
        "clinical_evidence_status": "exploratory_directional_not_significant" if len(significant_primary) == 0 else "primary_terms_nominally_significant",
        "survival_scope": {
            "pass_cohorts": cox_pass_cohorts,
            "pass_term_rows": cox_pass_count,
            "note": "COAD survival is unavailable/insufficient; current Cox evidence is PDAC-only when pass_cohorts has one entry.",
        },
        "outputs": {
            "morphology_risk_oof": str((out_dir / "morphology_risk_oof.tsv").relative_to(root)),
            "discordance_risk": str((out_dir / "discordance_risk.tsv").relative_to(root)),
            "hidden_aggressive_groups": str((out_dir / "hidden_aggressive_groups.tsv").relative_to(root)),
            "clinical_association_models": str((out_dir / "clinical_association_models.tsv").relative_to(root)),
            "meta_analysis_results": str((out_dir / "meta_analysis_results.tsv").relative_to(root)),
            "hidden_aggressive_group_summary": str((out_dir / "hidden_aggressive_group_summary.tsv").relative_to(root)),
            "model_diagnostics": str((out_dir / "model_diagnostics.tsv").relative_to(root)),
            "bootstrap_effects": str((out_dir / "bootstrap_effects.tsv").relative_to(root)),
            "stage8_diagnostic_summary": str((out_dir / "stage8_diagnostic_summary.tsv").relative_to(root)),
        },
        "checks": checks,
        "notes": [
            "No Stage 8 archetype is significantly positively associated with advanced_at_presentation at alpha=0.05; A2 is used only as a non-significant directional proxy when present.",
            "Clinical association results are exploratory: discordance/hidden aggressive primary terms do not provide definitive clinical significance in this run unless clinical_evidence_status says otherwise.",
            "Recurrence is modeled only for explicit Yes records; Not Reported is treated as missing, so recurrence models may be skipped.",
            "Cox survival modeling is secondary and only fit where event counts and predictor variation are sufficient; COAD survival is unavailable/insufficient and current Cox evidence is PDAC-only.",
            "Survival meta-analysis rows with k_cohorts=1 are descriptive single-cohort summaries, not cross-cohort survival meta-analysis.",
            "RNA comparator remains skipped because matched CPTAC RNA target was not available in Stage 5.",
            "Calibration, bootstrap CI, and EPV diagnostics are recorded in model_diagnostics.tsv, bootstrap_effects.tsv, and stage8_diagnostic_summary.tsv.",
        ],
    }
    (out_dir / "stage8_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    qc_json = root / "logs" / "qc" / "stage8_clinical_qc.json"
    qc_tsv = root / "logs" / "qc" / "stage8_clinical_qc_summary.tsv"
    qc_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(checks).to_csv(qc_tsv, sep="\t", index=False)
    print(json.dumps({"status": summary["status"], "fail_count": fail_count, "warn_count": warn_count, "adverse_archetypes": adverse, "survival_status": cox_note}, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--embedding-table", default="data/processed/stage4/prov_gigapath_all_candidates_median_patient_multislide_median_embeddings.tsv")
    parser.add_argument("--clinical-endpoints", default="data/processed/stage3/clinical_endpoints.tsv")
    parser.add_argument("--archetype-scores", default="results/archetypes/sample_archetype_scores.tsv")
    parser.add_argument("--discordance-scores", default="results/discordance/sample_discordance_scores.tsv")
    parser.add_argument("--stage7-summary", default="results/archetypes/stage7_build_summary.json")
    parser.add_argument("--output-dir", default="results/clinical")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_stage8(args)


if __name__ == "__main__":
    main()
