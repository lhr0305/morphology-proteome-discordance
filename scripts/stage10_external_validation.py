#!/usr/bin/env python3
"""Stage 10 TCGA RNA proxy external validation."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 20260609
META_COLUMNS = {"cohort", "patient_id"}


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"patient_id": str})


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return read_tsv(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


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
    design = np.column_stack([np.ones(len(y)), x])
    try:
        from scipy import optimize

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
        terms = ["intercept"] + predictor_names
        rows: list[dict[str, Any]] = []
        for i, term in enumerate(terms):
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
                    "auc": float(roc_auc_score(y.astype(int), prob)),
                    "average_precision": float(average_precision_score(y.astype(int), prob)),
                    "brier": float(brier_score_loss(y.astype(int), prob)),
                }
            )
        return rows
    except Exception as exc:
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
    except Exception as exc:
        return [{"status": "SKIP", "skip_reason": f"cox_fit_error:{type(exc).__name__}:{exc}", "term": "", "n": int(len(sub))}]


def fixed_effect_meta(rows: pd.DataFrame, effect_col: str) -> pd.DataFrame:
    out_rows = []
    if rows.empty or "se" not in rows.columns:
        return pd.DataFrame()
    usable = rows[(rows["status"].isin(["PASS", "WARN"])) & rows[effect_col].notna() & rows["se"].notna() & (rows["se"] > 0)].copy()
    for keys, sub in usable.groupby(["endpoint", "model", "term"], sort=True):
        endpoint, model, term = keys
        if len(sub) == 0:
            continue
        yi = sub[effect_col].to_numpy(dtype=float)
        vi = sub["se"].to_numpy(dtype=float) ** 2
        wi = 1.0 / vi
        est = float(np.sum(wi * yi) / np.sum(wi))
        se = float(math.sqrt(1.0 / np.sum(wi)))
        z = est / se if se > 0 else np.nan
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        q = float(np.sum(wi * (yi - est) ** 2))
        df_q = max(0, len(yi) - 1)
        out_rows.append(
            {
                "endpoint": endpoint,
                "model": model,
                "term": term,
                "method": "fixed_effect_inverse_variance",
                "k_cohorts": int(len(yi)),
                effect_col: est,
                "se": se,
                "z": z,
                "p": p,
                "exp_effect": float(np.exp(est)),
                "ci95_low": float(np.exp(est - 1.96 * se)),
                "ci95_high": float(np.exp(est + 1.96 * se)),
                "q_heterogeneity": q,
                "q_df": df_q,
                "i2": float(max(0.0, (q - df_q) / q) * 100.0) if q > 0 else 0.0,
            }
        )
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["q_within_endpoint"] = np.nan
        for endpoint, idx in out.groupby("endpoint").groups.items():
            out.loc[idx, "q_within_endpoint"] = bh_adjust(out.loc[idx, "p"].tolist())
    return out


def load_stage9_gate(project_root: Path) -> dict[str, Any]:
    qc_path = project_root / "logs/qc/stage9_spatial_qc.json"
    summary_path = project_root / "results/spatial/stage9_build_summary.json"
    out: dict[str, Any] = {"stage9_qc_path": str(qc_path.relative_to(project_root))}
    if qc_path.exists():
        qc = json.loads(qc_path.read_text())
        out.update(
            {
                "stage9_status": qc.get("status"),
                "stage9_fail_count": qc.get("fail_count"),
                "stage9_warn_count": qc.get("warn_count"),
            }
        )
    else:
        out.update({"stage9_status": "MISSING", "stage9_fail_count": np.nan, "stage9_warn_count": np.nan})
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        out["stage9_target_status"] = s.get("target", {}).get("status")
        out["stage9_method"] = s.get("method")
    return out


def build_fixed_loading_proxy(
    *,
    rna: pd.DataFrame,
    loadings: pd.DataFrame,
    target_archetypes: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rna_features = feature_columns(rna)
    rna_feature_set = set(rna_features)
    rows = []
    weight_rows = []
    for archetype in target_archetypes:
        sub = loadings[
            (loadings["archetype"] == archetype)
            & (loadings["target_set"] == "protein_pathway")
            & (loadings["direction"] == "positive_discordance")
            & (loadings["feature"].isin(rna_feature_set))
        ].copy()
        if sub.empty:
            continue
        sub["weight"] = pd.to_numeric(sub["loading"], errors="coerce")
        sub = sub.dropna(subset=["weight"])
        sub = sub[sub["weight"] > 0].copy()
        if sub.empty:
            continue
        total = sub["weight"].sum()
        sub["normalized_weight"] = sub["weight"] / total
        for row in sub.itertuples(index=False):
            weight_rows.append(
                {
                    "proxy_name": f"{archetype}_rna_loading_proxy",
                    "archetype": archetype,
                    "feature": row.feature,
                    "direction": row.direction,
                    "weight": float(row.weight),
                    "normalized_weight": float(row.normalized_weight),
                    "loading_rank": int(row.loading_rank),
                }
            )
        x = rna[sub["feature"].tolist()].to_numpy(dtype=float)
        w = sub["normalized_weight"].to_numpy(dtype=float)
        score = x @ w
        rows.append(pd.DataFrame({"proxy_name": f"{archetype}_rna_loading_proxy", "proxy_score_raw": score}))
    if not rows:
        raise RuntimeError("No overlapping positive protein-pathway loadings found for TCGA RNA proxy")
    base = rna[["cohort", "patient_id"]].copy().reset_index(drop=True)
    wide = base.copy()
    for block in rows:
        name = str(block["proxy_name"].iloc[0])
        wide[name] = block["proxy_score_raw"].to_numpy(dtype=float)
    proxy_cols = [c for c in wide.columns if c.endswith("_rna_loading_proxy")]
    wide = cohort_zscore(wide, proxy_cols)
    if {"A2_rna_loading_proxy_z", "A1_rna_loading_proxy_z"}.issubset(wide.columns):
        wide["discordance_rna_proxy"] = wide["A2_rna_loading_proxy_z"] - wide["A1_rna_loading_proxy_z"]
    elif "A2_rna_loading_proxy_z" in wide.columns:
        wide["discordance_rna_proxy"] = wide["A2_rna_loading_proxy_z"]
    else:
        first = proxy_cols[0]
        wide["discordance_rna_proxy"] = wide[f"{first}_z"]
    wide = cohort_zscore(wide, ["discordance_rna_proxy"])
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(output_dir / "tcga_rna_proxy_signature.tsv", sep="\t", index=False)
    summary = {
        "proxy_method": "fixed_stage7_positive_protein_pathway_loading_overlap_with_tcga_rna_pathways",
        "proxy_columns": proxy_cols,
        "overlap_features_by_proxy": weights.groupby("proxy_name")["feature"].nunique().to_dict(),
    }
    return wide, weights, summary


def train_matched_rna_elastic_net_if_available(
    *,
    project_root: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Stage 1-3 currently keep TCGA RNA as external validation material and no
    # matched CPTAC COAD/PDAC RNA matrix is present. This function writes a real
    # audit table rather than silently skipping the proposal branch.
    candidate_paths = [
        project_root / "data/processed/stage3/cptac_rna_pathway_matrix.tsv",
        project_root / "data/processed/stage3/matched_rna_pathway_matrix.tsv",
        project_root / "data/processed/stage3/cptac_rna_log_tpm_matrix.parquet",
    ]
    rows = []
    for path in candidate_paths:
        rows.append(
            {
                "candidate_path": str(path.relative_to(project_root)),
                "exists": path.exists(),
                "status": "AVAILABLE" if path.exists() else "MISSING",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "matched_rna_training_audit.tsv", sep="\t", index=False)
    summary = {
        "matched_rna_training_status": "SKIPPED_INPUT_NOT_AVAILABLE",
        "reason": "No matched CPTAC COAD/PDAC RNA pathway matrix exists in current Stage 3 outputs; TCGA RNA is external-only.",
        "candidate_paths": rows,
    }
    return audit, summary


def run_external_associations(scores: pd.DataFrame, clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = scores.merge(
        clinical[
            [
                "cohort",
                "patient_id",
                "age",
                "sex",
                "stage",
                "advanced_at_presentation",
                "recurrence",
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
    for cohort, sub in df.groupby("cohort", sort=True):
        for model_name, predictors in [
            ("proxy_only", ["discordance_rna_proxy_z"]),
            ("clinic_plus_proxy", ["age_numeric_z", "sex_male", "discordance_rna_proxy_z"]),
        ]:
            usable = [p for p in predictors if p in sub.columns and sub[p].notna().sum() > 0 and sub[p].nunique(dropna=True) > 1]
            for fit_row in safe_logistic_fit(
                sub[usable].to_numpy(dtype=float),
                pd.to_numeric(sub["advanced_at_presentation"], errors="coerce").to_numpy(dtype=float),
                usable,
            ):
                fit_row.update({"endpoint": "advanced_at_presentation", "cohort": cohort, "model": model_name, "predictors": ",".join(usable)})
                rows.append(fit_row)
        for fit_row in safe_cox_fit(sub, ["discordance_rna_proxy_z"]):
            fit_row.update({"endpoint": "survival_time_to_event", "cohort": cohort, "model": "cox_proxy_only", "predictors": "discordance_rna_proxy_z"})
            rows.append(fit_row)
    results = pd.DataFrame(rows)
    if not results.empty and "p" in results.columns:
        results["q_within_endpoint"] = np.nan
        for endpoint, idx in results.groupby("endpoint").groups.items():
            results.loc[idx, "q_within_endpoint"] = bh_adjust(results.loc[idx, "p"].tolist())
    meta_logit = fixed_effect_meta(results[results["endpoint"] == "advanced_at_presentation"], "coef")
    meta_cox = fixed_effect_meta(results[results["endpoint"] == "survival_time_to_event"], "coef")
    meta = pd.concat([meta_logit, meta_cox], ignore_index=True) if not meta_logit.empty or not meta_cox.empty else pd.DataFrame()
    return results, meta


def build_wsi_transfer_audit(project_root: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    manifest_paths = sorted((project_root / "manifests/gdc").glob("TCGA-*_slide_images_manifest.tsv"))
    for path in manifest_paths:
        cohort = path.name.replace("_slide_images_manifest.tsv", "")
        manifest = pd.read_csv(path, sep="\t")
        expected_bytes = int(pd.to_numeric(manifest["size"], errors="coerce").fillna(0).sum()) if "size" in manifest.columns else 0
        download_dir = project_root / f"data/raw/gdc/{cohort}/slide_images"
        existing_files = list(download_dir.rglob("*")) if download_dir.exists() else []
        existing_files = [p for p in existing_files if p.is_file()]
        rows.append(
            {
                "cohort": cohort,
                "gdc_slide_manifest": str(path.relative_to(project_root)),
                "manifest_rows": int(len(manifest)),
                "expected_size_bytes": expected_bytes,
                "expected_size_tib": expected_bytes / 1024**4,
                "download_dir": str(download_dir.relative_to(project_root)),
                "download_dir_exists": download_dir.exists(),
                "downloaded_file_count": len(existing_files),
                "status": "SKIPPED_INPUT_NOT_DOWNLOADED",
                "reason": "TCGA slide manifests exist but TCGA WSI files and embeddings are not downloaded/extracted in current workspace.",
            }
        )
    idc_paths = sorted((project_root / "manifests/idc").glob("tcga_*_sm_s5cmd_manifest.txt"))
    for path in idc_paths:
        rows.append(
            {
                "cohort": path.stem.replace("_sm_s5cmd_manifest", "").upper().replace("_", "-"),
                "gdc_slide_manifest": "",
                "idc_s5cmd_manifest": str(path.relative_to(project_root)),
                "manifest_rows": int(sum(1 for _ in path.open())),
                "expected_size_bytes": np.nan,
                "expected_size_tib": np.nan,
                "download_dir": "",
                "download_dir_exists": False,
                "downloaded_file_count": 0,
                "status": "SKIPPED_INPUT_NOT_DOWNLOADED",
                "reason": "IDC TCGA SM manifests exist but TCGA DICOM WSI files and embeddings are not downloaded/extracted in current workspace.",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "tcga_wsi_transfer_input_audit.tsv", sep="\t", index=False)
    out = audit.copy()
    out["predicted_archetype_status"] = "SKIPPED"
    out["prediction_path"] = ""
    out.to_csv(output_dir / "tcga_wsi_predicted_archetypes.tsv", sep="\t", index=False)
    summary = {
        "wsi_transfer_status": "SKIPPED_INPUT_NOT_DOWNLOADED",
        "gdc_slide_manifest_count": int(sum(1 for r in rows if r.get("gdc_slide_manifest"))),
        "idc_slide_manifest_count": int(sum(1 for r in rows if r.get("idc_s5cmd_manifest", ""))),
        "gdc_expected_size_tib": float(np.nansum(audit["expected_size_tib"].to_numpy(dtype=float))) if not audit.empty else 0.0,
        "reason": "Stage 10 WSI transfer requires TCGA WSI download plus embedding extraction; current Stage 1 kept TCGA slides as manifest-only due disk/time constraints.",
    }
    return out, summary


def qc_outputs(
    *,
    summary: dict[str, Any],
    scores: pd.DataFrame,
    assoc: pd.DataFrame,
    meta: pd.DataFrame,
    wsi_audit: pd.DataFrame,
    output_dir: Path,
    project_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    checks: list[dict[str, Any]] = []

    def add(check: str, status: str, details: dict[str, Any]) -> None:
        checks.append({"check": check, "status": status, "details": details})

    stage9 = summary.get("stage9_gate", {})
    add("stage9_gate", "PASS" if stage9.get("stage9_status") == "PASS" else "FAIL", stage9)
    add("tcga_rna_scores", "PASS" if len(scores) >= 100 and scores["discordance_rna_proxy"].notna().all() else "FAIL", {"rows": int(len(scores)), "cohorts": scores["cohort"].value_counts().to_dict()})
    proxy_cols = [c for c in scores.columns if c.endswith("_rna_loading_proxy")]
    add("proxy_columns", "PASS" if proxy_cols else "FAIL", {"proxy_cols": proxy_cols})
    add("external_association_table", "PASS" if not assoc.empty else "FAIL", {"rows": int(len(assoc)), "pass_rows": int((assoc.get("status", pd.Series(dtype=str)) == "PASS").sum()) if not assoc.empty else 0})
    add("external_meta_table", "PASS" if not meta.empty else "WARN", {"rows": int(len(meta))})
    add("wsi_transfer_audit", "WARN", {"status": summary.get("wsi_transfer", {}).get("wsi_transfer_status"), "rows": int(len(wsi_audit))})
    add("matched_rna_training_audit", "WARN", {"status": summary.get("matched_rna_training", {}).get("matched_rna_training_status")})
    for name in [
        "tcga_rna_proxy_scores.tsv",
        "tcga_rna_proxy_signature.tsv",
        "external_validation_results.tsv",
        "external_validation_meta_analysis.tsv",
        "tcga_wsi_predicted_archetypes.tsv",
        "matched_rna_training_audit.tsv",
        "tcga_wsi_transfer_input_audit.tsv",
    ]:
        path = output_dir / name
        add(f"output_exists::{name}", "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL", {"path": str(path.relative_to(project_root)), "size": path.stat().st_size if path.exists() else 0})
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    warn_count = sum(1 for c in checks if c["status"] == "WARN")
    status = "PASS" if fail_count == 0 else "FAIL"
    qc = {
        "built_at": now_iso(),
        "status": status,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checks": checks,
        "notes": [
            "RNA proxy uses fixed overlap with Stage 7 protein-pathway positive loadings because matched CPTAC RNA is unavailable in current Stage 3 outputs.",
            "WSI transfer validation is not computed; TCGA slide manifests exist but WSI files and embeddings are not downloaded in the current workspace.",
            "This Stage 10 result is an external RNA proxy validation plus WSI-transfer input audit, not a completed TCGA WSI transfer analysis.",
        ],
    }
    qc_df = pd.DataFrame(
        [
            {"check": c["check"], "status": c["status"], "details": c["details"]}
            for c in checks
        ]
    )
    return qc, qc_df


def run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    ensure_dir(output_dir)
    ensure_dir(project_root / "logs/qc")

    stage9_gate = load_stage9_gate(project_root)
    if stage9_gate.get("stage9_status") != "PASS":
        raise RuntimeError(f"Stage 9 gate is not PASS: {stage9_gate}")

    rna = read_tsv(project_root / args.rna_pathway_matrix)
    clinical = read_tsv(project_root / args.clinical_endpoints)
    loadings = read_tsv(project_root / args.archetype_loadings)

    matched_audit, matched_summary = train_matched_rna_elastic_net_if_available(project_root=project_root, output_dir=output_dir)
    scores, weights, proxy_summary = build_fixed_loading_proxy(
        rna=rna,
        loadings=loadings,
        target_archetypes=args.target_archetypes.split(","),
        output_dir=output_dir,
    )
    scores.to_csv(output_dir / "tcga_rna_proxy_scores.tsv", sep="\t", index=False)

    assoc, meta = run_external_associations(scores, clinical)
    assoc.to_csv(output_dir / "external_validation_results.tsv", sep="\t", index=False)
    meta.to_csv(output_dir / "external_validation_meta_analysis.tsv", sep="\t", index=False)

    wsi_pred, wsi_summary = build_wsi_transfer_audit(project_root, output_dir)

    summary: dict[str, Any] = {
        "built_at": now_iso(),
        "status": "PASS",
        "method": "tcga_rna_fixed_stage7_loading_proxy_with_wsi_transfer_input_audit",
        "stage9_gate": stage9_gate,
        "inputs": {
            "rna_pathway_matrix": args.rna_pathway_matrix,
            "clinical_endpoints": args.clinical_endpoints,
            "archetype_loadings": args.archetype_loadings,
        },
        "rna_proxy": proxy_summary,
        "matched_rna_training": matched_summary,
        "wsi_transfer": wsi_summary,
        "outputs": {
            "tcga_rna_proxy_scores": str((output_dir / "tcga_rna_proxy_scores.tsv").relative_to(project_root)),
            "tcga_wsi_predicted_archetypes": str((output_dir / "tcga_wsi_predicted_archetypes.tsv").relative_to(project_root)),
            "external_validation_results": str((output_dir / "external_validation_results.tsv").relative_to(project_root)),
            "external_validation_meta_analysis": str((output_dir / "external_validation_meta_analysis.tsv").relative_to(project_root)),
        },
    }
    qc, qc_df = qc_outputs(
        summary=summary,
        scores=scores,
        assoc=assoc,
        meta=meta,
        wsi_audit=wsi_pred,
        output_dir=output_dir,
        project_root=project_root,
    )
    summary["qc"] = {"status": qc["status"], "fail_count": qc["fail_count"], "warn_count": qc["warn_count"]}
    (output_dir / "stage10_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (project_root / "logs/qc/stage10_external_validation_qc.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False) + "\n")
    qc_df.to_csv(project_root / "logs/qc/stage10_external_validation_qc_summary.tsv", sep="\t", index=False)
    print(json.dumps({"status": qc["status"], "fail_count": qc["fail_count"], "warn_count": qc["warn_count"], "tcga_rna_rows": int(len(scores))}, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--rna-pathway-matrix", default="data/processed/stage3/rna_pathway_matrix.tsv")
    parser.add_argument("--clinical-endpoints", default="data/processed/stage3/clinical_endpoints.tsv")
    parser.add_argument("--archetype-loadings", default="results/archetypes/archetype_loadings.tsv")
    parser.add_argument("--target-archetypes", default="A1,A2")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
