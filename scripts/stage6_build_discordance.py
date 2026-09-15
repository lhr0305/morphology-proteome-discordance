#!/usr/bin/env python3
"""Stage 6 discordance and blind-spot map construction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 20260609
ALPHAS = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], dtype=float)
META_COLUMNS = {"cohort", "patient_id"}


def feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...] | None = None) -> list[str]:
    excluded = {
        "cohort",
        "patient_id",
        "tile_count",
        "embedding_path",
        "model_repo",
        "encoder_slug",
        "encoder_role",
        "embedding_dim",
    }
    cols = [col for col in df.columns if col not in excluded]
    if prefixes:
        cols = [col for col in cols if col.startswith(prefixes)]
    return cols


def meta_key(df: pd.DataFrame) -> pd.DataFrame:
    return df[["cohort", "patient_id"]].astype({"cohort": str, "patient_id": str}).reset_index(drop=True)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"patient_id": str})


def write_matrix(path: Path, meta: pd.DataFrame, features: list[str], matrix: np.ndarray) -> None:
    out = pd.concat(
        [
            meta[["cohort", "patient_id"]].reset_index(drop=True),
            pd.DataFrame(matrix, columns=features),
        ],
        axis=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return out
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
    return out


def fit_predict_fold(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_test: np.ndarray,
    repeat: int,
    use_target_pca: bool,
    target_pca_max_components: int,
    target_pca_variance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_components = max(1, min(256, x_train.shape[1], x_train.shape[0] - 1))
    inner_splits = max(2, min(3, x_train.shape[0] // 10))
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
    y_train = y_scaler.fit_transform(y_train_raw)
    target_components = y_train.shape[1]
    target_variance_retained = 1.0
    if use_target_pca:
        max_target_components = max(1, min(target_pca_max_components, y_train.shape[1], x_train.shape[0] - 1))
        pilot_pca = PCA(n_components=max_target_components, random_state=seed + repeat + 2000)
        pilot_pca.fit(y_train)
        cumulative = np.cumsum(pilot_pca.explained_variance_ratio_)
        target_components = int(min(max_target_components, np.searchsorted(cumulative, target_pca_variance) + 1))
        y_pca = PCA(n_components=target_components, random_state=seed + repeat + 3000)
        y_model_train = y_pca.fit_transform(y_train)
        target_variance_retained = float(np.sum(y_pca.explained_variance_ratio_))
        model.fit(x_train, y_model_train)
        train_model = model.predict(x_train)
        test_model = model.predict(x_test)
        if train_model.ndim == 1:
            train_model = train_model.reshape(-1, 1)
            test_model = test_model.reshape(-1, 1)
        train_scaled = y_pca.inverse_transform(train_model)
        test_scaled = y_pca.inverse_transform(test_model)
    else:
        model.fit(x_train, y_train)
        train_scaled = model.predict(x_train)
        test_scaled = model.predict(x_test)
    train_pred = y_scaler.inverse_transform(train_scaled)
    test_pred = y_scaler.inverse_transform(test_scaled)
    train_residual = y_train_raw - train_pred
    train_sd = np.std(train_residual, axis=0, ddof=1)
    train_sd = np.where(np.isfinite(train_sd) & (train_sd > 1e-8), train_sd, np.nan)
    info = {
        "x_pca_components": int(n_components),
        "inner_cv_splits": int(inner_splits),
        "alpha": float(model.named_steps["ridge"].alpha_),
        "target_transform": "training_fold_pca" if use_target_pca else "direct_scaled_targets",
        "target_pca_components": int(target_components),
        "target_pca_variance_retained": float(target_variance_retained),
        "min_train_residual_sd": float(np.nanmin(train_sd)),
        "median_train_residual_sd": float(np.nanmedian(train_sd)),
    }
    return train_pred, test_pred, train_sd, info


def standardized_oof_residuals(
    *,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    saved_pred: np.ndarray,
    assignments: pd.DataFrame,
    target_set: str,
    target_pca_target_sets: set[str],
    folds: int,
    repeats: int,
    seed: int,
    target_pca_max_components: int,
    target_pca_variance: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    residual_sum = np.zeros_like(y, dtype=np.float64)
    pred_sum = np.zeros_like(y, dtype=np.float64)
    pred_count = np.zeros(y.shape[0], dtype=np.int32)
    meta_index = {
        (str(row.cohort), str(row.patient_id)): idx
        for idx, row in enumerate(meta[["cohort", "patient_id"]].itertuples(index=False))
    }
    rows: list[dict[str, Any]] = []
    use_target_pca = target_set in target_pca_target_sets
    for cohort in sorted(meta["cohort"].unique()):
        cohort_idx = np.flatnonzero(meta["cohort"].to_numpy() == cohort)
        cohort_keys = set((cohort, pid) for pid in meta.loc[cohort_idx, "patient_id"].astype(str))
        for repeat in range(repeats):
            for fold in range(folds):
                sub = assignments[
                    (assignments["target_set"] == target_set)
                    & (assignments["cohort"] == cohort)
                    & (assignments["repeat"] == repeat)
                    & (assignments["fold"] == fold)
                ]
                test_keys = [(str(row.cohort), str(row.patient_id)) for row in sub.itertuples(index=False)]
                test_idx = np.array([meta_index[key] for key in test_keys], dtype=int)
                test_key_set = set(test_keys)
                train_idx = np.array([meta_index[key] for key in cohort_keys if key not in test_key_set], dtype=int)
                train_idx.sort()
                test_idx.sort()
                if len(train_idx) + len(test_idx) != len(cohort_idx):
                    raise RuntimeError(f"Fold split size mismatch for {cohort}/{target_set}/repeat {repeat}/fold {fold}")
                _, pred, train_sd, info = fit_predict_fold(
                    x[train_idx],
                    y[train_idx],
                    x[test_idx],
                    repeat=repeat,
                    use_target_pca=use_target_pca,
                    target_pca_max_components=target_pca_max_components,
                    target_pca_variance=target_pca_variance,
                    seed=seed,
                )
                if np.isnan(train_sd).any():
                    raise RuntimeError(f"Non-positive train residual SD in {cohort}/{target_set}/repeat {repeat}/fold {fold}")
                pred_sum[test_idx] += pred
                pred_count[test_idx] += 1
                residual_sum[test_idx] += (y[test_idx] - pred) / train_sd
                rows.append(
                    {
                        "cohort": cohort,
                        "target_set": target_set,
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        **info,
                    }
                )
    if not np.all(pred_count == repeats):
        raise RuntimeError(f"OOF repeat count mismatch for {target_set}: {sorted(np.unique(pred_count).tolist())}")
    reconstructed_pred = pred_sum / pred_count[:, None]
    diff = reconstructed_pred - saved_pred
    diagnostics = {
        "target_set": target_set,
        "max_abs_prediction_diff_vs_stage5": float(np.nanmax(np.abs(diff))),
        "median_abs_prediction_diff_vs_stage5": float(np.nanmedian(np.abs(diff))),
        "standardized_residual_mean": float(np.nanmean(residual_sum / pred_count[:, None])),
        "standardized_residual_sd": float(np.nanstd(residual_sum / pred_count[:, None])),
    }
    return residual_sum / pred_count[:, None], pd.DataFrame(rows), diagnostics


def clinical_associations(
    residual_df: pd.DataFrame,
    metrics: pd.DataFrame,
    clinical: pd.DataFrame,
    target_set: str,
) -> pd.DataFrame:
    merged = residual_df.merge(
        clinical[["cohort", "patient_id", "advanced_at_presentation"]],
        on=["cohort", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    merged["advanced_at_presentation"] = pd.to_numeric(merged["advanced_at_presentation"], errors="coerce")
    feature_cols = [c for c in residual_df.columns if c not in META_COLUMNS]
    rows: list[dict[str, Any]] = []
    for cohort, sub in merged.groupby("cohort", sort=True):
        endpoint = sub["advanced_at_presentation"].to_numpy(dtype=float)
        for feature in feature_cols:
            values = sub[feature].to_numpy(dtype=float)
            ok = np.isfinite(values) & np.isfinite(endpoint)
            n = int(ok.sum())
            n_event = int(np.sum(endpoint[ok] == 1)) if n else 0
            n_nonevent = int(np.sum(endpoint[ok] == 0)) if n else 0
            p_value = np.nan
            point_biserial = np.nan
            mean_event = np.nan
            mean_nonevent = np.nan
            diff = np.nan
            if n >= 10 and n_event >= 3 and n_nonevent >= 3:
                event_values = values[ok & (endpoint == 1)]
                nonevent_values = values[ok & (endpoint == 0)]
                mean_event = float(np.mean(event_values))
                mean_nonevent = float(np.mean(nonevent_values))
                diff = mean_event - mean_nonevent
                point_biserial = float(np.corrcoef(values[ok], endpoint[ok])[0, 1])
                p_value = float(stats.ttest_ind(event_values, nonevent_values, equal_var=False, nan_policy="omit").pvalue)
            rows.append(
                {
                    "cohort": cohort,
                    "target_set": target_set,
                    "feature": feature,
                    "advanced_n": n,
                    "advanced_events": n_event,
                    "advanced_nonevents": n_nonevent,
                    "advanced_mean_event": mean_event,
                    "advanced_mean_nonevent": mean_nonevent,
                    "advanced_mean_diff_event_minus_nonevent": diff,
                    "advanced_point_biserial_r": point_biserial,
                    "advanced_ttest_p": p_value,
                }
            )
    assoc = pd.DataFrame(rows)
    assoc["advanced_ttest_fdr_bh"] = np.nan
    for (cohort, target), idx in assoc.groupby(["cohort", "target_set"]).groups.items():
        assoc.loc[idx, "advanced_ttest_fdr_bh"] = bh_adjust(assoc.loc[idx, "advanced_ttest_p"].to_numpy(dtype=float))
    out = assoc.merge(metrics, on=["cohort", "target_set", "feature"], how="left", validate="one_to_one")
    return out


def sample_scores(discordance: pd.DataFrame, blindspot: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in discordance.columns if c not in META_COLUMNS]
    protein_cols = [c for c in feature_cols if c.startswith("protein_pathway__")]
    phospho_cols = [c for c in feature_cols if c.startswith("phospho_kinase_ptm_exact__")]
    values = discordance[feature_cols]
    out = discordance[["cohort", "patient_id"]].copy()
    out["positive_discordance_burden_all"] = values.clip(lower=0).mean(axis=1)
    out["absolute_discordance_burden_all"] = values.abs().mean(axis=1)
    if protein_cols:
        out["positive_discordance_burden_protein"] = discordance[protein_cols].clip(lower=0).mean(axis=1)
        out["absolute_discordance_burden_protein"] = discordance[protein_cols].abs().mean(axis=1)
    if phospho_cols:
        out["positive_discordance_burden_phospho"] = discordance[phospho_cols].clip(lower=0).mean(axis=1)
        out["absolute_discordance_burden_phospho"] = discordance[phospho_cols].abs().mean(axis=1)
    adverse = blindspot[
        (blindspot["advanced_ttest_fdr_bh"] <= 0.10)
        & (blindspot["advanced_mean_diff_event_minus_nonevent"] > 0)
    ].copy()
    if adverse.empty:
        out["adverse_pathway_discordance_score"] = np.nan
        out["adverse_pathway_discordance_score_status"] = "no_stage6_adverse_features_at_fdr_0.10"
    else:
        cols = [f"{row.target_set}__{row.feature}" for row in adverse.itertuples(index=False)]
        cols = [c for c in cols if c in discordance.columns]
        if cols:
            out["adverse_pathway_discordance_score"] = discordance[cols].mean(axis=1)
            out["adverse_pathway_discordance_score_status"] = f"{len(cols)}_features"
        else:
            out["adverse_pathway_discordance_score"] = np.nan
            out["adverse_pathway_discordance_score_status"] = "no_matching_adverse_feature_columns"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", default="results/discordance")
    parser.add_argument("--embedding-table", required=True)
    parser.add_argument("--clinical-endpoints", default="data/processed/stage3/clinical_endpoints.tsv")
    parser.add_argument("--stage5-validation-json", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--prediction-diff-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    pred_dir = root / args.prediction_dir
    out_dir = root / args.output_dir
    qc_dir = root / "logs/qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, details: dict[str, Any] | None = None, severity: str = "FAIL") -> None:
        checks.append({"check": name, "status": "PASS" if ok else severity, "details": details or {}})

    validation = json.loads((root / args.stage5_validation_json).read_text(encoding="utf-8"))
    check(
        "stage5_validation_passed",
        validation.get("status") == "PASS" and validation.get("fail_count") == 0 and validation.get("warn_count") == 0,
        {
            "status": validation.get("status"),
            "fail_count": validation.get("fail_count"),
            "warn_count": validation.get("warn_count"),
            "check_count": validation.get("check_count"),
        },
    )
    build = json.loads((pred_dir / "stage5_prediction_build_summary.json").read_text(encoding="utf-8"))
    check("stage5_build_complete", build.get("status") == "COMPLETE", {"status": build.get("status")})
    check(
        "stage5_input_is_reviewed_exact_multislide_candidate",
        Path(args.prediction_dir).name == "prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity",
        {"prediction_dir": args.prediction_dir},
    )

    embedding_table = build.get("embedding_input_table")
    check(
        "embedding_table_matches_stage5_build",
        embedding_table == args.embedding_table,
        {"build_embedding_input_table": embedding_table, "stage6_embedding_table": args.embedding_table},
    )
    target_pca_target_sets = set(build.get("target_pca_target_sets", []))
    target_pca_max_components = int(build.get("target_pca_max_components", 30))
    target_pca_variance = float(build.get("target_pca_variance", 0.8))

    emb = read_table(root / args.embedding_table)
    meta = meta_key(emb)
    x_cols = feature_columns(emb, prefixes=("emb_",))
    x = emb[x_cols].to_numpy(dtype=np.float32)
    assignments = read_table(pred_dir / "stage5_oof_assignments.tsv")
    clinical = read_table(root / args.clinical_endpoints)
    metrics = read_table(pred_dir / "prediction_feature_metrics.tsv")

    target_specs = [
        {
            "target_set": "protein_pathway",
            "observed": pred_dir / "observed_protein_pathway_aligned.tsv",
            "predicted": pred_dir / "predicted_protein_pathway_oof.tsv",
            "residual_output": out_dir / "protein_residual_matrix.tsv",
        },
        {
            "target_set": "phospho_kinase_ptm_exact",
            "observed": pred_dir / "observed_phospho_kinase_ptm_aligned.tsv",
            "predicted": pred_dir / "predicted_phospho_kinase_ptm_oof.tsv",
            "residual_output": out_dir / "phospho_residual_matrix.tsv",
        },
    ]

    residual_tables: dict[str, pd.DataFrame] = {}
    fold_sd_tables = []
    diagnostics = []
    blindspot_parts = []
    for spec in target_specs:
        observed = read_table(spec["observed"])
        predicted = read_table(spec["predicted"])
        check(
            f"{spec['target_set']}::meta_alignment",
            meta.equals(meta_key(observed)) and meta.equals(meta_key(predicted)),
            {},
        )
        features = [c for c in observed.columns if c not in META_COLUMNS]
        y = observed[features].to_numpy(dtype=np.float32)
        saved_pred = predicted[features].to_numpy(dtype=np.float64)
        residual, fold_sd, diag = standardized_oof_residuals(
            x=x,
            y=y,
            meta=meta,
            saved_pred=saved_pred,
            assignments=assignments,
            target_set=spec["target_set"],
            target_pca_target_sets=target_pca_target_sets,
            folds=args.folds,
            repeats=args.repeats,
            seed=args.seed,
            target_pca_max_components=target_pca_max_components,
            target_pca_variance=target_pca_variance,
        )
        check(
            f"{spec['target_set']}::reconstructed_prediction_matches_stage5",
            diag["max_abs_prediction_diff_vs_stage5"] <= args.prediction_diff_tolerance,
            diag,
        )
        check(
            f"{spec['target_set']}::standardized_residuals_finite_nonzero",
            bool(np.isfinite(residual).all()) and float(np.std(residual)) > 0,
            {"std": float(np.std(residual)), "mean": float(np.mean(residual))},
        )
        write_matrix(spec["residual_output"], meta, features, residual)
        residual_df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(residual, columns=features)], axis=1)
        residual_tables[spec["target_set"]] = residual_df
        fold_sd_tables.append(fold_sd)
        diagnostics.append(diag)
        metric_subset = metrics[metrics["target_set"] == spec["target_set"]][
            ["cohort", "target_set", "feature", "pearson", "spearman", "r2"]
        ].copy()
        blindspot_parts.append(clinical_associations(residual_df, metric_subset, clinical, spec["target_set"]))

    combined = meta.copy()
    for target_set, table in residual_tables.items():
        feature_cols = [c for c in table.columns if c not in META_COLUMNS]
        renamed = table[feature_cols].rename(columns={c: f"{target_set}__{c}" for c in feature_cols})
        combined = pd.concat([combined, renamed.reset_index(drop=True)], axis=1)
    combined.to_csv(out_dir / "discordance_pathway_matrix.tsv", sep="\t", index=False)

    blindspot = pd.concat(blindspot_parts, ignore_index=True)
    blindspot["predictability_group_median_pearson"] = blindspot.groupby(["cohort", "target_set"])["pearson"].transform("median")
    blindspot["predictability_class"] = np.where(
        (blindspot["pearson"] > 0) & (blindspot["pearson"] >= blindspot["predictability_group_median_pearson"]),
        "higher_predictability",
        "lower_predictability",
    )
    blindspot["advanced_relevant_fdr10"] = (
        (blindspot["advanced_ttest_fdr_bh"] <= 0.10)
        & (blindspot["advanced_point_biserial_r"].abs() >= 0.15)
    )
    blindspot["stage6_candidate_class"] = "silent_or_unresolved"
    blindspot.loc[
        blindspot["advanced_relevant_fdr10"] & (blindspot["predictability_class"] == "lower_predictability"),
        "stage6_candidate_class",
    ] = "blind_spot_candidate"
    blindspot.loc[
        blindspot["advanced_relevant_fdr10"] & (blindspot["predictability_class"] == "higher_predictability"),
        "stage6_candidate_class",
    ] = "visible_risk_candidate"
    blindspot.to_csv(out_dir / "blindspot_pathway_table.tsv", sep="\t", index=False)

    scores = sample_scores(combined, blindspot)
    scores.to_csv(out_dir / "sample_discordance_scores.tsv", sep="\t", index=False)
    pd.concat(fold_sd_tables, ignore_index=True).to_csv(out_dir / "stage6_fold_residual_sd_diagnostics.tsv", sep="\t", index=False)

    output_files = {
        "protein_residual_matrix": (out_dir / "protein_residual_matrix.tsv").relative_to(root).as_posix(),
        "phospho_residual_matrix": (out_dir / "phospho_residual_matrix.tsv").relative_to(root).as_posix(),
        "discordance_pathway_matrix": (out_dir / "discordance_pathway_matrix.tsv").relative_to(root).as_posix(),
        "blindspot_pathway_table": (out_dir / "blindspot_pathway_table.tsv").relative_to(root).as_posix(),
        "sample_discordance_scores": (out_dir / "sample_discordance_scores.tsv").relative_to(root).as_posix(),
        "fold_residual_sd_diagnostics": (out_dir / "stage6_fold_residual_sd_diagnostics.tsv").relative_to(root).as_posix(),
    }
    for name, rel in output_files.items():
        path = root / rel
        check(f"output_exists::{name}", path.exists() and path.stat().st_size > 0, {"path": rel, "size": path.stat().st_size if path.exists() else 0})

    check(
        "discordance_shape",
        combined.shape == (236, 2 + 602 + 277),
        {"rows": combined.shape[0], "columns": combined.shape[1]},
    )
    check(
        "blindspot_table_shape",
        len(blindspot) == (602 + 277) * 2,
        {"rows": len(blindspot), "expected_rows": (602 + 277) * 2},
    )
    centering = []
    for target_set, table in residual_tables.items():
        features = [c for c in table.columns if c not in META_COLUMNS]
        for cohort, sub in table.groupby("cohort"):
            means = sub[features].mean(axis=0).abs()
            centering.append(
                {
                    "target_set": target_set,
                    "cohort": cohort,
                    "median_abs_feature_mean": float(means.median()),
                    "p95_abs_feature_mean": float(np.percentile(means, 95)),
                }
            )
    check(
        "residuals_approximately_centered",
        all(row["median_abs_feature_mean"] < 0.35 for row in centering),
        {"centering": centering},
        severity="WARN",
    )

    summary = {
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if not any(row["status"] == "FAIL" for row in checks) else "FAIL",
        "warn_count": sum(1 for row in checks if row["status"] == "WARN"),
        "fail_count": sum(1 for row in checks if row["status"] == "FAIL"),
        "prediction_dir": args.prediction_dir,
        "stage5_validation_json": args.stage5_validation_json,
        "target_sets": [spec["target_set"] for spec in target_specs],
        "samples": int(len(meta)),
        "cohort_counts": meta["cohort"].value_counts().sort_index().to_dict(),
        "standardization": "per-repeat held-out residual divided by the corresponding training-fold residual SD, then averaged over repeats",
        "prediction_reconstruction_diagnostics": diagnostics,
        "candidate_class_counts": blindspot["stage6_candidate_class"].value_counts().sort_index().to_dict(),
        "outputs": output_files,
        "checks": checks,
        "notes": [
            "Stage 6 uses only the fresh-reviewed Stage 5 exact multi-slide median phospho-direct candidate.",
            "Clinical relevance in blindspot_pathway_table is an initial advanced-at-presentation screen; Stage 8 remains the formal clinical modeling stage.",
            "Stage 5 null method remains posthoc label shuffle, not full retraining null, and should be disclosed downstream.",
        ],
    }
    (out_dir / "stage6_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    qc_json = qc_dir / "stage6_discordance_qc.json"
    qc_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(checks).assign(details=lambda df: df["details"].map(lambda x: json.dumps(x, ensure_ascii=False))).to_csv(
        qc_dir / "stage6_discordance_qc_summary.tsv",
        sep="\t",
        index=False,
    )
    print(json.dumps({k: summary[k] for k in ["status", "fail_count", "warn_count", "cohort_counts", "candidate_class_counts"]}, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
