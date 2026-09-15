#!/usr/bin/env python3
"""Stage 10 TCGA RNA proxy external validation."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings


RANDOM_SEED = 20260609
META_COLUMNS = {"cohort", "patient_id"}
CBIO_BASE_URL = "https://www.cbioportal.org/api"
MATCHED_CPTAC_RNA_SOURCES = [
    {
        "cohort": "CPTAC-COAD",
        "study_id": "coad_cptac_2019",
        "profile_id": "coad_cptac_2019_rna_seq_v2_mrna",
        "sample_list_id": "coad_cptac_2019_rna_seq_mrna",
        "value_scale": "RNA Seq V2 RSEM UQ Log2",
    },
    {
        "cohort": "CPTAC-PDAC",
        "study_id": "paad_cptac_2021",
        "profile_id": "paad_cptac_2021_mrna",
        "sample_list_id": "paad_cptac_2021_all",
        "value_scale": "log2 RSEM-UQ",
    },
]


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


def log_progress(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def numeric_part(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.drop(columns=["cohort", "patient_id"]).apply(pd.to_numeric, errors="coerce")


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=1).div(std, axis=1).fillna(0.0)


def read_gmt_dict(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = [gene.upper() for gene in parts[2:] if gene]
    return gene_sets


def standardize_within_cohort(values: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, idx in metadata.groupby("cohort").groups.items():
        parts.append(zscore_columns(values.loc[idx]))
    return pd.concat(parts).loc[values.index]


def score_unweighted(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    prefix: str,
    min_size: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = matrix.copy()
    work.columns = [str(col).upper() for col in work.columns]
    work = work.groupby(level=0, axis=1).mean()
    z = standardize_within_cohort(work, metadata)
    rows: dict[str, pd.Series] = {}
    size_rows: list[dict[str, Any]] = []
    available = set(z.columns)
    for name, members in sorted(gene_sets.items()):
        overlap = sorted(set(g.upper() for g in members) & available)
        if len(overlap) < min_size:
            continue
        col = f"{prefix}__{name}"
        rows[col] = z[overlap].mean(axis=1)
        size_rows.append({"score": col, "resource": prefix, "genes_in_set": len(set(members)), "genes_used": len(overlap)})
    return pd.DataFrame(rows, index=matrix.index), size_rows


def score_weighted_progeny(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    progeny: pd.DataFrame,
    min_size: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = matrix.copy()
    work.columns = [str(col).upper() for col in work.columns]
    work = work.groupby(level=0, axis=1).mean()
    z = standardize_within_cohort(work, metadata)
    rows: dict[str, pd.Series] = {}
    size_rows: list[dict[str, Any]] = []
    available = set(z.columns)
    net = progeny.copy()
    net["target"] = net["target"].astype(str).str.upper()
    for pathway, group in net.groupby("source"):
        group = group[group["target"].isin(available)].drop_duplicates("target")
        if len(group) < min_size:
            continue
        weights = group.set_index("target")["weight"].astype(float)
        denom = float(weights.abs().sum())
        if denom <= 0:
            continue
        rows[f"PROGENY__{pathway}"] = z[weights.index].mul(weights, axis=1).sum(axis=1) / denom
        size_rows.append({"score": f"PROGENY__{pathway}", "resource": "PROGENY", "genes_in_set": len(group), "genes_used": len(group)})
    return pd.DataFrame(rows, index=matrix.index), size_rows


def build_pathway_matrix_from_expression(
    expression_matrix: pd.DataFrame,
    resource_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    progeny = pd.read_csv(resource_dir / "progeny_human_top500_decoupler.tsv", sep="\t")
    hallmark = read_gmt_dict(resource_dir / "hallmark_human_decoupler.gmt")
    reactome = read_gmt_dict(resource_dir / "reactome_pathways_2024_enrichr.gmt")
    metadata = expression_matrix[["cohort", "patient_id"]].copy()
    values = numeric_part(expression_matrix)
    values.index = metadata["cohort"].astype(str) + "|" + metadata["patient_id"].astype(str)
    metadata.index = values.index
    pscore, psize = score_weighted_progeny(values, metadata, progeny, min_size=5)
    hscore, hsize = score_unweighted(values, metadata, hallmark, "HALLMARK", min_size=5)
    rscore, rsize = score_unweighted(values, metadata, reactome, "REACTOME", min_size=10)
    scored = pd.concat([metadata, pscore, hscore, rscore], axis=1).reset_index(drop=True)
    sizes = pd.DataFrame(psize + hsize + rsize)
    return scored, sizes


def retry_request(method: str, url: str, *, session: requests.Session, **kwargs: Any) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            timeout = kwargs.pop("timeout", 60)
            response = session.request(method, url, timeout=timeout, **kwargs)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2 + attempt * 2)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 + attempt * 2)
    raise RuntimeError(f"request_failed:{url}:{type(last_exc).__name__}:{last_exc}")


def collect_pathway_resource_genes(resource_dir: Path) -> list[str]:
    genes: set[str] = set()
    progeny = pd.read_csv(resource_dir / "progeny_human_top500_decoupler.tsv", sep="\t")
    genes.update(progeny["target"].astype(str).str.upper())
    for gmt_name in ["hallmark_human_decoupler.gmt", "reactome_pathways_2024_enrichr.gmt"]:
        for members in read_gmt_dict(resource_dir / gmt_name).values():
            genes.update(str(gene).upper() for gene in members)
    return sorted(g for g in genes if g and g != "NAN")


def map_symbols_to_entrez(symbols: list[str], cache_path: Path, chunk_size: int = 1000) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = pd.read_csv(cache_path, sep="\t", dtype=str).fillna("")
    else:
        cached = pd.DataFrame(columns=["query", "symbol", "entrezgene", "name", "status"])
    done = set(cached["query"].astype(str))
    missing = [symbol for symbol in sorted(set(symbols)) if symbol not in done]
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    endpoint = "https://mygene.info/v3/query"
    for i in range(0, len(missing), chunk_size):
        chunk = missing[i : i + chunk_size]
        if not chunk:
            continue
        payload = {
            "q": ",".join(chunk),
            "scopes": "symbol",
            "fields": "symbol,entrezgene,name",
            "species": "human",
        }
        last_error = ""
        result: list[dict[str, Any]] = []
        for attempt in range(4):
            try:
                response = session.post(endpoint, data=payload, timeout=60)
                response.raise_for_status()
                parsed = response.json()
                result = parsed if isinstance(parsed, list) else []
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}:{exc}"
                time.sleep(2 + attempt * 2)
        by_query: dict[str, dict[str, Any]] = {}
        for item in result:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query", "")).upper()
            symbol = str(item.get("symbol", "")).upper() if item.get("symbol") else ""
            entrez = str(item.get("entrezgene", ""))
            if query and symbol == query and entrez:
                by_query[query] = item
        for query in chunk:
            item = by_query.get(query, {})
            rows.append(
                {
                    "query": query,
                    "symbol": str(item.get("symbol", "")).upper() if item.get("symbol") else "",
                    "entrezgene": str(item.get("entrezgene", "")),
                    "name": item.get("name", ""),
                    "status": "mapped" if item.get("entrezgene") else f"unmapped:{last_error}" if last_error else "unmapped",
                }
            )
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached = cached.drop_duplicates("query", keep="last").sort_values("query")
        cached.to_csv(cache_path, sep="\t", index=False)
    return cached


def fetch_cbio_sample_ids(session: requests.Session, sample_list_id: str) -> list[str]:
    response = retry_request("GET", f"{CBIO_BASE_URL}/sample-lists/{sample_list_id}/sample-ids", session=session, timeout=60)
    values = response.json()
    return [str(v) for v in values]


def fetch_cbio_mrna_for_source(
    *,
    source: dict[str, str],
    sample_ids: list[str],
    gene_map: pd.DataFrame,
    raw_dir: Path,
    chunk_size: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cohort = source["cohort"]
    profile_id = source["profile_id"]
    stem = cohort.lower().replace("-", "_")
    wide_path = raw_dir / f"{stem}_cbio_mrna_wide.tsv"
    if wide_path.exists() and wide_path.stat().st_size > 0:
        wide = pd.read_csv(wide_path, sep="\t", dtype={"patient_id": str})
        audit = pd.DataFrame(
            [
                {
                    "cohort": cohort,
                    "profile_id": profile_id,
                    "chunk_index": "cached",
                    "genes_requested": np.nan,
                    "samples_requested": len(sample_ids),
                    "records_returned": int(max(0, len(wide) * (wide.shape[1] - 2))),
                    "status": "CACHED",
                }
            ]
        )
        return wide, audit
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
    usable_map = gene_map[(gene_map["status"] == "mapped") & (gene_map["entrezgene"].astype(str).str.len() > 0)].copy()
    usable_map["entrezgene"] = usable_map["entrezgene"].astype(str)
    entrez_ids = sorted({int(x) for x in usable_map["entrezgene"].dropna().astype(str) if x.isdigit()})
    entrez_to_symbol = dict(zip(usable_map["entrezgene"].astype(int), usable_map["query"].astype(str)))
    all_rows: list[pd.DataFrame] = []
    fetch_rows: list[dict[str, Any]] = []
    for i in range(0, len(entrez_ids), chunk_size):
        chunk = entrez_ids[i : i + chunk_size]
        payload = {"sampleIds": sample_ids, "entrezGeneIds": chunk}
        response = retry_request(
            "POST",
            f"{CBIO_BASE_URL}/molecular-profiles/{profile_id}/molecular-data/fetch",
            session=session,
            json=payload,
            timeout=120,
        )
        data = response.json()
        block = pd.DataFrame(data)
        if not block.empty:
            all_rows.append(block)
        fetch_rows.append(
            {
                "cohort": cohort,
                "profile_id": profile_id,
                "chunk_index": i // chunk_size,
                "genes_requested": len(chunk),
                "samples_requested": len(sample_ids),
                "records_returned": int(len(block)),
                "status": "PASS",
            }
        )
    raw_long = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["sampleId", "patientId", "entrezGeneId", "value"])
    raw_long.to_csv(raw_dir / f"{stem}_cbio_mrna_long.tsv", sep="\t", index=False)
    if raw_long.empty:
        return pd.DataFrame(columns=["cohort", "patient_id"]), pd.DataFrame(fetch_rows)
    raw_long["gene_symbol"] = raw_long["entrezGeneId"].map(entrez_to_symbol)
    raw_long["value"] = pd.to_numeric(raw_long["value"], errors="coerce")
    raw_long = raw_long.dropna(subset=["gene_symbol", "value"])
    raw_long.to_csv(raw_dir / f"{stem}_cbio_mrna_long_with_symbols.tsv", sep="\t", index=False)
    wide = raw_long.pivot_table(index="patientId", columns="gene_symbol", values="value", aggfunc="mean")
    wide = wide.reindex(sample_ids)
    wide = wide.dropna(axis=1, how="all").fillna(wide.median(axis=0, skipna=True)).fillna(0.0)
    metadata = pd.DataFrame({"cohort": cohort, "patient_id": wide.index.astype(str).to_numpy()})
    wide = pd.concat([metadata.reset_index(drop=True), wide.reset_index(drop=True)], axis=1)
    wide.to_csv(wide_path, sep="\t", index=False)
    return wide, pd.DataFrame(fetch_rows)


def oof_elastic_net_predictions(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    folds: int = 5,
    random_seed: int = RANDOM_SEED,
    max_prefilter_features: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = list(x.columns)
    target_names = list(y.columns)
    n = len(x)
    n_splits = max(2, min(folds, n))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    pred = pd.DataFrame(index=x.index, columns=target_names, dtype=float)
    coef_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    alpha_grid = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    l1_grid = [0.1, 0.5, 0.9]
    for target in target_names:
        y_values = y[target].to_numpy(dtype=float)
        for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
            train_x = x.iloc[train_idx].copy()
            train_y = pd.Series(y_values[train_idx], index=train_x.index)
            corr = train_x.apply(lambda col: col.corr(train_y), axis=0).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            selected_features = corr.sort_values(ascending=False).head(min(max_prefilter_features, len(corr))).index.tolist()
            inner_cv = max(2, min(5, len(train_idx)))
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "enet",
                        ElasticNetCV(
                            l1_ratio=l1_grid,
                            alphas=alpha_grid,
                            cv=inner_cv,
                            max_iter=5000,
                            tol=1e-3,
                            random_state=random_seed + fold,
                            n_jobs=4,
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train_x[selected_features].to_numpy(dtype=float), y_values[train_idx])
            fold_pred = model.predict(x.iloc[test_idx][selected_features].to_numpy(dtype=float))
            pred.iloc[test_idx, pred.columns.get_loc(target)] = fold_pred
            enet = model.named_steps["enet"]
            coefs = pd.Series(enet.coef_, index=selected_features)
            selected = coefs[coefs.abs() > 1e-10].sort_values(key=lambda s: s.abs(), ascending=False)
            fold_rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "alpha": float(enet.alpha_),
                    "l1_ratio": float(enet.l1_ratio_),
                    "prefilter_features": int(len(selected_features)),
                    "selected_features": int(len(selected)),
                }
            )
            for rank, (feature, coef) in enumerate(selected.head(50).items(), start=1):
                coef_rows.append({"target": target, "fold": fold, "feature": feature, "coef": float(coef), "abs_rank": rank})
    metrics_rows: list[dict[str, Any]] = []
    for target in target_names:
        obs = y[target].to_numpy(dtype=float)
        pr = pred[target].to_numpy(dtype=float)
        ok = np.isfinite(obs) & np.isfinite(pr)
        if ok.sum() >= 3 and np.nanstd(obs[ok]) > 1e-12 and np.nanstd(pr[ok]) > 1e-12:
            pearson = stats.pearsonr(obs[ok], pr[ok])
            spearman = stats.spearmanr(obs[ok], pr[ok])
            rmse = float(np.sqrt(np.mean((obs[ok] - pr[ok]) ** 2)))
            metrics_rows.append(
                {
                    "target": target,
                    "n": int(ok.sum()),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_r": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "rmse": rmse,
                    "status": "PASS" if np.isfinite(pearson.statistic) else "WARN",
                }
            )
        else:
            metrics_rows.append({"target": target, "n": int(ok.sum()), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan, "rmse": np.nan, "status": "WARN"})
    return pred, pd.DataFrame(metrics_rows), pd.concat([pd.DataFrame(fold_rows), pd.DataFrame(coef_rows)], ignore_index=True, sort=False)


def bootstrap_elastic_net_selection(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    bootstraps: int,
    random_seed: int = RANDOM_SEED,
    max_prefilter_features: int = 80,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    alpha_grid = np.array([0.05, 0.1, 0.5, 1.0, 5.0])
    l1_grid = [0.1, 0.5, 0.9]
    for target in y.columns:
        y_values = y[target].to_numpy(dtype=float)
        for b in range(bootstraps):
            idx = rng.integers(0, len(x), size=len(x))
            boot_x = x.iloc[idx].copy()
            boot_y = pd.Series(y_values[idx], index=boot_x.index)
            corr = boot_x.apply(lambda col: col.corr(boot_y), axis=0).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            selected_features = corr.sort_values(ascending=False).head(min(max_prefilter_features, len(corr))).index.tolist()
            inner_cv = max(2, min(3, len(idx) - 1))
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "enet",
                        ElasticNetCV(
                            l1_ratio=l1_grid,
                            alphas=alpha_grid,
                            cv=inner_cv,
                            max_iter=5000,
                            tol=1e-3,
                            random_state=random_seed + b,
                            n_jobs=4,
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(boot_x[selected_features].to_numpy(dtype=float), y_values[idx])
            coefs = pd.Series(model.named_steps["enet"].coef_, index=selected_features)
            selected = coefs[coefs.abs() > 1e-10]
            for feature, coef in selected.items():
                rows.append({"target": target, "bootstrap": b + 1, "feature": feature, "coef": float(coef)})
    if not rows:
        return pd.DataFrame(columns=["target", "feature", "selected_bootstraps", "selection_frequency", "mean_coef", "mean_abs_coef"])
    coef_df = pd.DataFrame(rows)
    summary = (
        coef_df.groupby(["target", "feature"])
        .agg(selected_bootstraps=("bootstrap", "nunique"), mean_coef=("coef", "mean"), mean_abs_coef=("coef", lambda s: float(np.mean(np.abs(s)))))
        .reset_index()
    )
    summary["selection_frequency"] = summary["selected_bootstraps"] / float(bootstraps)
    summary = summary.sort_values(["target", "selection_frequency", "mean_abs_coef", "feature"], ascending=[True, False, False, True])
    return summary


def keyed_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cohort"] = out["cohort"].astype(str)
    out["patient_id"] = out["patient_id"].astype(str)
    out.index = out["cohort"] + "|" + out["patient_id"]
    return out


def safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return np.nan, np.nan
    if float(np.nanstd(a[ok])) <= 1e-12 or float(np.nanstd(b[ok])) <= 1e-12:
        return np.nan, np.nan
    res = stats.pearsonr(a[ok], b[ok])
    return float(res.statistic), float(res.pvalue)


def safe_wilcoxon_greater(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return np.nan
    try:
        return float(stats.wilcoxon(a[ok], b[ok], alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def summarize_head_to_head_group(
    *,
    label: str,
    feature_metrics: pd.DataFrame,
    sample_metrics: pd.DataFrame,
    morph_residual: pd.DataFrame,
    rna_residual: pd.DataFrame,
) -> dict[str, Any]:
    fm = feature_metrics.copy()
    sm = sample_metrics.copy()
    finite_corr = fm["morph_pearson"].notna() & fm["rna_pearson"].notna()
    finite_rmse = fm["morph_rmse"].notna() & fm["rna_rmse"].notna()
    flat_m = morph_residual.to_numpy(dtype=float).ravel()
    flat_r = rna_residual.to_numpy(dtype=float).ravel()
    matrix_corr, matrix_corr_p = safe_pearson(flat_m, flat_r)
    return {
        "group": label,
        "samples": int(len(sm)),
        "features": int(len(fm)),
        "median_morph_pearson": float(np.nanmedian(fm["morph_pearson"])) if len(fm) else np.nan,
        "median_rna_pearson": float(np.nanmedian(fm["rna_pearson"])) if len(fm) else np.nan,
        "median_rna_minus_morph_pearson": float(np.nanmedian(fm["rna_minus_morph_pearson"])) if len(fm) else np.nan,
        "fraction_features_rna_higher_pearson": float((fm.loc[finite_corr, "rna_pearson"] > fm.loc[finite_corr, "morph_pearson"]).mean()) if finite_corr.any() else np.nan,
        "median_morph_rmse": float(np.nanmedian(fm["morph_rmse"])) if len(fm) else np.nan,
        "median_rna_rmse": float(np.nanmedian(fm["rna_rmse"])) if len(fm) else np.nan,
        "median_morph_minus_rna_rmse": float(np.nanmedian(fm["morph_minus_rna_rmse"])) if len(fm) else np.nan,
        "fraction_features_rna_lower_rmse": float((fm.loc[finite_rmse, "rna_rmse"] < fm.loc[finite_rmse, "morph_rmse"]).mean()) if finite_rmse.any() else np.nan,
        "feature_rmse_wilcoxon_p_morph_gt_rna": safe_wilcoxon_greater(fm["morph_rmse"].to_numpy(dtype=float), fm["rna_rmse"].to_numpy(dtype=float)),
        "median_sample_morph_abs_residual": float(np.nanmedian(sm["mean_abs_morphology_protein_residual"])) if len(sm) else np.nan,
        "median_sample_rna_abs_residual": float(np.nanmedian(sm["mean_abs_rna_protein_residual"])) if len(sm) else np.nan,
        "sample_abs_residual_wilcoxon_p_morph_gt_rna": safe_wilcoxon_greater(sm["mean_abs_morphology_protein_residual"].to_numpy(dtype=float), sm["mean_abs_rna_protein_residual"].to_numpy(dtype=float)),
        "matrix_residual_pearson": matrix_corr,
        "matrix_residual_pearson_p": matrix_corr_p,
    }


def build_rna_protein_residual_head_to_head(
    *,
    project_root: Path,
    output_dir: Path,
    rna_pathway: pd.DataFrame,
    stage5_prediction_dir: str,
    folds: int = 5,
) -> dict[str, Any]:
    pred_dir = project_root / stage5_prediction_dir
    observed = keyed_frame(read_tsv(pred_dir / "observed_protein_pathway_aligned.tsv"))
    morph_pred = keyed_frame(read_tsv(pred_dir / "predicted_protein_pathway_oof.tsv"))
    rna = keyed_frame(rna_pathway)
    common_index = sorted(set(observed.index) & set(morph_pred.index) & set(rna.index))
    protein_features = [c for c in feature_columns(observed) if c in morph_pred.columns]
    x_features = [c for c in protein_features if c in rna.columns]
    if len(common_index) < 20 or len(protein_features) < 20 or len(x_features) < 20:
        raise RuntimeError(
            f"Insufficient matched RNA/protein inputs for direct head-to-head: "
            f"samples={len(common_index)} protein_features={len(protein_features)} x_features={len(x_features)}"
        )
    meta = observed.loc[common_index, ["cohort", "patient_id"]].reset_index(drop=True)
    y = observed.loc[common_index, protein_features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    morph_yhat = morph_pred.loc[common_index, protein_features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    x = rna.loc[common_index, x_features].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    rna_yhat = pd.DataFrame(index=y.index, columns=protein_features, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    alphas = np.logspace(-2, 3, 16)
    for cohort, cohort_idx in meta.groupby("cohort", sort=True).groups.items():
        positions = np.array(list(cohort_idx), dtype=int)
        n_splits = max(2, min(folds, len(positions)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        for fold, (train_local, test_local) in enumerate(cv.split(positions), start=1):
            train_pos = positions[train_local]
            test_pos = positions[test_local]
            model = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("ridge", RidgeCV(alphas=alphas)),
                ]
            )
            model.fit(x.iloc[train_pos].to_numpy(dtype=float), y.iloc[train_pos].to_numpy(dtype=float))
            pred = model.predict(x.iloc[test_pos].to_numpy(dtype=float))
            rna_yhat.iloc[test_pos, :] = pred
            alpha = model.named_steps["ridge"].alpha_
            fold_rows.append(
                {
                    "cohort": cohort,
                    "fold": fold,
                    "train_n": int(len(train_pos)),
                    "test_n": int(len(test_pos)),
                    "features_in": int(len(x_features)),
                    "targets_out": int(len(protein_features)),
                    "ridge_alpha": json.dumps(np.asarray(alpha).tolist()) if np.ndim(alpha) else float(alpha),
                }
            )
    morph_residual = y - morph_yhat
    rna_residual = y - rna_yhat
    rna_pred_out = pd.concat([meta, rna_yhat.reset_index(drop=True)], axis=1)
    morph_resid_out = pd.concat([meta, morph_residual.reset_index(drop=True)], axis=1)
    rna_resid_out = pd.concat([meta, rna_residual.reset_index(drop=True)], axis=1)
    rna_pred_out.to_csv(output_dir / "matched_rna_predicted_protein_pathway_oof.tsv", sep="\t", index=False)
    morph_resid_out.to_csv(output_dir / "matched_morphology_protein_residual_matrix.tsv", sep="\t", index=False)
    rna_resid_out.to_csv(output_dir / "matched_rna_protein_residual_matrix.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "matched_rna_protein_head_to_head_folds.tsv", sep="\t", index=False)

    feature_rows: list[dict[str, Any]] = []
    for cohort, idx in meta.groupby("cohort", sort=True).groups.items():
        idx_list = list(idx)
        for feature in protein_features:
            obs = y.loc[idx_list, feature].to_numpy(dtype=float)
            mp = morph_yhat.loc[idx_list, feature].to_numpy(dtype=float)
            rp = rna_yhat.loc[idx_list, feature].to_numpy(dtype=float)
            mr = morph_residual.loc[idx_list, feature].to_numpy(dtype=float)
            rr = rna_residual.loc[idx_list, feature].to_numpy(dtype=float)
            morph_corr, morph_p = safe_pearson(obs, mp)
            rna_corr, rna_p = safe_pearson(obs, rp)
            resid_corr, resid_p = safe_pearson(mr, rr)
            feature_rows.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "n": int(np.isfinite(obs).sum()),
                    "morph_pearson": morph_corr,
                    "morph_pearson_p": morph_p,
                    "rna_pearson": rna_corr,
                    "rna_pearson_p": rna_p,
                    "rna_minus_morph_pearson": rna_corr - morph_corr if np.isfinite(rna_corr) and np.isfinite(morph_corr) else np.nan,
                    "morph_rmse": float(np.sqrt(np.nanmean((obs - mp) ** 2))),
                    "rna_rmse": float(np.sqrt(np.nanmean((obs - rp) ** 2))),
                    "morph_minus_rna_rmse": float(np.sqrt(np.nanmean((obs - mp) ** 2)) - np.sqrt(np.nanmean((obs - rp) ** 2))),
                    "morph_mae": float(np.nanmean(np.abs(obs - mp))),
                    "rna_mae": float(np.nanmean(np.abs(obs - rp))),
                    "morph_minus_rna_mae": float(np.nanmean(np.abs(obs - mp)) - np.nanmean(np.abs(obs - rp))),
                    "residual_pearson_morph_vs_rna": resid_corr,
                    "residual_pearson_p": resid_p,
                }
            )
    feature_metrics = pd.DataFrame(feature_rows)
    feature_metrics.to_csv(output_dir / "matched_rna_protein_head_to_head_feature_metrics.tsv", sep="\t", index=False)

    sample_metrics = meta.copy()
    sample_metrics["mean_abs_morphology_protein_residual"] = morph_residual.abs().mean(axis=1).to_numpy(dtype=float)
    sample_metrics["mean_abs_rna_protein_residual"] = rna_residual.abs().mean(axis=1).to_numpy(dtype=float)
    sample_metrics["mean_abs_morph_minus_rna"] = sample_metrics["mean_abs_morphology_protein_residual"] - sample_metrics["mean_abs_rna_protein_residual"]
    sample_metrics["mean_sq_morphology_protein_residual"] = (morph_residual**2).mean(axis=1).to_numpy(dtype=float)
    sample_metrics["mean_sq_rna_protein_residual"] = (rna_residual**2).mean(axis=1).to_numpy(dtype=float)
    sample_metrics["mean_sq_morph_minus_rna"] = sample_metrics["mean_sq_morphology_protein_residual"] - sample_metrics["mean_sq_rna_protein_residual"]
    sample_metrics.to_csv(output_dir / "matched_rna_protein_head_to_head_sample_metrics.tsv", sep="\t", index=False)

    summary_rows: list[dict[str, Any]] = []
    for cohort, idx in meta.groupby("cohort", sort=True).groups.items():
        idx_list = list(idx)
        summary_rows.append(
            summarize_head_to_head_group(
                label=cohort,
                feature_metrics=feature_metrics[feature_metrics["cohort"] == cohort],
                sample_metrics=sample_metrics.iloc[idx_list],
                morph_residual=morph_residual.iloc[idx_list],
                rna_residual=rna_residual.iloc[idx_list],
            )
        )
    summary_rows.append(
        summarize_head_to_head_group(
            label="pooled",
            feature_metrics=feature_metrics,
            sample_metrics=sample_metrics,
            morph_residual=morph_residual,
            rna_residual=rna_residual,
        )
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "matched_rna_protein_head_to_head_summary.tsv", sep="\t", index=False)
    return {
        "status": "PASS",
        "method": "cohort_stratified_oof_RNA_pathway_to_observed_protein_pathway_RidgeCV_compared_with_final_HE_to_protein_OOF_residuals",
        "stage5_prediction_dir": stage5_prediction_dir,
        "matched_rows": int(len(meta)),
        "cohort_counts": meta["cohort"].value_counts().to_dict(),
        "rna_pathway_features_used": int(len(x_features)),
        "protein_pathway_targets": int(len(protein_features)),
        "folds": int(folds),
        "outputs": {
            "matched_rna_predicted_protein_pathway_oof": "results/validation/matched_rna_predicted_protein_pathway_oof.tsv",
            "matched_morphology_protein_residual_matrix": "results/validation/matched_morphology_protein_residual_matrix.tsv",
            "matched_rna_protein_residual_matrix": "results/validation/matched_rna_protein_residual_matrix.tsv",
            "matched_rna_protein_head_to_head_feature_metrics": "results/validation/matched_rna_protein_head_to_head_feature_metrics.tsv",
            "matched_rna_protein_head_to_head_sample_metrics": "results/validation/matched_rna_protein_head_to_head_sample_metrics.tsv",
            "matched_rna_protein_head_to_head_summary": "results/validation/matched_rna_protein_head_to_head_summary.tsv",
        },
        "summary": summary_rows,
    }


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
    bootstraps: int = 200,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_dir = project_root / "data/raw/cbio"
    processed_dir = project_root / "data/processed/stage3"
    resource_dir = project_root / "data/external/stage3_resources"
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    session = requests.Session()
    session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
    log_progress("Stage 10 matched RNA: collecting pathway genes and MyGene symbol mappings")
    pathway_genes = collect_pathway_resource_genes(resource_dir)
    gene_map = map_symbols_to_entrez(pathway_genes, resource_dir / "cptac_rna_symbol_to_entrez_mygene.tsv")
    mapped_genes = gene_map[(gene_map["status"] == "mapped") & (gene_map["entrezgene"].astype(str).str.len() > 0)]["query"].nunique()

    expression_parts = []
    fetch_audits = []
    source_rows = []
    for source in MATCHED_CPTAC_RNA_SOURCES:
        log_progress(f"Stage 10 matched RNA: fetching/caching cBioPortal mRNA for {source['cohort']}")
        sample_ids = fetch_cbio_sample_ids(session, source["sample_list_id"])
        wide, fetch_audit = fetch_cbio_mrna_for_source(source=source, sample_ids=sample_ids, gene_map=gene_map, raw_dir=raw_dir)
        expression_parts.append(wide)
        fetch_audits.append(fetch_audit)
        source_rows.append(
            {
                "cohort": source["cohort"],
                "study_id": source["study_id"],
                "profile_id": source["profile_id"],
                "sample_list_id": source["sample_list_id"],
                "value_scale": source["value_scale"],
                "sample_ids_from_cbio": len(sample_ids),
                "expression_rows_built": int(len(wide)),
                "expression_genes_built": int(max(0, wide.shape[1] - 2)),
                "status": "PASS" if len(wide) > 0 and wide.shape[1] > 2 else "FAIL",
            }
        )
    source_audit = pd.DataFrame(source_rows)
    source_audit.to_csv(output_dir / "matched_rna_source_audit.tsv", sep="\t", index=False)
    fetch_audit_df = pd.concat(fetch_audits, ignore_index=True) if fetch_audits else pd.DataFrame()
    fetch_audit_df.to_csv(raw_dir / "matched_cptac_cbio_mrna_fetch_audit.tsv", sep="\t", index=False)

    expression = pd.concat(expression_parts, axis=0, ignore_index=True) if expression_parts else pd.DataFrame(columns=["cohort", "patient_id"])
    expression = expression.drop_duplicates(["cohort", "patient_id"], keep="first")
    expression.to_csv(processed_dir / "cptac_rna_log_expression_matrix.tsv", sep="\t", index=False)
    if expression.empty or expression.shape[1] <= 2:
        audit = pd.DataFrame(
            [
                {
                    "component": "cBioPortal matched CPTAC mRNA",
                    "status": "FAIL",
                    "details": "No expression matrix could be built from cBioPortal.",
                }
            ]
        )
        audit.to_csv(output_dir / "matched_rna_training_audit.tsv", sep="\t", index=False)
        return audit, {"matched_rna_training_status": "FAILED_NO_EXPRESSION", "source_audit": source_rows}

    log_progress("Stage 10 matched RNA: scoring RNA pathways with Stage 3 resources")
    rna_pathway, pathway_sizes = build_pathway_matrix_from_expression(expression, resource_dir)
    rna_pathway.to_csv(processed_dir / "cptac_rna_pathway_matrix.tsv", sep="\t", index=False)
    rna_pathway.to_csv(processed_dir / "matched_rna_pathway_matrix.tsv", sep="\t", index=False)
    pathway_sizes.to_csv(processed_dir / "cptac_rna_score_feature_counts.tsv", sep="\t", index=False)

    scores = read_tsv(project_root / "results/archetypes/sample_archetype_scores.tsv")
    hidden = read_tsv(project_root / "results/clinical/hidden_aggressive_groups.tsv")
    hidden = hidden[hidden["cutoff"] == "top33"].drop_duplicates(["cohort", "patient_id"])
    target_cols = ["A1_score", "A2_score"]
    targets = scores[["cohort", "patient_id"] + target_cols].merge(
        hidden[["cohort", "patient_id", "discordance_risk", "hidden_aggressive_indicator"]],
        on=["cohort", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    for col in ["A1_score", "A2_score", "discordance_risk", "hidden_aggressive_indicator"]:
        targets[col] = pd.to_numeric(targets[col], errors="coerce")
    model_df = rna_pathway.merge(targets, on=["cohort", "patient_id"], how="inner", validate="one_to_one")
    pathway_cols = [c for c in feature_columns(rna_pathway) if model_df[c].notna().any()]
    common_protein_pathways = [c for c in pathway_cols if c in read_tsv(project_root / "data/processed/stage3/protein_pathway_matrix.tsv").columns]
    x_cols = common_protein_pathways if len(common_protein_pathways) >= 20 else pathway_cols
    model_targets = ["A1_score", "A2_score", "discordance_risk"]
    usable = model_df.dropna(subset=model_targets).copy()
    x = usable[x_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(axis=0, skipna=True)).fillna(0.0)
    y = usable[model_targets].apply(pd.to_numeric, errors="coerce")
    log_progress(f"Stage 10 matched RNA: fitting OOF ElasticNet comparator on {len(usable)} matched samples and {len(x_cols)} pathway features")
    pred, metrics, fold_coef = oof_elastic_net_predictions(x, y, folds=5, random_seed=RANDOM_SEED)
    pred_out = usable[["cohort", "patient_id"] + model_targets + ["hidden_aggressive_indicator"]].copy().reset_index(drop=True)
    for target in model_targets:
        pred_out[f"{target}_rna_oof_pred"] = pred[target].to_numpy(dtype=float)
    pred_out.to_csv(output_dir / "matched_rna_elastic_net_oof_predictions.tsv", sep="\t", index=False)

    metrics["q_within_matched_rna"] = bh_adjust(metrics["pearson_p"].tolist()) if not metrics.empty else []
    metrics.to_csv(output_dir / "matched_rna_elastic_net_metrics.tsv", sep="\t", index=False)
    fold_coef.to_csv(output_dir / "matched_rna_elastic_net_fold_coefficients.tsv", sep="\t", index=False)
    log_progress(f"Stage 10 matched RNA: running {bootstraps} bootstrap ElasticNet selections")
    stability = bootstrap_elastic_net_selection(x, y, bootstraps=bootstraps, random_seed=RANDOM_SEED)
    stability.to_csv(output_dir / "matched_rna_elastic_net_bootstrap_signature.tsv", sep="\t", index=False)

    assoc_rows: list[dict[str, Any]] = []
    if "discordance_risk_rna_oof_pred" in pred_out.columns:
        tmp = pred_out.copy()
        tmp = cohort_zscore(tmp, ["discordance_risk_rna_oof_pred"])
        for cohort, sub in tmp.groupby("cohort", sort=True):
            for fit_row in safe_logistic_fit(
                sub[["discordance_risk_rna_oof_pred_z"]].to_numpy(dtype=float),
                pd.to_numeric(sub["hidden_aggressive_indicator"], errors="coerce").to_numpy(dtype=float),
                ["discordance_risk_rna_oof_pred_z"],
            ):
                fit_row.update({"endpoint": "hidden_aggressive_top33", "cohort": cohort, "model": "matched_rna_oof_proxy"})
                assoc_rows.append(fit_row)
    assoc = pd.DataFrame(assoc_rows)
    assoc.to_csv(output_dir / "matched_rna_hidden_aggressive_association.tsv", sep="\t", index=False)

    audit_rows = [
        {
            "component": "cBioPortal matched CPTAC mRNA download",
            "status": "PASS" if int((source_audit["status"] == "PASS").sum()) == len(source_audit) else "WARN",
            "details": json.dumps(source_audit.to_dict("records"), ensure_ascii=False),
        },
        {
            "component": "MyGene symbol to Entrez mapping",
            "status": "PASS" if mapped_genes >= 1000 else "WARN",
            "details": json.dumps({"pathway_genes": len(pathway_genes), "mapped_genes": int(mapped_genes)}, ensure_ascii=False),
        },
        {
            "component": "matched RNA pathway matrix",
            "status": "PASS" if len(rna_pathway) >= 100 and len(feature_columns(rna_pathway)) >= 100 else "WARN",
            "details": json.dumps({"rows": int(len(rna_pathway)), "pathway_scores": int(len(feature_columns(rna_pathway)))}, ensure_ascii=False),
        },
        {
            "component": "matched RNA archetype comparator",
            "status": "PASS" if len(usable) >= 100 and not metrics.empty else "WARN",
            "details": json.dumps({"matched_rows": int(len(usable)), "features_used": int(len(x_cols)), "targets": model_targets}, ensure_ascii=False),
        },
    ]
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "matched_rna_training_audit.tsv", sep="\t", index=False)
    metric_records = metrics.to_dict("records")
    summary = {
        "matched_rna_training_status": "PASS" if not metrics.empty and len(usable) >= 100 else "WARN",
        "method": "cBioPortal_CPTAC_mRNA_pathways_to_stage7_archetype_scores_elastic_net_oof",
        "sources": source_rows,
        "mygene_mapping": {"pathway_genes": len(pathway_genes), "mapped_genes": int(mapped_genes)},
        "expression_matrix": {
            "path": "data/processed/stage3/cptac_rna_log_expression_matrix.tsv",
            "rows": int(len(expression)),
            "genes": int(expression.shape[1] - 2),
        },
        "pathway_matrix": {
            "path": "data/processed/stage3/cptac_rna_pathway_matrix.tsv",
            "rows": int(len(rna_pathway)),
            "scores": int(len(feature_columns(rna_pathway))),
        },
        "modeling": {
            "matched_rows": int(len(usable)),
            "features_used": int(len(x_cols)),
            "targets": model_targets,
            "folds": 5,
            "bootstraps": int(bootstraps),
            "metrics": metric_records,
        },
        "outputs": {
            "matched_rna_training_audit": "results/validation/matched_rna_training_audit.tsv",
            "matched_rna_source_audit": "results/validation/matched_rna_source_audit.tsv",
            "matched_rna_oof_predictions": "results/validation/matched_rna_elastic_net_oof_predictions.tsv",
            "matched_rna_metrics": "results/validation/matched_rna_elastic_net_metrics.tsv",
            "matched_rna_bootstrap_signature": "results/validation/matched_rna_elastic_net_bootstrap_signature.tsv",
        },
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
    matched_status = summary.get("matched_rna_training", {}).get("matched_rna_training_status")
    add("matched_rna_training_audit", "PASS" if matched_status == "PASS" else "WARN", {"status": matched_status, "matched_rows": summary.get("matched_rna_training", {}).get("modeling", {}).get("matched_rows")})
    direct_status = summary.get("rna_protein_residual_head_to_head", {}).get("status")
    add(
        "rna_protein_residual_head_to_head",
        "PASS" if direct_status == "PASS" else "WARN",
        {
            "status": direct_status,
            "matched_rows": summary.get("rna_protein_residual_head_to_head", {}).get("matched_rows"),
            "protein_pathway_targets": summary.get("rna_protein_residual_head_to_head", {}).get("protein_pathway_targets"),
        },
    )
    for name in [
        "tcga_rna_proxy_scores.tsv",
        "tcga_rna_proxy_signature.tsv",
        "external_validation_results.tsv",
        "external_validation_meta_analysis.tsv",
        "tcga_wsi_predicted_archetypes.tsv",
        "matched_rna_training_audit.tsv",
        "matched_rna_source_audit.tsv",
        "matched_rna_elastic_net_oof_predictions.tsv",
        "matched_rna_elastic_net_metrics.tsv",
        "matched_rna_elastic_net_bootstrap_signature.tsv",
        "matched_rna_predicted_protein_pathway_oof.tsv",
        "matched_morphology_protein_residual_matrix.tsv",
        "matched_rna_protein_residual_matrix.tsv",
        "matched_rna_protein_head_to_head_feature_metrics.tsv",
        "matched_rna_protein_head_to_head_sample_metrics.tsv",
        "matched_rna_protein_head_to_head_summary.tsv",
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
            "TCGA RNA proxy uses fixed overlap with Stage 7 protein-pathway positive loadings.",
            "Matched CPTAC RNA comparator uses cBioPortal CPTAC COAD/PDAC mRNA profiles, Stage 3 pathway scoring resources, and OOF ElasticNet models against Stage 7 archetype scores.",
            "Direct matched RNA-to-protein head-to-head uses cohort-stratified OOF RidgeCV from CPTAC RNA pathways to observed protein pathways, compared with final Stage 5 H&E-to-protein OOF residuals.",
            "WSI transfer validation is not computed; TCGA slide manifests exist but WSI files and embeddings are not downloaded in the current workspace.",
            "This Stage 10 result is an external RNA proxy validation, matched CPTAC RNA comparator, direct RNA-vs-morph protein residual head-to-head, and WSI-transfer input audit; it is not a completed TCGA WSI transfer analysis.",
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

    matched_audit, matched_summary = train_matched_rna_elastic_net_if_available(
        project_root=project_root,
        output_dir=output_dir,
        bootstraps=args.matched_rna_bootstraps,
    )
    rna_matched = read_tsv(project_root / "data/processed/stage3/cptac_rna_pathway_matrix.tsv")
    log_progress("Stage 10 matched RNA: running direct RNA-to-protein residual head-to-head")
    head_to_head_summary = build_rna_protein_residual_head_to_head(
        project_root=project_root,
        output_dir=output_dir,
        rna_pathway=rna_matched,
        stage5_prediction_dir=args.stage5_prediction_dir,
        folds=args.rna_protein_folds,
    )
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
        "method": "tcga_rna_fixed_stage7_loading_proxy_plus_matched_cptac_rna_comparator_with_wsi_transfer_input_audit",
        "stage9_gate": stage9_gate,
        "inputs": {
            "rna_pathway_matrix": args.rna_pathway_matrix,
            "clinical_endpoints": args.clinical_endpoints,
            "archetype_loadings": args.archetype_loadings,
            "stage5_prediction_dir": args.stage5_prediction_dir,
        },
        "rna_proxy": proxy_summary,
        "matched_rna_training": matched_summary,
        "rna_protein_residual_head_to_head": head_to_head_summary,
        "wsi_transfer": wsi_summary,
        "outputs": {
            "tcga_rna_proxy_scores": str((output_dir / "tcga_rna_proxy_scores.tsv").relative_to(project_root)),
            "tcga_wsi_predicted_archetypes": str((output_dir / "tcga_wsi_predicted_archetypes.tsv").relative_to(project_root)),
            "external_validation_results": str((output_dir / "external_validation_results.tsv").relative_to(project_root)),
            "external_validation_meta_analysis": str((output_dir / "external_validation_meta_analysis.tsv").relative_to(project_root)),
            "matched_rna_metrics": str((output_dir / "matched_rna_elastic_net_metrics.tsv").relative_to(project_root)),
            "matched_rna_oof_predictions": str((output_dir / "matched_rna_elastic_net_oof_predictions.tsv").relative_to(project_root)),
            "matched_rna_protein_head_to_head_summary": str((output_dir / "matched_rna_protein_head_to_head_summary.tsv").relative_to(project_root)),
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
    parser.add_argument("--matched-rna-bootstraps", type=int, default=200)
    parser.add_argument("--stage5-prediction-dir", default="results/prediction_prov_gigapath_exact_multislide_median_phospho_direct_sensitivity")
    parser.add_argument("--rna-protein-folds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
