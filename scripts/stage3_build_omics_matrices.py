#!/usr/bin/env python3
"""Build Stage 3 omics matrices and pathway/activity layers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import decoupler as dc
import gseapy as gp
import numpy as np
import pandas as pd
import requests
from sklearn.impute import KNNImputer


RANDOM_SEED = 20260609
MAIN_COHORTS = ["CPTAC-COAD", "CPTAC-PDAC"]
TCGA_COHORTS = ["TCGA-COAD", "TCGA-PAAD", "TCGA-STAD"]
PDC_MATRIX_PATHS = {
    ("CPTAC-COAD", "proteome"): "data/raw/pdc_quant_api/PDC000116/PDC000116_unshared_log2_ratio.quantDataMatrix.tsv",
    ("CPTAC-COAD", "phosphoproteome"): "data/raw/pdc_quant_api/PDC000117/PDC000117_log2_ratio.quantDataMatrix.tsv",
    ("CPTAC-PDAC", "proteome"): "data/raw/pdc_quant_api/PDC000270/PDC000270_unshared_log2_ratio.quantDataMatrix.tsv",
    ("CPTAC-PDAC", "phosphoproteome"): "data/raw/pdc_quant_api/PDC000271/PDC000271_log2_ratio.quantDataMatrix.tsv",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_tsv(path: Path, matrix: pd.DataFrame, metadata: pd.DataFrame) -> None:
    out = pd.concat([metadata.reset_index(drop=True), matrix.reset_index(drop=True)], axis=1)
    out.to_csv(path, sep="\t", index=False)


def robust_mad(frame: pd.DataFrame) -> pd.Series:
    med = frame.median(axis=0, skipna=True)
    return (frame.sub(med, axis=1)).abs().median(axis=0, skipna=True)


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0).replace(0, np.nan)
    z = frame.sub(mean, axis=1).div(std, axis=1)
    return z.fillna(0.0)


def knn_impute_and_zscore(frame: pd.DataFrame, neighbors: int = 10) -> tuple[pd.DataFrame, float]:
    imputed_fraction = float(frame.isna().mean().mean())
    n_neighbors = max(1, min(neighbors, len(frame) - 1))
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    arr = imputer.fit_transform(frame)
    imputed = pd.DataFrame(arr, index=frame.index, columns=frame.columns)
    return zscore_columns(imputed), imputed_fraction


def selected_patients(analysis_sets: pd.DataFrame, set_name: str, cohort: str) -> list[str]:
    sub = analysis_sets[(analysis_sets["set_name"] == set_name) & (analysis_sets["cohort"] == cohort)]
    return sorted(sub["patient_id"].astype(str).unique())


def load_modality_matrix(root: Path, sample_selection: pd.DataFrame, cohort: str, modality: str, patients: list[str]) -> pd.DataFrame:
    path = root / PDC_MATRIX_PATHS[(cohort, modality)]
    selected = sample_selection[
        (sample_selection["cohort"] == cohort)
        & (sample_selection["modality"] == modality)
        & (sample_selection["patient_id"].isin(patients))
        & (sample_selection["sample_type"] == "Primary Tumor")
    ].copy()
    selected = selected.drop_duplicates("patient_id")
    columns = selected["matrix_column"].tolist()
    rename = dict(zip(selected["matrix_column"], selected["patient_id"]))
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    feature_col = header[0]
    missing_columns = sorted(set(columns) - set(header))
    if missing_columns:
        raise RuntimeError(f"{cohort} {modality} matrix is missing {len(missing_columns)} selected columns; examples={missing_columns[:5]}")
    usecols = [feature_col] + columns
    raw = pd.read_csv(path, sep="\t", usecols=usecols, na_values=["NaN", "nan", "NA", ""], low_memory=False)
    raw = raw.set_index(feature_col)
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.groupby(raw.index).mean()
    raw = raw.rename(columns=rename)
    raw = raw[selected["patient_id"].tolist()]
    return raw.T


def build_common_feature_matrix(
    cohort_frames: dict[str, pd.DataFrame],
    min_fraction: float,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    stats_by_cohort: dict[str, pd.DataFrame] = {}
    passing_sets: list[set[str]] = []
    for cohort, frame in cohort_frames.items():
        observed_fraction = 1.0 - frame.isna().mean(axis=0)
        mad = robust_mad(frame)
        stats = pd.DataFrame({"observed_fraction": observed_fraction, "mad": mad})
        stats_by_cohort[cohort] = stats
        passing_sets.append(set(stats.index[stats["observed_fraction"] >= min_fraction]))
    common = set.intersection(*passing_sets)
    rows = []
    for feature in sorted(common):
        row: dict[str, Any] = {"feature_id": feature}
        mads = []
        for cohort, stats in stats_by_cohort.items():
            row[f"{cohort}_observed_fraction"] = float(stats.at[feature, "observed_fraction"])
            row[f"{cohort}_mad"] = float(stats.at[feature, "mad"])
            mads.append(float(stats.at[feature, "mad"]))
        row["mean_mad"] = float(np.nanmean(mads))
        rows.append(row)
    feature_stats = pd.DataFrame(rows).sort_values(["mean_mad", "feature_id"], ascending=[False, True])
    selected = feature_stats.head(max_features)["feature_id"].tolist()
    qc_rows: list[dict[str, Any]] = []
    processed_parts = []
    raw_parts = []
    for cohort, frame in cohort_frames.items():
        sub = frame[selected].copy()
        processed, imputed_fraction = knn_impute_and_zscore(sub)
        processed.insert(0, "patient_id", processed.index)
        processed.insert(0, "cohort", cohort)
        raw = sub.copy()
        raw.insert(0, "patient_id", raw.index)
        raw.insert(0, "cohort", cohort)
        processed_parts.append(processed)
        raw_parts.append(raw)
        qc_rows.append(
            {
                "matrix": "protein" if max_features <= 3000 else "phosphosite",
                "cohort": cohort,
                "samples": len(sub),
                "features_before_filter": frame.shape[1],
                "features_common_after_filter": len(common),
                "features_selected": len(selected),
                "min_observed_fraction": min_fraction,
                "imputed_fraction_selected": round(imputed_fraction, 6),
                "knn_neighbors": max(1, min(10, len(sub) - 1)),
            }
        )
    processed_matrix = pd.concat(processed_parts, axis=0, ignore_index=True)
    raw_matrix = pd.concat(raw_parts, axis=0, ignore_index=True)
    return processed_matrix, raw_matrix, qc_rows + feature_stats.to_dict("records")


def get_sample_metadata(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix[["cohort", "patient_id"]].copy()


def numeric_part(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.drop(columns=["cohort", "patient_id"]).apply(pd.to_numeric, errors="coerce")


def cache_gmt_from_dict(path: Path, gene_sets: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, genes in sorted(gene_sets.items()):
            clean = sorted({str(g).strip().upper() for g in genes if str(g).strip()})
            if clean:
                handle.write(name + "\tna\t" + "\t".join(clean) + "\n")


def read_gmt_dict(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = [gene.upper() for gene in parts[2:] if gene]
    return gene_sets


def fetch_resources(resource_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], list[dict[str, Any]]]:
    resource_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    progeny_path = resource_dir / "progeny_human_top500_decoupler.tsv"
    if progeny_path.exists():
        progeny = pd.read_csv(progeny_path, sep="\t")
        status = "cached"
    else:
        progeny = dc.op.progeny(organism="human", top=500, license="academic")
        progeny.to_csv(progeny_path, sep="\t", index=False)
        status = "downloaded"
    manifest.append({"resource": "PROGENy", "source": "decoupler.op.progeny(top=500, license=academic)", "path": progeny_path.as_posix(), "rows": len(progeny), "status": status})

    hallmark_path = resource_dir / "hallmark_human_decoupler.gmt"
    if hallmark_path.exists():
        hallmark = read_gmt_dict(hallmark_path)
        status = "cached"
    else:
        hnet = dc.op.hallmark(organism="human", license="academic")
        hallmark = {name: group["target"].astype(str).str.upper().tolist() for name, group in hnet.groupby("source")}
        cache_gmt_from_dict(hallmark_path, hallmark)
        status = "downloaded"
    manifest.append({"resource": "Hallmark", "source": "decoupler.op.hallmark(license=academic)", "path": hallmark_path.as_posix(), "rows": len(hallmark), "status": status})

    def fetch_gseapy_library(name: str, filename: str) -> tuple[dict[str, list[str]], str]:
        path = resource_dir / filename
        if path.exists():
            return read_gmt_dict(path), "cached"
        lib = gp.get_library(name=name, organism="Human")
        cache_gmt_from_dict(path, lib)
        return read_gmt_dict(path), "downloaded"

    reactome, status = fetch_gseapy_library("Reactome_Pathways_2024", "reactome_pathways_2024_enrichr.gmt")
    manifest.append({"resource": "Reactome_Pathways_2024", "source": "gseapy.get_library Enrichr", "path": (resource_dir / "reactome_pathways_2024_enrichr.gmt").as_posix(), "rows": len(reactome), "status": status})
    kinase, status = fetch_gseapy_library("The_Kinase_Library_2024", "the_kinase_library_2024_enrichr.gmt")
    manifest.append({"resource": "The_Kinase_Library_2024", "source": "gseapy.get_library Enrichr", "path": (resource_dir / "the_kinase_library_2024_enrichr.gmt").as_posix(), "rows": len(kinase), "status": status})
    silac, status = fetch_gseapy_library("SILAC_Phosphoproteomics", "silac_phosphoproteomics_enrichr.gmt")
    manifest.append({"resource": "SILAC_Phosphoproteomics", "source": "gseapy.get_library Enrichr", "path": (resource_dir / "silac_phosphoproteomics_enrichr.gmt").as_posix(), "rows": len(silac), "status": status})
    return progeny, hallmark, reactome, kinase, silac, manifest


def standardize_within_cohort(values: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cohort, idx in metadata.groupby("cohort").groups.items():
        part = values.loc[idx]
        parts.append(zscore_columns(part))
    return pd.concat(parts).loc[values.index]


def score_unweighted(matrix: pd.DataFrame, metadata: pd.DataFrame, gene_sets: dict[str, list[str]], prefix: str, min_size: int = 5, max_sets: int | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    genes = [str(col).upper() for col in matrix.columns]
    work = matrix.copy()
    work.columns = genes
    work = work.groupby(level=0, axis=1).mean()
    z = standardize_within_cohort(work, metadata)
    rows: dict[str, pd.Series] = {}
    size_rows: list[dict[str, Any]] = []
    items = sorted(gene_sets.items())
    if max_sets is not None:
        items = items[:max_sets]
    available = set(z.columns)
    for name, members in items:
        overlap = sorted(set(g.upper() for g in members) & available)
        if len(overlap) < min_size:
            continue
        score = z[overlap].mean(axis=1)
        col = f"{prefix}__{name}"
        rows[col] = score
        size_rows.append({"score": col, "resource": prefix, "genes_in_set": len(set(members)), "genes_used": len(overlap)})
    return pd.DataFrame(rows, index=matrix.index), size_rows


def score_weighted_progeny(matrix: pd.DataFrame, metadata: pd.DataFrame, progeny: pd.DataFrame, min_size: int = 5) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
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
        if denom == 0:
            continue
        rows[f"PROGENY__{pathway}"] = z[weights.index].mul(weights, axis=1).sum(axis=1) / denom
        size_rows.append({"score": f"PROGENY__{pathway}", "resource": "PROGENY", "genes_in_set": len(group), "genes_used": len(group)})
    return pd.DataFrame(rows, index=matrix.index), size_rows


def build_pathway_matrix(feature_matrix: pd.DataFrame, progeny: pd.DataFrame, hallmark: dict[str, list[str]], reactome: dict[str, list[str]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    metadata = get_sample_metadata(feature_matrix)
    values = numeric_part(feature_matrix)
    values.index = metadata["cohort"].astype(str) + "|" + metadata["patient_id"].astype(str)
    metadata.index = values.index
    pscore, psize = score_weighted_progeny(values, metadata, progeny, min_size=5)
    hscore, hsize = score_unweighted(values, metadata, hallmark, "HALLMARK", min_size=5)
    rscore, rsize = score_unweighted(values, metadata, reactome, "REACTOME", min_size=10)
    scored = pd.concat([metadata, pscore, hscore, rscore], axis=1).reset_index(drop=True)
    return scored, psize + hsize + rsize


def refseq_accession(feature_id: str) -> str:
    return feature_id.split(":", 1)[0].strip()


def map_refseq_to_gene(accessions: list[str], cache_path: Path, chunk_size: int = 1000) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = pd.read_csv(cache_path, sep="\t", dtype=str).fillna("")
    else:
        cached = pd.DataFrame(columns=["query", "symbol", "entrezgene", "name", "status"])
    done = set(cached["query"].astype(str))
    missing = [acc for acc in sorted(set(accessions)) if acc not in done]
    rows: list[dict[str, Any]] = []
    endpoint = "https://mygene.info/v3/query"
    for i in range(0, len(missing), chunk_size):
        chunk = missing[i : i + chunk_size]
        if not chunk:
            continue
        payload = {"q": ",".join(chunk), "scopes": "refseq.protein", "fields": "symbol,entrezgene,name", "species": "human"}
        last_error = ""
        result: list[dict[str, Any]] = []
        for attempt in range(3):
            try:
                response = requests.post(endpoint, data=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}:{exc}"
                time.sleep(2 + attempt)
        found = {str(item.get("query", "")): item for item in result if isinstance(item, dict)}
        for query in chunk:
            item = found.get(query, {})
            symbol = str(item.get("symbol", "")).upper() if item.get("symbol") else ""
            rows.append(
                {
                    "query": query,
                    "symbol": symbol,
                    "entrezgene": item.get("entrezgene", ""),
                    "name": item.get("name", ""),
                    "status": "mapped" if symbol else f"unmapped:{last_error}" if last_error else "unmapped",
                }
            )
    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached = cached.drop_duplicates("query", keep="last").sort_values("query")
        cached.to_csv(cache_path, sep="\t", index=False)
    qc = [
        {
            "resource": "MyGene.info refseq.protein mapping",
            "path": cache_path.as_posix(),
            "queries": len(set(accessions)),
            "mapped": int((cached["symbol"].astype(str).str.len() > 0).sum()),
            "mapping_fraction": round(float((cached["symbol"].astype(str).str.len() > 0).mean()), 4) if len(cached) else 0,
        }
    ]
    return cached, qc


def build_phospho_gene_matrix(phosphosite_matrix: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = get_sample_metadata(phosphosite_matrix)
    values = numeric_part(phosphosite_matrix)
    site_ids = values.columns.tolist()
    map_dict = dict(zip(mapping["query"], mapping["symbol"]))
    gene_cols = [map_dict.get(refseq_accession(site), "") for site in site_ids]
    keep = [bool(gene) for gene in gene_cols]
    kept_values = values.loc[:, keep].copy()
    kept_values.columns = [gene_cols[i] for i, flag in enumerate(keep) if flag]
    gene_matrix = kept_values.groupby(level=0, axis=1).mean()
    gene_matrix.insert(0, "patient_id", metadata["patient_id"].values)
    gene_matrix.insert(0, "cohort", metadata["cohort"].values)
    site_map = pd.DataFrame({"phosphosite_id": site_ids, "refseq_protein": [refseq_accession(site) for site in site_ids], "gene_symbol": gene_cols})
    return gene_matrix, site_map


def build_activity_matrix(phospho_gene_matrix: pd.DataFrame, kinase_sets: dict[str, list[str]], silac_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    metadata = get_sample_metadata(phospho_gene_matrix)
    values = numeric_part(phospho_gene_matrix)
    values.index = metadata["cohort"].astype(str) + "|" + metadata["patient_id"].astype(str)
    metadata.index = values.index
    kscore, ksize = score_unweighted(values, metadata, kinase_sets, "KINASE_LIBRARY_2024", min_size=5)
    pscore, psize = score_unweighted(values, metadata, silac_sets, "SILAC_PHOSPHO", min_size=5)
    scored = pd.concat([metadata, kscore, pscore], axis=1).reset_index(drop=True)
    return scored, ksize + psize


def locate_gdc_rna_files(root: Path, cohort: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    base = root / "data/raw/gdc" / cohort / "rna_star_counts"
    if not base.exists():
        return mapping
    for path in base.glob("*/*.rna_seq.augmented_star_gene_counts.tsv"):
        mapping[path.name] = path
    return mapping


def read_rna_file(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#", usecols=["gene_id", "gene_name", "gene_type", "tpm_unstranded"])
    df = df[(df["gene_type"] == "protein_coding") & df["gene_name"].notna() & (df["gene_name"].astype(str) != "")]
    df["gene_name"] = df["gene_name"].astype(str).str.upper()
    df["tpm_unstranded"] = pd.to_numeric(df["tpm_unstranded"], errors="coerce").fillna(0.0)
    return np.log2(df.groupby("gene_name")["tpm_unstranded"].mean() + 1.0)


def build_rna_matrix(root: Path, master: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    records = []
    path_maps = {cohort: locate_gdc_rna_files(root, cohort) for cohort in TCGA_COHORTS}
    external = master[(master["data_scope"] == "external_validation") & (master["rna_available"].astype(str).str.lower() == "true")].copy()
    external = external[external["sample_type"] == "Primary Tumor"]
    for _, row in external.iterrows():
        path = path_maps.get(row["cohort"], {}).get(row["rna_file_name"])
        records.append({"cohort": row["cohort"], "patient_id": row["patient_id"], "rna_file_name": row["rna_file_name"], "path": path})
    manifest = pd.DataFrame(records)
    missing = manifest["path"].isna().sum()
    ok = manifest.dropna(subset=["path"]).copy()
    rows: dict[str, pd.Series] = {}
    metadata_rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {pool.submit(read_rna_file, Path(row["path"])): row for _, row in ok.iterrows()}
        for future in as_completed(future_map):
            row = future_map[future]
            sample_id = f"{row['cohort']}|{row['patient_id']}"
            rows[sample_id] = future.result()
            metadata_rows.append({"cohort": row["cohort"], "patient_id": row["patient_id"], "sample_id": sample_id})
    expr = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    metadata = pd.DataFrame(metadata_rows).sort_values("sample_id").set_index("sample_id")
    expr = expr.loc[metadata.index]
    expr = expr.fillna(0.0)
    out = expr.copy()
    out.insert(0, "patient_id", metadata["patient_id"])
    out.insert(0, "cohort", metadata["cohort"])
    qc = [
        {
            "matrix": "rna_log_tpm",
            "manifest_rows": len(manifest),
            "files_found": len(ok),
            "files_missing": int(missing),
            "samples_built": len(expr),
            "genes": expr.shape[1],
            "workers": workers,
        }
    ]
    return out.reset_index(drop=True), manifest.drop(columns=["path"]).assign(path=manifest["path"].astype(str)), qc


def clinical_endpoints(master: pd.DataFrame, analysis_sets: pd.DataFrame) -> pd.DataFrame:
    set_memberships = analysis_sets.groupby(["cohort", "patient_id"])["set_name"].apply(lambda x: ",".join(sorted(set(x)))).reset_index()
    cols = [
        "cohort",
        "patient_id",
        "data_scope",
        "sample_type",
        "age",
        "sex",
        "stage",
        "T_stage",
        "N_stage",
        "M_stage",
        "grade",
        "recurrence",
        "survival_time",
        "survival_event",
        "advanced_at_presentation",
        "advanced_endpoint_source",
        "clinical_source",
    ]
    out = master[cols].merge(set_memberships, on=["cohort", "patient_id"], how="left")
    out["analysis_sets"] = out["set_name"].fillna("")
    return out.drop(columns=["set_name"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("STAGE3_WORKERS", "16")))
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out_dir = root / "data/processed/stage3"
    resource_dir = root / "data/external/stage3_resources"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    master = pd.read_csv(root / "data/processed/stage2/master_case_table.tsv", sep="\t", dtype=str).fillna("")
    analysis_sets = pd.read_csv(root / "data/processed/stage2/analysis_sets.tsv", sep="\t", dtype=str).fillna("")
    sample_selection = pd.read_csv(root / "data/processed/stage2/pdc_sample_selection.tsv", sep="\t", dtype=str).fillna("")

    protein_frames = {}
    phospho_frames = {}
    for cohort in MAIN_COHORTS:
        protein_patients = selected_patients(analysis_sets, "set_protein", cohort)
        phospho_patients = selected_patients(analysis_sets, "set_phospho", cohort)
        protein_frames[cohort] = load_modality_matrix(root, sample_selection, cohort, "proteome", protein_patients)
        phospho_frames[cohort] = load_modality_matrix(root, sample_selection, cohort, "phosphoproteome", phospho_patients)

    protein_matrix, protein_raw, protein_qc_and_features = build_common_feature_matrix(protein_frames, min_fraction=0.70, max_features=3000)
    phospho_matrix, phospho_raw, phospho_qc_and_features = build_common_feature_matrix(phospho_frames, min_fraction=0.50, max_features=5000)
    protein_matrix.to_csv(out_dir / "protein_matrix.tsv", sep="\t", index=False)
    protein_raw.to_csv(out_dir / "protein_matrix_raw_selected.tsv", sep="\t", index=False)
    phospho_matrix.to_csv(out_dir / "phosphosite_matrix.tsv", sep="\t", index=False)
    phospho_raw.to_csv(out_dir / "phosphosite_matrix_raw_selected.tsv", sep="\t", index=False)

    progeny, hallmark, reactome, kinase_sets, silac_sets, resource_manifest = fetch_resources(resource_dir)
    protein_pathway, protein_pathway_sizes = build_pathway_matrix(protein_matrix, progeny, hallmark, reactome)
    protein_pathway.to_csv(out_dir / "protein_pathway_matrix.tsv", sep="\t", index=False)

    accessions = [refseq_accession(col) for col in numeric_part(phospho_matrix).columns]
    mapping, mapping_qc = map_refseq_to_gene(accessions, resource_dir / "refseq_protein_to_gene_mygene.tsv")
    phospho_gene_matrix, site_map = build_phospho_gene_matrix(phospho_matrix, mapping)
    phospho_gene_matrix.to_csv(out_dir / "phospho_gene_matrix.tsv", sep="\t", index=False)
    site_map.to_csv(out_dir / "phosphosite_gene_map.tsv", sep="\t", index=False)
    kinase_ptm, kinase_ptm_sizes = build_activity_matrix(phospho_gene_matrix, kinase_sets, silac_sets)
    kinase_ptm.to_csv(out_dir / "kinase_ptm_activity.tsv", sep="\t", index=False)

    rna_matrix, rna_manifest, rna_qc = build_rna_matrix(root, master, args.workers)
    rna_matrix.to_parquet(out_dir / "rna_log_tpm_matrix.parquet", index=False)
    rna_manifest.to_csv(out_dir / "rna_file_manifest_stage3.tsv", sep="\t", index=False)
    rna_pathway, rna_pathway_sizes = build_pathway_matrix(rna_matrix, progeny, hallmark, reactome)
    rna_pathway.to_csv(out_dir / "rna_pathway_matrix.tsv", sep="\t", index=False)

    endpoints = clinical_endpoints(master, analysis_sets)
    endpoints.to_csv(out_dir / "clinical_endpoints.tsv", sep="\t", index=False)

    matrix_qc_rows = []
    for row in protein_qc_and_features:
        if "cohort" in row:
            matrix_qc_rows.append(row)
    for row in phospho_qc_and_features:
        if "cohort" in row:
            row = row.copy()
            row["matrix"] = "phosphosite"
            matrix_qc_rows.append(row)
    matrix_qc_rows.extend(rna_qc)
    matrix_qc_rows.extend(mapping_qc)
    write_tsv(out_dir / "stage3_matrix_qc.tsv", matrix_qc_rows)

    pd.DataFrame([row for row in protein_qc_and_features if "feature_id" in row]).to_csv(out_dir / "protein_feature_qc.tsv", sep="\t", index=False)
    pd.DataFrame([row for row in phospho_qc_and_features if "feature_id" in row]).to_csv(out_dir / "phosphosite_feature_qc.tsv", sep="\t", index=False)
    pd.DataFrame(resource_manifest).to_csv(out_dir / "stage3_resource_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(protein_pathway_sizes + kinase_ptm_sizes + rna_pathway_sizes).to_csv(out_dir / "stage3_score_feature_counts.tsv", sep="\t", index=False)

    summary = {
        "random_seed": RANDOM_SEED,
        "workers": args.workers,
        "protein_matrix": {"samples": int(len(protein_matrix)), "features": int(numeric_part(protein_matrix).shape[1])},
        "phosphosite_matrix": {"samples": int(len(phospho_matrix)), "features": int(numeric_part(phospho_matrix).shape[1])},
        "protein_pathway_matrix": {"samples": int(len(protein_pathway)), "scores": int(protein_pathway.shape[1] - 2)},
        "phospho_gene_matrix": {"samples": int(len(phospho_gene_matrix)), "genes": int(numeric_part(phospho_gene_matrix).shape[1])},
        "kinase_ptm_activity": {"samples": int(len(kinase_ptm)), "scores": int(kinase_ptm.shape[1] - 2)},
        "rna_log_tpm_matrix": {"samples": int(len(rna_matrix)), "genes": int(rna_matrix.shape[1] - 2)},
        "rna_pathway_matrix": {"samples": int(len(rna_pathway)), "scores": int(rna_pathway.shape[1] - 2)},
        "clinical_endpoints": {"rows": int(len(endpoints))},
        "resource_manifest": resource_manifest,
        "limitations": [
            "Exact site-level KSEAapp/PTM-SEA GMT resources were not fetched from GitHub raw because remote GitHub raw requests timed out during probing; kinase/PTM activity uses real Enrichr The_Kinase_Library_2024 and SILAC_Phosphoproteomics gene-set activity on RefSeq-mapped phosphosite gene means.",
            "Stage 3 main matrices are restricted to COAD+PDAC matched Primary Tumor samples from Stage 2; STAD omics is not promoted to pathology-linked modeling until STAD pathology is obtained.",
        ],
    }
    (out_dir / "stage3_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
