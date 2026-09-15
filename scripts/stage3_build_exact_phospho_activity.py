#!/usr/bin/env python3
"""Build site-specific KSEA/PTM-SEA-like phospho activity diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SITE_RE = re.compile(r"([sty])(\d+)", re.IGNORECASE)
RANDOM_SEED = 20260609


def numeric_part(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.drop(columns=["cohort", "patient_id"]).apply(pd.to_numeric, errors="coerce")


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=0)
    std = frame.std(axis=0, ddof=0).replace(0, np.nan)
    return frame.sub(mean, axis=1).div(std, axis=1).fillna(0.0)


def standardize_within_cohort(values: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, idx in metadata.groupby("cohort").groups.items():
        parts.append(zscore_columns(values.loc[idx]))
    return pd.concat(parts).loc[values.index]


def read_gmt(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out[parts[0]] = parts[2:]
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def submit_uniprot_mapping(ids: list[str]) -> str:
    payload = urllib.parse.urlencode(
        {"from": "RefSeq_Protein", "to": "UniProtKB", "ids": ",".join(ids)}
    ).encode()
    request = urllib.request.Request("https://rest.uniprot.org/idmapping/run", data=payload, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode())
    return str(data["jobId"])


def read_json_url(url: str, timeout: int = 60) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_uniprot_job(job_id: str, poll_seconds: int = 5, max_wait_seconds: int = 900) -> None:
    deadline = time.time() + max_wait_seconds
    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    while True:
        data = read_json_url(status_url)
        if "results" in data or data.get("jobStatus") == "FINISHED":
            return
        if data.get("jobStatus") in {"FAILED", "ERROR"}:
            raise RuntimeError(f"UniProt mapping job {job_id} failed: {data}")
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for UniProt mapping job {job_id}: {data}")
        time.sleep(poll_seconds)


def fetch_uniprot_results(job_id: str) -> tuple[list[dict[str, str]], list[str]]:
    results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}?size=500"
    mappings: list[dict[str, str]] = []
    failed: list[str] = []
    while results_url:
        data = read_json_url(results_url)
        for row in data.get("results", []):
            mappings.append({"from": str(row["from"]), "to": str(row["to"])})
        failed.extend(str(x) for x in data.get("failedIds", []))
        results_url = data.get("next")
    return mappings, failed


def load_or_build_refseq_uniprot_map(refseq_ids: list[str], cache_path: Path, chunk_size: int) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path, sep="\t")
        if set(refseq_ids).issubset(set(cached["refseq_acc"].astype(str))):
            return cached

    rows: list[dict[str, Any]] = []
    for start in range(0, len(refseq_ids), chunk_size):
        chunk = refseq_ids[start : start + chunk_size]
        job_id = submit_uniprot_mapping(chunk)
        wait_uniprot_job(job_id)
        mappings, failed = fetch_uniprot_results(job_id)
        for item in mappings:
            rows.append(
                {
                    "refseq_acc": item["from"],
                    "uniprot_acc": item["to"],
                    "uniprot_canonical_acc": item["to"].split("-")[0],
                    "status": "mapped",
                    "job_id": job_id,
                }
            )
        for refseq_acc in failed:
            rows.append(
                {
                    "refseq_acc": refseq_acc,
                    "uniprot_acc": "",
                    "uniprot_canonical_acc": "",
                    "status": "failed",
                    "job_id": job_id,
                }
            )
    mapped = pd.DataFrame(rows)
    present = set(mapped["refseq_acc"].astype(str))
    for missing in sorted(set(refseq_ids) - present):
        mapped.loc[len(mapped)] = {
            "refseq_acc": missing,
            "uniprot_acc": "",
            "uniprot_canonical_acc": "",
            "status": "not_returned",
            "job_id": "",
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(cache_path, sep="\t", index=False)
    return mapped


def parse_feature_sites(feature_id: str) -> tuple[str, list[str]]:
    refseq_acc, site_text = feature_id.split(":", 1)
    sites = [f"{res.upper()}{pos}" for res, pos in SITE_RE.findall(site_text)]
    return refseq_acc, sites


def build_site_matrix(phospho: pd.DataFrame, mapping: pd.DataFrame, out_map_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = numeric_part(phospho)
    metadata = phospho[["cohort", "patient_id"]].copy()
    refseq_to_uniprot: dict[str, set[str]] = {}
    for row in mapping.itertuples(index=False):
        if str(row.status) != "mapped":
            continue
        refseq_to_uniprot.setdefault(str(row.refseq_acc), set()).add(str(row.uniprot_acc))
        refseq_to_uniprot.setdefault(str(row.refseq_acc), set()).add(str(row.uniprot_canonical_acc))

    site_series: dict[str, list[pd.Series]] = {}
    map_rows: list[dict[str, Any]] = []
    for feature_id in values.columns:
        refseq_acc, sites = parse_feature_sites(feature_id)
        mapped = sorted(x for x in refseq_to_uniprot.get(refseq_acc, set()) if x and x != "nan")
        for site in sites:
            for uniprot_acc in mapped:
                key = f"{uniprot_acc};{site}-p"
                site_series.setdefault(key, []).append(values[feature_id])
                map_rows.append(
                    {
                        "feature_id": feature_id,
                        "refseq_acc": refseq_acc,
                        "site": site,
                        "uniprot_acc": uniprot_acc,
                        "site_key": key,
                    }
                )
        if not sites or not mapped:
            map_rows.append(
                {
                    "feature_id": feature_id,
                    "refseq_acc": refseq_acc,
                    "site": ";".join(sites),
                    "uniprot_acc": ";".join(mapped),
                    "site_key": "",
                }
            )
    site_matrix = pd.DataFrame(
        {key: pd.concat(series_list, axis=1).mean(axis=1) for key, series_list in site_series.items()},
        index=values.index,
    )
    site_matrix = standardize_within_cohort(site_matrix, metadata)
    site_matrix.insert(0, "patient_id", metadata["patient_id"].to_numpy())
    site_matrix.insert(0, "cohort", metadata["cohort"].to_numpy())
    site_map = pd.DataFrame(map_rows)
    site_map.to_csv(out_map_path, sep="\t", index=False)
    return site_matrix, site_map


def ksea_score(site_values: pd.DataFrame, metadata: pd.DataFrame, ksea: pd.DataFrame, min_sites: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ksea = ksea[(ksea["KIN_ORGANISM"].astype(str).str.lower() == "human") & (ksea["SUB_ORGANISM"].astype(str).str.lower() == "human")].copy()
    available = set(site_values.columns)
    rows: dict[str, pd.Series] = {}
    sizes: list[dict[str, Any]] = []
    for kinase_gene, group in ksea.groupby("GENE"):
        keys: set[str] = set()
        for row in group.itertuples(index=False):
            mod = str(getattr(row, "SUB_MOD_RSD", "")).upper()
            if not SITE_RE.match(mod):
                continue
            acc = str(getattr(row, "SUB_ACC_ID", ""))
            candidates = {f"{acc};{mod}-p", f"{acc.split('-')[0]};{mod}-p"}
            keys.update(key for key in candidates if key in available)
        if len(keys) < min_sites:
            continue
        col = f"KSEA__{kinase_gene}"
        rows[col] = site_values[sorted(keys)].mean(axis=1)
        sizes.append({"score": col, "resource": "KSEA_PSP_NetworKIN_2016", "sites_used": len(keys)})
    scored = pd.DataFrame(rows, index=site_values.index)
    scored = standardize_within_cohort(scored, metadata)
    return scored, sizes


def ptmsea_score(site_values: pd.DataFrame, metadata: pd.DataFrame, signatures: dict[str, list[str]], min_sites: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    available = set(site_values.columns)
    rows: dict[str, pd.Series] = {}
    sizes: list[dict[str, Any]] = []
    for name, members in sorted(signatures.items()):
        up_keys: set[str] = set()
        down_keys: set[str] = set()
        for member in members:
            parts = member.split(";")
            if len(parts) < 3:
                continue
            acc = parts[0]
            site = parts[1]
            direction = parts[2].lower()
            key = f"{acc};{site}"
            if key not in available:
                continue
            if direction.startswith("u"):
                up_keys.add(key)
            elif direction.startswith("d"):
                down_keys.add(key)
        used = sorted(up_keys | down_keys)
        if len(used) < min_sites:
            continue
        score = pd.Series(0.0, index=site_values.index)
        if up_keys:
            score = score + site_values[sorted(up_keys)].mean(axis=1)
        if down_keys:
            score = score - site_values[sorted(down_keys)].mean(axis=1)
        if up_keys and down_keys:
            score = score / 2.0
        col = f"PTMSEA__{name}"
        rows[col] = score
        sizes.append(
            {
                "score": col,
                "resource": "PTMsigDB_uniprot_human_v2.0.0",
                "sites_used": len(used),
                "up_sites_used": len(up_keys),
                "down_sites_used": len(down_keys),
            }
        )
    scored = pd.DataFrame(rows, index=site_values.index)
    scored = standardize_within_cohort(scored, metadata)
    return scored, sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--min-sites", type=int, default=5)
    parser.add_argument("--uniprot-chunk-size", type=int, default=400)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / "data/processed/stage3_exact"
    log_dir = root / "logs/qc"
    resource_dir = root / "data/external/stage3_resources"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    phospho = pd.read_csv(root / "data/processed/stage3/phosphosite_matrix.tsv", sep="\t")
    metadata = phospho[["cohort", "patient_id"]].copy()
    features = [col for col in phospho.columns if col not in {"cohort", "patient_id"}]
    refseq_ids = sorted({feature.split(":", 1)[0] for feature in features})

    mapping = load_or_build_refseq_uniprot_map(
        refseq_ids,
        resource_dir / "refseq_protein_to_uniprot.tsv",
        args.uniprot_chunk_size,
    )
    site_matrix, site_map = build_site_matrix(
        phospho,
        mapping,
        out_dir / "phosphosite_uniprot_site_map.tsv",
    )
    site_matrix.to_csv(out_dir / "phosphosite_uniprot_site_matrix.tsv", sep="\t", index=False)

    site_values = numeric_part(site_matrix)
    ksea = pd.read_csv(resource_dir / "PSP%26NetworKIN_Kinase_Substrate_Dataset_July2016.csv")
    ksea_activity, ksea_sizes = ksea_score(site_values, metadata, ksea, args.min_sites)
    ptm_signatures = read_gmt(resource_dir / "ptm.sig.db.all.uniprot.human.v2.0.0.gmt")
    ptmsea_activity, ptmsea_sizes = ptmsea_score(site_values, metadata, ptm_signatures, args.min_sites)

    combined = pd.concat([ksea_activity, ptmsea_activity], axis=1)
    combined.insert(0, "patient_id", metadata["patient_id"].to_numpy())
    combined.insert(0, "cohort", metadata["cohort"].to_numpy())
    ksea_out = pd.concat([metadata, ksea_activity], axis=1)
    ptm_out = pd.concat([metadata, ptmsea_activity], axis=1)
    ksea_out.to_csv(out_dir / "ksea_activity_exact.tsv", sep="\t", index=False)
    ptm_out.to_csv(out_dir / "ptmsea_activity_exact.tsv", sep="\t", index=False)
    combined.to_csv(out_dir / "kinase_ptm_activity_exact.tsv", sep="\t", index=False)
    write_tsv(out_dir / "exact_activity_score_feature_counts.tsv", ksea_sizes + ptmsea_sizes)

    mapped_features = site_map[site_map["site_key"].astype(str) != ""]["feature_id"].nunique()
    summary = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "random_seed": RANDOM_SEED,
        "min_sites": args.min_sites,
        "phosphosite_features": len(features),
        "unique_refseq": len(refseq_ids),
        "refseq_mapped_rows": int((mapping["status"] == "mapped").sum()),
        "refseq_unique_mapped": int(mapping.loc[mapping["status"] == "mapped", "refseq_acc"].nunique()),
        "phosphosite_features_with_uniprot_site": int(mapped_features),
        "uniprot_site_columns": int(site_values.shape[1]),
        "ksea_scores": int(ksea_activity.shape[1]),
        "ptmsea_scores": int(ptmsea_activity.shape[1]),
        "combined_scores": int(combined.shape[1] - 2),
        "cohort_counts": metadata["cohort"].value_counts().to_dict(),
        "source_resources": {
            "KSEA": "PSP&NetworKIN_Kinase_Substrate_Dataset_July2016.csv",
            "PTMsigDB": "ptm.sig.db.all.uniprot.human.v2.0.0.gmt",
        },
        "outputs": {
            "site_map": "data/processed/stage3_exact/phosphosite_uniprot_site_map.tsv",
            "site_matrix": "data/processed/stage3_exact/phosphosite_uniprot_site_matrix.tsv",
            "ksea_activity": "data/processed/stage3_exact/ksea_activity_exact.tsv",
            "ptmsea_activity": "data/processed/stage3_exact/ptmsea_activity_exact.tsv",
            "combined_activity": "data/processed/stage3_exact/kinase_ptm_activity_exact.tsv",
        },
    }
    (log_dir / "stage3_exact_phospho_activity_diagnostic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(log_dir / "stage3_exact_phospho_activity_diagnostic.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
