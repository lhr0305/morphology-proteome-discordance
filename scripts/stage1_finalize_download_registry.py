#!/usr/bin/env python3
"""Add final Stage 1 GDC and IDC download entries to manifest registries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "dataset_name",
    "source_url",
    "download_method",
    "downloaded_at",
    "raw_path",
    "file_size",
    "md5_or_sha256",
    "expected_files",
    "actual_files",
    "status",
    "notes",
]

GDC_REQUIRED = [
    ("TCGA-COAD", "rna_star_counts"),
    ("TCGA-COAD", "clinical_supplement"),
    ("TCGA-COAD", "biospecimen_supplement"),
    ("TCGA-PAAD", "rna_star_counts"),
    ("TCGA-PAAD", "clinical_supplement"),
    ("TCGA-PAAD", "biospecimen_supplement"),
    ("TCGA-STAD", "rna_star_counts"),
    ("TCGA-STAD", "clinical_supplement"),
    ("TCGA-STAD", "biospecimen_supplement"),
]

IDC_REQUIRED = {
    "cptac_coad": {"resource": "SM", "expected_series": 372, "expected_patients": 178},
    "cptac_pda": {"resource": "SM", "expected_series": 557, "expected_patients": 168},
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def gdc_target_count(root: Path, cohort: str, kind: str) -> tuple[int, int]:
    manifest = root / f"manifests/gdc/{cohort}_{kind}_manifest.tsv"
    outdir = root / f"data/raw/gdc/{cohort}/{kind}"
    rows = read_tsv(manifest)
    present = 0
    for row in rows:
        target = outdir / row["id"] / row["filename"]
        if target.exists() and target.stat().st_size > 0:
            present += 1
    return len(rows), present


def idc_counts(root: Path, collection: str) -> tuple[int, int]:
    base = root / f"data/raw/idc_dicom/{collection}/{collection}"
    patients = {child.name for child in base.iterdir() if child.is_dir()} if base.exists() else set()
    series = set()
    if base.exists():
        for path in base.rglob("*.dcm"):
            series.add(str(path.parent))
    return len(patients), len(series)


def merge_rows(existing: list[dict[str, str]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {row.get("dataset_name", ""): dict(row) for row in existing if row.get("dataset_name")}
    for row in additions:
        merged[row["dataset_name"]] = {key: str(row.get(key, "")) for key in FIELDNAMES}
    return [merged[key] for key in sorted(merged)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    additions: list[dict[str, Any]] = []

    for cohort, kind in GDC_REQUIRED:
        manifest = root / f"manifests/gdc/{cohort}_{kind}_manifest.tsv"
        outdir = root / f"data/raw/gdc/{cohort}/{kind}"
        expected, actual = gdc_target_count(root, cohort, kind)
        additions.append(
            {
                "dataset_name": f"GDC:{cohort}:{kind}",
                "source_url": "https://api.gdc.cancer.gov/files",
                "download_method": "gdc-client 2.3 manifest download; selected low-concurrency retry for biospecimen gaps",
                "downloaded_at": now,
                "raw_path": str(outdir.relative_to(root)),
                "file_size": dir_bytes(outdir),
                "md5_or_sha256": f"manifest_sha256:{sha256_file(manifest)}",
                "expected_files": expected,
                "actual_files": actual,
                "status": "downloaded" if expected == actual and expected > 0 else "partial",
                "notes": f"manifest={manifest.relative_to(root)}; target file presence counted as manifest id/filename, excluding gdc-client parcel logs",
            }
        )

    for collection, meta in IDC_REQUIRED.items():
        manifest = root / f"manifests/idc/{collection}_sm_s5cmd_manifest.txt"
        outdir = root / f"data/raw/idc_dicom/{collection}"
        patients, series = idc_counts(root, collection)
        additions.append(
            {
                "dataset_name": f"IDC:{collection}:SM",
                "source_url": "s3://public-datasets-idc",
                "download_method": "idc-index download-from-manifest with s5cmd sync",
                "downloaded_at": now,
                "raw_path": str(outdir.relative_to(root)),
                "file_size": dir_bytes(outdir),
                "md5_or_sha256": f"manifest_sha256:{sha256_file(manifest)}",
                "expected_files": meta["expected_series"],
                "actual_files": series,
                "status": "downloaded" if series >= meta["expected_series"] and patients >= meta["expected_patients"] else "partial",
                "notes": f"patients={patients}; expected_patients={meta['expected_patients']}; manifest={manifest.relative_to(root)}",
            }
        )

    for rel in ["manifests/download_registry.tsv", "data/manifests/DATA_MANIFEST.tsv"]:
        path = root / rel
        write_tsv(path, merge_rows(read_tsv(path), additions))
    print(f"updated_registries\t{len(additions)}\t{now}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
