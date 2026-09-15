#!/usr/bin/env python3
"""Build Stage 2 patient-level master table and analysis sets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    import pydicom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pydicom is required in the project environment") from exc


GDC_API = "https://api.gdc.cancer.gov"
RANDOM_SEED = 20260609

CPTAC_COHORTS = {
    "CPTAC-COAD": {
        "short": "COAD",
        "collection": "cptac_coad",
        "proteome": "PDC000116",
        "phospho": "PDC000117",
    },
    "CPTAC-PDAC": {
        "short": "PDAC",
        "collection": "cptac_pda",
        "proteome": "PDC000270",
        "phospho": "PDC000271",
    },
    "CPTAC-STAD": {
        "short": "STAD",
        "collection": "",
        "proteome": "PDC000614",
        "phospho": "PDC000615",
    },
}

TCGA_PROJECTS = ["TCGA-COAD", "TCGA-PAAD", "TCGA-STAD"]

ANALYSIS_SET_DEFINITIONS = {
    "set_protein": "downloaded representative slide + primary-tumor proteome",
    "set_phospho": "downloaded representative slide + primary-tumor phosphoproteome",
    "set_triplet": "downloaded representative slide + primary-tumor proteome + primary-tumor phosphoproteome",
    "set_plus_rna": "set_triplet + matched open RNA when available",
    "external_rna": "TCGA external validation RNA case from Stage 1 GDC open RNA download",
}

GDC_CASE_FIELDS = [
    "case_id",
    "submitter_id",
    "project.project_id",
    "diagnoses.primary_diagnosis",
    "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_clinical_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.ajcc_clinical_t",
    "diagnoses.ajcc_clinical_n",
    "diagnoses.ajcc_clinical_m",
    "diagnoses.tumor_grade",
    "diagnoses.days_to_recurrence",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.last_known_disease_status",
    "demographic.gender",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.days_to_last_follow_up",
    "follow_ups.days_to_follow_up",
    "follow_ups.disease_response",
    "follow_ups.progression_or_recurrence",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_missing(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "not reported", "unknown", "not available", "not applicable"}:
        return ""
    return text


def clean_text(value: Any) -> str:
    text = norm_missing(value)
    return text if text else "Not Reported"


def normalize_table_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Make mixed metadata columns explicit before writing parquet."""
    out = df.copy()
    bool_columns = [
        "pathology_available",
        "proteome_available",
        "phosphoproteome_available",
        "rna_available",
    ]
    integer_columns = ["gdc_slide_manifest_count"]
    for column in bool_columns:
        if column in out:
            out[column] = out[column].fillna(False).astype(bool)
    for column in integer_columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("int64")
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else str(value))
    return out


def normalize_sample_type(value: Any) -> str:
    text = str(value or "").strip()
    low = text.lower()
    if "primary" in low and "tumor" in low:
        return "Primary Tumor"
    if "normal" in low:
        return "Solid Tissue Normal"
    if low in {"", "nan", "none", "not reported", "unknown"}:
        return "Not Reported"
    return text


def parse_matrix_columns(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n\r").split("\t")[1:]
    out: dict[str, str] = {}
    for column in header:
        aliquot_submitter = column.split(":", 1)[1] if ":" in column else column
        out[aliquot_submitter] = column
    return out


def matrix_missing_fraction(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, sep="\t", index_col=0, na_values=["NaN", "nan", "NA", ""])
    return df.isna().mean(axis=0).to_dict()


def pdc_matrix_path(root: Path, study_id: str) -> Path:
    matches = sorted((root / f"data/raw/pdc_quant_api/{study_id}").glob("*.quantDataMatrix.tsv"))
    if not matches:
        raise FileNotFoundError(f"No quantDataMatrix TSV for {study_id}")
    return matches[0]


def load_pdc_study(root: Path, study_id: str, modality: str) -> pd.DataFrame:
    clinical = pd.DataFrame(read_tsv(root / f"manifests/pdc/{study_id}_clinical_metadata.tsv"))
    biospecimen = pd.DataFrame(read_tsv(root / f"manifests/pdc/{study_id}_biospecimen_metadata.tsv"))
    merged = biospecimen.merge(clinical, on=["aliquot_id", "aliquot_submitter_id"], how="left")
    matrix_path = pdc_matrix_path(root, study_id)
    matrix_columns = parse_matrix_columns(matrix_path)
    missing_fraction = matrix_missing_fraction(matrix_path)
    merged["matrix_column"] = merged["aliquot_submitter_id"].map(matrix_columns)
    merged["matrix_available"] = merged["matrix_column"].notna()
    merged["matrix_missing_fraction"] = merged["matrix_column"].map(missing_fraction)
    merged["sample_type_normalized"] = merged["sample_type"].map(normalize_sample_type)
    merged["modality"] = modality
    merged["pdc_study_id"] = study_id
    merged["matrix_path"] = str(matrix_path.relative_to(root))
    return merged


def select_modality_sample(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if df.empty:
        return rows
    work = df[df["matrix_available"]].copy()
    if work.empty:
        return rows
    priority = {"Primary Tumor": 0, "Not Reported": 1, "Solid Tissue Normal": 2}
    work["sample_type_priority"] = work["sample_type_normalized"].map(priority).fillna(3)
    work["missing_sort"] = work["matrix_missing_fraction"].fillna(1.0)
    work = work.sort_values(
        ["case_submitter_id", "sample_type_priority", "missing_sort", "aliquot_submitter_id", "matrix_column"],
        kind="mergesort",
    )
    for patient_id, group in work.groupby("case_submitter_id", sort=True):
        selected = group.iloc[0].to_dict()
        rows[patient_id] = selected
    return rows


def list_series_files(series_dir: Path) -> tuple[int, int, Path | None]:
    files = [path for path in series_dir.glob("*.dcm") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    first = sorted(files)[0] if files else None
    return len(files), total, first


def dicom_metadata(first_file: Path | None) -> dict[str, Any]:
    if first_file is None:
        return {}
    try:
        ds = pydicom.dcmread(str(first_file), stop_before_pixels=True, force=True)
    except Exception as exc:
        return {"dicom_read_error": str(exc)[:300]}
    manufacturer = clean_text(getattr(ds, "Manufacturer", ""))
    model = clean_text(getattr(ds, "ManufacturerModelName", ""))
    objective = clean_text(getattr(ds, "ObjectiveLensPower", ""))
    scanner = " ".join(x for x in [manufacturer, model] if x and x != "Not Reported").strip()
    optical_path_description = ""
    try:
        seq = getattr(ds, "OpticalPathSequence", [])
        if seq:
            optical_path_description = clean_text(getattr(seq[0], "OpticalPathDescription", ""))
    except Exception:
        optical_path_description = ""
    return {
        "scanner": scanner or "Not Reported",
        "scanner_manufacturer": manufacturer,
        "scanner_model": model,
        "magnification": objective,
        "total_pixel_matrix_columns": getattr(ds, "TotalPixelMatrixColumns", ""),
        "total_pixel_matrix_rows": getattr(ds, "TotalPixelMatrixRows", ""),
        "imaged_volume_width": getattr(ds, "ImagedVolumeWidth", ""),
        "imaged_volume_height": getattr(ds, "ImagedVolumeHeight", ""),
        "optical_path_description": optical_path_description,
    }


def build_slide_table(root: Path) -> pd.DataFrame:
    manifest = pd.DataFrame(read_tsv(root / "manifests/idc/idc_sm_series_manifest.tsv"))
    manifest = manifest[manifest["collection_id"].isin(["cptac_coad", "cptac_pda"])].copy()
    manifest["series_size_MB"] = pd.to_numeric(manifest["series_size_MB"], errors="coerce")
    series_dirs: dict[tuple[str, str], Path] = {}
    for collection in ["cptac_coad", "cptac_pda"]:
        base = root / f"data/raw/idc_dicom/{collection}/{collection}"
        if not base.exists():
            continue
        for dcm in base.rglob("*.dcm"):
            series_dir = dcm.parent
            uid = series_dir.name[3:] if series_dir.name.startswith("SM_") else series_dir.name
            series_dirs.setdefault((collection, uid), series_dir)

    def build_one(row: dict[str, Any]) -> dict[str, Any]:
        collection = row["collection_id"]
        uid = row["SeriesInstanceUID"]
        series_dir = series_dirs.get((collection, uid))
        file_count = 0
        byte_count = 0
        first_file = None
        if series_dir is not None:
            file_count, byte_count, first_file = list_series_files(series_dir)
        meta = dicom_metadata(first_file)
        out = {
            "cohort": "CPTAC-COAD" if collection == "cptac_coad" else "CPTAC-PDAC",
            "collection_id": collection,
            "patient_id": row["PatientID"],
            "slide_id": uid,
            "study_instance_uid": row["StudyInstanceUID"],
            "series_instance_uid": uid,
            "slide_path": str(series_dir.relative_to(root)) if series_dir else "",
            "modality": row["Modality"],
            "sop_class_name": row["sop_class_name"],
            "instance_count_manifest": row["instanceCount"],
            "series_size_MB": row["series_size_MB"],
            "downloaded_file_count": file_count,
            "downloaded_bytes": byte_count,
            "he_eligible": True,
            "stain_status": "assumed_he_from_cptac_pathology_sm",
            "histoqc_status": "not_run_stage4",
            "usable_tissue_area_source": "stage2_pre_histoqc_series_size_MB_proxy",
            "usable_tissue_area_value": row["series_size_MB"],
        }
        out.update(meta)
        return out

    rows = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(build_one, row) for row in manifest.to_dict("records")]
        for future in as_completed(futures):
            rows.append(future.result())
    slide_df = pd.DataFrame(rows)
    slide_df = slide_df.sort_values(["cohort", "patient_id", "series_size_MB", "slide_id"], ascending=[True, True, False, True])
    slide_df["representative_rank_stage2"] = slide_df.groupby(["cohort", "patient_id"]).cumcount() + 1
    slide_df["is_representative_stage2"] = slide_df["representative_rank_stage2"] == 1
    slide_df["representative_rule_stage2"] = "H&E-eligible SM; pre-HistoQC largest series_size_MB proxy; tie-break lexicographic slide_id"
    return slide_df


def request_gdc(params: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            response = requests.get(f"{GDC_API}/cases", params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_gdc_cases_for_submitters(submitter_ids: list[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    fields = ",".join(GDC_CASE_FIELDS)
    ids = sorted(set(x for x in submitter_ids if x))
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        filters = {"op": "in", "content": {"field": "submitter_id", "value": batch}}
        payload = request_gdc({"filters": json.dumps(filters), "fields": fields, "format": "JSON", "size": str(len(batch))})
        hits.extend(payload.get("data", {}).get("hits", []))
    return hits


def fetch_gdc_cases_for_project(project_id: str) -> list[dict[str, Any]]:
    fields = ",".join(GDC_CASE_FIELDS)
    filters = {"op": "in", "content": {"field": "project.project_id", "value": [project_id]}}
    first = request_gdc({"filters": json.dumps(filters), "fields": fields, "format": "JSON", "size": "1"})
    total = first.get("data", {}).get("pagination", {}).get("total", 0)
    payload = request_gdc({"filters": json.dumps(filters), "fields": fields, "format": "JSON", "size": str(max(total, 1))})
    return payload.get("data", {}).get("hits", [])


def first_value(items: list[dict[str, Any]], keys: list[str]) -> Any:
    for item in items:
        for key in keys:
            value = item.get(key)
            if norm_missing(value):
                return value
    return None


def max_numeric(values: list[Any]) -> float | None:
    nums = []
    for value in values:
        try:
            if value is not None and str(value).strip() != "":
                nums.append(float(value))
        except ValueError:
            pass
    return max(nums) if nums else None


def flatten_gdc_case(case: dict[str, Any]) -> dict[str, Any]:
    diagnoses = case.get("diagnoses") or []
    follow_ups = case.get("follow_ups") or []
    demographic = case.get("demographic") or {}
    age_days = first_value(diagnoses, ["age_at_diagnosis"])
    try:
        age_years = round(float(age_days) / 365.25, 2) if age_days not in [None, ""] else ""
    except ValueError:
        age_years = ""
    vital = clean_text(demographic.get("vital_status"))
    death_days = demographic.get("days_to_death")
    follow_days = [demographic.get("days_to_last_follow_up")]
    follow_days += [d.get("days_to_last_follow_up") for d in diagnoses]
    follow_days += [f.get("days_to_follow_up") for f in follow_ups]
    max_follow = max_numeric(follow_days)
    survival_time = death_days if death_days not in [None, ""] else max_follow
    survival_event = "1" if str(vital).lower() == "dead" else ("0" if vital != "Not Reported" else "")
    recurrence_days = first_value(diagnoses, ["days_to_recurrence"])
    recurrence_terms = [str(f.get("progression_or_recurrence", "")).lower() for f in follow_ups]
    recurrence = ""
    if recurrence_days not in [None, ""]:
        recurrence = "Yes"
    elif any(x in {"yes", "progression", "recurrence"} for x in recurrence_terms):
        recurrence = "Yes"
    elif recurrence_terms:
        recurrence = "Not Reported"
    stage = clean_text(first_value(diagnoses, ["ajcc_pathologic_stage", "ajcc_clinical_stage"]))
    n_stage = clean_text(first_value(diagnoses, ["ajcc_pathologic_n", "ajcc_clinical_n"]))
    m_stage = clean_text(first_value(diagnoses, ["ajcc_pathologic_m", "ajcc_clinical_m"]))
    return {
        "gdc_case_id": case.get("case_id") or case.get("id", ""),
        "patient_id": case.get("submitter_id", ""),
        "gdc_project_id": (case.get("project") or {}).get("project_id", ""),
        "age": age_years,
        "sex": clean_text(demographic.get("gender")),
        "stage": stage,
        "T_stage": clean_text(first_value(diagnoses, ["ajcc_pathologic_t", "ajcc_clinical_t"])),
        "N_stage": n_stage,
        "M_stage": m_stage,
        "grade": clean_text(first_value(diagnoses, ["tumor_grade"])),
        "primary_diagnosis": clean_text(first_value(diagnoses, ["primary_diagnosis"])),
        "recurrence": recurrence or "Not Reported",
        "days_to_recurrence": recurrence_days if recurrence_days not in [None, ""] else "",
        "survival_time": survival_time if survival_time not in [None, ""] else "",
        "survival_event": survival_event,
        "vital_status": vital,
        "clinical_source": "GDC cases API",
    }


def stage_advanced(stage: str) -> bool | None:
    text = norm_missing(stage).upper().replace("STAGE", "").strip()
    if not text:
        return None
    if text.startswith("III") or text.startswith("IV"):
        return True
    if text.startswith("0") or text.startswith("I") or text.startswith("II"):
        return False
    return None


def n_positive(n_stage: str) -> bool | None:
    text = norm_missing(n_stage).upper()
    if not text:
        return None
    if text.startswith("N0"):
        return False
    if text.startswith("N") and any(ch.isdigit() and ch != "0" for ch in text):
        return True
    return None


def m_positive(m_stage: str) -> bool | None:
    text = norm_missing(m_stage).upper()
    if not text:
        return None
    if text.startswith("M0"):
        return False
    if text.startswith("M1"):
        return True
    return None


def endpoint_from_row(row: dict[str, Any]) -> tuple[str, str]:
    signals = [stage_advanced(str(row.get("stage", ""))), n_positive(str(row.get("N_stage", ""))), m_positive(str(row.get("M_stage", "")))]
    known = [x for x in signals if x is not None]
    if not known:
        return "", "missing_stage_n_m"
    return ("1" if any(known) else "0"), "AJCC stage III/IV OR N+ OR M+"


def build_rna_maps(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rna: dict[str, dict[str, Any]] = {}
    slide_manifest_counts: dict[str, int] = {}
    for project in TCGA_PROJECTS:
        rows = read_tsv(root / f"manifests/gdc/{project}_rna_star_counts_manifest.tsv")
        work = []
        for row in rows:
            patient_ids = [x for x in row.get("case_submitter_ids", "").split(",") if x]
            for patient_id in patient_ids:
                sample_type = normalize_sample_type(row.get("sample_types", ""))
                work.append((patient_id, sample_type, row))
        for patient_id, sample_type, row in sorted(work, key=lambda x: (x[0], 0 if x[1] == "Primary Tumor" else 1, x[2]["id"])):
            key = f"{project}:{patient_id}"
            rna.setdefault(
                key,
                {
                    "rna_sample_id": row["id"],
                    "rna_file_name": row["filename"],
                    "rna_sample_type": sample_type,
                    "rna_manifest": f"manifests/gdc/{project}_rna_star_counts_manifest.tsv",
                },
            )
        slide_rows = read_tsv(root / f"manifests/gdc/{project}_slide_images_manifest.tsv")
        for row in slide_rows:
            for patient_id in [x for x in row.get("case_submitter_ids", "").split(",") if x]:
                slide_manifest_counts[f"{project}:{patient_id}"] = slide_manifest_counts.get(f"{project}:{patient_id}", 0) + 1
    return rna, slide_manifest_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    out_dir = root / "data/processed/stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / "logs/preprocessing").mkdir(parents=True, exist_ok=True)

    slide_df = build_slide_table(root)
    slide_df.to_csv(out_dir / "slide_series_table.tsv", sep="\t", index=False)
    representative = slide_df[slide_df["is_representative_stage2"]].copy()
    slide_by_key = {(row["cohort"], row["patient_id"]): row for row in representative.to_dict("records")}
    slide_df.to_csv(out_dir / "multi_slide_sensitivity_table.tsv", sep="\t", index=False)

    pdc_selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    pdc_all_patients: dict[str, set[str]] = {cohort: set() for cohort in CPTAC_COHORTS}
    pdc_selection_rows = []
    for cohort, meta in CPTAC_COHORTS.items():
        for modality, study_id in [("proteome", meta["proteome"]), ("phosphoproteome", meta["phospho"])]:
            study_df = load_pdc_study(root, study_id, modality)
            pdc_all_patients[cohort].update(study_df["case_submitter_id"].dropna().astype(str).tolist())
            selected = select_modality_sample(study_df)
            for patient_id, row in selected.items():
                pdc_selected[(cohort, patient_id, modality)] = row
                pdc_selection_rows.append(
                    {
                        "cohort": cohort,
                        "patient_id": patient_id,
                        "modality": modality,
                        "pdc_study_id": study_id,
                        "sample_id": row.get("sample_id", ""),
                        "sample_submitter_id": row.get("sample_submitter_id", ""),
                        "aliquot_id": row.get("aliquot_id", ""),
                        "aliquot_submitter_id": row.get("aliquot_submitter_id", ""),
                        "matrix_column": row.get("matrix_column", ""),
                        "sample_type": row.get("sample_type_normalized", ""),
                        "matrix_missing_fraction": row.get("matrix_missing_fraction", ""),
                    }
                )
    write_tsv(out_dir / "pdc_sample_selection.tsv", pdc_selection_rows)

    cptac_ids = sorted({pid for ids in pdc_all_patients.values() for pid in ids} | set(slide_df["patient_id"].astype(str).tolist()))
    gdc_cases = fetch_gdc_cases_for_submitters(cptac_ids)
    for project in TCGA_PROJECTS:
        gdc_cases.extend(fetch_gdc_cases_for_project(project))
    dedup = {case.get("submitter_id", "") + "|" + ((case.get("project") or {}).get("project_id", "")): case for case in gdc_cases}
    gdc_cases = list(dedup.values())
    (out_dir / "gdc_cases_stage2.json").write_text(json.dumps(gdc_cases, indent=2) + "\n", encoding="utf-8")
    gdc_flat = [flatten_gdc_case(case) for case in gdc_cases if case.get("submitter_id")]
    gdc_by_patient_project = {(row["patient_id"], row["gdc_project_id"]): row for row in gdc_flat}
    write_tsv(out_dir / "gdc_case_clinical.tsv", gdc_flat)

    rna_map, gdc_slide_counts = build_rna_maps(root)
    master_rows: list[dict[str, Any]] = []

    for cohort, meta in CPTAC_COHORTS.items():
        patient_ids = sorted(pdc_all_patients[cohort] | {pid for c, pid in slide_by_key if c == cohort})
        project_hint = "CPTAC-2" if cohort == "CPTAC-COAD" else "CPTAC-3"
        for patient_id in patient_ids:
            clinical = dict(gdc_by_patient_project.get((patient_id, project_hint), {}))
            proteome = pdc_selected.get((cohort, patient_id, "proteome"), {})
            phospho = pdc_selected.get((cohort, patient_id, "phosphoproteome"), {})
            slide = slide_by_key.get((cohort, patient_id), {})
            if not clinical:
                clinical = {"clinical_source": "PDC minimal metadata"}
            for source in [proteome, phospho]:
                if source:
                    clinical.setdefault("stage", clean_text(source.get("tumor_stage", "")))
                    clinical.setdefault("grade", clean_text(source.get("tumor_grade", "")))
                    clinical.setdefault("primary_diagnosis", clean_text(source.get("primary_diagnosis", "")))
            sample_type = proteome.get("sample_type_normalized") or phospho.get("sample_type_normalized") or "Not Reported"
            row = {
                "cohort": cohort,
                "patient_id": patient_id,
                "data_scope": "discovery_extension",
                "slide_id": slide.get("slide_id", ""),
                "slide_path": slide.get("slide_path", ""),
                "scanner": slide.get("scanner", "Not Reported") if slide else "Not Reported",
                "magnification": slide.get("magnification", "Not Reported") if slide else "Not Reported",
                "histoqc_status": slide.get("histoqc_status", "not_available_no_slide"),
                "slide_selection_rule": slide.get("representative_rule_stage2", ""),
                "proteome_sample_id": proteome.get("aliquot_submitter_id", ""),
                "proteome_matrix_column": proteome.get("matrix_column", ""),
                "proteome_sample_type": proteome.get("sample_type_normalized", ""),
                "phospho_sample_id": phospho.get("aliquot_submitter_id", ""),
                "phospho_matrix_column": phospho.get("matrix_column", ""),
                "phospho_sample_type": phospho.get("sample_type_normalized", ""),
                "rna_sample_id": "",
                "sample_type": sample_type,
                "age": clinical.get("age", ""),
                "sex": clinical.get("sex", "Not Reported"),
                "stage": clinical.get("stage", "Not Reported"),
                "T_stage": clinical.get("T_stage", "Not Reported"),
                "N_stage": clinical.get("N_stage", "Not Reported"),
                "M_stage": clinical.get("M_stage", "Not Reported"),
                "grade": clinical.get("grade", "Not Reported"),
                "primary_diagnosis": clinical.get("primary_diagnosis", "Not Reported"),
                "recurrence": clinical.get("recurrence", "Not Reported"),
                "survival_time": clinical.get("survival_time", ""),
                "survival_event": clinical.get("survival_event", ""),
                "clinical_source": clinical.get("clinical_source", "Not Reported"),
                "pathology_available": bool(slide),
                "proteome_available": bool(proteome),
                "phosphoproteome_available": bool(phospho),
                "rna_available": False,
            }
            endpoint, endpoint_source = endpoint_from_row(row)
            row["advanced_at_presentation"] = endpoint
            row["advanced_endpoint_source"] = endpoint_source
            master_rows.append(row)

    for project in TCGA_PROJECTS:
        patient_ids = sorted({key.split(":", 1)[1] for key in rna_map if key.startswith(project + ":")} | {key.split(":", 1)[1] for key in gdc_slide_counts if key.startswith(project + ":")})
        for patient_id in patient_ids:
            clinical = dict(gdc_by_patient_project.get((patient_id, project), {}))
            rna = rna_map.get(f"{project}:{patient_id}", {})
            row = {
                "cohort": project,
                "patient_id": patient_id,
                "data_scope": "external_validation",
                "slide_id": "",
                "slide_path": "",
                "scanner": "Not Reported",
                "magnification": "Not Reported",
                "histoqc_status": "not_downloaded_tcga_slide_manifest_only",
                "slide_selection_rule": "TCGA slides manifest-only in Stage 1",
                "proteome_sample_id": "",
                "proteome_matrix_column": "",
                "proteome_sample_type": "",
                "phospho_sample_id": "",
                "phospho_matrix_column": "",
                "phospho_sample_type": "",
                "rna_sample_id": rna.get("rna_sample_id", ""),
                "rna_file_name": rna.get("rna_file_name", ""),
                "sample_type": rna.get("rna_sample_type", "Not Reported"),
                "age": clinical.get("age", ""),
                "sex": clinical.get("sex", "Not Reported"),
                "stage": clinical.get("stage", "Not Reported"),
                "T_stage": clinical.get("T_stage", "Not Reported"),
                "N_stage": clinical.get("N_stage", "Not Reported"),
                "M_stage": clinical.get("M_stage", "Not Reported"),
                "grade": clinical.get("grade", "Not Reported"),
                "primary_diagnosis": clinical.get("primary_diagnosis", "Not Reported"),
                "recurrence": clinical.get("recurrence", "Not Reported"),
                "survival_time": clinical.get("survival_time", ""),
                "survival_event": clinical.get("survival_event", ""),
                "clinical_source": clinical.get("clinical_source", "GDC cases API"),
                "pathology_available": False,
                "gdc_slide_manifest_count": gdc_slide_counts.get(f"{project}:{patient_id}", 0),
                "proteome_available": False,
                "phosphoproteome_available": False,
                "rna_available": bool(rna),
            }
            endpoint, endpoint_source = endpoint_from_row(row)
            row["advanced_at_presentation"] = endpoint
            row["advanced_endpoint_source"] = endpoint_source
            master_rows.append(row)

    master = pd.DataFrame(master_rows).sort_values(["cohort", "patient_id"])
    master = normalize_table_for_parquet(master)
    master.to_parquet(out_dir / "master_case_table.parquet", index=False)
    master.to_csv(out_dir / "master_case_table.tsv", sep="\t", index=False)

    analysis_rows: list[dict[str, Any]] = []
    for _, row in master.iterrows():
        primary_ok = row.get("sample_type") == "Primary Tumor"
        protein_ok = bool(row.get("pathology_available")) and bool(row.get("proteome_available")) and row.get("proteome_sample_type") == "Primary Tumor"
        phospho_ok = bool(row.get("pathology_available")) and bool(row.get("phosphoproteome_available")) and row.get("phospho_sample_type") == "Primary Tumor"
        memberships = {
            "set_protein": protein_ok,
            "set_phospho": phospho_ok,
            "set_triplet": protein_ok and phospho_ok,
            "set_plus_rna": protein_ok and phospho_ok and bool(row.get("rna_available")),
            "external_rna": row.get("data_scope") == "external_validation" and bool(row.get("rna_available")),
        }
        for set_name, include in memberships.items():
            if include:
                analysis_rows.append(
                    {
                        "set_name": set_name,
                        "cohort": row["cohort"],
                        "patient_id": row["patient_id"],
                        "sample_type": row.get("sample_type", ""),
                        "reason": "meets Stage 2 availability and primary-tumor rules" if set_name != "external_rna" else "TCGA external RNA case",
                    }
                )
    write_tsv(out_dir / "analysis_sets.tsv", analysis_rows, ["set_name", "cohort", "patient_id", "sample_type", "reason"])
    write_tsv(
        out_dir / "analysis_set_definitions.tsv",
        [{"set_name": key, "definition": value} for key, value in ANALYSIS_SET_DEFINITIONS.items()],
        ["set_name", "definition"],
    )
    analysis = pd.DataFrame(analysis_rows)
    analysis_completeness_rows: list[dict[str, Any]] = []
    if not analysis.empty:
        for (set_name, cohort), set_sub in analysis.groupby(["set_name", "cohort"]):
            keys = set(cohort + "|" + patient_id for cohort, patient_id in zip(set_sub["cohort"], set_sub["patient_id"]))
            master_sub = master[(master["cohort"] + "|" + master["patient_id"]).isin(keys)]
            endpoint_available = master_sub["advanced_at_presentation"].astype(str).str.len() > 0
            analysis_completeness_rows.append(
                {
                    "set_name": set_name,
                    "cohort": cohort,
                    "n_patients": len(master_sub),
                    "advanced_endpoint_available_n": int(endpoint_available.sum()),
                    "advanced_endpoint_completeness": round(float(endpoint_available.mean()), 4) if len(master_sub) else 0.0,
                    "stage_available_n": int((master_sub["stage"].map(norm_missing).astype(str).str.len() > 0).sum()),
                    "n_stage_available_n": int((master_sub["N_stage"].map(norm_missing).astype(str).str.len() > 0).sum()),
                    "m_stage_available_n": int((master_sub["M_stage"].map(norm_missing).astype(str).str.len() > 0).sum()),
                }
            )
    write_tsv(
        out_dir / "analysis_set_endpoint_completeness.tsv",
        analysis_completeness_rows,
        [
            "set_name",
            "cohort",
            "n_patients",
            "advanced_endpoint_available_n",
            "advanced_endpoint_completeness",
            "stage_available_n",
            "n_stage_available_n",
            "m_stage_available_n",
        ],
    )

    completeness_rows = []
    for cohort in ["CPTAC-COAD", "CPTAC-PDAC", "CPTAC-STAD", "TCGA-COAD", "TCGA-PAAD", "TCGA-STAD"]:
        sub = master[master["cohort"] == cohort]
        if sub.empty:
            continue
        endpoint_available = sub["advanced_at_presentation"].astype(str).str.len() > 0
        completeness_rows.append(
            {
                "cohort": cohort,
                "n_patients": len(sub),
                "advanced_endpoint_available_n": int(endpoint_available.sum()),
                "advanced_endpoint_completeness": round(float(endpoint_available.mean()), 4),
                "stage_available_n": int((sub["stage"].map(norm_missing).astype(str).str.len() > 0).sum()),
                "n_stage_available_n": int((sub["N_stage"].map(norm_missing).astype(str).str.len() > 0).sum()),
                "m_stage_available_n": int((sub["M_stage"].map(norm_missing).astype(str).str.len() > 0).sum()),
            }
        )
    write_tsv(out_dir / "clinical_endpoint_completeness.tsv", completeness_rows)

    analysis_set_counts = {key: 0 for key in ANALYSIS_SET_DEFINITIONS}
    if analysis_rows:
        analysis_set_counts.update(pd.DataFrame(analysis_rows).groupby("set_name").size().to_dict())
    summary = {
        "master_rows": len(master),
        "analysis_set_counts": analysis_set_counts,
        "slide_series_rows": len(slide_df),
        "representative_slide_rows": int(slide_df["is_representative_stage2"].sum()),
        "clinical_endpoint_completeness": completeness_rows,
        "analysis_set_endpoint_completeness": analysis_completeness_rows,
        "random_seed": RANDOM_SEED,
    }
    (out_dir / "stage2_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
