#!/usr/bin/env python3
"""Stage 7 recurrent discordance archetype discovery."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


META_COLUMNS = {"cohort", "patient_id"}
RANDOM_SEED = 20260609


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"patient_id": str})


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in META_COLUMNS]


def within_cohort_zscore(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = np.zeros((len(df), len(cols)), dtype=np.float64)
    values = df[cols].to_numpy(dtype=np.float64)
    for cohort in sorted(df["cohort"].unique()):
        idx = np.flatnonzero(df["cohort"].to_numpy() == cohort)
        block = values[idx]
        center = np.nanmean(block, axis=0)
        scale = np.nanstd(block, axis=0, ddof=1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        out[idx] = (block - center) / scale
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite values after within-cohort z-score")
    return out


def split_positive_negative(z: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    pos = np.clip(z, 0.0, None)
    neg = np.clip(-z, 0.0, None)
    names = [f"pos__{name}" for name in feature_names] + [f"neg__{name}" for name in feature_names]
    return np.concatenate([pos, neg], axis=1), names


def balanced_bootstrap_indices(cohorts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cohort_values = sorted(set(cohorts.tolist()))
    min_n = min(int(np.sum(cohorts == cohort)) for cohort in cohort_values)
    sampled: list[np.ndarray] = []
    for cohort in cohort_values:
        idx = np.flatnonzero(cohorts == cohort)
        sampled.append(rng.choice(idx, size=min_n, replace=True))
    out = np.concatenate(sampled)
    rng.shuffle(out)
    return out


def fit_nmf(x: np.ndarray, rank: int, seed: int, max_iter: int) -> tuple[NMF, np.ndarray, np.ndarray, float]:
    model = NMF(
        n_components=rank,
        init="random",
        random_state=seed,
        max_iter=max_iter,
        tol=1e-4,
        solver="cd",
        beta_loss="frobenius",
    )
    w = model.fit_transform(x)
    h = model.components_
    return model, w, h, float(model.reconstruction_err_)


def fit_consensus_start(
    x: np.ndarray,
    sampled: np.ndarray,
    rank: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, float, float]:
    model, _, _, err = fit_nmf(x[sampled], rank, seed, max_iter)
    w_full = model.transform(x)
    labels = np.argmax(w_full, axis=1)
    return labels, float(err / math.sqrt(float(x[sampled].size))), float(np.mean(np.linalg.norm(w_full, axis=1)))


def consensus_for_rank(
    x: np.ndarray,
    cohorts: np.ndarray,
    rank: int,
    starts: int,
    seed: int,
    max_iter: int,
    n_jobs: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + rank * 1000)
    n = x.shape[0]
    same = np.zeros((n, n), dtype=np.float64)
    reconstruction_errors: list[float] = []
    score_norms: list[float] = []
    assignment_counts = np.zeros((n, rank), dtype=np.int32)
    sampled_indices = [balanced_bootstrap_indices(cohorts, rng) for _ in range(starts)]
    print(f"rank={rank} consensus starts={starts} n_jobs={n_jobs}", flush=True)
    if n_jobs == 1:
        results = [
            fit_consensus_start(x, sampled, rank, seed + rank * 100000 + start, max_iter)
            for start, sampled in enumerate(sampled_indices)
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(fit_consensus_start)(x, sampled, rank, seed + rank * 100000 + start, max_iter)
            for start, sampled in enumerate(sampled_indices)
        )
    for start, (labels, err, score_norm) in enumerate(results):
        same += labels[:, None] == labels[None, :]
        for sample_idx, label in enumerate(labels):
            assignment_counts[sample_idx, label] += 1
        reconstruction_errors.append(err)
        score_norms.append(score_norm)
        if (start + 1) % max(1, starts // 10) == 0:
            print(f"rank={rank} start={start + 1}/{starts}", flush=True)
    consensus = same / float(starts)
    distance = 1.0 - consensus
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if np.allclose(condensed, 0):
        coph = float("nan")
        silhouette = float("nan")
        labels = np.zeros(n, dtype=int)
    else:
        z_link = linkage(condensed, method="average")
        coph, _ = cophenet(z_link, condensed)
        _, w_final, h_final, _ = best_full_nmf(x, rank, seed + rank * 7777, max_iter, n_starts=20, n_jobs=n_jobs)
        labels = np.argmax(w_final, axis=1)
        if len(set(labels.tolist())) > 1:
            silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
        else:
            silhouette = float("nan")
    within_values = []
    between_values = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within_values.append(consensus[i, j])
            else:
                between_values.append(consensus[i, j])
    within_mean = float(np.mean(within_values)) if within_values else float("nan")
    between_mean = float(np.mean(between_values)) if between_values else float("nan")
    margin = within_mean - between_mean if np.isfinite(within_mean) and np.isfinite(between_mean) else float("nan")
    return {
        "rank": rank,
        "starts": starts,
        "consensus": consensus,
        "labels": labels,
        "assignment_counts": assignment_counts,
        "mean_reconstruction_error": float(np.mean(reconstruction_errors)),
        "sd_reconstruction_error": float(np.std(reconstruction_errors, ddof=1)) if len(reconstruction_errors) > 1 else 0.0,
        "mean_score_norm": float(np.mean(score_norms)),
        "cophenetic_correlation": float(coph),
        "consensus_silhouette": silhouette,
        "within_consensus_mean": within_mean,
        "between_consensus_mean": between_mean,
        "within_minus_between_consensus": margin,
    }


def best_full_nmf(
    x: np.ndarray,
    rank: int,
    seed: int,
    max_iter: int,
    n_starts: int,
    n_jobs: int,
) -> tuple[NMF, np.ndarray, np.ndarray, float]:
    if n_jobs == 1:
        results = [fit_nmf(x, rank, seed + i, max_iter) for i in range(n_starts)]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(fit_nmf)(x, rank, seed + i, max_iter) for i in range(n_starts)
        )
    return min(results, key=lambda item: item[3])


def choose_rank(stability: pd.DataFrame) -> int:
    df = stability.copy()
    rec = df["mean_reconstruction_error"].to_numpy(dtype=float)
    rec_scaled = (rec.max() - rec) / (rec.max() - rec.min() + 1e-12)
    coph = df["cophenetic_correlation"].fillna(0).to_numpy(dtype=float)
    sil = df["consensus_silhouette"].fillna(0).to_numpy(dtype=float)
    margin = df["within_minus_between_consensus"].fillna(0).to_numpy(dtype=float)
    rank_penalty = (df["rank"].to_numpy(dtype=float) - df["rank"].min()) / (df["rank"].max() - df["rank"].min() + 1e-12)
    score = 0.35 * coph + 0.25 * sil + 0.25 * margin + 0.15 * rec_scaled - 0.05 * rank_penalty
    df["selection_score"] = score
    best_idx = int(np.nanargmax(score))
    return int(df.iloc[best_idx]["rank"])


def parse_split_feature(name: str) -> dict[str, str]:
    direction_token, original = name.split("__", 1)
    target_set, feature = original.split("__", 1)
    direction = "positive_discordance" if direction_token == "pos" else "negative_discordance"
    return {"direction": direction, "target_set": target_set, "feature": feature}


def write_loadings(out_dir: Path, h: np.ndarray, split_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component_idx in range(h.shape[0]):
        values = h[component_idx]
        order = np.argsort(values)[::-1]
        total = float(values.sum()) if float(values.sum()) > 0 else 1.0
        for rank_idx, feature_idx in enumerate(order, start=1):
            parsed = parse_split_feature(split_names[feature_idx])
            rows.append(
                {
                    "archetype": f"A{component_idx + 1}",
                    "loading_rank": rank_idx,
                    "split_feature": split_names[feature_idx],
                    **parsed,
                    "loading": float(values[feature_idx]),
                    "loading_fraction": float(values[feature_idx] / total),
                }
            )
    loadings = pd.DataFrame(rows)
    loadings.to_csv(out_dir / "archetype_loadings.tsv", sep="\t", index=False)
    loadings.groupby("archetype", group_keys=False).head(50).to_csv(
        out_dir / "archetype_top_loadings.tsv",
        sep="\t",
        index=False,
    )
    return loadings


def write_scores(out_dir: Path, meta: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    score_cols = [f"A{i + 1}_score" for i in range(w.shape[1])]
    score_df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(w, columns=score_cols)], axis=1)
    total = w.sum(axis=1)
    total = np.where(total > 0, total, 1.0)
    normalized = w / total[:, None]
    for i in range(w.shape[1]):
        score_df[f"A{i + 1}_fraction"] = normalized[:, i]
    dominant = np.argmax(w, axis=1)
    score_df["dominant_archetype"] = [f"A{i + 1}" for i in dominant]
    score_df["dominant_archetype_fraction"] = normalized[np.arange(w.shape[0]), dominant]
    score_df.to_csv(out_dir / "sample_archetype_scores.tsv", sep="\t", index=False)
    return score_df


def leave_one_cohort_projection(
    x: np.ndarray,
    meta: pd.DataFrame,
    full_h: np.ndarray,
    rank: int,
    seed: int,
    max_iter: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    score_parts: list[pd.DataFrame] = []
    cohorts = sorted(meta["cohort"].unique())
    for cohort in cohorts:
        train_idx = np.flatnonzero(meta["cohort"].to_numpy() != cohort)
        test_idx = np.flatnonzero(meta["cohort"].to_numpy() == cohort)
        model, w_train, h_train, train_err = best_full_nmf(x[train_idx], rank, seed + len(rows) * 1000, max_iter, n_starts=30, n_jobs=n_jobs)
        w_test = model.transform(x[test_idx])
        sim = cosine_similarity(h_train, full_h)
        assigned_full = np.argmax(sim, axis=1)
        for local_idx in range(rank):
            rows.append(
                {
                    "heldout_cohort": cohort,
                    "train_cohorts": ",".join(c for c in cohorts if c != cohort),
                    "local_archetype": f"local_A{local_idx + 1}",
                    "matched_full_archetype": f"A{assigned_full[local_idx] + 1}",
                    "loading_cosine_to_full": float(sim[local_idx, assigned_full[local_idx]]),
                    "train_samples": int(len(train_idx)),
                    "test_samples": int(len(test_idx)),
                    "train_reconstruction_error": float(train_err / math.sqrt(float(x[train_idx].size))),
                    "test_mean_score": float(np.mean(w_test[:, local_idx])),
                    "test_median_score": float(np.median(w_test[:, local_idx])),
                }
            )
        part = meta.iloc[test_idx][["cohort", "patient_id"]].reset_index(drop=True).copy()
        for local_idx in range(rank):
            part[f"local_A{local_idx + 1}_score"] = w_test[:, local_idx]
        score_parts.append(part)
    return pd.DataFrame(rows), pd.concat(score_parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input", default="results/discordance/discordance_pathway_matrix.tsv")
    parser.add_argument("--stage6-summary", default="results/discordance/stage6_build_summary.json")
    parser.add_argument("--output-dir", default="results/archetypes")
    parser.add_argument("--rank-min", type=int, default=2)
    parser.add_argument("--rank-max", type=int, default=8)
    parser.add_argument("--starts", type=int, default=200)
    parser.add_argument("--final-starts", type=int, default=80)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / args.output_dir
    qc_dir = root / "logs/qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, details: dict[str, Any] | None = None, severity: str = "FAIL") -> None:
        checks.append({"check": name, "status": "PASS" if ok else severity, "details": details or {}})

    stage6 = json.loads((root / args.stage6_summary).read_text(encoding="utf-8"))
    check("stage6_summary_pass", stage6.get("status") == "PASS" and stage6.get("fail_count") == 0, {"status": stage6.get("status"), "fail_count": stage6.get("fail_count"), "warn_count": stage6.get("warn_count")})
    check("stage7_input_is_stage6_discordance_matrix", args.input == "results/discordance/discordance_pathway_matrix.tsv", {"input": args.input})
    df = read_table(root / args.input)
    meta = df[["cohort", "patient_id"]].copy()
    cols = feature_columns(df)
    check("input_shape", df.shape == (236, 881) and len(cols) == 879, {"rows": df.shape[0], "columns": df.shape[1], "features": len(cols)})
    z = within_cohort_zscore(df, cols)
    x, split_names = split_positive_negative(z, cols)
    check("nonnegative_split_matrix", bool(np.isfinite(x).all()) and float(x.min()) >= 0.0 and x.shape == (236, 1758), {"rows": x.shape[0], "columns": x.shape[1], "min": float(x.min()), "max": float(x.max())})

    rank_results: list[dict[str, Any]] = []
    for rank in range(args.rank_min, args.rank_max + 1):
        result = consensus_for_rank(
            x,
            meta["cohort"].to_numpy(dtype=str),
            rank=rank,
            starts=args.starts,
            seed=args.seed,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
        )
        rank_results.append(result)

    stability_rows = []
    for result in rank_results:
        row = {k: v for k, v in result.items() if k not in {"consensus", "labels", "assignment_counts"}}
        stability_rows.append(row)
    stability = pd.DataFrame(stability_rows)
    selected_rank = choose_rank(stability)
    stability["selection_score"] = (
        0.35 * stability["cophenetic_correlation"].fillna(0)
        + 0.25 * stability["consensus_silhouette"].fillna(0)
        + 0.25 * stability["within_minus_between_consensus"].fillna(0)
        + 0.15
        * (
            (stability["mean_reconstruction_error"].max() - stability["mean_reconstruction_error"])
            / (stability["mean_reconstruction_error"].max() - stability["mean_reconstruction_error"].min() + 1e-12)
        )
        - 0.05
        * (
            (stability["rank"] - stability["rank"].min())
            / (stability["rank"].max() - stability["rank"].min() + 1e-12)
        )
    )
    stability["selected"] = stability["rank"] == selected_rank
    stability.to_csv(out_dir / "archetype_stability.tsv", sep="\t", index=False)

    model, w, h, full_err = best_full_nmf(x, selected_rank, args.seed + 424242, args.max_iter, args.final_starts, args.n_jobs)
    loadings = write_loadings(out_dir, h, split_names)
    scores = write_scores(out_dir, meta, w)
    projection, projection_scores = leave_one_cohort_projection(x, meta, h, selected_rank, args.seed + 777, args.max_iter, args.n_jobs)
    projection.to_csv(out_dir / "projection_results.tsv", sep="\t", index=False)
    projection_scores.to_csv(out_dir / "leave_one_cohort_projection_scores.tsv", sep="\t", index=False)

    selected = stability[stability["selected"]].iloc[0].to_dict()
    dominant_counts = scores.groupby(["cohort", "dominant_archetype"]).size().reset_index(name="n")
    dominant_counts.to_csv(out_dir / "archetype_cohort_counts.tsv", sep="\t", index=False)
    top_summary = (
        loadings.groupby(["archetype", "direction", "target_set"], as_index=False)
        .head(15)
        [["archetype", "direction", "target_set", "feature", "loading_rank", "loading"]]
    )
    top_summary.to_csv(out_dir / "archetype_top_feature_summary.tsv", sep="\t", index=False)

    check("selected_rank_in_range", args.rank_min <= selected_rank <= args.rank_max, {"selected_rank": selected_rank})
    check("selected_rank_stability_finite", all(np.isfinite(float(selected.get(k, np.nan))) for k in ["cophenetic_correlation", "within_minus_between_consensus", "mean_reconstruction_error"]), selected)
    check("sample_scores_shape", scores.shape[0] == 236 and "dominant_archetype" in scores.columns, {"rows": scores.shape[0], "columns": scores.shape[1]})
    check("loadings_shape", len(loadings) == selected_rank * x.shape[1], {"rows": len(loadings), "expected_rows": selected_rank * x.shape[1]})
    check("projection_results_shape", len(projection) == selected_rank * len(meta["cohort"].unique()), {"rows": len(projection), "expected_rows": selected_rank * len(meta["cohort"].unique())})
    check("dominant_archetypes_present", set(scores["dominant_archetype"]) == {f"A{i + 1}" for i in range(selected_rank)}, {"counts": scores["dominant_archetype"].value_counts().sort_index().to_dict()}, severity="WARN")
    min_projection_cosine = float(projection["loading_cosine_to_full"].min()) if len(projection) else float("nan")
    check("leave_one_cohort_projection_nontrivial", np.isfinite(min_projection_cosine) and min_projection_cosine > 0.20, {"min_loading_cosine_to_full": min_projection_cosine}, severity="WARN")
    for rel in [
        "archetype_loadings.tsv",
        "sample_archetype_scores.tsv",
        "archetype_stability.tsv",
        "projection_results.tsv",
        "archetype_top_loadings.tsv",
        "archetype_cohort_counts.tsv",
    ]:
        path = out_dir / rel
        check(f"output_exists::{rel}", path.exists() and path.stat().st_size > 0, {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size if path.exists() else 0})

    fail_count = sum(1 for row in checks if row["status"] == "FAIL")
    warn_count = sum(1 for row in checks if row["status"] == "WARN")
    summary = {
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if fail_count == 0 else "FAIL",
        "fail_count": fail_count,
        "warn_count": warn_count,
        "input": args.input,
        "stage6_summary": args.stage6_summary,
        "preprocessing": "within-cohort z-score, then D+ and D- non-negative split",
        "rank_range": [args.rank_min, args.rank_max],
        "starts_per_rank": args.starts,
        "final_starts": args.final_starts,
        "n_jobs": args.n_jobs,
        "selected_rank": selected_rank,
        "selected_rank_metrics": selected,
        "full_model_reconstruction_error": float(full_err / math.sqrt(float(x.size))),
        "dominant_archetype_counts": scores["dominant_archetype"].value_counts().sort_index().to_dict(),
        "cohort_dominant_counts": dominant_counts.to_dict(orient="records"),
        "projection_min_loading_cosine_to_full": min_projection_cosine,
        "outputs": {
            "archetype_loadings": (out_dir / "archetype_loadings.tsv").relative_to(root).as_posix(),
            "sample_archetype_scores": (out_dir / "sample_archetype_scores.tsv").relative_to(root).as_posix(),
            "archetype_stability": (out_dir / "archetype_stability.tsv").relative_to(root).as_posix(),
            "projection_results": (out_dir / "projection_results.tsv").relative_to(root).as_posix(),
            "leave_one_cohort_projection_scores": (out_dir / "leave_one_cohort_projection_scores.tsv").relative_to(root).as_posix(),
            "archetype_top_loadings": (out_dir / "archetype_top_loadings.tsv").relative_to(root).as_posix(),
            "archetype_cohort_counts": (out_dir / "archetype_cohort_counts.tsv").relative_to(root).as_posix(),
            "stage7_build_summary": (out_dir / "stage7_build_summary.json").relative_to(root).as_posix(),
        },
        "checks": checks,
        "notes": [
            "Archetype discovery is unsupervised and does not use clinical outcome labels.",
            "Stage 8 remains responsible for clinical association and hidden aggressive group testing.",
            "Stage 7 uses only the Stage 6 QC-passed discordance pathway matrix.",
        ],
    }
    (out_dir / "stage7_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    qc_json = qc_dir / "stage7_archetype_qc.json"
    qc_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(checks).assign(details=lambda frame: frame["details"].map(lambda x: json.dumps(x, ensure_ascii=False))).to_csv(
        qc_dir / "stage7_archetype_qc_summary.tsv",
        sep="\t",
        index=False,
    )
    print(json.dumps({k: summary[k] for k in ["status", "fail_count", "warn_count", "selected_rank", "dominant_archetype_counts", "projection_min_loading_cosine_to_full"]}, ensure_ascii=False))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
