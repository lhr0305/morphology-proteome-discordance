#!/usr/bin/env python3
"""Stage 9 spatial localization for discordance archetype proxies.

The primary scorer is a learned attention-MIL slide-level regressor trained on
tile embeddings and archetype scores. A linear ridge slide-regression scorer is
kept as a comparator because earlier Stage 9 runs used it as the fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixels import pixel_array
from scipy import ndimage, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 20260609
TILE_SIZE = 256
CELL_PROXY_MIN_NUCLEUS_AREA = 8
CELL_PROXY_MAX_NUCLEUS_AREA = 2000
META_COLUMNS = {
    "cohort",
    "patient_id",
    "slide_candidate_key",
    "slide_id",
    "tile_count",
    "embedding_path",
    "model_repo",
}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"patient_id": str})


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("emb_") and col not in META_COLUMNS]


def safe_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        text = "unknown"
    keep = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text]
    return "".join(keep)[:160]


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


def cohort_zscore(df: pd.DataFrame, value_col: str) -> pd.Series:
    out = np.full(len(df), np.nan, dtype=np.float64)
    for cohort in sorted(df["cohort"].dropna().unique()):
        idx = df["cohort"].to_numpy() == cohort
        values = pd.to_numeric(df.loc[idx, value_col], errors="coerce").to_numpy(dtype=np.float64)
        mean = np.nanmean(values)
        sd = np.nanstd(values, ddof=1)
        if not np.isfinite(sd) or sd <= 1e-8:
            sd = 1.0
        out[idx] = (values - mean) / sd
    return pd.Series(out, index=df.index)


def load_stage8_target(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads((root / args.stage8_summary).read_text())
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Stage 8 summary is not PASS: {args.stage8_summary}")
    strict = list(summary.get("strict_adverse_archetypes", []) or [])
    proxy = list(summary.get("directional_proxy_adverse_archetypes", []) or [])
    if args.target_archetype:
        target = args.target_archetype
        source = "user_requested_target"
        status = "manual_target_from_cli"
    elif strict:
        target = strict[0]
        source = "strict_adverse_archetype"
        status = "strict_significant_adverse_archetype"
    elif proxy:
        target = proxy[0]
        source = "directional_proxy_adverse_archetype"
        status = "exploratory_directional_proxy_no_significant_adverse_archetype"
    else:
        raise RuntimeError("No strict or directional proxy adverse archetype available for Stage 9")
    return {
        "target_archetype": target,
        "target_key": safe_component(f"{target}_proxy" if source != "strict_adverse_archetype" else target),
        "score_col": f"{target}_score",
        "source": source,
        "status": status,
        "stage8_warn_count": int(summary.get("warn_count", 0)),
        "stage8_summary": args.stage8_summary,
        "stage8_notes": summary.get("notes", []),
    }


def train_slide_ridge(
    slide_table: pd.DataFrame,
    emb_cols: list[str],
    target_col: str,
    groups: np.ndarray,
    alphas: list[float],
    folds: int,
) -> tuple[dict[str, Any], StandardScaler, Ridge, pd.DataFrame]:
    x = slide_table[emb_cols].to_numpy(dtype=np.float32)
    y = slide_table[target_col].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Non-finite slide-level training data")
    unique_groups = np.unique(groups)
    n_splits = max(2, min(folds, len(unique_groups)))
    fold_rows: list[dict[str, Any]] = []
    alpha_scores: list[dict[str, Any]] = []
    best_alpha = alphas[0]
    best_mse = float("inf")
    for alpha in alphas:
        pred = np.full(len(y), np.nan, dtype=np.float64)
        cv = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups=groups)):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_test = scaler.transform(x[test_idx])
            model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
            model.fit(x_train, y[train_idx])
            pred[test_idx] = model.predict(x_test)
            fold_rows.append(
                {
                    "alpha": alpha,
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_patients": int(len(np.unique(groups[test_idx]))),
                }
            )
        mse = float(mean_squared_error(y, pred))
        pearson = float(stats.pearsonr(y, pred).statistic) if len(y) > 2 else np.nan
        spearman = float(stats.spearmanr(y, pred).statistic) if len(y) > 2 else np.nan
        r2 = float(r2_score(y, pred))
        alpha_scores.append({"alpha": alpha, "mse": mse, "pearson": pearson, "spearman": spearman, "r2": r2})
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
    model.fit(x_scaled, y)
    train_pred = model.predict(x_scaled)
    metrics = {
        "best_alpha": float(best_alpha),
        "n_slides": int(len(slide_table)),
        "n_patients": int(len(unique_groups)),
        "folds": int(n_splits),
        "target_mean": float(np.mean(y)),
        "target_sd": float(np.std(y, ddof=1)),
        "train_r2": float(r2_score(y, train_pred)),
        "train_pearson": float(stats.pearsonr(y, train_pred).statistic),
        "train_spearman": float(stats.spearmanr(y, train_pred).statistic),
        "alpha_scores": alpha_scores,
    }
    fold_table = pd.DataFrame(fold_rows)
    return metrics, scaler, model, fold_table


def torch_import():
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - depends on project env
        raise RuntimeError(f"PyTorch is required for Stage 9 attention MIL but could not be imported: {exc}") from exc
    return torch, nn


def read_h5_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        embeddings = handle["embeddings"][:].astype(np.float32)
        coords = handle["coords_x_y_tissue"][:].astype(np.float32)
        attrs = {key: handle.attrs[key] for key in handle.attrs.keys()}
    return embeddings, coords, attrs


def score_tile_matrix(embeddings: np.ndarray, scaler: StandardScaler, model: Ridge) -> np.ndarray:
    scale = scaler.scale_.astype(np.float32)
    scale = np.where(scale > 1e-8, scale, 1.0)
    centered = (embeddings - scaler.mean_.astype(np.float32)) / scale
    return centered @ model.coef_.astype(np.float32)


@dataclass
class RidgeTileScorer:
    scaler: StandardScaler
    model: Ridge
    embedding_dim: int
    method: str = "slide_level_ridge_linear_per_tile_scoring_comparator"
    score_name: str = "linear_contribution"

    def score(self, embeddings: np.ndarray) -> tuple[np.ndarray, float]:
        scores = score_tile_matrix(embeddings, self.scaler, self.model).astype(np.float32)
        pred = float(np.mean(scores)) if len(scores) else np.nan
        return scores, pred


def resolve_torch_device(requested: str) -> tuple[Any, str]:
    torch, _ = torch_import()
    if requested == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        name = requested
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA for attention MIL, but torch.cuda.is_available() is False")
    return torch.device(name), name


def scale_embeddings(embeddings: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scale = scaler.scale_.astype(np.float32)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return ((embeddings.astype(np.float32, copy=False) - scaler.mean_.astype(np.float32)) / scale).astype(np.float32)


def sample_tile_embeddings(
    embeddings: np.ndarray,
    max_tiles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_tiles <= 0 or len(embeddings) <= max_tiles:
        return embeddings.astype(np.float32, copy=False)
    idx = np.sort(rng.choice(np.arange(len(embeddings)), size=max_tiles, replace=False).astype(np.int64))
    return embeddings[idx].astype(np.float32, copy=False)


def preload_mil_training_bags(
    root: Path,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED + 1901)
    bags: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    total_tiles_before = 0
    total_tiles_after = 0
    for row_idx, row in manifest.reset_index(drop=True).iterrows():
        if args.max_slides and row_idx >= args.max_slides:
            break
        path = root / str(row["embedding_path"])
        if not path.exists():
            continue
        embeddings, _, _ = read_h5_embeddings(path)
        total_tiles_before += int(len(embeddings))
        sampled = sample_tile_embeddings(embeddings, args.mil_train_max_tiles, rng)
        total_tiles_after += int(len(sampled))
        bags.append(sampled)
        rows.append(
            {
                "cohort": str(row["cohort"]),
                "patient_id": str(row["patient_id"]),
                "slide_candidate_key": str(row["slide_candidate_key"]),
                "slide_id": str(row.get("slide_id", "")),
                "target_score_z": float(row["target_score_z"]),
                "embedding_path": str(row["embedding_path"]),
                "sampled_tiles": int(len(sampled)),
                "available_tiles": int(len(embeddings)),
            }
        )
    if not bags:
        raise RuntimeError("No MIL training bags could be loaded from embedding manifest")
    bag_table = pd.DataFrame(rows)
    y = bag_table["target_score_z"].to_numpy(dtype=np.float32)
    summary = {
        "training_bags": int(len(bags)),
        "training_patients": int(bag_table["patient_id"].nunique()),
        "tiles_before_sampling": int(total_tiles_before),
        "tiles_after_sampling": int(total_tiles_after),
        "mil_train_max_tiles": int(args.mil_train_max_tiles),
    }
    return bags, y, bag_table, summary


def make_scaled_bags(raw_bags: list[np.ndarray], scaler: StandardScaler) -> list[np.ndarray]:
    return [scale_embeddings(bag, scaler) for bag in raw_bags]


def build_attention_model(input_dim: int, args: argparse.Namespace) -> Any:
    _, nn = torch_import()

    class AttentionMILRegressor(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, attn_dim: int, dropout: float) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, attn_dim),
                nn.Tanh(),
                nn.Linear(attn_dim, 1),
            )
            self.regressor = nn.Linear(hidden_dim, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            h = self.encoder(x)
            logits = self.attention(h).squeeze(-1)
            weights = logits.softmax(dim=0)
            pooled = (weights.unsqueeze(-1) * h).sum(dim=0)
            pred = self.regressor(pooled).squeeze(-1)
            return pred, weights

    return AttentionMILRegressor(input_dim, args.mil_hidden_dim, args.mil_attention_dim, args.mil_dropout)


def evaluate_attention_model(model: Any, bags: list[np.ndarray], y: np.ndarray, indices: np.ndarray, device: Any) -> tuple[np.ndarray, dict[str, Any]]:
    torch, _ = torch_import()
    model.eval()
    preds = np.full(len(indices), np.nan, dtype=np.float64)
    with torch.no_grad():
        for out_i, bag_i in enumerate(indices):
            xb = torch.as_tensor(bags[int(bag_i)], dtype=torch.float32, device=device)
            pred, _ = model(xb)
            preds[out_i] = float(pred.detach().cpu().item())
    truth = y[indices].astype(np.float64)
    finite = np.isfinite(preds) & np.isfinite(truth)
    if finite.sum() >= 2:
        pearson = float(stats.pearsonr(truth[finite], preds[finite]).statistic)
        spearman = float(stats.spearmanr(truth[finite], preds[finite]).statistic)
        r2 = float(r2_score(truth[finite], preds[finite]))
        mse = float(mean_squared_error(truth[finite], preds[finite]))
    else:
        pearson = spearman = r2 = mse = np.nan
    return preds, {"mse": mse, "pearson": pearson, "spearman": spearman, "r2": r2}


def fit_attention_model(
    bags: list[np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    input_dim: int,
    args: argparse.Namespace,
    device: Any,
    epochs: int,
    seed_offset: int,
) -> tuple[Any, dict[str, Any], np.ndarray | None]:
    torch, _ = torch_import()
    torch.manual_seed(RANDOM_SEED + seed_offset)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(RANDOM_SEED + seed_offset)
    model = build_attention_model(input_dim, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mil_lr, weight_decay=args.mil_weight_decay)
    rng = np.random.default_rng(RANDOM_SEED + 2400 + seed_offset)
    train_losses: list[float] = []
    best_state = None
    best_val_mse = float("inf")
    best_epoch = 0
    patience_left = args.mil_patience
    y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.array(train_idx, dtype=np.int64)
        rng.shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for bag_i in order:
            xb = torch.as_tensor(bags[int(bag_i)], dtype=torch.float32, device=device)
            pred, _ = model(xb)
            loss = (pred - y_tensor[int(bag_i)]).pow(2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu().item()))
        train_losses.append(float(np.mean(losses)) if losses else np.nan)
        if val_idx is not None and len(val_idx):
            _, val_metrics = evaluate_attention_model(model, bags, y, np.asarray(val_idx, dtype=np.int64), device)
            val_mse = float(val_metrics["mse"])
            if np.isfinite(val_mse) and val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = args.mil_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_pred, train_metrics = evaluate_attention_model(model, bags, y, np.asarray(train_idx, dtype=np.int64), device)
    final_val_pred = None
    val_metrics: dict[str, Any] = {}
    if val_idx is not None and len(val_idx):
        final_val_pred, val_metrics = evaluate_attention_model(model, bags, y, np.asarray(val_idx, dtype=np.int64), device)
    metrics = {
        "epochs_requested": int(epochs),
        "epochs_run": int(len(train_losses)),
        "best_epoch": int(best_epoch or len(train_losses)),
        "final_train_loss": float(train_losses[-1]) if train_losses else np.nan,
        "best_val_mse": float(best_val_mse) if np.isfinite(best_val_mse) else np.nan,
        "train_mse": train_metrics["mse"],
        "train_pearson": train_metrics["pearson"],
        "train_spearman": train_metrics["spearman"],
        "train_r2": train_metrics["r2"],
        "val_mse": val_metrics.get("mse", np.nan),
        "val_pearson": val_metrics.get("pearson", np.nan),
        "val_spearman": val_metrics.get("spearman", np.nan),
        "val_r2": val_metrics.get("r2", np.nan),
        "train_predictions": train_pred,
    }
    return model, metrics, final_val_pred


@dataclass
class AttentionMILTileScorer:
    scaler: StandardScaler
    model: Any
    device: Any
    embedding_dim: int
    method: str = "attention_mil_regression_tile_attention"
    score_name: str = "attention_weight"

    def score(self, embeddings: np.ndarray) -> tuple[np.ndarray, float]:
        torch, _ = torch_import()
        x = scale_embeddings(embeddings, self.scaler)
        self.model.eval()
        with torch.no_grad():
            xb = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            pred, weights = self.model(xb)
            scores = weights.detach().cpu().numpy().astype(np.float32)
            slide_pred = float(pred.detach().cpu().item())
        return scores, slide_pred


def train_attention_mil(
    root: Path,
    manifest: pd.DataFrame,
    slide_meta: pd.DataFrame,
    emb_cols: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], StandardScaler, AttentionMILTileScorer, pd.DataFrame, pd.DataFrame]:
    device, device_name = resolve_torch_device(args.mil_device)
    raw_bags, y, bag_table, preload_summary = preload_mil_training_bags(root, manifest, args)
    input_dim = int(raw_bags[0].shape[1])
    groups = bag_table["patient_id"].to_numpy(dtype=str)
    unique_groups = np.unique(groups)
    n_splits = max(2, min(args.folds, len(unique_groups)))
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    oof_pred = np.full(len(y), np.nan, dtype=np.float64)
    cv = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups)):
        train_keys = bag_table.iloc[train_idx][["cohort", "patient_id", "slide_candidate_key"]]
        train_slide_meta = slide_meta.merge(train_keys, on=["cohort", "patient_id", "slide_candidate_key"], how="inner")
        fold_scaler = StandardScaler()
        fold_scaler.fit(train_slide_meta[emb_cols].to_numpy(dtype=np.float32))
        scaled_bags = make_scaled_bags(raw_bags, fold_scaler)
        fold_model, fold_metrics, fold_pred = fit_attention_model(
            scaled_bags,
            y,
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
            input_dim,
            args,
            device,
            args.mil_cv_epochs,
            seed_offset=fold,
        )
        del fold_model
        if fold_pred is not None:
            oof_pred[test_idx] = fold_pred
        row = {
            "fold": int(fold),
            "n_train_slides": int(len(train_idx)),
            "n_test_slides": int(len(test_idx)),
            "n_train_patients": int(len(np.unique(groups[train_idx]))),
            "n_test_patients": int(len(np.unique(groups[test_idx]))),
        }
        row.update({key: value for key, value in fold_metrics.items() if key != "train_predictions"})
        fold_rows.append(row)
        for local_i, pred in zip(test_idx, fold_pred if fold_pred is not None else np.full(len(test_idx), np.nan)):
            meta = bag_table.iloc[int(local_i)].to_dict()
            pred_rows.append(
                {
                    "fold": int(fold),
                    "cohort": meta["cohort"],
                    "patient_id": meta["patient_id"],
                    "slide_candidate_key": meta["slide_candidate_key"],
                    "slide_id": meta["slide_id"],
                    "target_score_z": float(y[int(local_i)]),
                    "attention_mil_oof_pred": float(pred),
                    "sampled_tiles": int(meta["sampled_tiles"]),
                    "available_tiles": int(meta["available_tiles"]),
                }
            )
    finite = np.isfinite(oof_pred) & np.isfinite(y)
    if finite.sum() >= 2:
        oof_metrics = {
            "oof_mse": float(mean_squared_error(y[finite], oof_pred[finite])),
            "oof_pearson": float(stats.pearsonr(y[finite], oof_pred[finite]).statistic),
            "oof_spearman": float(stats.spearmanr(y[finite], oof_pred[finite]).statistic),
            "oof_r2": float(r2_score(y[finite], oof_pred[finite])),
        }
    else:
        oof_metrics = {"oof_mse": np.nan, "oof_pearson": np.nan, "oof_spearman": np.nan, "oof_r2": np.nan}

    final_scaler = StandardScaler()
    final_scaler.fit(slide_meta[emb_cols].to_numpy(dtype=np.float32))
    final_bags = make_scaled_bags(raw_bags, final_scaler)
    all_idx = np.arange(len(final_bags), dtype=np.int64)
    final_model, final_metrics, _ = fit_attention_model(
        final_bags,
        y,
        all_idx,
        None,
        input_dim,
        args,
        device,
        args.mil_epochs,
        seed_offset=999,
    )
    scorer = AttentionMILTileScorer(final_scaler, final_model, device, input_dim)
    metrics = {
        "method": "attention_mil_regression_over_tile_embeddings",
        "device": device_name,
        "input_dim": input_dim,
        "hidden_dim": int(args.mil_hidden_dim),
        "attention_dim": int(args.mil_attention_dim),
        "dropout": float(args.mil_dropout),
        "learning_rate": float(args.mil_lr),
        "weight_decay": float(args.mil_weight_decay),
        "folds": int(n_splits),
        **preload_summary,
        **oof_metrics,
        "final_train_mse": final_metrics["train_mse"],
        "final_train_pearson": final_metrics["train_pearson"],
        "final_train_spearman": final_metrics["train_spearman"],
        "final_train_r2": final_metrics["train_r2"],
        "final_epochs_run": final_metrics["epochs_run"],
    }
    return metrics, final_scaler, scorer, pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    if len(values) == 1:
        ranks[0] = 1.0
        return ranks
    ranks[order] = np.arange(len(values), dtype=np.float32) / float(len(values) - 1)
    return ranks


def nearest_control_indices(scores: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(scores) <= n:
        return np.arange(len(scores), dtype=np.int64)
    cutoff = np.nanmedian(scores)
    pool = np.flatnonzero(scores <= cutoff)
    if len(pool) < n:
        pool = np.arange(len(scores), dtype=np.int64)
    return np.sort(rng.choice(pool, size=n, replace=False).astype(np.int64))


def collect_scores_and_tiles(
    root: Path,
    manifest: pd.DataFrame,
    slide_meta: pd.DataFrame,
    scorer: Any,
    target: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    out_h5 = root / args.output_dir / "tile_attention_scores.h5"
    tmp_h5 = out_h5.with_suffix(".tmp.h5")
    if tmp_h5.exists():
        tmp_h5.unlink()
    rng = np.random.default_rng(RANDOM_SEED + 900)
    top_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    top_embeddings: list[np.ndarray] = []
    control_embeddings: list[np.ndarray] = []
    total_tiles = 0
    processed_slides = 0
    target_key = target["target_key"]
    slide_lookup = slide_meta.set_index(["cohort", "patient_id", "slide_candidate_key"])
    with h5py.File(tmp_h5, "w") as out:
        out.attrs["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        out.attrs["method"] = scorer.method
        out.attrs["score_name"] = scorer.score_name
        out.attrs["target_archetype"] = target["target_archetype"]
        out.attrs["target_status"] = target["status"]
        out.attrs["stage8_summary"] = target["stage8_summary"]
        target_group = out.create_group(target_key)
        for row_idx, row in manifest.reset_index(drop=True).iterrows():
            if args.max_slides and row_idx >= args.max_slides:
                break
            path = root / str(row["embedding_path"])
            if not path.exists():
                continue
            embeddings, coords, attrs = read_h5_embeddings(path)
            if args.max_tiles_per_slide and len(embeddings) > args.max_tiles_per_slide:
                embeddings = embeddings[: args.max_tiles_per_slide]
                coords = coords[: args.max_tiles_per_slide]
            scores, slide_pred = scorer.score(embeddings)
            scores = scores.astype(np.float32)
            if len(scores) == 0:
                continue
            score_mean = float(np.mean(scores))
            score_sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else 1.0
            if not np.isfinite(score_sd) or score_sd <= 1e-8:
                score_sd = 1.0
            score_z = ((scores - score_mean) / score_sd).astype(np.float32)
            pct = rank_percentiles(scores)
            slide_name = safe_component(f"{row['cohort']}__{row['patient_id']}__{row['slide_candidate_key']}")
            group = target_group.create_group(slide_name)
            group.create_dataset("coords_x_y_tissue", data=coords, compression="gzip")
            group.create_dataset("attention_score_raw", data=scores, compression="gzip")
            group.create_dataset("attention_score_z", data=score_z, compression="gzip")
            group.create_dataset("attention_score_percentile", data=pct, compression="gzip")
            group.create_dataset("contribution_raw", data=scores, compression="gzip")
            group.create_dataset("contribution_z", data=score_z, compression="gzip")
            group.create_dataset("contribution_percentile", data=pct, compression="gzip")
            for key in ["cohort", "patient_id", "slide_candidate_key", "slide_id", "embedding_path", "tile_coords_path"]:
                group.attrs[key] = str(row.get(key, ""))
            for key in ["dicom_level_file", "overview_file", "tile_native_px", "native_mpp", "target_mpp", "model_repo"]:
                if key in attrs:
                    group.attrs[key] = attrs[key]
            group.attrs["tile_count"] = int(len(scores))
            group.attrs["score_mean"] = score_mean
            group.attrs["score_sd"] = score_sd
            group.attrs["slide_prediction"] = slide_pred
            total_tiles += int(len(scores))
            processed_slides += 1

            key_tuple = (str(row["cohort"]), str(row["patient_id"]), str(row["slide_candidate_key"]))
            clinical = slide_lookup.loc[key_tuple].to_dict() if key_tuple in slide_lookup.index else {}
            n_top = min(args.top_tiles_per_slide, len(scores))
            top_idx = np.argsort(scores)[-n_top:][::-1].astype(np.int64)
            control_idx = nearest_control_indices(scores, n_top, rng)
            top_embeddings.append(embeddings[top_idx].astype(np.float32))
            control_embeddings.append(embeddings[control_idx].astype(np.float32))
            for rank, tile_idx in enumerate(top_idx, start=1):
                top_rows.append(
                    tile_record(row, clinical, coords, scores, score_z, pct, tile_idx, rank, "top_attention")
                )
            for rank, tile_idx in enumerate(control_idx, start=1):
                control_rows.append(
                    tile_record(row, clinical, coords, scores, score_z, pct, tile_idx, rank, "random_tissue_control")
                )
            if args.progress_every_slides and processed_slides % args.progress_every_slides == 0:
                print(json.dumps({"status": "SCORED", "slides": processed_slides, "tiles": total_tiles}, ensure_ascii=False), flush=True)
    tmp_h5.replace(out_h5)
    top_table = pd.DataFrame(top_rows)
    control_table = pd.DataFrame(control_rows)
    top_matrix = np.vstack(top_embeddings) if top_embeddings else np.zeros((0, scorer.embedding_dim), dtype=np.float32)
    control_matrix = np.vstack(control_embeddings) if control_embeddings else np.zeros((0, scorer.embedding_dim), dtype=np.float32)
    score_summary = {
        "processed_slides": processed_slides,
        "total_tiles": total_tiles,
        "top_tiles": int(len(top_table)),
        "control_tiles": int(len(control_table)),
        "method": scorer.method,
        "score_name": scorer.score_name,
        "tile_attention_scores_h5": str(out_h5.relative_to(root)),
    }
    return top_table, control_table, top_matrix, control_matrix, score_summary


def tile_record(
    row: pd.Series,
    clinical: dict[str, Any],
    coords: np.ndarray,
    scores: np.ndarray,
    score_z: np.ndarray,
    pct: np.ndarray,
    tile_idx: int,
    rank: int,
    selection_type: str,
) -> dict[str, Any]:
    coord = coords[int(tile_idx)]
    return {
        "target": clinical.get("target_archetype", ""),
        "cohort": str(row["cohort"]),
        "patient_id": str(row["patient_id"]),
        "slide_candidate_key": str(row["slide_candidate_key"]),
        "slide_id": str(row.get("slide_id", "")),
        "selection_type": selection_type,
        "selection_rank_in_slide": int(rank),
        "tile_index": int(tile_idx),
        "x": float(coord[0]),
        "y": float(coord[1]),
        "tissue_occupancy": float(coord[2]) if coord.shape[0] > 2 else np.nan,
        "contribution_raw": float(scores[int(tile_idx)]),
        "contribution_z": float(score_z[int(tile_idx)]),
        "contribution_percentile": float(pct[int(tile_idx)]),
        "target_score": clinical.get("target_score", np.nan),
        "target_score_z": clinical.get("target_score_z", np.nan),
        "discordance_risk_z": clinical.get("discordance_risk_z", np.nan),
        "hidden_aggressive_group": clinical.get("hidden_aggressive_group", ""),
        "morph_group": clinical.get("morph_group", ""),
        "discordance_group": clinical.get("discordance_group", ""),
        "embedding_path": str(row.get("embedding_path", "")),
        "tile_coords_path": str(row.get("tile_coords_path", "")),
    }


def cluster_tiles(
    top_table: pd.DataFrame,
    control_table: pd.DataFrame,
    top_matrix: np.ndarray,
    control_matrix: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], PCA, KMeans, StandardScaler]:
    if len(top_table) == 0:
        raise RuntimeError("No top attention tiles available for motif clustering")
    max_n = min(args.max_cluster_tiles, len(top_table))
    if len(top_table) > max_n:
        order = np.argsort(top_table["contribution_raw"].to_numpy(dtype=float))[-max_n:]
        order = np.sort(order)
        fit_table = top_table.iloc[order].reset_index(drop=True)
        fit_matrix = top_matrix[order]
    else:
        fit_table = top_table.reset_index(drop=True)
        fit_matrix = top_matrix
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(fit_matrix)
    n_pca = max(2, min(args.pca_components, x_scaled.shape[1], x_scaled.shape[0] - 1))
    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    x_pca = pca.fit_transform(x_scaled)
    n_clusters = max(2, min(args.motif_clusters, max(2, x_pca.shape[0] // 20)))
    labels_by_seed = []
    inertias = []
    for seed_offset in range(args.cluster_seed_repeats):
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_SEED + seed_offset)
        labels_by_seed.append(km.fit_predict(x_pca))
        inertias.append(float(km.inertia_))
    ari_values = [
        float(adjusted_rand_score(labels_by_seed[0], labels))
        for labels in labels_by_seed[1:]
    ]
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(x_pca)
    distances = np.linalg.norm(x_pca - kmeans.cluster_centers_[labels], axis=1)
    fit_table = fit_table.copy()
    fit_table["motif_cluster"] = [f"M{int(label) + 1:02d}" for label in labels]
    fit_table["cluster_distance"] = distances
    fit_table["pca_components"] = n_pca

    if len(control_table) and len(control_matrix):
        c_scaled = scaler.transform(control_matrix)
        c_pca = pca.transform(c_scaled)
        control_labels = kmeans.predict(c_pca)
        control_table = control_table.copy()
        control_table["motif_cluster"] = [f"M{int(label) + 1:02d}" for label in control_labels]
    else:
        control_table = control_table.copy()
        control_table["motif_cluster"] = ""

    summary_rows = []
    for cluster, sub in fit_table.groupby("motif_cluster"):
        summary_rows.append(
            {
                "motif_cluster": cluster,
                "n_tiles": int(len(sub)),
                "n_patients": int(sub["patient_id"].nunique()),
                "n_slides": int(sub[["cohort", "patient_id", "slide_candidate_key"]].drop_duplicates().shape[0]),
                "mean_contribution_raw": float(sub["contribution_raw"].mean()),
                "median_contribution_percentile": float(sub["contribution_percentile"].median()),
                "mean_tissue_occupancy": float(sub["tissue_occupancy"].mean()),
                "hidden_aggressive_tiles": int((sub["hidden_aggressive_group"] == "hidden_aggressive").sum()),
                "true_low_risk_tiles": int((sub["hidden_aggressive_group"] == "true_low_risk").sum()),
            }
        )
    cluster_summary = pd.DataFrame(summary_rows).sort_values("motif_cluster")
    stability = {
        "n_clusters": int(n_clusters),
        "pca_components": int(n_pca),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "seed_repeats": int(args.cluster_seed_repeats),
        "mean_adjusted_rand_to_seed0": float(np.mean(ari_values)) if ari_values else 1.0,
        "min_adjusted_rand_to_seed0": float(np.min(ari_values)) if ari_values else 1.0,
        "inertia_mean": float(np.mean(inertias)),
        "inertia_sd": float(np.std(inertias, ddof=1)) if len(inertias) > 1 else 0.0,
    }
    return fit_table, control_table, cluster_summary, stability, pca, kmeans, scaler


def motif_burden(top_table: pd.DataFrame, cluster_summary: pd.DataFrame, group_table: pd.DataFrame) -> pd.DataFrame:
    clusters = sorted(cluster_summary["motif_cluster"].tolist())
    patients = top_table[["cohort", "patient_id"]].drop_duplicates()
    rows = []
    for _, patient in patients.iterrows():
        sub = top_table[(top_table["cohort"] == patient["cohort"]) & (top_table["patient_id"] == patient["patient_id"])]
        total = max(1, len(sub))
        meta = group_table[(group_table["cohort"] == patient["cohort"]) & (group_table["patient_id"] == patient["patient_id"])]
        meta_row = meta.iloc[0].to_dict() if len(meta) else {}
        for cluster in clusters:
            count = int((sub["motif_cluster"] == cluster).sum())
            rows.append(
                {
                    "cohort": patient["cohort"],
                    "patient_id": patient["patient_id"],
                    "motif_cluster": cluster,
                    "selected_top_tile_count": int(total),
                    "motif_tile_count": count,
                    "motif_burden_fraction": count / float(total),
                    "hidden_aggressive_group": meta_row.get("hidden_aggressive_group", ""),
                    "morph_group": meta_row.get("morph_group", ""),
                    "discordance_group": meta_row.get("discordance_group", ""),
                    "discordance_risk_z": meta_row.get("discordance_risk_z", np.nan),
                }
            )
    return pd.DataFrame(rows)


def motif_enrichment(burden: pd.DataFrame, top_table: pd.DataFrame, control_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cluster, sub in burden.groupby("motif_cluster"):
        hidden = pd.to_numeric(sub.loc[sub["hidden_aggressive_group"] == "hidden_aggressive", "motif_burden_fraction"], errors="coerce").dropna()
        low = pd.to_numeric(sub.loc[sub["hidden_aggressive_group"] == "true_low_risk", "motif_burden_fraction"], errors="coerce").dropna()
        if len(hidden) and len(low):
            stat = stats.mannwhitneyu(hidden, low, alternative="two-sided")
            p = float(stat.pvalue)
            delta = float(hidden.mean() - low.mean())
        else:
            p = np.nan
            delta = np.nan
        rows.append(
            {
                "motif_cluster": cluster,
                "comparison": "hidden_aggressive_vs_true_low_risk_sample_burden",
                "hidden_n": int(len(hidden)),
                "true_low_n": int(len(low)),
                "hidden_mean_burden": float(hidden.mean()) if len(hidden) else np.nan,
                "true_low_mean_burden": float(low.mean()) if len(low) else np.nan,
                "mean_delta": delta,
                "p": p,
            }
        )
    sample_enrichment = pd.DataFrame(rows)
    if not sample_enrichment.empty:
        sample_enrichment["q"] = bh_adjust(sample_enrichment["p"].tolist())

    top_counts = top_table["motif_cluster"].value_counts().to_dict()
    control_counts = control_table["motif_cluster"].value_counts().to_dict()
    all_clusters = sorted(set(top_counts) | set(control_counts))
    tile_rows = []
    for cluster in all_clusters:
        top_c = int(top_counts.get(cluster, 0))
        control_c = int(control_counts.get(cluster, 0))
        table = [[top_c, max(0, len(top_table) - top_c)], [control_c, max(0, len(control_table) - control_c)]]
        try:
            odds, p = stats.fisher_exact(table)
        except Exception:
            odds, p = np.nan, np.nan
        tile_rows.append(
            {
                "motif_cluster": cluster,
                "comparison": "top_attention_vs_random_tissue_tiles",
                "top_tiles": top_c,
                "top_other": max(0, len(top_table) - top_c),
                "control_tiles": control_c,
                "control_other": max(0, len(control_table) - control_c),
                "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            }
        )
    tile_enrichment = pd.DataFrame(tile_rows)
    if not tile_enrichment.empty:
        tile_enrichment["q"] = bh_adjust(tile_enrichment["p"].tolist())
    return sample_enrichment, tile_enrichment


def read_tile_from_dataset(
    ds: pydicom.Dataset,
    x: int,
    y: int,
    tile_native_px: int,
    frame_w: int,
    frame_h: int,
    total_cols: int,
    total_rows: int,
    frames_per_row: int,
) -> Image.Image:
    x_end = min(x + tile_native_px, total_cols)
    y_end = min(y + tile_native_px, total_rows)
    fx0 = x // frame_w
    fy0 = y // frame_h
    fx1 = math.ceil(x_end / frame_w)
    fy1 = math.ceil(y_end / frame_h)
    canvas = np.full((fy1 * frame_h - fy0 * frame_h, fx1 * frame_w - fx0 * frame_w, 3), 255, dtype=np.uint8)
    for fy in range(fy0, fy1):
        for fx in range(fx0, fx1):
            frame_index = fy * frames_per_row + fx
            arr = pixel_array(ds, index=frame_index)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            arr = arr[:, :, :3].astype(np.uint8)
            yy = (fy - fy0) * frame_h
            xx = (fx - fx0) * frame_w
            canvas[yy : yy + arr.shape[0], xx : xx + arr.shape[1], :] = arr
    crop = canvas[y - fy0 * frame_h : y_end - fy0 * frame_h, x - fx0 * frame_w : x_end - fx0 * frame_w, :]
    image = Image.fromarray(crop)
    if image.size != (TILE_SIZE, TILE_SIZE):
        image = image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR)
    return image


def image_artifact_metrics(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    gy, gx = np.gradient(gray)
    return {
        "white_fraction": float(np.mean((arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240))),
        "dark_fraction": float(np.mean(gray < 20)),
        "saturation_mean": float(np.mean(saturation)),
        "gradient_variance": float(np.var(gx) + np.var(gy)),
    }


def image_cell_proxy_metrics(image: Image.Image) -> dict[str, float | int | str]:
    """Estimate conservative image-level cellularity proxies from exported H&E tiles.

    This is not a HoVer-Net replacement and does not classify cell types. It gives
    a reproducible nuclei-candidate and stain proxy for every exported tile so the
    spatial catalog has a real cell-layer sanity signal instead of only embeddings.
    """
    arr_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = arr_u8.astype(np.float32)
    gray = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    tissue_mask = ((gray < 235) & (saturation > 5)) | (gray < 210)
    tissue_pixels = int(tissue_mask.sum())
    total_pixels = int(tissue_mask.size)
    if tissue_pixels == 0:
        return {
            "cell_proxy_status": "WARN",
            "cell_proxy_reason": "no_tissue_proxy_pixels",
            "tissue_proxy_area_fraction": 0.0,
            "nuclei_candidate_count": 0,
            "nuclei_density_per_10k_tissue_px": 0.0,
            "nuclei_area_fraction_of_tissue": 0.0,
            "hematoxylin_od_mean": np.nan,
            "eosin_od_mean": np.nan,
            "hematoxylin_eosin_ratio": np.nan,
            "hematoxylin_dominant_fraction": 0.0,
            "eosin_dominant_fraction": 0.0,
            "low_cellularity_tissue_fraction": 1.0,
        }

    rgb01 = np.clip(arr_u8.astype(np.float32) / 255.0, 1e-6, 1.0)
    od = -np.log(rgb01)
    r_od, g_od, b_od = od[:, :, 0], od[:, :, 1], od[:, :, 2]
    # Dependency-free H&E proxies. Hematoxylin-rich nuclei are blue/purple
    # optical-density dominant; eosin-rich tissue is red/green dominant.
    hematoxylin = np.maximum(0.0, b_od - 0.35 * r_od - 0.15 * g_od)
    eosin = np.maximum(0.0, 0.55 * r_od + 0.45 * g_od - 0.20 * b_od)
    h_tissue = hematoxylin[tissue_mask]
    e_tissue = eosin[tissue_mask]
    gray_tissue = gray[tissue_mask]
    sat_tissue = saturation[tissue_mask]
    h_threshold = max(float(np.nanpercentile(h_tissue, 75)), float(np.nanmean(h_tissue) + 0.25 * np.nanstd(h_tissue)))
    gray_threshold = min(185.0, float(np.nanpercentile(gray_tissue, 45)))
    nuclei_mask = tissue_mask & (hematoxylin > h_threshold) & (gray < gray_threshold)
    if int(nuclei_mask.sum()) < max(10, int(0.002 * tissue_pixels)):
        nuclei_mask = tissue_mask & (gray < min(170.0, float(np.nanpercentile(gray_tissue, 35)))) & (saturation > max(12.0, float(np.nanpercentile(sat_tissue, 40))))
    mask = ndimage.binary_opening(nuclei_mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labels, n_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if n_labels > 0:
        areas = np.asarray(ndimage.sum(mask, labels, index=np.arange(1, n_labels + 1)), dtype=np.int64)
        valid = (areas >= CELL_PROXY_MIN_NUCLEUS_AREA) & (areas <= CELL_PROXY_MAX_NUCLEUS_AREA)
        nuclei_count = int(valid.sum())
        nuclei_area = int(areas[valid].sum())
    else:
        nuclei_count = 0
        nuclei_area = 0
    h_median = float(np.nanmedian(h_tissue))
    e_p60 = float(np.nanpercentile(e_tissue, 60))
    hematoxylin_dominant = tissue_mask & (hematoxylin > h_threshold)
    eosin_dominant = tissue_mask & (eosin > e_p60) & (hematoxylin < h_median)
    tissue_fraction = tissue_pixels / float(total_pixels)
    nuclei_area_fraction = nuclei_area / float(tissue_pixels)
    status = "PASS"
    reason = ""
    if tissue_fraction < 0.25:
        status = "WARN"
        reason = "low_tissue_proxy_fraction"
    elif nuclei_count < 3 and nuclei_area_fraction < 0.005:
        status = "WARN"
        reason = "low_nuclei_candidate_signal"
    return {
        "cell_proxy_status": status,
        "cell_proxy_reason": reason,
        "tissue_proxy_area_fraction": float(tissue_fraction),
        "nuclei_candidate_count": nuclei_count,
        "nuclei_density_per_10k_tissue_px": float(nuclei_count / float(tissue_pixels) * 10000.0),
        "nuclei_area_fraction_of_tissue": float(nuclei_area_fraction),
        "hematoxylin_od_mean": float(np.nanmean(h_tissue)),
        "eosin_od_mean": float(np.nanmean(e_tissue)),
        "hematoxylin_eosin_ratio": float((np.nanmean(h_tissue) + 1e-6) / (np.nanmean(e_tissue) + 1e-6)),
        "hematoxylin_dominant_fraction": float(hematoxylin_dominant.sum() / float(tissue_pixels)),
        "eosin_dominant_fraction": float(eosin_dominant.sum() / float(tissue_pixels)),
        "low_cellularity_tissue_fraction": float(1.0 - min(1.0, nuclei_area_fraction * 4.0)),
    }


def quantify_catalog_cell_proxy(root: Path, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in catalog.iterrows():
        base = {
            "export_type": row.get("export_type", ""),
            "motif_cluster": row.get("motif_cluster", ""),
            "cohort": row.get("cohort", ""),
            "patient_id": row.get("patient_id", ""),
            "slide_candidate_key": row.get("slide_candidate_key", ""),
            "tile_index": int(row.get("tile_index", -1)),
            "image_path": row.get("image_path", ""),
            "image_status": row.get("image_status", ""),
            "image_status_reason": row.get("image_status_reason", ""),
        }
        try:
            image = Image.open(root / str(row["image_path"]))
            metrics = image_cell_proxy_metrics(image)
        except Exception as exc:  # noqa: BLE001
            metrics = {
                "cell_proxy_status": "FAILED",
                "cell_proxy_reason": f"{type(exc).__name__}: {exc}",
                "tissue_proxy_area_fraction": np.nan,
                "nuclei_candidate_count": np.nan,
                "nuclei_density_per_10k_tissue_px": np.nan,
                "nuclei_area_fraction_of_tissue": np.nan,
                "hematoxylin_od_mean": np.nan,
                "eosin_od_mean": np.nan,
                "hematoxylin_eosin_ratio": np.nan,
                "hematoxylin_dominant_fraction": np.nan,
                "eosin_dominant_fraction": np.nan,
                "low_cellularity_tissue_fraction": np.nan,
            }
        rows.append({**base, **metrics})
    cell_proxy = pd.DataFrame(rows)
    motif_rows: list[dict[str, Any]] = []
    motif_proxy = cell_proxy[cell_proxy["export_type"] == "motif_representative"].copy() if not cell_proxy.empty else pd.DataFrame()
    if not motif_proxy.empty:
        for cluster, sub in motif_proxy.groupby("motif_cluster"):
            usable = sub[sub["cell_proxy_status"] != "FAILED"]
            motif_rows.append(
                {
                    "motif_cluster": cluster,
                    "representative_tiles": int(len(sub)),
                    "image_pass_tiles": int((sub["image_status"] == "PASS").sum()),
                    "image_warn_tiles": int((sub["image_status"] == "WARN").sum()),
                    "cell_proxy_pass_tiles": int((sub["cell_proxy_status"] == "PASS").sum()),
                    "cell_proxy_warn_tiles": int((sub["cell_proxy_status"] == "WARN").sum()),
                    "cell_proxy_failed_tiles": int((sub["cell_proxy_status"] == "FAILED").sum()),
                    "mean_tissue_proxy_area_fraction": float(pd.to_numeric(usable["tissue_proxy_area_fraction"], errors="coerce").mean()) if len(usable) else np.nan,
                    "mean_nuclei_candidate_count": float(pd.to_numeric(usable["nuclei_candidate_count"], errors="coerce").mean()) if len(usable) else np.nan,
                    "mean_nuclei_density_per_10k_tissue_px": float(pd.to_numeric(usable["nuclei_density_per_10k_tissue_px"], errors="coerce").mean()) if len(usable) else np.nan,
                    "mean_nuclei_area_fraction_of_tissue": float(pd.to_numeric(usable["nuclei_area_fraction_of_tissue"], errors="coerce").mean()) if len(usable) else np.nan,
                    "mean_hematoxylin_dominant_fraction": float(pd.to_numeric(usable["hematoxylin_dominant_fraction"], errors="coerce").mean()) if len(usable) else np.nan,
                    "mean_eosin_dominant_fraction": float(pd.to_numeric(usable["eosin_dominant_fraction"], errors="coerce").mean()) if len(usable) else np.nan,
                    "artifact_like_motif": bool((sub["image_status"] == "PASS").sum() == 0 or (sub["cell_proxy_status"] == "PASS").sum() == 0),
                }
            )
    motif_summary = pd.DataFrame(motif_rows).sort_values("motif_cluster") if motif_rows else pd.DataFrame()
    summary = {
        "catalog_rows": int(len(cell_proxy)),
        "cell_proxy_success": int((cell_proxy["cell_proxy_status"] != "FAILED").sum()) if not cell_proxy.empty else 0,
        "cell_proxy_failed": int((cell_proxy["cell_proxy_status"] == "FAILED").sum()) if not cell_proxy.empty else 0,
        "motif_representative_rows": int(len(motif_proxy)),
        "motif_representative_cell_proxy_pass": int((motif_proxy["cell_proxy_status"] == "PASS").sum()) if not motif_proxy.empty else 0,
        "motif_representative_cell_proxy_warn": int((motif_proxy["cell_proxy_status"] == "WARN").sum()) if not motif_proxy.empty else 0,
        "artifact_like_motifs": sorted(motif_summary.loc[motif_summary["artifact_like_motif"], "motif_cluster"].tolist()) if not motif_summary.empty else [],
        "method": "rgb_to_hed_hematoxylin_proxy_connected_components; not HoVer-Net and no cell-type classification",
    }
    return cell_proxy, motif_summary, summary


def export_tile_catalog(
    root: Path,
    top_table: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    target_key: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    catalog_dir = root / args.output_dir / "representative_tile_catalog" / target_key
    if catalog_dir.exists():
        shutil.rmtree(catalog_dir)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    export_rows = []
    failures = 0
    ds_cache: dict[str, tuple[pydicom.Dataset, int, int, int, int, int]] = {}

    def prepare_image(row: pd.Series) -> tuple[Image.Image, dict[str, float], str, str]:
        rel_embedding = str(row["embedding_path"])
        image_status = "PASS"
        reason = ""
        with h5py.File(root / rel_embedding, "r") as handle:
            level_file = root / str(handle.attrs["dicom_level_file"])
            tile_native_px = int(handle.attrs["tile_native_px"])
        cache_key = str(level_file)
        if cache_key not in ds_cache:
            ds = pydicom.dcmread(str(level_file))
            frame_h = int(ds.Rows)
            frame_w = int(ds.Columns)
            total_cols = int(ds.TotalPixelMatrixColumns)
            total_rows = int(ds.TotalPixelMatrixRows)
            frames_per_row = math.ceil(total_cols / frame_w)
            ds_cache[cache_key] = (ds, frame_w, frame_h, total_cols, total_rows, frames_per_row)
        ds, frame_w, frame_h, total_cols, total_rows, frames_per_row = ds_cache[cache_key]
        image = read_tile_from_dataset(
            ds,
            int(float(row["x"])),
            int(float(row["y"])),
            tile_native_px,
            frame_w,
            frame_h,
            total_cols,
            total_rows,
            frames_per_row,
        )
        metrics = image_artifact_metrics(image)
        if float(row.get("tissue_occupancy", 0.0)) < 0.70 or metrics["white_fraction"] > 0.75 or metrics["saturation_mean"] < 3.0:
            image_status = "WARN"
            reason = "artifact_proxy_warn"
        return image, metrics, image_status, reason

    def save_record(
        serial: int,
        cluster: str,
        export_type: str,
        row: pd.Series,
        image: Image.Image | None,
        metrics: dict[str, float],
        image_status: str,
        reason: str,
    ) -> dict[str, Any]:
        out_subdir = catalog_dir / ("top100_tiles" if export_type == "top100_review" else safe_component(cluster))
        out_subdir.mkdir(parents=True, exist_ok=True)
        name = f"{serial:04d}_{safe_component(str(row['cohort']))}_{safe_component(str(row['patient_id']))}_{safe_component(str(row['slide_candidate_key']))}_tile{int(row['tile_index']):06d}.png"
        out_path = out_subdir / name
        if image is not None:
            image.save(out_path)
        export_rows.append(
            {
                "export_type": export_type,
                "motif_cluster": cluster,
                "cohort": row["cohort"],
                "patient_id": row["patient_id"],
                "slide_candidate_key": row["slide_candidate_key"],
                "tile_index": int(row["tile_index"]),
                "x": row["x"],
                "y": row["y"],
                "tissue_occupancy": row["tissue_occupancy"],
                "contribution_raw": row["contribution_raw"],
                "contribution_percentile": row["contribution_percentile"],
                "image_path": str(out_path.relative_to(root)),
                "image_status": image_status,
                "image_status_reason": reason,
                **metrics,
            }
        )
        return export_rows[-1]

    serial = 1
    candidate_limit = max(args.representatives_per_motif, args.representatives_per_motif * args.representative_candidate_multiplier)
    for cluster in sorted(cluster_summary["motif_cluster"].tolist()):
        sub = top_table[top_table["motif_cluster"] == cluster].copy()
        sub = sub.sort_values(["cluster_distance", "contribution_raw"], ascending=[True, False]).head(candidate_limit)
        selected = 0
        fallback: list[tuple[pd.Series, Image.Image, dict[str, float], str, str]] = []
        for _, row in sub.iterrows():
            try:
                image, metrics, image_status, reason = prepare_image(row)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                continue
            if image_status == "PASS" and selected < args.representatives_per_motif:
                save_record(serial, cluster, "motif_representative", row, image, metrics, image_status, reason)
                serial += 1
                selected += 1
            else:
                fallback.append((row, image, metrics, image_status, reason))
            if selected >= args.representatives_per_motif:
                break
        for row, image, metrics, image_status, reason in fallback:
            if selected >= args.representatives_per_motif:
                break
            save_record(serial, cluster, "motif_representative", row, image, metrics, image_status, reason)
            serial += 1
            selected += 1

    global_top = top_table.sort_values("contribution_raw", ascending=False).head(args.top100_review_tiles)
    for _, row in global_top.iterrows():
        metrics = {"white_fraction": np.nan, "dark_fraction": np.nan, "saturation_mean": np.nan, "gradient_variance": np.nan}
        image_status = "PASS"
        reason = ""
        image = None
        try:
            image, metrics, image_status, reason = prepare_image(row)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            image_status = "FAILED"
            reason = f"{type(exc).__name__}: {exc}"
        save_record(serial, "global_top100", "top100_review", row, image, metrics, image_status, reason)
        serial += 1

    catalog = pd.DataFrame(export_rows)
    motif_catalog = catalog[catalog["export_type"] == "motif_representative"] if not catalog.empty else pd.DataFrame()
    summary = {
        "catalog_rows": int(len(catalog)),
        "png_success": int((catalog["image_status"] != "FAILED").sum()) if not catalog.empty else 0,
        "png_failures": int(failures),
        "motif_representative_rows": int(len(motif_catalog)),
        "motif_representative_pass": int((motif_catalog["image_status"] == "PASS").sum()) if not motif_catalog.empty else 0,
        "motif_representative_warn": int((motif_catalog["image_status"] == "WARN").sum()) if not motif_catalog.empty else 0,
        "top100_rows": int((catalog["export_type"] == "top100_review").sum()) if not catalog.empty else 0,
        "top100_proxy_pass_or_warn": int((catalog.loc[catalog["export_type"] == "top100_review", "image_status"] != "FAILED").sum()) if not catalog.empty else 0,
    }
    return catalog, summary


def build_stage9(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / "logs" / "qc").mkdir(parents=True, exist_ok=True)
    target = load_stage8_target(root, args)
    manifest = read_table(root / args.embedding_manifest)
    slide_emb = read_table(root / args.slide_embedding_table)
    arch = read_table(root / args.archetype_scores)
    discordance = read_table(root / args.discordance_risk)
    groups = read_table(root / args.hidden_groups)
    groups = groups[groups["cutoff"] == args.hidden_group_cutoff].copy()
    if target["score_col"] not in arch.columns:
        raise RuntimeError(f"Missing target score column {target['score_col']}")
    arch = arch.copy()
    arch["target_score"] = pd.to_numeric(arch[target["score_col"]], errors="coerce")
    arch["target_score_z"] = cohort_zscore(arch, "target_score")
    arch["target_archetype"] = target["target_archetype"]
    slide_meta = (
        slide_emb.merge(arch[["cohort", "patient_id", "target_archetype", "target_score", "target_score_z"]], on=["cohort", "patient_id"], how="inner")
        .merge(discordance[["cohort", "patient_id", "discordance_risk_z"]], on=["cohort", "patient_id"], how="left")
        .merge(
            groups[["cohort", "patient_id", "hidden_aggressive_group", "morph_group", "discordance_group"]],
            on=["cohort", "patient_id"],
            how="left",
        )
    )
    emb_cols = feature_columns(slide_meta)
    if len(emb_cols) == 0:
        raise RuntimeError("No slide embedding feature columns found")
    slide_meta = slide_meta.dropna(subset=["target_score_z"]).reset_index(drop=True)
    alpha_values = [float(x) for x in args.alphas.split(",") if x.strip()]
    train_metrics, scaler, model, fold_table = train_slide_ridge(
        slide_meta,
        emb_cols,
        "target_score_z",
        slide_meta["patient_id"].to_numpy(dtype=str),
        alpha_values,
        args.folds,
    )
    fold_table.to_csv(out_dir / "stage9_regression_folds.tsv", sep="\t", index=False)
    pd.DataFrame(train_metrics["alpha_scores"]).to_csv(out_dir / "stage9_regression_alpha_scores.tsv", sep="\t", index=False)

    manifest = manifest.merge(
        slide_meta[[
            "cohort",
            "patient_id",
            "slide_candidate_key",
            "target_archetype",
            "target_score",
            "target_score_z",
            "discordance_risk_z",
            "hidden_aggressive_group",
            "morph_group",
            "discordance_group",
        ]],
        on=["cohort", "patient_id", "slide_candidate_key"],
        how="inner",
    )
    if args.scorer == "attention_mil":
        mil_metrics, mil_scaler, scorer, mil_fold_table, mil_oof_table = train_attention_mil(
            root,
            manifest,
            slide_meta,
            emb_cols,
            args,
        )
        mil_fold_table.to_csv(out_dir / "stage9_attention_mil_folds.tsv", sep="\t", index=False)
        mil_oof_table.to_csv(out_dir / "stage9_attention_mil_oof_predictions.tsv", sep="\t", index=False)
        (out_dir / "stage9_attention_mil_metrics.json").write_text(json.dumps(mil_metrics, indent=2, ensure_ascii=False) + "\n")
    elif args.scorer == "ridge":
        mil_metrics = {
            "method": "not_run",
            "reason": "CLI scorer=ridge requested; ridge is retained as fallback/comparator only.",
        }
        scorer = RidgeTileScorer(scaler, model, len(emb_cols), method="slide_level_ridge_linear_per_tile_scoring_fallback")
    else:
        raise RuntimeError(f"Unsupported scorer: {args.scorer}")
    top_table, control_table, top_matrix, control_matrix, score_summary = collect_scores_and_tiles(
        root, manifest, manifest, scorer, target, args
    )
    top_table.to_csv(out_dir / "top_attention_tile_table.tsv", sep="\t", index=False)
    control_table.to_csv(out_dir / "random_control_tile_table.tsv", sep="\t", index=False)

    motif_table, control_table_labeled, cluster_summary, stability, pca, kmeans, cluster_scaler = cluster_tiles(
        top_table, control_table, top_matrix, control_matrix, args
    )
    motif_table.to_csv(out_dir / "motif_cluster_table.tsv", sep="\t", index=False)
    control_table_labeled.to_csv(out_dir / "random_control_tile_cluster_table.tsv", sep="\t", index=False)
    cluster_summary.to_csv(out_dir / "motif_cluster_summary.tsv", sep="\t", index=False)

    burden = motif_burden(motif_table, cluster_summary, groups)
    burden.to_csv(out_dir / "motif_burden_by_sample.tsv", sep="\t", index=False)
    sample_enrichment, tile_enrichment = motif_enrichment(burden, motif_table, control_table_labeled)
    sample_enrichment.to_csv(out_dir / "motif_burden_hidden_aggressive_enrichment.tsv", sep="\t", index=False)
    tile_enrichment.to_csv(out_dir / "motif_top_vs_random_enrichment.tsv", sep="\t", index=False)

    catalog, catalog_summary = export_tile_catalog(root, motif_table, cluster_summary, target["target_key"], args)
    catalog.to_csv(out_dir / "representative_tile_catalog.tsv", sep="\t", index=False)
    cell_proxy, cell_proxy_motif_summary, cell_proxy_summary = quantify_catalog_cell_proxy(root, catalog)
    cell_proxy.to_csv(out_dir / "tile_cell_proxy_quantification.tsv", sep="\t", index=False)
    cell_proxy_motif_summary.to_csv(out_dir / "motif_cell_proxy_summary.tsv", sep="\t", index=False)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, status: str, details: dict[str, Any] | None = None) -> None:
        checks.append({"check": name, "status": status, "details": details or {}})

    add_check("stage8_summary_pass", "PASS", {"stage8_summary": args.stage8_summary, "target_status": target["status"]})
    add_check("strict_adverse_absent_fallback_recorded", "WARN" if target["source"] != "strict_adverse_archetype" else "PASS", target)
    expected_processed_slides = min(int(args.max_slides), len(manifest)) if args.max_slides else len(manifest)
    score_summary["expected_processed_slides"] = int(expected_processed_slides)
    add_check("manifest_rows", "PASS" if len(manifest) > 0 else "FAIL", {"slides": int(len(manifest)), "patients": int(manifest["patient_id"].nunique()), "expected_processed_slides": int(expected_processed_slides)})
    add_check("slide_regression_comparator_completed", "PASS" if np.isfinite(train_metrics["train_pearson"]) else "FAIL", train_metrics)
    add_check(
        "attention_mil_completed",
        "PASS" if args.scorer == "attention_mil" and np.isfinite(mil_metrics.get("oof_pearson", np.nan)) else "WARN",
        mil_metrics,
    )
    add_check("tile_scores_written", "PASS" if score_summary["processed_slides"] == expected_processed_slides and score_summary["total_tiles"] > 0 else "FAIL", score_summary)
    add_check("motif_clusters_written", "PASS" if len(cluster_summary) >= 2 else "FAIL", {"clusters": int(len(cluster_summary)), **stability})
    add_check("cluster_seed_stability", "PASS" if stability["mean_adjusted_rand_to_seed0"] >= args.seed_stability_pass_ari else "WARN", stability)
    top100 = catalog[catalog["export_type"] == "top100_review"] if not catalog.empty else pd.DataFrame()
    top100_nonfailed = int((top100["image_status"] != "FAILED").sum()) if not top100.empty else 0
    top100_pass_fraction = top100_nonfailed / max(1, len(top100))
    add_check("top100_tile_review_exported", "PASS" if len(top100) >= min(args.top100_review_tiles, len(motif_table)) and top100_pass_fraction >= 0.90 else "WARN", {"top100_rows": int(len(top100)), "nonfailed_fraction": top100_pass_fraction})
    high_tissue = pd.to_numeric(motif_table["tissue_occupancy"], errors="coerce")
    add_check("high_attention_tissue_proxy", "PASS" if len(high_tissue.dropna()) > 0 and float((high_tissue >= 0.70).mean()) >= 0.90 else "WARN", {"mean_tissue_occupancy": float(high_tissue.mean()), "fraction_ge_0.70": float((high_tissue >= 0.70).mean())})
    add_check("representative_png_export", "PASS" if catalog_summary["png_success"] > 0 and catalog_summary["png_failures"] == 0 else "WARN", catalog_summary)
    add_check(
        "tile_cell_proxy_quantification",
        "PASS" if cell_proxy_summary["cell_proxy_success"] == cell_proxy_summary["catalog_rows"] and cell_proxy_summary["motif_representative_cell_proxy_pass"] > 0 else "WARN",
        cell_proxy_summary,
    )
    add_check("hovernet_cell_quantification", "WARN", {"status": "not_run", "reason": "HoVer-Net dependency not available in current project env; proposal fallback uses embedding-cluster prototypes plus tissue/artifact proxies."})
    for path in [
        "tile_attention_scores.h5",
        "motif_cluster_table.tsv",
        "motif_burden_by_sample.tsv",
        "representative_tile_catalog.tsv",
        "tile_cell_proxy_quantification.tsv",
        "motif_cell_proxy_summary.tsv",
        "motif_cluster_summary.tsv",
        "motif_burden_hidden_aggressive_enrichment.tsv",
        "motif_top_vs_random_enrichment.tsv",
        "stage9_regression_folds.tsv",
        "stage9_regression_alpha_scores.tsv",
    ]:
        p = out_dir / path
        add_check(f"output_exists::{path}", "PASS" if p.exists() and p.stat().st_size > 0 else "FAIL", {"path": str(p.relative_to(root)), "size": p.stat().st_size if p.exists() else 0})
    if args.scorer == "attention_mil":
        for path in [
            "stage9_attention_mil_folds.tsv",
            "stage9_attention_mil_oof_predictions.tsv",
            "stage9_attention_mil_metrics.json",
        ]:
            p = out_dir / path
            add_check(f"output_exists::{path}", "PASS" if p.exists() and p.stat().st_size > 0 else "FAIL", {"path": str(p.relative_to(root)), "size": p.stat().st_size if p.exists() else 0})

    fail_count = sum(1 for check in checks if check["status"] == "FAIL")
    warn_count = sum(1 for check in checks if check["status"] == "WARN")
    summary = {
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS" if fail_count == 0 else "FAIL",
        "fail_count": fail_count,
        "warn_count": warn_count,
        "method": scorer.method,
        "fallback_reason": "" if args.scorer == "attention_mil" else "CLI scorer=ridge; attention MIL not used in this run.",
        "target": target,
        "inputs": {
            "embedding_manifest": args.embedding_manifest,
            "slide_embedding_table": args.slide_embedding_table,
            "archetype_scores": args.archetype_scores,
            "discordance_risk": args.discordance_risk,
            "hidden_groups": args.hidden_groups,
        },
        "regression_comparator_metrics": train_metrics,
        "attention_mil_metrics": mil_metrics,
        "score_summary": score_summary,
        "cluster_stability": stability,
        "catalog_summary": catalog_summary,
        "cell_proxy_summary": cell_proxy_summary,
        "outputs": {
            "tile_attention_scores_h5": str((out_dir / "tile_attention_scores.h5").relative_to(root)),
            "motif_cluster_table": str((out_dir / "motif_cluster_table.tsv").relative_to(root)),
            "motif_burden_by_sample": str((out_dir / "motif_burden_by_sample.tsv").relative_to(root)),
            "representative_tile_catalog": str((out_dir / "representative_tile_catalog.tsv").relative_to(root)),
            "representative_tile_catalog_dir": str((out_dir / "representative_tile_catalog").relative_to(root)),
            "tile_cell_proxy_quantification": str((out_dir / "tile_cell_proxy_quantification.tsv").relative_to(root)),
            "motif_cell_proxy_summary": str((out_dir / "motif_cell_proxy_summary.tsv").relative_to(root)),
            "attention_mil_metrics": str((out_dir / "stage9_attention_mil_metrics.json").relative_to(root)) if args.scorer == "attention_mil" else "",
            "attention_mil_folds": str((out_dir / "stage9_attention_mil_folds.tsv").relative_to(root)) if args.scorer == "attention_mil" else "",
            "attention_mil_oof_predictions": str((out_dir / "stage9_attention_mil_oof_predictions.tsv").relative_to(root)) if args.scorer == "attention_mil" else "",
        },
        "checks": checks,
        "notes": [
            "A2 is a non-significant directional proxy from Stage 8, not a strict clinically significant adverse archetype.",
            "Tile scores in tile_attention_scores.h5 are learned attention-MIL weights when scorer=attention_mil; ridge outputs are retained as comparator tables only.",
            "HoVer-Net cell classification was not run; image-level hematoxylin/nuclei candidate cell proxies, embedding motif prototypes and tissue/artifact proxies are retained as the proposal fallback.",
            "Cell proxy quantification uses RGB-to-HED hematoxylin thresholding and connected components on exported tile PNGs; it is not HoVer-Net and does not classify tumor, stromal, or immune cell types.",
            "Top100 tile review here is an exported PNG catalog plus automated tissue/artifact proxy, not manual pathologist annotation.",
        ],
    }
    (out_dir / "stage9_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    qc_json = root / "logs" / "qc" / "stage9_spatial_qc.json"
    qc_tsv = root / "logs" / "qc" / "stage9_spatial_qc_summary.tsv"
    qc_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(checks).to_csv(qc_tsv, sep="\t", index=False)
    print(json.dumps({"status": summary["status"], "fail_count": fail_count, "warn_count": warn_count, "target": target["target_key"], "method": scorer.method, "slides": score_summary["processed_slides"], "tiles": score_summary["total_tiles"]}, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--embedding-manifest", default="data/processed/stage4/embedding_manifest_prov_gigapath_prov_gigapath_all_candidates.tsv")
    parser.add_argument("--slide-embedding-table", default="data/processed/stage4/prov_gigapath_all_candidates_median_slide_mean_embeddings.tsv")
    parser.add_argument("--archetype-scores", default="results/archetypes/sample_archetype_scores.tsv")
    parser.add_argument("--discordance-risk", default="results/clinical/discordance_risk.tsv")
    parser.add_argument("--hidden-groups", default="results/clinical/hidden_aggressive_groups.tsv")
    parser.add_argument("--stage8-summary", default="results/clinical/stage8_build_summary.json")
    parser.add_argument("--output-dir", default="results/spatial")
    parser.add_argument("--target-archetype", default="")
    parser.add_argument("--hidden-group-cutoff", default="top33")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--scorer", choices=["attention_mil", "ridge"], default="attention_mil")
    parser.add_argument("--mil-device", default="auto", help="auto, cpu, cuda, or cuda:N for attention MIL")
    parser.add_argument("--mil-train-max-tiles", type=int, default=512)
    parser.add_argument("--mil-hidden-dim", type=int, default=256)
    parser.add_argument("--mil-attention-dim", type=int, default=128)
    parser.add_argument("--mil-dropout", type=float, default=0.15)
    parser.add_argument("--mil-lr", type=float, default=1e-4)
    parser.add_argument("--mil-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mil-cv-epochs", type=int, default=12)
    parser.add_argument("--mil-epochs", type=int, default=24)
    parser.add_argument("--mil-patience", type=int, default=4)
    parser.add_argument("--top-tiles-per-slide", type=int, default=20)
    parser.add_argument("--motif-clusters", type=int, default=12)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--cluster-seed-repeats", type=int, default=5)
    parser.add_argument("--seed-stability-pass-ari", type=float, default=0.40)
    parser.add_argument("--max-cluster-tiles", type=int, default=12000)
    parser.add_argument("--representatives-per-motif", type=int, default=3)
    parser.add_argument("--representative-candidate-multiplier", type=int, default=10)
    parser.add_argument("--top100-review-tiles", type=int, default=100)
    parser.add_argument("--progress-every-slides", type=int, default=50)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--max-tiles-per-slide", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    build_stage9(parse_args())


if __name__ == "__main__":
    main()
