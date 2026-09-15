#!/usr/bin/env python3
"""Extract real DICOM tile embeddings with an open fallback pathology encoder."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from types import SimpleNamespace
from pathlib import Path
from queue import Empty
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixels import pixel_array


TILE_SIZE = 256
MACENKO_TARGET_STAIN_MATRIX = np.array(
    [
        [0.650, 0.072],
        [0.704, 0.990],
        [0.286, 0.105],
    ],
    dtype=np.float32,
)
MACENKO_TARGET_MAX_C = np.array([1.9705, 1.0308], dtype=np.float32)


def model_slug(repo: str) -> str:
    return repo.replace("/", "_").replace("-", "_")


def encoder_role(repo: str) -> str:
    if repo == "bioptimus/H-optimus-0":
        return "proposal_default_H_Optimus_0_pathology_encoder"
    if repo == "prov-gigapath/prov-gigapath":
        return "proposal_reproducibility_Prov_GigaPath_pathology_encoder"
    return "open_fallback_pathology_encoder_not_H_Optimus_or_Prov_GigaPath"


def timm_model_name(repo: str) -> str:
    if repo == "prov-gigapath/prov-gigapath":
        return "hf_hub:prov-gigapath/prov-gigapath"
    return f"hf-hub:{repo}"


class TimmImageProcessor:
    def __init__(self, transform: Any):
        self.transform = transform

    def __call__(self, images: list[Image.Image], return_tensors: str = "pt") -> dict[str, Any]:
        del return_tensors
        return {"pixel_values": self.transform(images[0]).unsqueeze(0) if len(images) == 1 else self._stack(images)}

    def _stack(self, images: list[Image.Image]) -> Any:
        import torch

        return torch.stack([self.transform(image) for image in images], dim=0)


class TimmModelWrapper:
    def __init__(self, model: Any, hidden_size: int):
        self.model = model
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def to(self, device: Any) -> "TimmModelWrapper":
        self.model.to(device)
        return self

    def eval(self) -> "TimmModelWrapper":
        self.model.eval()
        return self

    def __call__(self, pixel_values: Any) -> Any:
        output = self.model(pixel_values)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if hasattr(output, "last_hidden_state"):
            pooled = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
        else:
            pooled = output
            if getattr(pooled, "ndim", 0) > 2:
                pooled = pooled[:, 0]
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=None)


def build_timm_processor_and_model(repo: str, cache_dir: Path) -> tuple[Any, Any]:
    import timm
    from torchvision import transforms

    if repo == "bioptimus/H-optimus-0":
        transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
            ]
        )
        model = timm.create_model(
            timm_model_name(repo),
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
            cache_dir=str(cache_dir),
        )
        return TimmImageProcessor(transform), TimmModelWrapper(model, hidden_size=1536)
    if repo == "prov-gigapath/prov-gigapath":
        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        model = timm.create_model(timm_model_name(repo), pretrained=True, cache_dir=str(cache_dir))
        return TimmImageProcessor(transform), TimmModelWrapper(model, hidden_size=1536)
    raise ValueError(f"Unsupported timm model repo: {repo}")


def read_tile_from_dataset(
    ds: pydicom.Dataset,
    level_file: Path,
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


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    return -np.log((rgb.astype(np.float32) + 1.0) / 255.0)


def estimate_macenko_fit(images: list[Image.Image]) -> dict[str, np.ndarray] | None:
    if not images:
        return None
    pixels = np.concatenate([np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3) for image in images], axis=0)
    od = rgb_to_od(pixels)
    od = od[np.linalg.norm(od, axis=1) > 0.15]
    if len(od) < 100:
        return None
    try:
        _, _, vh = np.linalg.svd(od, full_matrices=False)
        basis = vh[:2].T
        projected = od @ basis
        phi = np.arctan2(projected[:, 1], projected[:, 0])
        min_phi, max_phi = np.percentile(phi, [1, 99])
        v1 = basis @ np.array([np.cos(min_phi), np.sin(min_phi)], dtype=np.float32)
        v2 = basis @ np.array([np.cos(max_phi), np.sin(max_phi)], dtype=np.float32)
        he = np.stack([v1, v2], axis=1) if v1[0] < v2[0] else np.stack([v2, v1], axis=1)
        he = he / np.maximum(np.linalg.norm(he, axis=0, keepdims=True), 1e-8)
        concentrations, *_ = np.linalg.lstsq(he, od.T, rcond=None)
        max_c = np.percentile(concentrations, 99, axis=1).astype(np.float32)
        if not np.isfinite(max_c).all() or np.any(max_c <= 1e-6):
            return None
        return {"source_stain_matrix": he.astype(np.float32), "source_max_c": max_c}
    except Exception:
        return None


def apply_macenko(image: Image.Image, fit: dict[str, np.ndarray] | None) -> Image.Image:
    if fit is None:
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    shape = rgb.shape
    od = rgb_to_od(rgb.reshape(-1, 3))
    try:
        concentrations, *_ = np.linalg.lstsq(fit["source_stain_matrix"], od.T, rcond=None)
        scale = MACENKO_TARGET_MAX_C[:, None] / np.maximum(fit["source_max_c"][:, None], 1e-6)
        normalized_od = MACENKO_TARGET_STAIN_MATRIX @ (concentrations * scale)
        normalized = 255.0 * np.exp(-normalized_od.T)
        normalized = np.clip(normalized.reshape(shape), 0, 255).astype(np.uint8)
        return Image.fromarray(normalized)
    except Exception:
        return image


def output_is_complete(path: Path, expected_tiles: int, expected_dim: int | None = None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r") as handle:
            if "embeddings" not in handle:
                return False
            shape = handle["embeddings"].shape
            if shape[0] != expected_tiles:
                return False
            if expected_dim is not None and shape[1] != expected_dim:
                return False
        return True
    except Exception:
        return False


def process_slide(row: dict[str, Any], cfg: dict[str, Any], processor: Any, model: Any, device: Any, torch_module: Any) -> dict[str, Any]:
    root = Path(cfg["root"])
    encoder_slug = cfg["encoder_slug"]
    batch_size = int(cfg["batch_size"])
    overwrite = bool(cfg["overwrite"])
    use_amp = bool(cfg["amp"])
    max_tiles_per_slide = int(cfg.get("max_tiles_per_slide", 0) or 0)
    progress_every_tiles = int(cfg.get("progress_every_tiles", 0) or 0)
    stain_normalization = str(cfg.get("stain_normalization", "none"))
    macenko_reference_tiles = int(cfg.get("macenko_reference_tiles", 64) or 64)

    cohort = str(row["cohort"])
    patient_id = str(row["patient_id"])
    tile_path = root / str(row["tile_coords_path"])
    out_dir = root / "features/embeddings" / encoder_slug / cohort / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tile_embeddings.h5"
    expected_tiles = int(row["selected_tiles"])
    if max_tiles_per_slide > 0:
        expected_tiles = min(expected_tiles, max_tiles_per_slide)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if not overwrite and output_is_complete(out_path, expected_tiles, hidden_size):
        return {
            "cohort": cohort,
            "patient_id": patient_id,
            "status": "SKIPPED_EXISTS",
            "tile_count": expected_tiles,
            "embedding_dim": hidden_size,
            "embedding_path": str(out_path.relative_to(root)),
        }

    started = time.time()
    with h5py.File(tile_path, "r") as coord_handle:
        coords = coord_handle["coords_x_y_tissue"][:]
        if max_tiles_per_slide > 0:
            coords = coords[:max_tiles_per_slide]
        level_file = root / str(coord_handle.attrs["dicom_level_file"])
        tile_native_px = int(coord_handle.attrs["tile_native_px"])
        attrs = {key: coord_handle.attrs[key] for key in coord_handle.attrs.keys()}
    ds = pydicom.dcmread(str(level_file))
    frame_h = int(ds.Rows)
    frame_w = int(ds.Columns)
    total_cols = int(ds.TotalPixelMatrixColumns)
    total_rows = int(ds.TotalPixelMatrixRows)
    frames_per_row = math.ceil(total_cols / frame_w)
    n_tiles = int(coords.shape[0])
    embeddings = np.zeros((n_tiles, hidden_size), dtype=np.float16)
    batch_images: list[Image.Image] = []
    batch_indices: list[int] = []
    macenko_fit: dict[str, np.ndarray] | None = None

    if stain_normalization == "macenko" and n_tiles > 0:
        ref_count = min(n_tiles, max(1, macenko_reference_tiles))
        ref_indices = np.unique(np.linspace(0, n_tiles - 1, ref_count).round().astype(int))
        ref_images = [
            read_tile_from_dataset(
                ds,
                level_file,
                int(coords[i][0]),
                int(coords[i][1]),
                tile_native_px,
                frame_w,
                frame_h,
                total_cols,
                total_rows,
                frames_per_row,
            )
            for i in ref_indices
        ]
        macenko_fit = estimate_macenko_fit(ref_images)

    def flush_batch() -> None:
        if not batch_images:
            return
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        with torch_module.inference_mode():
            if use_amp:
                with torch_module.autocast(device_type="cuda", dtype=torch_module.float16):
                    output = model(**inputs)
            else:
                output = model(**inputs)
            emb = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
        embeddings[np.asarray(batch_indices, dtype=np.int64), :] = emb.detach().float().cpu().numpy().astype(np.float16)
        batch_images.clear()
        batch_indices.clear()

    for idx, coord in enumerate(coords):
        image = read_tile_from_dataset(
            ds,
            level_file,
            int(coord[0]),
            int(coord[1]),
            tile_native_px,
            frame_w,
            frame_h,
            total_cols,
            total_rows,
            frames_per_row,
        )
        if stain_normalization == "macenko":
            image = apply_macenko(image, macenko_fit)
        batch_images.append(image)
        batch_indices.append(idx)
        if len(batch_images) >= batch_size:
            flush_batch()
        if progress_every_tiles > 0 and (idx + 1) % progress_every_tiles == 0:
            print(
                json.dumps(
                    {
                        "status": "SLIDE_PROGRESS",
                        "cohort": cohort,
                        "patient_id": patient_id,
                        "tiles_done": idx + 1,
                        "tiles_total": n_tiles,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    flush_batch()

    tmp_path = out_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as handle:
        handle.create_dataset("embeddings", data=embeddings, chunks=(min(1024, n_tiles), hidden_size))
        handle.create_dataset("coords_x_y_tissue", data=coords, compression="gzip")
        for key, value in attrs.items():
            handle.attrs[key] = value
        handle.attrs["model_repo"] = cfg["model_repo"]
        handle.attrs["encoder_slug"] = encoder_slug
        handle.attrs["encoder_role"] = cfg["encoder_role"]
        handle.attrs["embedding_dim"] = hidden_size
        handle.attrs["embedding_dtype"] = "float16"
        handle.attrs["tile_count"] = n_tiles
        handle.attrs["batch_size"] = batch_size
        handle.attrs["stain_normalization"] = stain_normalization
        handle.attrs["macenko_reference_tiles"] = macenko_reference_tiles if stain_normalization == "macenko" else 0
        handle.attrs["macenko_fit_status"] = "PASS" if stain_normalization == "macenko" and macenko_fit is not None else ("NOT_RUN" if stain_normalization == "none" else "FALLBACK_ORIGINAL")
    tmp_path.replace(out_path)
    elapsed = time.time() - started
    return {
        "cohort": cohort,
        "patient_id": patient_id,
        "status": "PASS",
        "tile_count": n_tiles,
        "embedding_dim": hidden_size,
        "embedding_path": str(out_path.relative_to(root)),
        "seconds": round(elapsed, 3),
        "tiles_per_second": round(n_tiles / elapsed, 3) if elapsed > 0 else 0.0,
        "model_repo": cfg["model_repo"],
        "stain_normalization": stain_normalization,
        "macenko_fit_status": "PASS" if stain_normalization == "macenko" and macenko_fit is not None else ("NOT_RUN" if stain_normalization == "none" else "FALLBACK_ORIGINAL"),
    }


def worker_main(worker_id: int, gpu_id: str, task_queue: Any, result_queue: Any, cfg: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_ENDPOINT", cfg["hf_endpoint"])
    import torch

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(cfg["root"]) / "env/hf_cache"
    if cfg["model_repo"] in {"bioptimus/H-optimus-0", "prov-gigapath/prov-gigapath"}:
        processor, model = build_timm_processor_and_model(cfg["model_repo"], cache_dir)
    else:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(cfg["model_repo"], cache_dir=str(cache_dir))
        model = AutoModel.from_pretrained(cfg["model_repo"], cache_dir=str(cache_dir))
    model = model.to(device)
    model.eval()
    result_queue.put({"worker_id": worker_id, "gpu_id": gpu_id, "status": "WORKER_READY", "device": str(device)})
    while True:
        try:
            row = task_queue.get(timeout=5)
        except Empty:
            continue
        if row is None:
            result_queue.put({"worker_id": worker_id, "gpu_id": gpu_id, "status": "WORKER_DONE"})
            return
        try:
            result = process_slide(row, cfg, processor, model, device, torch)
            result["worker_id"] = worker_id
            result["gpu_id"] = gpu_id
            result_queue.put(result)
        except Exception as exc:  # noqa: BLE001
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "cohort": row.get("cohort", ""),
                    "patient_id": row.get("patient_id", ""),
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-repo", default="owkin/phikon-v2")
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--procs-per-gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-patients", type=int, default=0)
    parser.add_argument("--max-tiles-per-slide", type=int, default=0)
    parser.add_argument("--progress-every-tiles", type=int, default=512)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--stain-normalization", choices=["none", "macenko"], default="none")
    parser.add_argument("--macenko-reference-tiles", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    encoder_slug = model_slug(args.model_repo)
    if args.output_suffix:
        encoder_slug = f"{encoder_slug}_{args.output_suffix}"
    pass_set = pd.read_csv(root / "data/processed/stage4/stage4_pathology_pass_set.tsv", sep="\t", dtype=str).fillna("")
    pass_set["selected_tiles"] = pd.to_numeric(pass_set["selected_tiles"], errors="coerce").fillna(0).astype(int)
    pass_set = pass_set.sort_values(["cohort", "patient_id"])
    if args.limit_patients:
        pass_set = pass_set.head(args.limit_patients)
    rows = pass_set.to_dict("records")

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    worker_gpus = [gpu for gpu in gpus for _ in range(max(1, args.procs_per_gpu))]
    cfg = {
        "root": str(root),
        "model_repo": args.model_repo,
        "encoder_slug": encoder_slug,
        "encoder_role": encoder_role(args.model_repo),
        "batch_size": args.batch_size,
        "overwrite": args.overwrite,
        "amp": not args.no_amp,
        "hf_endpoint": args.hf_endpoint,
        "max_tiles_per_slide": args.max_tiles_per_slide,
        "progress_every_tiles": args.progress_every_tiles,
        "stain_normalization": args.stain_normalization,
        "macenko_reference_tiles": args.macenko_reference_tiles,
    }
    out_dir = root / "data/processed/stage4"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"embedding_manifest_{encoder_slug}.tsv"
    summary_path = out_dir / f"embedding_summary_{encoder_slug}.json"

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for row in rows:
        task_queue.put(row)
    for _ in worker_gpus:
        task_queue.put(None)
    workers = [
        ctx.Process(target=worker_main, args=(idx, gpu_id, task_queue, result_queue, cfg), daemon=False)
        for idx, gpu_id in enumerate(worker_gpus)
    ]
    started = time.time()
    for proc in workers:
        proc.start()

    results: list[dict[str, Any]] = []
    ready = 0
    done_workers = 0
    completed_slides = 0
    while done_workers < len(workers):
        item = result_queue.get()
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if item.get("status") == "WORKER_READY":
            ready += 1
        elif item.get("status") == "WORKER_DONE":
            done_workers += 1
        else:
            results.append(item)
            completed_slides += 1
    for proc in workers:
        proc.join()
    failed = [row for row in results if row.get("status") == "FAILED"]
    pd.DataFrame(results).to_csv(manifest_path, sep="\t", index=False)
    summary = {
        "model_repo": args.model_repo,
        "encoder_slug": encoder_slug,
        "encoder_role": encoder_role(args.model_repo),
        "slides_requested": len(rows),
        "slides_completed": completed_slides,
        "pass_or_skipped_slides": sum(1 for row in results if row.get("status") in {"PASS", "SKIPPED_EXISTS"}),
        "failed_slides": len(failed),
        "total_tiles": int(sum(int(row.get("tile_count", 0) or 0) for row in results if row.get("status") in {"PASS", "SKIPPED_EXISTS"})),
        "gpus": gpus,
        "workers": len(workers),
        "procs_per_gpu": args.procs_per_gpu,
        "batch_size": args.batch_size,
        "max_tiles_per_slide": args.max_tiles_per_slide,
        "stain_normalization": args.stain_normalization,
        "macenko_reference_tiles": args.macenko_reference_tiles,
        "amp": not args.no_amp,
        "seconds": round(time.time() - started, 3),
        "failed": failed[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not failed and completed_slides == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
