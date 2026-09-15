#!/usr/bin/env python3
"""Extract TCGA SVS tile embeddings for Stage 10 WSI transfer."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image


TILE_SIZE = 256


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


def read_tile_from_svs(slide: Any, x: int, y: int, tile_native_px: int) -> Image.Image:
    image = slide.read_region((x, y), 0, (tile_native_px, tile_native_px)).convert("RGB")
    if image.size != (TILE_SIZE, TILE_SIZE):
        image = image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.BILINEAR)
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
    import openslide

    root = Path(cfg["root"])
    encoder_slug = cfg["encoder_slug"]
    batch_size = int(cfg["batch_size"])
    overwrite = bool(cfg["overwrite"])
    use_amp = bool(cfg["amp"]) and str(device).startswith("cuda")
    max_tiles_per_slide = int(cfg.get("max_tiles_per_slide", 0) or 0)
    progress_every_tiles = int(cfg.get("progress_every_tiles", 0) or 0)

    cohort = str(row["cohort"])
    patient_id = str(row["patient_id"])
    slide_key = str(row["slide_candidate_key"])
    tile_path = root / str(row["tile_coords_path"])
    out_dir = root / "features/tcga_wsi_embeddings" / encoder_slug / cohort / patient_id / slide_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tile_embeddings.h5"
    expected_tiles = int(float(row["selected_tiles"]))
    if max_tiles_per_slide > 0:
        expected_tiles = min(expected_tiles, max_tiles_per_slide)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if not overwrite and output_is_complete(out_path, expected_tiles, hidden_size):
        return {
            "cohort": cohort,
            "patient_id": patient_id,
            "slide_id": row.get("slide_id", ""),
            "slide_candidate_key": slide_key,
            "status": "SKIPPED_EXISTS",
            "tile_count": expected_tiles,
            "embedding_dim": hidden_size,
            "embedding_path": str(out_path.relative_to(root)),
            "model_repo": cfg["model_repo"],
        }

    started = time.time()
    with h5py.File(tile_path, "r") as coord_handle:
        coords = coord_handle["coords_x_y_tissue"][:]
        if max_tiles_per_slide > 0:
            coords = coords[:max_tiles_per_slide]
        svs_path = root / str(coord_handle.attrs["svs_file"])
        tile_native_px = int(coord_handle.attrs["tile_native_px"])
        attrs = {key: coord_handle.attrs[key] for key in coord_handle.attrs.keys()}
    n_tiles = int(coords.shape[0])
    embeddings = np.zeros((n_tiles, hidden_size), dtype=np.float16)
    batch_images: list[Image.Image] = []
    batch_indices: list[int] = []

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

    slide = openslide.OpenSlide(str(svs_path))
    try:
        for idx, coord in enumerate(coords):
            image = read_tile_from_svs(slide, int(coord[0]), int(coord[1]), tile_native_px)
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
                            "slide_candidate_key": slide_key,
                            "tiles_done": idx + 1,
                            "tiles_total": n_tiles,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        flush_batch()
    finally:
        slide.close()

    tmp_path = out_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as handle:
        handle.create_dataset("embeddings", data=embeddings, chunks=(min(1024, max(1, n_tiles)), hidden_size))
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
        handle.attrs["source_tile_qc_table"] = cfg["pass_table"]
    tmp_path.replace(out_path)
    elapsed = time.time() - started
    return {
        "cohort": cohort,
        "patient_id": patient_id,
        "slide_id": row.get("slide_id", ""),
        "slide_candidate_key": slide_key,
        "status": "PASS",
        "tile_count": n_tiles,
        "embedding_dim": hidden_size,
        "embedding_path": str(out_path.relative_to(root)),
        "seconds": round(elapsed, 3),
        "tiles_per_second": round(n_tiles / elapsed, 3) if elapsed > 0 else 0.0,
        "model_repo": cfg["model_repo"],
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
                    "slide_id": row.get("slide_id", ""),
                    "slide_candidate_key": row.get("slide_candidate_key", ""),
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--pass-table", default="data/processed/stage10/tcga_wsi_pilot_tile_qc.tsv")
    parser.add_argument("--model-repo", default="prov-gigapath/prov-gigapath")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--procs-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-slides", type=int, default=0)
    parser.add_argument("--max-tiles-per-slide", type=int, default=0)
    parser.add_argument("--progress-every-tiles", type=int, default=256)
    parser.add_argument("--output-suffix", default="tcga_wsi_pilot")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    encoder_slug = model_slug(args.model_repo)
    if args.output_suffix:
        encoder_slug = f"{encoder_slug}_{args.output_suffix}"
    pass_path = root / args.pass_table
    pass_set = pd.read_csv(pass_path, sep="\t", dtype=str).fillna("")
    pass_set["selected_tiles"] = pd.to_numeric(pass_set["selected_tiles"], errors="coerce").fillna(0).astype(int)
    pass_set = pass_set[pass_set["status"].isin(["PASS", "SKIPPED_EXISTS"])].copy()
    pass_set = pass_set.sort_values(["cohort", "patient_id", "slide_candidate_key"])
    if args.limit_slides:
        pass_set = pass_set.head(args.limit_slides)
    rows = pass_set.to_dict("records")
    if not rows:
        raise RuntimeError(f"No PASS rows found in {pass_path}")

    out_dir = root / "data/processed/stage10"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{args.output_suffix}_embedding_manifest_{encoder_slug}.tsv"
    summary_path = out_dir / f"{args.output_suffix}_embedding_summary_{encoder_slug}.json"

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    worker_gpus = [gpu for gpu in gpus for _ in range(max(1, args.procs_per_gpu))]
    if not worker_gpus:
        worker_gpus = [""]
    cfg = {
        "root": str(root),
        "pass_table": args.pass_table,
        "model_repo": args.model_repo,
        "encoder_slug": encoder_slug,
        "encoder_role": encoder_role(args.model_repo),
        "batch_size": args.batch_size,
        "overwrite": args.overwrite,
        "amp": not args.no_amp,
        "hf_endpoint": args.hf_endpoint,
        "max_tiles_per_slide": args.max_tiles_per_slide,
        "progress_every_tiles": args.progress_every_tiles,
    }
    started = time.time()
    if args.dry_run:
        preview = pd.DataFrame(rows)
        preview["status"] = "DRY_RUN"
        preview.to_csv(manifest_path, sep="\t", index=False)
        summary = {
            "model_repo": args.model_repo,
            "encoder_slug": encoder_slug,
            "encoder_role": encoder_role(args.model_repo),
            "slides_requested": len(rows),
            "dry_run": True,
            "gpus": gpus,
            "workers": len(worker_gpus),
            "max_tiles_per_slide": args.max_tiles_per_slide,
            "pass_table": args.pass_table,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0

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
    for proc in workers:
        proc.start()

    results: list[dict[str, Any]] = []
    done_workers = 0
    completed_slides = 0
    while done_workers < len(workers):
        item = result_queue.get()
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if item.get("status") == "WORKER_DONE":
            done_workers += 1
        elif item.get("status") != "WORKER_READY" and item.get("status") != "SLIDE_PROGRESS":
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
        "amp": not args.no_amp,
        "seconds": round(time.time() - started, 3),
        "failed": failed[:20],
        "pass_table": args.pass_table,
        "manifest": str(manifest_path.relative_to(root)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not failed and completed_slides == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
