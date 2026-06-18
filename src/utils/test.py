from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.datasets.cityscapes import Cityscapes as CS
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as nnF
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
try:
    from thop import profile
except ImportError:
    profile = None
from src.model_architecture.model_factory import ModelBuilder
from src.utils.ablation import make_run_tag, parse_ablation_spec

MEAN = np.array([0.286, 0.325, 0.283], dtype=np.float32)
STD = np.array([0.176, 0.181, 0.177], dtype=np.float32)

cityscapes_classes = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# ------------------------------------------------------------
# Helpers: mapping + logits selection
# ------------------------------------------------------------
def _select_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _build_id_to_trainid_lut(device) -> torch.Tensor:
    id_to_trainid = {c.id: c.train_id for c in CS.classes if c.id >= 0}
    max_id = max(id_to_trainid.keys())
    lut = torch.full((max_id + 1,), 255, dtype=torch.long, device=device)

    for label_id, train_id in id_to_trainid.items():
        if train_id < 0:
            lut[label_id] = 255
        else:
            lut[label_id] = int(train_id)

    return lut


def _map_labelids_to_trainids(labels: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """
    labels: [B,H,W] labelId (0..33) OR 255
    returns: [B,H,W] trainId (0..18) OR 255
    """
    if labels.numel() == 0:
        return labels
    # keep 255 as 255
    labels_clamped = labels.clamp(0, lut.numel() - 1)
    mapped = lut[labels_clamped]
    mapped = torch.where(labels == 255, torch.full_like(mapped, 255), mapped)
    return mapped


def _fast_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    preds, labels: flattened (or any shape) with labels in [0..num_classes-1], ignore as -1
    """
    mask = (labels >= 0) & (labels < num_classes)
    labels = labels[mask]
    preds = preds[mask]
    if labels.numel() == 0:
        return torch.zeros((num_classes, num_classes), device=preds.device, dtype=torch.int64)
    preds = preds.clamp(0, num_classes - 1)
    idx = num_classes * labels + preds
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _extract_state_dict(ckpt_obj, use_ema: bool = True):
    if isinstance(ckpt_obj, dict):
        # 1. Próba załadowania wag EMA (najlepsze wyniki)
        if use_ema and "ema_model" in ckpt_obj:
            ema_dict = ckpt_obj["ema_model"]
            clean_dict = {}
            for k, v in ema_dict.items():
                # Usuwamy prefix "module.", który dodaje PyTorchowy AveragedModel
                clean_k = k.replace("module.", "") if k.startswith("module.") else k
                # Pomijamy licznik iteracji z AveragedModel
                if clean_k != "n_averaged":
                    clean_dict[clean_k] = v
            logging.getLogger(__name__).info("Loaded EMA weights from checkpoint.")
            return clean_dict
            
        # 2. Fallback na zwykły model
        if isinstance(ckpt_obj.get("model"), dict):
            logging.getLogger(__name__).info("Loaded STANDARD weights from checkpoint.")
            return ckpt_obj["model"]
        if isinstance(ckpt_obj.get("state_dict"), dict):
            return ckpt_obj["state_dict"]
            
    return ckpt_obj

def _resolve_weights_path(root: str, model_name: str, run_tag: str, weights_path: Optional[str] = None) -> str:
    if weights_path:
        if os.path.isabs(weights_path):
            return weights_path
        return os.path.join(root, weights_path)

    ckpt_best = os.path.join(root, "checkpoints", model_name, "best.pt")
    ckpt_last = os.path.join(root, "checkpoints", model_name, "last.pt")
    legacy = os.path.join(root, f"best_model/best_model_{run_tag}.pth")

    if os.path.isfile(ckpt_best):
        return ckpt_best
    if os.path.isfile(ckpt_last):
        return ckpt_last
    if os.path.isfile(legacy):
        return legacy

    raise FileNotFoundError(
        "No checkpoint found. Tried:\n"
        f"  {ckpt_best}\n"
        f"  {ckpt_last}\n"
        f"  {legacy}"
    )


def _build_model_with_weights(model_name, device_obj, ablation_cfg, patch_cfg, weights_path, use_ema=True) -> Tuple[torch.nn.Module, int]:
    ckpt_obj = torch.load(weights_path, map_location=device_obj)
    state_dict = _extract_state_dict(ckpt_obj, use_ema=use_ema)
    
    base_model_name = model_name
    for suffix in ["_512x1024", "_512x512"]:
        if base_model_name.endswith(suffix):
            base_model_name = base_model_name[:-len(suffix)]
            break

    last_error = None
    for num_classes in (19, 34):
        builder = ModelBuilder(
            num_classes=num_classes,
            device=device_obj,
            ablation_cfg=ablation_cfg,
            patch_cfg=patch_cfg,
        )
        model = builder.build(base_model_name)
        try:
            # Używamy strict=True (domyślnie), żeby upewnić się, że nie ma problemów z kluczami
            model.load_state_dict(state_dict)
            return model, num_classes
        except RuntimeError as exc:
            last_error = exc
            
    raise last_error

def _ensure_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def _latency_stats(times_ms: np.ndarray):
    return {
        "mean_ms": float(times_ms.mean()) if times_ms.size else 0.0,
        "p50_ms": float(np.percentile(times_ms, 50)) if times_ms.size else 0.0,
        "p90_ms": float(np.percentile(times_ms, 90)) if times_ms.size else 0.0,
        "p95_ms": float(np.percentile(times_ms, 95)) if times_ms.size else 0.0,
        "p99_ms": float(np.percentile(times_ms, 99)) if times_ms.size else 0.0,
        "min_ms": float(times_ms.min()) if times_ms.size else 0.0,
        "max_ms": float(times_ms.max()) if times_ms.size else 0.0,
    }

def _plot_latency_hist(times_s: np.ndarray, out_path: str, title: str):
    ms = times_s * 1000.0
    plt.figure(figsize=(10, 5))
    plt.hist(ms, bins=40)
    plt.xlabel("Latency [ms]")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def _plot_per_class_iou(ious: np.ndarray, out_path: str, title: str):
    ious = _ensure_numpy(ious)
    plt.figure(figsize=(12, 4))
    x = np.arange(len(ious))
    plt.bar(x, np.nan_to_num(ious, nan=0.0))
    plt.ylim(0, 1)
    plt.xlabel("Class index (trainId 0..18)")
    plt.ylabel("IoU")
    plt.title(title)
    plt.grid(True, axis="y")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def _plot_confusion_matrix(conf: np.ndarray, out_path: str, title: str, normalize: bool = True):
    conf = conf.astype(np.float64)
    if normalize:
        row_sum = conf.sum(axis=1, keepdims=True)
        conf = np.divide(conf, np.maximum(row_sum, 1.0))
    plt.figure(figsize=(7, 6))
    plt.imshow(conf, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("GT")
    plt.colorbar()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def _plot_miou_vs_latency_point(miou: float, p50_ms: float, out_path: str, title: str):
    plt.figure(figsize=(6, 5))
    plt.scatter([p50_ms], [miou])
    plt.xlabel("Latency P50 [ms] (batch)")
    plt.ylabel("mIoU")
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# @torch.no_grad()
# def benchmark_cpu_e2e(
#     model: torch.nn.Module,
#     dataset,
#     device: torch.device,
#     H: int,
#     W: int,
#     cpu_threads: int = 1,
#     warmup_iters: int = 20,
#     timed_iters: int = 200,
# ):
#     """
#     Fair-ish CPU benchmark:
#       - batch=1
#       - num_workers=0 (no background prefetch)
#       - includes forward + argmax
#       - includes dataset __getitem__ (transform) time implicitly, if we fetch inside loop
#         (more "end-to-end"). If you want pure model-only CPU, see notes below.
#     """
#     assert device.type == "cpu"

#     # torch.set_num_threads(int(cpu_threads))
#     # torch.set_num_interop_threads(1)

#     model = model.to(device)
    
#     model.eval()

#     # Warmup (includes data + forward)
#     n = min(len(dataset), warmup_iters + timed_iters)
#     idxs = list(range(n))

#     # warmup
#     for i in range(min(warmup_iters, n)):
#         img, _ = dataset[idxs[i]]               # includes transforms
#         img = img.unsqueeze(0).to(device)       # [1,3,H,W]
#         out = model(img)
#         logits = out[0] if isinstance(out, (tuple, list)) else out
#         _ = torch.argmax(logits, dim=1)

#     # timed
#     times = []
#     start = time.perf_counter()
#     for i in range(warmup_iters, min(warmup_iters + timed_iters, n)):
#         t0 = time.perf_counter()
#         img, _ = dataset[idxs[i]]
#         img = img.unsqueeze(0).to(device)
#         out = model(img)
#         logits = out[0] if isinstance(out, (tuple, list)) else out
#         _ = torch.argmax(logits, dim=1)
#         t1 = time.perf_counter()
#         times.append(t1 - t0)
#     end = time.perf_counter()

#     times = np.array(times, dtype=np.float64)
#     total = float(end - start)
#     fps = (len(times) / total) if total > 0 else 0.0
#     return times, fps

@torch.no_grad()
def benchmark_cpu_model_only(
    model: torch.nn.Module,
    device: torch.device,
    H: int,
    W: int,
    cpu_threads: int = 1,
    warmup_iters: int = 20,
    timed_iters: int = 200,
    run_argmax: bool = False,
):
    assert device.type == "cpu"

    torch.set_num_threads(int(cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = model.to(device)
    model.eval()

    x = torch.randn(1, 3, H, W, device=device)

    # warmup
    for _ in tqdm(range(warmup_iters),desc="CPU warmup"):
        out = model(x)
        if run_argmax:
            logits = out[0] if isinstance(out, (tuple, list)) else out
            _ = torch.argmax(logits, dim=1)

    times = []
    for _ in tqdm(range(timed_iters),desc="CPU timed"):
        t0 = time.perf_counter()
        out = model(x)
        if run_argmax:
            logits = out[0] if isinstance(out, (tuple, list)) else out
            _ = torch.argmax(logits, dim=1)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times, dtype=np.float64) * 1000.0  # Dodaj * 1000.0
    mean_ms = times.mean()
    fps = 1000.0 / mean_ms  # Zmień z 1.0 na 1000.0

    return  times, float(fps)


# @torch.no_grad()
# def benchmark_gpu_forward_only(
#     model: torch.nn.Module,
#     device: torch.device,
#     H: int,
#     W: int,
#     batch_size: int = 4,
#     warmup_iters: int = 50,
#     timed_iters: int = 200,
# ):
#     """
#     GPU benchmark (forward-only):
#       - synthetic input on GPU
#       - synchronize before/after
#       - measures forward + argmax (optional; here included to match output handling)
#     """
#     assert device.type == "cuda"
#     model.eval()

#     model = model.to(device)
    
#     x = torch.randn(batch_size, 3, H, W, device=device)

#     # warmup
#     for _ in range(warmup_iters):
#         torch.cuda.synchronize()
#         out = model(x)
#         logits = out[0] if isinstance(out, (tuple, list)) else out
#         _ = torch.argmax(logits, dim=1)
#         torch.cuda.synchronize()

#     # timed
#     times = []
#     for _ in range(timed_iters):
#         torch.cuda.synchronize()
#         t0 = time.perf_counter()
#         out = model(x)
#         logits = out[0] if isinstance(out, (tuple, list)) else out
#         _ = torch.argmax(logits, dim=1)
#         torch.cuda.synchronize()
#         t1 = time.perf_counter()
#         times.append(t1 - t0)

#     times = np.array(times, dtype=np.float64)
#     fps = (batch_size / (times.mean() if times.size else 1.0))
#     return times, float(fps)


@torch.no_grad()
def benchmark_gpu_paper_style(
    model: torch.nn.Module,
    device: torch.device,
    H: int,
    W: int,
    channels: int = 3,
    batch_size: int = 1,          # paper-style: 1
    warmup_iters: int = 50,
    timed_iters: int = 200,
    run_argmax: bool = False,     # zwykle False dla "forward-only"
    dtype: torch.dtype = torch.float32,
):
    assert device.type == "cuda", "Ten benchmark jest do GPU/CUDA."
    assert batch_size == 1, "Do paper-style latency używaj batch_size=1."

    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, channels, H, W, device=device, dtype=dtype)

    # Warmup
    for _ in tqdm(range(warmup_iters), desc="GPU warmup"):
        out = model(x)
        if run_argmax:
            logits = out[0] if isinstance(out, (tuple, list)) else out
            _ = torch.argmax(logits, dim=1)

    torch.cuda.synchronize()

    # CUDA events dają dokładniejszy pomiar GPU niż perf_counter wokół async kerneli
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    times_ms = []

    for _ in tqdm(range(timed_iters), desc="GPU timed"):
        starter.record()

        out = model(x)
        if run_argmax:
            logits = out[0] if isinstance(out, (tuple, list)) else out
            _ = torch.argmax(logits, dim=1)

        ender.record()
        torch.cuda.synchronize()

        elapsed_ms = starter.elapsed_time(ender)  # ms
        times_ms.append(elapsed_ms)

    times_ms = np.array(times_ms, dtype=np.float64)

    mean_ms = float(times_ms.mean())
    std_ms = float(times_ms.std())
    p50_ms = float(np.percentile(times_ms, 50))
    p95_ms = float(np.percentile(times_ms, 95))
    p99_ms = float(np.percentile(times_ms, 99))
    fps = 1000.0 / mean_ms

    return times_ms, float(fps)
    # return {
    #     "mean_ms": mean_ms,
    #     "std_ms": std_ms,
    #     "p50_ms": p50_ms,
    #     "p95_ms": p95_ms,
    #     "p99_ms": p99_ms,
    #     "fps": fps,
    #     "num_iters": timed_iters,
    #     "warmup_iters": warmup_iters,
    #     "batch_size": batch_size,
    #     "input_size": (H, W),
    #     "argmax_included": run_argmax,
    #     "dtype": str(dtype),
    # }

@torch.no_grad()
def count_pred_pixels(model, val_loader, num_classes=19, ignore_index=255, max_batches=50, device="cuda"):
    model.eval()
    pred_counts = torch.zeros(num_classes, dtype=torch.long)
    gt_counts = torch.zeros(num_classes, dtype=torch.long)

    lut = _build_id_to_trainid_lut(torch.device(device))

    for i, (x, y_labelid) in enumerate(val_loader):
        if i >= max_batches:
            break

        x = x.to(device)
        y_labelid = y_labelid.to(device).long()  # [B,H,W]
        y_trainid = _map_labelids_to_trainids(y_labelid, lut).squeeze(0)

        outputs = _select_logits(model(x))
        pred = outputs.argmax(1)  # [B,H,W]

        if outputs.shape[1] == num_classes:
            pred_trainid = pred
        else:
            pred_trainid = lut[pred.clamp(0, lut.numel() - 1)]

        pred_trainid = pred_trainid.squeeze(0)

        mask = y_trainid != ignore_index
        yv = y_trainid[mask]
        pv = pred_trainid[mask]

        for c in range(num_classes):
            gt_counts[c] += (yv == c).sum().item()
            pred_counts[c] += (pv == c).sum().item()

    return gt_counts, pred_counts


@torch.no_grad()
def direct_iou_for_class(model, val_loader, cls=18, ignore_index=255, device="cuda", max_batches=200):
    model.eval()
    inter = 0
    union = 0
    gt_cnt = 0
    pr_cnt = 0

    lut = _build_id_to_trainid_lut(torch.device(device))
    num_eval_classes = 19

    for i, (x, y_labelid) in enumerate(val_loader):
        if i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y_labelid = y_labelid.to(device, non_blocking=True).long()   # [B,H,W]

        y_trainid = _map_labelids_to_trainids(y_labelid, lut).squeeze(0)  # [H,W]

        outputs = _select_logits(model(x))
        pred = outputs.argmax(1)  # [B,H,W]

        # map pred do trainId jeśli model ma 34 wyjścia
        if outputs.shape[1] == num_eval_classes:
            pred_trainid = pred
        else:
            pred_trainid = lut[pred.clamp(0, lut.numel() - 1)]

        pred_trainid = pred_trainid.squeeze(0)

        m = (y_trainid != ignore_index)
        yv = y_trainid[m]
        pv = pred_trainid[m]

        gt = (yv == cls)
        pr = (pv == cls)

        inter += (gt & pr).sum().item()
        union += (gt | pr).sum().item()
        gt_cnt += gt.sum().item()
        pr_cnt += pr.sum().item()

    iou = inter / union if union > 0 else float("nan")
    return iou, inter, union, gt_cnt, pr_cnt

@torch.no_grad()
def count_gt_pixels_only(val_loader, device, lut, num_classes=19, ignore_index=255):
    gt_counts = torch.zeros(num_classes, dtype=torch.long)

    for _, y_labelid in val_loader:
        y_labelid = y_labelid.to(device).long()
        y_trainid = _map_labelids_to_trainids(y_labelid, lut)

        mask = y_trainid != ignore_index
        yv = y_trainid[mask]

        for c in range(num_classes):
            gt_counts[c] += (yv == c).sum().item()

    return gt_counts

# ------------------------------------------------------------
# Model utils
# ------------------------------------------------------------

def switch_model_to_deploy(module: nn.Module):
    """
    Rekurencyjnie przechodzi po modelu i wywołuje switch_to_deploy(),
    jeśli dany submodule taką metodę posiada.
    """
    for m in module.modules():
        if m is not module and hasattr(m, "switch_to_deploy") and callable(m.switch_to_deploy):
            m.switch_to_deploy()


def _get_model_complexity(model: nn.Module, input_size: Tuple[int, int], device: torch.device):
    """Returns GFLOPs and Parameters in Millions."""
    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w).to(device)
    if profile is None:
        return None, None
    try:
        # profile returns (macs, params)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        # GFLOPs = (macs * 2) / 1e9
        gflops = (macs * 2) / 1e9
        params_m = params / 1e6
        return float(gflops), float(params_m)
    except Exception as exc:
        logging.getLogger(__name__).warning(f"Complexity calculation failed: {exc}")
        return None, None



# ------------------------------------------------------------
# Test main using torchvision.datasets.Cityscapes
# ------------------------------------------------------------
def test_main(
    batch_size: int = 4,
    model_name: str = "FAscnn_pp_V13",
    device: str = "auto",
    ablation: Optional[str] = None,
    patch_cfg: Optional[dict] = None,
    image_size: Tuple[int, int] = (1024, 2048),  # (H,W) for fast eval
    cityscapes_root: Optional[str] = None,
    num_workers: int = 4,
    save_vis_per_batch: int = 2,
    warmup_iters: int = 10,
    measure_cpu: bool = True,
    measure_gpu: bool = True,
    cpu_threads: int = 1,
    cpu_warmup_iters: int = 20,
    cpu_timed_iters: int = 200,
    gpu_warmup_iters: int = 50,
    gpu_timed_iters: int = 200,
    gpu_batch_size: int = 1,  # paper-style latency: batch=1
    use_ema: bool = True,
    weights_path: Optional[str] = None,
    rescale_mask: bool = True,
    run_name: Optional[str] = None,
):
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    patch_cfg = patch_cfg or {}
    ablation_cfg = parse_ablation_spec(ablation)
    if ablation_cfg.id() and not model_name.startswith("FAscnn_pp_"):
        raise ValueError(f"Ablation is supported mainly for FAscnn_pp_* models, got {model_name}.")

    # Resolve project root (as you had)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # If user did not pass explicit cityscapes_root, try typical locations in repo
    if cityscapes_root is None:
        # You previously had cityscapes folder at repo root (matches your screenshot)
        candidate = os.path.join(root, "cityscapes")
        cityscapes_root = candidate if os.path.isdir(candidate) else root

    # Adjust model_name based on image_size if different from default
    if image_size and tuple(image_size) != (1024, 2048):
        h, w = image_size
        model_name = f"{model_name}_{h}x{w}"

    # Logging
    run_tag = make_run_tag(model_name, ablation_cfg)
    log_path = os.path.join(root, "log", "test", f"test_{run_tag}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # Load model
    weights_path = _resolve_weights_path(root, model_name, run_tag, weights_path=weights_path)
    model, num_classes = _build_model_with_weights(
        model_name,
        device_obj,
        ablation_cfg,
        patch_cfg,
        weights_path,
        use_ema=use_ema # <-- DODANE
    )
    
    switch_model_to_deploy(model)
    model = model.to(device_obj).eval()

    logger.info(f"Using device: {device_obj}")
    logger.info(f"Model type: {model.__class__.__name__}")
    logger.info(f"Loaded weights: {weights_path}")
    logger.info(f"Model output classes: {num_classes}")
    if ablation_cfg.id():
        logger.info(f"Ablation: {ablation_cfg.id()}")

    # Model complexity
    gflops, params_m = _get_model_complexity(model, image_size, device_obj)
    if gflops is not None:
        logger.info(f"GFLOPs (@{image_size[0]}x{image_size[1]}): {gflops:.2f}")
        logger.info(f"Parameters: {params_m:.2f} M")

    # Build LUT and dataset
    lut = _build_id_to_trainid_lut(device_obj)  # labelId -> trainId(0..18) or 255
    num_eval_classes = 19

    H, W = image_size

    # torchvision Cityscapes returns (PIL img, PIL mask)
    img_tf = T.Compose([
        T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(MEAN.tolist(), STD.tolist()),
    ])

    def target_tf(mask_pil):
        # Rescale mask if not full resolution (1024x2048) and rescale_mask is True
        if rescale_mask and (H, W) != (1024, 2048):
            mask_pil = F.resize(mask_pil, (H, W), interpolation=T.InterpolationMode.NEAREST)
        mask = np.array(mask_pil, dtype=np.int64)   # labelId
        return torch.from_numpy(mask).long()

    test_dataset = Cityscapes(
        root=cityscapes_root,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_tf,
        target_transform=target_tf,
    )
    
    # -----------------------
    # FAIR benchmarks: CPU (E2E) and GPU (forward-only)
    # -----------------------
    cpu_bench = None
    gpu_bench = None

    if measure_cpu:
        logger.info(f"[CPU BENCH] threads={cpu_threads}, warmup={cpu_warmup_iters}, timed={cpu_timed_iters}, batch=1, E2E(data+forward+argmax)")
        # Uwaga: model musi być na CPU do pomiaru CPU
        model_cpu = model.to("cpu")
        # cpu_times, cpu_fps = benchmark_cpu_e2e(
        #     model=model_cpu,
        #     dataset=test_dataset,
        #     device=torch.device("cpu"),
        #     H=H,
        #     W=W,
        #     cpu_threads=cpu_threads,
        #     warmup_iters=cpu_warmup_iters,
        #     timed_iters=cpu_timed_iters,
        # )
        cpu_times, cpu_fps = benchmark_cpu_model_only(
            model=model_cpu,
            device=torch.device("cpu"),
            H=H,
            W=W,
            cpu_threads=cpu_threads,
            warmup_iters=cpu_warmup_iters,
            timed_iters=cpu_timed_iters,
            run_argmax=False,
        )
        
        cpu_lat = _latency_stats(cpu_times)
        cpu_bench = {"fps": cpu_fps, "latency_ms": cpu_lat}
        logger.info(f"[CPU BENCH] FPS={cpu_fps:.2f} | p50={cpu_lat['p50_ms']:.2f}ms p95={cpu_lat['p95_ms']:.2f}ms p99={cpu_lat['p99_ms']:.2f}ms")

        # wróć modelem na pierwotne urządzenie do reszty testu (mIoU liczysz dalej jak chcesz)
        model = model.to(device_obj).eval()

    if measure_gpu and torch.cuda.is_available():
        logger.info(f"[GPU BENCH] batch={1}, warmup={gpu_warmup_iters}, timed={gpu_timed_iters}, Paper-style latency (forward-only + argmax, batch=1)")
        model_gpu = model.to("cuda").eval()
        # gpu_times, gpu_fps = benchmark_gpu_forward_only(
        #     model=model_gpu,
        #     device=torch.device("cuda"),
        #     H=H,
        #     W=W,
        #     batch_size=gpu_batch_size,
        #     warmup_iters=gpu_warmup_iters,
        #     timed_iters=gpu_timed_iters,
        # )
        gpu_times, gpu_fps = benchmark_gpu_paper_style(
            model=model_gpu,
            device=torch.device("cuda"),
            H=H,
            W=W,
            batch_size=1,  # paper-style latency: batch=1
            warmup_iters=gpu_warmup_iters,
            timed_iters=gpu_timed_iters,
            run_argmax=False,  
        )
        
        
        gpu_lat = _latency_stats(gpu_times)
        gpu_bench = {"fps": gpu_fps, "latency_ms": gpu_lat, "batch_size": gpu_batch_size}
        logger.info(f"[GPU BENCH] FPS~{gpu_fps:.2f} | p50={gpu_lat['p50_ms']:.2f}ms p95={gpu_lat['p95_ms']:.2f}ms p99={gpu_lat['p99_ms']:.2f}ms")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device_obj.type == "cuda"),
    )

    # Output dirs
    results_root = os.path.join(root, "results")
    model_dir = os.path.join(results_root, model_name)
    base_run_name = run_name if run_name else (ablation_cfg.id() if ablation_cfg.id() else "no_ablation")
    run_dir = os.path.join(model_dir, f"{base_run_name}_{H}x{W}")
    images_dir = os.path.join(run_dir, "images")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # TensorBoard
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Log “static” info (paper table helpers)
    writer.add_text("run/model_name", model_name)
    writer.add_text("run/ablation", ablation_cfg.id() if ablation_cfg.id() else "none")
    writer.add_text("run/weights_path", weights_path)
    writer.add_text("run/image_size", f"{H}x{W}")
    writer.add_text("run/device", str(device_obj))
    if gflops is not None:
        writer.add_scalar("model/gflops", gflops, 0)
        writer.add_scalar("model/params_m", params_m, 0)
    if writer is not None and cpu_bench is not None:
        writer.add_scalar("bench_cpu/fps_e2e", cpu_bench["fps"], 0)
        writer.add_scalar("bench_cpu/lat_p50_ms_e2e", cpu_bench["latency_ms"]["p50_ms"], 0)
        writer.add_scalar("bench_cpu/lat_p95_ms_e2e", cpu_bench["latency_ms"]["p95_ms"], 0)
        writer.add_scalar("bench_cpu/lat_p99_ms_e2e", cpu_bench["latency_ms"]["p99_ms"], 0)

    if writer is not None and gpu_bench is not None:
        writer.add_scalar("bench_gpu/fps_forward_only", gpu_bench["fps"], 0)
        writer.add_scalar("bench_gpu/lat_p50_ms_forward_only", gpu_bench["latency_ms"]["p50_ms"], 0)
        writer.add_scalar("bench_gpu/lat_p95_ms_forward_only", gpu_bench["latency_ms"]["p95_ms"], 0)
        writer.add_scalar("bench_gpu/lat_p99_ms_forward_only", gpu_bench["latency_ms"]["p99_ms"], 0)
        writer.add_text("bench_gpu/batch_size", str(gpu_bench["batch_size"]))

    def denormalize(img_tensor):
        img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
        img = (img * STD + MEAN).clip(0, 1)
        return img

    def visualize_results(img, mask_gt_trainid, mask_pred_trainid, idx):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(denormalize(img))
        axes[0].set_title("Image")
        axes[1].imshow(mask_gt_trainid.cpu().numpy(), cmap="jet", vmin=0, vmax=18)
        axes[1].set_title("GT (trainId)")
        axes[2].imshow(mask_pred_trainid.cpu().numpy(), cmap="jet", vmin=0, vmax=18)
        axes[2].set_title("Pred (trainId)")
        for ax in axes:
            ax.axis("off")
        plt.savefig(os.path.join(images_dir, f"results_{idx}.png"))
        plt.close()

    # Metrics accumulators
    conf_mat = torch.zeros((num_eval_classes, num_eval_classes), device=device_obj, dtype=torch.int64)

    pixel_correct = 0
    pixel_total = 0
    batch_pixel_acc = []
    batch_miou = []

    # Timing
    batch_times = []
    per_image_times = []  # seconds per image (approx = batch_time / B)
    
    # Warmup (important for fair-ish GPU timing)
    if device_obj.type == "cuda":
        logger.info(f"Warmup: {warmup_iters} iterations...")
        dummy = torch.randn(batch_size, 3, H, W, device=device_obj)
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = _select_logits(model(dummy))
        torch.cuda.synchronize()

    with torch.no_grad():
        for batch_idx, (images, masks_labelid) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device_obj, non_blocking=True)
            masks_labelid = masks_labelid.to(device_obj, non_blocking=True).long()

            # Map GT labelId -> trainId
            masks_trainid = _map_labelids_to_trainids(masks_labelid, lut)  # [B,H,W] in 0..18 or 255

            # Inference timing (GPU-safe)
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = _select_logits(model(images))  # [B,C,H,W]

            # Upsample if model output doesn't match GT resolution (e.g. rescale_mask=False or model downsamples)
            if outputs.shape[-2:] != masks_trainid.shape[-2:]:
                outputs = nnF.interpolate(
                    outputs,
                    size=masks_trainid.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            batch_times.append(end_time - start_time)
            bt = (end_time - start_time)
            bsz = images.size(0)
            per_image_times.append(bt / max(bsz, 1))

            preds = torch.argmax(outputs, dim=1)  # [B,H,W] in 0..C-1

            # Convert predictions to trainId if model outputs 34 classes
            if num_classes == num_eval_classes:
                preds_trainid = preds
            else:
                # Model predicts "labelId-like" indices 0..33; map to trainId via lut
                preds_trainid = lut[preds.clamp(0, lut.numel() - 1)]
                # ensure ignore stays ignore
                preds_trainid = preds_trainid.clamp(0, 255)

            # Pixel accuracy on valid pixels (trainId != 255)
            valid = masks_trainid != 255
            if valid.any():
                correct = (preds_trainid[valid] == masks_trainid[valid]).sum().item()
                total = valid.sum().item()
                pixel_correct += correct
                pixel_total += total
                batch_pixel_acc.append(correct / total)
            else:
                batch_pixel_acc.append(0.0)

            # Confusion for mIoU: ignore -> -1
            labels_eval = masks_trainid.clone()
            labels_eval[labels_eval == 255] = -1
            preds_eval = preds_trainid.clone().clamp(0, num_eval_classes - 1)

            conf_batch = _fast_confusion_matrix(preds_eval, labels_eval, num_eval_classes)
            conf_mat += conf_batch

            inter = torch.diag(conf_batch).to(torch.float32)
            union = conf_batch.sum(0).to(torch.float32) + conf_batch.sum(1).to(torch.float32) - inter
            ious = torch.where(union > 0, inter / union.clamp(min=1), torch.nan)
            batch_miou.append(torch.nanmean(ious).item())
            
                        # TensorBoard scalars per batch
            if writer is not None:
                writer.add_scalar("test/batch_pixel_acc", batch_pixel_acc[-1], batch_idx)
                writer.add_scalar("test/batch_miou", batch_miou[-1], batch_idx)
                writer.add_scalar("test/batch_latency_ms", (batch_times[-1] * 1000.0), batch_idx)
                writer.add_scalar("test/per_image_latency_ms", (per_image_times[-1] * 1000.0), batch_idx)

            # Visualize a few
            for i in range(min(save_vis_per_batch, images.size(0))):
                visualize_results(
                    images[i],
                    masks_trainid[i],
                    preds_trainid[i],
                    f"{batch_idx}_{i}",
                )

    # Final metrics
    inter = torch.diag(conf_mat).to(torch.float32)
    union = conf_mat.sum(0).to(torch.float32) + conf_mat.sum(1).to(torch.float32) - inter
    ious = torch.where(union > 0, inter / union.clamp(min=1), torch.nan)

    mean_iou_per_class = ious.detach().cpu().numpy()
    mean_iou_all = float(np.nanmean(mean_iou_per_class))

    all_pixel_acc = (pixel_correct / pixel_total) if pixel_total else 0.0

    total_time = float(np.sum(batch_times))
    fps = (len(test_dataset) / total_time) if total_time > 0 else 0.0
    
    # Latency stats
    batch_times_np = np.array(batch_times, dtype=np.float64) * 1000.0
    per_image_times_np = np.array(per_image_times, dtype=np.float64) * 1000.0

    batch_lat = _latency_stats(batch_times_np)
    img_lat = _latency_stats(per_image_times_np)

    logger.info(f"Batch latency ms: mean={batch_lat['mean_ms']:.2f} p50={batch_lat['p50_ms']:.2f} p95={batch_lat['p95_ms']:.2f} p99={batch_lat['p99_ms']:.2f}")
    logger.info(f"Per-image latency ms: mean={img_lat['mean_ms']:.2f} p50={img_lat['p50_ms']:.2f} p95={img_lat['p95_ms']:.2f} p99={img_lat['p99_ms']:.2f}")

    # TensorBoard summary scalars (single “run” point)
    if writer is not None:
        writer.add_scalar("test/pixel_acc", all_pixel_acc, 0)
        writer.add_scalar("test/miou", mean_iou_all, 0)
        writer.add_scalar("test/fps_dataset", fps, 0)

        writer.add_scalar("latency/batch_mean_ms", batch_lat["mean_ms"], 0)
        writer.add_scalar("latency/batch_p50_ms", batch_lat["p50_ms"], 0)
        writer.add_scalar("latency/batch_p95_ms", batch_lat["p95_ms"], 0)
        writer.add_scalar("latency/batch_p99_ms", batch_lat["p99_ms"], 0)

        writer.add_scalar("latency/per_image_mean_ms", img_lat["mean_ms"], 0)
        writer.add_scalar("latency/per_image_p50_ms", img_lat["p50_ms"], 0)
        writer.add_scalar("latency/per_image_p95_ms", img_lat["p95_ms"], 0)
        writer.add_scalar("latency/per_image_p99_ms", img_lat["p99_ms"], 0)

    logger.info(f"Pixel Accuracy: {all_pixel_acc:.4f}")
    logger.info(f"Mean IoU per class: {mean_iou_per_class}")
    logger.info(f"Mean IoU (overall): {mean_iou_all:.4f}")
    logger.info(f"Model FPS (val resized {H}x{W}): {fps:.2f}")

    # Plots
    plt.figure(figsize=(10, 5))
    plt.plot(batch_pixel_acc, label="Pixel Accuracy")
    plt.xlabel("Batch")
    plt.ylabel("Pixel Accuracy")
    plt.title("Pixel Accuracy per batch")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "pixel_accuracy.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(batch_miou, label="Mean IoU per batch")
    plt.xlabel("Batch")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU per batch")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "mean_iou.png"))
    plt.close()

        # Paper-ready plots
    _plot_latency_hist(
        per_image_times_np,
        out_path=os.path.join(plots_dir, "latency_hist_per_image_ms.png"),
        title=f"Per-image latency histogram ({model_name} @ {H}x{W})"
    )
    _plot_latency_hist(
        batch_times_np,
        out_path=os.path.join(plots_dir, "latency_hist_batch_ms.png"),
        title=f"Batch latency histogram ({model_name} @ {H}x{W})"
    )

    _plot_per_class_iou(
        mean_iou_per_class,
        out_path=os.path.join(plots_dir, "per_class_iou.png"),
        title=f"Per-class IoU (trainId) ({model_name})"
    )

    conf_cpu = conf_mat.detach().cpu().numpy()
    _plot_confusion_matrix(
        conf_cpu,
        out_path=os.path.join(plots_dir, "confusion_matrix_norm.png"),
        title=f"Confusion Matrix (row-normalized) ({model_name})",
        normalize=True
    )

    _plot_miou_vs_latency_point(
        miou=mean_iou_all,
        p50_ms=img_lat["p50_ms"],
        out_path=os.path.join(plots_dir, "miou_vs_latency_p50.png"),
        title=f"mIoU vs Latency (P50) ({model_name})"
    )
    
    # Save metrics
    metrics_path = os.path.join(run_dir, "metrics.txt")
    
    with open(metrics_path, "w", encoding="utf-8") as f:

        f.write("========== MODEL ==========\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Ablation: {ablation_cfg.id() or 'none'}\n")
        f.write(f"Weights: {weights_path}\n")
        f.write(f"Device (eval): {device_obj}\n")
        f.write(f"Image size: {H}x{W}\n")
        if gflops is not None:
            f.write(f"GFLOPs (@{H}x{W}): {gflops:.3f}\n")
            f.write(f"Parameters: {params_m:.3f} M\n")
        f.write(f"Output classes in model: {num_classes}\n")
        f.write(f"Eval classes: 19 (trainId)\n\n")

        f.write("========== QUALITY ==========\n")
        f.write(f"Pixel Accuracy: {all_pixel_acc:.6f}\n")
        f.write(f"Mean IoU (overall): {mean_iou_all:.6f}\n")
        f.write("IoU per class (trainId 0..18):\n")
        for i, v in enumerate(mean_iou_per_class):
            class_name = cityscapes_classes[i]
            f.write(f"  Class {i:02d} Name {class_name:<15}: {float(v):.6f}\n")
        f.write("\n")

        f.write("========== DATASET LOOP PERFORMANCE ==========\n")
        f.write(f"Dataset FPS (@{H}x{W}): {fps:.2f}\n")
        if 'batch_lat' in locals():
            f.write("\n[Batch latency over DataLoader loop]\n")
            f.write(f"Mean [ms]: {batch_lat['mean_ms']:.3f}\n")
            f.write(f"P50  [ms]: {batch_lat['p50_ms']:.3f}\n")
            f.write(f"P95  [ms]: {batch_lat['p95_ms']:.3f}\n")
            f.write(f"P99  [ms]: {batch_lat['p99_ms']:.3f}\n")
            f.write(f"Min  [ms]: {batch_lat['min_ms']:.3f}\n")
            f.write(f"Max  [ms]: {batch_lat['max_ms']:.3f}\n")

        if 'img_lat' in locals():
            f.write("\n[Per-image latency over DataLoader loop]\n")
            f.write(f"Mean [ms]: {img_lat['mean_ms']:.3f}\n")
            f.write(f"P50  [ms]: {img_lat['p50_ms']:.3f}\n")
            f.write(f"P95  [ms]: {img_lat['p95_ms']:.3f}\n")
            f.write(f"P99  [ms]: {img_lat['p99_ms']:.3f}\n")
            f.write(f"Min  [ms]: {img_lat['min_ms']:.3f}\n")
            f.write(f"Max  [ms]: {img_lat['max_ms']:.3f}\n")

        # ---------------- CPU BENCH ----------------
        if cpu_bench is not None:
            cpu_fps = cpu_bench["fps"]
            cpu_lat = cpu_bench["latency_ms"]
            f.write("\n========== CPU BENCH (E2E: data+forward+argmax | batch=1) ==========\n")
            f.write(f"CPU threads: {cpu_threads}\n")
            f.write(f"Warmup iters: {cpu_warmup_iters}\n")
            f.write(f"Timed iters: {cpu_timed_iters}\n")
            f.write(f"FPS: {cpu_fps:.2f}\n")
            f.write(f"Latency mean [ms]: {cpu_lat['mean_ms']:.3f}\n")
            f.write(f"Latency P50  [ms]: {cpu_lat['p50_ms']:.3f}\n")
            f.write(f"Latency P95  [ms]: {cpu_lat['p95_ms']:.3f}\n")
            f.write(f"Latency P99  [ms]: {cpu_lat['p99_ms']:.3f}\n")
            f.write(f"Latency min  [ms]: {cpu_lat['min_ms']:.3f}\n")
            f.write(f"Latency max  [ms]: {cpu_lat['max_ms']:.3f}\n")

        # ---------------- GPU BENCH ----------------
        if gpu_bench is not None:
            gpu_fps = gpu_bench["fps"]
            gpu_lat = gpu_bench["latency_ms"]
            gbs = gpu_bench["batch_size"]
            f.write("\n========== GPU BENCH (forward-only | synthetic | sync) ==========\n")
            f.write(f"GPU batch size: {gbs}\n")
            f.write(f"Warmup iters: {gpu_warmup_iters}\n")
            f.write(f"Timed iters: {gpu_timed_iters}\n")
            f.write(f"FPS (throughput): {gpu_fps:.2f}\n")
            f.write(f"Latency mean [ms]: {gpu_lat['mean_ms']:.3f}\n")
            f.write(f"Latency P50  [ms]: {gpu_lat['p50_ms']:.3f}\n")
            f.write(f"Latency P95  [ms]: {gpu_lat['p95_ms']:.3f}\n")
            f.write(f"Latency P99  [ms]: {gpu_lat['p99_ms']:.3f}\n")
            f.write(f"Latency min  [ms]: {gpu_lat['min_ms']:.3f}\n")
            f.write(f"Latency max  [ms]: {gpu_lat['max_ms']:.3f}\n")     
            
    if writer is not None:
        writer.close()
        
    logger.info("Testing completed!")

    # iou18, inter, union, gt_cnt, pr_cnt = direct_iou_for_class(model, test_loader, cls=18)
    # print("IoU18 direct:", iou18)
    # print("inter:", inter, "union:", union, "gt:", gt_cnt, "pred:", pr_cnt)
    
    # gtc, prc = count_pred_pixels(model, test_loader, num_classes=num_eval_classes, device=device_obj)
    # print("GT class18 pixels:", gtc[18].item())
    # print("Pred class18 pixels:", prc[18].item())
    
    # gt_counts = count_gt_pixels_only(test_loader, device_obj, lut)
    # print("GT counts:", gt_counts.tolist())
    
    
    
if __name__ == "__main__":
    # for c in CS.classes:
    #     print(c)
    test_main()
