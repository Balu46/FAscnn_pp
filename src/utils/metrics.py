import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision.datasets import Cityscapes
from torchvision.datasets.cityscapes import Cityscapes as CS


@torch.no_grad()
def fast_confusion_miou(model: nn.Module,
                        val_loader: DataLoader,
                        num_classes: int,
                        device: torch.device,
                        ignore_index: int = 255) -> float:
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)  # [B,H,W]
        logits = model(images)                           # [B,C,H,W] or dict
        if isinstance(logits, dict):
            logits = logits["main"]
        
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        preds = torch.argmax(logits, dim=1)              # [B,H,W]

        # mask ignore
        mask = targets != ignore_index
        t = targets[mask].view(-1)
        p = preds[mask].view(-1)

        # bincount for confusion
        k = num_classes
        idx = t * k + p
        conf += torch.bincount(idx, minlength=k*k).view(k, k)

    # IoU per class
    tp = torch.diag(conf).to(torch.float32)
    fp = conf.sum(0).to(torch.float32) - tp
    fn = conf.sum(1).to(torch.float32) - tp
    denom = tp + fp + fn

    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    miou = float(iou.mean().item())
    return miou

@torch.no_grad()
def compute_pixel_accuracy(model: nn.Module,
                           val_loader: DataLoader,
                           device: torch.device,
                           ignore_index: int = 255):

    model.eval()

    correct = 0
    total = 0

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["main"]
            
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            
        preds = logits.argmax(dim=1)

        mask = targets != ignore_index

        correct += (preds[mask] == targets[mask]).sum().item()
        total += mask.sum().item()

    acc = correct / total if total > 0 else 0.0
    return acc

@torch.no_grad()
def compute_class_weights_from_loader(train_loader, num_classes: int, ignore_index: int = 255,
                                      max_batches: int | None = 200):
    counts = torch.zeros(num_classes, dtype=torch.long)
    total = 0

    for i, (_, y) in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break
        y = y.view(-1)
        y = y[y != ignore_index]
        if y.numel() == 0:
            continue
        total += y.numel()
        counts += torch.bincount(y, minlength=num_classes).cpu()

    # uniknij dzielenia przez zero
    counts = counts.clamp_min(1)

    # wagi odwrotnie proporcjonalne do częstości
    freq = counts.float() / counts.sum().float()
    weights = 1.0 / torch.log(1.02 + freq)   # stabilne, często działa lepiej niż 1/freq

    # normalizacja: średnia waga = 1 (żeby LR/loss scale nie odjechały)
    weights = weights / weights.mean()

    return weights, counts


@torch.no_grad()
def weights_logarithmic(counts: torch.Tensor, c: float = 1.02, eps: float = 1e-6):
    counts = counts.float()
    prob = counts / counts.sum()
    w = 1.0 / torch.log(c + prob + eps)
    w = w / w.mean()
    return w

@torch.no_grad()
def weights_median_frequency(counts: torch.Tensor, eps: float = 1e-6):
    counts = counts.float()
    freq = counts / counts.sum()
    med = freq[freq > 0].median()
    w = med / (freq + eps)
    w = w / w.mean()
    return w

@torch.no_grad()
def weights_inverse_freq(counts: torch.Tensor, eps: float = 1e-6):
    counts = counts.float()
    w = 1.0 / (counts + eps)
    w = w / w.mean()
    return w


def build_id_to_trainid():
    return {c.id: c.train_id for c in CS.classes}

def compute_class_counts_raw(city_root: str, num_classes: int = 19, ignore_index: int = 255):
    ds = Cityscapes(
        root=city_root,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=None,
        target_transform=None,
    )

    id_to_trainid = build_id_to_trainid()
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, mask_pil in ds:
        mask = np.array(mask_pil, dtype=np.int64)   # raw labelIds

        mapped = np.full_like(mask, ignore_index, dtype=np.int64)
        for label_id, train_id in id_to_trainid.items():
            if train_id == -1:
                continue
            mapped[mask == label_id] = train_id

        valid = mapped != ignore_index
        vals, cnts = np.unique(mapped[valid], return_counts=True)

        for v, c in zip(vals, cnts):
            if 0 <= v < num_classes:
                counts[v] += c

    return torch.tensor(counts, dtype=torch.long)