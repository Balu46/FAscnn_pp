#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse multiple metrics.txt files (one per model) and generate paper-style comparative plots.

Usage examples:
  python plot_metrics.py --root /path/to/experiments --pattern "metrics.txt" --outdir ./plots
  python plot_metrics.py --files a/metrics.txt b/metrics.txt c/metrics.txt --outdir ./plots
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import glob
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


CITYSCAPES_19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]


@dataclass
class Metrics:
    path: str
    model: str = "unknown"
    ablation: str = "unknown"
    image_size: str = "unknown"

    pixel_acc: Optional[float] = None
    miou: Optional[float] = None

    # Speed / latency
    dataset_fps: Optional[float] = None

    cpu_fps: Optional[float] = None
    cpu_latency_mean_ms: Optional[float] = None
    cpu_latency_p50_ms: Optional[float] = None
    cpu_latency_p95_ms: Optional[float] = None
    cpu_latency_p99_ms: Optional[float] = None

    gpu_fps: Optional[float] = None
    gpu_latency_mean_ms: Optional[float] = None
    gpu_latency_p50_ms: Optional[float] = None
    gpu_latency_p95_ms: Optional[float] = None
    gpu_latency_p99_ms: Optional[float] = None

    # IoU per class: trainId -> float
    iou_per_class: Dict[int, float] = field(default_factory=dict)


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_metrics_file(path: str) -> Metrics:
    m = Metrics(path=path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Header-ish fields
    # Model: FAscnn_pp_V14
    mm = re.search(r"^\s*Model:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if mm:
        m.model = mm.group(1).strip()

    # Ablation: none
    mm = re.search(r"^\s*Ablation:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if mm:
        m.ablation = mm.group(1).strip()

    # Image size: 1024x2048
    mm = re.search(r"^\s*Image size:\s*([0-9]+x[0-9]+)\s*$", text, flags=re.MULTILINE)
    if mm:
        m.image_size = mm.group(1).strip()

    # Quality
    mm = re.search(r"^\s*Pixel Accuracy:\s*([0-9]*\.?[0-9]+)\s*$", text, flags=re.MULTILINE)
    if mm:
        m.pixel_acc = _safe_float(mm.group(1))

    mm = re.search(r"^\s*Mean IoU\s*\(overall\):\s*([0-9]*\.?[0-9]+)\s*$", text, flags=re.MULTILINE)
    if mm:
        m.miou = _safe_float(mm.group(1))

    # Dataset FPS
    mm = re.search(r"^\s*Dataset FPS.*?:\s*([0-9]*\.?[0-9]+)\s*$", text, flags=re.MULTILINE)
    if mm:
        m.dataset_fps = _safe_float(mm.group(1))

    # CPU BENCH FPS and latency stats
    # FPS: 0.66
    cpu_block = re.search(r"={5,}\s*CPU BENCH.*?={5,}([\s\S]*?)(?:(={5,}\s*GPU BENCH)|\Z)", text)
    if cpu_block:
        block = cpu_block.group(1)
        mm = re.search(r"^\s*FPS:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.cpu_fps = _safe_float(mm.group(1))
        mm = re.search(r"^\s*Latency mean \[ms\]:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.cpu_latency_mean_ms = _safe_float(mm.group(1))
        mm = re.search(r"^\s*Latency P50\s*\[ms\]:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.cpu_latency_p50_ms = _safe_float(mm.group(1))
        mm = re.search(r"^\s*Latency P95\s*\[ms\]:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.cpu_latency_p95_ms = _safe_float(mm.group(1))
        mm = re.search(r"^\s*Latency P99\s*\[ms\]:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.cpu_latency_p99_ms = _safe_float(mm.group(1))

    # GPU BENCH FPS and latency stats
    gpu_block = re.search(r"={5,}\s*GPU BENCH.*?={5,}([\s\S]*?)\Z", text)
    if gpu_block:
        block = gpu_block.group(1)
        mm = re.search(r"^\s*FPS\s*\(throughput\):\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.gpu_fps = _safe_float(mm.group(1))
        mm = re.search(r"^\s*Latency mean \[ms\]:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.gpu_latency_mean_ms = _safe_float(mm.group(1))
        mm = re.search(r"^\s*Latency P50\s*\[ms\]:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.gpu_latency_p50_ms = _safe_float(mm.group(1))
        mm = re.search(r"^\s*Latency P95\s*\[ms\]:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.gpu_latency_p95_ms = _safe_float(mm.group(1))
        mm = re.search(r"^\s*Latency P99\s*\[ms\]:\s*([0-9]*\.?[0-9]+)\s*$", block, flags=re.MULTILINE)
        if mm:
            m.gpu_latency_p99_ms = _safe_float(mm.group(1))

    # IoU per class lines:
    #   Class 00 Name road           : 0.933303
    for cls, name, val in re.findall(
        r"^\s*Class\s+(\d+)\s+Name\s+(.+?)\s*:\s*([0-9]*\.?[0-9]+)\s*$",
        text,
        flags=re.MULTILINE,
    ):
        cid = int(cls)
        m.iou_per_class[cid] = float(val)

    return m


def discover_files(root: str, pattern: str) -> List[str]:
    # Recursively find matches (pattern like "metrics.txt")
    # Use ** to search deeper.
    glob_pat = os.path.join(root, "**", pattern)
    files = glob.glob(glob_pat, recursive=True)
    files = [p for p in files if os.path.isfile(p)]
    # Filter out ablation steps
    files = [p for p in files if "Step" not in p]
    return sorted(files)


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def paper_style():
    # "Paper-like" defaults (no seaborn).
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "pdf.fonttype": 42,   # better embedding in PDFs
        "ps.fonttype": 42,
    })


def model_label(m: Metrics) -> str:
    # Compact label for plots; you can change this formatting.
    if m.ablation and m.ablation.lower() != "unknown" and m.ablation.lower() != "none":
        return f"{m.model}\n({m.ablation})"
    return m.model


def save_csv(rows: List[Metrics], out_csv: str) -> None:
    # Flatten IoU per class into columns iou_00..iou_18
    fieldnames = [
        "model", "ablation", "image_size", "path",
        "pixel_acc", "miou",
        "dataset_fps",
        "cpu_fps", "cpu_latency_mean_ms", "cpu_latency_p50_ms", "cpu_latency_p95_ms", "cpu_latency_p99_ms",
        "gpu_fps", "gpu_latency_mean_ms", "gpu_latency_p50_ms", "gpu_latency_p95_ms", "gpu_latency_p99_ms",
    ] + [f"iou_{i:02d}" for i in range(19)]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = {
                "model": r.model,
                "ablation": r.ablation,
                "image_size": r.image_size,
                "path": r.path,
                "pixel_acc": r.pixel_acc,
                "miou": r.miou,
                "dataset_fps": r.dataset_fps,
                "cpu_fps": r.cpu_fps,
                "cpu_latency_mean_ms": r.cpu_latency_mean_ms,
                "cpu_latency_p50_ms": r.cpu_latency_p50_ms,
                "cpu_latency_p95_ms": r.cpu_latency_p95_ms,
                "cpu_latency_p99_ms": r.cpu_latency_p99_ms,
                "gpu_fps": r.gpu_fps,
                "gpu_latency_mean_ms": r.gpu_latency_mean_ms,
                "gpu_latency_p50_ms": r.gpu_latency_p50_ms,
                "gpu_latency_p95_ms": r.gpu_latency_p95_ms,
                "gpu_latency_p99_ms": r.gpu_latency_p99_ms,
            }
            for i in range(19):
                d[f"iou_{i:02d}"] = r.iou_per_class.get(i, np.nan)
            w.writerow(d)


def bar_plot(values: List[Optional[float]], labels: List[str], title: str, ylabel: str,
             outpath_base: str, sort_desc: bool = True) -> None:
    v = np.array([np.nan if x is None else float(x) for x in values], dtype=float)
    order = np.argsort(-v) if sort_desc else np.arange(len(v))
    v = v[order]
    labs = [labels[i] for i in order]

    plt.figure(figsize=(max(6.5, 0.45 * len(labs)), 3.8))
    plt.bar(np.arange(len(v)), v)
    plt.xticks(np.arange(len(v)), labs, rotation=35, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.savefig(outpath_base + ".pdf")
    plt.savefig(outpath_base + ".png")
    plt.close()


def scatter_tradeoff(rows: List[Metrics], outpath_base: str) -> None:
    # Scatter: mIoU vs FPS for dataset/cpu/gpu (whichever exists)
    plt.figure(figsize=(6.2, 4.2))
    for kind, fps_getter, marker in [
        ("Dataset FPS", lambda r: r.dataset_fps, "o"),
        ("CPU FPS", lambda r: r.cpu_fps, "s"),
        ("GPU FPS", lambda r: r.gpu_fps, "^"),
    ]:
        xs, ys, labs = [], [], []
        for r in rows:
            fps = fps_getter(r)
            if fps is None or r.miou is None:
                continue
            xs.append(fps)
            ys.append(r.miou)
            labs.append(r.model)
        if xs:
            plt.scatter(xs, ys, marker=marker, label=kind)

    plt.xlabel("FPS (higher is better)")
    plt.ylabel("mIoU (higher is better)")
    plt.title("Speed–Accuracy Trade-off")
    plt.legend(frameon=True)
    plt.tight_layout()
    # plt.savefig(outpath_base + ".pdf")
    plt.savefig(outpath_base + ".png")
    plt.close()


def heatmap_iou(rows: List[Metrics], outpath_base: str, class_names: List[str] = CITYSCAPES_19) -> None:
    labels = [model_label(r) for r in rows]
    data = np.full((len(rows), 19), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for c in range(19):
            if c in r.iou_per_class:
                data[i, c] = r.iou_per_class[c]

    # Sort by mIoU if available
    miou = np.array([np.nan if r.miou is None else r.miou for r in rows], dtype=float)
    order = np.argsort(-miou)
    data = data[order]
    labels = [labels[i] for i in order]

    plt.figure(figsize=(10.8, max(3.2, 0.38 * len(labels))))
    im = plt.imshow(data, aspect="auto")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(19), class_names, rotation=35, ha="right")
    plt.title("IoU per class (Cityscapes trainId 0..18)")
    cbar = plt.colorbar(im, fraction=0.02, pad=0.02)
    cbar.set_label("IoU")
    plt.tight_layout()
    # plt.savefig(outpath_base + ".pdf")
    plt.savefig(outpath_base + ".png")
    plt.close()


def latency_plot(rows: List[Metrics], outpath_base: str) -> None:
    # Plot P50/P95/P99 for CPU and GPU (if available)
    labels = [model_label(r) for r in rows]

    def _stack(getters: List[Tuple[str, callable]]):
        mat = []
        for name, fn in getters:
            mat.append([np.nan if fn(r) is None else float(fn(r)) for r in rows])
        return np.array(mat, dtype=float)

    cpu = _stack([
        ("CPU p50", lambda r: r.cpu_latency_p50_ms),
        ("CPU p95", lambda r: r.cpu_latency_p95_ms),
        ("CPU p99", lambda r: r.cpu_latency_p99_ms),
    ])
    gpu = _stack([
        ("GPU p50", lambda r: r.gpu_latency_p50_ms),
        ("GPU p95", lambda r: r.gpu_latency_p95_ms),
        ("GPU p99", lambda r: r.gpu_latency_p99_ms),
    ])

    def _plot(mat: np.ndarray, title: str, outbase: str):
        if np.all(np.isnan(mat)):
            return
        plt.figure(figsize=(max(6.8, 0.45 * len(labels)), 3.8))
        x = np.arange(len(labels))
        w = 0.25
        for i in range(mat.shape[0]):
            plt.bar(x + (i - 1) * w, mat[i], width=w, label=["p50", "p95", "p99"][i])
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Latency [ms] (lower is better)")
        plt.title(title)
        plt.legend(frameon=True)
        plt.tight_layout()
        # plt.savefig(outbase + ".pdf")
        plt.savefig(outbase + ".png")
        plt.close()

    _plot(cpu, "CPU latency percentiles (E2E, batch=1)", outpath_base + "_cpu")
    _plot(gpu, "GPU latency percentiles (forward-only, synthetic)", outpath_base + "_gpu")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=str, help="Root directory to search recursively.")
    src.add_argument("--files", nargs="+", help="Explicit list of metrics.txt files.")
    ap.add_argument("--pattern", type=str, default="metrics.txt", help='Filename pattern, e.g. "metrics.txt"')
    ap.add_argument("--outdir", type=str, default="./plots", help="Output directory for plots and CSV.")
    ap.add_argument("--nosort", action="store_true", help="Disable sorting (default sorts by metric).")
    args = ap.parse_args()

    if args.files:
        files = [os.path.abspath(p) for p in args.files]
    else:
        files = discover_files(args.root, args.pattern)

    if not files:
        raise SystemExit("No metrics files found. Check --root/--pattern or --files.")

    rows: List[Metrics] = []
    for p in files:
        try:
            rows.append(parse_metrics_file(p))
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")

    # Keep stable order unless you want a different sorting policy globally.
    # We'll still sort in plot functions when it makes sense.
    ensure_outdir(args.outdir)
    paper_style()

    out_csv = os.path.join(args.outdir, "metrics_summary.csv")
    save_csv(rows, out_csv)
    print(f"[OK] Wrote CSV: {out_csv}")

    labels = [model_label(r) for r in rows]

    # Bars
    bar_plot([r.miou for r in rows], labels,
             title="Mean IoU (overall)", ylabel="mIoU",
             outpath_base=os.path.join(args.outdir, "bar_miou"),
             sort_desc=not args.nosort)

    bar_plot([r.pixel_acc for r in rows], labels,
             title="Pixel Accuracy", ylabel="Accuracy",
             outpath_base=os.path.join(args.outdir, "bar_pixel_acc"),
             sort_desc=not args.nosort)

    bar_plot([r.dataset_fps for r in rows], labels,
             title="Dataset FPS (@given image size)", ylabel="FPS",
             outpath_base=os.path.join(args.outdir, "bar_dataset_fps"),
             sort_desc=not args.nosort)

    bar_plot([r.cpu_fps for r in rows], labels,
             title="CPU FPS (E2E, batch=1)", ylabel="FPS",
             outpath_base=os.path.join(args.outdir, "bar_cpu_fps"),
             sort_desc=not args.nosort)

    bar_plot([r.gpu_fps for r in rows], labels,
             title="GPU FPS (throughput)", ylabel="FPS",
             outpath_base=os.path.join(args.outdir, "bar_gpu_fps"),
             sort_desc=not args.nosort)

    # Trade-off scatter
    scatter_tradeoff(rows, os.path.join(args.outdir, "scatter_speed_vs_miou"))

    # IoU heatmap
    heatmap_iou(rows, os.path.join(args.outdir, "heatmap_iou_per_class"), CITYSCAPES_19)

    # Latency
    latency_plot(rows, os.path.join(args.outdir, "latency_percentiles"))

    print(f"[OK] Plots saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()  
    
    
# odpalenie 
# python visualisation/comparison.py --root results --pattern metrics.txt --outdir results/plots