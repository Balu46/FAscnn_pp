#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FigureSpec:
    alias: str
    patterns: tuple[str, ...]


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUTDIR = REPO_ROOT / "praca_licencjacka" / "cała_praca" / "figures_auto"
SUPPORTED_EXTS = (".png", ".pdf", ".jpg", ".jpeg")

FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        alias="bar_miou",
        patterns=(
            "**/bar_miou.*",
            "**/*bar*miou*.*",
            "**/*miou*bar*.*",
        ),
    ),
    FigureSpec(
        alias="bar_pixel_acc",
        patterns=(
            "**/bar_pixel_acc.*",
            "**/*pixel*acc*.*",
            "**/*accuracy*bar*.*",
        ),
    ),
    FigureSpec(
        alias="bar_cpu_fps",
        patterns=(
            "**/bar_cpu_fps.*",
            "**/*cpu*fps*.*",
        ),
    ),
    FigureSpec(
        alias="bar_gpu_fps",
        patterns=(
            "**/bar_gpu_fps.*",
            "**/*gpu*fps*.*",
        ),
    ),
    FigureSpec(
        alias="bar_dataset_fps",
        patterns=(
            "**/bar_dataset_fps.*",
            "**/*dataset*fps*.*",
        ),
    ),
    FigureSpec(
        alias="scatter_speed_vs_miou",
        patterns=(
            "**/scatter_speed_vs_miou.*",
            "**/*speed*miou*.*",
            "**/*miou*speed*.*",
            "**/*trade*off*.*",
        ),
    ),
    FigureSpec(
        alias="heatmap_iou_per_class",
        patterns=(
            "**/heatmap_iou_per_class.*",
            "**/*heatmap*iou*.*",
            "**/*iou*per*class*.*",
        ),
    ),
    FigureSpec(
        alias="latency_percentiles_cpu",
        patterns=(
            "**/latency_percentiles_cpu.*",
            "**/*latency*cpu*.*",
        ),
    ),
    FigureSpec(
        alias="latency_percentiles_gpu",
        patterns=(
            "**/latency_percentiles_gpu.*",
            "**/*latency*gpu*.*",
        ),
    ),
)


def is_supported(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def newest_match(spec: FigureSpec) -> Path | None:
    for pattern in spec.patterns:
        seen: dict[Path, float] = {}
        for path in RESULTS_DIR.glob(pattern):
            if is_supported(path):
                seen[path.resolve()] = path.stat().st_mtime
        if seen:
            return max(seen, key=seen.get)
    return None


def unlink_old_aliases(alias: str) -> None:
    for ext in SUPPORTED_EXTS:
        candidate = OUTDIR / f"{alias}{ext}"
        if candidate.exists() or candidate.is_symlink():
            candidate.unlink()


def make_relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel_src)


def write_manifest(rows: list[tuple[str, Path | None]]) -> None:
    manifest = OUTDIR / "manifest.txt"
    lines = []
    for alias, path in rows:
        if path is None:
            lines.append(f"{alias}: MISSING")
        else:
            lines.append(f"{alias}: {path.relative_to(REPO_ROOT)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[tuple[str, Path | None]] = []
    for spec in FIGURES:
        unlink_old_aliases(spec.alias)
        match = newest_match(spec)
        manifest_rows.append((spec.alias, match))
        if match is None:
            print(f"[WARN] No figure found for alias: {spec.alias}")
            continue

        dst = OUTDIR / f"{spec.alias}{match.suffix.lower()}"
        make_relative_symlink(match, dst)
        print(f"[OK] {spec.alias} -> {match.relative_to(REPO_ROOT)}")

    write_manifest(manifest_rows)


if __name__ == "__main__":
    main()
