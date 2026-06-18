#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as ea_mod
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[2]
TB_ROOT = REPO_ROOT / "log" / "tensorboard"
FIG_DIR = REPO_ROOT / "praca_licencjacka" / "cała_praca" / "figures_auto"
OUT_TEX = REPO_ROOT / "praca_licencjacka" / "rozdzialy" / "05_Wyniki_Eksperymentow" / "auto_tensorboard_summary.tex"
OUT_CSV = REPO_ROOT / "praca_licencjacka" / "wyniki_auto" / "tensorboard_summary.csv"
SIZE_GUIDANCE = {ea_mod.SCALARS: 2000}


@dataclass
class RunSummary:
    model: str
    event_path: Path
    train_loss_avg: list[tuple[int, float]]
    val_miou: list[tuple[int, float]]
    val_acc: list[tuple[int, float]]


def fmt_num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", "{,}")


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def select_best_event(model_dir: Path) -> Path | None:
    candidates = sorted(model_dir.glob("events.out.tfevents.*"))
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: (path.stat().st_size, path.stat().st_mtime),
    )


def collect_runs() -> list[RunSummary]:
    runs: list[RunSummary] = []
    for model_dir in sorted(path for path in TB_ROOT.iterdir() if path.is_dir()):
        event_path = select_best_event(model_dir)
        if event_path is None:
            continue
        print(f"[TB] Loading {model_dir.name} from {event_path.name}")
        ea = EventAccumulator(str(event_path), size_guidance=SIZE_GUIDANCE)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])

        def series(tag: str) -> list[tuple[int, float]]:
            if tag not in tags:
                return []
            return [(item.step, item.value) for item in ea.Scalars(tag)]

        runs.append(
            RunSummary(
                model=model_dir.name,
                event_path=event_path,
                train_loss_avg=series("train/loss_avg"),
                val_miou=series("val/mIoU"),
                val_acc=series("val/acc"),
            )
        )
    return runs


def plot_runs(runs: list[RunSummary], attr: str, title: str, ylabel: str, out_name: str) -> None:
    plt.figure(figsize=(9.4, 5.0))
    plotted = 0
    for run in runs:
        series: list[tuple[int, float]] = getattr(run, attr)
        if not series:
            continue
        plotted += 1
        plt.plot(
            [step for step, _ in series],
            [value for _, value in series],
            linewidth=1.4,
            label=run.model,
        )
    if plotted == 0:
        plt.close()
        return
    plt.xlabel("Krok treningowy")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{out_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_tex(runs: list[RunSummary]) -> None:
    lines: list[str] = []
    lines.append("% This file is auto-generated. Do not edit manually.")
    lines.append(r"\subsection{Automatyczne zestawienie TensorBoard dla wszystkich modeli}")
    lines.append(
        f"Na podstawie logów TensorBoard wykryto {len(runs)} modeli z przebiegami treningowymi. "
        "Dla każdego modelu automatycznie wybierany jest najpełniejszy plik eventów, a następnie "
        "wyliczane są końcowe wartości \\texttt{train/loss\\_avg}, \\texttt{val/mIoU} oraz \\texttt{val/acc}. "
        f"Zbiorcze dane tekstowe dla tej sekcji zapisane są w pliku \\texttt{{{tex_escape(str(OUT_CSV.relative_to(REPO_ROOT)))}}}."
    )
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Końcowe wartości metryk TensorBoard dla najpełniejszych przebiegów każdego modelu.}")
    lines.append(r"  \label{tab:tensorboard-summary}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & train/loss\_avg & val/mIoU & val/acc \\")
    lines.append(r"    \midrule")
    for run in runs:
        loss = run.train_loss_avg[-1][1] if run.train_loss_avg else None
        miou = run.val_miou[-1][1] if run.val_miou else None
        acc = run.val_acc[-1][1] if run.val_acc else None
        lines.append(
            "    "
            f"\\texttt{{{tex_escape(run.model)}}} & "
            f"{fmt_num(loss) if loss is not None else '--'} & "
            f"{fmt_num(miou) if miou is not None else '--'} & "
            f"{fmt_num(acc) if acc is not None else '--'} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(
        "Wykresy w dalszej części rozdziału pokazują zbiorczo przebiegi średniej straty treningowej, "
        "wartości \\textit{mIoU} na walidacji oraz dokładności walidacyjnej dla wszystkich dostępnych modeli."
    )
    lines.append("")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(runs: list[RunSummary]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "event_file", "train_loss_avg", "val_miou", "val_acc"])
        for run in runs:
            loss = run.train_loss_avg[-1][1] if run.train_loss_avg else ""
            miou = run.val_miou[-1][1] if run.val_miou else ""
            acc = run.val_acc[-1][1] if run.val_acc else ""
            writer.writerow([run.model, run.event_path.name, loss, miou, acc])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    if not runs:
        raise SystemExit(f"No TensorBoard runs found in {TB_ROOT}")

    # Wykresy zbiorcze (pozostawiamy jako ogólny przegląd)
    plot_runs(
        runs,
        attr="train_loss_avg",
        title="TensorBoard: train/loss_avg dla wszystkich modeli",
        ylabel="train/loss_avg",
        out_name="tb_all_train_loss_avg",
    )
    plot_runs(
        runs,
        attr="val_miou",
        title="TensorBoard: val/mIoU dla wszystkich modeli",
        ylabel="val/mIoU",
        out_name="tb_all_val_miou",
    )

    # Indywidualne wykresy dla 3 najlepszych architektur
    best_models = ["FastSCNN", "FAscnn_pp_V29", "FAscnn_pp_V31"]
    for model_name in best_models:
        model_runs = [r for r in runs if r.model == model_name]
        if not model_runs:
            print(f"[WARN] Brak danych dla modelu {model_name}")
            continue
        
        # Wykres mIoU
        plot_runs(
            model_runs,
            attr="val_miou",
            title=f"Krzywa uczenia mIoU: {model_name}",
            ylabel="val/mIoU",
            out_name=f"plot_miou_{model_name.lower()}",
        )
        # Wykres Loss
        plot_runs(
            model_runs,
            attr="train_loss_avg",
            title=f"Krzywa straty (loss): {model_name}",
            ylabel="train/loss_avg",
            out_name=f"plot_loss_{model_name.lower()}",
        )

    write_csv(runs)
    write_tex(runs)
    print(f"[OK] Wrote LaTeX summary: {OUT_TEX.relative_to(REPO_ROOT)}")
    print(f"[OK] Wrote CSV summary: {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"[OK] Processed TensorBoard models: {len(runs)}")


if __name__ == "__main__":
    main()
