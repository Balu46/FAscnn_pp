#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "results" / "plots" / "metrics_summary.csv"
OUT_PATH = REPO_ROOT / "praca_licencjacka" / "rozdzialy" / "05_Wyniki_Eksperymentow" / "auto_metrics.tex"
EXPORTED_CSV_PATH = REPO_ROOT / "praca_licencjacka" / "wyniki_auto" / "metrics_summary.csv"


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt_num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", "{,}")


def tex_escape(text: str) -> str:
    # Używamy ++ zamiast _pp zgodnie z prośbą użytkownika
    text = text.replace("_pp", "++")
    return text.replace("_", r"\_")


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {CSV_PATH}")
    # Filter out ablation steps
    rows = [r for r in rows if "Step" not in r["path"]]
    return rows


def best_by(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    return max(rows, key=lambda row: as_float(row, key))


def format_cell(row: dict[str, str], key: str, digits: int, baselines: list[dict[str, str]]) -> str:
    val = as_float(row, key)
    text = fmt_num(val, digits)
    if row["model"].startswith("FAscnn_pp_"):
        if baselines:
            max_base = max(as_float(b, key) for b in baselines)
            if val > max_base:
                return f"\\underline{{{text}}}"
    return text


def write_tex(rows: list[dict[str, str]]) -> None:
    # Grupowanie
    baseline_rows = sorted([row for row in rows if not row["model"].startswith("FAscnn_pp_")], 
                          key=lambda r: as_float(r, "miou"), reverse=True)
    fascnn_pp_rows = sorted([row for row in rows if row["model"].startswith("FAscnn_pp_")], 
                      key=lambda r: as_float(r, "miou"), reverse=True)
    
    best_models_list = ["FastSCNN", "FAscnn_pp_V17", "FAscnn_pp_V18"]
    best_rows = [r for r in rows if r["model"] in best_models_list]
    best_rows = sorted(best_rows, key=lambda r: as_float(r, "miou"), reverse=True)

    best_overall = best_by(rows, "miou")
    best_baseline = best_by(baseline_rows, "miou")
    best_fascnn_pp = best_by(fascnn_pp_rows, "miou")

    lines: list[str] = []
    lines.append("% This file is auto-generated. Do not edit manually.")
    lines.append(r"\subsection{Zestawienie wyników eksperymentalnych}")
    
    lines.append(
        "W niniejszej sekcji przedstawiono szczegółowe zestawienie wyników uzyskanych dla wszystkich "
        "testowanych architektur. Porównanie obejmuje zarówno miary jakości segmentacji (mIoU, Pixel Accuracy), "
        "powiązanych z rodziną FAscnn++ i modelami referencyjnymi, jak i parametry wydajnościowe (FPS na CPU i GPU)."
    )
    lines.append("")

    # Tabela 1: Najlepsze modele
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Porównanie kluczowych architektur (baseline vs najlepsze FAscnn++).}")
    lines.append(r"  \label{tab:best-models-comparison}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & mIoU & Pixel Acc. & CPU FPS & GPU FPS \\")
    lines.append(r"    \midrule")
    t1_baselines = [r for r in best_rows if not r["model"].startswith("FAscnn_pp_")]
    for row in best_rows:
        lines.append(
            "    "
            f"\\textbf{{{tex_escape(row['model'])}}} & "
            f"{format_cell(row, 'miou', 3, t1_baselines)} & "
            f"{format_cell(row, 'pixel_acc', 3, t1_baselines)} & "
            f"{format_cell(row, 'cpu_fps', 2, t1_baselines)} & "
            f"{format_cell(row, 'gpu_fps', 2, t1_baselines)} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Tabela 2: Wszystkie modele (zgrupowane)
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Pełne zestawienie wyników dla wszystkich przetestowanych modeli.}")
    lines.append(r"  \label{tab:all-models-summary}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & mIoU & Pixel Acc. & CPU FPS & GPU FPS \\")
    lines.append(r"    \midrule")
    lines.append(r"    \textit{Modele bazowe} & & & & \\")
    for row in baseline_rows:
        lines.append(
            "    "
            f"\\texttt{{{tex_escape(row['model'])}}} & "
            f"{fmt_num(as_float(row, 'miou'))} & "
            f"{fmt_num(as_float(row, 'pixel_acc'))} & "
            f"{fmt_num(as_float(row, 'cpu_fps'), 2)} & "
            f"{fmt_num(as_float(row, 'gpu_fps'), 2)} \\\\"
        )
    lines.append(r"    \midrule")
    lines.append(r"    \textit{Warianty FAscnn++} & & & & \\")
    for row in fascnn_pp_rows:
        # Pogrubienie najlepszych wariantów w dużej tabeli
        fmt = "\\textbf" if row["model"] in best_models_list else "\\texttt"
        lines.append(
            "    "
            f"{fmt}{{{tex_escape(row['model'])}}} & "
            f"{format_cell(row, 'miou', 3, baseline_rows)} & "
            f"{format_cell(row, 'pixel_acc', 3, baseline_rows)} & "
            f"{format_cell(row, 'cpu_fps', 2, baseline_rows)} & "
            f"{format_cell(row, 'gpu_fps', 2, baseline_rows)} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    write_tex(rows)
    print(f"[OK] Wrote LaTeX summary: {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
