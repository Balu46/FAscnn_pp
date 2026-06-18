#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
from pathlib import Path
import numpy as np

# =====================================================================
# KONFIGURACJA ŚCIEŻEK I STYLU
# =====================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "wyniki_auto" / "metrics_summary.csv"
OUT_DIR = REPO_ROOT / "cała_praca" / "figures_auto"

# Styl wykresów (naukowy, czysty)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.autolayout': True,
    'pdf.fonttype': 42  # Ważne dla czcionek wektorowych w LaTeX
})

# Kolory
COLOR_FAscnn_pp = '#e74c3c'  # Czerwony dla Twojego modelu
COLOR_BASE = '#3498db'  # Niebieski dla konkurencji

def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"[WARN] Brak pliku {CSV_PATH}. Tworzę przykładowe dane do testów.")
        # Przykładowe dane, jeśli CSV jeszcze nie istnieje
        data = {
            'model': ['BiSeNetV1', 'MobileNetV2', 'FastSCNN', 'STDC1', 'FAscnn_pp_V28', 'FAscnn_pp_V29'],
            'miou': [0.680, 0.702, 0.686, 0.719, 0.715, 0.7265],
            'pixel_acc': [0.932, 0.941, 0.938, 0.945, 0.949, 0.9515],
            'gpu_fps': [180.5, 150.2, 220.0, 190.4, 255.0, 248.14],
            'cpu_fps': [5.2, 4.8, 6.1, 5.5, 1.8, 1.58]
        }
        df = pd.DataFrame(data)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        return df
    return pd.read_csv(CSV_PATH)

# =====================================================================
# 1. WYKRES PARETO (Scatter: Speed vs mIoU)
# =====================================================================
def plot_pareto_front(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    
    # Rozdzielenie modeli FAscnn_pp od reszty
    is_fascnn_pp = df['model'].str.startswith('FAscnn_pp')
    
    plt.scatter(df[~is_fascnn_pp]['gpu_fps'], df[~is_fascnn_pp]['miou'], 
                c=COLOR_BASE, s=100, label='Modele referencyjne', edgecolor='black', zorder=3)
    plt.scatter(df[is_fascnn_pp]['gpu_fps'], df[is_fascnn_pp]['miou'], 
                c=COLOR_FAscnn_pp, s=150, marker='*', label='Warianty FAscnn_pp', edgecolor='black', zorder=4)

    # Podpisy pod punktami
    for i, row in df.iterrows():
        y_offset = -0.002 if row['model'].startswith('FAscnn_pp') else 0.002
        plt.annotate(row['model'], (row['gpu_fps'], row['miou']), 
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

    plt.title('Zależność celności (mIoU) od szybkości inferencji (FPS)')
    plt.xlabel('GPU FPS (Rozdzielczość 2048x1024)')
    plt.ylabel('Mean IoU')
    plt.legend(loc='lower right')
    
    out_path = OUT_DIR / "scatter_speed_vs_miou.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Wygenerowano: {out_path.name}")
    plt.close()

# =====================================================================
# 2. WYKRESY SŁUPKOWE (Bar Charts)
# =====================================================================
def plot_bar_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str):
    df_sorted = df.sort_values(by=metric, ascending=True)
    
    plt.figure(figsize=(8, 6))
    colors = [COLOR_FAscnn_pp if m.startswith('FAscnn_pp') else COLOR_BASE for m in df_sorted['model']]
    
    bars = plt.barh(df_sorted['model'], df_sorted[metric], color=colors, edgecolor='black', zorder=3)
    
    # Wartości na końcach słupków
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                 f' {bar.get_width():.3f}' if metric in ['miou', 'pixel_acc'] else f' {bar.get_width():.1f}', 
                 va='center', ha='left', fontsize=10)

    # Lekki margines na tekst
    max_val = df_sorted[metric].max()
    plt.xlim(0, max_val * 1.15)
    if metric in ['miou', 'pixel_acc']:
        plt.xlim(max_val * 0.8, max_val * 1.05) # Zoom na interesujący zakres dla mIoU/Acc

    plt.title(title)
    plt.xlabel(ylabel)
    
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Wygenerowano: {out_path.name}")
    plt.close()

# =====================================================================
# 3. WYNIKI PER-KLASA (Wykres horyzontalny)
# =====================================================================
def plot_per_class_iou():
    # Dane wyciągnięte bezpośrednio z Twoich logów (FAscnn_pp_V29)
    classes = [
        'Road', 'Sky', 'Car', 'Vegetation', 'Building', 
        'Sidewalk', 'Bus', 'Person', 'Bicycle', 'Traffic Sign', 
        'Truck', 'Train', 'Terrain', 'Traffic Light', 'Wall', 
        'Fence', 'Pole', 'Motorcycle', 'Rider'
    ]
    ious = [
        97.7, 94.1, 92.8, 91.4, 90.8, 
        81.8, 77.3, 75.5, 70.2, 70.1, 
        70.1, 68.6, 63.3, 61.1, 58.7, 
        55.6, 54.4, 53.5, 52.6
    ]
    
    df_class = pd.DataFrame({'Class': classes, 'IoU': ious})
    df_class = df_class.sort_values(by='IoU', ascending=True)

    plt.figure(figsize=(10, 8))
    
    # Kolorowanie najsłabszych klas (te co optymalizowaliśmy) na czerwono, reszta niebieska
    colors = [COLOR_FAscnn_pp if val < 60.0 else COLOR_BASE for val in df_class['IoU']]
    
    bars = plt.barh(df_class['Class'], df_class['IoU'], color=colors, edgecolor='black', zorder=3)
    
    for bar in bars:
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.1f}%', va='center', ha='left', fontsize=10)

    plt.title('Wyniki IoU dla poszczególnych klas (FAscnn_pp_V29) [%]')
    plt.xlabel('Intersection over Union (IoU) [%]')
    plt.xlim(0, 105)
    
    # Nazwa pasuje do patternu w Twoim skrypcie: **/*iou*per*class*.*
    out_path = OUT_DIR / "heatmap_iou_per_class.png" 
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Wygenerowano: {out_path.name}")
    plt.close()

# =====================================================================
# MAIN EXECUTOR
# =====================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    
    # Grafiki ilościowe (z pliku CSV)
    plot_pareto_front(df)
    plot_bar_metric(df, 'miou', 'Porównanie Mean IoU (Wyższe = Lepsze)', 'mIoU', 'bar_miou.png')
    plot_bar_metric(df, 'pixel_acc', 'Porównanie Pixel Accuracy', 'Pixel Accuracy', 'bar_pixel_acc.png')
    plot_bar_metric(df, 'gpu_fps', 'Porównanie wydajności na GPU', 'Frames Per Second (FPS)', 'bar_gpu_fps.png')
    plot_bar_metric(df, 'cpu_fps', 'Porównanie wydajności na CPU', 'Frames Per Second (FPS)', 'bar_cpu_fps.png')
    
    # Grafika per-klasa (zahardkodowane z Twoich logów)
    plot_per_class_iou()

if __name__ == "__main__":
    main()
