# FAscnn_pp
**Optymalizacja architektury pod kątem systemów autonomicznych i urządzeń Edge.**

- **Domena:** Real-Time Computer Vision / Efficient Deep Learning
- **Target:** Cityscapes Benchmark (Native Resolution 1024x2048)
- **Target devices** Jetson Orin Nano / Xavier

---

## Kluczowe liczby
### **Przełamanie barier FPS/mIoU**
Kluczowe metryki modelu **FAscnn_pp_V34** na zbiorze Cityscapes (Val):

*   **Accuracy:** **73.21% mIoU** (natywna rozdzielczość 2MPix).
*   **Koszt obliczeniowy:** **16.014 GFLOPs** przy **1.176 M parametrów**.
*   **Przepustowość (Native 1024x2048, FP32):**
    *   RTX 3090: **165 FPS**
    *   RTX 5080: **285 FPS**


---

## Architektura 

### **Core Contributions & Architectural Novelties**

*   **Structural Reparameterization:**
    *   Trening: Wielogałęziowa architektura ekstrahująca bogate cechy przestrzenne.
    *   Inference: Metoda `switch_to_deploy()` – algebraiczna fuzja wag do jednej konwolucji.
*   **AxialBoundaryCalibration:**
    *   Mechanizm autokalibracji i wyostrzania granic obiektów.
    *   Kompresja osiowa wsparta konwolucjami (1xk, kx1) użyta do przestrzennego bramkowania cech wewnątrz pojedynczej warstwy.
*   **GuidedFusionGateV2:**
    *   Mechanizm fuzji cech sterowany krawędziami.
    *   Mapa wysokiej rozdzielczości (H/4) wsparta konwolucjami krzyżowymi użyta do bramkowania cech (H/8).
*   **Boundary-Aware Deep Supervision:**
    *   Zastosowanie głowic pomocniczych: `aux_border` oraz `aux_context`.
    *   GT dla aux border jest obliczane za pomocą maski krawędzi
---
## Protokół treningu 
###  Kluczowe parametry:

   * **Reżim optymalizacji**:
       * Schedule: Poly LR decay 
       * Batch Size: 8 (efektywny trening na natywnej rozdzielczości).
       * Użycie mixed precision dla zwiększenia batch size.
   * **Strategia Augmentacji**:
       * Native Resolution Crop: 1024x2048 – eliminacja artefaktów interpolacji, kluczowa dla segmentacji "cienkich" obiektów.
       * Randome Scale Training: Zakres skalowania [0.75, 2.0].
       * Random Flip 
       * Random Color jitter
       * Copy-Paste (p=0.3): Celowane wstrzykiwanie obiektów klas rzadkich (traffic light, traffic sign, rider, motorcycle, bicycle).
   * **Funkcja Straty**:
       * OHEM (Online Hard Example Mining): Parametr min_kept = 100,000 (wymuszona koncentracja na najtrudniejszych pikselach brzegowych).
       * Deep Supervision: Wsparcie głowic pomocniczych (aux_border, aux_context) z wagami odpowiednio 1.0 i 0.4.
   * **Class Balancing**: Rezygnacja z surowej statystyki (Inverse Frequency) na rzecz Manual Weight Boost dla krytycznych obiektów drogowych.

---

## Droga do 73.2% (Tabela ablacyjna)
### **Analiza Ablacyjna i Wnioski Badawcze (treningi po 100 epok)**

| Konfiguracja Modelu | Rozdzielczość | Zmiany Arch./Trening | mIoU |
| :--- | :--- | :--- | :--- |
| **Baseline (FastSCNN)** | 1024x2048 | – | 64.46% |
| **V17** | 1024x2048 | FastSCNN + bisenetFFM | 64.36% |
| **V18** | 1024x2048 | V17 + gated fast att | 65.50% |
| **V29** | 1024x2048 | V18 + border path + FAscnn_ppLoss (OHEM + Focal) + wagi boostowane manualnie | 67.53% | 
| **V29** | 1024x2048 | V29 + augmentacja copy paste | 67.77% |
| **V34** | 1024x2048 | poprzednie V29 + Reparametryzacja | 68.29% |
| **FINAL (V34)** | 1024x2048 | **Full Training (1000 epok)** | **73.21%** |

---

##  Porównanie mIoU per Class: V29 vs V34

### Tu wygrywa V34:
| Klasa | FAscnn_pp_V29 | FAscnn_pp_V34 | **Delta** | 
| :--- | :---: | :---: | :---: | 
| **Traffic Light** | 59.59% | **65.58%** | **+5.99%** | 
| **Pole (Słup)** | 54.10% | **60.18%** | **+6.08%** | 
| **Traffic Sign** | 69.71% | **74.72%** | **+5.01%** | 
| **Person** | 75.19% | **77.85%** | **+2.66%** | 
| **Bicycle** | 70.17% | **72.36%** | **+2.19%** |  
| **Motorcycle** | 54.55% | **57.75%** | **+3.20%** |


### Tu wygrywa V29:
| Klasa | FAscnn_pp_V29  | FAscnn_pp_V34 | **Delta** | 
| :--- | :---: | :---: | :---: | 
| **Train (Pociąg)** | **65.84%**| 58.75% | **+7.09%** | 
| **Wall (Ściana)** | **57.34%** | 52.72% | **+4.62%** |
| **Bus (Autobus)** | **79.10%** | 75.86% | **+3.24%** | 
| **Terrain (Teren)** | **63.18%** | 62.75% | **+0.43%** | 
| **Truck (Ciężarówka)** | **72.07%** | 71.72% | **+0.35%** |
| **Fence (Płot)** | **55.65%** | 55.41% | **+0.24%** | 

---


## Podsumowanie i Wizualizacja
### **Visual Performance & Architectural Insights**

![Główna Wizualizacja](../results/FAscnn_pp_V34/no_ablation/images/results_110_0.png)

*(Pełny widok: Image | Ground Truth | FAscnn_pp_V34.)*

