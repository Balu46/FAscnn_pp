# FAscnn_pp - Szybka Segmentacja Semantyczna (Fast Attention SCNN)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

## 📖 O czym jest ten projekt

**FAscnn_pp (Fast Atencion SCNN)** to wysokowydajny framework do semantycznej segmentacji obrazu w czasie rzeczywistym, zaprojektowany z myślą o pojazdach autonomicznych i urządzeniach brzegowych. Flagowy model, **FAscnn_pp_V18**, osiąga wysoką wydajność poprzez zrównoważenie dokładności (mIoU) z wysoką przepustowością (FPS - klatki na sekundę). 

Kluczowe innowacje i zalety tego projektu:
1. **FastAttention**: Lekki mechanizm uwagi, który wzmacnia globalne cechy na gałęzi kontekstowej bez znaczącego obciążania zasobów obliczeniowych.
2. **Fuzja w stylu BiSeNet**: Efektywne łączenie cech przestrzennych i kontekstowych.
3. **Głęboki nadzór (Deep Supervision)**: Wykorzystanie dodatkowych głowic w trakcie treningu poprawiających zbieżność i dokładność modelu.

Najlepszy model osiąga **70.68% mIoU** na zbiorze Cityscapes przy natywnej rozdzielczości 1024x2048 pikseli i imponującą prędkość **~343 FPS** na karcie graficznej GPU (RTX 5080) (tylko wnioskowanie).

---

## 🚀 Jak sklonować i skonfigurować projekt

### 1. Pobranie repozytorium
Sklonuj repozytorium projektu na swój komputer i przejdź do głównego katalogu:
```bash
git clone https://github.com/Balu46/FAscnn_pp.git
cd FAMB-fast-segmentation
```

### 2. Konfiguracja środowiska
Aby zainstalować wszystkie wymagane biblioteki, środowisko wirtualne dla Python 3.10+ oraz PyTorch z obsługą CUDA, wystarczy uruchomić przygotowany skrypt:
```bash
./setup.sh
```
Skrypt ten automatycznie utworzy wirtualne środowisko (`.venv`) i zainstaluje w nim pakiety z pliku `requirements.txt`. Wirtualne środowisko włącza się automatycznie w kolejnych skryptach.

### 3. Pobranie zbioru danych (Cityscapes)
Do treningu i ewaluacji modeli niezbędny jest zbiór Cityscapes. Możesz go w pełni automatycznie pobrać i odpowiednio zreorganizować, używając skryptu:
```bash
./download_data.sh
```
*(Skrypt pobierze dane oraz ułoży je we właściwej strukturze w folderze `cityscapes/`)*

---

## 🏃 Jak odtworzyć wyniki (Uruchamianie)

Do projektu dołączono dedykowane skrypty Bash, ułatwiające uruchamianie poszczególnych akcji (trening, testowanie). Przed ich wywołaniem upewnij się, że jesteś w głównym folderze projektu.

### Trening
Aby uruchomić trening modelu z domyślnymi parametrami (bądź z wybranego pliku konfiguracji), użyj skryptu:
```bash
./run_train.sh -c config_1024x2048.yaml
```

### Testowanie i Ewaluacja (na zbiorze walidacyjnym Cityscapes)
Aby przetestować jakość modelu (wygenerować metryki tj. mIoU):
```bash
./run_test.sh -m FAscnn_pp_V18
```
Dla sprawdzenia wszystkich wariantów modelu naraz, można użyć dołączonego skryptu zbiorczego:
```bash
./test_all_models.sh
```

### Odtworzenie pełnych wyników badawczych (Ablacje)
W procesie tworzenia architektury FAscnn_pp zrealizowano szereg badań ablacyjnych. Aby sekwencyjnie i w 100% odtworzyć kroki opisane w Pracy Licencjackiej (Tabela 5.1), od prostego modelu FastSCNN (Baseline) aż do ostatecznej zboostowanej architektury, uruchom po prostu:
```bash
./run_all_ablations.sh
```
Kolejne kroki zapiszą swoje wyniki, checkpointy oraz wykresy w odpowiednich podfolderach w katalogu `ablations/`.

---

## 📂 Struktura katalogów

*   `src/model_architecture/FAscnn_pp/`: Definicja rdzenia architektury modeli FAscnn_pp.
*   `src/data/`: Moduły do wgrywania obrazów i augmentacji na zbiorze Cityscapes.
*   `checkpoints/`: Przechowywane zapisy i pre-trenowane wagi.
*   `results/` / `ablations/`: Folder przechowujący wyjściowe dane z eksperymentów, metryki i wykresy.
*   `praca_licencjacka/`: Pliki źródłowe LaTeX zawierające właściwy tekst pracy.

---

## ⚖️ Licencja i cytowanie

Projekt udostępniany jest na warunkach **Licencji Apache 2.0**. 
Szczegóły znajdują się w pliku [LICENSE](LICENSE).

Jeżeli chcesz wspomnieć lub wykorzystać kod FAscnn_pp w swoich projektach/badaniach, prosimy o podanie referencji:
```text
[Jan Zakroczymski], "FAscnn_pp: Fast Atencion SCNN", Praca Licencjacka, UKSW, 2026.
```
