# FAscnn_pp Fast Segmentation - Szybki Start (Quick Start)

Ten przewodnik pomoże Ci błyskawicznie skonfigurować środowisko, pobrać dane i odtworzyć wszystkie wyniki projektu.

## 1. Konfiguracja Środowiska

Najpierw skonfiguruj środowisko. Skrypt `setup.sh` automatycznie stworzy wirtualne środowisko (`.venv`) dla Pythona i zainstaluje niezbędne biblioteki.

```bash
./setup.sh
```

*(Wszystkie poniższe skrypty powłoki automatycznie aktywują to środowisko, nie musisz robić tego ręcznie).*

## 2. Pobranie Danych (Cityscapes)

Pobierz zbiór danych Cityscapes. Skrypt pobierze archiwum i wypakuje obrazy do odpowiedniej struktury folderów:

```bash
./download_data.sh
```

## 3. Trening i Ewaluacja

### Trening
Do trenowania służy skrypt `run_train.sh`. Oczekuje on podania pliku konfiguracyjnego YAML (np. `config_1024x2048.yaml`):

```bash
./run_train.sh -c config_1024x2048.yaml
```

### Testowanie
Do ewaluacji i przetestowania modelu na zbiorze walidacyjnym (wyliczenie mIoU i FPS) uruchom:

```bash
./run_test.sh -m FAscnn_pp_V18
```
*(Możesz użyć flag -b dla batch size, -d dla urządzenia np. cuda/cpu, czy -w by podać ścieżkę do wag).*

## 4. Odtworzenie pełnych wyników (Ablacje i zestawienia)

Jeśli chcesz krok po kroku odtworzyć przebieg badań i testów zawartych w pracy licencjackiej, dołączono do tego celu specjalne skrypty zbiorcze:

```bash
# Uruchamia sekwencyjnie badanie ablacyjne (Kroki od 1 do 6 z Tabeli 5.1):
./run_all_ablations.sh

# Przetestowanie wszystkich przygotowanych architektur w katalogu z jednoczesnym logowaniem metryk:
./test_all_models.sh
```

Wyniki, wykresy i checkpointy poszczególnych przebiegów znajdą się w nowych folderach: `results/` oraz `ablations/`.
