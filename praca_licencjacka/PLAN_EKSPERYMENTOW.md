# Plan eksperymentow: retroaktywnie i na przyszlosc

## Zalozenie czasu

Przyjmuje start projektu: **1 pazdziernika 2025**.  
Stan biezacy planu: **9 marca 2026**.

## A. Plan retroaktywny 

### Etap 0: 2025-10-01 do 2025-10-07 - ustawienie srodowiska

Cele:
- stabilne srodowisko uruchomieniowe
- poprawne pobranie danych
- test "czy pipeline dziala"

Zadania:
1. Instalacja i aktywacja venv.
2. Instalacja zaleznosci z `requirements.txt`.
3. Pobranie Cityscapes przez `download_data.sh`.
4. Suchy test treningu i testu (po 1-2 batchach).

Komendy bazowe:
```bash
./setup.sh
./download_data.sh
./run_train.sh -c config.yaml
./run_test.sh -c config.yaml
```

### Etap 1: 2025-10-08 do 2025-10-31 - baseline i walidacja pipeline

Cele:
- ustalic punkt odniesienia
- potwierdzic logowanie metryk

Zadania:
1. Uruchomic `FastSCNN` jako baseline.
2. Zweryfikowac: `mIoU`, `Pixel Accuracy`, `FPS`, latency.
3. Zapisac wyniki do jednej tabeli porownawczej.

### Etap 2: 2025-11-01 do 2025-12-15 - modele klasyczne

Cele:
- porownac latwe baseline'y z literatura

Zadania:
1. Trening i test: `ENet`, `ENetv2`, `ENetv3`, `FastSCNN`.
2. Te same ustawienia danych i epok dla uczciwego porownania.
3. Kontrola jakosci klas rzadkich (w tym trainId 18).

### Etap 3: 2025-12-16 do 2026-01-31 - rozwoj FAscnn_pp i porzadkowanie repo

Cele:
- wejscie w architekture FAscnn_pp
- przygotowac repo pod eksperymenty porownawcze

Zadania:
1. Trening i test: `FAscnn_pp_V3`, `FAscnn_pp_V6`, `FAscnn_pp_V11`, `FAscnn_pp_V12`.
2. Ustalic standard nazewnictwa wynikow i checkpointow.
3. Zamrozic jeden "konfig referencyjny" dla dalszych testow.

### Etap 4: 2026-02-01 do 2026-03-09 - testy porownawcze nowych wersji

Cele:
- porownac nowe wersje FAscnn_pp z baseline
- domknac etap retro na bazie historii git

Kluczowe commity do porownan:
- `a59a555` (2026-01-29) - train loop + IoU (przed migracja)
- `169ead5` (2026-02-24) - migracja train loop
- `c10b9c5` (2026-02-26) - `FAscnn_pp_V14` + poprawki `FAscnn_pp_V13`
- `d0e7c00` (2026-03-03) - poprawki FAscnn_pp + wizualizacje
- `b30b25d` (2026-03-03) - fix `ENetv2`

Aktualny snapshot wynikow (logi 2026-02-25 do 2026-02-27):
- `FastSCNN`: mIoU `0.6117`, GPU FPS `267.65`, CPU FPS `1.37`
- `FAscnn_pp_V13`: mIoU `0.5036`, GPU FPS `50.43`, CPU FPS `0.69`
- `FAscnn_pp_V14`: mIoU `0.5037`, GPU FPS `49.96`, CPU FPS `0.66`
- `FAscnn_pp_V15`: mIoU `0.3436`, GPU FPS `166.39`, CPU FPS `1.29`

Retro-aktywne zadania do domkniecia teraz:
1. Odtworzyc A/B: `a59a555` vs `169ead5` na `FastSCNN` i `FAscnn_pp_V13`.
2. Potwierdzic roznice miedzy `FAscnn_pp_V13`, `FAscnn_pp_V14`, `FAscnn_pp_V15` na tych samych seedach.
3. Sprawdzic przyczyne IoU `0.0` dla klasy 18.

## B. Plan na przyszlosc (od 2026-03-10)

### Etap 5: 2026-03-10 do 2026-03-31 - ablacje i stabilnosc

1. Ablacje dla najlepszego wariantu FAscnn_pp (`FAscnn_pp_V14`):
   - `none`
   - `no_attn`
   - `no_branch`
   - `no_fa1`
   - `no_fa2`
   - `no_fa3`
2. Kazdy wariant: minimum 3 seedy.
3. Raport: srednia i odchylenie dla mIoU oraz FPS.

### Etap 6: 2026-04-01 do 2026-04-20 - trade-off jakosc/szybkosc

1. Testy rozdzielczosci:
   - `512x1024`
   - `768x1536`
   - `1024x2048`
2. Testy CPU threads:
   - `1`
   - `4`
   - `8`
3. Wykres glowny: `mIoU vs FPS` + percentyle opoznien.

### Etap 7: 2026-04-21 do 2026-05-15 - finalizacja modelu

1. Wybor 1 modelu glównego + 1 modelu szybkiego.
2. Pelne testy koncowe (jeden config finalny).
3. Przygotowanie tabel do rozdzialu eksperymentalnego.

### Etap 8: 2026-05-16 do 2026-06-15 - material do pracy

1. Tabela: `commit -> model -> mIoU/FPS/latency`.
2. Tabela: ablacje i ich wplyw.
3. Wnioski: co dalo jakosc, co dalo szybkosc, jaki kompromis jest najlepszy.

## C. Standard dla kazdego eksperymentu

1. Ustalic model, commit, seed, config.
2. Trening przez `python3 -m src train --config ...`.
3. Test przez `python3 -m src test --config ...`.
4. Zapis metryk do jednego CSV zbiorczego.
5. Krotki wpis do dziennika: co zmieniono i po co.

## D. Kryteria sukcesu

- Powtarzalnosc: kluczowe wyniki potwierdzone na min. 3 seedach.
- Porownywalnosc: te same warunki danych i ewaluacji.
- Uzytecznosc do pracy: gotowe tabele i wykresy bez brakujacych metryk.
