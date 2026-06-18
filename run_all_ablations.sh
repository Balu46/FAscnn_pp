#!/bin/bash
# Skrypt uruchamiający wszystkie 6 etapów ablacji zgodnie z Tabelą 5.1 z pracy licencjackiej

echo "=========================================================="
echo " Uruchamianie ablacji: Krok 1/6 - FastSCNN (Baseline)"
echo "=========================================================="
./run_ablation.sh --model FastSCNN --class-weights-type none --copy-paste-prob 0.0 --save-dir "ablations/Step1_Baseline"

echo "=========================================================="
echo " Uruchamianie ablacji: Krok 2/6 - Inna fuzja (BiSeNetFFM)"
echo " (FAscnn_pp_V18 bez modułu Fast Attention, bez CP, bez Wag)"
echo "=========================================================="
./run_ablation.sh --model FAscnn_pp_V18 --ablation no_attn --class-weights-type none --copy-paste-prob 0.0 --save-dir "ablations/Step2_no_attn"

echo "=========================================================="
echo " Uruchamianie ablacji: Krok 3/6 - Fast Attention"
echo " (FAscnn_pp_V18 z Fast Attention, bez CP, bez Wag)"
echo "=========================================================="
./run_ablation.sh --model FAscnn_pp_V18 --ablation none --class-weights-type none --copy-paste-prob 0.0 --save-dir "ablations/Step3_FastAttn"

echo "=========================================================="
echo " Uruchamianie ablacji: Krok 4/6 - Dodanie Copy-Paste"
echo " (FAscnn_pp_V18 z Fast Attention i Copy-Paste = 0.3, bez Wag)"
echo "=========================================================="
./run_ablation.sh --model FAscnn_pp_V18 --ablation none --class-weights-type none --copy-paste-prob 0.3 --save-dir "ablations/Step4_CP"

echo "=========================================================="
echo " Uruchamianie ablacji: Krok 5/6 - Class Weights (Standard)"
echo " (FAscnn_pp_V18 z Fast Attention, Copy-Paste = 0.3 i Standardowymi Wagami Klas)"
echo "=========================================================="
./run_ablation.sh --model FAscnn_pp_V18 --ablation none --class-weights-type standard --copy-paste-prob 0.3 --save-dir "ablations/Step5_Weights"

echo "=========================================================="
echo " Uruchamianie ablacji: Krok 6/6 - Manualny Boost Klas"
echo " (FAscnn_pp_V18 z Fast Attention, Copy-Paste = 0.3 i Ręcznie Zboostowanymi Wagami)"
echo "=========================================================="
./run_ablation.sh --model FAscnn_pp_V18 --ablation none --class-weights-type boosted --copy-paste-prob 0.3 --save-dir "ablations/Step6_Boosted"

echo "Wszystkie ablacje zostały zakończone!"
