#!/bin/bash
# Skrypt do ewaluacji wszystkich wyuczonych modeli ablacyjnych
# oraz pozostałych flagowych modeli do licencjatu.
#
# Wyniki będą logowane na ekranie (możesz też dopisać '> wyniki.txt' na końcu komendy).

set -e

CONFIG="config_1024x2048.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$VENV_DIR/bin/activate"

echo "=========================================================="
echo " ROZPOCZYNAM TESTOWANIE MODELI ABLACYJNYCH"
echo "=========================================================="

# Zwróć uwagę, by podać właściwe flagi ablacyjne zgodne z treningiem

echo "--- Krok 1: FastSCNN (Baseline) ---"
if [ -f "ablations/Step1_Baseline/best.pt" ]; then
    python3 -m src test --config "$CONFIG" --model FastSCNN \
        --weights-path "ablations/Step1_Baseline/best.pt" \
        --run-name "Step1_Baseline"
else
    echo "Brak wag dla Krok 1 (ablations/Step1_Baseline/best.pt)"
fi

echo "--- Krok 2: Inna fuzja (BiSeNetFFM, bez modułu Fast Attention) ---"
if [ -f "ablations/Step2_no_attn/best.pt" ]; then
    python3 -m src test --config "$CONFIG" --model FAscnn_pp_V18 --ablation no_attn \
        --weights-path "ablations/Step2_no_attn/best.pt" \
        --run-name "Step2_no_attn"
else
    echo "Brak wag dla Krok 2 (ablations/Step2_no_attn/best.pt)"
fi

echo "--- Krok 3: Fast Attention ---"
if [ -f "ablations/Step3_FastAttn/best.pt" ]; then
    python3 -m src test --config "$CONFIG" --model FAscnn_pp_V18 --ablation none \
        --weights-path "ablations/Step3_FastAttn/best.pt" \
        --run-name "Step3_FastAttn"
else
    echo "Brak wag dla Krok 3 (ablations/Step3_FastAttn/best.pt)"
fi

echo "--- Krok 4: Dodanie Copy-Paste ---"
if [ -f "ablations/Step4_CP/best.pt" ]; then
    python3 -m src test --config "$CONFIG" --model FAscnn_pp_V18 --ablation none \
        --weights-path "ablations/Step4_CP/best.pt" \
        --run-name "Step4_CP"
else
    echo "Brak wag dla Krok 4 (ablations/Step4_CP/best.pt)"
fi

echo "--- Krok 5: Class Weights (Standard) ---"
if [ -f "ablations/Step5_Weights/best.pt" ]; then
    python3 -m src test --config "$CONFIG" --model FAscnn_pp_V18 --ablation none \
        --weights-path "ablations/Step5_Weights/best.pt" \
        --run-name "Step5_Weights"
else
    echo "Brak wag dla Krok 5 (ablations/Step5_Weights/best.pt)"
fi

echo "--- Krok 6: Manualny Boost Klas ---"
if [ -f "ablations/Step6_Boosted/best.pt" ]; then
    python3 -m src test --config "$CONFIG" --model FAscnn_pp_V18 --ablation none \
        --weights-path "ablations/Step6_Boosted/best.pt" \
        --run-name "Step6_Boosted"
else
    echo "Brak wag dla Krok 6 (ablations/Step6_Boosted/best.pt)"
fi

echo "=========================================================="
echo " ROZPOCZYNAM TESTOWANIE POZOSTAŁYCH MODELI (np. pełne treningi z checkpoints/)"
echo "=========================================================="

# Testowanie wszystkich modeli z katalogu checkpoints/
for MODEL_DIR in checkpoints/*; do
    if [ -d "$MODEL_DIR" ]; then
        MODEL_NAME=$(basename "$MODEL_DIR")
        
        # Określ konfigurację i bazową nazwę modelu
        CONF="$CONFIG"
        BASE_MODEL="$MODEL_NAME"
        
        if [[ "$MODEL_NAME" == *"_512x1024" ]]; then
            CONF="config_512x1024.yaml"
            BASE_MODEL="${MODEL_NAME%_512x1024}"
        elif [[ "$MODEL_NAME" == *"_512x512" ]]; then
            CONF="config_512x512.yaml"
            BASE_MODEL="${MODEL_NAME%_512x512}"
        fi
        
        echo "--- $MODEL_NAME ---"
        if [ -f "checkpoints/$MODEL_NAME/best.pt" ]; then
            python3 -m src test --config "$CONF" --model "$BASE_MODEL" \
                --weights-path "checkpoints/$MODEL_NAME/best.pt" \
                --run-name "$MODEL_NAME"
        else
            echo "Brak wag dla $MODEL_NAME (checkpoints/$MODEL_NAME/best.pt)"
        fi
        echo ""
    fi
done

echo "Testy zakończone!"
