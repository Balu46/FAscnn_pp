#!/bin/bash
# Skrypt do uruchamiania szybkich ablacji na 100 epok
# Zapisuje wagi do folderu 'ablations' zamiast 'checkpoints'
# Użycie: ./run_ablation.sh [dodatkowe opcje]
# Przykłady:
#   ./run_ablation.sh                                     # uruchomi z domyślnym config_1024x2048.yaml
#   ./run_ablation.sh --model FAscnn_pp_V17               # uruchomi ablacje dla konkretnego modelu
#   ./run_ablation.sh --config config_512x1024.yaml       # uruchomi z innym configiem

set -e

CONFIG="config_1024x2048.yaml"
EXTRA_ARGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
source "$VENV_DIR/bin/activate"
echo "[INFO] Activated virtual environment at $VENV_DIR"

# Parse arguments to catch custom config if provided, rest goes to python script
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "================================================="
echo " Rozpoczynam trening ablacyjny (100 epok)"
echo " Config: $CONFIG"
echo " Zapis wag: ablations/{model}"
echo " Dodatkowe argumenty: ${EXTRA_ARGS[@]}"
echo "================================================="

python3 -m src train \
    --config "$CONFIG" \
    --num-epochs 100 \
    --save-dir "ablations/{model}" \
    "${EXTRA_ARGS[@]}"
