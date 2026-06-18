#!/bin/bash
# Testing script for FAscnn_pp Fast Segmentation
# Usage: ./run_test.sh [options]
set -e

CONFIG="config.yaml"
BATCH_SIZE=""
MODEL=""
DEVICE=""
ABLATION=""
WEIGHTS_PATH=""

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VENV_DIR="$PROJECT_DIR/.venv"
# Aktywuj środowisko venv
source "$VENV_DIR/bin/activate"
echo "[INFO] Activated virtual environment at $VENV_DIR"


# Help message
show_help() {
    echo "Usage: ./run_test.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --config PATH      YAML config (default: config.yaml)"
    echo "  -b, --batch-size NUM   Batch size (override config)"
    echo "  -m, --model NAME       Model name: ENet, ENetv2, ENetv3, FastSCNN, FAscnn_pp_V6 (override config)"
    echo "  -d, --device DEVICE    Device: auto, cuda, cpu (override config)"
    echo "  -a, --ablation SPEC    Ablation tokens for FAscnn_pp_V6 (comma-separated, override config)"
    echo "  -w, --weights PATH     Path to weights (override config)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_test.sh                        # Test with config.yaml"
    echo "  ./run_test.sh -c configs/exp.yaml    # Test with custom config"
    echo "  ./run_test.sh -b 4 -d cpu            # Override batch size on CPU"
    echo "  ./run_test.sh -m ENetv2              # Override model"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -a|--ablation)
            ABLATION="$2"
            shift 2
            ;;
        -w|--weights)
            WEIGHTS_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run testing
echo "Starting testing..."
echo "  Config: $CONFIG"
if [ -n "$BATCH_SIZE" ]; then echo "  Override batch size: $BATCH_SIZE"; fi
if [ -n "$MODEL" ]; then echo "  Override model: $MODEL"; fi
if [ -n "$DEVICE" ]; then echo "  Override device: $DEVICE"; fi
if [ -n "$ABLATION" ]; then echo "  Override ablation: $ABLATION"; fi
if [ -n "$WEIGHTS_PATH" ]; then echo "  Override weights: $WEIGHTS_PATH"; fi
echo ""

ARGS=(--config "$CONFIG")
if [ -n "$BATCH_SIZE" ]; then ARGS+=(--batch-size "$BATCH_SIZE"); fi
if [ -n "$MODEL" ]; then ARGS+=(--model "$MODEL"); fi
if [ -n "$DEVICE" ]; then ARGS+=(--device "$DEVICE"); fi
if [ -n "$ABLATION" ]; then ARGS+=(--ablation "$ABLATION"); fi
if [ -n "$WEIGHTS_PATH" ]; then ARGS+=(--weights-path "$WEIGHTS_PATH"); fi

python3 -m src test "${ARGS[@]}"
