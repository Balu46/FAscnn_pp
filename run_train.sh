#!/bin/bash
# Training script for FAscnn_pp Fast Segmentation
# Usage: ./run_train.sh [options]
set -e

CONFIG="config.yaml"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
# Aktywuj środowisko venv
source "$VENV_DIR/bin/activate"
echo "[INFO] Activated virtual environment at $VENV_DIR"

# Help message
show_help() {
    echo "Usage: ./run_train.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --config PATH      YAML config (default: config.yaml)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_train.sh                                    # Train with config.yaml"
    echo "  ./run_train.sh -c configs/exp.yaml                # Train with custom config"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
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

# Run training
echo "Starting training..."
echo "  Config: $CONFIG"
echo ""

python3 -m src train --config "$CONFIG"
