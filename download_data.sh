#!/bin/bash
# Data download script for FAscnn_pp Fast Segmentation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[INFO] Activated virtual environment at $VENV_DIR"
fi

echo "=========================================="
echo "  FAscnn_pp Fast Segmentation - Data Download"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed or not in PATH"
    exit 1
fi

# Skip if already prepared
if [ -d "cityscapes/leftImg8bit" ] && [ -d "cityscapes/gtFine" ]; then
    echo "[INFO] Clean Cityscapes structure already exists — skipping."
    exit 0
fi

echo "[INFO] Starting data download..."
python3 src/data/data_download.py "$@"

echo "[INFO] Restructuring dataset..."

# Create base dir
mkdir -p cityscapes

# Move leftImg8bit
if [ -d "cityscapes/leftImg8bit_trainvaltest/leftImg8bit" ]; then
    mv cityscapes/leftImg8bit_trainvaltest/leftImg8bit cityscapes/
    rm -rf cityscapes/leftImg8bit_trainvaltest
fi

# Move gtFine
if [ -d "cityscapes/gtFine_trainvaltest/gtFine" ]; then
    mv cityscapes/gtFine_trainvaltest/gtFine cityscapes/
    rm -rf cityscapes/gtFine_trainvaltest
fi

echo ""
echo "=========================================="
echo "  Download Complete!"
echo "=========================================="
echo ""
echo "Final structure:"
echo "  cityscapes/"
echo "  ├── leftImg8bit/"
echo "  │   ├── train/"
echo "  │   ├── val/"
echo "  │   └── test/"
echo "  └── gtFine/"
echo "      ├── train/"
echo "      ├── val/"
echo "      └── test/"
echo ""
echo "You can now run training with:"
echo "  ./run_train.sh"
echo ""