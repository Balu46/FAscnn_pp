#!/bin/bash

# Automatycznie znajdź katalog skryptu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit

if [ ! -f requirements.txt ]; then
    echo "Brak pliku requirements.txt!"
    exit 1
fi

python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete."
