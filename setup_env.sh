#!/bin/bash
# Quick environment setup script using UV

set -e  # Exit on error

echo "=========================================="
echo "Impact Echo Evidential Learning Setup"
echo "=========================================="
echo ""

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed!"
    echo ""
    echo "Install UV with one of these methods:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  brew install uv"
    echo "  pip install uv"
    echo ""
    exit 1
fi

echo "✓ UV found: $(uv --version)"
echo ""

# Create virtual environment
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists at .venv"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        echo "✓ Removed old environment"
    else
        echo "Using existing environment"
    fi
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.12.11..."
    uv venv --python 3.12.11
    echo "✓ Virtual environment created"
fi

echo ""
echo "Activating environment..."
source .venv/bin/activate

echo "✓ Environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take a few minutes..."
echo ""

uv pip install -r requirements-uv.txt

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import audiomentations; print(\"Audiomentations OK\")'"
echo ""
echo "To profile training performance:"
echo "  python profile_training.py"
echo ""
echo "To start training:"
echo "  python train_evidential_model.py"
echo ""
