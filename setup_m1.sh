#!/bin/bash
# Quick setup script for M1 Mac

set -e  # Exit on error

echo "=========================================="
echo "Scout RL Training - M1 Setup"
echo "=========================================="
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is designed for macOS (M1/M2)"
    echo "For other systems, see the installation instructions in README.md"
    exit 1
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+ first:"
    echo "   brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $(python3 --version)"

# Create virtual environment
if [ ! -d "venv_m1" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv_m1
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv_m1/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "  - PyTorch (with MPS support)"
pip install --quiet --upgrade pip
pip install --quiet torch torchvision torchaudio
echo "  - NumPy, Matplotlib, Pandas"
pip install --quiet numpy matplotlib pandas

# Install RLCard in development mode
echo "  - RLCard (development mode)"
pip install --quiet -e .

echo ""
echo "✓ All dependencies installed"

# Check MPS availability
echo ""
echo "Checking GPU acceleration..."
python3 -c "import torch; print('✓ MPS available!' if torch.backends.mps.is_available() else '⚠️  MPS not available (will use CPU)')"

# Run tests
echo ""
echo "Running tests..."
if python examples/test_scout_improvements.py 2>&1 | grep -q "ALL TESTS PASSED"; then
    echo "✓ All tests passed!"
else
    echo "⚠️  Some tests failed. Check output above."
    exit 1
fi

# Print next steps
echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Your virtual environment is activated."
echo ""
echo "Quick start commands:"
echo ""
echo "  # Test run (1000 episodes, ~5-10 minutes)"
echo "  python examples/train_scout_improved_m1.py --num_episodes 1000"
echo ""
echo "  # Full training (10000 episodes, ~1-2 hours)"
echo "  python examples/train_scout_improved_m1.py --num_episodes 10000"
echo ""
echo "  # Resume this session later:"
echo "  source venv_m1/bin/activate"
echo ""
echo "For more options, see M1_SETUP_GUIDE.md"
echo ""
