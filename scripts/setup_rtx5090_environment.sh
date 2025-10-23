#!/bin/bash
# DeepSeek-OCR Environment Setup Script for RTX 5090
# This script sets up the WORKING environment based on our successful configuration
# Solution: PyTorch 2.10.0 nightly with CUDA 12.8 + Transformers backend

set -e  # Exit on error

echo "=========================================="
echo "DeepSeek-OCR RTX 5090 Setup"
echo "Working Solution: PyTorch 2.10+ CUDA 12.8"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda not found${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo -e "${GREEN}Conda found: $(which conda)${NC}"
echo ""

# Step 1: Create conda environment
echo -e "${YELLOW}Step 1: Creating conda environment 'deepseek-ocr'...${NC}"
if conda env list | grep -q "^deepseek-ocr "; then
    echo -e "${YELLOW}Environment 'deepseek-ocr' already exists.${NC}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n deepseek-ocr -y
    else
        echo -e "${YELLOW}Using existing environment. Skipping to installation...${NC}"
    fi
fi

if ! conda env list | grep -q "^deepseek-ocr "; then
    conda create -n deepseek-ocr python=3.12.9 -y
    echo -e "${GREEN}Environment created successfully!${NC}"
fi

# Step 2: Activate environment
echo ""
echo -e "${YELLOW}Step 2: Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate deepseek-ocr

# Step 3: Install PyTorch nightly with CUDA 12.8 (CRITICAL!)
echo ""
echo -e "${YELLOW}Step 3: Installing PyTorch 2.10+ nightly with CUDA 12.8...${NC}"
echo -e "${RED}NOTE: CUDA 12.8 is REQUIRED for RTX 5090 (sm_120) support!${NC}"
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 4: Install DeepSeek-OCR dependencies
echo ""
echo -e "${YELLOW}Step 4: Installing DeepSeek-OCR requirements...${NC}"

# Change to repo directory
REPO_DIR="/home/ye/ml-experiments/DeepSeek-OCR"
if [ ! -d "$REPO_DIR" ]; then
    echo -e "${RED}Error: Repository not found at $REPO_DIR${NC}"
    exit 1
fi

cd "$REPO_DIR"

# Install from requirements.txt
pip install -r requirements.txt

echo ""
echo -e "${GREEN}=========================================="
echo "Installation Complete!"
echo "==========================================${NC}"
echo ""
echo -e "${YELLOW}Installed versions:${NC}"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
echo ""
echo -e "${YELLOW}Key points:${NC}"
echo "  ✅ Using PyTorch nightly with CUDA 12.8"
echo "  ✅ Using Transformers backend (no vLLM)"
echo "  ✅ Using eager attention mode (no flash-attention)"
echo ""
echo -e "${GREEN}To use the environment:${NC}"
echo "  conda activate deepseek-ocr"
echo ""
echo -e "${GREEN}To test:${NC}"
echo "  cd $REPO_DIR"
echo "  python examples/test_transformers_inference_no_flash.py"
echo ""
echo -e "${YELLOW}For more info, see: docs/SUCCESS_SOLUTION.md${NC}"
echo ""
