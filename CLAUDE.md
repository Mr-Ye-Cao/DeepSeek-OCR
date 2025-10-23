# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DeepSeek-OCR is a vision-language model designed for Optical Character Recognition with contextual optical compression. The model investigates vision encoders from an LLM-centric viewpoint and supports multiple resolution modes for document and image processing.

## Environment Setup

The project requires CUDA 11.8 and PyTorch 2.6.0.

### Installation Steps

1. Create conda environment:
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

2. Install dependencies:
```bash
# Download vllm-0.8.5 wheel from https://github.com/vllm-project/vllm/releases/tag/v0.8.5
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

Note: Ignore transformers version conflicts between vllm and requirements.txt - they can coexist.

## Running Inference

The repository provides two inference backends: vLLM (for production/batch) and Transformers (for simple cases).

### vLLM Inference (Production)

**Configuration:** Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` before running:
- Set `INPUT_PATH` and `OUTPUT_PATH`
- Configure resolution mode (BASE_SIZE, IMAGE_SIZE, CROP_MODE)
- Adjust `MAX_CONCURRENCY` and `NUM_WORKERS` based on GPU memory
- Set `MODEL_PATH` to model location (default: 'deepseek-ai/DeepSeek-OCR')

**Commands:**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Single image with streaming output
python run_dpsk_ocr_image.py

# PDF processing with concurrency (~2500 tokens/s on A100-40G)
python run_dpsk_ocr_pdf.py

# Batch evaluation for benchmarks
python run_dpsk_ocr_eval_batch.py
```

### Transformers Inference (Simple)

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

Or use the API directly:
```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2',
                                  trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

res = model.infer(tokenizer, prompt=prompt, image_file=image_file,
                  output_path=output_path, base_size=1024, image_size=640,
                  crop_mode=True, save_results=True, test_compress=True)
```

## Resolution Modes

Configure via `BASE_SIZE`, `IMAGE_SIZE`, and `CROP_MODE` in config.py:

- **Tiny:** base_size=512, image_size=512, crop_mode=False (64 vision tokens)
- **Small:** base_size=640, image_size=640, crop_mode=False (100 vision tokens)
- **Base:** base_size=1024, image_size=1024, crop_mode=False (256 vision tokens)
- **Large:** base_size=1280, image_size=1280, crop_mode=False (400 vision tokens)
- **Gundam (Dynamic):** base_size=1024, image_size=640, crop_mode=True (adaptive tiles)

Dynamic resolution automatically tiles images larger than 640×640 using n×640×640 crops plus 1×1024×1024 global view.

## Prompt Examples

Common prompts (from config.py):
```python
# Documents
"<image>\n<|grounding|>Convert the document to markdown."

# General OCR
"<image>\n<|grounding|>OCR this image."

# Without layouts
"<image>\nFree OCR."

# Figures in documents
"<image>\nParse the figure."

# General description
"<image>\nDescribe this image in detail."

# Object localization
"<image>\nLocate <|ref|>xxxx<|/ref|> in the image."
```

## Architecture

### Project Structure

- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/`: vLLM-based production inference
  - `deepseek_ocr.py`: Main model implementation (DeepseekOCRForCausalLM)
  - `config.py`: Central configuration file
  - `run_dpsk_ocr_*.py`: Inference scripts for different use cases
  - `deepencoder/`: Vision encoder components (SAM, CLIP, projector)
  - `process/`: Image preprocessing and n-gram no-repeat logic

- `DeepSeek-OCR-master/DeepSeek-OCR-hf/`: Transformers-based simple inference

### Model Architecture

The model combines multiple vision encoders with a language model:

1. **Vision Encoders:**
   - SAM (Segment Anything Model) encoder: `deepencoder/sam_vary_sdpa.py`
   - CLIP encoder: `deepencoder/clip_sdpa.py`
   - Both use SDPA (Scaled Dot-Product Attention) for efficiency

2. **Projector:** MLP projector in `deepencoder/build_linear.py` projects concatenated SAM+CLIP features (2048-dim) to language model dimension (1280-dim)

3. **Image Processing Pipeline:**
   - Global view: Base image at BASE_SIZE resolution
   - Local views: Dynamic crops at IMAGE_SIZE resolution (when CROP_MODE=True)
   - Features are arranged with 2D tile tags: `image_newline` separates rows, `view_seperator` separates views
   - Final embedding: [local_features, global_features, view_separator]

4. **Language Model:** DeepSeek V2/V3 architecture (auto-selected based on config)

### Key Implementation Details

- **Multi-modal token merging:** Vision embeddings replace `<image>` tokens in input sequence (see `get_input_embeddings` in deepseek_ocr.py:508-528)
- **Dynamic tiling:** Computed in `count_tiles()` function in `process/image_process.py`
- **No-repeat processing:** `process/ngram_norepeat.py` prevents repetitive outputs using n-gram blocking
- **Environment flags:** Set `VLLM_USE_V1='0'` and configure TRITON_PTXAS_PATH for CUDA 11.8

## Performance Tuning

- `MAX_CONCURRENCY`: Controls vLLM batch size (lower if GPU memory limited)
- `MAX_CROPS`: Maximum dynamic tiles (default 6, max 9)
- `NUM_WORKERS`: Image preprocessing parallelism (default 64)
- `gpu_memory_utilization`: 0.75 for image script, 0.9 for PDF/batch scripts
- `SKIP_REPEAT`: Skip pages without proper EOS token in PDF processing

## Output Format

Results include:
- `.mmd` files with markdown output (including detection tags)
- Processed markdown with image references replaced
- Bounding box visualizations (when using grounding prompts)
- Extracted images from documents saved to `OUTPUT_PATH/images/`
