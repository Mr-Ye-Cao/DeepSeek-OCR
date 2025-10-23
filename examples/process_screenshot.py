#!/usr/bin/env python3
"""
Process screenshot with DeepSeek-OCR
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os

print("=" * 70)
print("DeepSeek-OCR Screenshot Processing")
print("=" * 70)

# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
image_path = '/home/ye/ml-experiments/DeepSeek-OCR/Screenshot from 2025-10-22 20-44-59.png'
output_dir = '/home/ye/ml-experiments/DeepSeek-OCR/screenshot_output'
os.makedirs(output_dir, exist_ok=True)

# Load model
print("\n1. Loading DeepSeek-OCR model...")
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
print("   ✅ Model loaded on RTX 5090")

# Process image
print(f"\n2. Processing screenshot...")
print(f"   Input: {image_path}")

prompt = "<image>\n<|grounding|>Convert the document to markdown."

try:
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=1024,      # Base mode for good quality
        image_size=1024,
        crop_mode=False,
        save_results=True,
        test_compress=True
    )

    print("\n" + "=" * 70)
    print("✅ SUCCESS!")
    print("=" * 70)

    # Read and display the result
    result_file = f"{output_dir}/result.mmd"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            markdown_content = f.read()

        print(f"\n📄 Extracted Markdown ({len(markdown_content)} chars):")
        print("=" * 70)
        print(markdown_content)
        print("=" * 70)

        print(f"\n💾 Output files:")
        print(f"   - Markdown: {result_file}")
        print(f"   - With boxes: {output_dir}/result_with_boxes.jpg")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
