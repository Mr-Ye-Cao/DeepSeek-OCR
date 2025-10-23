#!/usr/bin/env python3
"""
Transcribe DeepSeek-OCR Research Paper
Process all 22 pages and convert to markdown
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import fitz  # PyMuPDF
from PIL import Image
import io
from datetime import datetime

print("=" * 80)
print("DeepSeek-OCR Research Paper Transcription")
print("22-Page PDF to Markdown Conversion")
print("=" * 80)

# Setup
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
pdf_path = '/home/ye/ml-experiments/DeepSeek-OCR/DeepSeek_OCR_paper.pdf'
output_dir = '/home/ye/ml-experiments/DeepSeek-OCR/paper_output'
os.makedirs(output_dir, exist_ok=True)

# Load model
print("\n[1/4] Loading DeepSeek-OCR model...")
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
print("      ✅ Model loaded on RTX 5090")

# Open PDF
print(f"\n[2/4] Opening PDF...")
pdf_document = fitz.open(pdf_path)
num_pages = len(pdf_document)
print(f"      Found {num_pages} pages to process")

# Process each page
print(f"\n[3/4] Processing {num_pages} pages with high-quality OCR...")
print("      Using Base mode (1024×1024) for academic paper quality")
print("      " + "=" * 70)

all_markdown = []
start_time = datetime.now()

for page_num in range(num_pages):
    page_start = datetime.now()
    print(f"\n      📄 Page {page_num + 1}/{num_pages}...", end=" ", flush=True)

    # Extract page as high-res image
    page = pdf_document[page_num]
    mat = fitz.Matrix(2.5, 2.5)  # 2.5x zoom for academic quality
    pix = page.get_pixmap(matrix=mat)

    # Convert to PIL Image
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))

    # Save temporary image
    temp_image_path = f"{output_dir}/temp_page_{page_num + 1}.png"
    image.save(temp_image_path)

    # Process with OCR
    prompt = "<image>\n<|grounding|>Convert the document to markdown."

    try:
        # Save to separate page directory
        page_output_dir = f"{output_dir}/page_{page_num + 1}"
        os.makedirs(page_output_dir, exist_ok=True)

        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=temp_image_path,
            output_path=page_output_dir,
            base_size=1024,      # Base mode
            image_size=1024,     # Base mode
            crop_mode=False,
            save_results=True,
            test_compress=False  # Don't print compression stats for cleaner output
        )

        # Read the generated markdown
        result_file = f"{page_output_dir}/result.mmd"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                page_markdown = f.read()

            # Add page marker and content
            all_markdown.append(f"\n\n---\n<!-- Page {page_num + 1} -->\n\n")
            all_markdown.append(page_markdown)

            page_time = (datetime.now() - page_start).total_seconds()
            print(f"✅ ({len(page_markdown)} chars, {page_time:.1f}s)")
        else:
            print(f"❌ No output file")
            all_markdown.append(f"\n\n---\n<!-- Page {page_num + 1}: Error - no output -->\n\n")

        # Clean up temp image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        all_markdown.append(f"\n\n---\n<!-- Page {page_num + 1}: Error - {str(e)[:100]} -->\n\n")

# Save combined markdown
print(f"\n[4/4] Combining all pages into final document...")
output_md_path = f"{output_dir}/DeepSeek_OCR_paper_transcribed.md"

with open(output_md_path, 'w', encoding='utf-8') as f:
    # Write header
    f.write("# DeepSeek-OCR: Contexts Optical Compression\n\n")
    f.write("**Research Paper Transcription**\n\n")
    f.write("- **Original PDF:** DeepSeek_OCR_paper.pdf (22 pages)\n")
    f.write("- **Transcribed by:** DeepSeek-OCR on NVIDIA RTX 5090\n")
    f.write(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("- **Mode:** Base (1024×1024, 256 vision tokens per page)\n")
    f.write("- **Authors:** Haoran Wei, Yaofeng Sun, Yukun Li (DeepSeek-AI)\n\n")
    f.write("---\n\n")

    # Write all pages
    f.writelines(all_markdown)

total_time = (datetime.now() - start_time).total_seconds()
file_size = os.path.getsize(output_md_path)

print("\n" + "=" * 80)
print("✅ TRANSCRIPTION COMPLETE!")
print("=" * 80)
print(f"\n📄 Output file: {output_md_path}")
print(f"📊 Statistics:")
print(f"   - Pages processed: {num_pages}")
print(f"   - Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"   - Average: {total_time/num_pages:.1f} seconds per page")
print(f"   - Output size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
print(f"\n🔍 View with:")
print(f"   cat {output_md_path}")
print(f"   less {output_md_path}")
print("=" * 80)
