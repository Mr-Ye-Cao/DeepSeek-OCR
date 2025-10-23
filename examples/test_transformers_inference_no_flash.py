from transformers import AutoModel, AutoTokenizer
import torch
import os

print("=" * 60)
print("DeepSeek-OCR Transformers Inference Test (No Flash Attention)")
print("=" * 60)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR'

print(f"\n1. Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("   ✅ Tokenizer loaded")

print(f"\n2. Loading model from {model_name}...")
print("   Using eager mode (no flash attention), bfloat16, and CUDA")
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',  # Use eager instead of flash_attention_2
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
print("   ✅ Model loaded and moved to GPU")

# Tiny mode settings
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = '/home/ye/ml-experiments/DeepSeek-OCR/test_data/sample.jpg'
output_path = '/home/ye/ml-experiments/DeepSeek-OCR/output'

print(f"\n3. Running inference...")
print(f"   Image: {image_file}")
print(f"   Mode: Tiny (512x512, no cropping)")
print(f"   Output: {output_path}")

os.makedirs(output_path, exist_ok=True)

try:
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        base_size=512,      # Tiny mode
        image_size=512,     # Tiny mode
        crop_mode=False,    # Tiny mode
        save_results=True,
        test_compress=True
    )

    print("\n" + "=" * 60)
    print("✅ SUCCESS! Inference completed")
    print("=" * 60)
    print(f"\nResult:\n{res}")
    print(f"\nCheck output directory: {output_path}")

except Exception as e:
    print("\n" + "=" * 60)
    print("❌ ERROR during inference")
    print("=" * 60)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
