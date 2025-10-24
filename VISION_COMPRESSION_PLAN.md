# Vision Token Compression for Context Extension

**Objective:** Extend gpt-oss-20b's effective context length (8000 tokens) using 20x vision token compression of personal knowledge (files + screenshots), with adaptive forgetting mechanism.

**Author:** Claude Code
**Date:** 2025-10-22
**Repository:** DeepSeek-OCR
**Use Case:** Personal knowledge base compression and retrieval with LLM-centric forgetting

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Detailed Implementation Plan](#detailed-implementation-plan)
4. [Phase 1: Knowledge Base Compression](#phase-1-knowledge-base-compression)
5. [Phase 2: Training Data Generation](#phase-2-training-data-generation)
6. [Phase 3: Model Adaptation & Fine-tuning](#phase-3-model-adaptation--fine-tuning)
7. [Phase 4: Forgetting Mechanism](#phase-4-forgetting-mechanism)
8. [Phase 5: Integration & End-to-End Pipeline](#phase-5-integration--end-to-end-pipeline)
9. [Technical Specifications](#technical-specifications)
10. [Expected Results & Limitations](#expected-results--limitations)
11. [Timeline & Resources](#timeline--resources)

---

## Executive Summary

### The Problem
- gpt-oss-20b has 8000 token context limit
- Personal knowledge base (files, screenshots) far exceeds this
- Cannot load all knowledge into model context at once

### The Solution
- Use DeepSeek-OCR's vision encoder (DeepEncoder) to compress documents → vision tokens
- **20x compression ratio:** 8000 text tokens → 400 vision tokens
- Fine-tune gpt-oss-20b to accept and reason over compressed vision embeddings
- Implement adaptive forgetting: old/unused memories degrade gracefully

### Key Innovation
- **Forgetting as a feature:** Higher compression = more information loss = natural forgetting
- **Adaptive memory:** Importance-based pruning when memory exceeds budget
- **Persistent context:** Compressed knowledge survives across conversations

### Expected Performance
| Metric | Value |
|---|---|
| Compression Ratio | 20x (8000 → 400 tokens) |
| Memory Fidelity | ~60% (gist preserved, details lossy) |
| Effective Context | 400 vision tokens + 7600 text tokens = 8000 total |
| Supported Files | PDFs, docs, code, screenshots, images |
| Training Time (RTX 5090) | 1-2 days (5k-10k examples) |
| Memory Usage (inference) | Same as base gpt-oss-20b |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Your Knowledge Base                            │
│          (Personal files: PDFs, docs, code, screenshots)             │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │  Image Preprocessing Layer   │
                │  (render non-images as PNG)  │
                │  - PDFs → page images        │
                │  - Docs → rendered images    │
                │  - Code → screenshot         │
                │  - Screenshots → as-is       │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │   DeepEncoder (pretrained)   │
                │   - SAM: perception (80M)    │
                │   - CLIP: knowledge (300M)   │
                │   - 16x compressor layer     │
                │   Input: 1024×1024 image     │
                │   Output: 256 tokens @ base, │
                │           64 tokens @ tiny   │
                └──────────┬───────────────────┘
                           │
                           ▼
                ┌──────────────────────────────┐
                │  Compressed Memory Bank      │
                │  - Persistent embeddings     │
                │  - Metadata (importance,     │
                │    timestamp, access_count)  │
                │  - Storage: ~400 tokens      │
                │  - Format: PyTorch tensors   │
                └──────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────┐
         │    Vision Token Projector (NEW)          │
         │    - Adapter layer: 1280-dim → GPT-dim  │
         │    - LoRA trainable                      │
         │    - Learns to map vision embeddings     │
         │      into gpt-oss-20b's embedding space  │
         └────────────┬────────────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────────────┐
         │    gpt-oss-20b (LoRA fine-tuned)         │
         │    Input: [vision_context, text_query]   │
         │    - Vision context: 400 tokens          │
         │    - Text query: 7600 tokens max         │
         │    - Total context: 8000 tokens          │
         │    - Fine-tuned on (vision, query, ans)  │
         │      triplets                            │
         └────────────┬────────────────────────────┘
                      │
                      ▼
         ┌─────────────────────────────────────────┐
         │         Generated Response               │
         │  (grounded in compressed personal        │
         │   knowledge + reasoning capability)      │
         └─────────────────────────────────────────┘
```

### Data Flow Example

```python
# User's daily workflow
query = "What did I learn about transformers in my notes?"

# System retrieves relevant compressed memories
vision_context = memory.get_context_for_query(query)  # 400 tokens
# → Selected: important_ml_paper.pdf (64 tokens) + notes_transformers.png (100 tokens)

# System constructs input
input_tokens = concat([
    vision_projector(vision_context),  # 400 vision tokens → GPT embedding space
    tokenize(query)                    # "What did I learn..." → ~50 text tokens
])
# Total: 400 + 50 = 450 tokens (well under 8000 limit)

# gpt-oss-20b generates response
response = gpt_oss_20b.generate(input_tokens)
# Model can now reason over compressed knowledge + generate informed response

# System tracks memory access
memory.update_access_count(vision_context)  # Increment importance of accessed memories
```

---

## Detailed Implementation Plan

### Project Structure

```
/home/ye/ml-experiments/DeepSeek-OCR/
├── vision_compression/                    # NEW: Main implementation
│   ├── __init__.py
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── memory_bank.py                # Compressed memory storage
│   │   ├── image_processor.py            # Render files as images
│   │   └── deepencoder_wrapper.py        # Wrapper around DeepSeek-OCR encoder
│   ├── adaptation/
│   │   ├── __init__.py
│   │   ├── vision_projector.py           # Vision token → GPT embedding adapter
│   │   ├── gpt_vision_model.py           # VisionAugmentedGPT class
│   │   └── lora_config.py                # LoRA configuration
│   ├── training/
│   │   ├── __init__.py
│   │   ├── data_generator.py             # Generate Q&A from compressed images
│   │   ├── dataset.py                    # PyTorch Dataset class
│   │   └── trainer.py                    # Training loop with validation
│   ├── memory/
│   │   ├── __init__.py
│   │   └── adaptive_memory.py            # Forgetting mechanism implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                     # Configuration constants
│   │   ├── io_utils.py                   # Save/load utilities
│   │   └── metrics.py                    # Evaluation metrics
│   └── pipelines/
│       ├── __init__.py
│       ├── compress_knowledge_base.py    # Phase 1: Initial compression
│       ├── generate_training_data.py     # Phase 2: Data generation
│       ├── finetune_gpt.py               # Phase 3: Fine-tuning
│       └── end_to_end_pipeline.py        # Phase 5: Full system
├── VISION_COMPRESSION_PLAN.md            # This file
└── examples/
    ├── example_compress.py               # Compress a single file
    ├── example_query.py                  # Query compressed memory
    └── example_forgetting.py             # Test forgetting mechanism
```

---

## Phase 1: Knowledge Base Compression

**Goal:** Convert all personal files into compressed vision token embeddings.

**Duration:** 1-2 days
**Output:** Persistent memory bank (400-800 vision tokens total)

### 1.1 Image Preprocessing

**File:** `vision_compression/compression/image_processor.py`

**Requirements:**
- Convert PDFs to images (PyMuPDF/fitz)
- Render documents/code as images
- Keep screenshots as-is
- Normalize all to consistent resolution

**Implementation:**

```python
class FileToImageConverter:
    def __init__(self):
        """Initialize converters for different file types"""
        pass

    def convert_pdf(self, pdf_path: str) -> List[Image]:
        """
        Convert PDF to page images using PyMuPDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Images (one per page)
        """
        # Use fitz (PyMuPDF) to render pages
        # Resolution: 200 DPI recommended
        # Return list of PIL Images
        pass

    def convert_doc(self, doc_path: str) -> List[Image]:
        """
        Convert Word/Markdown documents to images

        Args:
            doc_path: Path to .docx, .md, .txt file

        Returns:
            List of PIL Images (one per logical section)
        """
        # Use python-pptx for docx/doc
        # Use markdown-to-html + html2image for md/txt
        # Return list of PIL Images
        pass

    def convert_code(self, code_path: str) -> Image:
        """
        Render code file as image (syntax-highlighted)

        Args:
            code_path: Path to code file

        Returns:
            PIL Image with highlighted code
        """
        # Use Pygments for syntax highlighting
        # Render to image (Pillow or PIL)
        # Return single Image
        pass

    def load_image(self, image_path: str) -> Image:
        """
        Load image file directly (JPEG, PNG, etc.)
        """
        return Image.open(image_path)

    def batch_convert(self, file_paths: List[str]) -> Dict[str, List[Image]]:
        """
        Convert multiple files in batch

        Returns:
            Dict mapping file path → list of images
        """
        pass
```

**Dependencies:**
```
PyMuPDF (fitz)
Pillow
python-pptx
markdown
Pygments
html2image (or wkhtmltoimage)
```

### 1.2 DeepEncoder Wrapper

**File:** `vision_compression/compression/deepencoder_wrapper.py`

**Purpose:** Encapsulate DeepSeek-OCR's DeepEncoder for easy compression

**Implementation:**

```python
class DeepEncoderCompressor:
    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-OCR',
                 resolution: str = 'tiny',  # or 'small', 'base', 'large'
                 device: str = 'cuda'):
        """
        Initialize DeepEncoder

        Args:
            model_name: HuggingFace model identifier
            resolution: 'tiny' (64 tokens), 'small' (100), 'base' (256), 'large' (400)
            device: 'cuda' or 'cpu'
        """
        # Load pretrained DeepSeek-OCR model
        # Extract DeepEncoder component
        # Set to eval mode
        # Resolution modes:
        # - tiny: 512×512 → 64 tokens (20x compression)
        # - small: 640×640 → 100 tokens (~10x compression)
        # - base: 1024×1024 → 256 tokens (~7x compression)
        # - large: 1280×1280 → 400 tokens (~5x compression)
        pass

    def compress(self, image: Image) -> torch.Tensor:
        """
        Compress image to vision embeddings

        Args:
            image: PIL Image

        Returns:
            Vision embeddings tensor
            Shape: (num_tokens, 1280)
            where num_tokens depends on resolution:
            - tiny: 64
            - small: 100
            - base: 256
            - large: 400
        """
        # Preprocess image to target resolution
        # Forward through DeepEncoder
        # Return embeddings (no text generation needed yet)
        pass

    def compress_batch(self, images: List[Image]) -> torch.Tensor:
        """
        Compress multiple images in parallel

        Returns:
            Stacked embeddings tensor
            Shape: (batch_size * num_tokens, 1280)
        """
        pass

    @torch.no_grad()
    def __call__(self, image: Image) -> torch.Tensor:
        return self.compress(image)
```

### 1.3 Memory Bank

**File:** `vision_compression/compression/memory_bank.py`

**Purpose:** Persistent storage of compressed embeddings with metadata

**Implementation:**

```python
class CompressedMemoryBank:
    def __init__(self, save_dir: str = './compressed_memory'):
        """
        Initialize memory bank

        Args:
            save_dir: Directory to persist embeddings
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.memories = []  # List of memory chunks
        self.index = {}     # Quick lookup: file_hash → memory_chunk

    def add_memory(self, file_path: str, embeddings: torch.Tensor,
                   metadata: Dict = None) -> str:
        """
        Add compressed memory to bank

        Args:
            file_path: Original file path (for reference)
            embeddings: Vision token embeddings (num_tokens, 1280)
            metadata: Optional dict with:
                - timestamp: when compressed
                - source: original file path
                - importance: manual importance score (0-10)
                - category: document type
                - access_count: number of times retrieved

        Returns:
            memory_id: Unique identifier for this memory chunk
        """
        memory_id = self._generate_id()
        memory_chunk = {
            'id': memory_id,
            'embeddings': embeddings,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'access_count': 0,
        }

        self.memories.append(memory_chunk)
        self.index[memory_id] = len(self.memories) - 1
        self._save_to_disk(memory_chunk)
        return memory_id

    def get_memory(self, memory_id: str) -> torch.Tensor:
        """Retrieve embeddings by memory ID"""
        pass

    def get_all_memories(self) -> torch.Tensor:
        """Get concatenated embeddings of all memories"""
        pass

    def get_total_tokens(self) -> int:
        """Calculate total vision tokens across all memories"""
        return sum(m['embeddings'].shape[0] for m in self.memories)

    def list_memories(self) -> pd.DataFrame:
        """Return pandas DataFrame with memory metadata"""
        pass

    def save(self):
        """Persist entire memory bank to disk"""
        # Save embeddings in efficient format (.pt or .npy)
        # Save metadata as JSON
        pass

    def load(self):
        """Load memory bank from disk"""
        pass

    def _generate_id(self) -> str:
        """Generate unique memory ID (UUID)"""
        return str(uuid.uuid4())[:8]

    def _save_to_disk(self, memory_chunk: Dict):
        """Save individual memory chunk to disk"""
        pass
```

### 1.4 Phase 1 Pipeline

**File:** `vision_compression/pipelines/compress_knowledge_base.py`

**Entry point for Phase 1:**

```python
def compress_knowledge_base(
    input_dirs: List[str],
    output_dir: str = './compressed_memory',
    resolution: str = 'tiny',  # 64 tokens, 20x compression
    batch_size: int = 4,
    device: str = 'cuda'
) -> CompressedMemoryBank:
    """
    Main Phase 1 pipeline

    Args:
        input_dirs: Directories containing personal files
        output_dir: Where to save compressed memory bank
        resolution: 'tiny' for aggressive 20x, 'small' for balanced 10x
        batch_size: Images to process in parallel
        device: 'cuda' or 'cpu'

    Returns:
        CompressedMemoryBank: Populated memory bank

    Example:
        >>> memory = compress_knowledge_base(
        ...     input_dirs=['~/Documents', '~/Pictures'],
        ...     resolution='tiny'
        ... )
        >>> print(f"Total tokens: {memory.get_total_tokens()}")
    """

    # Step 1: Discover all files
    file_paths = discover_files(input_dirs)
    print(f"Found {len(file_paths)} files to compress")

    # Step 2: Initialize components
    converter = FileToImageConverter()
    compressor = DeepEncoderCompressor(resolution=resolution, device=device)
    memory_bank = CompressedMemoryBank(save_dir=output_dir)

    # Step 3: Process files
    for file_path in tqdm(file_paths):
        try:
            # Convert file to image(s)
            images = converter.batch_convert([file_path])

            # Compress images
            for img_idx, image in enumerate(images):
                embeddings = compressor.compress(image)

                # Add to memory bank with metadata
                metadata = {
                    'source': str(file_path),
                    'page': img_idx,
                    'category': infer_category(file_path),
                    'importance': 5.0,  # Default importance
                }
                memory_bank.add_memory(str(file_path), embeddings, metadata)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Step 4: Save memory bank
    memory_bank.save()

    # Step 5: Report statistics
    print(f"\n=== Compression Complete ===")
    print(f"Total memories: {len(memory_bank.memories)}")
    print(f"Total vision tokens: {memory_bank.get_total_tokens()}")
    print(f"Saved to: {output_dir}")

    return memory_bank

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True,
                       help='Directories with personal files')
    parser.add_argument('--output', default='./compressed_memory',
                       help='Output directory')
    parser.add_argument('--resolution', default='tiny',
                       help='tiny|small|base|large')
    args = parser.parse_args()

    memory = compress_knowledge_base(
        input_dirs=args.input,
        output_dir=args.output,
        resolution=args.resolution
    )
```

---

## Phase 2: Training Data Generation

**Goal:** Create (vision_tokens, query, answer) dataset for fine-tuning.

**Duration:** 2-3 days
**Output:** 5k-10k training examples

### 2.1 Text Extraction from Compressed Images

**File:** `vision_compression/training/data_generator.py`

**Purpose:** Use DeepSeek-OCR to extract full text from compressed images

**Implementation:**

```python
class TrainingDataGenerator:
    def __init__(self, memory_bank: CompressedMemoryBank,
                 deepseek_ocr_model_name: str = 'deepseek-ai/DeepSeek-OCR'):
        """
        Initialize training data generator

        Args:
            memory_bank: Compressed memory from Phase 1
            deepseek_ocr_model_name: DeepSeek-OCR model for text extraction
        """
        # Load full DeepSeek-OCR model (for OCR, not just encoder)
        # This is used to decompress and extract text from images
        pass

    def extract_text_from_image(self, image: Image) -> str:
        """
        Use DeepSeek-OCR to extract full text from image

        Args:
            image: PIL Image

        Returns:
            Extracted text (OCR output)
        """
        # Use DeepSeek-OCR in "Free OCR" mode
        # prompt = "<image>\nFree OCR."
        # Return full text
        pass

    def extract_text_from_memory(self, memory_id: str,
                                  memory_bank: CompressedMemoryBank) -> str:
        """
        Extract text from a memory chunk

        Args:
            memory_id: ID of memory to extract text from
            memory_bank: CompressedMemoryBank instance

        Returns:
            Full extracted text
        """
        # Get original image from memory bank metadata
        # Extract text using DeepSeek-OCR
        pass

    def generate_qa_pairs(self, text: str, num_questions: int = 3) -> List[Tuple[str, str]]:
        """
        Generate Q&A pairs from extracted text

        Args:
            text: Extracted text from image
            num_questions: Number of Q&A pairs to generate

        Returns:
            List of (question, answer) tuples
        """
        # Use gpt-oss-20b or another model to generate questions
        # Questions should be diverse:
        # - Factual ("What is mentioned about X?")
        # - Summarization ("Summarize the main points")
        # - Reasoning ("How does this relate to Y?")
        pass

    def create_training_dataset(self, memory_bank: CompressedMemoryBank,
                               num_questions_per_memory: int = 3) -> List[Dict]:
        """
        Generate full training dataset from memory bank

        Args:
            memory_bank: CompressedMemoryBank with all personal files
            num_questions_per_memory: Questions to generate per memory chunk

        Returns:
            List of training examples:
            [
                {
                    'vision_embeddings': torch.Tensor (num_tokens, 1280),
                    'query': 'What is discussed?',
                    'answer': 'The document discusses...',
                    'memory_id': 'abc123',
                },
                ...
            ]
        """
        training_data = []

        for memory_chunk in tqdm(memory_bank.memories):
            memory_id = memory_chunk['id']
            embeddings = memory_chunk['embeddings']

            # Extract text from this memory
            text = self.extract_text_from_memory(memory_id, memory_bank)

            # Generate Q&A pairs
            qa_pairs = self.generate_qa_pairs(text, num_questions_per_memory)

            # Create training examples
            for question, answer in qa_pairs:
                training_data.append({
                    'vision_embeddings': embeddings.cpu(),
                    'query': question,
                    'answer': answer,
                    'memory_id': memory_id,
                })

        return training_data
```

### 2.2 PyTorch Dataset

**File:** `vision_compression/training/dataset.py`

```python
class VisionCompressedDataset(torch.utils.data.Dataset):
    def __init__(self, training_examples: List[Dict],
                 tokenizer: transformers.PreTrainedTokenizer):
        """
        PyTorch Dataset for vision-compressed Q&A pairs

        Args:
            training_examples: Output from data_generator.create_training_dataset()
            tokenizer: gpt-oss-20b tokenizer
        """
        self.examples = training_examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Vision embeddings (already processed)
        vision_embeddings = example['vision_embeddings']

        # Tokenize query and answer
        query_tokens = self.tokenizer(
            example['query'],
            return_tensors='pt',
            padding=False,
        )

        answer_tokens = self.tokenizer(
            example['answer'],
            return_tensors='pt',
            padding=False,
        )

        return {
            'vision_embeddings': vision_embeddings,
            'query_input_ids': query_tokens['input_ids'].squeeze(0),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
            'answer_input_ids': answer_tokens['input_ids'].squeeze(0),
            'answer_attention_mask': answer_tokens['attention_mask'].squeeze(0),
            'memory_id': example['memory_id'],
        }

class VisionCompressedDataCollator:
    def __init__(self, tokenizer, max_length: int = 8000):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of training examples

        Returns:
            {
                'vision_embeddings': (batch_size, max_vision_tokens, 1280),
                'input_ids': (batch_size, max_length),
                'attention_mask': (batch_size, max_length),
                'labels': (batch_size, max_length),
            }
        """
        # Concatenate vision embeddings + query + answer
        # Pad to max_length
        # Return batch dict
        pass
```

### 2.3 Phase 2 Pipeline

**File:** `vision_compression/pipelines/generate_training_data.py`

```python
def generate_training_data(
    memory_bank: CompressedMemoryBank,
    output_path: str = './training_data.json',
    num_questions_per_memory: int = 3,
) -> List[Dict]:
    """
    Phase 2: Generate training dataset

    Args:
        memory_bank: Compressed memory from Phase 1
        output_path: Where to save training data
        num_questions_per_memory: Diversity of Q&A pairs

    Returns:
        Training examples list
    """

    generator = TrainingDataGenerator(memory_bank)
    training_data = generator.create_training_dataset(
        memory_bank,
        num_questions_per_memory=num_questions_per_memory
    )

    print(f"Generated {len(training_data)} training examples")

    # Save to JSON
    import json
    with open(output_path, 'w') as f:
        json.dump([{
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in ex.items()
        } for ex in training_data], f)

    return training_data
```

---

## Phase 3: Model Adaptation & Fine-tuning

**Goal:** Fine-tune gpt-oss-20b to understand and reason over vision embeddings.

**Duration:** 3-5 days
**Output:** LoRA-adapted gpt-oss-20b model

### 3.1 Vision Token Projector

**File:** `vision_compression/adaptation/vision_projector.py`

**Purpose:** Map vision embeddings (1280-dim) to gpt-oss-20b's embedding space

```python
class VisionTokenProjector(nn.Module):
    def __init__(self, vision_dim: int = 1280,
                 gpt_hidden_dim: int = 4096,
                 use_lora: bool = True):
        """
        Project vision embeddings to GPT embedding space

        Args:
            vision_dim: Dimension of vision embeddings (1280 for DeepEncoder)
            gpt_hidden_dim: Hidden dimension of gpt-oss-20b
            use_lora: Use LoRA for efficient adaptation
        """
        super().__init__()

        # Simple linear projection
        self.linear = nn.Linear(vision_dim, gpt_hidden_dim)

        # Optional: Layer normalization
        self.ln = nn.LayerNorm(gpt_hidden_dim)

        # LoRA if specified
        if use_lora:
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=['linear'],
                lora_dropout=0.05,
            )
            # Apply LoRA (will wrap in training)

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project vision embeddings

        Args:
            vision_embeddings: (batch_size, num_vision_tokens, 1280)

        Returns:
            Projected embeddings: (batch_size, num_vision_tokens, gpt_hidden_dim)
        """
        x = self.linear(vision_embeddings)
        x = self.ln(x)
        return x
```

### 3.2 Vision-Augmented GPT Model

**File:** `vision_compression/adaptation/gpt_vision_model.py`

```python
class VisionAugmentedGPT(nn.Module):
    def __init__(self, gpt_model: transformers.PreTrainedModel,
                 use_lora: bool = True):
        """
        Augment gpt-oss-20b to accept vision embeddings

        Args:
            gpt_model: Pretrained gpt-oss-20b model
            use_lora: Apply LoRA fine-tuning
        """
        super().__init__()

        self.gpt = gpt_model
        self.hidden_dim = gpt_model.config.hidden_size

        # Vision projector
        self.vision_projector = VisionTokenProjector(
            vision_dim=1280,
            gpt_hidden_dim=self.hidden_dim,
            use_lora=use_lora
        )

        # LoRA for GPT
        if use_lora:
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=['q_proj', 'v_proj'],
                lora_dropout=0.05,
            )
            self.gpt = get_peft_model(self.gpt, lora_config)

    def forward(self,
                vision_embeddings: torch.Tensor = None,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs) -> transformers.modeling_outputs.CausalLMOutput:
        """
        Forward pass combining vision and text

        Args:
            vision_embeddings: (batch, num_vision_tokens, 1280)
            input_ids: (batch, seq_len) - text tokens
            attention_mask: (batch, seq_len)
            labels: (batch, seq_len) - for training

        Returns:
            CausalLMOutput with loss if labels provided
        """

        # Get text embeddings
        text_embeddings = self.gpt.get_input_embeddings()(input_ids)

        # Project vision embeddings
        if vision_embeddings is not None:
            projected_vision = self.vision_projector(vision_embeddings)

            # Concatenate: [vision_context, text_embeddings]
            combined_embeddings = torch.cat(
                [projected_vision, text_embeddings],
                dim=1
            )

            # Extend attention mask for vision tokens
            if attention_mask is not None:
                vision_mask = torch.ones(
                    (vision_embeddings.shape[0], vision_embeddings.shape[1]),
                    device=attention_mask.device
                )
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            combined_embeddings = text_embeddings

        # Forward through GPT
        outputs = self.gpt(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def generate(self, vision_embeddings: torch.Tensor,
                input_ids: torch.Tensor,
                max_length: int = 2048,
                **kwargs):
        """
        Generate text conditioned on vision embeddings and text prompt

        Args:
            vision_embeddings: (batch, num_vision_tokens, 1280)
            input_ids: (batch, prompt_len) - query
            max_length: Maximum generation length

        Returns:
            Generated token ids
        """
        # Not directly supported; use text-only generation
        # after converting embeddings to text (decode step)
        pass
```

### 3.3 LoRA Configuration

**File:** `vision_compression/adaptation/lora_config.py`

```python
def get_lora_config(target_modules: str = 'full') -> 'LoraConfig':
    """
    Get LoRA configuration for efficient fine-tuning

    Args:
        target_modules: 'full' (all linear), 'attention' (q,v only), 'minimal'

    Returns:
        peft.LoraConfig
    """
    from peft import LoraConfig

    if target_modules == 'attention':
        modules = ['q_proj', 'v_proj']
    elif target_modules == 'full':
        modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'fc1', 'fc2']
    elif target_modules == 'minimal':
        modules = ['q_proj', 'v_proj']
    else:
        raise ValueError(f"Unknown target_modules: {target_modules}")

    return LoraConfig(
        r=16,                          # LoRA rank
        lora_alpha=32,                 # Scaling factor
        target_modules=modules,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )

def get_training_args(output_dir: str = './checkpoints',
                      num_epochs: int = 3,
                      batch_size: int = 4,
                      learning_rate: float = 3e-5) -> 'TrainingArguments':
    """
    Get TrainingArguments for Hugging Face Trainer
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=learning_rate,
        lr_scheduler_type='cosine',
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        eval_strategy='no',  # Or 'steps' with eval_steps
        fp16=True,          # Use mixed precision
        tf32=True,          # Use TF32 on A100-like GPUs
        gradient_checkpointing=True,  # Save memory
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
```

### 3.4 Training Script

**File:** `vision_compression/training/trainer.py`

```python
class VisionCompressedTrainer:
    def __init__(self, model: VisionAugmentedGPT,
                 train_dataset: VisionCompressedDataset,
                 output_dir: str = './checkpoints'):
        """
        Trainer for vision-augmented GPT
        """
        self.model = model
        self.train_dataset = train_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def train(self, num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 3e-5,
              warmup_steps: int = 500):
        """
        Train the model with LoRA

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
        """
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=50,
            save_steps=500,
            fp16=True,
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=VisionCompressedDataCollator(
                self.train_dataset.tokenizer
            ),
        )

        trainer.train()
        self.model.save_pretrained(self.output_dir / 'final_model')

def train_vision_augmented_gpt(
    gpt_model_name: str = 'gpt-oss-20b',
    training_data_path: str = './training_data.json',
    output_dir: str = './checkpoints',
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
):
    """
    Phase 3 pipeline: Fine-tune gpt-oss-20b

    Args:
        gpt_model_name: Model identifier
        training_data_path: Path to training data JSON
        output_dir: Where to save checkpoints
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """

    # Load models
    tokenizer = transformers.AutoTokenizer.from_pretrained(gpt_model_name)
    gpt_model = transformers.AutoModelForCausalLM.from_pretrained(
        gpt_model_name,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    # Load training data
    import json
    with open(training_data_path) as f:
        training_examples = json.load(f)

    # Create dataset
    dataset = VisionCompressedDataset(training_examples, tokenizer)

    # Wrap model
    model = VisionAugmentedGPT(gpt_model, use_lora=True)

    # Train
    trainer = VisionCompressedTrainer(model, dataset, output_dir)
    trainer.train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    print(f"\nTraining complete! Model saved to {output_dir}")
```

### 3.5 Phase 3 Pipeline

**File:** `vision_compression/pipelines/finetune_gpt.py`

```python
def finetune_gpt_oss(
    gpt_model_name: str = 'gpt-oss-20b',
    training_data_path: str = './training_data.json',
    output_dir: str = './finetuned_gpt',
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
):
    """
    Main Phase 3 entry point: Fine-tune gpt-oss-20b
    """
    print("=== Phase 3: Fine-tuning gpt-oss-20b ===\n")

    train_vision_augmented_gpt(
        gpt_model_name=gpt_model_name,
        training_data_path=training_data_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
```

---

## Phase 4: Forgetting Mechanism

**Goal:** Implement adaptive memory with importance-based pruning.

**Duration:** 1-2 days
**Output:** AdaptiveCompressedMemory class

### 4.1 Adaptive Memory Implementation

**File:** `vision_compression/memory/adaptive_memory.py`

```python
class AdaptiveCompressedMemory:
    """
    Adaptive memory system with importance scoring and forgetting
    """

    def __init__(self, max_vision_tokens: int = 400,
                 encoder: 'DeepEncoderCompressor' = None,
                 memory_bank: 'CompressedMemoryBank' = None):
        """
        Initialize adaptive memory

        Args:
            max_vision_tokens: Maximum total vision tokens to keep
            encoder: DeepEncoder for re-compression
            memory_bank: Existing memory bank to load from
        """
        self.max_vision_tokens = max_vision_tokens
        self.encoder = encoder
        self.memory_bank = memory_bank or CompressedMemoryBank()
        self.current_tokens = self.memory_bank.get_total_tokens()

    def add_new_knowledge(self, file_paths: List[str],
                         importance_scores: List[float] = None) -> bool:
        """
        Add new knowledge and trigger forgetting if needed

        Args:
            file_paths: Paths to new files
            importance_scores: Optional manual importance (0-10)

        Returns:
            True if added, False if rejected due to budget
        """
        if not file_paths:
            return True

        # Compress new files
        converter = FileToImageConverter()
        new_tokens = []

        for file_path, importance in zip(
            file_paths,
            importance_scores or [5.0] * len(file_paths)
        ):
            try:
                images = converter.batch_convert([file_path])
                for img in images:
                    embeddings = self.encoder.compress(img)
                    memory_id = self.memory_bank.add_memory(
                        file_path,
                        embeddings,
                        metadata={'importance': importance}
                    )
                    new_tokens.append((memory_id, embeddings.shape[0]))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # Check budget
        new_total = sum(t[1] for t in new_tokens)
        if self.current_tokens + new_total > self.max_vision_tokens:
            # Trigger forgetting
            excess = self.current_tokens + new_total - self.max_vision_tokens
            self._forget(excess_tokens=excess)

        self.current_tokens = self.memory_bank.get_total_tokens()
        return True

    def _importance_score(self, memory_chunk: Dict) -> float:
        """
        Compute importance score for a memory chunk

        Factors:
        1. Manual importance (0-10)
        2. Recency: newer = more important
        3. Access frequency: accessed = more important

        Returns:
            Combined importance score (higher = more important)
        """
        metadata = memory_chunk['metadata']
        timestamp = memory_chunk['timestamp']
        access_count = memory_chunk['access_count']

        # Base scores
        manual_importance = metadata.get('importance', 5.0)  # 0-10

        # Recency: decay over days
        days_old = (time.time() - timestamp) / (24 * 3600)
        recency_score = 10.0 / (1.0 + days_old / 30)  # Half-life ~30 days

        # Frequency: log scale
        frequency_score = 10.0 * (1.0 - np.exp(-access_count / 5))

        # Weighted combination
        importance = (
            manual_importance * 0.5 +  # User input is most important
            recency_score * 0.3 +       # Recent = somewhat important
            frequency_score * 0.2       # Used = less important
        )

        return importance

    def _forget(self, excess_tokens: int):
        """
        Implement forgetting to reduce tokens below budget

        Strategies (applied in order):
        1. Drop least important memories entirely
        2. Re-compress high-access memories at higher ratio
        3. Merge similar memories

        Args:
            excess_tokens: Number of tokens to remove
        """
        print(f"\n=== Triggering Forgetting ===")
        print(f"Need to remove {excess_tokens} tokens")

        # Strategy 1: Drop least important
        scored_memories = [
            (i, self._importance_score(m), m)
            for i, m in enumerate(self.memory_bank.memories)
        ]
        scored_memories.sort(key=lambda x: x[1])  # Sort by importance

        removed_tokens = 0
        removed_indices = []

        for idx, importance, memory in scored_memories:
            if removed_tokens >= excess_tokens:
                break

            num_tokens = memory['embeddings'].shape[0]
            if importance < 3.0:  # Low importance threshold
                removed_tokens += num_tokens
                removed_indices.append(idx)
                print(f"  Dropped {memory['metadata']['source']} "
                      f"(importance={importance:.2f}, tokens={num_tokens})")

        # Remove from memory bank (in reverse order to preserve indices)
        for idx in sorted(removed_indices, reverse=True):
            del self.memory_bank.memories[idx]

        # Strategy 2: Re-compress memories (if still over budget)
        remaining_excess = excess_tokens - removed_tokens
        if remaining_excess > 0:
            print(f"\nRe-compressing frequently-accessed memories...")
            for memory in self.memory_bank.memories:
                if memory['access_count'] > 10 and remaining_excess > 0:
                    # Re-compress at higher ratio (e.g., 256 → 128 tokens)
                    old_tokens = memory['embeddings'].shape[0]
                    # Apply pooling or re-compression
                    memory['embeddings'] = self._recompress_embeddings(
                        memory['embeddings'],
                        target_tokens=old_tokens // 2
                    )
                    new_tokens = memory['embeddings'].shape[0]
                    freed_tokens = old_tokens - new_tokens
                    remaining_excess -= freed_tokens
                    print(f"  Compressed {memory['metadata']['source']}: "
                          f"{old_tokens} → {new_tokens} tokens")

        self.current_tokens = self.memory_bank.get_total_tokens()
        print(f"\nForgetting complete. New total: {self.current_tokens} tokens\n")

    def _recompress_embeddings(self, embeddings: torch.Tensor,
                               target_tokens: int) -> torch.Tensor:
        """
        Re-compress embeddings to fewer tokens (lossy)

        Methods:
        1. Average pooling
        2. Max pooling
        3. Learned pooling
        """
        current_tokens = embeddings.shape[0]
        if current_tokens <= target_tokens:
            return embeddings

        # Simple average pooling
        pool_size = (current_tokens + target_tokens - 1) // target_tokens
        reshaped = embeddings[:target_tokens * pool_size]
        pooled = reshaped.reshape(target_tokens, pool_size, -1).mean(dim=1)
        return pooled

    def get_context_for_query(self, query: str,
                             top_k: int = None) -> torch.Tensor:
        """
        Retrieve relevant compressed memories for a query

        Args:
            query: User query text
            top_k: Max number of memories to return

        Returns:
            Concatenated vision embeddings
            Shape: (num_selected_tokens, 1280)
        """
        if not self.memory_bank.memories:
            return torch.empty((0, 1280))

        # Rank memories by relevance to query
        ranked = self._rank_by_relevance(query, self.memory_bank.memories)

        # Select memories that fit in context
        selected_embeddings = []
        token_count = 0

        for memory_id, score in ranked:
            memory = self._get_memory_by_id(memory_id)
            num_tokens = memory['embeddings'].shape[0]

            # Check if it fits
            if token_count + num_tokens <= self.max_vision_tokens:
                selected_embeddings.append(memory['embeddings'])
                token_count += num_tokens

                # Update access count (for importance scoring)
                memory['access_count'] += 1

        if selected_embeddings:
            return torch.cat(selected_embeddings, dim=0)
        else:
            return torch.empty((0, 1280))

    def _rank_by_relevance(self, query: str,
                          memories: List[Dict]) -> List[Tuple[str, float]]:
        """
        Rank memories by relevance to query

        Strategies:
        1. Keyword matching
        2. Semantic similarity (embedding-based)
        3. Metadata matching

        Returns:
            List of (memory_id, relevance_score) sorted by score
        """
        from sklearn.metrics.pairwise import cosine_similarity

        ranked = []

        for memory in memories:
            score = 0.0

            # Strategy 1: Keyword matching
            source = memory['metadata']['source'].lower()
            category = memory['metadata'].get('category', '').lower()
            query_lower = query.lower()

            keyword_hits = sum(
                1 for word in query_lower.split()
                if len(word) > 3 and word in source + ' ' + category
            )
            score += keyword_hits * 2.0

            # Strategy 2: Recency boost
            days_old = (time.time() - memory['timestamp']) / (24 * 3600)
            recency = 5.0 / (1.0 + days_old / 7)  # Half-life 7 days
            score += recency

            # Strategy 3: Manual importance
            score += memory['metadata'].get('importance', 5.0) * 0.5

            ranked.append((memory['id'], score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _get_memory_by_id(self, memory_id: str) -> Dict:
        """Look up memory by ID"""
        idx = self.memory_bank.index.get(memory_id)
        if idx is not None:
            return self.memory_bank.memories[idx]
        return None

    def save(self, path: str):
        """Save adaptive memory state"""
        self.memory_bank.save()

    def load(self, path: str):
        """Load adaptive memory state"""
        self.memory_bank.load()

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_memories': len(self.memory_bank.memories),
            'total_tokens': self.current_tokens,
            'max_tokens': self.max_vision_tokens,
            'utilization': f"{100 * self.current_tokens / self.max_vision_tokens:.1f}%",
            'avg_importance': np.mean([
                self._importance_score(m)
                for m in self.memory_bank.memories
            ]) if self.memory_bank.memories else 0,
        }
```

---

## Phase 5: Integration & End-to-End Pipeline

**Goal:** Build complete system for daily use with compressed memory.

**Duration:** 1-2 days
**Output:** End-to-end pipeline scripts and API

### 5.1 End-to-End Pipeline

**File:** `vision_compression/pipelines/end_to_end_pipeline.py`

```python
class CompressedMemoryLLMSystem:
    """
    Complete system for querying gpt-oss-20b with compressed personal knowledge
    """

    def __init__(self, model_path: str,
                 memory_bank_path: str,
                 device: str = 'cuda'):
        """
        Initialize the complete system

        Args:
            model_path: Path to fine-tuned vision-augmented gpt-oss-20b
            memory_bank_path: Path to compressed memory bank
            device: 'cuda' or 'cpu'
        """
        # Load fine-tuned model
        self.model = VisionAugmentedGPT.from_pretrained(model_path)
        self.model = self.model.to(device)
        self.model.eval()

        # Load memory
        self.memory = AdaptiveCompressedMemory.load(memory_bank_path)

        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('gpt-oss-20b')

        self.device = device

    def query(self, query_text: str,
              max_new_tokens: int = 512,
              temperature: float = 0.7,
              top_p: float = 0.9) -> str:
        """
        Query the system with compressed memory context

        Args:
            query_text: User's question
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated response grounded in personal knowledge
        """

        # Retrieve relevant memories
        vision_context = self.memory.get_context_for_query(query_text)

        if vision_context.shape[0] == 0:
            # No relevant memories found
            return "No relevant knowledge found in memory. Please add related files."

        # Tokenize query
        query_tokens = self.tokenizer(
            query_text,
            return_tensors='pt',
            padding=True,
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                vision_embeddings=vision_context.unsqueeze(0),  # Add batch dim
                input_ids=query_tokens['input_ids'],
                attention_mask=query_tokens['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        # Decode output
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def add_knowledge(self, file_paths: List[str],
                     importance_scores: List[float] = None):
        """
        Add new files to knowledge base

        Automatically triggers forgetting if memory exceeds budget
        """
        self.memory.add_new_knowledge(file_paths, importance_scores)
        print(f"Knowledge added. Memory stats: {self.memory.get_stats()}")

    def set_importance(self, file_path_pattern: str, importance: float):
        """
        Manually set importance for files matching pattern

        Args:
            file_path_pattern: Pattern to match files (e.g., "important_*.pdf")
            importance: Importance score 0-10
        """
        import fnmatch
        for memory in self.memory.memory_bank.memories:
            source = memory['metadata']['source']
            if fnmatch.fnmatch(source, file_path_pattern):
                memory['metadata']['importance'] = importance
                print(f"Set importance={importance} for {source}")

    def show_memory_stats(self):
        """Display current memory statistics"""
        stats = self.memory.get_stats()
        print("\n=== Memory Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

    def list_memories(self) -> pd.DataFrame:
        """List all compressed memories"""
        return self.memory.memory_bank.list_memories()
```

### 5.2 Usage Examples

**File:** `examples/example_full_workflow.py`

```python
def main():
    """Complete workflow example"""

    print("=== Vision Token Compression for LLM Context ===\n")

    # === INITIALIZATION (one-time) ===

    # Phase 1: Compress personal knowledge
    print("Phase 1: Compressing personal knowledge...")
    memory = compress_knowledge_base(
        input_dirs=[
            os.path.expanduser('~/Documents'),
            os.path.expanduser('~/Pictures'),
        ],
        output_dir='./compressed_memory',
        resolution='tiny',  # 64 tokens, 20x compression
    )

    # Phase 2: Generate training data
    print("\nPhase 2: Generating training data...")
    training_data = generate_training_data(
        memory_bank=memory,
        output_path='./training_data.json',
        num_questions_per_memory=3,
    )

    # Phase 3: Fine-tune gpt-oss-20b
    print("\nPhase 3: Fine-tuning gpt-oss-20b...")
    finetune_gpt_oss(
        gpt_model_name='gpt-oss-20b',
        training_data_path='./training_data.json',
        output_dir='./finetuned_gpt',
        num_epochs=3,
        batch_size=4,
    )

    # === DAILY USAGE ===

    # Load the system
    print("\nInitializing system...")
    system = CompressedMemoryLLMSystem(
        model_path='./finetuned_gpt/final_model',
        memory_bank_path='./compressed_memory',
    )

    system.show_memory_stats()

    # Query with compressed memory
    print("\n=== Interactive Mode ===\n")
    while True:
        query = input("Your question (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break

        response = system.query(query)
        print(f"\nResponse:\n{response}\n")

    # Add new knowledge
    print("\nAdding new document...")
    system.add_knowledge(
        file_paths=['new_document.pdf'],
        importance_scores=[8.0],  # High importance
    )

    system.show_memory_stats()

if __name__ == '__main__':
    main()
```

---

## Technical Specifications

### Model Specifications

| Component | Details |
|---|---|
| **gpt-oss-20b** | Base LLM (text-only) |
| **Context Length** | 8000 tokens |
| **Hidden Dimension** | 4096 |
| **DeepEncoder** | 380M parameters (SAM 80M + CLIP 300M) |
| **Vision Embedding Dim** | 1280 |
| **Vision Token Projector** | 1280 → 4096 linear + LayerNorm |

### Compression Specs

| Setting | Tokens | Compression | Quality |
|---|---|---|---|
| Tiny (512×512) | 64 | 20x | Gist (~60%) |
| Small (640×640) | 100 | 10x | Good (~90%) |
| Base (1024×1024) | 256 | 4x | Excellent (~97%) |
| Large (1280×1280) | 400 | 2x | Very High |

### Training Specs

| Parameter | Value |
|---|---|
| Fine-tuning Method | LoRA (r=16, alpha=32) |
| Target Modules | q_proj, v_proj (+ vision_projector) |
| Training Examples | 5,000-10,000 |
| Batch Size | 4 (gradient acc. = 4 → eff. 16) |
| Learning Rate | 3e-5 |
| Warmup Steps | 500 |
| Max Epochs | 3 |
| Mixed Precision | FP16 |
| Gradient Checkpointing | Enabled |
| RTX 5090 Memory | ~40GB (QLoRA: ~20GB) |

---

## Expected Results & Limitations

### What Works Well

✅ **High-level reasoning** over documents
✅ **Semantic search** through personal knowledge
✅ **Summarization** of compressed documents
✅ **Cross-document synthesis** and analysis
✅ **Automatic memory management** with forgetting
✅ **Fast inference** (just 400 vision tokens + prompt)

### What's Lossy

❌ **Exact quotes**: 60% fidelity at 20x compression
❌ **Precise numbers**: May hallucinate without being in image
❌ **Fine-grained details**: Lost in aggressive compression
❌ **Special formatting**: Code indentation, layout details

### When to Use Different Compression Ratios

- **20x (64 tokens)**: Gist of documents, high-level ideas
- **10x (100 tokens)**: Balanced trade-off (default)
- **4x (256 tokens)**: Precise information needed
- **2x (400 tokens)**: Maximum fidelity, rare updates

---

## Timeline & Resources

### Development Timeline

| Phase | Duration | Compute | Output |
|---|---|---|---|
| Phase 1: Compression | 1-2 days | 1×GPU | Memory bank (~400 tokens) |
| Phase 2: Data Gen | 2-3 days | 1×GPU | 5k-10k training examples |
| Phase 3: Fine-tuning | 3-5 days | RTX 5090 | LoRA-adapted gpt-oss-20b |
| Phase 4: Forgetting | 1-2 days | CPU | Importance-based pruning |
| Phase 5: Integration | 1-2 days | CPU | End-to-end scripts |
| **Total** | **~2 weeks** | **RTX 5090** | **Complete system** |

### Resource Requirements

- **Compute**: RTX 5090 (12-24GB VRAM for fine-tuning with LoRA)
- **Storage**: ~10GB for models + embeddings
- **Time**: ~2-3 weeks part-time development

### Deliverables

1. ✅ Compressed memory bank (400-800 vision tokens)
2. ✅ Training dataset (5k-10k examples)
3. ✅ Fine-tuned gpt-oss-20b (LoRA weights)
4. ✅ Adaptive memory system (with forgetting)
5. ✅ End-to-end pipeline (query → response)
6. ✅ Usage scripts and examples
7. ✅ Documentation and implementation guide

---

## Running the Implementation

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Compress your knowledge base
python -m vision_compression.pipelines.compress_knowledge_base \
    --input ~/Documents ~/Pictures \
    --output ./compressed_memory \
    --resolution tiny

# Phase 2: Generate training data
python -m vision_compression.pipelines.generate_training_data \
    --memory-bank ./compressed_memory \
    --output ./training_data.json

# Phase 3: Fine-tune
python -m vision_compression.pipelines.finetune_gpt \
    --training-data ./training_data.json \
    --output ./finetuned_gpt \
    --epochs 3 \
    --batch-size 4

# Phase 5: Run interactive system
python -m examples.example_full_workflow
```

### Advanced Usage

```python
# Load and query the system
from vision_compression.pipelines.end_to_end_pipeline import CompressedMemoryLLMSystem

system = CompressedMemoryLLMSystem(
    model_path='./finetuned_gpt/final_model',
    memory_bank_path='./compressed_memory',
)

# Query
response = system.query("What did I learn about transformers?")

# Add new knowledge with forgetting
system.add_knowledge(['new_paper.pdf'], importance_scores=[9.0])

# Manage memory
system.show_memory_stats()
```

---

## References & Related Work

### Key Papers

- DeepSeek-OCR: Context Optical Compression (Wei et al., 2025)
- Vision Language Models: Survey and Taxonomy (Tsimpoukelli et al., 2021)

### Related Technologies

- **LoRA**: Low-Rank Adaptation for Large Language Models (Hu et al., 2021)
- **Dynamic Prompting**: Retrieval-Augmented Generation for LLMs
- **Forgetting in LLMs**: Mechanisms and Applications

---

## Notes & Future Work

### Future Enhancements

1. **Semantic-aware forgetting**: Use embeddings to merge similar memories
2. **Multi-modal fusion**: Combine vision + text embeddings
3. **Retrieval optimization**: BM25 + semantic search
4. **Streaming compression**: Process large files incrementally
5. **Knowledge distillation**: Compress already-compressed memories

### Known Limitations

- 60% fidelity at aggressive 20x compression
- No support for very small text (< 6pt)
- Layout information partially lost
- Requires fine-tuning (not zero-shot)

---

**Date Created**: 2025-10-22
**Author**: Claude Code
**Status**: Ready for Implementation
