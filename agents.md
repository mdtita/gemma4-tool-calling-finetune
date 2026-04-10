# Agentic Fine-Tuning — Project Knowledge Base

> **Purpose**: This document captures all research, decisions, gotchas, and hard-won learnings from fine-tuning models for agentic tool-calling. It covers **four separate training pipelines** across two model families. It is designed so that any AI agent or developer can pick up this project without repeating the research.

> **⚠️ IMPORTANT**: This repo contains scripts for **multiple models and pipelines**. They share some datasets and infrastructure but have different architectures, configs, and gotchas. Do NOT mix settings between them.

---

## Table of Contents

1. [Project Overview & Pipeline Map](#project-overview--pipeline-map)
2. [Hardware Environment](#hardware-environment)
3. [Datasets](#datasets)
4. [PIPELINE A: Gemma 4 E4B — Local SFT](#pipeline-a-gemma-4-e4b--local-sft)
5. [PIPELINE B: Gemma 4 31B — Cloud SFT](#pipeline-b-gemma-4-31b--cloud-sft)
6. [PIPELINE C: Qwen 3.5 9B — Local SFT + GRPO](#pipeline-c-qwen-35-9b--local-sft--grpo)
7. [PIPELINE D: Gemma 4 E4B — Arabic Agent (CPT → SFT → GRPO)](#pipeline-d-gemma-4-e4b--arabic-agent-cpt--sft--grpo)
8. [Critical Gotchas & Solved Issues](#critical-gotchas--solved-issues)
9. [Evaluation](#evaluation)
10. [File Inventory](#file-inventory)
11. [Next Steps](#next-steps)
12. [Reference Links](#reference-links)

---

## Project Overview & Pipeline Map

**Goal**: Train models into high-performance agentic coding assistants capable of autonomous tool-calling, multi-turn reasoning, and complex coding workflows.

**Multi-phase approach** (varies by pipeline):
1. **CPT (Continued Pretraining)** — Strengthens language token representations (e.g., Arabic)
2. **SFT (Supervised Fine-Tuning)** — Teaches tool-calling format, JSON structure, multi-turn patterns
3. **GRPO (Group Relative Policy Optimization)** — RL phase to sharpen quality using reward functions

### Pipeline Summary

| Pipeline | Model | Where | Status | Script |
|----------|-------|-------|--------|--------|
| **A** | Gemma 4 E4B (~8B) | Local (single GPU) | ✅ SFT converged (loss 0.14-0.38) | `Gemma4_E4B_Agentic_SFT.py` |
| **B** | Gemma 4 31B | Cloud (OneClickAMD) | ✅ SFT complete, GGUF exported | `Gemma4_31B_Arabic_Agent_Cloud.ipynb` |
| **C** | Qwen 3.5 9B | Local (dual GPU) | ⚠️ SFT had OOM issues, GRPO planned | `Qwen3.5_9B_Agentic_SFT.py` |
| **D** | Gemma 4 E4B (~8B) | Local (single GPU) | ✅ SFT done → GRPO in progress | `Gemma4_E4B_Arabic_*.py` |

---

## Hardware Environment

### Local System
- **GPUs**: 2x AMD Radeon RX 9060 XT (16 GB VRAM each, 32 GB total)
- **Platform**: Linux, ROCm 7.1.25424
- **PyTorch**: 2.10.0+rocm7.1
- **Triton**: 3.6.0
- **Python**: 3.14

### Key Hardware Constraints
- **No Flash Attention 2** on ROCm — Unsloth falls back to Xformers (harmless warning, no perf impact)
- **`expandable_segments` recommended** — Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256` for all scripts to reduce HIP memory fragmentation
- **GPU 1 is the busier GPU** on this system (likely runs display server or other tasks)
- **GPU 0 is less busy** → preferred for training when single-GPU
- **Multi-GPU model parallelism (device_map) does NOT work with Unsloth + 4-bit** — BnB quantization conflicts with CPU/disk offload
- **Multi-GPU NOT supported by Unsloth natively** — Unsloth is designed for single-GPU speed; use parameter tuning to fit within 16GB
- **Multi-GPU `device_map="balanced"` works for Qwen bf16** — but must carefully cap `max_memory` per GPU

### Cloud (OneClickAMD) — Pipeline B only
- Used for Gemma 4 31B training (model too large for local GPUs)
- 50-minute VM sessions with automatic kernel crashes
- Required chunked training with HF Hub checkpoint sync (see Pipeline B)

---

## Datasets

### Dataset Files
| File | Size | Used By | Description |
|------|------|---------|-------------|
| `gemma4_e4b_agentic_dataset.jsonl` | 82 MB | Pipeline A | 15k curated, Gemma 4 format |
| `arabic_agentic_dataset.jsonl` | 37 MB | Pipeline D (SFT) | Arabic agentic training data |
| `curated_agentic_dataset.jsonl` | 150 MB | Pipeline C | ~41k curated, ShareGPT format (proven on Qwen) |
| `enhanced_agentic_dataset.jsonl` | 304 MB | Pipeline C (GRPO) | Larger raw dataset |
| Arabic GSM8K (HF Hub) | ~7.5k | Pipeline D (GRPO) | `Omartificial-Intelligence-Space/Arabic-gsm8k-v2` |

### Dataset Sources (for `curate_gemma4_dataset.py`)

| # | Source | Quality | Gated? | Size |
|---|--------|---------|--------|------|
| 1 | `Salesforce/xlam-function-calling-60k` | Gold standard | ✅ Yes (request access) | 60k |
| 2 | `Salesforce/APIGen-MT-5k` | Multi-turn verified | No | 5k |
| 3 | `NousResearch/hermes-function-calling-v1` | High quality FC | No | ~12k |
| 4 | `glaiveai/glaive-function-calling-v2` | Broad coverage | No | 100k+ |
| 5 | Existing `curated_agentic_dataset.jsonl` | Proven on Qwen | Local | ~41k |

### Quality Scoring Formula (curate_gemma4_dataset.py)
```python
score = tool_score * 0.35 + turn_score * 0.20 + content_score * 0.15 + source_weight * 0.30
```
- **tool_score**: 1.0 if tool call + result, 0.6 tool call only, 0.3 neither
- **turn_score**: peaks at 4-10 turns (1.0)
- **content_score**: peaks at 500-3000 chars (1.0)
- **source_weight**: per-source quality weight (xlam=1.0, glaive=0.8, etc.)

---

## PIPELINE A: Gemma 4 E4B — Local SFT

### Script: `Gemma4_E4B_Agentic_SFT.py`
### Model: `unsloth/gemma-4-E4B-it` (~8B dense, multimodal-capable)
### Dataset: `gemma4_e4b_agentic_dataset.jsonl` (15k samples)
### HF Output: `mtita/gemma4-e4b-agentic-lora`

### Architecture Notes
- Despite being called "E4B", it has **8B parameters**
- Has a **SigLip vision encoder** (multimodal-capable) but we train **text-only**
- Unsloth classifies it as "processor-based" → packing is skipped
- Uses `FastModel` (NOT `FastVisionModel`, even though it has vision)

### Gemma 4 Chat Template
```
<bos><|turn>user\nHello<turn|>\n<|turn>model\nHi there!<turn|>\n
```
**CRITICAL**: Opening tag is `<|turn>` (pipe on LEFT only), NOT `<|turn|>`.

### Role Normalization (Gemma 4 has NO tool role)
```python
# System → user + assistant acknowledgment
{"role": "user", "content": "[System Instructions]\n{content}"}
{"role": "assistant", "content": "Understood."}

# Tool results → user message
{"role": "user", "content": "[Tool Result: {tool_name}]\n{output}"}

# Consecutive same-role → merged with \n\n
```

### Proven Configuration
```python
MODEL_NAME        = "unsloth/gemma-4-E4B-it"
MAX_SEQ_LENGTH    = 2048
DTYPE             = torch.bfloat16
LOAD_IN_4BIT      = True           # 4-bit QLoRA on single GPU
LORA_R            = 16
LORA_ALPHA        = 16
BATCH_SIZE        = 1              # batch_size=2 OOMs
GRAD_ACCUM        = 4              # Effective batch = 4
LEARNING_RATE     = 2e-4
OPTIM             = "adamw_8bit"
SAVE_STEPS        = 250

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Pin to less-busy GPU
```

### Key Settings
```python
from unsloth import FastModel  # NOT FastVisionModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

# Response-only masking
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",
    response_part    = "<|turn>model\n",
)

# BOS stripping (processor adds it)
if text.startswith("<bos>"):
    text = text[5:]
```

### Training Results
```
Step    5: loss=12.19  (random baseline)
Step   10: loss=7.63   (rapid learning)
Step   15: loss=4.40
Step   20: loss=3.20
Step   25: loss=2.65
...
Step 2142: loss=0.14-0.38  (fully converged at 57% epoch)
```
- **Trainable**: 36.7M / 8B (0.46%)
- **VRAM**: 9.6 GB reserved (6 GB headroom on 16 GB GPU)
- **Speed**: ~13-20 s/step
- **Checkpoints**: checkpoint-1500, checkpoint-1750, checkpoint-2000

---

## PIPELINE B: Gemma 4 31B — Unified Cloud Pipeline (SFT + GRPO)

### Notebook: `Gemma4_31B_Arabic_Agent_Cloud.ipynb`

### The Cloud Advantage
Local AMD GPUs (16GB) completely fail on Gemma 4 GRPO due to the massive 256,000 vocabulary size `lm_head` tensor (~1.25 GB allocated instantly during backward pass). To solve this, the pipeline was moved to OneClickAMD VMs providing >100GB VRAM GPUs (e.g. MI300X).

### SFT Phase Configuration
- **Model**: `unsloth/gemma-4-31B-it`
- `max_seq_length`: **8192** (VRAM limit removed)
- `load_in_4bit`: `False` (Train in pure bfloat16 for max quality)
- `optim`: `adamw_torch`

### GRPO Phase Configuration
- **Dataset**: `Arabic-gsm8k-v2`
- `num_generations`: **8** (Max quality, mathematically impossible locally)
- `max_completion_length`: **1024** (Deep reasoning room)
- `max_prompt_length`: **256** (Stops TRL padding overhead)

### Cloud Crash Resilience Strategy
OneClickAMD VMs die every ~50 minutes. The solution:
1. **Chunked training**: 600-step chunks for SFT, 100-step chunks for GRPO.
2. **Checkpoint sync to HF Hub**: After each chunk, synchronous upload via `HfApi.upload_folder()`.
3. **Auto-resume**: On VM restart, pull checkpoints from Hub via `snapshot_download()`.
4. **No async push**: `push_to_hub=False` because the async background thread crashes.

```python
# Core crash-resilience loop
CHUNK_SIZE = 600
TOTAL_CHUNKS = 30
for i in range(TOTAL_CHUNKS):
    LAST_CHECKPOINT = get_last_checkpoint(TRAINING_OUTPUT_DIR)
    trainer.args.max_steps = (i + 1) * CHUNK_SIZE
    if completed_steps >= current_max_steps:
        continue  # Skip completed chunks
    trainer.train(resume_from_checkpoint=LAST_CHECKPOINT)
    # Synchronous upload after each chunk
    api.upload_folder(folder_path=..., repo_id=..., allow_patterns=["checkpoint-*/*"])
```

### Configuration
```python
model = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-31B-it",
    dtype = torch.bfloat16,
    max_seq_length = 4096,
    load_in_4bit = False,  # bf16 (cloud has enough VRAM)
    full_finetuning = False,
)
model = FastModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    finetune_vision_layers=False,
    use_gradient_checkpointing=True,  # Not "unsloth" — standard for cloud stability
)
```

### Training Config
```python
SFTConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_train_epochs = 2,
    learning_rate = 1e-4,       # Note: lower than E4B's 2e-4
    optim = "adamw_torch",      # Note: NOT adamw_8bit (cloud has VRAM)
    lr_scheduler_type = "linear",
    save_steps = 600,
)
```

### GGUF Export
```python
model.save_pretrained_gguf("gemma_4_finetune", tokenizer, quantization_method="q4_k_m")
model.push_to_hub_gguf("mtita/gemma4-31b-tool-calling-q4_k_m-GGUF", tokenizer,
                        quantization_method="q4_k_m", token=HF_TOKEN)
```

### Downloaded GGUF
- **Local path**: `models/gemma-4-31B-it.Q4_K_M.gguf` (~18.7 GB)
- **Ollama Modelfile**: `models/Modelfile`

### Cloud Notebook: `Gemma4_31B_Arabic_Agent_Cloud.ipynb`
Updated version of the cloud pipeline for Arabic agentic training.
- Install cell now includes all required packages (`datasets`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `huggingface_hub`)
- Uses `processing_class=tokenizer` (not deprecated `tokenizer=`)
- Stray `packing=True` removed from `FastModel.from_pretrained()` (only valid in `SFTConfig`)
- Graceful `try/except` on `unsloth` uninstall (prevents crash on fresh environments)

---

## PIPELINE C: Qwen 3.5 9B — Local SFT + GRPO

### Scripts:
- SFT: `Qwen3.5_9B_Agentic_SFT.py`
- GRPO: `Qwen3.5_9B_Agentic_GRPO.py`

### Model: `unsloth/Qwen3.5-9B`
### Dataset (SFT): `curated_agentic_dataset.jsonl` (~41k samples)
### Dataset (GRPO): `enhanced_agentic_dataset.jsonl` (~304 MB)

### Critical: Qwen 3.5 Architecture Differences from Gemma 4

| Feature | Qwen 3.5 | Gemma 4 |
|---------|----------|---------|
| **API** | `FastLanguageModel` | `FastModel` |
| **Chat template** | `qwen-2.5` | `gemma-4` |
| **Quantization** | ❌ bf16 only (QLoRA degrades quality) | ✅ 4-bit QLoRA works well |
| **Multi-GPU** | ✅ `device_map="balanced"` works | ❌ device_map breaks with Unsloth |
| **VRAM (bf16)** | ~19 GB (needs 2 GPUs) | ~16 GB (needs 2 GPUs) |
| **Has DeltaNet** | ✅ (FLA kernels needed) | ❌ |
| **Tool role** | ✅ Native `tool` role | ❌ Fold into `user` |

### ROCm FLA Patch (CRITICAL for Qwen 3.5)
Qwen 3.5 uses **GatedDeltaNet layers** which require Flash Linear Attention (FLA). On ROCm, `is_torch_cuda_available()` returns False, hiding FLA. Must monkey-patch BEFORE model loading:

```python
import importlib
def _patch_fla_availability():
    import fla
    import transformers.utils.import_utils as _iu
    _iu.is_flash_linear_attention_available = lambda: True
    _iu.is_flash_linear_attention_available.cache_clear = lambda: None
    import transformers.models.qwen3_5.modeling_qwen3_5 as _m
    importlib.reload(_m)
_patch_fla_availability()
```

### Qwen SFT Configuration
```python
max_seq_length = 2048
max_memory = {0: "14GiB", 1: "11GiB"}  # Skew to GPU 0 (less busy)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-9B",
    load_in_4bit = False,          # ⚠️ Unsloth warns: QLoRA degrades Qwen 3.5
    load_in_16bit = True,          # bf16 LoRA as recommended
    device_map = "balanced",       # Split across both GPUs
    max_memory = max_memory,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,    # Lower rank for bf16 VRAM budget
    use_gradient_checkpointing = "unsloth",
)

# Chat template
tokenizer = get_chat_template(
    tokenizer, chat_template="qwen-2.5",
    mapping={"role": "role", "content": "content", "user": "user",
             "assistant": "assistant", "system": "system"}
)
```

### Qwen Training Config
```python
SFTConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-5,         # Lower LR for bf16 LoRA
    optim = "adamw_8bit",
    lr_scheduler_type = "cosine",
    num_train_epochs = 1,
    save_steps = 500,
    dataloader_pin_memory = False,  # Not useful on ROCm
)
```

### Conversation Splitting
Qwen script includes `split_long_conversation()` which splits conversations exceeding `max_tokens` at turn boundaries, prepending the system message to each chunk. This is unique to Pipeline C.

### GRPO Configuration (Planned)
```python
from trl import GRPOConfig, GRPOTrainer

# Reward functions
def tool_format_reward(completions):    # Is tool call valid JSON? +2.0 / +0.5 / -0.5
def reasoning_structure_reward(completions):  # Has <think>...</think>? +1.0 / 0.0

GRPOConfig(
    learning_rate = 5e-6,
    num_generations = 4,         # 4 completions per prompt
    max_completion_length = 2048,
    max_prompt_length = 1024,
    max_steps = 500,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2,
    save_steps = 50,
    loss_type = "grpo",
    mask_truncated_completions = True,
)
```

### Qwen OOM Issues (Known)
- bf16 Qwen 3.5 needs ~19 GB total → must split across 2 GPUs
- `max_memory = {0: "14GiB", 1: "11GiB"}` prevents driver-level OOM
- NO CPU spillover — kills performance
- `gc.collect(); torch.cuda.empty_cache()` after model load and adapter setup
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation

---

## PIPELINE D: Gemma 4 E4B — Arabic Agent (CPT → SFT → GRPO)

### Scripts:
- CPT: `Gemma4_E4B_Arabic_CPT.py`
- SFT: `Gemma4_E4B_Arabic_Agent_SFT.py`
- GRPO: `Gemma4_E4B_Arabic_GRPO.py`

### Model: `unsloth/gemma-4-E4B-it`
### HF Outputs:
- SFT LoRA: `mtita/gemma4-e4b-arabic-agent-lora`
- CPT LoRA: `mtita/gemma4-e4b-arabic-cpt-lora`
- GRPO LoRA: `mtita/gemma4-e4b-arabic-agent-grpo-lora`

### Three-Phase Pipeline
```
CPT (optional) → SFT → GRPO
  │                │       └── Reinforcement learning with Arabic reward functions
  │                └── Supervised fine-tuning on Arabic agentic conversations
  └── Strengthen Arabic token representations on raw text
```

### Phase 1: CPT (Continued Pretraining)
**Script**: `Gemma4_E4B_Arabic_CPT.py`
**Purpose**: Improve Arabic language model quality before instruction tuning.

```python
# CPT requires training embed_tokens + lm_head with a smaller LR
# Uses UnslothTrainer + UnslothTrainingArguments (NOT SFTTrainer)
from unsloth import UnslothTrainer, UnslothTrainingArguments

model = FastModel.get_peft_model(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head", "embed_tokens",  # Critical for CPT!
    ],
    r=32, lora_alpha=32,     # Higher rank for CPT
    use_rslora=True,          # rsLoRA for better scaling
    use_gradient_checkpointing="unsloth",
)

trainer = UnslothTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=UnslothTrainingArguments(
        dataset_text_field="text",
        learning_rate=5e-5,
        embedding_learning_rate=5e-6,  # 10x smaller for embeddings!
        packing=True,
    ),
)
```

### Phase 2: SFT (Arabic Agentic)
**Script**: `Gemma4_E4B_Arabic_Agent_SFT.py`
**Dataset**: `arabic_agentic_dataset.jsonl` (local) or `mtita/gemma4-arabic-agent-training` (HF Hub)
**Status**: ✅ Complete — loss converged to 0.14-0.38

```python
# Same architecture as Pipeline A, with Arabic system prompt
MODEL_NAME    = "unsloth/gemma-4-E4B-it"
LORA_R        = 16
BATCH_SIZE    = 1
GRAD_ACCUM    = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 2
PACKING       = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
```

### Phase 3: GRPO (Arabic Reasoning RL)
**Script**: `Gemma4_E4B_Arabic_GRPO.py`
**Dataset**: Arabic GSM8K (`Omartificial-Intelligence-Space/Arabic-gsm8k-v2`, 7473 samples)
**Status**: 🔄 In progress

#### GRPO Dataset Format (CRITICAL)
TRL requires the dataset to have **only** `prompt` and optional reward-function kwargs columns.
The source dataset has a pre-existing `messages` column — **must remove it** or TRL crashes:
```python
# ✅ Correct: remove all source columns, keep only prompt + answer
dataset = ds.map(format_grpo, remove_columns=ds.column_names)

# ❌ Wrong: leaves 'messages' column → KeyError: "Invalid keys: {'messages', 'prompt'}"
dataset = ds.map(format_grpo)
```

#### GRPO Configuration (Tuned for 16GB VRAM)
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

NUM_GENERATIONS       = 4     # 4 sufficient per DeepSeek paper (was 8 → OOM)
MAX_COMPLETION_LENGTH = 256   # GSM8K answers are short math (was 512 → OOM)
MAX_SEQ_LENGTH        = 1024  # GSM8K questions are short (was 2048 → OOM)
GRAD_ACCUM            = 4     # Compensates for fewer generations
LEARNING_RATE         = 5e-6
BATCH_SIZE            = 1

GRPOConfig(
    use_vllm=False,           # Gemma 4 not in vLLM; no vLLM on ROCm
    loss_type="gspo",
    epsilon_high=0.28,
)

GRPOTrainer(
    model=model,
    processing_class=tokenizer,  # NOT tokenizer= (deprecated)
    reward_funcs=[...],
)
```

#### Arabic Reward Functions (5 total)
| Reward | Max | What It Measures |
|--------|-----|------------------|
| `arabic_language_reward` | +2.0 | Arabic character ratio in response |
| `format_reward` | +1.5 | `<تفكير>...<إجابة>` XML structure |
| `tool_call_reward` | +2.0 | Valid JSON tool call blocks |
| `quality_reward` | +1.0 | Appropriate response length |
| `correctness_reward` | +3.0 | Answer matches GSM8K ground truth |

#### Reward Function Signature (CRITICAL)
TRL passes dataset columns as **lists** (one per sample in batch):
```python
# ✅ Correct: answer is a list, zip with completions
def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    for completion, expected in zip(completions, answer):
        ...

# ❌ Wrong: treats answer as a single string
def correctness_reward(completions, answer=None, **kwargs):
    norm_expected = answer.strip()  # Crashes: list has no .strip()
```

---

## Critical Gotchas & Solved Issues

### Gemma 4 Specific

#### 1. Turn Marker Typo (CRITICAL — Pipeline A & B)
**Problem**: `<|turn|>user\n` → all 15,000 samples removed (all labels -100)
**Fix**: `<|turn>user\n` (no pipe before `>`)
**Verify**: `print(repr(tokenizer.apply_chat_template(msgs, tokenize=False)))`

#### 2. Model Name Missing `-it` Suffix (Pipeline A)
**Problem**: `unsloth/gemma-4-E4B` → `ValueError: not supported in transformers`
**Fix**: `unsloth/gemma-4-E4B-it`

#### 3. Packing Skipped (Pipeline A — Expected)
**Warning**: "Sample packing skipped (processor-based model detected)"
**Cause**: E4B has a vision processor, Unsloth classifies it as multimodal
**Impact**: No packing speedup (3-5x), but padding-free batching still works (~2x)

#### 4. Multi-GPU Model Split Doesn't Work (Pipeline A)
**Problem**: `device_map="auto"` + `load_in_4bit=True` → ValueError about CPU offload
**Also fails**: bf16 + device_map → Unsloth internal errors
**Solution**: Single GPU with `CUDA_VISIBLE_DEVICES="0"`

#### 5. Cloud Kernel Crashes (Pipeline B)
**Problem**: OneClickAMD VMs die every ~50 minutes
**Solution**: Chunked training + synchronous HF Hub upload after each chunk + auto-resume (see Pipeline B section)

### Qwen 3.5 Specific

#### 6. QLoRA Degrades Qwen 3.5 Quality
**Warning**: Unsloth explicitly warns: "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models due to higher than normal quantization differences."
**Solution**: Must use bf16 LoRA (`load_in_16bit=True`)

#### 7. FLA Not Detected on ROCm (Pipeline C)
**Problem**: `is_torch_cuda_available()` returns False on ROCm, hiding FLA kernels
**Impact**: Qwen 3.5 DeltaNet layers fall back to slow torch implementation
**Solution**: Monkey-patch before model import (see Pipeline C section)

#### 8. Dual-GPU OOM Balancing (Pipeline C)
**Problem**: Even split causes GPU 1 (busier) to OOM
**Solution**: Asymmetric `max_memory = {0: "14GiB", 1: "11GiB"}`

### Python 3.14 Specific (Pipeline D — Local)

#### 9. `dill` / `pickle` Serialization Crash
**Problem**: Python 3.14 changed the internal C-API signature of `pickle._Pickler._batch_setitems` to require an `obj` argument. `dill` and `datasets` call this without `obj`, causing `TypeError`.
**Fix**: Patched `dill/_dill.py` and `datasets/utils/_dill.py` to accept optional `obj=None` and forward it correctly.
**Root Cause**: Python 3.14 is pre-release; `dill` and `datasets` haven't updated yet.

#### 10. Unsloth Ghost Module Injection
**Problem**: `unsloth_zoo` injects dummy modules (`vllm`, `mergekit`, `llm_blender`) into `sys.modules` to bypass HuggingFace availability checks. TRL then eagerly imports deep submodules from these ghosts and crashes.
**Fix**: Wrapped eager imports in `try/except ImportError` blocks in:
- `trl/extras/vllm_client.py` (lines 35-41)
- `trl/mergekit_utils.py` (lines 21-24)
- `trl/trainer/judges.py` (lines 28-29)
- `trl/trainer/grpo_trainer.py` (lines 87-89)
**Note**: These are `site-packages` patches — technical debt caused by Python 3.14 alpha + Unsloth. Use Python 3.12 to avoid entirely.

### Shared Issues

#### 11. Batch Size 2 OOMs (Pipeline A)
**Problem**: batch_size=2 → OOM on 16 GB GPU with 4-bit QLoRA
**Solution**: batch_size=1 is the maximum

#### 12. Flash Attention 2 Broken on ROCm
**Warning**: "Flash Attention 2 installation seems to be broken. Using Xformers instead."
**Impact**: None — Xformers is equivalent
**Action**: Ignore

#### 13. transformers Version
**Requirement**: `transformers >= 5.5.0` for Gemma 4 E4B support
**Command**: `pip install --upgrade transformers`

#### 14. Import Order
**Rule**: `import unsloth` MUST be before any other ML imports
This patches the environment for 2x faster training.

#### 15. `processing_class` vs `tokenizer` in Trainers
**Rule**: All TRL trainers (`SFTTrainer`, `GRPOTrainer`) now require `processing_class=tokenizer` instead of the deprecated `tokenizer=tokenizer`. Using the old name may cause warnings or errors.

---

## Evaluation

### Test Script: `test_gemma4_31b_toolcalling.py`
Uses Ollama for inference on the 31B GGUF model.

### Test Categories (10 tests)
| Category | Tests | What It Measures |
|----------|-------|-----------------|
| Single tool call | 2 | Correct JSON format, right tool name, path arg |
| Multi-tool chain | 2 | Sequential tool use (read→edit), multi-step reasoning |
| Tool selection | 2 | write_file vs edit_file differentiation |
| Refusal | 2 | Don't hallucinate tools not in schema |
| Reasoning | 2 | Diagnose before acting, outline plan before editing |

### Running the Eval
```bash
# 1. Import GGUF into Ollama
cd models/
ollama create gemma4-31b-toolcall -f Modelfile

# 2. Run eval
python test_gemma4_31b_toolcalling.py

# 3. Compare against base model
python test_gemma4_31b_toolcalling.py --compare gemma3:27b
```

### Modelfile (Ollama)
```
FROM ./gemma-4-31B-it.Q4_K_M.gguf
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<turn|>"
```

---

## File Inventory

```
gemma4-finetune-repo-backup/
│
│  ── PIPELINE A: Gemma 4 E4B Agentic (Local) ──
├── Gemma4_E4B_Agentic_SFT.py              # SFT training (single GPU, 4-bit QLoRA)
├── curate_gemma4_dataset.py                # Dataset curation pipeline
├── gemma4_e4b_agentic_dataset.jsonl        # 15k curated training data (82 MB)
│
│  ── PIPELINE B: Gemma 4 31B (Cloud) ──
├── Gemma4_31B_Arabic_Agent_Cloud.ipynb     # Cloud notebook (FINAL, crash-resilient)
├── Gemma4_31B_ToolCalling_FineTune.ipynb               # Cloud notebook v1 (legacy)
├── Gemma4_31B_ToolCalling_FineTune_V2.ipynb             # Cloud notebook v2 (legacy)
├── Gemma4_31B_ToolCalling_FineTune_Cloud_Stable.ipynb   # Cloud notebook v3 (legacy)
├── models/
│   ├── gemma-4-31B-it.Q4_K_M.gguf         # Fine-tuned GGUF (18.7 GB)
│   └── Modelfile                           # Ollama import config
│
│  ── PIPELINE C: Qwen 3.5 9B (Local) ──
├── Qwen3.5_9B_Agentic_SFT.py              # SFT training (dual GPU, bf16 LoRA)
├── Qwen3.5_9B_Agentic_GRPO.py             # GRPO RL training (planned)
├── qwen35_9b_agentic_sft/                  # Qwen checkpoints dir
│
│  ── PIPELINE D: Gemma 4 E4B Arabic Agent (Local) ──
├── Gemma4_E4B_Arabic_CPT.py               # Phase 0: Continued pretraining on Arabic text
├── Gemma4_E4B_Arabic_Agent_SFT.py         # Phase 1: Arabic agentic SFT
├── Gemma4_E4B_Arabic_GRPO.py              # Phase 2: Arabic GRPO with reward functions
├── arabic_agentic_dataset.jsonl            # Arabic agentic training data (37 MB)
├── build_arabic_dataset.py                 # Arabic dataset builder
├── gemma4_e4b_arabic_lora/                 # SFT LoRA output (adapter files)
│
│  ── SHARED ──
├── curated_agentic_dataset.jsonl           # 41k curated (Qwen + fallback) (150 MB)
├── enhanced_agentic_dataset.jsonl          # Larger raw dataset (304 MB)
├── build_dataset.py                        # Legacy dataset builder
├── curate_dataset.py                       # Legacy curation script
├── test_gemma4_31b_toolcalling.py          # Evaluation suite (Ollama)
├── requirements.txt                        # Python dependencies
├── agents.md                               # This file
└── unsloth_compiled_cache/                 # Unsloth kernel cache
```

### Checkpoint Locations
```
~/gemma4_runs/e4b_agentic_sft/          # Pipeline A
├── checkpoint-1500/
├── checkpoint-1750/
└── checkpoint-2000/    ← Latest, loss ~0.14-0.38, fully converged

qwen35_9b_agentic_sft/                  # Pipeline C
└── (checkpoint dirs if training completed)
```

---

## Next Steps

### Immediate (Pipeline D)
1. **Complete GRPO training** — Run `Gemma4_E4B_Arabic_GRPO.py` (500 steps, ~4-8 hours)
2. **Export Arabic GRPO LoRA** — Push to `mtita/gemma4-e4b-arabic-agent-grpo-lora`
3. **Convert to GGUF** — `model.save_pretrained_gguf()` with `q4_k_m`
4. **Arabic inference validation** — Test tool-calling + reasoning in Arabic

### Pending
1. **Test 31B model** — Run eval suite against `gemma4-31b-toolcall` via Ollama
2. **Qwen 3.5 GRPO** — Design reward functions, run GRPO
3. **Run cloud notebook** — `Gemma4_31B_Arabic_Agent_Cloud.ipynb` on OneClickAMD

### Future
- DPO/ORPO as alternative to GRPO (requires preference pairs)
- Continued pretraining on coding corpora
- Context extension beyond 2048 tokens
- Vision fine-tuning for multimodal agentic tasks (Gemma 4 has SigLip encoder)
- **Migrate to Python 3.12** to eliminate all `dill`/`pickle`/`trl` serialization patches

---

## Reference Links

### Unsloth Documentation
- **Gemma 4 Training**: https://unsloth.ai/docs/models/gemma-4/train
- **Qwen 3.5 Training**: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- **Multi-GPU DDP**: https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp
- **RL/GRPO Guide**: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- **Tool Calling Guide**: https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms
- **3x Faster Packing**: https://unsloth.ai/docs/blog/3x-faster-training-packing

### Datasets
- **xLAM**: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k (gated — request access)
- **APIGen**: https://huggingface.co/datasets/Salesforce/APIGen-MT-5k
- **Hermes FC**: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- **Glaive FC**: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

### Models
- **Gemma 4 E4B base**: https://huggingface.co/unsloth/gemma-4-E4B-it
- **Gemma 4 31B base**: https://huggingface.co/unsloth/gemma-4-31B-it
- **Gemma 4 31B fine-tuned GGUF**: https://huggingface.co/mtita/gemma4-31b-tool-calling-q4_k_m-GGUF
- **Qwen 3.5 9B base**: https://huggingface.co/unsloth/Qwen3.5-9B

### Dependencies
```
transformers>=5.5.0
unsloth (latest from pip or source)
trl (latest)
torch>=2.10.0 (ROCm build)
datasets
peft
bitsandbytes
xformers>=0.0.35
fla                    # Required for Qwen 3.5 DeltaNet layers
```




