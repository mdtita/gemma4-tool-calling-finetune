"""
Gemma 4 E4B Arabic GRPO — Reinforcement Learning
==================================================
Phase 2: Sharpen Arabic agentic behaviors AFTER SFT using GRPO.
Loads the SFT adapter and applies reinforcement learning with
Arabic-specific reward functions.

Pipeline: (CPT →) SFT → GRPO (this)

What GRPO does:
- Generates multiple responses per prompt
- Scores each with Arabic-specific reward functions
- Increases probability of "good" responses, decreases "bad"
- No reward model needed — uses verifiable reward functions (RLVR)

Arabic reward functions:
1. Language adherence — responds in Arabic when prompted in Arabic
2. Tool call format  — valid JSON tool calls
3. Translation quality — contains target language text
4. Response quality  — appropriate length and structure
5. Dialect consistency — MSA for formal, Egyptian for casual

VRAM: E4B (4B params) → ~4-5 GB for QLoRA GRPO — fits easily on 16 GB

Usage:
  python Gemma4_E4B_Arabic_GRPO.py
  python Gemma4_E4B_Arabic_GRPO.py --max-steps 1000
  python Gemma4_E4B_Arabic_GRPO.py --sft-adapter gemma4_e4b_arabic_lora
"""

import os
import re
import json
import argparse

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Prevent torch.compile from allocating huge temp buffers

# ── Config ────────────────────────────────────────────────────────────────
# Load from SFT adapter (post-SFT model)
SFT_ADAPTER_PATH = "gemma4_e4b_arabic_lora"    # Local SFT output
BASE_MODEL       = "unsloth/gemma-4-E4B-it"     # Fallback if no adapter
MAX_SEQ_LENGTH   = 512       # Reduced from 1024. GSM8K questions are short.
DTYPE            = torch.bfloat16
LOAD_IN_4BIT     = True
OUTPUT_DIR       = os.path.expanduser("~/gemma4_runs/e4b_arabic_grpo")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
HF_REPO_ID       = "mtita/gemma4-e4b-arabic-agent-grpo-lora"

# GRPO hyperparams — tuned for 16 GB single-GPU
NUM_GENERATIONS  = 2        # Minimum for GRPO relative comparison (higher OOMs on 16GB)
MAX_STEPS        = 500      # Minimum 300 recommended
LEARNING_RATE    = 5e-6     # Lower than SFT — fine-tuning existing knowledge
BATCH_SIZE       = 1
GRAD_ACCUM       = 4        # Effective batch = 4 (compensates for fewer generations)
MAX_COMPLETION_LENGTH = 200 # Room for Arabic reasoning chains + short math answer


# ── Arabic System Prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """\
أنت مساعد ذكي ثنائي اللغة (عربي/إنجليزي).
عند الإجابة، فكر أولاً ثم أجب.
ضع تفكيرك داخل <تفكير> ... </تفكير>
ثم ضع إجابتك النهائية داخل <إجابة> ... </إجابة>
"""

ARABIC_RESPONSE_FORMAT = """\
<تفكير>
{reasoning}
</تفكير>
<إجابة>
{answer}
</إجابة>
"""


# ══════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTIONS — Arabic-specific
# ══════════════════════════════════════════════════════════════════════════

def has_arabic_chars(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', text))


def extract_arabic_answer(text: str) -> str:
    """Extract content from <إجابة> tags."""
    match = re.search(r'<إجابة>(.*?)</إجابة>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_arabic_reasoning(text: str) -> str:
    """Extract content from <تفكير> tags."""
    match = re.search(r'<تفكير>(.*?)</تفكير>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


# ── Reward 1: Language Adherence ──────────────────────────────────────
def arabic_language_reward(completions, **kwargs) -> list[float]:
    """
    Reward for responding in Arabic when the prompt is in Arabic.
    +2.0 if response contains substantial Arabic text
    +0.5 if response contains some Arabic
    -1.0 if response has no Arabic at all
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = max(len(text), 1)
        arabic_ratio = arabic_chars / total_chars

        if arabic_ratio > 0.3:
            rewards.append(2.0)    # Strong Arabic response
        elif arabic_ratio > 0.1:
            rewards.append(0.5)    # Some Arabic
        else:
            rewards.append(-1.0)   # Not Arabic enough
    return rewards


# ── Reward 2: Structured Format ──────────────────────────────────────
def format_reward(completions, **kwargs) -> list[float]:
    """
    Reward for using the <تفكير>...</تفكير><إجابة>...</إجابة> format.
    +1.5 if both tags present and properly nested
    +0.5 if at least answer tag present
    -0.5 if neither tag present
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        has_thinking = bool(re.search(r'<تفكير>.*?</تفكير>', text, re.DOTALL))
        has_answer = bool(re.search(r'<إجابة>.*?</إجابة>', text, re.DOTALL))

        if has_thinking and has_answer:
            rewards.append(1.5)
        elif has_answer:
            rewards.append(0.5)
        else:
            rewards.append(-0.5)
    return rewards


# ── Reward 3: Tool Call Format ───────────────────────────────────────
def tool_call_reward(completions, **kwargs) -> list[float]:
    """
    Reward for valid tool calling JSON when tool usage is expected.
    Checks if JSON blocks are parseable and have 'tool' + 'arguments' keys.
    +2.0 for valid tool call JSON
    +0.0 for no JSON (neutral — not all prompts need tools)
    -1.0 for malformed JSON
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        # Look for JSON blocks
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)

        if not json_blocks:
            rewards.append(0.0)  # No tool call — neutral
            continue

        valid = False
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "tool" in parsed:
                    valid = True
                    break
            except json.JSONDecodeError:
                pass

        rewards.append(2.0 if valid else -1.0)
    return rewards


# ── Reward 4: Response Quality ───────────────────────────────────────
def quality_reward(completions, **kwargs) -> list[float]:
    """
    Reward for appropriate response length and substance.
    Too short → bad (lazy response)
    Too long → bad (rambling)
    Just right → good
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        length = len(text)

        if length < 20:
            rewards.append(-1.0)   # Too short — empty/lazy
        elif length < 50:
            rewards.append(-0.5)   # Still too short
        elif length > 2000:
            rewards.append(-0.5)   # Too verbose
        elif length > 1000:
            rewards.append(0.5)    # Detailed but maybe too long
        else:
            rewards.append(1.0)    # Good length
    return rewards


# ── Reward 5: Correctness (answer match) ─────────────────────────────
def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward for matching the expected answer (when available).
    Uses proximity — partial matches get partial reward.
    +3.0 for exact match
    +1.0 for partial match (answer is contained in response)
    -1.0 for wrong answer

    Note: 'answer' is a list (one per sample in the batch), pulled from
    the dataset's 'answer' column automatically by TRL.
    """
    rewards = []
    for completion, expected in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        extracted = extract_arabic_answer(text) or text

        # Normalize both for comparison
        norm_expected = str(expected).strip().lower()
        norm_got = extracted.strip().lower()

        if norm_expected == norm_got:
            rewards.append(3.0)
        elif norm_expected in norm_got or norm_got in norm_expected:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


# ══════════════════════════════════════════════════════════════════════════
#  DATASET — Arabic prompts for GRPO
# ══════════════════════════════════════════════════════════════════════════

def build_grpo_dataset():
    """
    Build a dataset of Arabic prompts for GRPO training.
    We are using a translated Arabic-GSM8K dataset to ensure we have >500 complex
    math reasoning trajectories to properly train the RL loop.
    """
    from datasets import load_dataset
    
    print("  Loading translated Arabic GSM8K for GRPO...")
    ds = load_dataset("Omartificial-Intelligence-Space/Arabic-gsm8k-v2", split="main_train")

    # Filter out empty answers or weird parsed artifacts
    ds = ds.filter(lambda x: x.get("question") and x.get("answer"))

    # Convert to standard GRPO dict format:
    # Must have 'prompt' column (list of message dicts) + optional extra columns for reward kwargs.
    # Must NOT have 'messages' column — TRL rejects datasets with both 'prompt' and 'messages'.
    def format_grpo(row):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            "answer": row["answer"]
        }

    # Remove all source columns, keep only 'prompt' and 'answer'
    dataset = ds.map(format_grpo, remove_columns=ds.column_names)
    
    # GSM8K is 7K+ rows, which is perfect for GRPO
    print(f"  GRPO dataset prepared with {len(dataset)} reasoning samples.")
    return dataset


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Arabic GRPO for Gemma 4 E4B")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--sft-adapter", type=str, default=SFT_ADAPTER_PATH,
                        help="Path to SFT adapter to start from")
    args = parser.parse_args()

    # ── 1. Load SFT Model ──────────────────────────────────────────────
    print("=== Loading Model (from SFT adapter) ===")
    import unsloth
    from unsloth import FastModel

    # Try loading SFT adapter first, fall back to base model
    model_name = args.sft_adapter
    if not os.path.isdir(model_name):
        print(f"  SFT adapter '{model_name}' not found, using base model")
        model_name = BASE_MODEL

    model, tokenizer = FastModel.from_pretrained(
        model_name     = model_name,
        dtype          = DTYPE,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = LOAD_IN_4BIT,
        full_finetuning = False,
        fast_inference  = False,    # Disabled because vLLM is not installed on this ROCm environment
        token          = HF_TOKEN or None,
    )

    # ── 2. LoRA for GRPO ───────────────────────────────────────────────
    if model_name == BASE_MODEL:
        print("=== Adding new LoRA Adapters for GRPO ===")
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers     = False,
            finetune_language_layers   = True,
            finetune_attention_modules = True,
            finetune_mlp_modules       = True,
            r               = 16,
            lora_alpha       = 16,
            lora_dropout     = 0,
            bias             = "none",
            random_state     = 3407,
            use_gradient_checkpointing = "unsloth",
        )
    else:
        print(f"=== Successfully loaded existing adapters from {model_name}. Skipping get_peft_model. ===")
        # CRITICAL: Since get_peft_model is skipped, gradient checkpointing was off!
        # This causes massive 14GB VRAM spikes during the backward pass.
        model.gradient_checkpointing_enable()

    # ── 3. Chat Template ───────────────────────────────────────────────
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-4",
    )

    # ── 4. Build Dataset ───────────────────────────────────────────────
    print("=== Building GRPO Dataset ===")
    dataset = build_grpo_dataset()

    # ── 5. Setup GRPO Trainer ──────────────────────────────────────────
    print("=== Setting up GRPO Trainer ===")
    from trl import GRPOTrainer, GRPOConfig

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grpo_config = GRPOConfig(
        # Gemma 4 not in vLLM yet — use native generation
        use_vllm              = False,

        # Generation
        num_generations       = NUM_GENERATIONS,
        max_prompt_length     = 256,   # Critical to stop TRL from padding to 512/1024
        max_completion_length = MAX_COMPLETION_LENGTH,

        # Training
        learning_rate         = LEARNING_RATE,
        max_steps             = args.max_steps,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps          = 20,
        optim                 = "adamw_8bit",
        weight_decay          = 0.01,
        lr_scheduler_type     = "cosine",
        seed                  = 3407,
        report_to             = "none",
        output_dir            = OUTPUT_DIR,
        save_strategy         = "steps",
        save_steps            = 100,
        save_total_limit      = 3,
        bf16                  = True,
        logging_steps         = 1,

        # GRPO specific
        loss_type             = "gspo",
        epsilon_high          = 0.28,
    )

    trainer = GRPOTrainer(
        model     = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        reward_funcs = [
            arabic_language_reward,     # +2 for Arabic responses
            format_reward,              # +1.5 for proper structure
            tool_call_reward,           # +2 for valid tool calls
            quality_reward,             # +1 for good length
            correctness_reward,         # +3 for correct answers
        ],
        args = grpo_config,
    )

    # ── 6. Train ───────────────────────────────────────────────────────
    print(f"\n=== Starting GRPO Training ({args.max_steps} steps) ===")
    print(f"  {NUM_GENERATIONS} generations per prompt")
    print(f"  5 reward functions")
    print(f"  Expect rewards to increase after ~300 steps")

    gpu_stats = torch.cuda.get_device_properties(0)
    max_mem = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"  GPU: {gpu_stats.name} | Max: {max_mem} GB")

    trainer_stats = trainer.train()

    # ── 7. Stats ───────────────────────────────────────────────────────
    used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print(f"\n=== GRPO Complete ===")
    print(f"Peak VRAM: {used} GB ({used/max_mem*100:.1f}%)")

    # ── 8. Save ────────────────────────────────────────────────────────
    print("=== Saving GRPO LoRA ===")
    model.save_pretrained("gemma4_e4b_arabic_grpo_lora")
    tokenizer.save_pretrained("gemma4_e4b_arabic_grpo_lora")

    if HF_TOKEN:
        import huggingface_hub
        huggingface_hub.login(token=HF_TOKEN)
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"Pushed to {HF_REPO_ID}")

    # ── 9. Test ────────────────────────────────────────────────────────
    print("\n=== Arabic GRPO Inference Test ===")
    messages = [
        {"role": "user", "content": (
            "لديك أداة search(query) للبحث في الإنترنت.\n"
            "ابحث عن أحدث تطورات الذكاء الاصطناعي في العالم العربي."
        )}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to("cuda")

    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs, streamer=streamer, max_new_tokens=512,
        temperature=0.7, top_p=0.9,
    )

    print("\n=== Done ===")
    print("Full pipeline complete: (CPT →) SFT → GRPO → Ready for deployment!")


if __name__ == "__main__":
    main()
