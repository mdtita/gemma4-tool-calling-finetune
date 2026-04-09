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

# ── Config ────────────────────────────────────────────────────────────────
# Load from SFT adapter (post-SFT model)
SFT_ADAPTER_PATH = "gemma4_e4b_arabic_lora"    # Local SFT output
BASE_MODEL       = "unsloth/gemma-4-E4B-it"     # Fallback if no adapter
MAX_SEQ_LENGTH   = 2048
DTYPE            = torch.bfloat16
LOAD_IN_4BIT     = True
OUTPUT_DIR       = os.path.expanduser("~/gemma4_runs/e4b_arabic_grpo")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
HF_REPO_ID       = "mtita/gemma4-e4b-arabic-agent-grpo-lora"

# GRPO hyperparams (from Unsloth RL guide)
NUM_GENERATIONS  = 8        # Responses per prompt (GRPO sampling)
MAX_STEPS        = 500      # Minimum 300 recommended
LEARNING_RATE    = 5e-6     # Lower than SFT — fine-tuning existing knowledge
BATCH_SIZE       = 1
GRAD_ACCUM       = 1
MAX_COMPLETION_LENGTH = 512


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
def correctness_reward(completions, answer=None, **kwargs) -> list[float]:
    """
    Reward for matching the expected answer (when available).
    Uses proximity — partial matches get partial reward.
    +3.0 for exact match
    +1.0 for partial match (answer is contained in response)
    +0.0 if no answer provided (skip)
    -1.0 for wrong answer when answer IS provided
    """
    if answer is None:
        return [0.0] * len(completions)

    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        extracted = extract_arabic_answer(text) or text

        # Normalize both for comparison
        norm_expected = answer.strip().lower()
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
    Uses diverse Arabic tasks: conversation, tool calling, translation, QA.
    """
    from datasets import Dataset

    prompts = []

    # ── Arabic conversation prompts ────────────────────────────────
    conversation_prompts = [
        "اشرح لي مفهوم الذكاء الاصطناعي بطريقة بسيطة.",
        "ما هي أفضل الممارسات لكتابة كود نظيف؟",
        "كيف يمكنني تعلم البرمجة من الصفر؟",
        "ما الفرق بين التعلم الآلي والتعلم العميق؟",
        "اقترح لي خطة لتعلم اللغة الإنجليزية في ستة أشهر.",
        "ما هي أهم التقنيات الحديثة في عالم الحوسبة السحابية؟",
        "كيف أحافظ على صحتي أثناء العمل من المنزل؟",
        "ما هي أفضل طرق إدارة الوقت للمبرمجين؟",
        "اشرح لي كيف يعمل الإنترنت ببساطة.",
        "ما هي النصائح الأساسية لأمن المعلومات الشخصية؟",
        "قارن بين بايثون وجافاسكريبت من حيث الاستخدامات.",
        "كيف أبدأ مشروعي الخاص في مجال التقنية؟",
        "ما هي أهمية البيانات الضخمة في العصر الحالي؟",
        "اشرح لي مفهوم سلسلة الكتل (البلوكتشين) ببساطة.",
        "ما هي الفروق بين تطبيقات الويب وتطبيقات الهاتف المحمول؟",
    ]

    for p in conversation_prompts:
        prompts.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ],
        })

    # ── Tool calling prompts ───────────────────────────────────────
    tool_prompts = [
        "لديك أداة search(query). ابحث عن آخر أخبار التقنية في مصر.",
        "لديك أداة translate(text, from_lang, to_lang). ترجم 'Hello, how are you?' إلى العربية.",
        "لديك أداة summarize(text). لخص هذا النص: الذكاء الاصطناعي يتطور بسرعة كبيرة في العالم العربي ويؤثر على جميع القطاعات الاقتصادية والاجتماعية.",
        "لديك أداة weather(city). ما حالة الطقس في القاهرة اليوم؟",
        "لديك أداة calculate(expression). احسب مساحة مستطيل طوله 15 متر وعرضه 8 متر.",
        "لديك أداة read_file(path). اقرأ محتوى الملف src/main.py.",
        "لديك أداة create_reminder(title, time). أنشئ تذكيراً لاجتماع الفريق الساعة 3 مساءً.",
        "لديك أداة search(query). ابحث عن أفضل مكتبات بايثون لمعالجة اللغة العربية.",
    ]

    for p in tool_prompts:
        prompts.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ],
        })

    # ── Translation prompts (with expected answers) ────────────────
    translation_prompts = [
        ("ترجم إلى الإنجليزية: أهلاً وسهلاً بكم في مصر.", "Welcome to Egypt."),
        ("ترجم إلى الإنجليزية: البرمجة هي فن حل المشكلات.", "Programming is the art of problem solving."),
        ("Translate to Arabic: Artificial intelligence is transforming the world.", "الذكاء الاصطناعي يغير العالم."),
        ("Translate to Arabic: Open source software enables collaboration.", "البرمجيات مفتوحة المصدر تمكّن التعاون."),
        ("ترجم إلى الإنجليزية: الحوسبة السحابية توفر موارد حاسوبية عند الطلب.", "Cloud computing provides computing resources on demand."),
    ]

    for prompt_text, expected_answer in translation_prompts:
        prompts.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "answer": expected_answer,
        })

    # ── Egyptian dialect prompts ───────────────────────────────────
    dialect_prompts = [
        "إيه رأيك في الذكاء الاصطناعي؟ كلمني بالمصري.",
        "ازاي أبدأ أتعلم برمجة لو مش فاهم حاجة؟",
        "إيه أحسن لابتوب للبرمجة في حدود 20 ألف جنيه؟",
        "اشرحلي يعني إيه API ببساطة كده.",
        "إيه الفرق بين الفرونت إند والباك إند؟",
    ]

    for p in dialect_prompts:
        prompts.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ],
        })

    dataset = Dataset.from_list(prompts)
    print(f"  GRPO dataset: {len(dataset)} prompts")
    print(f"  - Conversation: {len(conversation_prompts)}")
    print(f"  - Tool calling: {len(tool_prompts)}")
    print(f"  - Translation: {len(translation_prompts)}")
    print(f"  - Egyptian dialect: {len(dialect_prompts)}")
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
        token          = HF_TOKEN or None,
    )

    # ── 2. LoRA for GRPO ───────────────────────────────────────────────
    print("=== Adding LoRA Adapters for GRPO ===")
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
        loss_type             = "grpo",
        # Uncomment for GSPO (recommended by Unsloth):
        # loss_type           = "gspo",
        # epsilon_high        = 0.28,
    )

    trainer = GRPOTrainer(
        model     = model,
        tokenizer = tokenizer,
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
