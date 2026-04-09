"""
Gemma 4 E4B Arabic Agentic SFT
================================
Single-GPU, ~10 GB VRAM (4-bit QLoRA) — Arabic-focused agentic model.
Based on Unsloth Gemma 4 guide: https://unsloth.ai/docs/models/gemma-4/train

Capabilities trained:
- Arabic conversation (MSA فصحى + Egyptian dialect)
- Translation (AR ↔ EN)
- Information gathering and report generation in Arabic
- Tool calling / function calling with Arabic context

Key design decisions (from Unsloth docs):
- FastModel (text-only, NOT FastVisionModel)
- packing = True for 3-5x faster training
- use_gradient_checkpointing = "unsloth" for max VRAM savings
- chat_template = "gemma-4" (non-thinking, correct for E4B)
- train_on_responses_only with Gemma 4 turn markers
- .removeprefix('<bos>') since the processor adds <bos>
- Gemma 4 has NO native 'tool' role: tool results → user messages
- Loss of 13-15 is normal for E4B — multimodal model quirk

Usage:
  python Gemma4_E4B_Arabic_Agent_SFT.py
"""

import os
import gc
import json

import torch

# Single GPU — no display GPU issues
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME        = "unsloth/gemma-4-E4B-it"
MAX_SEQ_LENGTH    = 2048         # Sufficient for Arabic tool-calling chains
DTYPE             = torch.bfloat16
LOAD_IN_4BIT      = True         # QLoRA — ~10 GB VRAM on single GPU
DATASET_FILE      = "arabic_agentic_dataset.jsonl"
OUTPUT_DIR        = os.path.expanduser("~/gemma4_runs/e4b_arabic_agent_sft")
HF_REPO_ID        = "mtita/gemma4-e4b-arabic-agent-lora"
HF_TOKEN          = os.environ.get("HF_TOKEN", "")
PUSH_TO_HUB       = bool(HF_TOKEN)

# LoRA config
LORA_R             = 16
LORA_ALPHA         = 16
LORA_DROPOUT       = 0

# Training hyperparams
BATCH_SIZE         = 1
GRAD_ACCUM         = 4           # Effective batch = 1 * 4 = 4
LEARNING_RATE      = 2e-4        # Unsloth default for QLoRA
NUM_EPOCHS         = 2           # 2 epochs for better Arabic absorption
WARMUP_STEPS       = 20
SAVE_STEPS         = 250         # Frequent saves — crash-resilient
LOGGING_STEPS      = 5
PACKING            = True        # 3-5x faster via Unsloth kernels

# Arabic system prompt (MSA + Egyptian)
ARABIC_SYSTEM_PROMPT = (
    "أنت مساعد ذكي ثنائي اللغة (عربي/إنجليزي). يمكنك:\n"
    "- إجراء محادثات باللغة العربية الفصحى والعامية المصرية\n"
    "- ترجمة النصوص بين العربية والإنجليزية\n"
    "- البحث عن المعلومات وجمعها وتلخيصها في تقارير عربية\n"
    "- استخدام الأدوات المتاحة لتنفيذ المهام\n\n"
    "عند استخدام الأدوات، اكتب استدعاء الأداة بصيغة JSON المحددة.\n"
    "أجب دائماً باللغة التي يستخدمها المستخدم ما لم يُطلب خلاف ذلك."
)


def main():
    # ── 1. Load Model ──────────────────────────────────────────────────
    print("=== Loading Model ===")
    import unsloth                    # Must be first import
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name     = MODEL_NAME,
        dtype          = DTYPE,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = LOAD_IN_4BIT,
        full_finetuning = False,
        token          = HF_TOKEN or None,
    )

    # ── 2. Add LoRA Adapters ───────────────────────────────────────────
    print("=== Adding LoRA Adapters ===")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False,   # Text-only
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r               = LORA_R,
        lora_alpha       = LORA_ALPHA,
        lora_dropout     = LORA_DROPOUT,
        bias             = "none",
        random_state     = 3407,
        use_gradient_checkpointing = "unsloth",  # Max VRAM savings
    )

    # ── 3. Chat Template ───────────────────────────────────────────────
    print("=== Setting up Chat Template ===")
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-4",   # Non-thinking template for E4B
    )

    # ── 4. Load Dataset ────────────────────────────────────────────────
    print("=== Loading Dataset ===")
    from datasets import load_dataset

    if os.path.isfile(DATASET_FILE):
        dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
        print(f"Loaded {len(dataset)} examples from {DATASET_FILE}")
    else:
        dataset = load_dataset(
            "mtita/gemma4-arabic-agent-training",
            data_files="arabic_agentic_dataset.jsonl",
            split="train",
        )
        print(f"Loaded {len(dataset)} examples from HF Hub")

    # ── 5. Format for Gemma 4 ──────────────────────────────────────────
    print("=== Formatting Dataset ===")
    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            formatted = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "") or ""
                if not content.strip():
                    continue

                # Gemma 4 roles: system, user, model (assistant → model)
                if role == "system":
                    # System messages: wrap as user instruction
                    formatted.append({"role": "user", "content": f"[System Instructions]\n{content}"})
                    formatted.append({"role": "assistant", "content": "مفهوم."})
                elif role == "user":
                    if formatted and formatted[-1]["role"] == "user":
                        formatted[-1]["content"] += f"\n\n{content}"
                    else:
                        formatted.append({"role": "user", "content": content})
                elif role == "assistant":
                    if formatted and formatted[-1]["role"] == "assistant":
                        formatted[-1]["content"] += f"\n\n{content}"
                    else:
                        formatted.append({"role": "assistant", "content": content})
                elif role in ("tool", "function"):
                    # Gemma 4 has no tool role — fold into user
                    tool_name = msg.get("name", "tool")
                    tool_text = f"[نتيجة الأداة: {tool_name}]\n{content}"
                    if formatted and formatted[-1]["role"] == "user":
                        formatted[-1]["content"] += f"\n\n{tool_text}"
                    else:
                        formatted.append({"role": "user", "content": tool_text})

            # Validate: starts with user, ends with assistant
            if not formatted or formatted[0]["role"] != "user":
                texts.append("")
                continue
            if formatted[-1]["role"] != "assistant":
                texts.append("")
                continue

            try:
                text = tokenizer.apply_chat_template(
                    formatted, tokenize=False, add_generation_prompt=False
                )
                # Remove leading <bos> — processor adds it
                if text.startswith("<bos>"):
                    text = text[5:]
                texts.append(text)
            except Exception:
                texts.append("")

        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=2)
    dataset = dataset.filter(lambda x: len(x["text"]) > 50)
    print(f"Formatted: {len(dataset)} examples")
    if len(dataset) > 0:
        print(dataset[0]["text"][:500])

    # ── 6. Setup Trainer ───────────────────────────────────────────────
    print("=== Setting up Trainer ===")
    from trl import SFTTrainer, SFTConfig

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = SFTTrainer(
        model     = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset  = None,
        args = SFTConfig(
            dataset_text_field        = "text",
            max_seq_length            = MAX_SEQ_LENGTH,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps              = WARMUP_STEPS,
            num_train_epochs          = NUM_EPOCHS,
            learning_rate             = LEARNING_RATE,
            logging_steps             = LOGGING_STEPS,
            optim                     = "adamw_8bit",
            weight_decay              = 0.01,
            lr_scheduler_type         = "cosine",
            seed                      = 3407,
            report_to                 = "none",
            output_dir                = OUTPUT_DIR,
            save_strategy             = "steps",
            save_steps                = SAVE_STEPS,
            save_total_limit          = 3,
            bf16                      = True,
            packing                   = PACKING,     # 3-5x faster training!
        ),
    )

    # ── 7. Train on Responses Only ─────────────────────────────────────
    print("=== Masking: Train on Responses Only ===")
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|turn>user\n",
        response_part    = "<|turn>model\n",
    )

    # Verify masking
    if len(trainer.train_dataset) > 0:
        sample_labels = trainer.train_dataset[0]["labels"]
        n_trained = sum(1 for x in sample_labels if x != -100)
        n_total = len(sample_labels)
        print(f"  Masking check: {n_trained}/{n_total} tokens trained ({n_trained/n_total*100:.1f}%)")

    # ── 8. Train (with auto-resume) ──────────────────────────────────
    print("=== Starting Training ===")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_mem = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"GPU: {gpu_stats.name} | Max: {max_mem} GB | Reserved: {start_mem} GB")

    # Auto-resume from last checkpoint if one exists
    resume_ckpt = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = sorted(
            [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1])
        )
        if checkpoints:
            resume_ckpt = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"  Resuming from: {resume_ckpt}")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── 9. Post-Training Stats ─────────────────────────────────────────
    used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print(f"\n=== Training Complete ===")
    print(f"Runtime: {trainer_stats.metrics['train_runtime']:.0f}s ({trainer_stats.metrics['train_runtime']/60:.1f}m)")
    print(f"Peak VRAM: {used} GB ({used/max_mem*100:.1f}% of {max_mem} GB)")

    # ── 10. Save ───────────────────────────────────────────────────────
    print("=== Saving Model ===")
    model.save_pretrained("gemma4_e4b_arabic_lora")
    tokenizer.save_pretrained("gemma4_e4b_arabic_lora")

    if PUSH_TO_HUB:
        import huggingface_hub
        huggingface_hub.login(token=HF_TOKEN)
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"Pushed to {HF_REPO_ID}")

    # ── 11. Arabic Inference Test ─────────────────────────────────────
    print("\n=== Arabic Inference Test ===")
    messages = [{"role": "user", "content": (
        "لديك الأدوات التالية:\n"
        "- search(query): البحث في الإنترنت\n"
        "- translate(text, from_lang, to_lang): ترجمة النصوص\n"
        "- summarize(text): تلخيص النصوص\n\n"
        "ابحث عن آخر أخبار التقنية في العالم العربي وقدم لي تقريراً موجزاً."
    )}]

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

    # ── 12. Translation Test ──────────────────────────────────────────
    print("\n=== Translation Test ===")
    messages_translate = [{"role": "user", "content": (
        "ترجم الجملة التالية إلى الإنجليزية:\n\n"
        "الذكاء الاصطناعي يغير طريقة عملنا وحياتنا اليومية بشكل جذري."
    )}]

    inputs_translate = tokenizer.apply_chat_template(
        messages_translate, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to("cuda")

    _ = model.generate(
        **inputs_translate, streamer=streamer, max_new_tokens=256,
        temperature=0.3, top_p=0.9,
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
