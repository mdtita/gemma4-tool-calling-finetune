"""
Gemma 4 E4B Arabic Continued Pretraining (CPT)
=================================================
Phase 0: Strengthen the model's Arabic token representations BEFORE SFT.
Trains on raw Arabic text (Wikipedia, news) — no instruction pairs needed.

Run this BEFORE Gemma4_E4B_Arabic_Agent_SFT.py if the base model's Arabic
is weak (garbled text, script mixing, poor grammar).

Pipeline: CPT (this) → SFT → GRPO

Key CPT techniques (from Unsloth docs + blog):
- Add lm_head + embed_tokens to LoRA target modules
- Use 2-10x smaller learning rate for embeddings (embedding_learning_rate)
- Use UnslothTrainer + UnslothTrainingArguments (not SFTTrainer)
- Use rsLoRA for better scaling at higher ranks
- Train on ALL linear layers including gate_proj (Unsloth blog finding)
- Unsloth automatically offloads embeddings to disk to save VRAM

Usage:
  python Gemma4_E4B_Arabic_CPT.py
  python Gemma4_E4B_Arabic_CPT.py --max-samples 50000
  python Gemma4_E4B_Arabic_CPT.py --resume
"""

import os
import gc
import argparse

import torch

# Single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME       = "unsloth/gemma-4-E4B-it"
MAX_SEQ_LENGTH   = 2048
DTYPE            = torch.bfloat16
LOAD_IN_4BIT     = True
OUTPUT_DIR       = os.path.expanduser("~/gemma4_runs/e4b_arabic_cpt")
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
HF_REPO_ID       = "mtita/gemma4-e4b-arabic-cpt-lora"

# CPT hyperparams (from Unsloth blog best practices)
LORA_R           = 32          # Higher rank for CPT (more capacity for new language)
LORA_ALPHA       = 32
LEARNING_RATE    = 5e-5        # Main LR
EMBED_LR         = 5e-6        # 10x smaller for lm_head / embed_tokens
BATCH_SIZE       = 1
GRAD_ACCUM       = 4
NUM_EPOCHS       = 1           # 1 epoch over raw text is usually enough
WARMUP_STEPS     = 50
SAVE_STEPS       = 500
LOGGING_STEPS    = 10


def main():
    parser = argparse.ArgumentParser(description="Arabic CPT for Gemma 4 E4B")
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Max text samples to train on")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    # ── 1. Load Model ──────────────────────────────────────────────────
    print("=== Loading Model for CPT ===")
    import unsloth
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name     = MODEL_NAME,
        dtype          = DTYPE,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = LOAD_IN_4BIT,
        full_finetuning = False,
        token          = HF_TOKEN or None,
    )

    # ── 2. LoRA + Embeddings ───────────────────────────────────────────
    # CPT requires training embed_tokens + lm_head with a smaller LR
    # Unsloth offloads these to disk automatically to save VRAM
    print("=== Adding LoRA Adapters (with embed_tokens + lm_head) ===")
    model = FastModel.get_peft_model(
        model,
        # finetune_language_layers = True normally doesn't include lm_head,
        # so we explicitly list target modules for CPT
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head", "embed_tokens",
        ],
        r               = LORA_R,
        lora_alpha       = LORA_ALPHA,
        lora_dropout     = 0,
        bias             = "none",
        random_state     = 3407,
        use_rslora       = True,    # rsLoRA for better scaling at higher ranks
        use_gradient_checkpointing = "unsloth",
    )

    # ── 3. Load Arabic Raw Text Corpus ─────────────────────────────────
    print("=== Loading Arabic Text Corpus ===")
    from datasets import load_dataset, concatenate_datasets

    corpus_parts = []

    # Arabic Wikipedia
    print("  Loading Arabic Wikipedia...")
    try:
        wiki = load_dataset(
            "wikimedia/wikipedia", "20231101.ar",
            split="train",
            streaming=False,
        )
        # Take a subset
        wiki = wiki.select(range(min(args.max_samples, len(wiki))))
        wiki = wiki.map(lambda x: {"text": x["text"]}, remove_columns=wiki.column_names)
        corpus_parts.append(wiki)
        print(f"    → {len(wiki)} articles")
    except Exception as e:
        print(f"    ⚠️ Wikipedia failed: {e}")

    # Arabic OSCAR (web text)
    print("  Loading Arabic OSCAR (web text)...")
    try:
        oscar = load_dataset(
            "oscar-corpus/OSCAR-2301", "ar",
            split="train",
            trust_remote_code=True,
            streaming=True,
        )
        # Stream a subset
        oscar_texts = []
        for i, row in enumerate(oscar):
            text = row.get("text", "")
            if len(text) > 100:  # Skip very short texts
                oscar_texts.append({"text": text})
            if len(oscar_texts) >= min(args.max_samples // 2, 50000):
                break
        from datasets import Dataset
        oscar_ds = Dataset.from_list(oscar_texts)
        corpus_parts.append(oscar_ds)
        print(f"    → {len(oscar_ds)} texts")
    except Exception as e:
        print(f"    ⚠️ OSCAR failed: {e}")
        # Fallback: use CC-100 Arabic
        print("  Trying CC-100 Arabic fallback...")
        try:
            cc100 = load_dataset("cc100", "ar", split="train", streaming=True)
            cc100_texts = []
            for i, row in enumerate(cc100):
                text = row.get("text", "")
                if len(text) > 100:
                    cc100_texts.append({"text": text})
                if len(cc100_texts) >= min(args.max_samples // 2, 50000):
                    break
            from datasets import Dataset
            cc100_ds = Dataset.from_list(cc100_texts)
            corpus_parts.append(cc100_ds)
            print(f"    → {len(cc100_ds)} texts")
        except Exception as e2:
            print(f"    ⚠️ CC-100 also failed: {e2}")

    if not corpus_parts:
        print("❌ No Arabic text corpus available! Cannot do CPT.")
        print("   Please download Arabic text data manually.")
        return

    # Merge all corpus parts
    if len(corpus_parts) == 1:
        dataset = corpus_parts[0]
    else:
        dataset = concatenate_datasets(corpus_parts)

    print(f"\n  Total CPT corpus: {len(dataset)} texts")

    # Filter: ensure text has Arabic
    def has_arabic(text):
        return any('\u0600' <= c <= '\u06FF' for c in text[:200])

    dataset = dataset.filter(lambda x: has_arabic(x["text"]) and len(x["text"]) > 200)
    print(f"  After Arabic filter: {len(dataset)} texts")

    # ── 4. Tokenize for text completion ────────────────────────────────
    print("=== Tokenizing ===")
    EOS_TOKEN = tokenizer.eos_token

    def tokenize_func(examples):
        # Simple text completion — no instruction format
        texts = [text + EOS_TOKEN for text in examples["text"]]
        return {"text": texts}

    dataset = dataset.map(tokenize_func, batched=True, num_proc=2)
    print(f"  Tokenized: {len(dataset)} samples")

    # ── 5. Setup UnslothTrainer ────────────────────────────────────────
    # Must use UnslothTrainer + UnslothTrainingArguments for CPT
    # (supports embedding_learning_rate)
    print("=== Setting up Trainer ===")
    from unsloth import UnslothTrainer, UnslothTrainingArguments

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = UnslothTrainer(
        model     = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = UnslothTrainingArguments(
            dataset_text_field        = "text",
            max_seq_length            = MAX_SEQ_LENGTH,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            warmup_steps              = WARMUP_STEPS,
            num_train_epochs          = NUM_EPOCHS,
            learning_rate             = LEARNING_RATE,
            embedding_learning_rate    = EMBED_LR,    # 10x smaller for embeddings!
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
            packing                   = True,
        ),
    )

    # ── 6. Train ───────────────────────────────────────────────────────
    print("=== Starting CPT ===")
    gpu_stats = torch.cuda.get_device_properties(0)
    max_mem = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"GPU: {gpu_stats.name} | Max: {max_mem} GB")

    # Auto-resume
    resume_ckpt = None
    if args.resume and os.path.isdir(OUTPUT_DIR):
        checkpoints = sorted(
            [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1])
        )
        if checkpoints:
            resume_ckpt = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"  Resuming from: {resume_ckpt}")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── 7. Stats ───────────────────────────────────────────────────────
    used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print(f"\n=== CPT Complete ===")
    print(f"Runtime: {trainer_stats.metrics['train_runtime']:.0f}s")
    print(f"Peak VRAM: {used} GB ({used/max_mem*100:.1f}%)")
    print(f"Final loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")

    # ── 8. Save CPT Adapter ────────────────────────────────────────────
    print("=== Saving CPT LoRA ===")
    model.save_pretrained("gemma4_e4b_arabic_cpt_lora")
    tokenizer.save_pretrained("gemma4_e4b_arabic_cpt_lora")

    if HF_TOKEN:
        import huggingface_hub
        huggingface_hub.login(token=HF_TOKEN)
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"Pushed to {HF_REPO_ID}")

    # ── 9. Quick Arabic Generation Test ────────────────────────────────
    print("\n=== Arabic Generation Test (post-CPT) ===")
    prompt = "الذكاء الاصطناعي هو مجال من مجالات علوم الحاسوب الذي"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=200,
        temperature=0.7, top_p=0.9,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("\n=== Done ===")
    print("Next step: Run Gemma4_E4B_Arabic_Agent_SFT.py")
    print("  Load this CPT adapter by changing MODEL_NAME to 'gemma4_e4b_arabic_cpt_lora'")


if __name__ == "__main__":
    main()
