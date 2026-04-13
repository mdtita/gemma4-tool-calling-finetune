import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Unsloth must be imported before torch, trl, transformers to ensure compiler optimizations apply
import unsloth
from unsloth import FastLanguageModel
import torch
import gc
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Configuration Toggles
EXPORT_TO_GGUF = False # Change to True to automatically export Q4_K_M GGUF format upon completion

# ── ROCm compatibility patch for Qwen 3.5 DeltaNet ──────────────────────
import importlib
def _patch_fla_availability():
    try:
        import fla
        import transformers.utils.import_utils as _iu
        _iu.is_flash_linear_attention_available = lambda: True
        _iu.is_flash_linear_attention_available.cache_clear = lambda: None
        import transformers.models.qwen3_5.modeling_qwen3_5 as _m
        importlib.reload(_m)
        print("✅ Patched FLA availability for ROCm")
    except ImportError:
        pass
_patch_fla_availability()

def main():
    print("=== Loading Qwen 3.5 9B Model ===")
    max_seq_length = 2048 # SFT length
    
    # Dual-GPU strategy: GPU 0 is the workhorse, GPU 1 is the display GPU.
    # GPU 1 runs gnome-shell/Xwayland which eats ~3-4 GB. If training pushes
    # GPU 1 to capacity, the compositor can't allocate display buffers and
    # the entire system hard-freezes (amdgpu OOM).
    #
    # Final VRAM Mathematics Calibration: 
    # Qwen 3.5's massive vocabulary size (151,936) causes a huge VRAM explosion ~4GB 
    # when calculating cross-entropy loss. Because of Pipeline Parallelism, this loss 
    # ONLY occurs on the GPU hosting the final layer (`lm_head`).
    # Thus: We heavily load GPU0 with weights (14 GB), leaving it an exact 2GB buffer for forward pass memory.
    # The remaining weights (~4 GB) spill onto GPU1. GPU1 is left with a gargantuan 12 GB
    # buffer specifically to swallow the `lm_head` float32 loss spike seamlessly!
    max_memory = {0: "14GiB", 1: "15GiB"}
    
    # Unsloth Warning: Qwen 3.5 natively struggles with 4-bit quantization differences.
    # Because we have dual 16GB GPUs (32GB total), we safely use Native bf16.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3.5-9B",
        max_seq_length = max_seq_length,
        load_in_4bit = False,      # Disabled! Qwen3.5 doesn't like 4-bit
        load_in_16bit = True,      # Native bf16
        fast_inference = False,
        device_map = "sequential", # Sequentially overflow weight tail to GPU1 
        max_memory = max_memory,
    )
    
    # Execute explicit garbage collection to purge transient caching memory pre-training
    gc.collect()
    torch.cuda.empty_cache()
    
    print("=== Adding LoRA Adapters ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",
                          "out_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Crucial Unsloth memory optimization
        random_state = 3407,
        max_seq_length = max_seq_length,
    )
    
    print("=== Loading Curated Arabic Agentic Dataset ===")
    # Using the universally standardized <think> and explicitly curated trajectories
    dataset = load_dataset('json', data_files='qwen_unified_arabic_agent.jsonl', split='train')
    
    # Qwen 3.5 tokenizer already includes the correct chat template.
    # No get_chat_template() call needed — just use tokenizer.apply_chat_template() directly.
    # Ref: https://github.com/unslothai/notebooks/blob/main/nb/Qwen_3_5_27B_A100(80GB).ipynb

    def formatting_prompts_func(examples):
        texts = []
        for msgs in examples["messages"]:
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        return {"text": texts}
    
    dataset = dataset.map(
        formatting_prompts_func, 
        batched=True, 
        remove_columns=dataset.column_names,
        num_proc=16
    )
    
    print("=== Configuring SFT Trainer ===")
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 16,
            packing = True,            # 3-5x Speedup via Unsloth context stuffing
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_steps = 20,
            learning_rate = 2e-5,
            num_train_epochs = 1,
            logging_steps = 5,
            optim = "adamw_8bit",       # Standard 8-bit optimizer for bf16 LoRA
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs_qwen35_9B_arabic_sft",
            save_strategy = "steps",
            save_steps = 200,
            save_total_limit = 3,
            report_to = "none",
            bf16 = True,
        ),
    )
    
    print("=== Starting Qwen 9B SFT ===")
    
    # ── Masking: Train on Responses Only ──
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )
    
    checkpoints = [d for d in os.listdir("outputs_qwen35_9B_arabic_sft") if d.startswith("checkpoint-")] if os.path.exists("outputs_qwen35_9B_arabic_sft") else []
    if checkpoints:
        print(f"  Found {len(checkpoints)} checkpoints. Safely resuming from latest...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    print("=== Saving SFT LoRA ===")
    model.save_pretrained("qwen35_9b_arabic_lora")
    tokenizer.save_pretrained("qwen35_9b_arabic_lora")
    
    if EXPORT_TO_GGUF:
        print("=== Exporting to GGUF format ===")
        model.save_pretrained_gguf("qwen35_9b_arabic_sft_gguf", tokenizer, quantization_method="q4_k_m")
        print("GGUF Export Complete!")
        
    print("SFT complete! Ready for GRPO.")

if __name__ == "__main__":
    main()
