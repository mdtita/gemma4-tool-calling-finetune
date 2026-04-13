# ── Unsloth must be imported before transformers ──────────────────────────
import unsloth  # noqa: F401  — ensures all Unsloth optimizations are applied

# ── ROCm compatibility patch ──────────────────────────────────────────────
# transformers gates FLA behind is_torch_cuda_available() which is False on
# ROCm/HIP. FLA uses pure Triton kernels and works fine on AMD GPUs.
# Monkey-patch BEFORE any model import so Qwen 3.5 GatedDeltaNet layers
# pick up the optimized FLA kernels instead of the slow torch fallback.
import importlib
def _patch_fla_availability():
    try:
        import fla  # noqa: F401
        import transformers.utils.import_utils as _iu
        _orig = _iu.is_flash_linear_attention_available
        _iu.is_flash_linear_attention_available = lambda: True
        _iu.is_flash_linear_attention_available.cache_clear = lambda: None
        # Reload the Qwen 3.5 modeling module so it picks up the patched check
        import transformers.models.qwen3_5.modeling_qwen3_5 as _m
        importlib.reload(_m)
        print("✅ Patched FLA availability for ROCm — fast DeltaNet kernels enabled")
    except ImportError:
        print("⚠️  FLA not installed — using torch fallback for DeltaNet layers")
_patch_fla_availability()

from unsloth import FastLanguageModel  # unsloth already imported above
import os
import torch
import gc
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback


def main():
    print("=== Loading Model ===")
    # Unsloth docs: Qwen3.5 bf16 LoRA 9B needs ~22 GB.
    # 2×16 GB GPUs = 32 GB total. Balanced split = ~11 GB/GPU for weights,
    # leaving ~5 GB/GPU for LoRA adapters, optimizer states, and activations.
    # Seq length 2048 keeps activation memory reasonable under gradient checkpointing.
    max_seq_length = 2048
    
    # Reduce VRAM fragmentation on ROCm
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    
    # Cap each GPU — GPU 1 is the DISPLAY GPU (gnome-shell, Xwayland, browser, etc.)
    # eat ~3-4 GB of its VRAM. If training pushes GPU 1 to capacity, the compositor
    # can't allocate display buffers and the whole system hard-freezes (amdgpu OOM).
    #
    # "sequential" fills GPU 0 first (up to 15 GiB ≈ most of the 19.3 GB model),
    # then puts only the tail layers (~4 GiB) on GPU 1, leaving it with ~8-9 GB
    # free for the desktop compositor. No disk/CPU offloading needed.
    max_memory = {0: "15GiB", 1: "9GiB"}
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3.5-9B",
        max_seq_length = max_seq_length,
        # Unsloth WARNING: "It is not recommended to do QLoRA (4-bit) training
        # on the Qwen3.5 models due to higher than normal quantization differences.
        # Best to use bf16 setups." — https://unsloth.ai/docs/models/qwen3.5/fine-tune
        load_in_4bit = False,
        load_in_16bit = True,      # bf16 LoRA as recommended
        device_map = "sequential", # Fill GPU 0 first → minimal load on display GPU 1
        max_memory = max_memory,   # Prevent amdgpu driver-level OOM
    )
    
    # Flush any loading intermediates
    gc.collect()
    torch.cuda.empty_cache()
    
    print("=== Adding LoRA Adapters ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,                # Lower rank to fit bf16 VRAM budget (was 32 for QLoRA)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",
                          "out_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Crucial for 16 GB GPUs
        random_state = 3407,
        max_seq_length = max_seq_length,
    )
    
    # Reclaim any adapter-setup temporaries
    gc.collect()
    torch.cuda.empty_cache()
    
    print("=== Loading Dataset ===")
    dataset = load_dataset('json', data_files='qwen_unified_arabic_agent.jsonl', split='train')
    
    # Qwen 3.5 tokenizer already includes the correct chat template.
    # No get_chat_template() call needed — just use tokenizer.apply_chat_template() directly.
    # Ref: https://github.com/unslothai/notebooks/blob/main/nb/Qwen_3_5_27B_A100(80GB).ipynb
    
    # Normalize non-standard roles (APIGen uses function_call/observation)
    ROLE_MAP = {"function_call": "assistant", "observation": "tool", "human": "user", "gpt": "assistant"}

    def split_long_conversation(messages, max_tokens):
        """Split conversations exceeding max_tokens at turn boundaries.
        Each chunk gets the system message prepended for context.
        Returns a list of message lists, each fitting within max_tokens."""
        # Normalize roles first
        normalized = [{"role": ROLE_MAP.get(m["role"], m["role"]), "content": m.get("content", "") or ""} for m in messages]
        
        # Access underlying tokenizer (tokenizer is a Qwen3VLProcessor)
        _tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        
        # Extract system message if present
        system_msgs = []
        turn_msgs = []
        for m in normalized:
            if m["role"] == "system":
                system_msgs.append(m)
            else:
                turn_msgs.append(m)
        
        # Check if the full conversation fits
        full_text = tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)
        if len(_tok.encode(full_text, truncation=False)) <= max_tokens:
            return [normalized]
        
        # Split at turn boundaries — group user+assistant pairs
        chunks = []
        current_chunk = list(system_msgs)
        
        for msg in turn_msgs:
            current_chunk.append(msg)
            
            # When we reach an assistant response, check if chunk is too long
            if msg["role"] == "assistant":
                text = tokenizer.apply_chat_template(current_chunk, tokenize=False, add_generation_prompt=False)
                tok_len = len(_tok.encode(text, truncation=False))
                
                if tok_len > max_tokens and len(current_chunk) > len(system_msgs) + 2:
                    # Save everything except the last user+assistant pair
                    save_chunk = current_chunk[:-2]
                    if len(save_chunk) > len(system_msgs):
                        chunks.append(save_chunk)
                    # Start new chunk with system + this user+assistant pair
                    current_chunk = list(system_msgs) + current_chunk[-2:]
        
        # Don't forget the last chunk
        if len(current_chunk) > len(system_msgs):
            chunks.append(current_chunk)
        
        return chunks if chunks else [normalized]  # Fallback to original if splitting fails

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = []
        for convo in convos:
            chunks = split_long_conversation(convo, max_seq_length)
            for chunk in chunks:
                texts.append(tokenizer.apply_chat_template(chunk, tokenize=False, add_generation_prompt=False))
        return { "text" : texts, }
    
    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
    
    # ── VRAM Watchdog — prevent display-GPU OOM freezes ──────────────────
    class VRAMWatchdog(TrainerCallback):
        """Monitor both GPUs and defensively clear caches when the display GPU
        (GPU 1) approaches its VRAM ceiling.  Logs telemetry every `interval` steps."""
        DISPLAY_GPU = 1
        THRESHOLD   = 0.75           # trigger GC when >75 % of total VRAM used

        def __init__(self, interval: int = 25):
            self.interval = interval

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.interval != 0:
                return
            lines = []
            for gpu_id in range(torch.cuda.device_count()):
                used  = torch.cuda.memory_reserved(gpu_id) / 1024**3
                total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                pct   = used / total * 100
                lines.append(f"GPU{gpu_id}: {used:.1f}/{total:.1f} GB ({pct:.0f}%)")
            print(f"  [VRAM step {state.global_step}] {' | '.join(lines)}")

            # Defensive flush on display GPU
            disp_used  = torch.cuda.memory_reserved(self.DISPLAY_GPU) / 1024**3
            disp_total = torch.cuda.get_device_properties(self.DISPLAY_GPU).total_memory / 1024**3
            if disp_used / disp_total > self.THRESHOLD:
                gc.collect()
                torch.cuda.empty_cache()
                after = torch.cuda.memory_reserved(self.DISPLAY_GPU) / 1024**3
                print(f"  ⚠️  GPU{self.DISPLAY_GPU} above {self.THRESHOLD*100:.0f}% — flushed cache: {disp_used:.1f} → {after:.1f} GB")

    print("=== Setting up Trainer ===")
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False,           # Unsloth skips packing for multimodal models anyway
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 20,
            num_train_epochs = 1,      # 1 pass over 38k examples is plenty for SFT
            learning_rate = 2e-5,      # Lower LR for bf16 LoRA (was 1e-4 for QLoRA)
            lr_scheduler_type = "cosine",
            optim = "adamw_8bit",      # 8-bit optimizer to save VRAM
            logging_steps = 5,
            save_steps = 250,          # Safer checkpoint frequency for bf16 dual-GPU
            save_total_limit = 3,
            save_strategy = "steps",
            output_dir = "qwen35_9b_agentic_sft",
            report_to = "none",
            seed = 3407,
            bf16 = True,               # Explicit bf16 training
            dataloader_num_workers = 2, # Overlap data loading with GPU compute
            dataloader_pin_memory = False, # Not useful on ROCm
            # NOTE: DO NOT set gradient_checkpointing here — Unsloth's "unsloth"
            # version (line 68) is faster. HF's default would override it.
        ),
        callbacks = [VRAMWatchdog(interval=25)],
    )
    # ── Masking: Train on Responses Only ──
    print("=== Masking: Train on Responses Only ===")
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )
    
    # Resume from checkpoint if one exists (crash-resilient)
    output_dir = "qwen35_9b_agentic_sft"
    resume_ckpt = None
    if os.path.isdir(output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1])
        )
        if checkpoints:
            resume_ckpt = os.path.join(output_dir, checkpoints[-1])
            print(f"=== Resuming from checkpoint (found {len(checkpoints)}: {resume_ckpt}) ===")

    print("=== Starting Training ===")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_mem = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"GPU: {gpu_stats.name} | Max: {max_mem} GB | Reserved: {start_mem} GB")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_ckpt)

    print("=== Saving Final Model ===")
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print(f"\n=== Training Complete ===")
    print(f"Runtime: {trainer_stats.metrics['train_runtime']:.0f}s ({trainer_stats.metrics['train_runtime']/60:.1f}m)")
    print(f"Peak VRAM: {used} GB ({used/max_mem*100:.1f}% of {max_mem} GB)")

if __name__ == "__main__":
    main()
