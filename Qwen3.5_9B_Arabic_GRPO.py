import os
# Memory & Compilation optimizations MUST be set BEFORE importing torch
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES here — we need both GPUs for bf16 LoRA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import gc
from datasets import load_dataset
import unsloth
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Configuration Toggles
EXPORT_TO_GGUF = False # Change to True to automatically export Q4_K_M GGUF format upon completion

# ══════════════════════════════════════════════════════════════════════════
#  REWARDS (Continuous / Tie-breaking for NUM_GENERATIONS=2)
# ══════════════════════════════════════════════════════════════════════════

def tool_format_reward(completions, **kwargs):
    """Reward for generating universally valid tool calls."""
    rewards = []
    for comp in completions:
        comp_str = comp if isinstance(comp, str) else (comp[0]["content"] if isinstance(comp, list) and len(comp) > 0 else str(comp))
        
        # Continuous tie-breaker based on output length
        length_bonus = min((len(comp_str) / 2000.0) * 0.1, 0.1)
        
        if "<tool_call>" in comp_str and "</tool_call>" in comp_str:
            import json
            import re
            match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', comp_str, re.DOTALL)
            if match:
                try:
                    json.loads(match.group(1))
                    rewards.append(2.0 + length_bonus)
                except:
                    rewards.append(0.5 + length_bonus) # Attempted but invalid json
            else:
                rewards.append(0.5 + length_bonus)
        elif "tool_call" in comp_str:
            rewards.append(0.2 + length_bonus)
        else:
            # If no tool is generated, mildly penalize but allow length bonus
            rewards.append(-0.5 + length_bonus)
    return rewards

def reasoning_structure_reward(completions, **kwargs):
    """Reward for utilizing the universally compatible <think> structure correctly."""
    rewards = []
    for comp in completions:
        comp_str = comp if isinstance(comp, str) else (comp[0]["content"] if isinstance(comp, list) and len(comp) > 0 else str(comp))
        
        # Tie breaker based on reasoning length
        length_bonus = min((len(comp_str) / 2000.0) * 0.1, 0.1)
        
        if "<think>" in comp_str and "</think>" in comp_str:
            import re
            match = re.search(r'<think>(.*?)</think>', comp_str, re.DOTALL)
            if match and len(match.group(1).strip()) > 20:
                rewards.append(1.0 + length_bonus)
            else:
                rewards.append(0.5 + length_bonus)
        else:
            rewards.append(0.0 + length_bonus)
    return rewards


def main():
    print("=== Loading Qwen 3.5 9B SFT Model ===")
    max_seq_length = 2048
    
    # Intelligently load the best SFT weights
    model_name = "qwen35_9b_arabic_lora"
    if not os.path.isdir(model_name):
        sft_output_dir = "outputs_qwen35_9B_arabic_sft"
        if os.path.exists(sft_output_dir):
            checkpoints = [d for d in os.listdir(sft_output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                latest_ckpt = os.path.join(sft_output_dir, checkpoints[-1])
                print(f"Final SFT adapter not found. Auto-detecting latest SFT checkpoint: {latest_ckpt}")
                model_name = latest_ckpt
            else:
                print(f"SFT adapter '{model_name}' not found, falling back to base model.")
                model_name = "unsloth/Qwen3.5-9B"
        else:
            print(f"SFT adapter '{model_name}' not found, falling back to base model.")
            model_name = "unsloth/Qwen3.5-9B"

    # Qwen 3.5 Documentation forbids 4-bit due to quant losses.
    # We heavily load GPU0 with weights (14 GB), leaving it an exact buffer for forward passes.
    # The remaining weights (~4 GB) spill onto GPU1, which calculates the massively bloated vocab-loss calculations.
    max_memory = {0: "14GiB", 1: "15GiB"}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = False,
        load_in_16bit = True,
        fast_inference = False, # Mandatory for GRPO unsloth without vLLM
        device_map = "sequential",
        max_memory = max_memory,
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if model_name != "qwen35_9b_arabic_lora":
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",
                              "out_proj",],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            max_seq_length = max_seq_length,
        )

    # Use pure conversational logic for GRPO dataset extraction
    dataset = load_dataset('json', data_files='qwen_arabic_curated.jsonl', split='train')
    
    def extract_prompt(examples):
        prompts = []
        for msgs in examples["messages"]:
            last_user_idx = -1
            for i, m in enumerate(msgs):
                if m.get("role") == "user": last_user_idx = i
            prompt_msgs = msgs[:last_user_idx+1] if last_user_idx != -1 else msgs
            prompts.append(prompt_msgs)
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True) for convo in prompts]
        return {"prompt": texts}
        
    grpo_dataset = dataset.map(extract_prompt, batched=True, num_proc=16)
    
    output_dir = "outputs_qwen35_9B_arabic_grpo"
    os.makedirs(output_dir, exist_ok=True)
    
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer, # Required for TRL >= 0.15
        reward_funcs = [tool_format_reward, reasoning_structure_reward],
        train_dataset = grpo_dataset,
        args = GRPOConfig(
            learning_rate = 5e-6,
            num_generations = 2,   # Essential for 16GB restrictions
            max_completion_length = 512, # Enough for detailed thinking
            max_prompt_length = 512,
            num_train_epochs = 1, # Full sweep of the dataset
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            save_steps = 100,
            save_total_limit = 3,
            save_strategy = "steps",
            logging_steps = 5,
            output_dir = output_dir,
            optim = "adamw_8bit",       # Standard 8-bit optimizer for bf16 LoRA
            loss_type = "grpo",
            beta = 0.0, # Math stability
            max_grad_norm = 1.0,
            mask_truncated_completions = True,
            bf16 = True, # Critical! Defends against fp32 explosions!
            report_to = "none",
        ),
    )
    
    # ── CRITICAL FIX: Vocab-chunked logit computation ─────────────────
    # Must be applied AFTER GRPOTrainer is constructed, which triggers
    # Unsloth to load its compiled cache into sys.modules.
    # Root cause: Qwen 3.5 has a 151,936 vocab size. Matmul causes ~1GB transpose on ROCm.
    # Fix: Split VOCAB dimension into slices and use online logsumexp.
    import sys as _sys
    _mod = _sys.modules.get('UnslothGRPOTrainer')
    if _mod and hasattr(_mod, 'chunked_hidden_states_selective_log_softmax'):
        _VOCAB_CHUNK = 8192
        def _vocab_chunked_log_softmax(hidden_states, lm_head, index, chunks=4,
                                        logit_scale_multiply=0.0, logit_scale_divide=0.0,
                                        logit_softcapping=0.0, temperature=1.0):
            flat_h = hidden_states.reshape(-1, hidden_states.shape[-1])
            flat_idx = index.reshape(-1)
            T = flat_h.shape[0]
            V = lm_head.shape[0]
            flat_h_cast = flat_h.to(lm_head.dtype)

            selected_logits = torch.zeros(T, device=flat_h.device, dtype=torch.float32)
            running_max = torch.full((T,), -1e30, device=flat_h.device, dtype=torch.float32)
            running_sumexp = torch.zeros(T, device=flat_h.device, dtype=torch.float32)

            for v_start in range(0, V, _VOCAB_CHUNK):
                v_end = min(v_start + _VOCAB_CHUNK, V)
                w_slice = lm_head[v_start:v_end]
                logits_slice = flat_h_cast @ w_slice.t()

                if logit_scale_multiply != 0.0: logits_slice = logits_slice * logit_scale_multiply
                if logit_scale_divide != 0.0: logits_slice = logits_slice / logit_scale_divide
                if logit_softcapping != 0.0: logits_slice = logits_slice * torch.tanh(logits_slice / logit_softcapping)
                
                logits_slice = logits_slice.to(torch.float32)
                if temperature != 1.0: logits_slice = logits_slice / temperature

                mask = (flat_idx >= v_start) & (flat_idx < v_end)
                if mask.any():
                    local_idx = flat_idx[mask] - v_start
                    rows = mask.nonzero(as_tuple=True)[0]
                    selected_logits[rows] = logits_slice[rows, local_idx]

                chunk_max = logits_slice.max(dim=-1).values
                new_max = torch.maximum(running_max, chunk_max)
                running_sumexp = (running_sumexp * torch.exp(running_max - new_max) +
                                  torch.exp(logits_slice - new_max.unsqueeze(-1)).sum(dim=-1))
                running_max = new_max

            log_probs = selected_logits - (torch.log(running_sumexp) + running_max)
            return log_probs.reshape(hidden_states.shape[0], hidden_states.shape[1])

        _mod.chunked_hidden_states_selective_log_softmax = _vocab_chunked_log_softmax
        print(f"✅ Patched: Qwen Vocab-chunked logit to prevent 1GB VRAM OOM.")
    else:
        print(f"⚠️ Could not patch logit function. Modules: {[n for n in _sys.modules if 'GRPO' in n]}")
    
    # Safe resuming
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")] if os.path.exists(output_dir) else []
    if checkpoints:
        print(f"  Found {len(checkpoints)} checkpoints. Safely resuming from latest...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    print("=== Training Complete ===")
    model.save_pretrained("qwen35_9b_arabic_grpo_lora")
    tokenizer.save_pretrained("qwen35_9b_arabic_grpo_lora")
    
    if EXPORT_TO_GGUF:
        print("=== Saving GGUF ===")
        model.save_pretrained_gguf("qwen35_9b_arabic_grpo_gguf", tokenizer, quantization_method="q4_k_m")
        print("Exported to Q4_K_M GGUF format!")

if __name__ == "__main__":
    main()
