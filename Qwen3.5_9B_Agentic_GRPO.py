import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

# Simple custom reward functions for our agentic training
def tool_format_reward(completions, **kwargs):
    """Reward for generating valid tool calls or attempting to do so."""
    rewards = []
    for comp in completions:
        comp_str = comp if isinstance(comp, str) else (comp[0]["content"] if isinstance(comp, list) and len(comp) > 0 else "")
        if "<tool_call>" in comp_str and "</tool_call>" in comp_str:
            # Check if valid JSON inside
            import json
            import re
            match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', comp_str, re.DOTALL)
            if match:
                try:
                    json.loads(match.group(1))
                    rewards.append(2.0)
                except:
                    rewards.append(0.5) # Attempted but invalid json
            else:
                rewards.append(0.5)
        elif "tool_call" in comp_str:
            rewards.append(0.2)
        else:
            rewards.append(-0.5) # No tool calls when there probably should be
    return rewards

def reasoning_structure_reward(completions, **kwargs):
    """Reward for reasoning before acting."""
    rewards = []
    for comp in completions:
        comp_str = comp if isinstance(comp, str) else (comp[0]["content"] if isinstance(comp, list) and len(comp) > 0 else "")
        if "<think>" in comp_str and "</think>" in comp_str:
            # Reward more for longer thoughts
            import re
            match = re.search(r'<think>(.*?)</think>', comp_str, re.DOTALL)
            if match and len(match.group(1).strip()) > 20:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def main():
    print("=== Loading SFT Checkpoint for GRPO ===")
    max_seq_length = 4096
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check if the SFT output directory exists.
    # In a real run, this would be the actual checkpoint folder.
    # For dry-run, we just load the base model and apply PEFT.
    
    checkpoint_dir = "qwen35_9b_agentic_sft/checkpoint-best"
    
    # For dry run if it doesn't exist, we load base
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: {checkpoint_dir} not found. Loading base Qwen3.5-9B for dry-run.")
        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3.5-9B",
        max_seq_length = max_seq_length,
        fast_inference = False,    # Required for Qwen3.5 (no vLLM yet for this)
        load_in_16bit = True,
        load_in_4bit = False,
        device_map = "balanced",   # Dual 16GB GPUs
    )
    
    print("=== Adding LoRA Adapters ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = max_seq_length,
    )
    
    # If checkpoint existed, we would load_lora instead of creating new peft.
    # We'll just continue for the dry run.
    
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen3.5",
        mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant", "system": "system"}
    )
    
    print("=== Loading Dataset ===")
    # Load same dataset
    dataset = load_dataset('json', data_files='qwen_unified_arabic_agent.jsonl', split='train')
    
    # GRPO requires only standard queries, we don't need formatting prompts func
    # Usually you format just the prompt for GRPO to generate the response
    
    def extract_prompt(examples):
        # We take all messages up to the last user message to form the prompt
        prompts = []
        for msgs in examples["messages"]:
            # Find last user message
            last_user_idx = -1
            for i, m in enumerate(msgs):
                if m.get("role") == "user":
                    last_user_idx = i
            if last_user_idx != -1:
                prompt_msgs = msgs[:last_user_idx+1]
                prompts.append(prompt_msgs)
            else:
                prompts.append(msgs)
        
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True) for convo in prompts]
        return {"prompt": texts}
        
    grpo_dataset = dataset.map(extract_prompt, batched=True)
    
    print("=== Setting up GRPO Trainer ===")
    trainer = GRPOTrainer(
        model = model,
        reward_funcs = [tool_format_reward, reasoning_structure_reward],
        train_dataset = grpo_dataset,
        args = GRPOConfig(
            learning_rate = 5e-6,
            num_generations = 4,   # 4 generations to learn from
            max_completion_length = 2048,
            max_prompt_length = 1024,
            max_steps = 500,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 2,
            save_steps = 50,
            save_total_limit = 5,
            save_strategy = "steps",
            logging_steps = 5,
            output_dir = "qwen35_9b_agentic_grpo",
            loss_type = "grpo",
            beta = 0.0,
            max_grad_norm = 1.0,
            mask_truncated_completions = True,
            report_to = "none",
        ),
    )
    
    print("=== Starting Dry Run Check ===")
    print("Environment OK, model loaded, dataset ready.")
    print("If you want to train, run: trainer.train(resume_from_checkpoint=True)")
    
    print("Dry run successful!")

if __name__ == "__main__":
    main()
