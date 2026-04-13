import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Configuration Toggles
EXPORT_TO_GGUF = False # Change to True to automatically export Q4_K_M GGUF format upon completion

# This script is designed for Cloud GPU environments (Runpod, Kaggle, Colab Pro)
# WARNING: Qwen 3.5 documentation absolutely forbids 4-bit QLoRA due to severe quantization loss.
# As a result, training Qwen3.5-32B natively in bf16 requires ~64GB+ VRAM.
# You MUST provision an A100 80GB or dual A6000/H100 instance for this cloud script.
def main():
    print("=== Loading Qwen 3.5 32B Cloud Pipeline ===")
    
    # 1. Model Initialization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3.5-32B",
        max_seq_length = 2048,
        load_in_4bit = False,  # Qwen 3.5 degrades fatally in 4-bit
        load_in_16bit = True,  # Native bf16 required
        fast_inference = False,
        device_map = "auto",
    )
    
    # 2. LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32, # Higher rank for 32B capacity
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",
                          "out_proj",],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = 2048,
    )
    
    # 3. Download and prepare the curated dataset (Cloud instances start fresh)
    # Ensure qwen_unified_arabic_agent.jsonl has been uploaded to the cloud instance!
    if not os.path.exists("qwen_unified_arabic_agent.jsonl"):
        print("CRITICAL: Please upload 'qwen_unified_arabic_agent.jsonl' to this cloud directory before running!")
        return

    dataset = load_dataset('json', data_files='qwen_unified_arabic_agent.jsonl', split='train')
    
    # Qwen 3.5 tokenizer already includes the correct chat template.
    # No get_chat_template() call needed.
    # Ref: https://github.com/unslothai/notebooks/blob/main/nb/Qwen_3_5_27B_A100(80GB).ipynb

    def formatting_prompts_func(examples):
        texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in examples["messages"]]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
    
    # 4. Standardized Pre-training setup
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_seq_length = 2048,
            dataset_num_proc = 4,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 16, # Stabilize 32B model gradients
            warmup_steps = 50,
            learning_rate = 1e-5,
            num_train_epochs = 1,
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            bf16 = True,
            output_dir = "outputs_qwen35_32B_cloud",
            save_strategy = "steps",
            save_steps = 200,
            save_total_limit = 3,
            report_to = "none",
        ),
    )
    
    # ── Masking: Train on Responses Only ──
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )
    
    # Cloud resilience - auto-resume if container restarts
    checkpoints = [d for d in os.listdir("outputs_qwen35_32B_cloud") if d.startswith("checkpoint-")] if os.path.exists("outputs_qwen35_32B_cloud") else []
    if checkpoints:
        print(f"Resuming cloud training from {len(checkpoints)} checkpoints...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    print("=== Training Complete. Saving Cloud Checkpoints ===")
    model.save_pretrained("qwen35_32b_arabic_lora")
    tokenizer.save_pretrained("qwen35_32b_arabic_lora")
    
    if EXPORT_TO_GGUF:
        print("Exporting Q4 GGUF for download...")
        model.save_pretrained_gguf("qwen35_32b_arabic_gguf", tokenizer, quantization_method="q4_k_m")
        print("GGUF Export Complete!")
        
    print("Cloud Pipeline Fully Completed!")

if __name__ == "__main__":
    main()
