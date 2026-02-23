import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import json


def train_baseline():
    # 1. Load Model (CPU Mode)
    model_name = "Qwen/Qwen2.5-0.5B"
    print("Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Data
    dataset = load_dataset("gsm8k", "main", split="train[:20]")

    # --- STEP 3: PRE-PROCESS DATA MANUALLY ---
    # We do the formatting here, outside the trainer. It's much safer.
    def format_batch(batch):
        # This function processes a whole chunk of data at once
        texts = []
        for q, a in zip(batch["question"], batch["answer"]):
            texts.append(f"Question: {q}\nAnswer: {a}")
        return {"text": texts}  # We create a new column called "text"

    print("Formatting dataset...")
    dataset = dataset.map(format_batch, batched=True)
    # -----------------------------------------

    # 4. Setup LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # 5. Trainer Config
    sft_config = SFTConfig(
        output_dir="./models/baseline",
        max_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        use_cpu=True,
        packing=False,
        dataset_text_field="text",  # Look at the column we created above
        # max_seq_length=512,  # Problem unexpected keyword argument
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_config,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    # 8. Save
    # Save the training logs to a JSON file
    log_history = trainer.state.log_history
    with open("results/baseline_res/training_logs.json", "w") as f:
        json.dump(log_history, f, indent=4)

    print("Training logs saved to results/baseline_res/training_logs.json")
    trainer.save_model("./models/baseline_final")
    tokenizer.save_pretrained("./models/baseline_final")
    print("Training complete.")


if __name__ == "__main__":
    train_baseline()
