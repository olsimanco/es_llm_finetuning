import json


def create_cell(cell_type, source_text):
    source_lines = [line + "\n" for line in source_text.split("\n")]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip("\n")

    cell = {"cell_type": cell_type, "metadata": {}, "source": source_lines}
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def build_notebook():
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": "es_finetune_project.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [],
    }

    # --- SETUP & INSTALL ---
    notebook["cells"].append(
        create_cell(
            "markdown",
            "# 🚀 ES Fine-Tuning Large Language Models\n\nWelcome to the Colab workspace for your Thesis. Let's set up the environment.",
        )
    )
    notebook["cells"].append(
        create_cell(
            "code",
            "!pip install -q transformers datasets trl peft accelerate\n!mkdir -p src results",
        )
    )

    # --- ACCURACY VALIDATOR ---
    acc_validator_code = """%%writefile src/accuracy_validator.py
import torch
from datasets import load_dataset
import re

class AccuracyValidator:
    def __init__(self, device="cpu", limit=10):
        self.device = device
        self.tokenizer = None
        print(f"Loading {limit} Validation Questions (GSM8k)...")
        self.dataset = load_dataset("gsm8k", "main", split=f"test[:{limit}]")

    def validate(self, model, soft_prompt):
        correct = 0
        total = len(self.dataset)
        soft_prompt_batch = soft_prompt.unsqueeze(0)
        print(f"  > Running Validation on {total} questions...")

        for i, item in enumerate(self.dataset):
            question = item["question"]
            target = item["answer"].split("####")[-1].strip()
            text = f"Question: {question}\\nAnswer:"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                input_embeds = model.get_input_embeddings()(inputs.input_ids)
                combined_embeds = torch.cat([soft_prompt_batch, input_embeds], dim=1)
                outputs = model.generate(
                    inputs_embeds=combined_embeds,
                    max_new_tokens=15,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if target in output_text:
                correct += 1

        score = correct / total
        print(f"  > Validation Complete. Accuracy: {score:.2%} ({correct}/{total})")
        return score
"""
    notebook["cells"].append(create_cell("code", acc_validator_code))

    # --- BPB WRAPPER ---
    bpb_wrapper_code = """%%writefile src/bpb_wrapper.py
import torch
from datasets import load_dataset
import numpy as np

class BPBScorer:
    def __init__(self, suite_config, device="cpu", limit=10):
        self.device = device
        self.tokenizer = None
        self.tasks = {}

        task_list = suite_config.get("tasks", [])
        if isinstance(task_list, dict):
            task_list = list(task_list.keys())

        print(f"Loading {len(task_list)} tasks for BPB evaluation...")

        for task_str in task_list:
            dataset_name = task_str.split(":")[0]
            print(f"  - Loading {dataset_name}...")
            try:
                if "minerva" in dataset_name:
                    ds = load_dataset("gsm8k", "main", split=f"test[:{limit}]")
                else:
                    ds = load_dataset(dataset_name, split=f"test[:{limit}]")
                self.tasks[dataset_name] = ds
            except Exception as e:
                print(f"    ! Could not load {dataset_name} ({e}). using GSM8k fallback.")
                self.tasks[dataset_name] = load_dataset("gsm8k", "main", split=f"test[:{limit}]")

    def get_score(self, model, soft_prompt):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set!")

        total_loss = 0
        total_tasks = 0

        for name, dataset in self.tasks.items():
            task_loss = 0
            count = 0

            for i in range(len(dataset)):
                q = dataset[i].get("question", dataset[i].get("problem", ""))
                a = dataset[i].get("answer", dataset[i].get("solution", ""))
                full_text = f"Question: {q}\\nAnswer: {a}"

                inputs = self.tokenizer(full_text, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)

                input_embeds = model.get_input_embeddings()(input_ids)
                soft_prompt_batch = soft_prompt.unsqueeze(0)
                combined_embeds = torch.cat([soft_prompt_batch, input_embeds], dim=1)

                ignore_labels = torch.full((1, soft_prompt.shape[0]), -100, dtype=torch.long, device=self.device)
                real_labels = input_ids.clone()
                combined_labels = torch.cat([ignore_labels, real_labels], dim=1)

                with torch.no_grad():
                    outputs = model(inputs_embeds=combined_embeds, labels=combined_labels)
                    loss = outputs.loss

                task_loss += loss.item()
                count += 1

            if count > 0:
                total_loss += task_loss / count
                total_tasks += 1

        if total_tasks == 0:
            return -99.0
        return -(total_loss / total_tasks)
"""
    notebook["cells"].append(create_cell("code", bpb_wrapper_code))

    # --- ES TRAINER ---
    # NOTE: I automatically changed cfg.device to "cuda" so Colab will use the GPU
    es_trainer_code = """%%writefile src/es_trainer_bpb.py
import torch
import numpy as np
import os
import csv
import json
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from bpb_wrapper import BPBScorer
from accuracy_validator import AccuracyValidator

TASK_SUITE_CONFIGS = {
    "modalities:base_easy:math_bpb": {
        "tasks": ["minerva_math_algebra:bpb::olmes"],
        "primary_metric": "macro",
    }
}

class ESConfig:
    run_name = "Run_Colab_GPU"
    model_name = "Qwen/Qwen2.5-0.5B"
    generations = 5
    population_size = 4
    sigma = 0.05
    learning_rate = 0.01
    validate_every = 5
    device = "cuda" # AUTOMATICALLY UPGRADED TO CUDA

class ExperimentLogger:
    def __init__(self, config):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_dir = os.path.join("results", f"{config.run_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_file = os.path.join(self.run_dir, "gen_stats.csv")
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gen", "best_loss", "avg_loss", "variance", "accuracy", "time_sec"])
        self.leaderboard_file = "results/leaderboard.csv"
        if not os.path.exists(self.leaderboard_file):
            with open(self.leaderboard_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Run_Name", "Model", "Gens", "Pop", "Sigma", "Global_Best_Loss", "Global_Avg_Loss", "Avg_Variance", "Final_Acc", "Total_Time_Min"])

    def log_gen(self, gen, best_loss, avg_loss, variance, accuracy, duration):
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, f"{best_loss:.4f}", f"{avg_loss:.4f}", f"{variance:.4f}", f"{accuracy:.2%}", f"{duration:.1f}"])

    def log_run_summary(self, config, global_best, global_avg, avg_variance, final_acc, total_time):
        with open(self.leaderboard_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([config.run_name, config.model_name, config.generations, config.population_size, config.sigma, f"{global_best:.4f}", f"{global_avg:.4f}", f"{avg_variance:.4f}", f"{final_acc:.2%}", f"{total_time/60:.2f}"])

    def save_prompt(self, prompt_tensor, filename):
        torch.save(prompt_tensor, os.path.join(self.run_dir, filename))

def main():
    cfg = ESConfig()
    logger = ExperimentLogger(cfg)
    target_suite = TASK_SUITE_CONFIGS["modalities:base_easy:math_bpb"]
    loss_scorer = BPBScorer(target_suite, device=cfg.device, limit=3)
    acc_validator = AccuracyValidator(device=cfg.device, limit=5)

    print(f"Loading Model: {cfg.model_name}")
    # AUTOMATICALLY MAPPED TO GPU
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.float32, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    loss_scorer.tokenizer = tokenizer
    acc_validator.tokenizer = tokenizer

    for param in model.parameters():
        param.requires_grad = False

    print("Initializing Prompt...")
    init_text = "You are a helpful expert mathematician. "
    init_ids = tokenizer(init_text, return_tensors="pt").input_ids.to(cfg.device)
    with torch.no_grad():
        master_prompt = model.get_input_embeddings()(init_ids).squeeze(0).clone()
    prompt_shape = master_prompt.shape

    print(f"Starting Run: {cfg.run_name}")
    start_time_global = time.time()
    current_acc = 0.0
    all_best_losses, all_avg_losses, all_variances = [], [], []
    best_loss_overall = -float("inf")
    best_prompt_overall = None

    for gen in range(cfg.generations + 1):
        gen_start = time.time()
        noises = torch.randn(cfg.population_size, *prompt_shape, device=cfg.device) * cfg.sigma
        rewards = []

        for i in range(cfg.population_size):
            candidate = master_prompt + noises[i]
            score = loss_scorer.get_score(model, candidate)
            rewards.append(score)

        rewards_tensor = torch.tensor(rewards, device=cfg.device)
        gen_best = rewards_tensor.max().item()
        gen_avg = rewards_tensor.mean().item()
        gen_var = rewards_tensor.var().item()

        all_best_losses.append(gen_best)
        all_avg_losses.append(gen_avg)
        all_variances.append(gen_var)

        if gen_best > best_loss_overall:
            best_loss_overall = gen_best
            best_prompt_overall = master_prompt.clone()
            logger.save_prompt(best_prompt_overall, "best_prompt_overall.pt")

        std = rewards_tensor.std()
        if std == 0: std = 1e-8
        standardized = (rewards_tensor - gen_avg) / std

        gradient = torch.zeros_like(master_prompt)
        for i in range(cfg.population_size):
            gradient += noises[i] * standardized[i]

        master_prompt += (cfg.learning_rate / (cfg.population_size * cfg.sigma)) * gradient

        if gen % cfg.validate_every == 0 or gen == cfg.generations:
            current_acc = acc_validator.validate(model, master_prompt)

        duration = time.time() - gen_start
        logger.log_gen(gen, gen_best, gen_avg, gen_var, current_acc, duration)
        print(f"Gen {gen:<3} | Best: {gen_best:<8.4f} | Avg: {gen_avg:<8.4f} | Acc: {current_acc:<6.1%} | {duration:<5.1f}s")

    total_time = time.time() - start_time_global
    global_avg_loss = sum(all_avg_losses) / len(all_avg_losses)
    global_avg_variance = sum(all_variances) / len(all_variances)
    logger.log_run_summary(cfg, best_loss_overall, global_avg_loss, global_avg_variance, current_acc, total_time)
    print("Training Complete!")

if __name__ == "__main__":
    main()
"""
    notebook["cells"].append(create_cell("code", es_trainer_code))

    # --- BASELINE SFT ---
    # NOTE: Also upgraded device_map to "auto"
    sft_code = """%%writefile src/baseline_sft.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import json

def train_baseline():
    model_name = "Qwen/Qwen2.5-0.5B"
    print("Loading model on GPU...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("gsm8k", "main", split="train[:20]")

    def format_batch(batch):
        texts = []
        for q, a in zip(batch["question"], batch["answer"]):
            texts.append(f"Question: {q}\\nAnswer: {a}")
        return {"text": texts}

    print("Formatting dataset...")
    dataset = dataset.map(format_batch, batched=True)

    peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_dropout=0.05)
    
    sft_config = SFTConfig(
        output_dir="./models/baseline",
        max_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        packing=False,
        dataset_text_field="text"
    )

    trainer = SFTTrainer(model=model, train_dataset=dataset, peft_config=peft_config, args=sft_config)
    
    print("Starting training...")
    trainer.train()

    import os
    os.makedirs("results/baseline_res", exist_ok=True)
    with open("results/baseline_res/training_logs.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    trainer.save_model("./models/baseline_final")
    tokenizer.save_pretrained("./models/baseline_final")
    print("Training complete.")

if __name__ == "__main__":
    train_baseline()
"""
    notebook["cells"].append(create_cell("code", sft_code))

    # --- EXECUTION BLOCKS ---
    notebook["cells"].append(
        create_cell(
            "markdown",
            "### Run The Experiments\nUse the cells below to actually run the code we just saved.",
        )
    )
    notebook["cells"].append(create_cell("code", "!python src/es_trainer_bpb.py"))
    notebook["cells"].append(create_cell("code", "!python src/baseline_sft.py"))

    # --- ZIP & DOWNLOAD ---
    notebook["cells"].append(
        create_cell(
            "markdown",
            "### Download Results\nRun this cell to zip all your results so you can download them to your local computer!",
        )
    )
    notebook["cells"].append(
        create_cell(
            "code",
            "!zip -r my_results.zip results/\nfrom google.colab import files\nfiles.download('my_results.zip')",
        )
    )

    with open("es_finetune_project.ipynb", "w") as f:
        json.dump(notebook, f, indent=4)

    print("✅ Created es_finetune_project.ipynb")


if __name__ == "__main__":
    build_notebook()
