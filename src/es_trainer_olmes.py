import torch
import numpy as np
import os
import csv
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

# Import the OPTIMIZED wrapper
from full_olmes_wrapper import FullOlmesWrapper


class ESConfig:
    task_name = "gsm8k"
    model_name = "Qwen/Qwen2.5-0.5B"
    generations = 3
    population_size = 2
    sigma = 0.1
    learning_rate = 0.1
    prompt_length = 10
    device = "cpu"
    limit_questions = 5


class ExperimentLogger:
    def __init__(self, config):
        self.run_dir = os.path.join("results", "es_res_real_olmes")
        os.makedirs(self.run_dir, exist_ok=True)

        config_dict = {
            k: v for k, v in config.__dict__.items() if not k.startswith("__")
        }
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        self.csv_file = os.path.join(self.run_dir, "log.csv")
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["generation", "mean_score", "max_score", "min_score", "time_taken"]
            )

        print(f"Logging results to: {self.run_dir}")

    def log_stats(self, generation, scores, duration):
        if isinstance(scores, list):
            scores = np.array(scores)
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [generation, scores.mean(), scores.max(), scores.min(), duration]
            )

    def save_model(self, prompt_tensor, filename):
        torch.save(prompt_tensor, os.path.join(self.run_dir, filename))


def main():
    cfg = ESConfig()
    logger = ExperimentLogger(cfg)

    # 1. Initialize Wrapper
    wrapper = FullOlmesWrapper(cfg.model_name, cfg.task_name, limit=cfg.limit_questions)

    # 2. Load Base Model
    print(f"Loading Base Model: {cfg.model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32, device_map=None
    )
    for param in base_model.parameters():
        param.requires_grad = False

    # 3. WRAP MODEL ONCE (The Fix)
    # We turn it into a PEFT model here, so we don't have to do it inside the loop
    print("Wrapping model in PEFT Prompt Tuning...")
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=cfg.prompt_length,
        tokenizer_name_or_path=cfg.model_name,
    )
    peft_model = get_peft_model(base_model, peft_config)

    # 4. Initialize Master Prompt
    # We can grab the random init from the PEFT model itself to start
    # This ensures shapes match perfectly
    print("Initializing Master Prompt...")
    if hasattr(peft_model.prompt_encoder, "default"):
        master_prompt = peft_model.prompt_encoder[
            "default"
        ].embedding.weight.data.clone()
    else:
        master_prompt = peft_model.prompt_encoder.embedding.weight.data.clone()

    prompt_shape = master_prompt.shape

    # 5. Evolution Loop
    print(
        f"Starting Real-OLMES Evolution: {cfg.generations} Gens, Pop {cfg.population_size}"
    )

    for gen in range(cfg.generations):
        gen_start = time.time()
        print(f"\n--- Generation {gen} ---")

        # A. Create Population
        noises = (
            torch.randn(cfg.population_size, *prompt_shape, device=cfg.device)
            * cfg.sigma
        )
        rewards = []

        # B. Evaluate
        for i in range(cfg.population_size):
            noise = noises[i]
            candidate = master_prompt + noise

            print(f"  Evaluating Child {i} via OLMES CLI...")

            # Pass the PRE-WRAPPED peft_model
            score = wrapper.get_score(peft_model, candidate, gen, i)

            rewards.append(score)
            print(f"  Child {i} Score: {score}")

        # C. Update
        rewards_tensor = torch.tensor(rewards, device=cfg.device)
        std = rewards_tensor.std()
        if std == 0:
            std = 1e-8

        standardized_rewards = (rewards_tensor - rewards_tensor.mean()) / std

        gradient = torch.zeros_like(master_prompt)
        for i in range(cfg.population_size):
            gradient += noises[i] * standardized_rewards[i]

        master_prompt += (
            cfg.learning_rate / (cfg.population_size * cfg.sigma)
        ) * gradient

        # D. Log
        duration = time.time() - gen_start
        logger.log_stats(gen, rewards, duration)
        logger.save_model(master_prompt, f"prompt_gen_{gen}.pt")

        print(f"Gen {gen} Done. Max Score: {max(rewards):.4f} | Time: {duration:.1f}s")


if __name__ == "__main__":
    main()
