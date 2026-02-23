import torch
import numpy as np
import os
import csv
import json
import time
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our "Lightweight OLMES" Scorer
from olmes_wrapper import OLMESScorer


# --- Configuration ---
class ESConfig:
    task_name = "gsm8k"
    model_name = "Qwen/Qwen2.5-0.5B"
    generations = 10  # Number of loops
    population_size = 10  # Number of children per loop
    sigma = 0.1  # Noise scale
    learning_rate = 0.1  # Step size
    prompt_length = 10  # Number of soft tokens
    device = "cpu"  # Force CPU


# --- Logger Class (Updated for specific folder) ---
class ExperimentLogger:
    def __init__(self, config):
        # 1. Define the specific folder you requested
        self.run_dir = os.path.join("results", "es_res")

        # 2. Create it (and clean it if it exists to start fresh)
        if os.path.exists(self.run_dir):
            print(f"Warning: Overwriting existing results in {self.run_dir}")
        else:
            os.makedirs(self.run_dir, exist_ok=True)

        # 3. Save Configuration
        config_dict = {
            k: v for k, v in config.__dict__.items() if not k.startswith("__")
        }
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        # 4. Initialize CSV Log
        self.csv_file = os.path.join(self.run_dir, "log.csv")
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "mean_score", "max_score", "min_score"])

        print(f"Logging results to: {self.run_dir}")

    def log_stats(self, generation, scores):
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        elif isinstance(scores, list):
            scores = np.array(scores)

        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([generation, scores.mean(), scores.max(), scores.min()])

    def save_model(self, prompt_tensor, filename="best_prompt.pt"):
        torch.save(prompt_tensor, os.path.join(self.run_dir, filename))


# --- Main Loop ---
def main():
    # 1. Setup
    cfg = ESConfig()
    logger = ExperimentLogger(cfg)

    print(f"Initializing Scorer for task: {cfg.task_name}...")
    # Limit to 5 questions so it runs fast!
    scorer = OLMESScorer(cfg.task_name, device=cfg.device, limit=5)

    print(f"Loading Model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # CRITICAL: Pass tokenizer to scorer
    scorer.tokenizer = tokenizer

    # Freeze Model
    for param in model.parameters():
        param.requires_grad = False

    print("Model Loaded and Frozen.")

    # 2. Create the "Individual" (Soft Prompt)
    # Shape: (10 tokens, embedding size)
    prompt_shape = (cfg.prompt_length, model.config.hidden_size)
    master_prompt = torch.randn(prompt_shape, device=cfg.device)

    # 3. Evolution Loop
    print(f"Starting Evolution: {cfg.generations} generations...")

    for gen in range(cfg.generations):
        start_time = time.time()

        # A. Create Population
        noise_list = []
        rewards = []

        # Generate noise: Shape [Pop_Size, Tokens, Hidden]
        noises = (
            torch.randn(cfg.population_size, *prompt_shape, device=cfg.device)
            * cfg.sigma
        )

        print(f"--- Generation {gen} Evaluation ---")
        for i in range(cfg.population_size):
            noise = noises[i]
            candidate = master_prompt + noise

            # Evaluate using the In-Memory Scorer
            score = scorer.get_score(model, candidate)
            rewards.append(score)
            print(f"  Child {i}: Score {score:.2f}")

        # B. Compute Update
        rewards_tensor = torch.tensor(rewards, device=cfg.device)

        std = rewards_tensor.std()
        if std == 0:
            std = 1e-8

        standardized_rewards = (rewards_tensor - rewards_tensor.mean()) / std

        gradient_approx = torch.zeros_like(master_prompt)
        for i in range(cfg.population_size):
            gradient_approx += noises[i] * standardized_rewards[i]

        # Update Master Prompt
        update_step = (
            cfg.learning_rate / (cfg.population_size * cfg.sigma)
        ) * gradient_approx
        master_prompt += update_step

        # C. Logging
        max_score = rewards_tensor.max().item()
        logger.log_stats(gen, rewards_tensor)
        logger.save_model(master_prompt, f"prompt_gen_{gen}.pt")

        duration = time.time() - start_time
        print(f"Gen {gen} Complete. Best: {max_score:.2f} | Time: {duration:.2f}s")

    print(f"Evolution Complete. Check {logger.run_dir} for results.")


if __name__ == "__main__":
    main()
