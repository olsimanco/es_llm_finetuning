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

# --- MENTOR'S CONFIG ---
TASK_SUITE_CONFIGS = {
    "modalities:base_easy:math_bpb": {
        "tasks": ["minerva_math_algebra:bpb::olmes"],
        "primary_metric": "macro",
    }
}


class ESConfig:
    # --- Experiment Settings ---
    run_name = "Run_01_WholeRunStats"
    model_name = "Qwen/Qwen2.5-0.5B"
    generations = 5
    population_size = 4
    sigma = 0.05
    learning_rate = 0.01

    # --- Validation Settings ---
    validate_every = 5
    device = "cpu"


class ExperimentLogger:
    def __init__(self, config):
        # 1. Setup Directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_dir = os.path.join("results", f"{config.run_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # 2. Detailed Gen-by-Gen Log
        self.csv_file = os.path.join(self.run_dir, "gen_stats.csv")
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["gen", "best_loss", "avg_loss", "variance", "accuracy", "time_sec"]
            )

        # 3. Leaderboard Log (Master File)
        self.leaderboard_file = "results/leaderboard.csv"
        if not os.path.exists(self.leaderboard_file):
            with open(self.leaderboard_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Run_Name",
                        "Model",
                        "Gens",
                        "Pop",
                        "Sigma",
                        "Global_Best_Loss",
                        "Global_Avg_Loss",
                        "Avg_Variance",
                        "Final_Acc",
                        "Total_Time_Min",
                    ]
                )

    def log_gen(self, gen, best_loss, avg_loss, variance, accuracy, duration):
        """Logs stats for a single generation."""
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    gen,
                    f"{best_loss:.4f}",
                    f"{avg_loss:.4f}",
                    f"{variance:.4f}",
                    f"{accuracy:.2%}",
                    f"{duration:.1f}",
                ]
            )

    def log_run_summary(
        self, config, global_best, global_avg, avg_variance, final_acc, total_time
    ):
        """Logs the final summary to the leaderboard."""
        with open(self.leaderboard_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    config.run_name,
                    config.model_name,
                    config.generations,
                    config.population_size,
                    config.sigma,
                    f"{global_best:.4f}",
                    f"{global_avg:.4f}",
                    f"{avg_variance:.4f}",
                    f"{final_acc:.2%}",
                    f"{total_time/60:.2f}",
                ]
            )
        print(f"\n>>> Run saved to Leaderboard: {self.leaderboard_file}")
        print(f">>> Full stats saved to: {self.run_dir}")

    def save_prompt(self, prompt_tensor, filename):
        """Saves the prompt tensor."""
        torch.save(prompt_tensor, os.path.join(self.run_dir, filename))


def main():
    cfg = ESConfig()
    logger = ExperimentLogger(cfg)

    # 1. Setup Scorers
    target_suite = TASK_SUITE_CONFIGS["modalities:base_easy:math_bpb"]
    loss_scorer = BPBScorer(target_suite, device=cfg.device, limit=3)
    acc_validator = AccuracyValidator(device=cfg.device, limit=5)

    # 2. Load Model
    print(f"Loading Model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    loss_scorer.tokenizer = tokenizer
    acc_validator.tokenizer = tokenizer

    for param in model.parameters():
        param.requires_grad = False

    # 3. Initialize Prompt
    print("Initializing Prompt...")
    init_text = "You are a helpful expert mathematician. "
    init_ids = tokenizer(init_text, return_tensors="pt").input_ids
    with torch.no_grad():
        master_prompt = model.get_input_embeddings()(init_ids).squeeze(0).clone()
    prompt_shape = master_prompt.shape

    # 4. Evolution Loop
    print(f"Starting Run: {cfg.run_name}")
    print(
        f"{'Gen':<5} | {'Best Loss':<10} | {'Avg Loss':<10} | {'Var':<8} | {'Acc':<6} | {'Time':<6}"
    )
    print("-" * 60)

    start_time_global = time.time()
    current_acc = 0.0

    # Stats Accumulators
    all_best_losses = []
    all_avg_losses = []
    all_variances = []

    best_loss_overall = -float("inf")
    best_prompt_overall = None

    for gen in range(cfg.generations + 1):
        gen_start = time.time()

        # A. Create Population
        noises = torch.randn(cfg.population_size, *prompt_shape) * cfg.sigma
        rewards = []

        # B. Evaluate
        for i in range(cfg.population_size):
            candidate = master_prompt + noises[i]
            score = loss_scorer.get_score(model, candidate)
            rewards.append(score)

        # C. Calculate Gen Stats
        rewards_tensor = torch.tensor(rewards)
        gen_best = rewards_tensor.max().item()
        gen_avg = rewards_tensor.mean().item()
        gen_var = rewards_tensor.var().item()

        # D. Accumulate for Whole Run Stats
        all_best_losses.append(gen_best)
        all_avg_losses.append(gen_avg)
        all_variances.append(gen_var)

        # E. Keep Track of Best Ever
        if gen_best > best_loss_overall:
            best_loss_overall = gen_best
            best_prompt_overall = master_prompt.clone()
            logger.save_prompt(best_prompt_overall, "best_prompt_overall.pt")

        # F. Update Master Prompt
        std = rewards_tensor.std()
        if std == 0:
            std = 1e-8
        standardized = (rewards_tensor - gen_avg) / std

        gradient = torch.zeros_like(master_prompt)
        for i in range(cfg.population_size):
            gradient += noises[i] * standardized[i]

        master_prompt += (
            cfg.learning_rate / (cfg.population_size * cfg.sigma)
        ) * gradient

        # G. Validate Accuracy
        if gen % cfg.validate_every == 0 or gen == cfg.generations:
            current_acc = acc_validator.validate(model, master_prompt)

        # H. Log & Print
        duration = time.time() - gen_start
        logger.log_gen(gen, gen_best, gen_avg, gen_var, current_acc, duration)

        print(
            f"{gen:<5} | {gen_best:<10.4f} | {gen_avg:<10.4f} | {gen_var:<8.4f} | {current_acc:<6.1%} | {duration:<6.1f}s"
        )

    # 5. Final WHOLE RUN Summary
    total_time = time.time() - start_time_global

    # Aggregated Stats
    global_avg_loss = sum(all_avg_losses) / len(all_avg_losses)
    global_avg_variance = sum(all_variances) / len(all_variances)

    # Save to Leaderboard
    logger.log_run_summary(
        cfg,
        best_loss_overall,
        global_avg_loss,
        global_avg_variance,
        current_acc,
        total_time,
    )

    print("\n" + "=" * 40)
    print("      WHOLE RUN STATISTICS")
    print("=" * 40)
    print(f"Total Runtime       : {total_time/60:.2f} minutes")
    print(f"Best Negative Loss  : {best_loss_overall:.4f}")
    print(f"Global Average Loss : {global_avg_loss:.4f}")
    print(f"Average Variance    : {global_avg_variance:.4f}")
    print(f"Final Accuracy      : {current_acc:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    main()
