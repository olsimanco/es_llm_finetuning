import subprocess
import json
import os
import shutil
import glob
import torch

print("Loaded OPTIMIZED FullOlmesWrapper (v6 - Explicit Base Model)")


class FullOlmesWrapper:
    def __init__(self, base_model_name, task_name, limit=None):
        self.base_model_name = base_model_name
        self.task_name = task_name
        self.limit = limit

        self.temp_base_dir = os.path.join("results", "temp_adapters")
        os.makedirs(self.temp_base_dir, exist_ok=True)

    def get_score(self, peft_model, soft_prompt_tensor, gen_idx, cand_idx):
        # --- Step A: Unique Path ---
        run_id = f"gen_{gen_idx}_cand_{cand_idx}"
        adapter_path = os.path.join(self.temp_base_dir, run_id)
        output_path = os.path.join(self.temp_base_dir, f"{run_id}_out")

        # Clean previous runs
        if os.path.exists(adapter_path):
            shutil.rmtree(adapter_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        # --- Step B: Inject Weights ---
        with torch.no_grad():
            if hasattr(peft_model.prompt_encoder, "default"):
                peft_model.prompt_encoder["default"].embedding.weight.copy_(
                    soft_prompt_tensor
                )
            elif hasattr(peft_model.prompt_encoder, "embedding"):
                peft_model.prompt_encoder.embedding.weight.copy_(soft_prompt_tensor)
            else:
                peft_model.prompt_encoder.embedding.weight.copy_(soft_prompt_tensor)

        # Save to Disk
        peft_model.save_pretrained(adapter_path)

        # --- Step C: Run OLMES CLI ---
        # FIX: We must explicitly tell it the base model via 'pretrained='
        # Otherwise it thinks the model name is "hf"
        model_args = f"pretrained={self.base_model_name},peft={adapter_path},trust_remote_code=True"

        cmd = [
            "olmes",
            "--model",
            "hf",
            "--model-args",
            model_args,
            "--task",
            self.task_name,
            "--output-dir",
            output_path,
            "--batch-size",
            "1",
        ]

        if self.limit:
            cmd.extend(["--limit", str(self.limit)])

        try:
            # Run OLMES
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # It is safer to catch the error and return 0.0 than to crash the whole loop
            print(
                f"OLMES Failed for {run_id}. \nCommand: {' '.join(cmd)}\nError: {e.stderr}"
            )
            score = 0.0
        else:
            score = self._parse_results(output_path)

        # --- Step D: Cleanup ---
        if os.path.exists(adapter_path):
            shutil.rmtree(adapter_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        return score

    def _parse_results(self, output_dir):
        try:
            json_files = glob.glob(f"{output_dir}/**/*.json", recursive=True)
            if not json_files:
                return 0.0

            with open(json_files[0], "r") as f:
                data = json.load(f)

            results = data.get("results", {})
            if not results:
                return 0.0

            task_key = list(results.keys())[0]
            metrics = results[task_key]

            return metrics.get("acc_norm", metrics.get("acc", 0.0))

        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return 0.0
