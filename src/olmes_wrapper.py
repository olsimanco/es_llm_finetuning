import subprocess
import json
import os
import shutil
import glob
import torch
from peft import PromptTuningConfig, TaskType, get_peft_model, PromptTuningInit

print("Loaded CORRECTED FullOlmesWrapper (v2)")  # <--- Verify this prints!


class FullOlmesWrapper:
    def __init__(self, base_model_name, task_name, limit=None):
        self.base_model_name = base_model_name
        self.task_name = task_name
        self.limit = limit

        # Directory to store temporary adapters for OLMES to read
        self.temp_base_dir = os.path.join("results", "temp_adapters")
        os.makedirs(self.temp_base_dir, exist_ok=True)

    def get_score(self, base_model, soft_prompt_tensor, gen_idx, cand_idx):
        # --- Step A: Unique Path ---
        run_id = f"gen_{gen_idx}_cand_{cand_idx}"
        adapter_path = os.path.join(self.temp_base_dir, run_id)
        output_path = os.path.join(self.temp_base_dir, f"{run_id}_out")

        # Clean previous runs
        if os.path.exists(adapter_path):
            shutil.rmtree(adapter_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        # --- Step B: Create Adapter ---
        num_tokens = soft_prompt_tensor.shape[0]
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=num_tokens,
            tokenizer_name_or_path=self.base_model_name,
        )

        # Wrap model
        peft_model = get_peft_model(base_model, peft_config)

        # --- THE FIX IS HERE (Line 67 approx) ---
        # We access ["default"] before .embedding
        with torch.no_grad():
            if hasattr(peft_model.prompt_encoder, "embedding"):
                # Old PEFT version fallback
                peft_model.prompt_encoder.embedding.weight.copy_(soft_prompt_tensor)
            else:
                # Modern PEFT version (ModuleDict)
                peft_model.prompt_encoder["default"].embedding.weight.copy_(
                    soft_prompt_tensor
                )

        # Save to Disk
        peft_model.save_pretrained(adapter_path)

        # UNWRAP model (Critical!)
        peft_model.unload()
        del peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Step C: Run OLMES CLI ---
        model_args = f"peft={adapter_path},trust_remote_code=True"

        cmd = [
            "olmes",
            "--model",
            "hf",
            "--model-args",
            model_args,
            "--tasks",
            self.task_name,
            "--output_path",
            output_path,
            "--batch_size",
            "1",
            "--device",
            "cpu",
        ]

        if self.limit:
            cmd.extend(["--limit", str(self.limit)])

        try:
            # Run OLMES
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"OLMES Failed for {run_id}: {e.stderr}")
            return 0.0

        # --- Step D: Parse Results ---
        score = self._parse_results(output_path)

        # Cleanup
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
