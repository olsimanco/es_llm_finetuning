import torch
from datasets import load_dataset
import numpy as np


class BPBScorer:
    def __init__(self, suite_config, device="cpu", limit=10):
        """
        Evaluates the model based on Loss (BPB) rather than Accuracy.
        Args:
            suite_config: The dictionary containing the list of tasks.
            limit: Max questions per task (keep small for CPU speed).
        """
        self.device = device
        self.tokenizer = None  # Set by trainer
        self.tasks = {}

        # Parse the tasks from the Mentor's config
        task_list = suite_config.get("tasks", [])
        if isinstance(task_list, dict):
            task_list = list(task_list.keys())

        print(f"Loading {len(task_list)} tasks for BPB evaluation...")

        for task_str in task_list:
            # Extract simple name (e.g., 'minerva_math_algebra' from 'minerva_math_algebra:bpb::olmes')
            dataset_name = task_str.split(":")[0]

            print(f"  - Loading {dataset_name}...")
            try:
                # 1. Try loading the specific dataset
                # Note: If minerva is not available, we map it to GSM8k for testing purposes
                if "minerva" in dataset_name:
                    ds = load_dataset("gsm8k", "main", split=f"test[:{limit}]")
                else:
                    ds = load_dataset(dataset_name, split=f"test[:{limit}]")

                self.tasks[dataset_name] = ds
            except Exception as e:
                print(
                    f"    ! Could not load {dataset_name} ({e}). using GSM8k fallback."
                )
                self.tasks[dataset_name] = load_dataset(
                    "gsm8k", "main", split=f"test[:{limit}]"
                )

    def get_score(self, model, soft_prompt):
        """
        Returns the average NEGATIVE LOSS across all tasks.
        (Higher is better: -2.0 is better than -5.0)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set!")

        total_loss = 0
        total_tasks = 0

        # Iterate over every math subject
        for name, dataset in self.tasks.items():
            task_loss = 0
            count = 0

            for i in range(len(dataset)):
                # Handle different dataset column names
                q = dataset[i].get("question", dataset[i].get("problem", ""))
                a = dataset[i].get("answer", dataset[i].get("solution", ""))

                # We want the model to predict the Answer given the Question
                # Format: "Question: ... Answer: ..."
                full_text = f"Question: {q}\nAnswer: {a}"

                inputs = self.tokenizer(full_text, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device)

                # --- INJECT SOFT PROMPT ---
                # 1. Get Embeddings of real text
                input_embeds = model.get_input_embeddings()(input_ids)

                # 2. Reshape Soft Prompt to [1, Length, Hidden]
                soft_prompt_batch = soft_prompt.unsqueeze(0)

                # 3. Concatenate: [Soft Prompt] + [Real Embeddings]
                combined_embeds = torch.cat([soft_prompt_batch, input_embeds], dim=1)

                # --- CALCULATE LOSS ---
                # We need to create labels.
                # The labels for the 'Soft Prompt' part should be -100 (ignored).
                # The labels for the 'Real Text' part should be input_ids.

                ignore_labels = torch.full(
                    (1, soft_prompt.shape[0]),
                    -100,
                    dtype=torch.long,
                    device=self.device,
                )
                real_labels = input_ids.clone()
                combined_labels = torch.cat([ignore_labels, real_labels], dim=1)

                with torch.no_grad():
                    outputs = model(
                        inputs_embeds=combined_embeds, labels=combined_labels
                    )
                    loss = outputs.loss

                task_loss += loss.item()
                count += 1

            # Add average loss for this task to total
            if count > 0:
                total_loss += task_loss / count
                total_tasks += 1

        # Return negative loss (Fitness)
        if total_tasks == 0:
            return -99.0
        return -(total_loss / total_tasks)
