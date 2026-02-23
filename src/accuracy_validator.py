import torch
from datasets import load_dataset
import re


class AccuracyValidator:
    def __init__(self, device="cpu", limit=10):
        """
        Runs a 'Real' test: Generates text and checks if the answer is correct.
        """
        self.device = device
        self.tokenizer = None
        # We use GSM8k as the standard 'Real World' test
        print(f"Loading {limit} Validation Questions (GSM8k)...")
        self.dataset = load_dataset("gsm8k", "main", split=f"test[:{limit}]")

    def validate(self, model, soft_prompt):
        """
        Returns: Accuracy (0.0 to 1.0)
        """
        correct = 0
        total = len(self.dataset)

        # 1. Expand Prompt: [Length, Hidden] -> [1, Length, Hidden]
        soft_prompt_batch = soft_prompt.unsqueeze(0)

        print(f"  > Running Validation on {total} questions...")

        for i, item in enumerate(self.dataset):
            question = item["question"]
            target = item["answer"].split("####")[-1].strip()  # Extract number

            # Input Text
            text = f"Question: {question}\nAnswer:"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            # 2. Inject Prompt & Generate
            with torch.no_grad():
                # Get embeddings of text
                input_embeds = model.get_input_embeddings()(inputs.input_ids)
                # Concatenate [Prompt, Text]
                combined_embeds = torch.cat([soft_prompt_batch, input_embeds], dim=1)

                # Generate new tokens
                outputs = model.generate(
                    inputs_embeds=combined_embeds,
                    max_new_tokens=15,  # Keep it short for speed
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # 3. Check Answer
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Simple check: is the target number in the output?
            # (Clean up output to avoid false negatives)
            if target in output_text:
                correct += 1

        score = correct / total
        print(f"  > Validation Complete. Accuracy: {score:.2%} ({correct}/{total})")
        return score
