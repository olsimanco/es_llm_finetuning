import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def speak_the_prompt(model_name, prompt_path):
    print(f"Loading Model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading Evolved Prompt: {prompt_path}...")
    # Load the prompt and add the batch dimension: [1, Length, Hidden_Size]
    soft_prompt = torch.load(prompt_path, map_location="cpu").unsqueeze(0)

    # --- THE ATTENTION MASK ---

    prompt_length = soft_prompt.shape[1]
    attention_mask = torch.ones((1, prompt_length), dtype=torch.long, device="cpu")
    # --------------------------------------

    print("\n" + "=" * 50)
    print(" WHAT THE AI 'HEARS' WHEN IT READS THE PROMPT")
    print("=" * 50)

    # Force the model to generate the next 15 tokens
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=soft_prompt,
            attention_mask=attention_mask,  # <--- PASS THE MASK HERE
            max_new_tokens=15,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
        )

    # Decode the generated tokens into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Generated Output: \n\n{generated_text}")
    print("\n" + "=" * 50)


if __name__ == "__main__":

    PROMPT_FILE = "results/es_res_bpb/prompt_gen_1.pt"

    try:
        speak_the_prompt("Qwen/Qwen2.5-0.5B", PROMPT_FILE)
    except FileNotFoundError:
        print(f"Could not find the file at {PROMPT_FILE}")
