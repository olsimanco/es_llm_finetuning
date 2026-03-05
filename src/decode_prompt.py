import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def decode_soft_prompt(model_name, prompt_path):
    print(f"Loading Model Vocabulary: {model_name}...")

    # Load the model and tokenizer on CPU (we only need the embedding layer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Extract the AI's entire dictionary of embeddings (Vocab Size x Hidden Size)
    word_embeddings = model.get_input_embeddings().weight.data

    # Load evolved soft prompt tensor
    print(f"Loading Evolved Prompt: {prompt_path}...")
    soft_prompt = torch.load(prompt_path, map_location="cpu")

    print("\n" + "=" * 45)
    print("   DECODED PROMPT (Nearest Neighbors)")
    print("=" * 45)

    # Loop through each individual "Token" in soft prompt
    for i in range(soft_prompt.shape[0]):
        vector = soft_prompt[i].unsqueeze(0)  # Isolate the specific evolved vector

        # Calculate Cosine Similarity between this vector and every human word in the dictionary
        similarities = F.cosine_similarity(vector, word_embeddings)

        # Grab the indices of the top 3 closest words
        top_k_indices = torch.topk(similarities, 3).indices

        # Translate the indices back to human text using the tokenizer
        words = [repr(tokenizer.decode([idx])) for idx in top_k_indices]

        # Get the actual distance/similarity scores
        scores = [similarities[idx].item() for idx in top_k_indices]

        print(f"Token {i+1}:")
        print(f"  1st closest: {words[0]:<15} | Similarity: {scores[0]:.4f}")
        print(f"  2nd closest: {words[1]:<15} | Similarity: {scores[1]:.4f}")
        print(f"  3rd closest: {words[2]:<15} | Similarity: {scores[2]:.4f}")
        print("-" * 40)


if __name__ == "__main__":

    PROMPT_FILE = "results/es_res_bpb/prompt_gen_49.pt"

    try:
        decode_soft_prompt("Qwen/Qwen2.5-0.5B", PROMPT_FILE)
    except FileNotFoundError:
        print(f"\nERROR: Could not find the file at '{PROMPT_FILE}'.")
        print(
            "Please check your results/ folder and update the PROMPT_FILE path in the script!"
        )
