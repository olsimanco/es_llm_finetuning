from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# The specific ID of the model on Hugging Face
model_id = "Qwen/Qwen2.5-0.5B"

print(f"Downloading {model_id}...")

# 1. Download the Tokenizer (The tool that turns text into numbers)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Download the Model (The brain)
# We use torch_dtype="auto" to download it in the most efficient format for your hardware
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",  # Automatically puts it on GPU if available
)

print("Download complete!")

# 3. Test it immediately to make sure it works
input_text = "The square root of 64 is"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate a completion
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)

print("Model Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
