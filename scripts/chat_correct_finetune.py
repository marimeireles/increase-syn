"""Chat with the metacognitive finetuned Gemma 3 4B-IT model (Steyvers method)."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "results/finetuning/merged_model"

print("Loading finetuned model from", MODEL_PATH, "...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Model loaded! Type 'quit' or 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit", "q"):
        break

    messages = [{"role": "user", "content": user_input}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    outputs = model.generate(
        inputs, max_new_tokens=512, do_sample=True, temperature=0.7
    )
    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:], skip_special_tokens=True
    )
    print(f"\nModel: {response}\n")
