import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

MODEL_DIR = "./model-v1"

print("Loading model...")
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()

if torch.cuda.is_available():
    model.to("cuda")

def complete_word(prefix, max_new_tokens=10, num_return_sequences=100):
    input_ids = tokenizer(f"<bos>{prefix}", return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.8,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    results = []
    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        if text.startswith("<bos>"):
            text = text.replace("<bos>", "", 1)
        results.append(text)
    return results

if __name__ == "__main__":
    while True:
        prefix = input("\nEnter first few characters: ").strip()
        if prefix.lower() == "quit":
            break
        completions = complete_word(prefix)
        print("\nPredictions:")
        for i, c in enumerate(completions, 1):
            print(f"{i}. {c}")
