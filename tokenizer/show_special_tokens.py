
from transformers import AutoTokenizer

# List of models to test
models = [
    "bert-base-uncased",
    "roberta-base",
    "gpt2",
    "t5-small",
    "xlm-roberta-base"
]

text = "The quick brown fox jumps over the lazy dog."

for model_name in models:
    print("=" * 30)
    print(f"Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Print special token mappings
    print("Special Tokens Map:", tokenizer.special_tokens_map)
    print("All Special Tokens:", tokenizer.all_special_tokens)

    # Add special tokens to the text
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt"
    )
    decoded = tokenizer.decode(encoded["input_ids"][0])

    print("Tokenized IDs:", encoded["input_ids"][0].tolist())
    print("Decoded Text with Special Tokens:")
    print(decoded)
