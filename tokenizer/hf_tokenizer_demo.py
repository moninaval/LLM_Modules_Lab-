from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Input sentences
text = "This is a sentence."
text_pair = "Here is another."

# Tokenize with special tokens
encoded = tokenizer(
    text,
    text_pair,
    add_special_tokens=True,
    return_token_type_ids=True
)

# Get tokens and token IDs
input_ids = encoded["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids)
token_types = encoded["token_type_ids"]

# Print token details
print("Index\tToken\t\tToken ID\tSegment (0=first, 1=second)")
for i, (tok, tid, seg) in enumerate(zip(tokens, input_ids, token_types)):
    print(f"{i:2d}\t{tok:12s}\t{tid:8d}\t{seg}")
