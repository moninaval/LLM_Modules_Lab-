import os
import torch
from transformers import AutoTokenizer

# --- 0. Setup: Define some text inputs ---
print("--- 0. Setup ---")
text_single = "Hello, how are you doing today?"
text_pair = "The quick brown fox jumps over the lazy dog."
text_pair_2 = "Dogs are great companions for humans."
batch_texts = [
    "This is the first sentence.",
    "And here is the second one, which is a bit longer.",
    "Short."
]
long_text = " ".join(["word" for _ in range(100)]) # A very long text


print("\n--- I. Core Tokenization & Encoding (Text to Numbers) ---")

# --- 1. from_pretrained(model_name_or_path) ---
# The primary way to load a pre-trained tokenizer.
print("\n1. Loading Tokenizer: tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Padding side: {tokenizer.padding_side}")


# --- 2. tokenize(text) ---
# Converts a raw string into a list of subword tokens (strings), without special tokens or IDs.
print("\n2. tokenize(text) - Raw tokenization:")
raw_tokens = tokenizer.tokenize(text_single)
print(f"Original text: '{text_single}'")
print(f"Raw tokens: {raw_tokens}")


# --- 3. convert_tokens_to_ids(tokens) ---
# Converts a list of token strings into their numerical IDs.
print("\n3. convert_tokens_to_ids(tokens) - Tokens to IDs:")
token_ids = tokenizer.convert_tokens_to_ids(raw_tokens)
print(f"Token IDs from raw tokens: {token_ids}")


# --- 4. __call__(text_or_texts, ...) - The main encoding method ---
print("\n4. __call__(...) - Comprehensive Encoding:")

# Encoding a single text
print("\n  a) Encoding a single text (with special tokens, padding, truncation, attention mask):")
encoded_single = tokenizer(
    text_single,
    padding='max_length',
    truncation=True,
    max_length=20, # Set a max length for demonstration
    return_tensors="pt"
)
print(f"Original text: '{text_single}'")
print(f"Input IDs (padded): {encoded_single['input_ids']}")
print(f"Attention Mask (0 for padding): {encoded_single['attention_mask']}")
print(f"Token Type IDs (BERT specific, 0 for first segment): {encoded_single['token_type_ids']}")
# Decode to show what's inside with special tokens
print(f"Decoded (with special tokens): '{tokenizer.decode(encoded_single['input_ids'][0], skip_special_tokens=False)}'")

# Encoding a pair of texts
print("\n  b) Encoding a pair of texts (for tasks like NLI):")
encoded_pair = tokenizer(
    text_pair, text_pair_2,
    padding='longest', # Pad to the longest sequence in the pair
    truncation=True,
    return_tensors="pt"
)
print(f"Text 1: '{text_pair}'")
print(f"Text 2: '{text_pair_2}'")
print(f"Input IDs (paired): {encoded_pair['input_ids']}")
print(f"Attention Mask: {encoded_pair['attention_mask']}")
print(f"Token Type IDs (0 for first, 1 for second segment): {encoded_pair['token_type_ids']}")
print(f"Decoded (paired, with special tokens): '{tokenizer.decode(encoded_pair['input_ids'][0], skip_special_tokens=False)}'")

# Encoding a batch of texts with padding and truncation
print("\n  c) Encoding a batch of texts:")
encoded_batch = tokenizer(
    batch_texts,
    padding='longest', # Pad all sequences to the length of the longest in the batch
    truncation=True,
    max_length=15, # Max length for all sequences in batch
    return_tensors="pt"
)
print(f"Original batch texts: {batch_texts}")
print(f"Batch Input IDs:\n{encoded_batch['input_ids']}")
print(f"Batch Attention Mask:\n{encoded_batch['attention_mask']}")
print("\nDecoded batch (with special tokens for inspection):")
for i, ids in enumerate(encoded_batch['input_ids']):
    print(f"  Seq {i}: '{tokenizer.decode(ids, skip_special_tokens=False)}'")


# --- 5. encode(text_or_texts, ...) ---
# A simpler version of __call__ that primarily returns input_ids.
print("\n5. encode(text) - Simplified encoding (returns only input_ids):")
encoded_simple_ids = tokenizer.encode(text_single, add_special_tokens=True)
print(f"Encoded IDs (simplified): {encoded_simple_ids}")


print("\n--- II. Decoding (Numbers to Text) ---")

# --- 1. decode(token_ids, ...) ---
# Converts a list of token IDs back into a single human-readable string.
print("\n1. decode(token_ids, skip_special_tokens=True/False) - Decoding IDs to text:")
print(f"IDs from previous single encoding: {encoded_single['input_ids'][0].tolist()}")
decoded_with_special = tokenizer.decode(encoded_single['input_ids'][0], skip_special_tokens=False)
decoded_without_special = tokenizer.decode(encoded_single['input_ids'][0], skip_special_tokens=True)
print(f"Decoded (with special tokens): '{decoded_with_special}'")
print(f"Decoded (without special tokens): '{decoded_without_special}'")

# --- 2. batch_decode(batch_of_token_ids, ...) ---
# Decodes a list of token ID sequences (a batch) into a list of human-readable strings.
print("\n2. batch_decode(batch_of_token_ids, skip_special_tokens=True) - Decoding a batch:")
decoded_batch = tokenizer.batch_decode(encoded_batch['input_ids'], skip_special_tokens=True)
print(f"Decoded batch (clean): {decoded_batch}")


print("\n--- III. Chat Templating (for LLMs - using a Llama-like template for demo) ---")
# Note: bert-base-uncased doesn't have a chat template by default.
# We'll demonstrate with a hypothetical template similar to Llama's or manually construct messages.
# For a real LLM, you'd load its specific tokenizer.

# Simulate a conversation for apply_chat_template
# This will use the default chat template of the Llama tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
print(f"Loaded Llama tokenizer: {llama_tokenizer.__class__.__name__}")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the largest planet in our solar system?"},
    {"role": "assistant", "content": "The largest planet is Jupiter."},
    {"role": "user", "content": "And what about the smallest?"}
]

print("\n1. apply_chat_template(conversation, ...) - Formatting for LLM chat:")
# Tokenize and format the conversation
chat_input_ids = llama_tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_special_tokens=True,
    return_tensors="pt",
    # add_generation_prompt=True # Often used when prompting the model for a response
)
print(f"Chat input IDs:\n{chat_input_ids}")
print(f"Decoded chat template (with special tokens):\n'{llama_tokenizer.decode(chat_input_ids[0], skip_special_tokens=False)}'")
print(f"Decoded chat template (without special tokens, but note: this will remove the formatting too!):\n'{llama_tokenizer.decode(chat_input_ids[0], skip_special_tokens=True)}'")

print("\n2. get_chat_template() - Inspecting the chat template:")
# Only available if the tokenizer has a chat_template defined
if llama_tokenizer.chat_template:
    print(f"Llama's chat template:\n{llama_tokenizer.chat_template}")
else:
    print("Tokenizer does not have a default chat template.")


print("\n--- IV. Vocabulary Management & Saving/Loading ---")

# --- 1. add_special_tokens(special_tokens_dict) ---
print("\n1. add_special_tokens(...) - Adding new special tokens:")
current_vocab_size = len(tokenizer)
print(f"Current vocabulary size: {current_vocab_size}")
num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['<DATE>', '<LOCATION>']})
print(f"Number of new special tokens added: {num_added_tokens}")
print(f"New vocabulary size: {len(tokenizer)}")
print(f"ID for '<DATE>': {tokenizer.convert_tokens_to_ids('<DATE>')}")
print(f"ID for '<LOCATION>': {tokenizer.convert_tokens_to_ids('<LOCATION>')}")

# Test new tokenization
print("\n  Testing tokenization with new special tokens:")
text_with_new_tokens = "I visited <LOCATION>Bengaluru</LOCATION> on <DATE>2025-07-28</DATE>."
encoded_with_new = tokenizer(text_with_new_tokens, return_tensors="pt")
print(f"Text with new tokens: '{text_with_new_tokens}'")
print(f"Encoded IDs: {encoded_with_new['input_ids']}")
print(f"Decoded: '{tokenizer.decode(encoded_with_new['input_ids'][0], skip_special_tokens=False)}'")


# --- 2. add_tokens(new_tokens) ---
print("\n2. add_tokens(...) - Adding new non-special tokens:")
current_vocab_size = len(tokenizer)
print(f"Current vocabulary size: {current_vocab_size}")
num_added_reg_tokens = tokenizer.add_tokens(['customword', 'foobar'])
print(f"Number of new regular tokens added: {num_added_reg_tokens}")
print(f"New vocabulary size: {len(tokenizer)}")
print(f"ID for 'customword': {tokenizer.convert_tokens_to_ids('customword')}")
print(f"ID for 'foobar': {tokenizer.convert_tokens_to_ids('foobar')}")


# --- 3. save_pretrained(save_directory) and from_pretrained(local_path) ---
print("\n3. save_pretrained(...) and load_pretrained(...) - Saving and Loading:")
save_dir = "./my_custom_tokenizer"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(f"Saving tokenizer to '{save_dir}'...")
tokenizer.save_pretrained(save_dir)
print("Tokenizer saved.")

print(f"Loading tokenizer from '{save_dir}'...")
loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
print("Tokenizer loaded successfully from local path.")
print(f"Loaded tokenizer's ID for '<DATE>': {loaded_tokenizer.convert_tokens_to_ids('<DATE>')}") # Should be the same
print(f"Loaded tokenizer's ID for 'customword': {loaded_tokenizer.convert_tokens_to_ids('customword')}") # Should be the same

# Clean up saved directory
import shutil
shutil.rmtree(save_dir)
print(f"Cleaned up directory: '{save_dir}'")


print("\n--- V. Utility Attributes ---")
print(f"Tokenizer type (is_fast): {tokenizer.is_fast}")
print(f"Padding side: {tokenizer.padding_side}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Pad token string: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"CLS token string: '{tokenizer.cls_token}' (ID: {tokenizer.cls_token_id})")
print(f"SEP token string: '{tokenizer.sep_token}' (ID: {tokenizer.sep_token_id})")
print(f"Unknown token string: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")

# --- VI. train_new_from_iterator (Illustrative - not typically run in a simple script) ---
# This is for training a brand new tokenizer, often on very large datasets.
# It requires `tokenizers` library and usually a large text corpus.
print("\n--- VI. train_new_from_iterator (Illustrative - requires `tokenizers` library and data) ---")
print("This function is used for training a new tokenizer from scratch on a custom dataset.")
print("It's computationally intensive and usually involves a custom `text_iterator`.")
print("Example (not executed):")
print("""
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer

# Create a dummy iterator for demonstration
def get_training_corpus():
    return (
        ["This is a sample sentence.", "Another sentence for training."]
        + ["More text to make the vocabulary diverse."] * 100
    )

# Initialize a new tokenizer (e.g., Byte-level BPE)
# new_tokenizer = ByteLevelBPETokenizer()

# Define trainer and special tokens
# trainer = BpeTrainer(vocab_size=500, min_frequency=2, special_tokens=[
#     "<s>", "</s>", "<unk>", "<pad>", "<mask>"
# ])

# Train the tokenizer
# new_tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# You can then convert this `tokenizers` object to a Hugging Face `PreTrainedTokenizerFast`
# from transformers import PreTrainedTokenizerFast
# hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=new_tokenizer)
# hf_tokenizer.save_pretrained("./my_new_tokenizer_dir")
""")

print("\n--- Script Finished ---")