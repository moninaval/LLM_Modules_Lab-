from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a tokenizer and model (e.g., for a Llama-like model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Prepare input with chat template (includes <s>, [INST], [/INST])
messages = [{"role": "user", "content": "What is the capital of France?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_special_tokens=True, return_tensors="pt")

# Generate output from the model
# Model will generate the answer and then typically the </s> token
output_ids = model.generate(input_ids, max_new_tokens=50, num_return_sequences=1)

# Decode the output
# This is the key step where special tokens are removed for display
decoded_output_with_special = tokenizer.decode(output_ids[0], skip_special_tokens=False)
decoded_output_without_special = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Output with special tokens:")
print(decoded_output_with_special)
# Expected: "<s> [INST] What is the capital of France? [/INST] Paris.</s>" (or similar)

print("\nOutput without special tokens (clean for display):")
print(decoded_output_without_special)
# Expected: "What is the capital of France? Paris." (or similar, stripping the instruction part too if it's within the generated range)