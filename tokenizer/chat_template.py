from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2") # Example Llama 2 model

messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris, of course!"},
    {"role": "user", "content": "And what about Germany?"}
]

# apply_chat_template handles the insertion of [INST], [/INST], <s>, </s> and tokenization
chat_input = tokenizer.apply_chat_template(messages, tokenize=True, add_special_tokens=True, return_tensors="pt")

# You'll see the full tokenized sequence, including the tokens for [INST] and [/INST]
print(tokenizer.decode(chat_input[0]))