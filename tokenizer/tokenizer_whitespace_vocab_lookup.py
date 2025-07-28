# Basic Whitespace Tokenizer + Vocab Lookup

# Sample vocab
vocab = {
    "my": 1,
    "name": 2,
    "is": 3,
    "naval": 4,
    "[UNK]": 0
}

def tokenize(text):
    return text.strip().lower().split()

def encode(tokens, vocab):
    return [vocab.get(token, vocab["[UNK]"]) for token in tokens]

def decode(token_ids, vocab):
    reverse_vocab = {id: word for word, id in vocab.items()}
    return " ".join([reverse_vocab.get(tid, "[UNK]") for tid in token_ids])

if __name__ == "__main__":
    text = "My name is Kaval"
    tokens = tokenize(text)
    token_ids = encode(tokens, vocab)
    decoded = decode(token_ids, vocab)

    print("Input Text: ", text)
    print("Tokens:     ", tokens)
    print("Token IDs:  ", token_ids)
    print("Decoded:    ", decoded)
# tokenizer_whitespace_vocab_lookup.py
