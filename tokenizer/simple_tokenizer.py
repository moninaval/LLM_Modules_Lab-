# tokenizer/simple_tokenizer.py

def simple_tokenizer(text):
    # Lowercase and split by space
    tokens = text.lower().replace(".", "").replace(",", "").split()
    vocab = {word: idx for idx, word in enumerate(sorted(set(tokens)))}
    token_ids = [vocab[token] for token in tokens]
    return tokens, token_ids, vocab


if __name__ == "__main__":
    input_text = "Attention is all you need and attention is powerful"
    tokens, ids, vocab = simple_tokenizer(input_text)

    print("Tokens:", tokens)
    print("Token IDs:", ids)
    print("Vocab:", vocab)
