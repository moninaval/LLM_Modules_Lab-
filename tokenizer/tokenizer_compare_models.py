from transformers import AutoTokenizer
import importlib.util

MODEL_INFO = {
    "bert-base-uncased": "WordPiece Tokenizer (BERT)",
    "gpt2": "Byte-Level BPE Tokenizer (GPT-2)",
    "roberta-base": "Byte-Level BPE Tokenizer (RoBERTa)",
    "distilbert-base-uncased": "WordPiece Tokenizer (DistilBERT)",
    "xlm-roberta-base": "SentencePiece Tokenizer (XLM-RoBERTa)",
}

def show_tokenizer_output(model_name, text):
    print(f"\n=== {model_name} ===")
    print(f"Tokenizer Type: {MODEL_INFO[model_name]}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded = tokenizer.decode(token_ids)
    print("Tokens:    ", tokens)
    print("Token IDs: ", token_ids)
    print("Decoded:   ", decoded)

def show_tiktoken_output(text):
    try:
        import tiktoken
    except ImportError:
        print("\n=== tiktoken (OpenAI tokenizer) ===")
        print("tiktoken not installed. Run: pip install tiktoken")
        return

    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    tokens = [enc.decode_single_token_bytes(tid).decode("utf-8", errors="replace") for tid in token_ids]
    decoded = enc.decode(token_ids)

    print("\n=== tiktoken (OpenAI GPT-4 tokenizer) ===")
    print("Tokenizer Type: Byte-Level BPE (tiktoken, cl100k_base)")
    print("Tokens:    ", tokens)
    print("Token IDs: ", token_ids)
    print("Decoded:   ", decoded)

if __name__ == "__main__":
    input_text = "monkeybananafruitxyz"
    for model in MODEL_INFO:
        show_tokenizer_output(model, input_text)

    show_tiktoken_output(input_text)
