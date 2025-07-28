# tokenizer_gpt2_demo.py
import tiktoken

# Load the GPT-4 tokenizer (same tokenizer used in GPT-3.5/4)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Input text (your message)
text = """Hello naval praSAD, To ensure a smooth filing experience once the portal is live, we recommend completing the following steps on your dashboard: Step 1 – Link PAN & Pre-fill • Navigate to Filing Status > Link PAN & Pre-fill • Enter your PAN and Date of Birth • Choose an OTP method to verify and complete the linking (This registers your details with the department.) Step 2 – Authorize Pre-fill Access • Re-enter OTP to complete authorization which comes to both your email and mobile number Please refer to the video for PAN Linking https://www.youtube.com/watch?v=MKUijIzQjF8 Regards, Clear Tax Team For any assistance, feel free to contact us at: 080-67458776 (Available every day from 8:00 AM to 12:00 AM) Email us at: services@cleartax.in We’re here to help!"""

# Tokenize
tokens = tokenizer.encode(text + "<|endoftext|>", allowed_special={"<|endoftext|>"})

# Display result
print("Total Tokens:", len(tokens))
print("Tokens:", tokens)
print("Decoded:", tokenizer.decode(tokens))
