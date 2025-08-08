# inspect_tokenizer.py

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from custom_tokenizer import CustomBPETokenizer

# 1. Load the tokenizer you just trained and saved
print("--- Loading tokenizer ---")
tokenizer_path = 'trainedBPE/tokenizer.json'
tokenizer = CustomBPETokenizer.load(tokenizer_path)
print("Tokenizer loaded successfully.\n")

# 2. Check Vocabulary and Special Tokens
print("--- Vocabulary and Special Token Inspection ---")
vocab_size = tokenizer.get_vocab_size()
pad_id = tokenizer.get_pad_token_id()
unk_id = tokenizer.get_unk_token_id()

print(f"Vocabulary Size: {vocab_size}")
print(f"PAD token ID: {pad_id}")
print(f"UNK token ID: {unk_id}")

# Let's see some of the actual learned vocabulary
# The get_vocab() method returns a dictionary of {token: id}
learned_vocab = tokenizer.tokenizer.get_vocab()
print("\nSample of learned vocabulary:")
# Print the first 20 learned tokens/subwords
for i, (token, token_id) in enumerate(learned_vocab.items()):
    if i >= 20: break
    print(f"  ID: {token_id}, Token: '{token}'")
print("...\n")


# 3. Test Encoding and Decoding with various sentences
print("--- Encoding/Decoding Tests ---")
sentences_to_test = [
    "Hello this is a test.",                     # Pure English
    "नमस्ते यो एउटा टेस्ट हो।",               # Pure Nepali
    "Mix of English and नेपाली words.",          # Mixed sentence
    "A word not in vocab: ZYXWVU",             # Out-of-vocabulary word
    "Punctuation! And numbers 123?"              # Punctuation and numbers
]

for sentence in sentences_to_test:
    print(f"Original: '{sentence}'")
    
    # Encode
    encoded_ids = tokenizer.encode(sentence)
    print(f" -> Encoded IDs: {encoded_ids}")
    
    # Check for UNK tokens
    if unk_id in encoded_ids:
        print(f"   -> WARNING: Found UNK token (ID: {unk_id})!")

    # Decode
    decoded_text = tokenizer.decode(encoded_ids)
    print(f" -> Decoded Text: '{decoded_text}'")
    
    # Verify round-trip
    assert sentence == decoded_text
    print("   -> Round-trip successful.\n")

print("--- Inspection Complete ---")