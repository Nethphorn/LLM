import os
import requests
import pickle

# --- Step 1: Download the Data ---
# This is the "Tiny Shakespeare" dataset, which contains all of Shakespeare's works.
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_file_path):
    print("Downloading Tiny Shakespeare dataset...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

print("Reading the text...")
with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset length: {len(text)} characters")

# --- Step 2: Extract the Vocabulary ---
# A vocabulary is just the unique characters that appear in the text.
# We sort them to ensure a consistent mapping every time we run.
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters (Vocabulary Size): {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")

# --- Step 3: Create the Tokenizer ---
# We need dictionaries that map a character to an integer (encode) and vice-versa (decode)
stoi = { ch:i for i,ch in enumerate(chars) } # stoi = "string to integer"
itos = { i:ch for i,ch in enumerate(chars) } # itos = "integer to string"

# lambda functions to do the actual mapping
encode = lambda s: [stoi[c] for c in s] # Takes a string, maps each char to an integer
decode = lambda l: ''.join([itos[i] for i in l]) # Takes a list of integers, maps to a string

print("\nLet's test the tokenizer!")
test_string = "Hello, LLM!"
test_encoded = encode(test_string)
print(f"'{test_string}' encodes to: {test_encoded}")
print(f"Decoding back: '{decode(test_encoded)}'")

# --- Step 4: Tokenize the entire dataset and save the mappings ---
# Now, we are actually encoding the entire works of Shakespeare into integers!
print("\nEncoding the entire dataset...")
encoded_data = encode(text)

# We will save the vocab mapping so we don't have to recalculate it later when we want to test our model
vocab_filepath = os.path.join(os.path.dirname(__file__), 'vocab.pkl')
with open(vocab_filepath, 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

print(f"Saved vocabulary to {vocab_filepath}")
print("Lesson 1 complete! We have numbers instead of text.")
