import torch
import pickle
import os
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4
block_size = 8

vocab_filepath = os.path.join(os.path.dirname(__file__), 'vocab.pkl')

with open(vocab_filepath, 'rb') as f: vocab_data = pickle.load(f)

stoi = vocab_data['stoi']
itos = vocab_data['itos']
vocab_size = vocab_data['vocab_size']

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_file_path, 'r', encoding='utf-8') as f: text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data tensor shape and type: {data.shape}, {data.dtype}")

# Split into training and validation sets
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


xb, yb = get_batch('train')
print('\ninputs (x):')
print(xb.shape)
print(xb)
print('\ntargets (y):')
print(yb.shape)
print(yb)

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
