import os
import pickle
import torch
from model import MiniGPT

# --- Config ---
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
device = 'cpu'
max_new_tokens = 300  # how many characters to generate

data_dir = os.path.join('data', 'shakespeare_char')
out_dir = 'out-shakespeare-char'

# Load encoding
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
stoi = meta['stoi']
vocab_size = meta['vocab_size']

# Load model
model = MiniGPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout).to(device)
model.load_state_dict(torch.load(os.path.join(out_dir, 'model.pt'), map_location=device))
model.eval()

def sample(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # last time step
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx

# Start prompt (can change to anything)
start = "Thou shall"
input_ids = torch.tensor([[stoi.get(c, 0) for c in start]], dtype=torch.long).to(device)
out = sample(model, input_ids, max_new_tokens)
out_str = ''.join([itos[i] for i in out[0].tolist()])

print(out_str)