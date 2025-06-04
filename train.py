import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from model import MiniGPT

# --- Configs (minimal, can be parsed from args or set here) ---
block_size = 64
batch_size = 12
n_layer = 4
n_head = 4
n_embd = 128
max_iters = 2000
eval_iters = 20
log_interval = 1
dropout = 0.0
learning_rate = 1e-3
device = 'cpu'

data_dir = os.path.join('data', 'shakespeare_char')
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

def load_bin(split):
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode='r')

train_data = load_bin('train')
val_data = load_bin('val')

def get_batch(data):
    ix = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = torch.from_numpy(np.stack([data[i:i+block_size] for i in ix])).long()
    y = torch.from_numpy(np.stack([data[i+1:i+1+block_size] for i in ix])).long()
    return x.to(device), y.to(device)

model = MiniGPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for iter in range(1, max_iters+1):
    model.train()
    x, y = get_batch(train_data)
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter % log_interval == 0:
        print(f"step {iter}: train loss {loss.item():.4f}")
    if iter % eval_iters == 0:
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch(val_data)
            val_logits = model(x_val)
            val_loss = loss_fn(val_logits.view(-1, vocab_size), y_val.view(-1))
            print(f"step {iter}: val loss {val_loss.item():.4f}")

os.makedirs('out-shakespeare-char', exist_ok=True)
torch.save(model.state_dict(), os.path.join('out-shakespeare-char', 'model.pt'))
print("Training done. Model saved.")