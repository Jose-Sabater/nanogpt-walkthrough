import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
torch.manual_seed(1337)
batch_size = 64  # how many independent sequences  in parallel?
block_size = 256  # also called context length in some sets // what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
dropout = 0.2
n_head = 6
n_layer = 6


with open("input.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # encoder
itos = {i: ch for ch, i in stoi.items()}  # decoder

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
# print(encode("Hi there"))
# print(decode(encode("Hi There")))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data[:1000])
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


######################################################
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint((len(data) - block_size), (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# xb, yb = get_batch("train")

# # lets check 1 batch contents
# print("input:", xb, "\nOutput:", yb)
# print(xb.shape)
# ######################################################


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,C) * (B,C,T)-> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)

        out = wei @ v  # (B,T,T) @ (B,T,C) -> (B,T,C)

        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    # Feed forward is so to say, allowing the different attentions captured to think, token by token

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: communication followed by computation"""

    # Here we basically intersperse the coomuncation and the computation

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the nr of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (Batch,    Time,Channel 4x8x65
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # Apply 1 head self attention
        logits = self.lm_head(x)  # (B,T,vocab_size)
        if targets == None:
            loss = 0
        # pytorch cross entropy wants B,C,T instead of BTC
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """The function of this method is to genete next tokens"""
        for _ in range(max_new_tokens):
            # crop idx to the last blocksize
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            # Focus only on the last time step (the last element of each group)
            logits = logits[:, -1, :]  # Becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)  # Cat concatenates

        return idx


model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# idx = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))
