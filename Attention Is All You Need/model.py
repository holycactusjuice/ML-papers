import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from collections import OrderedDict
from matplotlib import pyplot as plt
import time

# Hyperparameters
batch_size = 4
block_size = 8
n_embd = 32
n_head = 4
d_k = 32
d_v = d_k
n_block = 4
dropout = 0.0
learning_rate = 1e-3
max_iters = 100000

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

encode = lambda s : [stoi[ch] for ch in s]
decode = lambda l : ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # need to get batch_size number of samples
    # choose from the specified split
    split_data = train_data if split == 'train' else val_data

    # get batch_size number of starting indicies
    ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))

    # for each i in ix,
    # i is the starting index
    # so we take from i to i + block_size
    x = torch.stack([split_data[i:i+block_size] for i in ix])
    # y is just the element after each x
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y

class Head(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.d_k = n_embd // n_head
        
        assert (self.n_head * self.d_k == self.n_embd), "Embedding size not divisible by number of heads"

        self.query = nn.Linear(self.n_embd, self.d_k)
        self.key = nn.Linear(self.n_embd, self.d_k)
        self.value = nn.Linear(self.n_embd, self.d_k)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # (T, T)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        B, T, C = X.shape
        # essentially doing X @ W_Q
        # dims are (B, T, C) @ (T, n_embd) -> (B, T, n_embd)
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        # ----- QUESTION: Which dimensions do we transpose? -----
        # The shape of K is (B, T, n_embd)
        # We want to matrix multiply with Q, which is also (B, T, n_embd)
        # We ignore the batch dimension, so we transpose T and n_embd
        # ----- ANSWER: The last two dimensions, -2 and -1
        weights = Q @ K.transpose(-2, -1) # we want to transpose the last two dimensions
        weights /= self.d_k**0.5

        # dims are now (B, T, T)

        # Mask the weights to be lower-triangular
        # This prevents the model from attending to future tokens
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # ----- QUESTION: Which dimension do we want to softmax over? -----
        # The shape of out is currently (B, T, T)
        # - B is the batch dimension
        # - since we produced this through Q @ K^T,
        #     - the first T represents the query tokens (since it came from Q)
        #     - the second T represents the key tokens (since it came from K)
        # we want the values in the key dimension to become probabilities,
        # so we softmax over the second T
        # ----- ANSWER: The key dimension, -1 -----
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ V

        # dims are now (B, T, d_k)

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head

        self.heads = nn.ModuleList([Head(n_embd, n_head, block_size) for _ in range(n_head)])
        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, X):
        out = torch.cat([head(X) for head in self.heads], dim=-1)
        out = self.projection(out)
        return out
    
class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()

        # LayerNorm includes scale (gamma) and shift (beta) parameters
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, X):
        # ----- QUESTION: Which dimension do we want to normalize across? -----
        # Our dims are (B, T, n_embd)
        # - B is the number of batches
        # - T is the number of tokens in each sequence
        # - n_embd is the embedding dimension, or the number of features that represents each token
        # We want to normalize across the feature dimension to normalize the activations of each token independently.
        # This ensures that each token has a similar distrbution of activations, regardless of its position in the sequence.
        # ----- ANSWER: The feature dimension, -1
        X = X - X.mean(dim=-1, keepdim=True)
        X = X / torch.sqrt((X.var(dim=-1, keepdim=True) + 1e-5))
        X = self.gamma * X + self.beta
        
        return X
    
class FeedForward(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        hidden_dim = 4 * input_dim

        self.layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_dim, hidden_dim)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_dim, output_dim))
        ]))

    def forward(self, X):
        out = self.layers(X)
        return out
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        self.attention = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln1 = LayerNorm(n_embd)
        self.feedforward = FeedForward(n_embd, n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, X):
        X = X + self.attention(self.ln1(X))
        X = X + self.feedforward(self.ln2(X))
        return X
    
class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, block_size, n_block):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Needs same dimensions as input after token embedding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_block)]) # Need to unpack list
        # After the blocks, the dim is (B, T, n_embd)
        # We want to project this to probabiliites for the next character, so we need (n_embd, vocab_size)
        # ----- QUESTION: Why is the T dimension still here and what does it represent? -----
        # The T dimension is the time dimension. Recall that we are passing in multiple examples with one sequence.
        # This is because we are simultaneously predicting the 2nd char from the 1st, the 3rd from the 1st and 2nd, etc.
        # ----- ANSWER: We have multiple time-dependent examples in one input sequence.
        self.projection = nn.Linear(n_embd, vocab_size)

    def forward(self, X):
        B, T = X.shape
        # (B, T)
        X_embd = self.token_embedding_table(X) + self.position_embedding_table(torch.arange(T, device=device))
        # (B, T, n_embd)
        X_blocks = self.blocks(X_embd)
        # (B, T, n_embd)
        logits = self.projection(X_blocks)
        # (B, T, vocab_size)
        # probs = logits.softmax(dim=-1)
        # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # Keep in mind that idx is a sequence of encoded values
        for _ in range(max_new_tokens):
            # We can only take up to block_size, so crop idx
            # Don't forget about the batch dimension, even though it's 1
            idx_cropped = idx[:, -block_size:]
            logits = self(idx_cropped)
            # We will sample from the last time step
            # Keep in mind that logits is now (B, T, C)
            logits = logits[:, -1, :]
            # We want to softmax over the probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample
            next_idx = torch.multinomial(probs, num_samples=1)
            # idx has dim (B, T)
            # next_idx has dim (B, 1)
            # So we need to concatenate over the last dim
            idx = torch.cat((idx, next_idx), dim=-1)
        return idx
    
if __name__ == "__main__":
    
    model = LanguageModel(vocab_size, n_embd, n_head, block_size, n_block)
    model.to(device)

    # create a PyTorch optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    start_time = time.time()
    for iter in range(1, max_iters + 1):
        xb, yb = get_batch('train')
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), yb.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 1000 == 0:
            print(f"iter {iter}/{max_iters}: loss {loss.item()}")

        if iter % 5000 == 0:
            out = model.generate(context, max_new_tokens=500)
            print(decode(out[0].tolist()))
            print('---')
            
    # Save model
    torch.save(model.state_dict(), 'model.pth')
            
    print(f"Average time per 1000 iterations: {(time.time() - start_time) / (max_iters / 1000)} seconds")
