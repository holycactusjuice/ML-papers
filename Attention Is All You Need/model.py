import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

# dimension values
batch_size = 4
block_size = 8
n_embd = 32
n_head = 4
d_k = 32
d_v = d_k

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
val_data = data[:n]

def get_batch(split):
    # need to get batch_size number of samples
    # choose from the specified split
    split_data = train_data if split == 'train' else val_data

    # get batch_size number of starting indicies
    ix = torch.randint(len(split_data) - block_size, (batch_size,))

    print(ix.shape)

    # for each i in ix,
    # i is the starting index
    # so we take from i to i + block_size
    x = torch.stack([split_data[i:i+block_size] for i in ix])
    # y is just the element after each x
    y = torch.stack([split_data[i+1:i+block_size+1] for i in ix])

    return x, y

class Head(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        
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
        weights = weights.softmax(dim=-1)
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
    
class Transformer(nn.Module):

    def __init__(self, vocab_size, n_embd, n_head, block_size, n_encoder):

        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        # LEAVING OUT THE POSITION ENCODING FOR NOW
        self.blocks = nn.Sequential([Block(n_embd, n_head, block_size) for _ in range(n_encoder)])
        # After the blocks, the dim is (B, T, n_embd)
        # We want to project this to probabiliites for the next character, so we need (n_embd, vocab_size)
        # ----- QUESTION: Why is the T dimension still here and what does it represent? -----
        # The T dimension is the time dimension. Recall that we are passing in multiple examples with one sequence.
        # This is because we are simultaneously predicting the 2nd char from the 1st, the 3rd from the 1st and 2nd, etc.
        # ----- ANSWER: We have multiple time-dependent examples in one input sequence.
        self.projection = nn.Linear(n_embd, vocab_size)

    def forward(self, X):
        # (B, T)
        X_embd = self.embedding_table(X)
        # (B, T, n_embd)
        X_blocks = self.blocks(X_embd)
        # (B, T, n_embd)
        logits = self.projection(X_blocks)
        # (B, T, vocab_size)
        # probs = logits.softmax(dim=-1)
        # (B, T, vocab_size)
        return logits