import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from utils import load_pretrained_weights, run_inference
import math
import argparse


# Layer normalization class :-

class LayerNorm(nn.Module):
    def __init__(self, config):
      super().__init__()
      self.eps = config['layer_norm_epsilon']
      self.scale = nn.Parameter(torch.ones(config['n_embd']))
      self.shift = nn.Parameter(torch.zeros(config['n_embd']))

    def forward(self, x):
      mean = x.mean(dim=-1, keepdim=True)
      var = x.var(dim=-1, keepdim=True, unbiased=False)
      norm_x = (x - mean) / torch.sqrt(var + self.eps)
      return self.scale * norm_x + self.shift
    



# GELU activation class :-

class GELU(nn.Module):
  def __init__(self):
      super().__init__()

  def forward(self, x):
      return 0.5 * x * (1 + torch.tanh(
          torch.sqrt(torch.tensor(2.0 / torch.pi)) *
          (x + 0.044715 * torch.pow(x, 3))
      ))
  


# Feed forward neural network module :-

class MLP(nn.Module):
    """
    Position-wise feed-forward network.
    Consists of two linear layers with GELU activation in between.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], config['n_inner'])  # Expansion
        self.c_proj = nn.Linear(config['n_inner'], config['n_embd'])  # Projection back
        self.act = GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x
    



# multi-head attention class :-

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    Causal means tokens can only attend to previous tokens (autoregressive).
    """
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0

        # Linear layer for query, key, value projections
        self.W_query = nn.Linear(config['n_embd'], config['n_embd'])
        self.W_key = nn.Linear(config['n_embd'], config['n_embd'])
        self.W_value = nn.Linear(config['n_embd'], config['n_embd'])

        # Output projection
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])

        # Dropout layers
        self.attn_dropout = nn.Dropout(config['attn_pdrop'])

        self.n_head = config['n_head']
        self.n_embd = config['n_embd']

        # Causal mask to ensure attention only flows backwards
        # This is a lower triangular matrix
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config['n_positions'], config['n_positions']))
            .view(1, 1, config['n_positions'], config['n_positions'])
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Calculate query, key, values for all heads in batch
        k = self.W_key(x)
        q = self.W_query(x)
        v = self.W_value(x)

        # Reshape to separate heads: (B, T, C) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute attention scores: (B, n_head, T, head_size) x (B, n_head, head_size, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask (prevent attending to future tokens)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Normalize with softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values: (B, n_head, T, T) x (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        y = att @ v

        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and dropout
        y = self.c_proj(y)
        return y
    


# Transformer Block :-

class Block(nn.Module):
    """
    A single transformer block consisting of:
    1. Layer normalization
    2. Multi-head self-attention
    3. Residual connection
    4. Layer normalization
    5. Feed-forward network
    6. Residual connection
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attn = CausalMultiHeadSelfAttention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        self.drop_shortcut = nn.Dropout(config["resid_pdrop"])

    def forward(self, x):
        shortcut = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# GPT2 Model :-

class GPT2(nn.Module):
    """
    Complete GPT-2 model implementation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings: maps token IDs to vectors
        self.wte = nn.Embedding(config['vocab_size'], config['n_embd'])

        # Position embeddings: maps positions to vectors
        self.wpe = nn.Embedding(config['n_positions'], config['n_embd'])

        # Embedding dropout
        self.drop = nn.Dropout(config['embd_pdrop'])

        # Stack of transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])

        # Final layer normalization
        self.ln_f = LayerNorm(config)

        # Language modeling head (projects back to vocabulary)
        # In GPT-2, this shares weights with token embeddings
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # Weight sharing: output projection uses same weights as input embeddings
        self.lm_head.weight = self.wte.weight


    def forward(self, idx):
        """
        Forward pass through the model.

        Args:
            idx: Input token IDs of shape (batch_size, sequence_length)
            targets: Target token IDs for training (optional)

        Returns:
            logits: Predicted logits of shape (batch_size, sequence_length, vocab_size)
            loss: Cross-entropy loss if targets are provided
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config['n_positions'], f"Sequence length {t} exceeds max length {self.config['n_positions']}"

        # Get position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # Token embeddings + position embeddings
        tok_emb = self.wte(idx)  # (b, t, n_embd)
        pos_emb = self.wpe(pos)  # (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Pass through all transformer blocks
        for block in self.h:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt', default='My name is SHubham Ojha', type=str, help='Prompt to the model')
    parser.add_argument('--max_new_tokens', default=100, type=int, help='max_new_tokens generated by the model')

    args = parser.parse_args()



    # Configuration Dictionary for GPT2 124M Model :-
    GPT2_CONFIG_124M = {
        'vocab_size': 50257,      # Vocabulary size
        'n_positions': 1024,       # Maximum sequence length (context window)
        'n_embd': 768,            # Embedding dimension (hidden size)
        'n_layer': 12,            # Number of transformer blocks
        'n_head': 12,             # Number of attention heads
        'n_inner': 3072,          # Inner dimension of feed-forward network (4 * n_embd)
        'activation': 'gelu',     # Activation function
        'resid_pdrop': 0.1,       # Residual dropout probability
        'embd_pdrop': 0.1,        # Embedding dropout probability
        'attn_pdrop': 0.1,        # Attention dropout probability
        'layer_norm_epsilon': 1e-5  # Layer normalization epsilon
    }


    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Create and load model
    model = GPT2(GPT2_CONFIG_124M)
    model = load_pretrained_weights(model)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Run inference

    generated_text = run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=0.8,
        top_k=50
    )

    print()
    print()
    print("Generated Text:")
    print()
    print()
    print(generated_text)
    print()





if __name__ == "__main__":
    
    main()














