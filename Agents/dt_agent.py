import numpy
import math
import torch
from torch import nn
from .agent import Agent

class AttentionHead(nn.Module):
    def __init__(self, num_heads, embedding_dim, masked=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        """
        Each attention head learns a linear projection of the keys,
        queries and values onto a subspace.
        """
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        # Weights for keys, queries and values, in a batch
        self.w_attention = nn.Linear(embedding_dim, num_heads * 3 * embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.w_output = nn.Linear(num_heads * embedding_dim, embedding_dim)

        self.masked = masked

    def forward(self, x):
        """
        x - (batch_size x seq_length x embedding_dim)
        bs - batch size
        sql - sequence length
        edim - embedding dimension  (self.embedding_dim)
        nh - number of attention heads    (self.num_heads)
        """

        # Linearly project onto the learnt subspace.
        projections = self.w_attention(x)   # (bs x sql x edim) -> (bs x sql x nh*3*edim)
        projections = torch.split(projections, self.embedding_dim * self.num_heads, dim=-1)

        # Q, K, V is each (bs x sql x (nh * edim))
        Q, K, V = projections

        # Re-shape each of Q,K,V to (bs x nh x sql x edim)
        Q = Q.contiguous().view((-1, self.num_heads, x.shape[1], self.embedding_dim))
        K = K.contiguous().view((-1, self.num_heads, x.shape[1], self.embedding_dim))
        V = V.contiguous().view((-1, self.num_heads, x.shape[1], self.embedding_dim))

        # Attention
        # (bs x nh x sql x edim) * (bs x nh x edim x sql) -> (bs x nh x sql x sql)
        compatibility = Q @ torch.transpose(K, -1, -2)
        scaled_compatibility = torch.divide(compatibility, math.sqrt(self.embedding_dim))
        
        if self.masked:
            mask = torch.ones_like(scaled_compatibility) * float('-inf')
            mask = torch.triu(mask, 1)
            masked_compatibility = scaled_compatibility + mask
            attention_scores = self.softmax(masked_compatibility)
        else:
            attention_scores = self.softmax(scaled_compatibility)

        # Output
        output = attention_scores @ V   # (bs x nh x sql x sql) * (bs x nh x sql x edim) -> (bs x nh x sql x edim)
        output = output.contiguous().view((-1, x.shape[1], self.num_heads * self.embedding_dim))
        output = self.w_output(output)

        return output


class GPTBlock(nn.Module):
    def __init__(self, num_heads, embedding_dim, ff_dim, dropout, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention_block = AttentionHead(num_heads, embedding_dim, args, kwargs)
        self.ln1 = nn.LayerNorm(embedding_dim)
        
        # Feed forward network
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU,
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout)
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward through the masked multi-head
        attn_out = self.attention_block.forward(x)

        # Skip connection
        x = self.dropout(attn_out) + x

        # Layer Norm
        x = self.ln1(x)

        # Feed forward 
        ff_out = self.feedforward(x)
        
        # Skip conncetion 
        x = ff_out + x

        # Layer Norm
        x = self.ln2(x)

        return x
        

class DecisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class DTAgent(Agent):
    def __init__(self, env):
        super(DTAgent, self).__init__(env)
        self.model = DecisionTransformer()

    def act(self, state):
        return self.env.action_space.sample()
