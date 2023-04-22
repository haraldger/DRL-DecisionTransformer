import numpy
import math
import torch 
from torch import nn
import Agent

class MaskedAttentionHead(nn.Module):
    def __init__(self, num_heads, embedding_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        """
        Each attention head learns a linear projection of the keys,
        queries and values onto a subspace.
        """
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        # Weights for keys, queries and values, in a batch
        self.w_attention = nn.Linear(embedding_dim, 3*embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x - (seq_length x embedding_dim)
        """

        # Linearly project onto the learnt subspace.
        Q = self.w_q(x)     # (seq_length x k_dim)
        K = self.w_k(x)     # (seq_length x k_dim)
        V = self.w_v(x)     # (seq_length x v_dim)

        # Attention
        compatibility = Q @ K.T     # (seq_length x seq_length)
        scaled_compatibility = torch.divide(compatibility, math.sqrt(self.k_dim))
        attention_scores = self.softmax(scaled_compatibility)
        output = attention_scores @ V   # (seq_length x v_dim)

        return output


class DecisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class DTAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.model = DecisionTransformer()

    def act(self, observation, reward, done):
        raise NotImplementedError