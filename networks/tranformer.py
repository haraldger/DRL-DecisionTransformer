import numpy 
import math
import torch
from torch import nn 
from Agents.agent import Agent
from networks.resnet import resnet18, resnet34, resnet101

class AttentionHead(nn.Module):
    def __init__(
            self, 
            num_heads, 
            embedding_dim, 
            masked=False, 
            *args, 
            **kwargs
    ) -> None:
     
        super(AttentionHead, self).__init__(*args, **kwargs)

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
    def __init__(
            self,
            num_heads,
            embedding_dim, 
            ff_dim, 
            dropout, 
            *args, 
            **kwargs
    ) -> None:
        
        super(GPTBlock, self).__init__(*args, **kwargs)
        self.attention_block = AttentionHead(num_heads, embedding_dim, masked=True, *args, **kwargs)
        self.ln1 = nn.LayerNorm(embedding_dim)
        
        # Feed forward network
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
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
    def __init__(
            self, 
            num_blocks, 
            num_heads, 
            embedding_dim, 
            dropout, 
            max_ep_len, 
            img_channels=1,
            act_dim=9, 
            *args, 
            **kwargs
    ) -> None:

        super(DecisionTransformer, self).__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For now, like gpt2, use a ff size of 4*embedding_dim
        ff_dim = 4*embedding_dim

        self.embedding_dim = embedding_dim
        self.act_dim = act_dim

        # Embeddings and Encodings
        self.embed_timestep = nn.Embedding(max_ep_len, embedding_dim)
        self.embed_action = nn.Embedding(act_dim, embedding_dim)
        self.embed_return = nn.Linear(1, embedding_dim)
        self.embed_state = resnet101(in_channels=img_channels)


        self.embed_ln = nn.LayerNorm(embedding_dim)

        # input shape to decoder block:
        # (batch, sequencelength, embeddingdim)

        # GPT blocks
        self.gpt_blocks = nn.ModuleList([
            GPTBlock(num_heads, embedding_dim, ff_dim, dropout, *args, **kwargs)
            for _ in range(num_blocks)
        ])

        # output prediction layers
        # self.predict_state = nn.Linear(embedding_dim, self.state_dim)
        # self.predict_action = nn.Sequential(
        #     nn.Linear(embedding_dim, self.act_dim) + nn.Tanh() 
        # )
        # self.predict_return = nn.Linear(embedding_dim, 1)

        # NOTE: atm just using action prediction, use the output of the gpt blocks and a few more layers
        self.predict_action = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, self.act_dim),
            nn.Softmax()
        )


    def forward(
            self, 
            states, 
            actions, 
            returns_to_go, 
            timesteps
    ):
        # NOTE: actions should be in one-hot rep based on act_dim

        batch_size, seq_length, channels, y, x = states.shape

        print(torch.cuda.memory_reserved())
        # Embed each modality with a different head
        time_embeddings = self.embed_timestep(timesteps).reshape(batch_size, seq_length, self.embedding_dim)
        action_embeddings = self.embed_action(actions).reshape(batch_size, seq_length, self.embedding_dim)
        returns_embeddings = self.embed_return(returns_to_go).reshape(batch_size, seq_length, self.embedding_dim)
        
        print(torch.cuda.memory_reserved())
        # merge seq_length and batch_size dims for resenet
        state_merged = states.reshape(-1, channels, y, x)
        state_embeddings = self.embed_state(state_merged).reshape(batch_size, seq_length, self.embedding_dim)

        # time embeddings 
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack inputs
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.embedding_dim)

        stacked_data = self.embed_ln(stacked_inputs)

        # Pass through GPT Layers
        for block in self.gpt_blocks:
            stacked_data = block(stacked_data)

        # # Reshape so that second dim corresponds to the original:
        # # returns (0), states (1), or actions (2)
        stacked_transformer_output = stacked_data.reshape(batch_size, seq_length, 3, self.embedding_dim).permute(0, 2, 1, 3)

        # # get predictions
        # return_preds = self.predict_return(x[:,0])  # predict next return given state and action
        # state_preds = self.predict_state(x[:,1])    # predict next state given state and action
        action_preds = self.predict_action(stacked_transformer_output[:,2])  # predict next action given state

        return action_preds
