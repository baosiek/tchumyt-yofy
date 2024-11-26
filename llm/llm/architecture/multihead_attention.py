import torch
import torch.nn as nn

from llm.llm.architecture.attention import Attention


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 dropout: float,
                 qkv_bias: bool = False
                 ) -> None:
        '''
        Inputs:
            embedding_dim: The embedding dimension of the model
            num_heads: the number of heads
            dropout: a float in the range [0, 1]
            qkv_bias: flag that sets on or off working with bias.
            Defaults to false.
        '''
        super().__init__()

        '''
        Asserts that the embed_dim is divisible by num_heads as all
        heads ought to have the same dimension
        '''
        assert embedding_dim % num_heads == 0, \
            f'''model embedding dimension
             {embedding_dim} must be divisible by {num_heads}'''

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.dropout: float = dropout

        # dimension of each head
        self.head_dim = embedding_dim // num_heads

        # Initializing query weight matrix
        self.W_query: nn.Linear = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=qkv_bias
        )

        # Initializing key weight matrix
        self.W_key: nn.Linear = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=qkv_bias
        )

        # Initializing value weight matrix
        self.W_value: nn.Linear = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=qkv_bias
        )

        # Initializing projection layer
        self.o_proj: nn.Linear = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=qkv_bias
        )

        # Initializing dropout and projection layers
        self.dropout_layer: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        queries: torch.Tensor = self.W_query(x)
        queries = queries.reshape(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )
        queries = queries.permute(0, 2, 1, 3)

        keys: torch.Tensor = self.W_key(x)
        keys = keys.reshape(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )
        keys = keys.permute(0, 2, 1, 3)

        values: torch.Tensor = self.W_value(x)
        values = values.reshape(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )
        values = values.permute(0, 2, 1, 3)

        context_vector: torch.Tensor = Attention(
            query=queries,
            key=keys,
            value=values)
        context_vector = context_vector.permute(0, 2, 1, 3)
        context_vector = context_vector.reshape(
            batch_size, 
            seq_length,
            self.embed_dim)
        context_vector = self.dropout_layer(context_vector)

        output: torch.Tensor = self.o_proj(context_vector)

        return output