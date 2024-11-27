import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, dropout=0.1):
        """
        Initializes the Positional Encoding module.

        Args:
            embedding_dim: The dimension of the model (size of embeddings).
            max_len: The maximum sequence length.
            dropout: Dropout rate for regularization.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of [max_len, d_model] with positional encodings
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            -(math.log(10000.0) / embedding_dim)
            )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for applying positional encoding.

        Args:
            x: The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor of same shape as x with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
