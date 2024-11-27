import torch
import torch.nn as nn

from typing import Dict, Any

from llm.llm.architecture.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initializes the Transformer Block module.

        Args:
            cfg: Dict[str, Any] -> The dictionary with the configuration file.
        """
        super(TransformerBlock, self).__init__()

        # Initializes the Multihead attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg['embedding_dimension'],
            num_heads=cfg['number_heads'],
            dropout=cfg['drop_rate'],
            bias=cfg['bias']
            )

        # Initializes the feed forward layer
        self.ff = FeedForward(cfg=cfg)

        # Initialized the normalization layers
        self.norm1: nn.LayerNorm = nn.LayerNorm(cfg['embedding_dimension'])
        self.norm2: nn.LayerNorm = nn.LayerNorm(cfg['embedding_dimension'])

        # Initializes the dropout shortcut
        self.drop_shortcut: nn.Dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass for applying transformer block.

        Args:
            x: torch.Tensor -> The input tensor of shape
            (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor of same shape as x with attention added
        """

        # Store the input value
        shortcut: torch.Tensor = x

        # Applies the transformer block flow
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
