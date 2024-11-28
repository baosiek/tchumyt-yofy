import torch
import torch.nn as nn

from typing import Dict, Any

from llm.llm import logger
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
        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=cfg['embedding_dimension'],
            num_heads=cfg['number_heads'],
            dropout=cfg['drop_rate'],
            bias=cfg['bias'],
            batch_first=True
            )

        # Initializes the feed forward layer
        self.ff: FeedForward = FeedForward(cfg=cfg)

        # Initialized the normalization layers
        self.norm1: nn.LayerNorm = nn.LayerNorm(cfg['embedding_dimension'])
        self.norm2: nn.LayerNorm = nn.LayerNorm(cfg['embedding_dimension'])

        # Initializes the dropout shortcut
        self.drop_shortcut: nn.Dropout = nn.Dropout(cfg['drop_rate'])

        # Flag whether to set the attention_mask flag
        self.attention_flag: bool = cfg['attention_mask']

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass for applying transformer block.

        Args:
            - x: torch.Tensor -> The input tensor of shape
            (batch_size, seq_len, d_model).
            - attention_mask: bool -> sets the flag to create and pass
            the attention mask to the multihead attention layer.

        Returns:
            torch.Tensor of same shape as x with attention added
        """

        logger.debug(f"\tx input shape: {x.shape}")

        # Store the input value
        shortcut: torch.Tensor = x

        # Sets the attention mask if required.
        attention_mask: torch.Tensor = None
        if self.attention_flag:
            sequence_length: int = x.shape[1]
            attention_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length),
                diagonal=1
            ).bool()
            logger.debug(f"\tAttention mask shape: {attention_mask.shape}")

        # Applies the transformer block flow
        x = self.norm1(x)
        logger.debug(f"\tx after normalization shape: {x.shape}")
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        logger.debug(f"\tx after attention shape: {x.shape}")
        x = self.drop_shortcut(x)
        x = x + shortcut
        logger.debug(f"\tx after first shortcut shape: {x.shape}")

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        logger.debug(f"\tx after feed forward shape: {x.shape}")
        x = self.drop_shortcut(x)
        x = x + shortcut
        logger.debug(f"\tx after second shortcut shape: {x.shape}")
        return x
