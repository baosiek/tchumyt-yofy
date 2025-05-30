import torch
import torch.nn as nn

from typing import Any, Dict

from llm.llm.architecture.gpt.positional_encoding import PositionalEncoding
from llm.llm.architecture.gpt.transformer_block_v1 import TransformerBlockV1


class GPTModelV1(nn.Module):
    def __init__(
            self,
            cfg: Dict[str, Any]
    ):
        """
        Initializes the GPTModel module.

        Args:
            cfg: Dict[str, Any] -> The dictionary with the configuration file.
        """
        super(GPTModelV1, self).__init__()

        # The model's layers
        # The token embedding layer
        self.token_embedding: nn.Embedding = nn.Embedding(
            cfg['vocabulary_size'],
            cfg['embedding_dimension']
        )

        # The positional embedding layer
        self.positional_embedding: PositionalEncoding = PositionalEncoding(
            embedding_dim=cfg['embedding_dimension'],
            dropout=cfg['drop_rate']
        )

        # The transformer block layer
        self.transformer_blocks: nn.Sequential = nn.Sequential(
            *[TransformerBlockV1(cfg=cfg) for _ in range(
                cfg['number_layers']
            )]
        )

        # The normalization layer
        self.norm_layer: nn.LayerNorm = nn.LayerNorm(
            cfg['embedding_dimension']
        )

        # The output layer
        self.output: nn.Linear = nn.Linear(
            cfg['embedding_dimension'],
            cfg['vocabulary_size'],
            bias=cfg['bias']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for applying GPTModel.

            Args:
                x: torch.Tensor -> The input tensor of shape
                (batch_size, seq_len, d_model).

            Returns:
                torch.Tensor of same shape as x with token embedding and
                positional encoding added
        """

        x = self.token_embedding(x)
        x = self.positional_embedding(x)
        x = self.transformer_blocks(x)
        x = self.norm_layer(x)
        return self.output(x)
