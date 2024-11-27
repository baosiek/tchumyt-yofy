import torch
import torch.nn as nn

from llm.llm.architecture.positional_encoding import PositionalEncoding
from llm.llm.architecture.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super(GPTModel, self).__init__()

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

        # The dropout layer
        self.dropout: nn.Dropout = nn.Dropout(
            cfg['drop_rate']
        )

        # The transformer block layer
        self.transformer_blocks: nn.Sequential = nn.Sequential(
            *[TransformerBlock(cfg=cfg) for _ in range(
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

        # Token embedding
        token_embedding: torch.Tensor = self.token_embedding(x)

        # Positinal encoding
        positional_encoding: torch.Tensor = self.positional_embedding(x)

        x = token_embedding + positional_encoding
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm_layer(x)
        return self.output(x)
