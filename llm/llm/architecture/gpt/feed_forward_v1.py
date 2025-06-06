import torch
import torch.nn as nn

from typing import Dict


class FeedForwardV1(nn.Module):
    def __init__(self, cfg: Dict):
        super(FeedForwardV1, self).__init__()

        embedding_dimension: int = cfg['embedding_dimension']
        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                4 * embedding_dimension
            ),
            nn.GELU(),
            nn.Linear(
                4 * embedding_dimension,
                embedding_dimension
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
