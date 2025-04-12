import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int):
        super(FeedForward, self).__init__()

        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim * 4
            ),
            nn.GELU(),
            nn.Linear(
                embedding_dim * 4,
                embedding_dim
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
