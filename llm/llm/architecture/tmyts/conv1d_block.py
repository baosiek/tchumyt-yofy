import torch
import torch.nn as nn

from llm.llm.architecture.tmyts.tmyts_feed_forward import FeedForward


class Conv1DBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            kernel_size: int,
            stride: int,
            dropout_rate: float
            ):
        super(Conv1DBlock, self).__init__()

        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride, padding='same',
            groups=4
        )

        self.conv_1: nn.Conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=kernel_size * 2,
            stride=stride, padding='same',
            groups=4
        )

        self.conv_2: nn.Conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=kernel_size * 4,
            stride=stride, padding='same',
            groups=4
        )

        self.batch_norm_1: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=embedding_dim
        )

        self.layer_norm_2: nn.LayerNorm = nn.LayerNorm(embedding_dim)
        self.relu = nn.ReLU()

        self.dropout_conv: nn.Dropout = nn.Dropout(dropout_rate)

        self.ff = FeedForward(embedding_dim=embedding_dim)

        self.dropout_ff: nn.Dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor):

        shortcut: torch.Tensor = X

        X = shortcut.transpose(1, 2)
        X = self.conv(X)
        X = self.batch_norm_1(X)
        X = self.relu(X)
        X = self.dropout_conv(X)
        X = X.transpose(1, 2)
        X = X + shortcut

        X = self.ff(X)
        X = self.layer_norm_2(X)
        X = self.dropout_ff(X)
        X = X + shortcut

        return X
