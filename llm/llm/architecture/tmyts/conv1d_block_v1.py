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

        self.conv_0: nn.Conv1d = nn.Conv1d(
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
            groups=8
        )

        self.conv_2: nn.Conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=kernel_size * 4,
            stride=stride, padding='same',
            groups=16
        )

        self.batch_norm_0: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=embedding_dim
        )

        self.batch_norm_1: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=embedding_dim
        )

        self.batch_norm_2: nn.BatchNorm1d = nn.BatchNorm1d(
            num_features=embedding_dim
        )

        self.layer_norm_2: nn.LayerNorm = nn.LayerNorm(embedding_dim)
        self.relu = nn.ReLU()

        self.dropout_conv_0: nn.Dropout = nn.Dropout(dropout_rate)
        self.dropout_conv_1: nn.Dropout = nn.Dropout(dropout_rate)
        self.dropout_conv_2: nn.Dropout = nn.Dropout(dropout_rate)

        self.ff = FeedForward(embedding_dim=embedding_dim)

        self.dropout_ff: nn.Dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):

        shortcut: torch.Tensor = x

        x_0: torch.Tensor = shortcut.transpose(1, 2)
        x_0 = self.conv_0(x_0)
        x_0 = self.batch_norm_0(x_0)
        x_0 = self.relu(x_0)
        x_0 = self.dropout_conv_0(x_0)
        x_0 = x_0.transpose(1, 2)
        # x_0 = x_0 + shortcut

        x_1: torch.Tensor = shortcut.transpose(1, 2)
        x_1 = self.conv_1(x_1)
        x_1 = self.batch_norm_1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.dropout_conv_1(x_1)
        x_1 = x_1.transpose(1, 2)
        # x_1 = x_1 + shortcut

        x_2: torch.Tensor = shortcut.transpose(1, 2)
        x_2 = self.conv_2(x_2)
        x_2 = self.batch_norm_2(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.dropout_conv_2(x_2)
        x_2 = x_2.transpose(1, 2)
        # x_1 = x_1 + shortcut

        x_out: torch.Tensor = x_0 + x_1 + x_2 + shortcut

        x_out = self.ff(x_out)
        x_out = self.layer_norm_2(x_out)
        x_out = self.dropout_ff(x_out)
        x_out = x_out + shortcut

        return x_out
