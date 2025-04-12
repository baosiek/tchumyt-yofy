import torch
import torch.nn as nn

from collections import OrderedDict

from llm.llm.architecture.tmyts.min_gru import minGRU
from llm.llm.architecture.tmyts.conv1d_block import Conv1DBlock


class TymysLLM(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 seq_length: int,
                 vocabulary_size: int,
                 dropout_rate: float,
                 kernel_size: int,
                 stride: int,
                 num_layers: int,
                 *args,
                 **kwargs):

        super(TymysLLM, self).__init__(*args, **kwargs)

        self.hidden_dim: int = hidden_dim
        self.seq_length: int = seq_length  # used only for positional embedding
        self.vocabulary_size: int = vocabulary_size
        self.dropout_rate: float = dropout_rate

        self.embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.hidden_dim
        )

        self.minGRU_1: minGRU = minGRU(dim=self.hidden_dim)
        self.drop_gru_1: nn.Dropout = nn.Dropout(self.dropout_rate)
        self.norm_gru_1: nn.LayerNorm = nn.LayerNorm(self.hidden_dim)

        self.conv1d_layers: nn.Sequential = nn.Sequential(
            *[Conv1DBlock(
                embedding_dim=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)]
        )

        self.output = nn.Sequential(OrderedDict([
            ('out_linear_1', nn.Linear(
                self.hidden_dim, self.hidden_dim * 2
            )),
            ('out_act_1', nn.GELU()),
            ('out_linear_2', nn.Linear(
                self.hidden_dim * 2, self.hidden_dim * 4
            )),
            ('out_act_2', nn.GELU()),
            ('out_linear_3', nn.Linear(
                self.hidden_dim * 4, self.hidden_dim * 4
            )),
            ('out_act_3', nn.GELU()),
            ('out_linear_4', nn.Linear(
                self.hidden_dim * 4, self.hidden_dim * 2
            )),
            ('out_act_4', nn.GELU()),
            ('out_linear_5', nn.Linear(
                self.hidden_dim * 2, self.vocabulary_size
            )),
        ]))

    def forward(
            self,
            X: torch.Tensor
            ) -> torch.Tensor:
        # validate input:
        if len(X.shape) != 2:
            raise RuntimeError("Input must have shape of "
                               "[batch_size, sequence_length]. "
                               f"Shape of input is: {X.shape}")

        # embeddings
        X = self.embeddings(X)

        shortcut: torch.Tensor = X

        output, _ = self.minGRU_1(X, return_next_prev_hidden=True)
        X = output
        X = self.drop_gru_1(X)
        X = self.norm_gru_1(X)
        X = X + shortcut

        X = self.conv1d_layers(X)

        # output
        X = self.output(X)

        return X
