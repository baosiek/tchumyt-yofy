import torch
import torch.nn as nn

from collections import OrderedDict

from llm.llm.architecture.tmyts.min_gru import minGRU


class TymysLLM(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 seq_length: int,
                 vocabulary_size: int,
                 dropout_rate: float,
                 num_heads: int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size: int = hidden_dim
        self.seq_length: int = seq_length
        self.vocabulary_size: int = vocabulary_size
        self.dropout_rate: float = dropout_rate
        self.num_heads: int = num_heads

        self.embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.hidden_size
        )

        self.pos_embedding_layer: nn.Embedding = nn.Embedding(
            num_embeddings=self.seq_length,
            embedding_dim=self.hidden_size
        )

        self.minGRU_1: minGRU = minGRU(dim=self.hidden_size)
        # self.gru: nn.GRU = nn.GRU(
        #     input_size=self.hidden_size,
        #     hidden_size=self.hidden_size,
        #     batch_first=True,
        #     bidirectional=False
        # )

        self.conv1d_1: nn.Conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )
        self.batch_norm_1: nn.BatchNorm1d = nn.BatchNorm1d(self.hidden_size)

        self.drop_gru_1: nn.Dropout = nn.Dropout(self.dropout_rate)
        self.norm_1: nn.LayerNorm = nn.LayerNorm(self.hidden_size)

        self.output = nn.Sequential(OrderedDict([
            ('out_linear_1', nn.Linear(
                self.hidden_size, self.hidden_size * 2
            )),
            ('out_act_1', nn.ReLU()),
            ('out_linear_2', nn.Linear(
                self.hidden_size * 2, self.hidden_size * 2
            )),
            ('out_act_2', nn.ReLU()),
            ('out_linear_3', nn.Linear(
                self.hidden_size * 2, self.vocabulary_size
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

        # seq_length: int = X.shape[1]

        # embeddings
        X = self.embeddings(X)

        shortcut: torch.Tensor = X

        # # positional embeddings
        # pos_embeddings = self.pos_embedding_layer(
        #     torch.arange(seq_length).to(X.device)
        # )

        # X_conv: torch.Tensor = X.transpose(1, 2)
        # X_conv = self.conv1d_1(X_conv)
        # X_conv = self.batch_norm_1(X_conv)
        # X = X_conv.transpose(1, 2)

        output, _ = self.minGRU_1(X, return_next_prev_hidden=True)
        X = output
        X = self.drop_gru_1(X)
        X = self.norm_1(X)
        X = X + shortcut

        X_conv: torch.Tensor = X.transpose(1, 2)
        X_conv = self.conv1d_1(X_conv)
        X_conv = self.batch_norm_1(X_conv)
        X = X_conv.transpose(1, 2)

        # # new input
        # X = X + pos_embeddings

        # output
        X = self.output(X)

        return X
