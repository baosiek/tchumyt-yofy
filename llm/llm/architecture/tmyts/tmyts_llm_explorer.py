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
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size: int = hidden_dim
        self.seq_length: int = seq_length  # used only for positional embedding
        self.vocabulary_size: int = vocabulary_size
        self.dropout_rate: float = dropout_rate

        self.embeddings: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.hidden_size
        )

        # self.pos_embedding_layer: nn.Embedding = nn.Embedding(
        #     num_embeddings=self.seq_length,
        #     embedding_dim=self.hidden_size
        # )

        self.minGRU_1: minGRU = minGRU(dim=self.hidden_size)
        self.drop_gru_1: nn.Dropout = nn.Dropout(self.dropout_rate)
        self.norm_gru_1: nn.LayerNorm = nn.LayerNorm(self.hidden_size)

        self.conv1d_1: nn.Conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )
        self.batch_norm_1: nn.BatchNorm1d = nn.BatchNorm1d(self.hidden_size)

        self.ff_1 = nn.Sequential(OrderedDict([
            ('out_linear_1', nn.Linear(
                self.hidden_size, self.hidden_size * 4
            )),
            ('out_act_1', nn.GELU()),
            ('out_linear_2', nn.Linear(
                self.hidden_size * 4, self.hidden_size
            )),
            ('out_act_2', nn.GELU()),
        ]))

        self.drop_ff_1: nn.Dropout = nn.Dropout(self.dropout_rate)

        self.conv1d_2: nn.Conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )
        self.batch_norm_2: nn.BatchNorm1d = nn.BatchNorm1d(self.hidden_size)

        self.ff_2 = nn.Sequential(OrderedDict([
            ('out_linear_1', nn.Linear(
                self.hidden_size, self.hidden_size * 4
            )),
            ('out_act_1', nn.GELU()),
            ('out_linear_2', nn.Linear(
                self.hidden_size * 4, self.hidden_size
            )),
            ('out_act_2', nn.GELU()),
        ]))

        self.drop_ff_2: nn.Dropout = nn.Dropout(self.dropout_rate)

        self.conv1d_3: nn.Conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )
        self.batch_norm_3: nn.BatchNorm1d = nn.BatchNorm1d(self.hidden_size)

        self.ff_3 = nn.Sequential(OrderedDict([
            ('out_linear_1', nn.Linear(
                self.hidden_size, self.hidden_size * 4
            )),
            ('out_act_1', nn.GELU()),
            ('out_linear_2', nn.Linear(
                self.hidden_size * 4, self.hidden_size
            )),
            ('out_act_2', nn.GELU()),
        ]))

        self.drop_ff_3: nn.Dropout = nn.Dropout(self.dropout_rate)

        self.output = nn.Sequential(OrderedDict([
            ('out_linear_1', nn.Linear(
                self.hidden_size, self.hidden_size * 2
            )),
            ('out_act_1', nn.GELU()),
            ('out_linear_2', nn.Linear(
                self.hidden_size * 2, self.hidden_size * 4
            )),
            ('out_act_2', nn.GELU()),
            ('out_linear_3', nn.Linear(
                self.hidden_size * 4, self.hidden_size * 4
            )),
            ('out_act_3', nn.GELU()),
            ('out_linear_4', nn.Linear(
                self.hidden_size * 4, self.hidden_size * 2
            )),
            ('out_act_4', nn.GELU()),
            ('out_linear_5', nn.Linear(
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
        X = self.norm_gru_1(X)
        X = X + shortcut

        shortcut = X

        # starts block 1
        X_conv: torch.Tensor = X.transpose(1, 2)
        X_conv = self.conv1d_1(X_conv)
        X_conv = self.batch_norm_1(X_conv)
        X = X_conv.transpose(1, 2)
        X = X + shortcut

        X = self.ff_1(X)
        X = self.drop_ff_1(X)
        X = X + shortcut
        # end block 1

        # starts block 2
        X_conv: torch.Tensor = X.transpose(1, 2)
        X_conv = self.conv1d_2(X_conv)
        X_conv = self.batch_norm_2(X_conv)
        X = X_conv.transpose(1, 2)
        X = X + shortcut

        X = self.ff_2(X)
        X = self.drop_ff_2(X)
        X = X + shortcut
        # end block 2

        # starts block 3
        X_conv: torch.Tensor = X.transpose(1, 2)
        X_conv = self.conv1d_3(X_conv)
        X_conv = self.batch_norm_3(X_conv)
        X = X_conv.transpose(1, 2)
        X = X + shortcut

        X = self.ff_3(X)
        X = self.drop_ff_3(X)
        X = X + shortcut
        # end block 3

        # output
        X = self.output(X)

        return X
