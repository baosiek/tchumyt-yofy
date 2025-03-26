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

        self.minGRU_1: minGRU = minGRU(dim=self.hidden_size)

        self.conv1d_1: nn.Conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )

        self.conv1d_2: nn.Conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )

        self.conv1d_3: nn.Conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )

        self.drop_embedding: nn.Dropout = nn.Dropout(self.dropout_rate)
        self.drop_rnn: nn.Dropout = nn.Dropout(self.dropout_rate)
        self.drop_att: nn.Dropout = nn.Dropout(self.dropout_rate)
        self.drop_mental_model: nn.Dropout = nn.Dropout(self.dropout_rate)
        self.drop_mental_mlp: nn.Dropout = nn.Dropout(self.dropout_rate)
        self.drop_grammar_mlp: nn.Dropout = nn.Dropout(self.dropout_rate)

        self.norm0: nn.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.norm1: nn.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.norm2: nn.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.norm3: nn.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.norm4: nn.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.norm5: nn.LayerNorm = nn.LayerNorm(self.hidden_size)

        self.batch_norm_1: nn.BatchNorm1d = nn.BatchNorm1d(self.hidden_size)
        self.batch_norm_2: nn.BatchNorm1d = nn.BatchNorm1d(self.hidden_size)
        self.batch_norm_3: nn.BatchNorm1d = nn.BatchNorm1d(self.hidden_size)

        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            bias=False,
            batch_first=True
            )

        self.mental_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_1', nn.ReLU()),
            ('linear_2', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_2', nn.ReLU()),
            ('linear_3', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_3', nn.ReLU()),
            ('linear_4', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_4', nn.ReLU()),
            ('linear_5', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_5', nn.ReLU()),
            ('linear_6', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_6', nn.ReLU()),
            ('linear_7', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_7', nn.ReLU()),
        ]))

        self.grammar_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_1', nn.ReLU()),
            ('linear_2', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_2', nn.ReLU()),
            ('linear_3', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_3', nn.ReLU()),
            ('linear_4', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_4', nn.ReLU()),
            ('linear_5', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_5', nn.ReLU()),
            ('linear_6', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_6', nn.ReLU()),
            ('linear_7', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_7', nn.ReLU()),
        ]))

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

        # comments
        X = self.embeddings(X)
        X = self.drop_embedding(X)

        shortcut: torch.Tensor = X

        output, hidden_state = self.minGRU_1(X, return_next_prev_hidden=True)
        X = output + hidden_state
        X = self.drop_rnn(X)
        X = X + shortcut
        X = self.norm0(X)

        # multihead attention
        X, _ = self.multihead_attn(X, X, X)
        X = self.drop_att(X)
        X = X + shortcut
        X = self.norm1(X)

        # mental model
        # convolution
        X_conv: torch.Tensor = X.transpose(1, 2)
        X_conv = self.conv1d_1(X_conv)
        X_conv = self.batch_norm_1(X_conv)
        X_conv = self.conv1d_2(X_conv)
        X_conv = self.batch_norm_2(X_conv)
        X_conv = self.conv1d_3(X_conv)
        X_conv = self.batch_norm_3(X_conv)
        X_conv = X_conv.transpose(1, 2)
        X_conv = X + X_conv
        X_conv = nn.ReLU()(X_conv)

        # mental model
        X_m: torch.Tensor = self.mental_model(X_conv)
        X_m = self.drop_mental_mlp(X_m)
        X_m = X_m + shortcut
        X_m = self.norm3(X)

        # grammar model
        X_g: torch.Tensor = self.grammar_model(X)
        X_g = self.drop_grammar_mlp(X_g)
        X_g = X_g + shortcut
        X_g = self.norm4(X)

        # union
        X = X_m + X_g
        X = self.norm5(X)

        # output
        X = self.output(X)

        return X
