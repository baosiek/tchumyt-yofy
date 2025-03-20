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

        self.minGRU: minGRU = minGRU(dim=self.hidden_size)

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

        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            bias=False,
            batch_first=True
            )

        self.mental_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_1', nn.GELU()),
            ('linear_2', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_2', nn.GELU()),
            ('linear_3', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_3', nn.GELU()),
            ('linear_4', nn.Linear(self.hidden_size, self.hidden_size)),
            ('act_4', nn.GELU()),
        ]))

        self.grammar_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_1', nn.GELU()),
            ('linear_2', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_2', nn.GELU()),
            ('linear_3', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_3', nn.GELU()),
            ('linear_4', nn.Linear(
                self.hidden_size, self.hidden_size
            )),
            ('act_4', nn.GELU()),
        ]))

        self.output = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(
                self.hidden_size, self.vocabulary_size
            )),
            ('act_4', nn.GELU()),
            # ('linear_2', nn.Linear(
            #     self.vocabulary_size, self.vocabulary_size
            # )),
            ('out_act', nn.Sigmoid()),
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
        X: torch.Tensor = self.embeddings(X)
        X = self.drop_embedding(X)

        shortcut: torch.Tensor = X

        # X = self.norm0(X)
        output, hidden_state = self.minGRU(X, return_next_prev_hidden=True)
        X = output + hidden_state
        X = self.drop_rnn(X)
        X = X + shortcut

        # multihead
        X = self.norm1(X)
        X, _ = self.multihead_attn(X, X, X)
        X = self.drop_att(X)
        X = X + shortcut

        # mental model
        X_m: torch.Tensor = self.norm2(X)
        X_m = self.mental_model(X_m)
        X_m = self.drop_mental_mlp(X_m)
        X_m = X_m + shortcut

        # grammar model
        X_g: torch.Tensor = self.norm3(X)
        X_g = self.grammar_model(X_g)
        X_g = self.drop_grammar_mlp(X_g)
        X_g = X_g + shortcut

        # union
        X = X_m + X_g

        # output
        X = self.norm4(X)
        X = self.output(X)

        return X
