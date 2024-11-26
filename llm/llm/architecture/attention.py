import torch
import torch.nn as nn


class Attention(nn.Module):
    '''
    This class implements the self-attention mechanism as described in
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

    The forward method computes the scaled dot-product (the attention)
    between a query and a key

    Args:
        embedding_dimension: torch.Tensor -> the embedding dimension
        of the tokens.
        dropout: float -> the dropout rate. A number between [0, 1].
        qvk_bias: bool -> if to use the bias in the linear layers.
        mask: bool -> if the forward mask must be applied.
    '''

    def __init__(self,
                 embedding_dimension: int = 0,
                 dropout: float = 0.5,
                 qvk_bias: bool = False,
                 mask: bool = False):
        super().__init__()

        # Initializes the query layer.
        self.query: nn.Linear = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=qvk_bias
        )

        # Initializes the key layer.
        self.key: nn.Linear = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=qvk_bias
        )

        # Initializes the value layer.
        self.value: nn.Linear = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=qvk_bias
        )

        # Initializes the dropout layer
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        # Initializes mask
        self.mask: bool = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward method computes the scaled dot-product (the attention)
        between a query and a key

        Args: x
            x: torch.Tensor -> contains the input sequence

        Returns: context
            - context: torch.Tensor -> tensor containing the context vector
            from attention mechanism.
        """

        # Number of tokens in the input
        sequence_length: int = x.shape[1]

        #  Updates attention weights with the input
        queries: torch.Tensor = self.query(x)
        keys: torch.Tensor = self.key(x)
        values: torch.Tensor = self.value(x)

        #  Computes the attention scores
        attention_weights: torch.Tensor = torch.matmul(
            queries,
            keys.transpose(1, 2)
        )

        if self.mask:
            attention_weights.masked_fill_(
                torch.triu(
                    torch.ones(
                        sequence_length,
                        sequence_length),
                    diagonal=1).bool(),
                -torch.inf)

        # Effectively computes the attention weights
        attention_weights = torch.softmax(
            attention_weights /
            torch.sqrt(
                torch.tensor(
                    keys.shape[2]
                )
            ),
            dim=2
        )

        # Applies dropout
        attention_weights = self.dropout(attention_weights)

        # Returns the context vector
        return torch.matmul(attention_weights, values)
