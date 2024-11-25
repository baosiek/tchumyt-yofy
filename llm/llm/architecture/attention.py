import torch
import torch.nn as nn


class Attention(nn.Module):
    '''
    This class implements the self-attention mechanism as described in
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    '''

    def __init__(self, d_in: int,
                 d_out, dropout: float = 0.5,
                 qvk_bias: bool = False,
                 mask: torch.Tensor = None):
        super().__init__()

        # Initializes the input layer.
        self.query = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.key = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.value = nn.Linear(d_in, d_out, bias=qvk_bias)

        # Initializes the dropout layer
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        # Initializes the mask layer
        self.mask: torch.Tensor = mask

        # Register mask in the buffer to enable allocation into
        # the appropriate device together with the model    
        if mask is not None:
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):

        #  Updates attention weights with the input
        queries: torch.Tensor = self.query(x)
        keys: torch.Tensor = self.key(x)
        values: torch.Tensor = self.value(x)

        #  Computes the attention weights
        attention_weights = torch.softmax(
            torch.matmul(queries, keys.T) /
            torch.sqrt(torch.tensor(keys.shape[-1])), dim=-1
        )

        if self.mask is not None:
            attention_weights += (self.mask * -torch.inf)

        # Applies the dropout layer
        attention_weights = self.dropout(attention_weights)

        # Returns the context vector
        return torch.matmul(attention_weights, values)
