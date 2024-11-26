import torch
import torch.nn.functional as F
import math

'''
This function computes the scaled dot product as defined in
https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

Args:
    query: torch.Tensor -> the query tensor.
    key: torch.Tensor -> the query tensor.
    value: torch.Tensor -> the query tensor.

Returns:
    context_vector: torch.Tensor -> the context_vector, or,
    the scaled dot product
'''


def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
        ) -> torch.Tensor:

    # The embedding dimension
    embedding_dimension: torch.Tensor = torch.tensor(
        query.size()[-1],
        dtype=torch.float32
        )

    # The nominator of the self attention
    attention_score: torch.Tensor = torch.matmul(
        query,
        key.transpose(-2, -1)
        )

    # The attention weights
    attention_weights: torch.Tensor = \
        attention_score / math.sqrt(embedding_dimension)

    if mask is not None:
        attention_weights = attention_weights.masked_fill_(
            mask.bool(),
            -torch.inf
        )

    # Normalization
    attention_weights = F.softmax(attention_weights, dim=-1)

    # Returns the context vector
    return torch.matmul(attention_weights, value)
