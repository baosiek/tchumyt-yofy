import torch

from llm.llm.architecture.attention import Attention


def test_attention_computation():

    torch.manual_seed(123)

    # Input tensor
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],    # The
         [0.55, 0.87, 0.66],    # boy
         [0.57, 0.85, 0.64],    # is
         [0.22, 0.58, 0.33],    # at
         [0.77, 0.25, 0.10],    # his
         [0.05, 0.80, 0.55]]    # home
    )

    # Context tensor
    context_tensor: torch.Tensor = torch.tensor(
        [[[0.2633,  0.4277, -0.1353],
         [0.2641,  0.4296, -0.1350],
         [0.2641,  0.4296, -0.1350],
         [0.2647,  0.4316, -0.1381],
         [0.2642,  0.4303, -0.1373],
         [0.2648,  0.4316, -0.1375]]]
    )

    batch_size: int = 1

    # Temporary structure to enable torch.stack
    input_list = []
    for i in range(batch_size):
        input_list.append(inputs)

    batch = torch.stack(input_list, dim=0)

    # Embedding dimension
    embedding_dimension: int = batch.shape[2]

    # Initializes the attention layer
    attention = Attention(
        embedding_dimension=embedding_dimension,
        mask=False,
        dropout=0.0)

    context_vector: torch.Tensor = attention(batch)

    assert context_tensor.shape == context_vector.shape
