import pytest
import torch

from llm.llm import cfg
from llm.llm.architecture.gpt_model import GPTModel


@pytest.fixture
def get_batch() -> torch.Tensor:
    
    # vocabulary = {"at": 0, "home": 1, "is": 2, "The": 3, "his": 4, "boy": 5}
  
    sentence = [3, 5, 2, 0, 4, 1]
    lookup_tensor = torch.tensor(sentence, dtype=torch.long)
    # # Input tensor
    # inputs = torch.tensor(
    #     [[0.43, 0.15, 0.89],    # The
    #      [0.55, 0.87, 0.66],    # boy
    #      [0.57, 0.85, 0.64],    # is
    #      [0.22, 0.58, 0.33],    # at
    #      [0.77, 0.25, 0.10],    # his
    #      [0.05, 0.80, 0.55]]    # home
    # )

    batch_size: int = 2

    # Temporary structure to enable torch.stack
    input_list = []
    for i in range(batch_size):
        input_list.append(lookup_tensor)

    return torch.stack(input_list, dim=0)


def test_gpt_model(get_batch: torch.Tensor):
    torch.manual_seed(123)

    model: GPTModel = GPTModel(cfg=cfg)
    output: torch.Tensor = model(get_batch)
    print(output)
