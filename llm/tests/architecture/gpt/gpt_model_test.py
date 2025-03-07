import pytest
import torch

from llm.llm import cfg
from llm.llm.architecture.gpt.gpt_model import GPTModel


@pytest.fixture
def get_batch() -> torch.Tensor:

    sentence = [3, 5, 2, 0, 1, 6, 4]
    sentence_tensor = torch.tensor(sentence, dtype=torch.long)

    batch_size: int = 5

    # Temporary structure to enable torch.stack
    input_list = []
    for i in range(batch_size):
        input_list.append(sentence_tensor)

    return torch.stack(input_list, dim=0)


def test_gpt_model(get_batch: torch.Tensor):
    torch.manual_seed(123)

    model: GPTModel = GPTModel(cfg=cfg)
    output: torch.Tensor = model(get_batch)

    batch_size: int = get_batch.shape[0]
    vocabulary_size: int = cfg['vocabulary_size']
    sequence_size: int = get_batch.shape[1]

    total_params: int = sum([p.numel() for p in model.parameters()])
    print(f"Total number of parameters: {total_params}")

    total_size_mb: int = (total_params * 4) / (1024 * 1024)
    print("Total memory footprint required to"
          f" run the model: {total_size_mb:,.2f} MB")

    assert batch_size == output.shape[0]
    assert vocabulary_size == output.shape[2]
    assert sequence_size == output.shape[1]
