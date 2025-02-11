import pytest
import torch

from llm.llm.components.decoding_strategies import TopKScaling


@pytest.fixture
def logits() -> torch.Tensor:
    torch.manual_seed(123)
    return torch.normal(mean=torch.arange(1., 11.),
                        std=torch.arange(1, 0, -0.1)
                        )


def test_tok_k_sampling(logits: torch.Tensor):

    logits = torch.softmax(logits, dim=-1)
    top_k_scaling = TopKScaling(topk_k=3, temperature=0.5)
    next_token: torch.Tensor = top_k_scaling.decode(logits=logits)
    assert next_token.item() == 5
