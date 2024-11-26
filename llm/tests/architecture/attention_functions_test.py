import pytest
import torch
import torch.nn as nn

from typing import Dict
from llm.llm.architecture.attention_functions import scaled_dot_product


@pytest.fixture
def container_fixture() -> Dict[str, torch.Tensor]:

    torch.manual_seed(123)
    query: nn.Linear = nn.Linear(3,3)
    key: torch.Tensor = torch.rand(2, 6, 3)
    value: torch.Tensor = torch.rand(2, 6, 3)

    print(query)
    fixture: Dict[str, torch.Tensor] = {
        'query': query,
        'key': key,
        'value': value
    }

    return fixture


@pytest.fixture
def mask_fixture() -> torch.Tensor:
    mask: torch.Tensor = torch.tril(torch.ones(6, 6))

    return mask


def test_scaled_dot_product(container_fixture: torch.Tensor) -> None:
    context_vector: torch.Tensor = scaled_dot_product(
        q=container_fixture['query'],
        k=container_fixture['key'],
        v=container_fixture['value']
    )

    assert context_vector.shape == (2, 6, 3), f"Size should be ([2, 6, 3]) but was {context_vector.shape}"