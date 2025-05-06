import pytest
import torch
from llm.llm.architecture.tmyts.min_gru import minGRU, exists, default, \
    heinsen_associative_scan_log, g, log_g


@pytest.fixture
def sample_tensor():
    # Batch size: 2, Sequence length: 5, Feature size: 10
    return torch.randn(2, 5, 10)


@pytest.fixture
def sample_hidden():
    return torch.randn(2, 1, 20)  # Batch size: 2, Feature size: 10


def test_min_gru_forward_parallel(sample_tensor):
    model = minGRU(dim=10, expansion_factor=2.0)
    output = model(sample_tensor)
    assert output.shape == (2, 5, 10)


def test_min_gru_forward_sequential(
        sample_tensor,
        sample_hidden
):
    model = minGRU(dim=10, expansion_factor=2.0)
    output, next_hidden = model(
        sample_tensor[:, :1, :],
        prev_hidden=sample_hidden,
        return_next_prev_hidden=True
    )
    assert output.shape == (2, 1, 10)
    assert next_hidden.shape == (2, 1, 20)


def test_min_gru_no_projection(sample_tensor):
    model = minGRU(dim=10, expansion_factor=1.0, proj_out=None)
    output = model(sample_tensor)
    assert output.shape == (2, 5, 10)


def test_exists():
    assert exists(5) is True
    assert exists(None) is False


def test_default():
    assert default(5, 10) == 5
    assert default(None, 10) == 10


def test_heinsen_associative_scan_log():
    log_coefficients = torch.randn(2, 5)
    log_values = torch.randn(2, 5)
    result = heinsen_associative_scan_log(log_coefficients, log_values)
    assert result.shape == (2, 5)


def test_g():
    x = torch.tensor([-1.0, 0.0, 1.0])
    result = g(x)
    assert torch.all(result >= 0)


def test_log_g():
    x = torch.tensor([-1.0, 0.0, 1.0])
    result = log_g(x)
    assert result.shape == x.shape
