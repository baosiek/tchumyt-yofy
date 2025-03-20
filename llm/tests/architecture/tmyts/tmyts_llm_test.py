import pytest
import torch
from llm.llm.architecture.tmyts.tmyts_llm import TymysLLM


@pytest.fixture
def model():
    """Fixture to create a TymysLLM instance."""
    return TymysLLM(
        hidden_dim=128,
        seq_length=10,
        vocabulary_size=1000,
        dropout_rate=0.1,
        num_heads=4
    )


def test_model_initialization(model):
    """Test that the model initializes correctly."""
    assert isinstance(model, TymysLLM)
    assert model.hidden_size == 128
    assert model.seq_length == 10
    assert model.vocabulary_size == 1000
    assert model.dropout_rate == 0.1
    assert model.num_heads == 4


def test_forward_pass(model):
    """Test the forward pass of the model."""
    batch_size = 32
    seq_length = model.seq_length
    vocabulary_size = model.vocabulary_size

    # Create a random input tensor with shape (batch_size, seq_length)
    input_tensor = torch.randint(0, vocabulary_size, (batch_size, seq_length))

    # Perform a forward pass
    output = model(input_tensor)

    # Assert the output shape is correct
    assert output.shape == (batch_size, seq_length, vocabulary_size)
    assert output.dtype == torch.float32


def test_invalid_input_shape(model):
    """Test that the model raises an error for invalid input shapes."""
    # Invalid shape
    invalid_input = torch.randint(0, model.vocabulary_size, (5,))

    with pytest.raises(RuntimeError):
        model(invalid_input)


def test_dropout_rate_effect(model):
    """Test that the dropout layers are applied correctly."""
    model.eval()  # Set model to evaluation mode (dropout disabled)
    batch_size = 16
    seq_length = model.seq_length

    input_tensor = torch.randint(
        0, model.vocabulary_size, (batch_size, seq_length)
    )
    output_eval = model(input_tensor)

    model.train()  # Set model to training mode (dropout enabled)
    output_train = model(input_tensor)

    # Outputs in training mode should differ due to dropout
    assert not torch.equal(output_eval, output_train)
