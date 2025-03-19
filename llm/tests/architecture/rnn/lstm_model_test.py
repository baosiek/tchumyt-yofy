import pytest
import torch
from unittest.mock import patch
from llm.llm.architecture.rnn.lstm_model import LSTMText


@pytest.fixture
def lstm_model():
    return LSTMText(id=1, input_size=10, hidden_size=20)


def test_lstm_initialization(lstm_model):
    assert lstm_model.id == 1
    assert lstm_model.input_size == 10
    assert lstm_model.hidden_size == 20
    assert lstm_model.W_xi.shape == (10, 20)
    assert lstm_model.W_hi.shape == (20, 20)
    assert lstm_model.b_i.shape == (20,)
    assert lstm_model.W_xf.shape == (10, 20)
    assert lstm_model.W_hf.shape == (20, 20)
    assert lstm_model.b_f.shape == (20,)
    assert lstm_model.W_xo.shape == (10, 20)
    assert lstm_model.W_ho.shape == (20, 20)
    assert lstm_model.b_o.shape == (20,)
    assert lstm_model.W_xc.shape == (10, 20)
    assert lstm_model.W_hc.shape == (20, 20)
    assert lstm_model.b_c.shape == (20,)


def test_lstm_forward_shape(lstm_model):
    X = torch.randn(5, 7, 10)  # batch_size=5, sequence_size=7, input_size=10
    output, (H_t, C_t) = lstm_model(X)
    assert output.shape == (5, 7, 20)
    assert H_t.shape == (5, 20)
    assert C_t.shape == (5, 20)


def test_lstm_forward_invalid_input_size(lstm_model):
    X = torch.randn(5, 7, 15)  # input_size=15 does not match model
    # input_size=10
    with pytest.raises(ValueError):
        lstm_model(X)


def test_lstm_forward_with_initial_states(lstm_model):
    X = torch.randn(5, 7, 10)  # batch_size=5, sequence_size=7, input_size=10
    H_0 = torch.randn(5, 20)  # batch_size=5, hidden_size=20
    C_0 = torch.randn(5, 20)  # batch_size=5, hidden_size=20
    output, (H_t, C_t) = lstm_model(X, (H_0, C_0))
    assert output.shape == (5, 7, 20)
    assert H_t.shape == (5, 20)
    assert C_t.shape == (5, 20)


def test_lstm_initialize_weights(lstm_model):
    with patch.object(
        lstm_model, 'initialize_weights'
    ) as mock_initialize_weights:
        lstm_model.__init__(1, 10, 20)
        mock_initialize_weights.assert_called_once()
