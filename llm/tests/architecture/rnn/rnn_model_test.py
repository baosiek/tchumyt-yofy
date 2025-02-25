import pytest
import torch
from llm.llm.architecture.rnn.rnn_model import RNNModelV1

import torch.nn as nn


@pytest.fixture
def cfg():
    return {
        'vocabulary_size': 1000,
        'embedding_dim': 128,
        'context_length': 50,
        'num_layers': 2
    }


@pytest.fixture
def rnn_model(cfg):
    return RNNModelV1(
        cfg=cfg,
        device=torch.device("cpu")
    )


def test_rnn_model_initialization(rnn_model):
    assert isinstance(rnn_model, nn.Module)
    assert rnn_model.vocabulary_size == 1000
    assert rnn_model.embedding_dim == 128
    assert rnn_model.context_length == 50
    assert rnn_model.num_layers == 2
    assert rnn_model.device == torch.device("cpu")


def test_rnn_model_forward(rnn_model):
    input_tensor = torch.randint(0, 1000, (10, 50)).to(torch.device("cpu"))
    output = rnn_model(input_tensor)
    assert output.shape == (10, 50, 1000)


def test_token_embedding_layer(rnn_model):
    input_tensor = torch.randint(0, 1000, (10, 50)).to(torch.device("cpu"))
    token_embeddings = rnn_model.token_embedding_layer(input_tensor)
    assert token_embeddings.shape == (10, 50, 128)


def test_pos_embedding_layer(rnn_model):
    pos_embeddings = rnn_model.pos_embedding_layer(torch.arange(50).to(
        device=torch.device("cpu")
    ))
    assert pos_embeddings.shape == (50, 128)


def test_rnn_layers(rnn_model):
    input_tensor = torch.randn(10, 50, 128).to(torch.device("cpu"))
    states = None
    for rnn_layer in rnn_model.rnn_layers:
        output, states = rnn_layer(input_tensor, states)
    assert output.shape == (10, 50, 128)


def test_output_layer(rnn_model):
    input_tensor = torch.randn(10, 50, 128).to(torch.device("cpu"))
    output = rnn_model.output_layer(input_tensor)
    assert output.shape == (10, 50, 1000)
