import torch
import torch.nn as nn
from llm.llm.architecture.rnn.lstm_model import LSTMText
from llm.llm.architecture.abstract_model import AbstractModel

from typing import Any, Dict


class RNNModelV1(AbstractModel):
    '''
    RNNModelV1 is a recurrent neural network model that uses LSTM layers for
    sequence modeling.

    Args:
        embedding_dim (int): The dimension of the embeddings.
        vocabulary_size (int): The size of the vocabulary.
        context_length (int): The length of the input context.
        num_layers (int): The number of LSTM layers in the model.
        device (torch.device, optional): The device to run the model on.
        Defaults to torch.device("cpu").

    Attributes:
        vocabulary_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embeddings.
        context_length (int): The length of the input context.
        num_layers (int): The number of LSTM layers in the model.
        device (torch.device): The device to run the model on.
        token_embedding_layer (nn.Embedding): The embedding layer for tokens.
        pos_embedding_layer (nn.Embedding): The embedding layer for
        positional encodings.
        rnn_layers (nn.ModuleList): A list of LSTM layers.
        output_layer (nn.Linear): A linear layer to generate the output.

    Methods:
        forward(X: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the model.
            Args:
                X (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor.
    '''
    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(cfg=cfg, device=device)

        # the embedding layer
        self.token_embedding_layer: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.embedding_dim,
        )

        self.pos_embedding_layer: nn.Embedding = nn.Embedding(
            num_embeddings=self.context_length,
            embedding_dim=self.embedding_dim
        )

        # the rnn (LSTM or GRU) rnn
        self.rnn_layers: nn.ModuleList = nn.ModuleList(
            [
                LSTMText(
                    id=id, input_size=self.embedding_dim,
                    hidden_size=self.embedding_dim
                )
                for id in range(self.num_layers)
            ]
        )

        # Linear layer to generate the output
        self.output_layer: nn.Linear = nn.Linear(
            in_features=self.embedding_dim, out_features=self.vocabulary_size
        )

    def __repr__(self):
        repr: str = f"""RNNModel(input_size={self.input_size},
        hidden_size={self.hidden_size},
        vocabulary_size={self.vocabulary_size},
        context_length={self.context_length},
        output_dim={self.output_dim},
        out_features={self.output_features},
        (token_embedding_layer)={self.token_embedding_layer}),
        (pos_embedding_layer)={self.pos_embedding_layer},
        (rnn_layer)={self.rnn_layers},
        (output_layer)={self.output_layer}
        )
        """
        return repr

    def forward(self, X: torch.Tensor):

        seq_length: int = X.shape[1]

        token_embeddings_ = self.token_embedding_layer(X)
        pos_embeddings_ = self.pos_embedding_layer(
            torch.arange(seq_length).to(device=self.device)
        )

        # TODO: Insert a Dropout layer

        input_embeddings_ = token_embeddings_ + pos_embeddings_

        states = None
        for rnn_layer in self.rnn_layers:
            output_, states = rnn_layer(input_embeddings_, states)

        output_ = self.output_layer(output_)

        return output_
