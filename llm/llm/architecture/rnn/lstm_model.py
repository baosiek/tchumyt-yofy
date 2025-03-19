import torch
import math

import torch.nn as nn


class LSTMText(nn.Module):
    def __init__(self, id: int, input_size: int, hidden_size: int):
        """
        Args:
            id (int): Identifier for the LSTM model instance.
            input_size (int): Size of the input feature dimension
                (embedding dimension).
            hidden_size (int): Size of the hidden state dimension.
        """
        super().__init__()
        self.id: int = id
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size

        # Input gate (I):
        # I_t = sigma(X_t.W_xi + H_t-1.W_hi + b_i)
        # Where:
        # X_t is X(the sequence) at time t
        # H_t-1 is H(the hidden state) at time t-1
        # W_xi is the weight matrix of X to I gate
        self.W_xi: nn.Parameter = nn.Parameter(
            torch.Tensor(self.input_size, self.hidden_size)
        )

        # W_hi is the weight matrix of h to the I gate
        self.W_hi: torch.Tensor = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size)
        )

        # self.b_i is the bias to the I gate
        self.b_i: torch.Tensor = nn.Parameter(torch.Tensor(self.hidden_size))

        # Forget gate (F):
        # F_t = sigma(X_t.W_xf + H_t-1.W_hf + b_f)
        # Where:
        # X_t is X(the sequence) at time t
        # H_t-1 is H(the hidden state) at time t-1
        # W_xf is the weight matrix of X to F gate
        self.W_xf: nn.Parameter = nn.Parameter(
            torch.Tensor(self.input_size, self.hidden_size)
        )

        # W_hf is the weight matrix of h to the F gate
        self.W_hf: torch.Tensor = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size)
        )

        # self.b_f is the bias to the F gate
        self.b_f: torch.Tensor = nn.Parameter(torch.Tensor(self.hidden_size))

        # Output gate (O):
        # O_t = sigma(X_t.W_xo + H_t-1.W_ho + b_o)
        # Where:
        # X_t is X(the sequence) at time t
        # H_t-1 is H(the hidden state) at time t-1
        # W_xo is the weight matrix of X to O gate
        self.W_xo: nn.Parameter = nn.Parameter(
            torch.Tensor(self.input_size, self.hidden_size)
        )

        # W_ho is the weight matrix of h to the O gate
        self.W_ho: torch.Tensor = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size)
        )

        # self.b_o is the bias to the O gate
        self.b_o: torch.Tensor = nn.Parameter(torch.Tensor(self.hidden_size))

        # Cell (C):
        # C_t = sigma(X_t.W_xc + H_t-1.W_hc + b_c)
        # Where:
        # X_t is X(the sequence) at time t
        # H_t-1 is H(the hidden state) at time t-1
        # W_xc is the weight matrix of X to C cell
        self.W_xc: nn.Parameter = nn.Parameter(
            torch.Tensor(self.input_size, self.hidden_size)
        )

        # W_ho is the weight matrix of h to the C cell
        self.W_hc: torch.Tensor = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size)
        )

        # self.b_o is the bias to the C cell
        self.b_c: torch.Tensor = nn.Parameter(torch.Tensor(self.hidden_size))

        # Initializes all weights
        self.initialize_weights()

    def __repr__(self):
        repr: str = f"LSTMText(id={self.id}, input_size={self.input_size}, "
        f"hidden_size={self.hidden_size})"
        return repr

    def initialize_weights(self):
        stdev: float = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdev, stdev)

    def forward(self, X: torch.Tensor, states: torch.Tensor = None):
        """
        assumes x.shape represents (batch_size, sequence_size,
        embedding_dimension)
        """
        bs, sequence_size, input_size = X.size()

        if input_size != self.input_size:
            raise ValueError(
                f"Input shape: {input_size} is not equal "
                f"to model input size: {self.input_size}"
            )

        if states is None:
            H_t, C_t = (
                torch.zeros(bs, self.hidden_size).to(device=X.device),
                torch.zeros(bs, self.hidden_size).to(device=X.device),
            )
        else:
            H_t, C_t = states

        outputs = []
        for t in range(sequence_size):
            x = X[:, t, :]
            # I is the input gate
            I_t = torch.sigmoid(
                torch.matmul(x, self.W_xi) +
                torch.matmul(H_t, self.W_hi) +
                self.b_i
            )

            # F is the forget state
            F_t = torch.sigmoid(
                torch.matmul(x, self.W_xf) +
                torch.matmul(H_t, self.W_hf) +
                self.b_f
            )

            # O is the output state
            O_t = torch.sigmoid(
                torch.matmul(x, self.W_xo) +
                torch.matmul(H_t, self.W_ho) +
                self.b_o
            )

            # C_t, the memory (C)ell is:
            # C_t = F(.)C_t-1 + I_t(.)C_temp
            # C_temp = tanh(X_t.W_xc + H_t-1.W_hc + b_c)
            C_temp = torch.tanh(
                torch.matmul(x, self.W_xc) +
                torch.matmul(H_t, self.W_hc) +
                self.b_c
            )
            C_t = F_t * C_t + I_t * C_temp
            H_t = O_t * torch.tanh(C_t)
            outputs.append(H_t.unsqueeze(1))

        result = torch.cat(outputs, dim=1)
        return result, (H_t, C_t)
