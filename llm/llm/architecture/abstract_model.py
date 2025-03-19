from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn

"""
AbstractModel is an abstract base class for all models. It inherits from both
ABC (Abstract Base Class) and nn.Module (PyTorch's base class for all neural
network modules).

Attributes:
    cfg (Dict[str, Any]): Configuration dictionary for the model.
    device (torch.device): The device on which the model will be run.

Methods:
    __init__(cfg: Dict[str, Any], device: torch.device):
        Initializes the model with the given configuration and device.

Usage example:
    class MyModel(AbstractModel):
        def __init__(self, cfg: Dict[str, Any], device: torch.device = \
            torch.device("cpu")):
            super(MyModel, self).__init__(cfg, device)
            # Define your model layers here
            self.layer1 = nn.Linear(cfg['input_size'], cfg['hidden_size'])
            self.layer2 = nn.Linear(cfg['hidden_size'], cfg['output_size'])

        def forward(self, x):
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            return x

    # Example configuration
    cfg = {
        'input_size': 10,
        'hidden_size': 20,
        'output_size': 1
    }

    # Instantiate the model
    model = MyModel(cfg, device=torch.device
        ("cuda" if torch.cuda.is_available() else "cpu"))

    # Example input
    input_tensor = torch.randn(5, 10)

    # Forward pass
    output = model(input_tensor)
    print(output)
"""


class AbstractModel(ABC, nn.Module):
    @abstractmethod
    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        # The parameters
        self.vocabulary_size: int = cfg['vocabulary_size']
        self.embedding_dim: int = cfg['embedding_dim']
        self.context_length: int = cfg['context_length']
        self.num_layers: int = cfg['num_layers']
        self.device: torch.device = device
