import torch

from abc import ABC, abstractmethod


# Define an abstract class as an interface to various decoding strategies
class AbstractDecodeStrategy(ABC):

    @abstractmethod
    def decode(self, logits: torch.Tensor):
        pass


class GreedyDecoding(AbstractDecodeStrategy):
    def __init__(
            self,
    ):
        super().__init__()

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)


class TemperatureScaling(AbstractDecodeStrategy):
    def __init__(
            self,
            temperature: float,
            num_samples: int = 1
    ):
        super().__init__()
        self.temperature: float = temperature
        self.num_samples: int = num_samples

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        scaled_logits: torch.Tensor = logits / torch.tensor(self.temperature,
                                                            dtype=torch.float)

        return torch.multinomial(
            scaled_logits, num_samples=1
        )
