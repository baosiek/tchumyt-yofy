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
        self.__name__: str = "greedy_decoding"

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
        self.__name__: str = "temperature_scaling"

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        scaled_logits: torch.Tensor = logits / torch.tensor(self.temperature,
                                                            dtype=torch.float)
        scaled_logits = torch.softmax(scaled_logits, dim=-1)

        return torch.multinomial(
            scaled_logits, num_samples=1
        )


class TopKScaling(AbstractDecodeStrategy):
    def __init__(
            self,
            topk_k: int,
            temperature: float,
            num_samples: int = 1
    ):
        super().__init__()
        self.top_k: int = topk_k
        self.temperature: float = temperature
        self.num_samples: int = num_samples
        self.__name__: str = "top_k_sampling"

    def decode(self, logits: torch.Tensor) -> torch.Tensor:

        # Ranks the logits, returning the top k in descending order
        top_logits, top_pos = torch.topk(logits, self.top_k)

        # The lowest value is the last one
        lowest_logit: torch.Tensor = top_logits[0, -1]

        # New_logits is logits with only the top 3 values. The other
        # logits are set to -inf.
        new_logits: torch.Tensor = torch.where(
            condition=logits < lowest_logit,
            input=torch.tensor(float('-inf')),
            other=logits
        )

        # The new probability distribution
        new_logits = torch.softmax(new_logits, dim=0)

        # Scales the probabilities
        scaled_logits: torch.Tensor = logits / self.temperature

        # Turn scaled values into a probability distribution
        probabilities: torch.Tensor = torch.softmax(scaled_logits, dim=-1)

        # Draws the next token from a multinomial distribution and
        # returns it
        return torch.multinomial(probabilities, num_samples=1)
