from torch.utils.data import DataLoader
from torch.optim import AdamW

from typing import Any, Dict

from llm.llm.architecture.gpt_model import GPTModel
from llm.llm.pipelines.evaluation.evaluator import Evaluator
from llm.llm.pipelines.inference.text_generator import TextGenerator

"""
This Trainer class is responsible for thr whole training pipeline

Args:
    model: GPTModel -> The initialized model to be trained
    trainer_cfg: Dict[str, Any] -> The dictionary with the trainer
        configuration.
    device: str -> cuda, cpu, etc

Returns:
    None
"""


class Trainer():
    def __init__(
            self,
            model: GPTModel,
            trainer_cfg: Dict[str, Any],
            device: str
            ) -> None:

        # The initialized model to be trained
        self.model: GPTModel = model

        # The text generator class that uses the trained model
        # to generate new text
        text_generator: TextGenerator = TextGenerator(
            model=self.model,
            context_length=trainer_cfg["context_length"]
            )

        # The device on which to train
        self.device: str = device

        # The trainer pipeline evaluator
        self.evaluator: Evaluator = Evaluator(self.model, device=device)

        # The optimizer for the trining pipeline
        self.optimizer: AdamW = AdamW(
            model.parameters(),
            lr=trainer_cfg["lr_rate"],
            weight_decay=trainer_cfg["weight_decay"]
            )

    def train():
        pass

    def save_model():
        pass

