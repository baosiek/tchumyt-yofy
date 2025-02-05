import torch

from typing import Tuple
from torch.utils.data import DataLoader

from llm.llm.architecture.gpt_model import GPTModel
from llm.llm import logger


class Evaluator():
    def __init__(self, model: GPTModel, device: str):
        self.model: GPTModel = model.to(device=device)
        self.device: str = device

    def calculate_batch_loss(
            self,
            input_batch: torch.Tensor,
            target_batch: torch.Tensor
            ) -> torch.Tensor:

        input_batch: torch.Tensor = input_batch.to(device=self.device)
        target_batch: torch.Tensor = target_batch.to(device=self.device)

        logits: torch.Tensor = self.model(input_batch)
        loss: torch.Tensor = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten()
        )

        return loss

    def calculate_epoch_loss(
            self,
            data_loader: DataLoader,
            num_batches: int = None
            ) -> float:
        total_loss: float = 0.0

        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        logger.debug(f"Number of batches to evaluate: {num_batches}")

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss: torch.Tensor = self.calculate_batch_loss(
                    input_batch=input_batch,
                    target_batch=target_batch
                    )
                total_loss += loss.item()
            else:
                break

        return total_loss / num_batches

    def evaluate_model(self,
                       train_loader: DataLoader,
                       validation_loader: DataLoader,
                       eval_iter: int) -> Tuple[float, float]:
        self.model.eval()

        with torch.no_grad():
            train_loss: float = self.calculate_epoch_loss(
                data_loader=train_loader, num_batches=eval_iter
            )

            validation_loss: float = self.calculate_epoch_loss(
                data_loader=validation_loader,
                num_batches=eval_iter
            )

        self.model.train()

        return (train_loss, validation_loss)
