import torch

from llm.llm.architecture.gpt_model import GPTModel


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

    def calculate_epoch_loss():
        pass
