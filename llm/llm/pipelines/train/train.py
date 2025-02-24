import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from llm.llm import logger
from llm.llm.architecture.gpt.gpt_model import GPTModel
from llm.llm.pipelines.evaluation.evaluator import Evaluator
from llm.llm.pipelines.inference.text_inferencer import TextProvider


def train_model(
        model: GPTModel,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        text_provider: TextProvider,
        device: str,
        num_epochs: int,
        eval_freq: int,
        eval_iter: int,
        start_context: str,
        evaluator: Evaluator,
        optimizer: AdamW):

    train_losses, validation_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    logger.info(f"Train loader size: {len(iter(train_loader))}")

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss: torch.Tensor = evaluator.calculate_batch_loss(
                input_batch=input_batch, target_batch=target_batch
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluator.evaluate_model(
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                    eval_iter=eval_iter
                )
                train_losses.append(train_loss)
                validation_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                logger.info(f"Epoch: {epoch + 1} (Step {global_step:06d}): "
                            f"Train loss {train_loss:.3f}, "
                            f"Val loss {val_loss:.3f}")

        text_produced: str = text_provider.produce_text(
            start_context=start_context
        )
        logger.info(f"Model outlook: {text_produced}")
