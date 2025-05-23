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

import datetime
import pickle
from typing import Any, Dict, List, Tuple

import mlflow
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm.llm import logger
from llm.llm.architecture.abstract_model import AbstractModel
from llm.llm.pipelines.evaluation.evaluator import Evaluator
from llm.llm.pipelines.inference.text_generator import TextGenerator
from llm.llm.pipelines.train.early_stop import EarlyStop
from llm.llm.utils.commons import filter_string
from llm.llm.utils.format_timedelta import format_time_delta


class TrainerV1:
    """TODO: Documentation"""

    def __init__(
        self,
        model: AbstractModel,
        text_generator: TextGenerator,
        trainer_cfg: Dict[str, Any],
        device: str,
        to_early_stop: bool = False,
    ) -> None:
        # Trainer configuration
        self.trainer_cfg: Dict[str, Any] = trainer_cfg
        logger.info(f"Trainer configuration: {self.trainer_cfg}")

        # The initialized model to be trained
        self.model: AbstractModel = model.to(device)

        # The text generator class that uses the trained model
        # to generate new text
        self.text_generator: TextGenerator = text_generator

        # The device on which to train
        self.device: str = device

        # The trainer pipeline evaluator
        self.evaluator: Evaluator = Evaluator(self.model, device=device)

        # The optimizer for the trining pipeline
        self.optimizer: AdamW = AdamW(
            model.parameters(),
            lr=self.trainer_cfg["lr_rate"],
            weight_decay=self.trainer_cfg["weight_decay"],
        )

        # # Weather or not to early stop
        # self.early_stopping: bool = early_stopping
        # Initializes early stopping
        self.early_stop: EarlyStop = None
        if to_early_stop:
            self.early_stop = EarlyStop(
                patience=self.trainer_cfg["patience"],
                delta=self.trainer_cfg["delta"],
                best_model_path="llm/models/best_gpt_model.pth",
            )

        logger.info("Trainer initialized with the following configuration:")
        for key in self.trainer_cfg.keys():
            logger.info(f"{key}: {self.trainer_cfg[key]}")

    def evaluate(
        self, train_loader: DataLoader, validation_loader: DataLoader, eval_iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: Documentation"""

        train_loss, val_loss = self.evaluator.evaluate_model(
            train_loader=train_loader,
            validation_loader=validation_loader,
            eval_iter=eval_iter,
        )

        return train_loss, val_loss

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        start_context: str,
    ) -> Dict[str, List[Any]]:
        """TODO: Documentation"""

        # Logs the start of the training
        logger.info(f"Starting training on {self.device}...")

        # Initialize training performance parameters stores.
        train_losses, validation_losses = [], []
        track_tokens_seen, texts_generated = [], []
        start_training: datetime.datetime = datetime.datetime.now()

        # Initialize training progress tracking variables
        tokens_seen, global_step = 0, 0
        best_loss: float = None

        if self.early_stop is None:
            best_loss: float = float("inf")

        # Retrieves the number of epochs to iterate
        num_epochs: int = self.trainer_cfg["num_epochs"]

        # Retrieves the evaluation frequency
        eval_freq: int = self.trainer_cfg["eval_freq"]

        # Retrieves the number of batches to use in the evaluation
        eval_num_batches: int = self.trainer_cfg["eval_iter"]

        # Number of batches
        num_batches: int = len(list(train_loader))

        if num_batches < eval_freq:
            logger.warning(
                f"Evaluation frequency [{eval_freq}] is "
                f"bigger than number of batches: [{num_batches}]. "
                "Train loss, validation loss and tokens seen will not be registered"
            )

        # Total global steps
        total_global_steps: int = num_batches * num_epochs

        # Generated text before training
        text_generated: str = self.text_generator.generate_text(
            start_context=start_context
        )

        logger.info("Text generated before training ->")
        logger.info(f"[{filter_string(text_generated)}]")

        # tracks processing time for evaluation
        eval_start: datetime.datetime = datetime.datetime.now()

        # Evaluates the model before training and logs result just for fun
        train_loss, val_loss = self.evaluate(
            train_loader=train_loader,
            validation_loader=validation_loader,
            eval_iter=eval_num_batches,
        )

        # end of eval_freq
        eval_elapsed: datetime.timedelta = (
            datetime.datetime.now() - eval_start
        )

        mlflow.log_metric("train_loss", train_loss, step=0)
        mlflow.log_metric("val_loss", val_loss, step=0)

        # logs the first loss with untrained model
        logger.info(
            f"No Train:"
            f"({0:06d}/{num_batches:06d}) "
            f"(Global {global_step:07d}/"
            f"{total_global_steps:07d}): "
            f"Train loss {train_loss:12.8f}, "
            f"Val loss {val_loss:12.8f}, "
            f"Time eval_freq {format_time_delta(eval_elapsed)}"
        )

        # The training loop
        for epoch in range(num_epochs):
            # tracks processing time for batches
            start_epoch: datetime.datetime = datetime.datetime.now()

            # Epoch batches progress monitor
            epoch_batches: int = 0

            # Set the mode of the model to train
            self.model.train()

            # tracks the amount of batches processed
            start_eval_freq: datetime.datetime = datetime.datetime.now()

            # For each batch
            for input_batch, target_batch in train_loader:
                # reset the gradients
                self.optimizer.zero_grad()

                # computes the batch loss
                loss: torch.Tensor = self.evaluator.calculate_batch_loss(
                    input_batch=input_batch, target_batch=target_batch
                )

                # Back propagate
                loss.backward()

                # Adjusts all model parameters based on loss back propagation
                self.optimizer.step()

                # update training progress variables
                tokens_seen += input_batch.numel()
                global_step += 1
                epoch_batches += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate(
                        train_loader=train_loader,
                        validation_loader=validation_loader,
                        eval_iter=eval_num_batches,
                    )

                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)

                    # updates training performance data
                    train_losses.append(train_loss)
                    validation_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    # end of eval_freq
                    end_eval_freq: datetime.timedelta = (
                        datetime.datetime.now() - start_eval_freq
                    )

                    # logs the progress
                    logger.info(
                        f"Epoch: {epoch + 1} "
                        f"({epoch_batches:06d}/{num_batches:06d}) "
                        f"(Global {global_step:07d}/"
                        f"{total_global_steps:07d}): "
                        f"Train loss {train_loss:12.8f}, "
                        f"Val loss {val_loss:12.8f}, "
                        f"Time eval_freq {format_time_delta(end_eval_freq)}"
                    )

                    # Resets eval_freq chronograph
                    start_eval_freq: datetime.datetime = datetime.datetime.now()

            # Epochs last evaluation
            train_loss, val_loss = self.evaluate(
                train_loader=train_loader,
                validation_loader=validation_loader,
                eval_iter=eval_num_batches,
            )

            # Computes the time of the end of the epoch
            epoch_end: datetime.timedelta = datetime.datetime.now() - start_epoch

            # logs epoch's final losses
            logger.info(
                f"Final Epoch: {epoch + 1} "
                f"({epoch_batches:06d}/{num_batches:06d}) "
                f"(Global {global_step:07d}/{total_global_steps:07d}): "
                f"Train loss {train_loss:12.8f}, "
                f"Val loss {val_loss:12.8f} ,"
                f"Time {format_time_delta(epoch_end)}"
            )

            # saves model.
            self.save_model("llm/models/gpt_model.pth")

            # Serializes tracking variables
            self.serialize_objects(
                train_losses=train_losses,
                validation_losses=validation_losses,
                track_tokens_seen=track_tokens_seen,
                texts_generated=texts_generated,
            )

            text_generated: str = self.text_generator.generate_text(
                start_context=start_context
            )

            # Update texts generated
            texts_generated.append(text_generated)

            # logs epoch processing time
            elapsed_time: datetime.timedelta = datetime.datetime.now() - start_epoch
            logger.info(
                f"Epoch: {epoch + 1} Processing time: {format_time_delta(elapsed_time)}"
            )

            # Resets start epochs
            start_epoch = datetime.datetime.now()

            # Checks if early stopping is set
            if self.early_stop is not None:
                # Process early stop logic
                self.early_stop(model=self.model, validation_loss=val_loss)

                # If patience limit achieved
                if self.early_stop.early_stop:
                    logger.info("Early stopping. Exiting training.")
                    break

            else:
                # If val_loss improves, saves model as best one.
                if best_loss > val_loss:
                    self.save_model("llm/models/best_gpt_model.pth")
                    best_loss = val_loss

            # Logs the generated text
            logger.info(f"Epoch: {epoch + 1} Text ->")
            logger.info(f"[{filter_string(text_generated)}]")

        # register the moment epoch finishes
        end_training: datetime.datetime = datetime.datetime.now()

        # logs epoch processing time
        elapsed_time: datetime.timedelta = end_training - start_training

        logger.info(f"Training processing time: {format_time_delta(elapsed_time)}")

        return train_losses, validation_losses, track_tokens_seen, texts_generated

    def save_model(self, model_name: str) -> None:
        torch.save(self.model.state_dict(), model_name)
        logger.info(f"Model saved at: {model_name}")

    def serialize_objects(
        self,
        train_losses: List[float],
        validation_losses: List[float],
        track_tokens_seen: List[int],
        texts_generated: List[str],
    ):
        # Insert objects into a dictionary
        objects_to_serialize: Dict[str, List[Any]] = {}
        objects_to_serialize["train_losses"] = train_losses
        objects_to_serialize["validation_losses"] = validation_losses
        objects_to_serialize["track_tokens_seen"] = track_tokens_seen
        objects_to_serialize["texts_generated"] = texts_generated

        # serializes the dictionary
        path_name: str = "llm/pickle_objects/train_tracking_objects.pkl"
        with open(path_name, "wb") as file:
            pickle.dump(objects_to_serialize, file=file)

        logger.info(f"Tracking objects saved at : {path_name}")
