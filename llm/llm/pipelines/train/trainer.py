import datetime
import torch
import pickle

from datetime import timedelta
from torch.utils.data import DataLoader
from torch.optim import AdamW

from typing import Any, Dict, List

from llm.llm import logger
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
            text_generator: TextGenerator,
            trainer_cfg: Dict[str, Any],
            device: str
            ) -> None:

        # Trainer configuration
        self.trainer_cfg: Dict[str, Any] = trainer_cfg

        # The initialized model to be trained
        self.model: GPTModel = model.to(device)

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
            weight_decay=self.trainer_cfg["weight_decay"]
            )

        logger.info("Trainer initialized with the following configuration:")
        for key in self.trainer_cfg.keys():
            logger.info(f"{key}: {self.trainer_cfg[key]}")

    def train(
            self,
            train_loader: DataLoader,
            validation_loader: DataLoader,
            start_context: str
    ) -> Dict[str, List[Any]]:
        # Initialize training performance parameters stores.
        train_losses, validation_losses = [], []
        track_tokens_seen, texts_generated = [], []
        start_training: datetime.datetime = datetime.datetime.now()

        # Initialize training progress tracking variables
        tokens_seen, global_step = 0, -1
        best_loss: float = float('inf')

        # Retrieves the number of epochs to iterate
        num_epochs: int = self.trainer_cfg["num_epochs"]

        # Retrieves the evaluation frequency
        eval_freq: int = self.trainer_cfg["eval_freq"]

        # Retrieves the number of batches to use in the evaluation
        eval_num_batches: int = self.trainer_cfg["eval_iter"]

        # The training loop
        for epoch in range(num_epochs):

            # tracks processing time for batches
            start_epoch: datetime.datetime = datetime.datetime.now()

            # Epoch batches progress monitor
            epoch_batches: int = 0

            # Set the mode of the model to train
            self.model.train()

            # tracks the amount of batches processed
            start_batch: datetime.datetime = datetime.datetime.now()

            # For each batch
            for input_batch, target_batch in train_loader:

                # reset the gradients
                self.optimizer.zero_grad()

                # computes the batch loss
                loss: torch = self.evaluator.calculate_batch_loss(
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

                # end of batch
                elapsed = datetime.datetime.now() - start_batch

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluator.evaluate_model(
                        train_loader=train_loader,
                        validation_loader=validation_loader,
                        eval_iter=eval_num_batches
                    )

                    # updates training performance data
                    train_losses.append(train_loss)
                    validation_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    # logs the progress
                    logger.info(
                        f"Epoch: {epoch + 1} "
                        f"(Batches: {epoch_batches:06d}) "
                        f"(Step {global_step:06d}): "
                        f"Train loss {train_loss:.6f}, "
                        f"Val loss {val_loss:.6f}, "
                        f"Elapsed {(timedelta(
                            elapsed.days,
                            elapsed.seconds))}"
                        )

            # logs epoch final losses
            logger.info(
                f"Epoch: {epoch + 1} "
                f"(Batches: {epoch_batches:05d}) "
                f"(Step {global_step:06d}): "
                f"Train loss {train_loss:.6f}, "
                f"Val loss {val_loss:.6f} ,"
                f"Elapsed {(timedelta(
                    elapsed.days,
                    elapsed.seconds))}"
                )

            text_generated: str = self.text_generator.generate_text(
                start_context=start_context
            )

            # Update texts generated
            texts_generated.append(text_generated)

            # Logs the generated text
            logger.info(f"Epoch: {epoch + 1} Text ->"
                        f"\n[\"{text_generated}\"]")

            # If val_loss improves, saves model as best one.
            if best_loss > val_loss:
                self.save_model('llm/models/best_gpt_model.pth')
                best_loss = val_loss

            # saves model.
            self.save_model('llm/models/gpt_model.pth')

            # Serializes tracking variables
            self.serialize_objects(
                train_losses=train_losses,
                validation_losses=validation_losses,
                track_tokens_seen=track_tokens_seen,
                texts_generated=texts_generated
            )

            # register the moment epoch finishes
            end_epoch: datetime.datetime = datetime.datetime.now()

            # logs epoch processing time
            elapsed_time: datetime.timedelta = end_epoch - start_epoch
            logger.info(
                f"Epoch: {epoch + 1} "
                f"Processing time: {datetime.timedelta(
                    elapsed_time.days, elapsed_time.seconds
                )}"
            )

        # register the moment epoch finishes
        end_training: datetime.datetime = datetime.datetime.now()

        # logs epoch processing time
        elapsed_time = end_training - start_training

        logger.info(
            f"Training processing time: {datetime.timedelta(
                elapsed_time.days, elapsed_time.seconds
            )}"
        )

        # Serializes tracking variables
        self.serialize_objects(
            train_losses=train_losses,
            validation_losses=validation_losses,
            track_tokens_seen=track_tokens_seen,
            texts_generated=texts_generated
        )

        return train_losses, validation_losses, \
            track_tokens_seen, texts_generated

    def save_model(self, model_name: str) -> None:
        torch.save(self.model.state_dict(), model_name)
        logger.info(f"Model saved at: {model_name}")

    def serialize_objects(self,
                          train_losses: List[float],
                          validation_losses: List[float],
                          track_tokens_seen: List[int],
                          texts_generated: List[str]):

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
