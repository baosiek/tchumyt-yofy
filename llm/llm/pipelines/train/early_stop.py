import torch

from llm.llm import logger
from llm.llm.architecture.gpt.gpt_model import GPTModel


class EarlyStop():
    def __init__(
            self,
            patience: int,
            delta: float,
            best_model_path: str
    ) -> None:
        '''
        Early stop initialization
        Input:
            patience: int -> number of epochs before early stopping
            delta: float ->  minimum amount of improvement in the loss
                to NOT early stop
            best_model_path: str -> location where to save the best model

        Return:
            None
        '''
        self.patience: int = patience
        self.delta: float = delta

        # Holds the best score value
        self.best_score: float = None

        # Flags the early stop
        self.early_stop: bool = False

        # The counter up to patience
        self.counter: int = 0

        # Location of best model
        self.best_model_path: str = best_model_path

    def __call__(
            self,
            model: GPTModel,
            validation_loss: float
    ) -> None:

        '''
        Executes the logic of this class
        Input:
            validation_loss: float -> the validation loss to consider

        Return:
            None
        '''

        # First iteration sets the best score and saves the model
        if self.best_score is None:
            self.best_score = validation_loss
            self.save_model(model)

        # If validations has no improvement
        elif validation_loss < self.best_score + self.delta:

            # Update counter
            self.counter += 1

            # Checks to see if patience limit has been achieved
            # Sets early stop to True in case it has
            if self.counter >= self.patience:
                self.early_stop = True
        # Validation was improved
        else:
            self.best_score = validation_loss
            self.save_model(model=model)

            # Resets the counter
            self.counter = 0

    def save_model(self, model: GPTModel):
        torch.save(model.state_dict(), self.best_model_path)
        logger.info(f"Best model saved at: {self.best_model_path}")
