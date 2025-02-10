import numpy as np
import pickle
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from typing import List

from llm import logger


def plot_training_performance(
        epochs: np.ndarray,
        tokens_seen: List[float],
        train_losses: List[float],
        validation_losses: List[float]
        ):

    '''
    This method plots the training performance chart

    Inputs:
        epochs: numpy.ndarray -> a list of numbers from 0 to length of losses
        tokens_seen: List[int] -> a list of the total number of tokens seen
            up to the epoch
        train_losses: List[float] -> a list with the training losses
            for each epoch
        validation_losses: List[float] -> a list with the validation losses
            for each epoch

    Returns:
        None

    '''
    # A figure with one plot
    fig, ax1 = plt.subplots(figsize=(7, 4))

    # The first line reflecting the training loss
    ax1.plot(epochs, train_losses, label="Training loss")

    # The second line reflecting the validation loss
    ax1.plot(
        epochs,
        validation_losses,
        linestyle="-.",
        label="Validation loss"
    )

    # Set the label for both x and y axis
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    # Set the location of the legend
    ax1.legend(loc="upper right")

    # Set the location of the ticker 
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Sets the second x axis, ie, the upper horizontal axis
    ax2 = ax1.twiny()

    # Plots the second axis with the number of tokens seen
    ax2.plot(tokens_seen, train_losses, alpha=0)

    # Sets second axis label
    ax2.set_xlabel("Tokens seen")

    # Renders the chart
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    logger.info("Rendering training performance plot")

    with open("llm/pickle_objects/train_tracking_objects.pkl", 'rb') as file:
        objects = pickle.load(file=file)
        epochs: np.ndarray = np.arange(0, len(objects["train_losses"]))
        plot_training_performance(
            epochs=epochs,
            tokens_seen=objects["track_tokens_seen"],
            train_losses=objects["train_losses"],
            validation_losses=objects["validation_losses"]
        )
