import mlflow
import torch
import os
import shutil

from typing import Any, Dict, List, Tuple
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from mlflow import MlflowClient


from llm.llm import logger, cfg, init_cfg
from llm.llm.architecture.abstract_model import AbstractModel
from llm.llm.architecture.rnn.rnn_model import RNNModelV1
from llm.llm.pipelines.train.trainer_v1 import TrainerV1
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.inference.text_generator import TextGenerator
from llm.llm.pipelines.data_ingestion.data_loader import \
    create_crawl_dataset_loader
from llm.llm.components.decoding_strategies import TopKScaling, \
      AbstractDecodeStrategy


def get_loaders(query: Dict[str, Any] = None, limit: int = None) -> \
        Tuple[DataLoader, DataLoader]:
    # 1. Load datasets
    # 1.1 Initializes MongoDB client
    client: TchumytMongoClient = TchumytMongoClient(
        "llm/configs/init_config.yaml",
    )

    # 1.2 Generator to enabling split dataset into train and validation subsets
    generator1: torch.Generator = torch.Generator().manual_seed(918)

    # 1.3 Creates a list with both subsets, 90% training, 10% evaluation
    datasets: List[Subset] = torch.utils.data.random_split(
        CrawlDataset(client=client, limit=limit, query=query),
        [0.9, 0.1],
        generator=generator1,
    )

    # 1.4 Assigns train and validation datasets accordingly
    train_dataset: Subset = datasets[0]
    validation_dataset: Subset = datasets[1]

    # logger.info(f"Configuration cfg type: {cfg.keys()}")
    # logger.info(f"vocabulary_size: {cfg['vocabulary_size']}")

    # 1.5 Creates train and validation dataloaders
    train_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False
    )

    validation_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=validation_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False
    )

    # 1.5 Log their sizes
    logger.info(f"Train loader length: {len(list(train_loader))}")
    logger.info(f"Validation loader length: {len(list(validation_loader))}")

    return (train_loader, validation_loader)


def main(run_name: str) -> bool:
    # Set the device on which the model will be trained
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Add a query to filter the dataset.
    # TODO:Must adjust Dataset schema at MongoDB
    # Get loaders
    train_loader, validation_loader = get_loaders(limit=1000)

    if len(list(train_loader)) == 0 or len(list(validation_loader)) == 0:
        logger.error(
            "Train loader and/or validation loader is empty."
        )
        return False

    # Start context
    start_context: str = "Trump is the president of the"

    # Initialize model
    model: AbstractModel = RNNModelV1(
        cfg=cfg, device=device
    )

    # Sets the strategy for decoding
    decode_strategy: AbstractDecodeStrategy = TopKScaling(
        topk_k=cfg["top_k"], temperature=cfg["temperature"]
    )

    # Initializes text generator based with model initialized
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=cfg["context_length"],
        encoding=cfg["tiktoken_encoding"],
        decode_strategy=decode_strategy,
    )

    # Initialize trainer
    trainer: TrainerV1 = TrainerV1(
        model=model,
        text_generator=text_generator,
        trainer_cfg=cfg,
        device=device
    )

    with mlflow.start_run(
        run_name=run_name,
        description="Testing LSTM on RNN"
    ):
        mlflow.enable_system_metrics_logging()

        # Logs training parameters
        mlflow.log_params(cfg)

        # Log model summary to both local and MLFlow.
        with open("llm/reports/model_summary.txt", "w") as f:
            f.write(str(summary(model)))

        mlflow.log_artifact("llm/reports/model_summary.txt")

        train_losses, validation_losses, track_tokens_seen, _ = trainer.train(
            train_loader=train_loader,
            validation_loader=validation_loader,
            start_context=start_context,
        )

        # Initialize metrics
        metrics: Dict[str, Any] = {
            "train_loss": train_losses[-1],
            "validation_loss": validation_losses[-1],
            "track_tokens_seen": track_tokens_seen[-1],
        }

        # Log metrics that were calculated during training
        mlflow.log_metrics(metrics)

        '''
        While saving the model to MLFlow, the function
        mlflow.pytorch.log_model() is corrupting the logger
        file. It will take time to find why and correct it.

        The following code backups the log file as it is now.
        '''
        # Copy the contents of the source file to the destination file
        shutil.copyfile("llm/logs/project.log", "llm/logs/project.bak.log")

        # Define an artifact path that the model will be saved to.
        artifact_path_model = f"tchumyt/model/{init_cfg["collection"]}"
        # Logs the model
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path_model,
            registered_model_name=init_cfg["collection"],
        )

        # Recover original log file
        shutil.copyfile("llm/logs/project.bak.log", "llm/logs/project.log")

        # Check if the file exists before attempting to delete it
        if os.path.exists("llm/logs/project.bak.log"):
            os.remove("llm/logs/project.bak.log")

        # Saves the training log
        artifact_path_log = "logs"
        mlflow.log_artifact(
            "llm/logs/project.log",
            artifact_path_log
        )

        # Saves the training performance
        artifact_path_objects = "objects"
        mlflow.log_artifact(
            "llm/pickle_objects/train_tracking_objects.pkl",
            artifact_path_objects
        )

    return True


if __name__ == "__main__":
    # Use the fluent API to set the tracking uri and the active experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Sets the current active experiment to the "Politics GPTModel"
    # experiment and returns the Experiment metadata
    _experiment = mlflow.set_experiment("TMYTS Model")

    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.
    run_name = "training 001"

    # FIXME: artifact_path not recognized \
    # Define an artifact path that the model will be saved to.
    artifact_path = f"mlflow-artifacts:/tchumyt/model/{init_cfg["collection"]}"

    if not main(run_name):
        logger.error("Training failed. Exiting.")
        exit(1)

    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = client.get_latest_versions(
        init_cfg["collection"]
    )[0]
    client.set_registered_model_alias(
        init_cfg["collection"], "challenger", model_info.version
    )
    client.set_model_version_tag(
        name=init_cfg["collection"],
        version=model_info.version,
        key="nlp",
        value="text_generation",
    )

    logger.info("The End")
