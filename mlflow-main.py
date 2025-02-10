import mlflow
import torch

from typing import Any, Dict, List, Tuple
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from mlflow import MlflowClient


from llm.llm import logger, model_cfg, trainer_cfg
from llm.llm.architecture.gpt_model import GPTModel
from llm.llm.pipelines.train.trainer import Trainer
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.inference.text_generator import TextGenerator
from llm.llm.pipelines.data_ingestion.data_loader import \
     create_crawl_dataset_loader


def get_loaders() -> Tuple[DataLoader, DataLoader]:

    # 1. Load datasets
    # 1.1 Initializes MongoDB client
    client: TchumytMongoClient = TchumytMongoClient(
        "llm/configs/dataset_loader_config.yaml",
    )

    # 1.2 Generator to enabling split dataset into train and validation subsets
    generator1: torch.Generator = torch.Generator().manual_seed(918)

    # 1.3 Creates a list with both subsets, 90% training, 10% evaluation
    datasets: List[Subset] = torch.utils.data.random_split(
        CrawlDataset(client=client),
        [0.9, 0.1],
        generator=generator1
    )

    # 1.4 Assigns train and validation datasets accordingly
    train_dataset: Subset = datasets[0]
    validation_dataset: Subset = datasets[1]

    # 1.5 Creates train and validation dataloaders
    train_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=train_dataset,
        batch_size=trainer_cfg["batch_size"],
        shuffle=False
    )

    validation_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=validation_dataset,
        batch_size=trainer_cfg["batch_size"],
        shuffle=False
    )

    # 1.5 Log their sizes
    logger.info(f"Train loader length: {len(list(train_loader))}")
    logger.info(f"Validation loader length: {len(list(validation_loader))}")

    return (train_loader, validation_loader)


def main():

    # Set the device on which the model will be trained
    device: str = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Get loaders
    train_loader, validation_loader = get_loaders()

    # Start context
    start_context: str = "Trump is the president of the United"

    # Initialize model
    model: GPTModel = GPTModel(cfg=model_cfg)

    # Initializes text generator based with model initialized
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=trainer_cfg["context_length"],
        encoding=trainer_cfg["tiktoken_encoding"]
    )

    # Initialize trainer
    trainer: Trainer = Trainer(
        model=model,
        text_generator=text_generator,
        trainer_cfg=trainer_cfg,
        device=device
    )

    with mlflow.start_run(run_name=run_name):

        mlflow.enable_system_metrics_logging()

        # Logs training parameters
        mlflow.log_params(trainer_cfg)

        # Log model summary to both local and MLFlow.
        with open("llm/reports/model_summary.txt", "w") as f:
            f.write(str(summary(model)))

        mlflow.log_artifact("llm/reports/model_summary.txt")

        train_losses, validation_losses, \
            track_tokens_seen, _ = trainer.train(
                train_loader=train_loader,
                validation_loader=validation_loader,
                start_context=start_context
            )

        # Initialize metrics
        metrics: Dict[str, Any] = {
            "train_loss": train_losses[-1],
            "validation_loss": validation_losses[-1],
            "track_tokens_seen": track_tokens_seen[-1]
        }

        # Log metrics that were calculated during training
        mlflow.log_metrics(metrics)

        metadata = {"description": "this is a test",
                    "Description": "This is also a test"}

        # Logs the model
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name="gpt_model_politics",
            metadata=metadata
            )

        # Saves the training log
        mlflow.log_artifacts("llm/logs/project.log")

        # Saves the training performance
        mlflow.log_artifacts("llm/pickle_objects/train_tracking_objects.pkl")


if __name__ == "__main__":

    # Use the fluent API to set the tracking uri and the active experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Sets the current active experiment to the "Politics GPTModel"
    # experiment and returns the Experiment metadata
    _experiment = mlflow.set_experiment("Politics GPTModel")

    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.
    run_name = "politics_gptmodel_test"

    # Define an artifact path that the model will be saved to.
    artifact_path = \
        "mlflow-artifacts:/585006454050763634/model/politics_gptmodel"

    main()

    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = client.get_latest_versions('gpt_model_politics')[0]
    client.set_registered_model_alias(
        "gpt_model_politics",
        "challenger",
        model_info.version
    )
    client.set_model_version_tag(
        name='gpt_model_politics',
        version=model_info.version,
        key='nlp',
        value='text_generation'
    )

    logger.info("The End!")
