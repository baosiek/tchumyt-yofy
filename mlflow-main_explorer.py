import mlflow
import torch
import shutil

from typing import Any, Dict, List, Tuple
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from mlflow import MlflowClient


from llm.llm import logger, cfg, init_cfg
from llm.llm.architecture.tmyts.tmyts_llm_explorer import TymysLLM
from llm.llm.pipelines.train.trainer_v1 import TrainerV1
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.inference.text_generator_hfbpe import TextGenerator
from llm.llm.tokenizers.bpe_tokenizer import HFBPETokenizer
from llm.llm.pipelines.data_ingestion.data_loader import \
    create_crawl_dataset_loader
from llm.llm.components.decoding_strategies import AbstractDecodeStrategy, \
    get_decoder_factory

# mlflow.enable_system_metrics_logging()


def get_loaders(query: Dict[str, Any] = None, limit: int = 0) -> \
        Tuple[DataLoader, DataLoader]:
    # 1. Load datasets
    # 1.1 Initializes MongoDB client
    client: TchumytMongoClient = TchumytMongoClient(
        "llm/configs/init_config.yaml",
    )

    # 1.2 Generator to enabling split dataset into train and validation subsets
    generator1: torch.Generator = torch.Generator().manual_seed(918)

    # 1.3 Loads dataset
    dataset: CrawlDataset = CrawlDataset(
        client=client, limit=limit, query=query
    )

    # 1.3 Creates a list with both subsets, 90% training, 10% evaluation
    logger.info("Splitting dataset into train and validation subsets...")
    datasets: List[Subset] = torch.utils.data.random_split(
        dataset,
        [0.95, 0.05],
        generator=generator1,
    )

    # 1.4 Assigns train and validation datasets accordingly
    train_dataset: Subset = datasets[0]
    validation_dataset: Subset = datasets[1]
    logger.info(f"Train dataset length: {len(train_dataset)}")
    logger.info(f"Validation dataset length: {len(validation_dataset)}")

    # 1.5 Creates train and validation dataloaders
    logger.info("Creating train and validation dataloaders...")
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

    # 1.6 Log their sizes
    logger.info(f"Train loader length: {len(list(train_loader))}")
    logger.info(f"Validation loader length: {len(list(validation_loader))}")

    return (train_loader, validation_loader)


def main(
        run_name: str,
        limit: int = 0,
        decode_strategy: str = "greedy_decoding"
) -> str:
    # Set the device on which the model will be trained
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Add a query to filter the dataset.
    # TODO: Must adjust Dataset schema at MongoDB
    # Get loaders
    train_loader, validation_loader = get_loaders(limit=limit)

    if len(list(train_loader)) == 0 or len(list(validation_loader)) == 0:
        logger.error(
            "Train loader and/or validation loader is empty."
        )
        return False

    # Start context
    start_context: str = "Trump met for nearly two"
    # two hours with President Joe Biden in the Oval Office

    # Gets hyperparameters from configuration file
    hidden_dim: int = cfg["embedding_dim"]
    seq_length: int = cfg["context_length"]
    vocabulary_size: int = cfg["vocabulary_size"]
    dropout_rate: float = cfg["drop_rate"]
    num_layers: int = cfg["num_layers"]
    stride: int = cfg["model_stride"]
    kernel_size: int = cfg["kernel_size"]

    # Initialize model
    model: TymysLLM = TymysLLM(
         hidden_dim=hidden_dim,
         seq_length=seq_length,
         vocabulary_size=vocabulary_size,
         dropout_rate=dropout_rate,
         num_layers=num_layers,
         stride=stride,
         kernel_size=kernel_size
    )

    decode_strategy: AbstractDecodeStrategy = get_decoder_factory(
        "greedy_decoding"
    )

    # Initializes Tokenizer
    tokenizer: HFBPETokenizer = HFBPETokenizer(
        tokenizer_path="llm/resources/bpe_tokenizer.json"
    )

    # Initializes text generator based with model initialized
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=cfg["context_length"],
        tokenizer=tokenizer,
        decode_strategy=decode_strategy,
    )

    # Initialize trainer
    trainer: TrainerV1 = TrainerV1(
        model=model,
        text_generator=text_generator,
        trainer_cfg=cfg,
        device=device,
        to_early_stop=False
    )

    description: str = '''
    First Model to compete with GPT-2
    '''

    with mlflow.start_run(
        run_name=run_name,
        description=description,
        log_system_metrics=True
    ) as run:

        # Logs training parameters
        mlflow.log_params(cfg)

        # Log model summary to both local and MLFlow.
        with open("llm/reports/model_summary.txt", "w") as f:
            f.write(str(summary(model)))

        mlflow.log_artifact("llm/reports/model_summary.txt")

        # Starts the training
        _, _, track_tokens_seen, _ = trainer.train(
            train_loader=train_loader,
            validation_loader=validation_loader,
            start_context=start_context,
        )

        # Copy the contents of the source file to the destination file
        shutil.copyfile("llm/logs/training.log", "llm/logs/training_1.log")

        # Define an artifact path that the model will be saved to.
        artifact_path_model = f"tchumyt/model/{init_cfg["collection"]}"
        # Logs the model
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path_model,
            registered_model_name=init_cfg["collection"],
        )

        # TODO: There is a bug in track_tokens_seen as it is coming empty
        # Initialize metrics
        if len(track_tokens_seen) > 0:
            metrics: Dict[str, Any] = {
                "track_tokens_seen": track_tokens_seen[-1],
            }

            # Log metrics that were calculated during training
            mlflow.log_metrics(metrics)

    # Clear GPU cache
    torch.cuda.empty_cache()
    
    run_id: str = run.info.run_id
    return run_id


if __name__ == "__main__":
    # Use the fluent API to set the tracking uri and the active experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Sets the current active experiment to the "Politics GPTModel"
    # experiment and returns the Experiment metadata
    _experiment = mlflow.set_experiment(
        "TMYTS Explorer"
    )

    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.
    run_name = "Model TMYTS_0_2 - run: 02"

    # FIXME: artifact_path not recognized \
    # Define an artifact path that the model will be saved to.
    artifact_path = f"mlflow-artifacts:/tchumyt/model/{init_cfg["collection"]}"

    run_id: str = main(
        run_name, limit=560000, decode_strategy="greedy_decoding"
    )

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

    # # Saves the training log
    artifact_path_log = "logs"
    client.log_artifact(
        run_id,
        "llm/logs/training_1.log",
    )
    client.log_artifact(
        run_id,
        "llm/pickle_objects/train_tracking_objects.pkl",
    )

    logger.info("The End")
