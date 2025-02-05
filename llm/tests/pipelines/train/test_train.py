import pytest
import torch
import json

from typing import Tuple, Dict, List, Any
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from llm.llm.architecture.gpt_model import GPTModel
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.data_ingestion.data_loader import \
     create_crawl_dataset_loader
from llm.llm.pipelines.train.trainer import Trainer
from llm.llm.pipelines.inference.text_generator import TextGenerator

from llm.llm import cfg, trainer_cfg, logger


@pytest.fixture()
def start_context() -> str:
    return "Trump is the president"


@pytest.fixture()
def model() -> GPTModel:
    model: GPTModel = GPTModel(cfg=cfg)
    return model


@pytest.fixture
def mock_data() -> List[Dict[str, Any]]:
    with open("llm/resources/testing.json", 'r') as file:
        mock_data = json.load(file)
        return mock_data


@pytest.fixture
def loaders(mock_data, mocker) -> Tuple[DataLoader, DataLoader]:

    # Create a mock response object with a .query()
    # method that returns the mock data
    mock_response = mocker.MagicMock()
    mock_response.query.return_value = mock_data

    # # Patch 'requests.get' to return the mock response
    mocker.patch(
        "llm.llm.utils.tchumyt_mongo_client.TchumytMongoClient.query",
        return_value=mock_response.query.return_value
    )

    client: TchumytMongoClient = TchumytMongoClient(
        "llm/configs/dataset_loader_config.yaml"
    )

    # Generator to enabling split dataset into train and validation subsets
    generator1: torch.Generator = torch.Generator().manual_seed(42)

    # Creates a list with both subsets
    dataset: List[Subset] = torch.utils.data.random_split(
        CrawlDataset(client=client),
        [0.9, 0.1],
        generator=generator1
        )

    # Assigns train and validation datasets accordingly
    train_dataset: Subset = dataset[0]
    validation_dataset: Subset = dataset[1]

    train_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=train_dataset,
        batch_size=8,
        shuffle=False
    )

    validation_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=validation_dataset,
        batch_size=8,
        shuffle=False
    )

    logger.info(f"Train loader length: {len(list(train_loader))}")
    logger.info(f"Validation loader length: {len(list(validation_loader))}")

    return (train_loader, validation_loader)


def test_trainer_initialization(
        start_context: str,
        model: GPTModel,
        loaders
        ) -> None:

    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_generator: TextGenerator = TextGenerator(
        model=model, context_length=1024, encoding="gpt2"
    )

    trainer: Trainer = Trainer(
        model=model,
        text_generator=text_generator,
        trainer_cfg=trainer_cfg,
        device=device
    )

    assert trainer.model._get_name() == "GPTModel"

    logger.info(
        "Trainer was initialized for model: "
        f"{trainer.model._get_name()}"
    )


def test_trainer_train_method(
          start_context: str,
          loaders: Tuple[DataLoader, DataLoader],
          model: GPTModel
        ) -> None:

    device: str = torch.device(
             "cuda" if torch.cuda.is_available() else "cpu"
             )

    text_generator: TextGenerator = TextGenerator(
        model=model, context_length=1024, encoding="gpt2"
    )

    trainer: Trainer = Trainer(
        model=model,
        text_generator=text_generator,
        trainer_cfg=trainer_cfg,
        device=device
    )

    train_losses, validation_losses, track_tokens_seen, texts_generated = \
        trainer.train(
            loaders[0],
            loaders[1],
            start_context
        )

    assert len(texts_generated) == trainer_cfg["num_epochs"]
