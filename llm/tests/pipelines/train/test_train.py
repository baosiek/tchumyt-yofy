import pytest
import torch
import json

from typing import Tuple, Dict, List, Any
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from llm.llm.pipelines.train.train import train_model
from llm.llm.architecture.gpt_model import GPTModel
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.data_ingestion.data_loader import \
     create_crawl_dataset_loader
from llm.llm.pipelines.evaluation.evaluator import Evaluator
from llm.llm.pipelines.inference.text_inferencer import TextProvider
from llm.llm import cfg, logger


@pytest.fixture()
def start_context() -> str:
    return "Every step moves you"


@pytest.fixture()
def model() -> GPTModel:
    model: GPTModel = GPTModel(cfg=cfg)
    return model


@pytest.fixture
def loaders() -> Tuple[DataLoader, DataLoader]:
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

    return (train_loader, validation_loader)


@pytest.fixture
def mock_data() -> List[Dict[str, Any]]:
    with open("llm/resources/testing.json", 'r') as file:
        mock_data = json.load(file)
        return mock_data


def test_train(
        start_context: str,
        model: GPTModel,
        mock_data: List[Dict[str, Any]],
        mocker
        ) -> None:
    
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
    
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    evaluator: Evaluator = Evaluator(model, device=device)
    text_provider: TextProvider = TextProvider(cfg=cfg)

    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device, num_epochs=1,
        eval_iter=5,
        eval_freq=10,
        start_context=start_context,
        optimizer=optimizer,
        evaluator=evaluator,
        text_provider=text_provider
        )
