import pytest
import torch
import json

from typing import Tuple, Dict, List, Any
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from llm.llm.architecture.abstract_model import AbstractModel
from llm.llm.architecture.rnn.rnn_model import RNNModelV1
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.data_ingestion.data_loader import \
     create_crawl_dataset_loader
from llm.llm.pipelines.train.trainer import Trainer
from llm.llm.pipelines.inference.text_generator import TextGenerator
from llm.llm.components.decoding_strategies import AbstractDecodeStrategy, \
    TopKScaling

from llm.llm import trainer_cfg, logger


@pytest.fixture()
def start_context() -> str:
    return "Trump is the president"


@pytest.fixture()
def mock_cfg_model() -> Dict[str, Any]:

    mock_cfg_model: Dict[str, Any] = {
        "name": "RNNModelV1",
        "vocabulary_size": 50257,
        "context_length": 256,
        "embedding_dim": 1024,
        "num_layers": 6,
    }

    return mock_cfg_model


@pytest.fixture()
def mock_cfg_data() -> Dict[str, Any]:

    mock_cfg_data: Dict[str, Any] = {
        "name": "Trainer for GPTModel",
        "batch_size": 256,
        "lr_rate": 0.0004,
        "weight_decay": 0.1,
        "num_epochs": 2,
        "eval_freq": 100,
        "temperature": 0.1,
        "eval_iter": 100,
        "context_length": 16,
        "patience": 2,
        "delta": 1.0000,
        "tiktoken_encoding": "gpt2"
    }
    return mock_cfg_data


@pytest.fixture()
def model(mock_cfg_model: Dict[str, Any]) -> AbstractModel:
    model: AbstractModel = RNNModelV1(
        cfg=mock_cfg_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    return model.to(device="cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_data() -> List[Dict[str, Any]]:
    with open("llm/resources/testing.json", 'r') as file:
        mock_data = json.load(file)
        return mock_data


@pytest.fixture
def decode_strategy() -> AbstractDecodeStrategy:
    decode_strategy: TopKScaling = TopKScaling(temperature=0.5, topk_k=3)
    return decode_strategy


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
        model: AbstractModel,
        decode_strategy

        ) -> None:

    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=decode_strategy
    )

    trainer: Trainer = Trainer(
        model=model,
        text_generator=text_generator,
        trainer_cfg=trainer_cfg,
        device=device
    )

    assert trainer.model._get_name() == "RNNModelV1"

    logger.info(
        "Trainer was initialized for model: "
        f"{trainer.model._get_name()}"
    )


def test_trainer_train_method_no_early_stop(
          start_context: str,
          loaders: Tuple[DataLoader, DataLoader],
          model: AbstractModel,
          decode_strategy: AbstractDecodeStrategy,
          mock_cfg_data: Dict[str, Any],
        ) -> None:

    device: str = torch.device(
             "cuda" if torch.cuda.is_available() else "cpu"
             )

    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=decode_strategy
    )

    trainer: Trainer = Trainer(
        model=model,
        text_generator=text_generator,
        trainer_cfg=mock_cfg_data,
        device=device
    )

    _, _, _, texts_generated = \
        trainer.train(
            loaders[0],
            loaders[1],
            start_context
        )

    assert len(texts_generated) == mock_cfg_data["num_epochs"]


def test_trainer_train_method_early_stopping(
          start_context: str,
          loaders: Tuple[DataLoader, DataLoader],
          model: AbstractModel,
          mock_cfg_data: Dict[str, Any],
          decode_strategy: AbstractDecodeStrategy,
          mocker
        ) -> None:

    device: str = torch.device(
             "cuda" if torch.cuda.is_available() else "cpu"
             )

    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=decode_strategy
    )

    trainer: Trainer = Trainer(
        model=model,
        text_generator=text_generator,
        trainer_cfg=mock_cfg_data,
        device=device,
        early_stopping=True
    )

    _, _, _, texts_generated = \
        trainer.train(
            loaders[0],
            loaders[1],
            start_context
        )

    assert len(texts_generated) == mock_cfg_data["num_epochs"]
