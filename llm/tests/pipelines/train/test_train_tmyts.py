import pytest
import torch
import json

from typing import Tuple, Dict, List, Any
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from llm.llm.architecture.tmyts.tmyts_llm import TymysLLM
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.data_ingestion.data_loader import \
     create_crawl_dataset_loader
from llm.llm.pipelines.train.trainer_v1 import TrainerV1
from llm.llm.pipelines.inference.text_generator import TextGenerator
from llm.llm.components.decoding_strategies import AbstractDecodeStrategy, \
    TopKScaling

from llm.llm import logger, cfg


@pytest.fixture()
def start_context() -> str:
    return "President Trump of the United"


@pytest.fixture()
def mock_cfg() -> Dict[str, Any]:

    mock_cfg: Dict[str, Any] = {
        "name": "RNNModelV2",
        "vocabulary_size": 50257,
        "context_length": 64,
        "embedding_dim": 1024,
        "num_layers": 6,
        "batch_size": 8,
        "lr_rate": 0.0004,
        "weight_decay": 0.1,
        "num_epochs": 10,
        "eval_freq": 100,
        "temperature": 0.1,
        "eval_iter": 100,
        "patience": 2,
        "delta": 1.0000,
        "tiktoken_encoding": "gpt2",
        "stride": 1
    }

    return mock_cfg


@pytest.fixture()
def model(mock_cfg: Dict[str, Any]) -> TymysLLM:
    model: TymysLLM = TymysLLM(
        hidden_dim=1024,
        seq_length=8,
        vocabulary_size=50257,
        dropout_rate=0.5,
        num_heads=4
    )
    return model.to(device="cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_data() -> List[Dict[str, Any]]:
    # Context length of this mock data is 256
    with open("llm/resources/test_RNNMODEL_LSTM_1_1.json", 'r') as file:
        mock_data = json.load(file)
        return mock_data


@pytest.fixture
def decode_strategy() -> AbstractDecodeStrategy:
    decode_strategy: TopKScaling = TopKScaling(temperature=0.5, topk_k=3)
    return decode_strategy


@pytest.fixture
def loaders(mock_data,
            mock_cfg: Dict[str, Any],
            mocker
            ) -> Tuple[DataLoader, DataLoader]:

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
        "llm/configs/init_config.yaml"
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
        batch_size=mock_cfg["batch_size"],
        shuffle=False
    )

    validation_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=validation_dataset,
        batch_size=mock_cfg["batch_size"],
        shuffle=False
    )

    logger.info(f"Train loader length: {len(list(train_loader))}")
    logger.info(f"Validation loader length: {len(list(validation_loader))}")

    return (train_loader, validation_loader)


def test_trainer_initialization(
        start_context: str,
        model: TymysLLM,
        decode_strategy
        ) -> None:

    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=decode_strategy
    )

    trainer: TrainerV1 = TrainerV1(
        model=model,
        text_generator=text_generator,
        trainer_cfg=cfg,
        device=device
    )

    assert trainer.model._get_name() == "TymysLLM"

    logger.info(
        "Trainer was initialized for model: "
        f"{trainer.model._get_name()}"
    )


def test_trainer_train_method_no_early_stop(
          start_context: str,
          loaders: Tuple[DataLoader, DataLoader],
          model: TymysLLM,
          decode_strategy: AbstractDecodeStrategy,
          mock_cfg: Dict[str, Any],
        ) -> None:

    device: str = torch.device(
             "cuda" if torch.cuda.is_available() else "cpu"
             )

    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=mock_cfg['context_length'],
        encoding="gpt2",
        decode_strategy=decode_strategy
    )

    trainer: TrainerV1 = TrainerV1(
        model=model,
        text_generator=text_generator,
        trainer_cfg=mock_cfg,
        device=device
    )

    _, _, _, texts_generated = \
        trainer.train(
            loaders[0],
            loaders[1],
            start_context
        )

    assert len(texts_generated) == mock_cfg["num_epochs"]
