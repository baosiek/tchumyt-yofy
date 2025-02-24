import pytest
import torch
import json

from typing import Tuple, List, Dict, Any
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from llm.llm import logger, model_cfg
from llm.llm.pipelines.evaluation.evaluator import Evaluator
from llm.llm.architecture.gpt.gpt_model import GPTModel
from llm.llm.pipelines.inference.text_inferencer import TextProvider
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.data_ingestion.data_loader import \
     create_crawl_dataset_loader


torch.manual_seed(123)


@pytest.fixture
def loaders(mocker, mock_data) -> Tuple[DataLoader, DataLoader]:
    # Create a mock response object with a .query()
    # method that returns the mock data
    mock_response = mocker.MagicMock()
    mock_response.query.return_value = mock_data

    # Patch 'requests.get' to return the mock response
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

    return (train_loader, validation_loader)


@pytest.fixture
def model() -> GPTModel:
    model: GPTModel = GPTModel(cfg=model_cfg)
    return model


@pytest.fixture()
def in_an_out_cpu() -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:

    input_sentences: List[str] = ["every effort moves",
                                  "I really like"]
    target_sentences: List[str] = [" effort moves you",
                                   " really like chocolate"]

    cross_entropy: float = 10.296357

    device: str = "cpu"

    text_provider: TextProvider = TextProvider(cfg=model_cfg)

    inputs: torch.Tensor = torch.tensor([], dtype=int)
    targets: torch.Tensor = torch.tensor([], dtype=int)

    for sent in input_sentences:
        tk_ids = text_provider.text_to_token_ids(sent)
        inputs = torch.cat((inputs, tk_ids))

    for sent in target_sentences:
        tk_ids = text_provider.text_to_token_ids(sent)
        targets = torch.cat((targets, tk_ids))

    cross_entropy_tensor: torch.Tensor = torch.tensor(cross_entropy)

    return (inputs, targets, cross_entropy_tensor, device)


@pytest.fixture()
def in_an_out_gpu(in_an_out_cpu) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:

    cross_entropy: float = 10.384483

    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cross_entropy_tensor: torch.Tensor = torch.tensor(cross_entropy)

    return (in_an_out_cpu[0], in_an_out_cpu[1], cross_entropy_tensor, device)


@pytest.fixture
def mock_data() -> List[Dict[str, Any]]:
    with open("llm/resources/testing.json", 'r') as file:
        mock_data = json.load(file)
        return mock_data


def test_calculate_batch_loss_with_cpu(
      in_an_out_cpu: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
      model: GPTModel
      ) -> None:

    logger.debug(f"Input: {in_an_out_cpu[0]}")
    logger.debug(f"Target: {in_an_out_cpu[1]}")

    evaluator: Evaluator = Evaluator(model, device=in_an_out_cpu[3])

    with torch.no_grad():
        batch_loss = evaluator.calculate_batch_loss(in_an_out_cpu[0],
                                                    in_an_out_cpu[1])

    assert batch_loss.to(device="cpu").numpy() == \
        in_an_out_cpu[2].to(device="cpu").numpy()


def test_calculate_batch_loss_with_gpu(
      in_an_out_gpu: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]
      ) -> None:

    logger.debug(f"Input: {in_an_out_gpu[0]}")
    logger.debug(f"Target: {in_an_out_gpu[1]}")

    model = GPTModel(cfg=model_cfg)

    evaluator: Evaluator = Evaluator(model, device=in_an_out_gpu[3])

    with torch.no_grad():
        batch_loss = evaluator.calculate_batch_loss(in_an_out_gpu[0],
                                                    in_an_out_gpu[1])

    assert batch_loss.to(device="cpu").numpy() == \
        in_an_out_gpu[2].to(device="cpu").numpy()


def test_calculate_epoch_loss(
        model: GPTModel,
        in_an_out_gpu,
        mock_data: List[Dict[str, Any]],
        mocker
        ):

    # Create a mock response object with a .query()
    # method that returns the mock data
    mock_response = mocker.MagicMock()
    mock_response.query.return_value = mock_data

    # Patch 'requests.get' to return the mock response
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

    logger.info(f"Device set to: [{in_an_out_gpu[3]}]")

    evaluator: Evaluator = Evaluator(model, device=in_an_out_gpu[3])

    with torch.no_grad():
        train_loss = evaluator.calculate_epoch_loss(train_loader)
        validation_loss = evaluator.calculate_epoch_loss(train_loader)

    logger.info(f"Training loss: {train_loss:.5f}")
    logger.info(f"Training loss: {validation_loss:.5f}")

    assert train_loss == pytest.approx(11.02646)
    assert validation_loss == pytest.approx(11.02582)


def test_evaluate_model(
        model: GPTModel,
        loaders: Tuple[DataLoader, DataLoader],
        in_an_out_gpu
        ):

    logger.info(f"Device set to: [{in_an_out_gpu[3]}]")
    evaluator: Evaluator = Evaluator(model, device=in_an_out_gpu[3])

    train_loss, validation_loss = evaluator.evaluate_model(
        loaders[0],
        loaders[1],
        eval_iter=5
        )

    logger.info(f"Training loss: {train_loss:.5f}")
    logger.info(f"Training loss: {validation_loss:.5f}")

    assert train_loss == pytest.approx(11.02892)
    assert validation_loss == pytest.approx(11.04301)
