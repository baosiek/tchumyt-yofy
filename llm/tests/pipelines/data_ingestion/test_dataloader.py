import pytest
import json

from typing import List, Dict, Any
from torch.utils.data import DataLoader
from bson.objectid import ObjectId

from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.data_loader import \
     create_crawl_dataset_loader
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset


@pytest.fixture
def pathname() -> str:
    return "llm/configs/dataset_loader_config.yaml"


@pytest.fixture
def query() -> Dict[str, Any]:
    return {"_id": ObjectId("678001d24570296256278926")}


@pytest.fixture
def mock_data() -> List[Dict[str, Any]]:
    with open("llm/resources/testing.json", 'r') as file:
        mock_data = json.load(file)
        return mock_data


@pytest.fixture
def parameters() -> Dict[str, Any]:
    batch_size: int = 8
    shuffle: bool = False
    return {"batch_size": batch_size, "shuffle": shuffle}


def test_crawl_dataset_loader(mock_data, parameters: Dict,
                              mocker
                              ):

    # Create a mock response object with a .query()
    # method that returns the mock data
    mock_response = mocker.MagicMock()
    mock_response.query.return_value = mock_data

    # # Patch 'TchumytMongoClient.query' to return the mock response
    mocker.patch(
        "llm.llm.utils.tchumyt_mongo_client.TchumytMongoClient.query",
        return_value=mock_response.query.return_value
    )

    client: TchumytMongoClient = TchumytMongoClient(
        "llm/configs/dataset_loader_config.yaml"
    )

    dataset: CrawlDataset = CrawlDataset(client=client)

    loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=dataset,
        batch_size=parameters['batch_size'],
        shuffle=parameters['shuffle']
    )

    iterator: iter = iter(loader)
    first_batch = next(iterator)

    assert len(first_batch) == 2
    assert len(first_batch[0]) == parameters['batch_size']
