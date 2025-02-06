import pytest
import json

from typing import Any, Dict, List
from bson.objectid import ObjectId

from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
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


def test_dateset_len(pathname: str, query: Dict[str, Any], mock_data, mocker):
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
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    dataset: CrawlDataset = CrawlDataset(client=client, query=query)

    assert dataset.__len__() == 1000


def test_dataset_getItem(
        pathname: str,
        query: Dict[str, Any],
        mock_data,
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
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    dataset: CrawlDataset = CrawlDataset(client=client, query=query)
    x, y = dataset.__getitem__(0)

    assert len(x) == 256
    assert len(y) == 256
