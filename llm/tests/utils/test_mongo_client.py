import pytest
import json

from bson.objectid import ObjectId
from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from pymongo.cursor import Cursor
from typing import Any, Dict, List


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


def test_client_init_function(pathname: str):
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    client.database == "crawl_dataset"


def test_query_function_one(
        pathname: str,
        query: Dict[str, Any],
        mock_data,
        mocker):

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

    cursor: Cursor = client.query(query)
    records: List[str] = list(cursor)
    assert len(records) == 1000


def test_query_function_all(pathname: str, mock_data, mocker):
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
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    cursor: Cursor = client.query()
    records: List[str] = list(cursor)
    assert len(records) == 1000
