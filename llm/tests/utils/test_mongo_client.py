import pytest

from bson.objectid import ObjectId
from utils.tchumyt_mongo_client import TchumytMongoClient
from pymongo.cursor import Cursor
from typing import Any, Dict, List


@pytest.fixture
def pathname() -> str:
    return "llm/configs/dataset_config.yaml"


@pytest.fixture
def query() -> Dict[str, Any]:
    return {"_id": ObjectId("678001d24570296256278926")}


def test_client_init_function(pathname: str):
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    client.configuration['database'] == "crawl_dataset"


def test_query_function_one(pathname: str, query: Dict[str, Any]):
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    cursor: Cursor = client.query(query)
    records: List[str] = list(cursor)
    assert len(records) == 1


def test_query_function_all(pathname: str):
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    cursor: Cursor = client.query()
    records: List[str] = list(cursor)
    assert len(records) == 5736
