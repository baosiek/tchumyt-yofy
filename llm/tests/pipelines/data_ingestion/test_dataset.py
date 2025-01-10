import pytest

from typing import Any, Dict
from bson.objectid import ObjectId

from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset


@pytest.fixture
def pathname() -> str:
    return "llm/configs/dataset_loader_config.yaml"


@pytest.fixture
def query() -> Dict[str, Any]:
    return {"_id": ObjectId("678001d24570296256278926")}


def test_dateset_len(pathname: str, query: Dict[str, Any]):
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    dataset: CrawlDataset = CrawlDataset(client=client, query=query)

    assert dataset.__len__() == 1


def test_dataset_getItem(pathname: str, query: Dict[str, Any]):
    client: TchumytMongoClient = TchumytMongoClient(pathname)
    dataset: CrawlDataset = CrawlDataset(client=client, query=query)
    x, y = dataset.__getitem__(0)

    assert len(x) == 256
    assert len(y) == 256
