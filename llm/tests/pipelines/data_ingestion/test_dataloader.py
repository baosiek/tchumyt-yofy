import pytest

from typing import Dict, Any
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
def parameters() -> Dict[str, Any]:
    batch_size: int = 8
    shuffle: bool = False
    return {"batch_size": batch_size, "shuffle": shuffle}


def test_crawl_dataset_loader(pathname: str,
                              query: Dict[str, Any],
                              parameters: Dict
                              ):

    client: TchumytMongoClient = TchumytMongoClient(pathname)
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
