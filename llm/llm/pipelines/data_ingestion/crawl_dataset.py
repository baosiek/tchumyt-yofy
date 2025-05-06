from torch import tensor
from torch.utils.data import Dataset
from typing import List, Dict, Any

from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm import logger


class CrawlDataset(Dataset):
    def __init__(
            self,
            client: TchumytMongoClient,
            query: Dict[str, Any] = None,
            limit: int = 0
    ):
        super().__init__()
        self.client: TchumytMongoClient = client

        logger.info("Building dataset...")

        if query is None and limit == 0:
            logger.info(
                "No query and limit was defined, fetching all records..."
            )
        else:
            if limit == 0:
                logger.info(f"Retrieving all records with query: {query}")
            else:
                logger.info(f"Retrieving {limit} records with query: {query}")

        self.records: List[str] = list(client.query(query, limit=limit))

        client.close()

        logger.info(f"Records retrieved: {len(self.records)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return tensor(self.records[index]['x']), \
               tensor(self.records[index]['y'])
