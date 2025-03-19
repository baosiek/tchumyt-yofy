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
            limit: int = None
    ):
        super().__init__()
        self.client: TchumytMongoClient = client

        if query is None:
            logger.info("Querying DB...")
        else:
            logger.info(f"Querying DB with query: {query}")

        self.records: List[str] = list(client.query(query))
        logger.info(f"Records retrieved: {len(self.records)}")

        if limit is None:
            self.records: List[str] = list(client.query(query))
        else:
            self.records: List[str] = list(client.query_with_limit(
                query, limit=limit)
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return tensor(self.records[index]['x']), \
               tensor(self.records[index]['y'])
