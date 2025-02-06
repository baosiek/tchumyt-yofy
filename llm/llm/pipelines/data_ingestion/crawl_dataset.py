from torch import tensor
from torch.utils.data import Dataset
from typing import List

from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient


class CrawlDataset(Dataset):
    def __init__(self, client: TchumytMongoClient, query: str = None):
        super().__init__()
        self.client: TchumytMongoClient = client
        self.records: List[str] = list(client.query(query))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return tensor(self.records[index]['x']), \
               tensor(self.records[index]['y'])
