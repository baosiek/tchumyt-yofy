import pymongo
import os

from typing import Any, Dict
from pymongo.cursor import Cursor

from llm.llm.utils.commons import read_yaml


class TchumytMongoClient():
    def __init__(self, config: str):
        configuration: Dict[str, Any] = read_yaml(config)

        self.dataset_name: str = configuration["dataset"]
        self.collection_name: str = configuration["collection"]

        # Gets username
        username: str = os.getenv("MONGO_INITDB_ROOT_USERNAME")
        password: str = os.getenv("MONGO_INITDB_ROOT_PASSWORD")

        if username is None or password is None:
            raise ValueError("Environment variables MONGO_INITDB_ROOT_USERNAME"
                             " and/or MONGO_INITDB_ROOT_PASSWORD are not set.")

        # builds database uri
        uri: str = (f"mongodb://{username}:{password}"
                    f"@{configuration['mongo_host']}:"
                    f"{configuration['mongo_port']}")

        self.client: pymongo.MongoClient = pymongo.MongoClient(uri)
        self.database = self.client[self.dataset_name]
        self.collection = self.database[self.collection_name]

    def query(self, query: str = None, limit: int = 0) -> Cursor:

        if query is None:
            return self.collection.find().limit(limit=limit)
        else:
            self.collection.find(query).limit(limit=limit)

    def close(self):
        self.client.close()
