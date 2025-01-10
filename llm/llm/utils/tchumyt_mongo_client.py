import pymongo
import os

from typing import Any, Dict
from pymongo.cursor import Cursor

from llm.llm.utils.commons import read_yaml
from llm import logger


class TchumytMongoClient():
    def __init__(self, config: str):
        configuration: Dict[str, Any] = read_yaml(config)

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

        logger.info(f"MongoDB URI: {uri}")

        self.client: pymongo.MongoClient = pymongo.MongoClient(uri)

        self.database = self.client[configuration["dataset"]]
        self.collection = self.database[configuration['collection']]

    def query(self, query: str = None) -> Cursor:
        if query is None:
            return self.collection.find()
        else:
            return self.collection.find(query)
