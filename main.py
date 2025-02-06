import torch

from typing import List
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

from llm.llm.utils.tchumyt_mongo_client import TchumytMongoClient
from llm.llm.architecture.gpt_model import GPTModel
from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset
from llm.llm.pipelines.data_ingestion.data_loader import \
     create_crawl_dataset_loader
from llm.llm.pipelines.inference.text_generator import TextGenerator
from llm.llm.pipelines.train.trainer import Trainer

from llm.llm import logger, cfg, trainer_cfg

if __name__ == "__main__":

    # 1. Load datasets
    # 1.1 Initializes MongoDB client
    client: TchumytMongoClient = TchumytMongoClient(
        "llm/configs/dataset_loader_config.yaml"
    )

    # 1.2 Generator to enabling split dataset into train and validation subsets
    generator1: torch.Generator = torch.Generator().manual_seed(918)

    # 1.3 Creates a list with both subsets, 90% training, 10% evaluation
    datasets: List[Subset] = torch.utils.data.random_split(
        CrawlDataset(client=client),
        [0.9, 0.1],
        generator=generator1
    )

    # 1.4 Assigns train and validation datasets accordingly
    train_dataset: Subset = datasets[0]
    validation_dataset: Subset = datasets[1]

    # 1.5 Creates train and validation dataloaders
    train_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=train_dataset,
        batch_size=trainer_cfg["batch_size"],
        shuffle=False
    )

    validation_loader: DataLoader = create_crawl_dataset_loader(
        crawl_dataset=validation_dataset,
        batch_size=trainer_cfg["batch_size"],
        shuffle=False
    )

    # 1.5 Log their sizes
    logger.info(f"Train loader length: {len(list(train_loader))}")
    logger.info(f"Validation loader length: {len(list(validation_loader))}")

    # 3. Initializes the model to be trained
    model: GPTModel = GPTModel(cfg=cfg)

    # 4. Initializes text generator based with model initialized
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=trainer_cfg["context_length"],
        encoding=trainer_cfg["tiktoken_encoding"]
    )

    # 5. Initializes the trainer
    # 5.1 Set the device on which the model will be trained
    device: str = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 5.2 Initializes the trainer
    trainer: Trainer = Trainer(
        model=model,
        text_generator=text_generator,
        trainer_cfg=trainer_cfg,
        device=device
    )

    # trainer.train(
    #     train_loader=train_loader,
    #     validation_loader=validation_loader,
    #     start_context="Trump is the president of the United"
    # )
