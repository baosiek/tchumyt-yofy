from torch.utils.data import DataLoader

from llm.llm.pipelines.data_ingestion.crawl_dataset import CrawlDataset


# Helper method to create dataset loader
def create_crawl_dataset_loader(
        crawl_dataset: CrawlDataset,
        batch_size: int,
        shuffle: bool = False
        ) -> DataLoader:

    # Initialize the data loader
    crawl_dataset_loader: DataLoader = DataLoader(
        crawl_dataset,
        batch_size=batch_size,
        shuffle=shuffle
        )

    # Returns dataloader
    return crawl_dataset_loader
