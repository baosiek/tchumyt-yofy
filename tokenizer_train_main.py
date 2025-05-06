from typing import Any, Dict, List
from datasets import load_dataset, DatasetDict
from llm.llm.tokenizers.bpe_tokenizer import HFBPETokenizer


def load_additional_tokens(path: str) -> List[str]:
    '''
    The load_additional_tokens function loads a list of additional tokens from
    a file.

    Args:
        path (str): The path to the file containing the additional tokens.

    Returns:
        List[str]: A list of additional tokens.
    '''
    with open(path, 'r') as f:
        return f.read().splitlines()


def load_datasets(dataset_dir: str, filename: str) -> DatasetDict:
    '''
    The load_datasets function loads a dataset from a file from
    Hugging Face's datasets library.
    Args:
        dataset_dir (str): The directory containing the dataset file.
        filename (str): The name of the dataset file.

    Returns:
        DatasetDict: A DatasetDict object containing the dataset.
    '''

    datasets: DatasetDict = load_dataset(dataset_dir, filename)
    return datasets


def main(cfg: Dict[str, Any]) -> None:
    '''
    The main function trains a byte-pair encoding tokenizer on a dataset and
    saves the trained tokenizer to a file.'''

    additional_tokens: List[str] = load_additional_tokens(
        path=cfg['path_to_token_list']
    )

    datasets: DatasetDict = load_datasets(
        dataset_dir=cfg['dataset_dir'],
        filename=cfg['filename']
    )

    bpe_tokenizer: HFBPETokenizer = HFBPETokenizer()
    bpe_tokenizer.train(
        vocab_size=cfg['vocab_size'],
        save_to_path=cfg['save_to_path'],
        add_tokens=additional_tokens,
        dataset=datasets
    )


if __name__ == '__main__':

    cfg: Dict[str, Any] = {
        'path_to_token_list': 'llm/resources/emoji_list.txt',
        'dataset_dir': 'Salesforce/wikitext',
        'filename': 'wikitext-103-raw-v1',
        'vocab_size': 35000,
        'save_to_path': 'llm/resources/bpe_tokenizer.json'
    }

    main(cfg=cfg)
