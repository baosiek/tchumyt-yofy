from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from datasets import DatasetDict, Dataset
from typing import List, Generator, Any

from llm.llm import logger

'''
The HFBPETokenizer class is a wrapper around the Tokenizer class from
the Hugging Face Tokenizers library.
It provides a simple interface to train a Byte-Pair Encoding (BPE) tokenizer
on a dataset and to encode and decode text.

Args:
    special_tokens (List[str]): A list of special tokens to be used by the
        tokenizer. Default is ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"].

    tokenizer_path (str): The path to a pre-trained tokenizer. Default is None,
        which means a new tokenizer will be created.

Returns:
    None
'''


class HFBPETokenizer():
    def __init__(
            self,
            special_tokens: List[str] = ["[UNK]", "[CLS]", "[SEP]", "[PAD]",
                                         "[MASK]"],
            tokenizer_path: str = None
            ):

        '''
        The __init__ method initializes the HFBPETokenizer class.
        It creates a new Tokenizer object with a Byte-Pair Encoding (BPE) model
        and a Whitespace pre-tokenizer. If a tokenizer_path is provided, the
        tokenizer will be loaded from the file at that path. Otherwise this
        BPE tokenizer has to be trained first

        '''
        if tokenizer_path is not None:
            self.tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
            self.is_trained = True
        else:
            self.tokenizer: Tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            self.special_tokens = special_tokens
            self.is_trained = False

    def train(
            self,
            vocab_size: int,
            save_to_path: str,
            add_tokens: List[str] = None,
            dataset: DatasetDict = None,
            train_dataset: Dataset = None,
            test_dataset: Dataset = None,
            val_dataset: Dataset = None
            ) -> None:
        '''
        The train method trains the tokenizer on a dataset. The dataset can be
        provided as a DatasetDict object or as separate train, test, and
        validation datasets. The tokenizer is trained with a BpeTrainer object
        and the resulting tokenizer is saved to a file at the save_to_path.

        Args:
            vocab_size (int): The size of the vocabulary to be used by the
                tokenizer.
            save_to_path (str): The path to save the trained tokenizer to.
            dataset (DatasetDict): A DatasetDict object containing train, test,
                and validation datasets. Default is None.
            train_dataset (Dataset): A Dataset object containing the training
                data. Default is None.
            test_dataset (Dataset): A Dataset object containing the test data.
                Default is None.
            val_dataset (Dataset): A Dataset object containing the validation
                data. Default is None.
            add_tokens (List[str]): A list of additional tokens to add to the
                tokenizer. Default is None.

        Returns:
            None
        '''

        if add_tokens is not None:
            self.tokenizer.add_tokens(add_tokens)

        if dataset is not None:
            self.train_dataset: Dataset = dataset["train"]
            self.test_dataset: Dataset = dataset["test"]
            self.val_dataset: Dataset = dataset["validation"]
        else:
            self.train_dataset: Dataset = train_dataset
            self.test_dataset: Dataset = test_dataset
            self.val_dataset: Dataset = val_dataset

        if dataset is None and (train_dataset is None
                                or test_dataset is None
                                or val_dataset is None):
            raise ValueError("You must provide either a dataset or a train, "
                             "test, and validation dataset")

        trainer: BpeTrainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens
        )

        logger.info("Training tokenizer")
        self.tokenizer.train_from_iterator(
            self.dataset_iterator_(),
            trainer=trainer
        )

        logger.info("Saving tokenizer")

        # "/home/baosiek/Projects/tchumyt-yofy/llm/resources/wiki_tokenizer.json"
        self.tokenizer.save(save_to_path)
        self.is_trained = True

    def encode(self, text) -> List[int]:
        '''
        The encode method encodes a text string into a list of token ids using
        the trained tokenizer.

        Args:
            text (str): The text to be encoded.

        Returns:
            List[int]: A list of token ids representing the encoded text.
        '''
        if self.is_trained is False:
            raise ValueError(
                "You must train the tokenizer before encoding text"
            )
        return self.tokenizer.encode(text).ids

    def decode(self, tokens) -> str:
        '''
        The decode method decodes a list of token ids into a text string using
        the trained tokenizer.

        Args:
            tokens (List[int]): A list of token ids to be decoded.

        Returns:
            str: The decoded text string.
        '''

        if self.is_trained is False:
            raise ValueError(
                "You must train the tokenizer before encoding text"
            )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def dataset_iterator_(self) -> Generator[Any, Any, Any]:
        data_iterator: List[Dataset] = [self.train_dataset,
                                        self.test_dataset,
                                        self.val_dataset]
        '''
        The dataset_iterator method is a generator that yields text strings
        from a dataset. It is used to train the tokenizer on a dataset.

        Returns:
            Generator[Any, Any, Any]: A generator that yields text strings from
                a dataset.

            '''
        for data in data_iterator:
            for i, data in enumerate(data):
                text = data.get("text", None)
                # text cannot be None and must be a string
                if isinstance(text, str) and text is not None:
                    text = text.strip()  # removes leading and trailing spaces
                    if len(text) > 0:  # only text with content
                        if text.startswith("="):  # removes '=' from headings
                            text = text.replace("=", "").strip()
                else:
                    continue

                # Yield the text
                yield text
