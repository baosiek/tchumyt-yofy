import pytest
from tokenizers import Tokenizer
from datasets import Dataset, DatasetDict
from llm.llm.tokenizers.bpe_tokenizer import HFBPETokenizer


@pytest.fixture
def sample_dataset():
    data = {
        "train": Dataset.from_dict({"text": ["Hello world", "Test sentence"]}),
        "test": Dataset.from_dict({"text": ["Another test", "More data"]}),
        "validation": Dataset.from_dict({"text": ["Validation text"]}),
    }
    return DatasetDict(data)


@pytest.fixture
def tokenizer():
    return HFBPETokenizer()


def test_tokenizer_initialization_without_path():
    tokenizer = HFBPETokenizer()
    assert tokenizer.is_trained is False
    assert isinstance(tokenizer.tokenizer, Tokenizer)


def test_tokenizer_initialization_with_path(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = HFBPETokenizer()
    tokenizer.tokenizer.save(str(tokenizer_path))
    loaded_tokenizer = HFBPETokenizer(tokenizer_path=str(tokenizer_path))
    assert loaded_tokenizer.is_trained is True


def test_train_tokenizer(sample_dataset, tmp_path):
    tokenizer = HFBPETokenizer()
    save_path = tmp_path / "trained_tokenizer.json"
    tokenizer.train(
        vocab_size=50,
        save_to_path=str(save_path),
        dataset=sample_dataset
    )
    assert tokenizer.is_trained is True
    assert save_path.exists()


def test_encode_before_training(tokenizer):
    with pytest.raises(
        ValueError,
        match="You must train the tokenizer before encoding text"
    ):
        tokenizer.encode("Sample text")


def test_encode_after_training(sample_dataset, tmp_path):
    tokenizer = HFBPETokenizer()
    save_path = tmp_path / "trained_tokenizer.json"
    tokenizer.train(
        vocab_size=50,
        save_to_path=str(save_path),
        dataset=sample_dataset
    )
    encoded = tokenizer.encode("Hello world")
    assert isinstance(encoded, list)
    assert len(encoded) > 0


def test_decode_before_training(tokenizer):
    with pytest.raises(
        ValueError,
        match="You must train the tokenizer before encoding text"
    ):
        tokenizer.decode([1, 2, 3])


def test_decode_after_training(sample_dataset, tmp_path):
    tokenizer = HFBPETokenizer()
    save_path = tmp_path / "trained_tokenizer.json"
    tokenizer.train(
        vocab_size=50,
        save_to_path=str(save_path),
        dataset=sample_dataset
    )
    encoded = tokenizer.encode("Hello world")
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
    assert "He" in decoded


def test_dataset_iterator(sample_dataset):
    tokenizer = HFBPETokenizer()
    tokenizer.train_dataset = sample_dataset["train"]
    tokenizer.test_dataset = sample_dataset["test"]
    tokenizer.val_dataset = sample_dataset["validation"]
    iterator = tokenizer.dataset_iterator_()
    texts = list(iterator)
    assert len(texts) == 5
    assert all(isinstance(text, str) for text in texts)


def test_hello_world():
    tokenizer = HFBPETokenizer(tokenizer_path="llm/resources/bpe_tokenizer.json")
    encoded = tokenizer.encode("Hello, world!")
    print(encoded)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
    assert "He" in decoded
