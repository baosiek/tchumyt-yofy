import pytest
import torch

from typing import List, Tuple

from llm.llm.pipelines.inference.text_generator import TextGenerator
from llm.llm.architecture.gpt_model import GPTModel
from llm.llm import cfg


@pytest.fixture()
def text_with_ids() -> Tuple[str, List[int], str]:
    text: str = "Every effort moves you"
    ids: List[int] = [6109, 3626, 6100, 345]
    generated: str = ("Every effort moves you"
                      "encrypted quartz ESA inspired passport extra")
    return (text, ids, generated)


@pytest.fixture
def model() -> GPTModel:
    torch.manual_seed(456)
    model: GPTModel = GPTModel(cfg=cfg)
    return model


def test_text_generated_initialized_correctly():
    model: GPTModel = GPTModel(cfg=cfg)
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2"
        )

    # Vocabulary size of gpt-2 is 50257
    vocab_size = text_generator.tokenizer.n_vocab
    assert vocab_size == 50257


def test_text_to_token_ids(
        text_with_ids: Tuple[str, List[int], str],
        model: GPTModel
        ):

    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2"
    )

    output: torch.Tensor = text_generator.text_to_token_ids(
        text=text_with_ids[0]
        )

    expected: torch.Tensor = torch.tensor([
        text_with_ids[1]
        ])

    assert expected.equal(output)


def test_token_ids_to_text(
        text_with_ids: Tuple[str, List[int], str],
        model: GPTModel):

    text_provider: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2")

    output: str = text_provider.token_ids_to_text(
        token_ids=torch.tensor(text_with_ids[1])
        )

    expected: str = text_with_ids[0]

    assert expected == output


def test_to_text(
        text_with_ids: Tuple[str, List[int], str],
        model: GPTModel
        ):

    text_provider: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2"
    )
    output_ids: torch.Tensor = text_provider.to_text(
        torch.tensor([text_with_ids[1]]), max_new_tokens=6
    )

    output: str = text_provider.token_ids_to_text(output_ids)
    expected: str = text_with_ids[2]

    assert expected == output


def test_produce_text(model: GPTModel):

    text_provider: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2"
    )

    start_context: str = "Every effort moves you"
    text_produced: str = text_provider.generate_text(
        start_context=start_context
        )

    assert text_provider.text_to_token_ids(text_produced).shape[1] == 54
