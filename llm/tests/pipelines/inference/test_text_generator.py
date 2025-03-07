import pytest
import torch

from typing import List, Tuple

from llm.llm.pipelines.inference.text_generator import TextGenerator
from llm.llm.architecture.gpt.gpt_model import GPTModel
from llm.llm.components.decoding_strategies import GreedyDecoding, \
    TemperatureScaling, AbstractDecodeStrategy
from llm.llm import cfg, logger


@pytest.fixture()
def text_with_ids() -> Tuple[str, List[int], str]:
    text: str = "Every effort moves you"
    ids: List[int] = [6109, 3626, 6100, 345]
    generated: str = ("Every effort moves you"
                      "51ancouver considered donations Snyder1981")
    return (text, ids, generated)


@pytest.fixture
def model() -> GPTModel:
    torch.manual_seed(456)
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    model: GPTModel = GPTModel(cfg=cfg)
    model.to(device=device)
    return model


def test_text_generated_initialized_correctly():
    model: GPTModel = GPTModel(cfg=cfg)

    decode_strategy: AbstractDecodeStrategy = GreedyDecoding()
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=decode_strategy
        )

    # Vocabulary size of gpt-2 is 50257
    vocab_size = text_generator.tokenizer.n_vocab
    assert vocab_size == 50257


def test_text_to_token_ids(
        text_with_ids: Tuple[str, List[int], str],
        model: GPTModel
        ):

    decode_strategy: AbstractDecodeStrategy = GreedyDecoding()
    text_generator: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=decode_strategy
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

    decode_strategy: AbstractDecodeStrategy = GreedyDecoding()
    text_provider: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=decode_strategy
        )

    output: str = text_provider.token_ids_to_text(
        token_ids=torch.tensor(text_with_ids[1])
        )

    expected: str = text_with_ids[0]

    assert expected == output


def test_to_text_greedy_decoding(
        text_with_ids: Tuple[str, List[int], str],
        model: GPTModel
        ):

    device: str = ("cuda" if torch.cuda.is_available() else "cpu")

    greedy_decoding: GreedyDecoding = GreedyDecoding()
    text_provider: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=greedy_decoding
    )

    greedy_decoding: GreedyDecoding = GreedyDecoding()
    output_ids: torch.Tensor = text_provider.to_text(
        torch.tensor([text_with_ids[1]]).to(device=device),
        max_new_tokens=6
    )

    assert output_ids.shape[1] == 10


def test_to_text_temperature_scaling(
        text_with_ids: Tuple[str, List[int], str],
        model: GPTModel
        ):
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")

    decode_strategy: AbstractDecodeStrategy = TemperatureScaling(
        temperature=1.0
        )
    text_provider: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=decode_strategy
    )

    output_ids: torch.Tensor = text_provider.to_text(
        torch.tensor([text_with_ids[1]]).to(device=device),
        max_new_tokens=6
    )

    assert output_ids.shape[1] == 10


def test_produce_text_greedy(model: GPTModel):

    greedy_decoding: GreedyDecoding = GreedyDecoding()
    text_provider: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=greedy_decoding
    )

    greedy_decoding: GreedyDecoding = GreedyDecoding()
    start_context: str = "Every effort moves you"
    text_produced: str = text_provider.generate_text(
        start_context=start_context
        )

    assert text_provider.text_to_token_ids(text_produced).shape[1] == 54


def test_produce_text_temperature_scaling(model: GPTModel):

    temperature_scaling: TemperatureScaling = TemperatureScaling(temperature=1)

    text_provider: TextGenerator = TextGenerator(
        model=model,
        context_length=1024,
        encoding="gpt2",
        decode_strategy=temperature_scaling
    )

    start_context: str = "Every effort moves you"
    text_produced: str = text_provider.generate_text(
        start_context=start_context
        )

    logger.info(text_provider.text_to_token_ids(text_produced).shape)

    assert text_provider.text_to_token_ids(text_produced).shape[1] == 55
