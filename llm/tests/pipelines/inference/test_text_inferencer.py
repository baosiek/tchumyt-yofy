import pytest
import torch

from typing import List, Tuple

from llm.llm import cfg, logger
from llm.llm.pipelines.inference.text_inferencer import TextProvider
from llm.llm.architecture.gpt.gpt_model import GPTModel


@pytest.fixture()
def text_with_ids() -> Tuple[str, List[int], str]:
    text: str = "Every effort moves you"
    ids: List[int] = [6109, 3626, 6100, 345]
    generated: str = ("Every effort moves youencrypted "
                      "quartz ESA inspired passport extra")
    return (text, ids, generated)


def test_text_to_token_ids(text_with_ids: Tuple[str, List[int], str]):
    torch.manual_seed(456)
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    text_provider: TextProvider = TextProvider(cfg=cfg)

    output: torch.Tensor = text_provider.text_to_token_ids(
        text=text_with_ids[0]
        )

    expected: torch.Tensor = torch.tensor([
        text_with_ids[1]
        ]).to(device=device)

    assert expected.equal(output)


def test_token_ids_to_text(text_with_ids: Tuple[str, List[int], str]):
    torch.manual_seed(456)
    text_provider: TextProvider = TextProvider(cfg=cfg)

    output: str = text_provider.token_ids_to_text(
        token_ids=torch.tensor(text_with_ids[1])
        )

    logger.debug(f"Output: {output}")

    expected: str = text_with_ids[0]

    assert expected == output


def test_generate_text(text_with_ids: Tuple[str, List[int], str]):
    torch.manual_seed(456)
    text_provider: TextProvider = TextProvider(cfg=cfg)
    output_ids: torch.Tensor = text_provider.generate_text(
        torch.tensor([text_with_ids[1]]), max_new_tokens=6
    )

    output: str = text_provider.token_ids_to_text(output_ids)
    logger.debug(f"Generated text: {output}")
    expected: str = text_with_ids[2]

    assert expected == output


def test_produce_text():
    torch.manual_seed(456)
    device: str = ("cuda" if torch.cuda.is_available() else "cpu")
    model: GPTModel = GPTModel(cfg=cfg)
    model.to(device=device)
    text_provider: TextProvider = TextProvider(cfg=cfg)
    text_provider.set_model(model)

    start_context: str = "Every effort moves you"
    text_produced: str = text_provider.produce_text(
        start_context=start_context
        )

    logger.info(f"Produced_text: {text_produced}")

    assert text_provider.text_to_token_ids(text_produced).shape[1] == 54
