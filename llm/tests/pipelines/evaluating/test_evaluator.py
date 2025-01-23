import pytest
import torch

from typing import Tuple, List

from llm.llm import logger, cfg
from llm.llm.pipelines.evaluating.evaluator import Evaluator
from llm.llm.architecture.gpt_model import GPTModel
from llm.llm.pipelines.inferencing.text_inferencer import TextProvider


torch.manual_seed(123)

@pytest.fixture
def model() -> GPTModel:
    model: GPTModel = GPTModel(cfg=cfg)
    return model


@pytest.fixture()
def in_an_out_cpu() -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:

    input_sentences: List[str] = ["every effort moves",
                                  "I really like"]
    target_sentences: List[str] = [" effort moves you",
                                   " really like chocolate"]

    cross_entropy: float = 10.296357

    device: str = "cpu"

    text_provider: TextProvider = TextProvider(cfg=cfg)

    inputs: torch.Tensor = torch.tensor([], dtype=int)
    targets: torch.Tensor = torch.tensor([], dtype=int)

    for sent in input_sentences:
        tk_ids = text_provider.text_to_token_ids(sent)
        inputs = torch.cat((inputs, tk_ids))

    for sent in target_sentences:
        tk_ids = text_provider.text_to_token_ids(sent)
        targets = torch.cat((targets, tk_ids))

    cross_entropy_tensor: torch.Tensor = torch.tensor(cross_entropy)

    return (inputs, targets, cross_entropy_tensor, device)


@pytest.fixture()
def in_an_out_gpu(in_an_out_cpu) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:

    cross_entropy: float = 10.384483

    device: str = "cuda"

    cross_entropy_tensor: torch.Tensor = torch.tensor(cross_entropy)

    return (in_an_out_cpu[0], in_an_out_cpu[1], cross_entropy_tensor, device)


def test_calculate_batch_loss_with_cpu(
      in_an_out_cpu: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
      model: GPTModel
      ) -> None:

    logger.debug(f"Input: {in_an_out_cpu[0]}")
    logger.debug(f"Target: {in_an_out_cpu[1]}")

    evaluator: Evaluator = Evaluator(model, device=in_an_out_cpu[3])

    with torch.no_grad():
        batch_loss = evaluator.calculate_batch_loss(in_an_out_cpu[0],
                                                    in_an_out_cpu[1])

    assert batch_loss.to(device="cpu").numpy() == \
        in_an_out_cpu[2].to(device="cpu").numpy()


def test_calculate_batch_loss_with_gpu(
      in_an_out_gpu: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]
      ) -> None:

    logger.debug(f"Input: {in_an_out_gpu[0]}")
    logger.debug(f"Target: {in_an_out_gpu[1]}")

    model = GPTModel(cfg=cfg)

    evaluator: Evaluator = Evaluator(model, device=in_an_out_gpu[3])

    with torch.no_grad():
        batch_loss = evaluator.calculate_batch_loss(in_an_out_gpu[0],
                                                    in_an_out_gpu[1])

    assert batch_loss.to(device="cpu").numpy() == \
        in_an_out_gpu[2].to(device="cpu").numpy()


def test_calculate_epoch_loss(model: GPTModel):
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
