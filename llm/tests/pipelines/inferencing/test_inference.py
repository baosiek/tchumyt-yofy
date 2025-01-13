import pytest
import tiktoken
import torch

from llm.llm.architecture.gpt_model import GPTModel
from llm.llm import logger, cfg
from llm.llm.pipelines.inferencing.inference import generate_text


@pytest.fixture()
def text() -> str:
    return "Hello, I am"


def test_generate_text(text: str):
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    model: GPTModel = GPTModel(cfg)

    output = generate_text(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=cfg['sequence_length'])
    logger.info(f"Output: {output}")
    logger.info(f"Output length: {len(output[0])}")


