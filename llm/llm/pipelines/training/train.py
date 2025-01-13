import torch

from typing import Dict, Any

from llm.llm import logger
from llm.llm.architecture.gpt_model import GPTModel




def train(cfg: Dict[str, Any]):

    torch.manual_seed(456)
    model: GPTModel = GPTModel(cfg=cfg)
    model.eval()


if __name__ == "__main__":
    logger.info("Training started")
    train()
