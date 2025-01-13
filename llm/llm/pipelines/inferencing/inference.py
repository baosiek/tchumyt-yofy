import torch

from llm.llm.architecture.gpt_model import GPTModel


def generate_text(model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int):

    for _ in range(max_new_tokens):
        idx_cond: torch.Tensor = idx[:, -context_size:]
        with torch.no_grad():
            logits: torch.Tensor = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # greedy decoding
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
