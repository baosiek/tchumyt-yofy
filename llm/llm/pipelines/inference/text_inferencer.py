import tiktoken
import torch

from typing import Dict, Any, List
from tiktoken.core import Encoding

from llm.llm.architecture.gpt_model import GPTModel


class TextProvider():
    def __init__(
            self,
            cfg: Dict[str, Any],
            ) -> None:
        self.tokenizer: Encoding = tiktoken.get_encoding(
            cfg["tiktoken_encoding"]
            )

        self.model: GPTModel = GPTModel(cfg=cfg)
        self.sequence_length: int = cfg['sequence_length']

    def set_model(self, model: GPTModel) -> None:
        self.model = model

    def generate_text(
            self,
            input: torch.Tensor,
            max_new_tokens: int
            ) -> torch.Tensor:

        for _ in range(max_new_tokens):
            input_trimmed: torch.Tensor = input[:, -self.sequence_length:]
            with torch.no_grad():
                logits: torch.Tensor = self.model(input_trimmed)

            logits = logits[:, -1, :]
            prob: torch.Tensor = torch.softmax(logits, dim=-1)

            # greedy decoding
            next_token: torch.Tensor = torch.argmax(prob, dim=-1, keepdim=True)

            input = torch.cat((input, next_token), dim=1)

        return input

    def text_to_token_ids(self, text: str) -> torch.Tensor:
        encoded: List[int] = self.tokenizer.encode(
            text=text,
            allowed_special={'<|endoftext|>'}
            )

        return torch.tensor(encoded).unsqueeze(0)

    def token_ids_to_text(self, token_ids: torch.Tensor) -> str:
        flat: torch.Tensor = token_ids.squeeze(0)
        return self.tokenizer.decode(flat.tolist())

    def produce_text(self, start_context: str) -> str:

        self.model.eval()
        context_size: int = self.model.positional_embedding.pe.shape[1]
        encoded = self.text_to_token_ids(start_context)
        with torch.no_grad():
            token_ids = self.generate_text(encoded, max_new_tokens=50)

        decoded_text = self.token_ids_to_text(token_ids=token_ids)

        self.model.train()

        return decoded_text
