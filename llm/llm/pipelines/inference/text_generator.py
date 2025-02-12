import tiktoken
from tiktoken.core import Encoding
import torch

from typing import List

from llm.llm import logger
from llm.llm.architecture.gpt_model import GPTModel
from llm.llm.components.decoding_strategies import AbstractDecodeStrategy

"""
This Trainer class is responsible for the whole training pipeline

Args:
    model: GPTModel -> The initialized model to be trained
    context_length: int -> The context_length that trims or
        pad incoming text length
    encoding: str -> The encoding to initialize the TikToken tokenizer
    decode_strategy: AbstractDecodeStrategy -> The decode strategy like,
        greedy decoding or top K sampling

Returns:
    None
"""


# this function returns the max argument from a probability distribution
def greedy_decoding(logits: torch.Tensor) -> torch.Tensor:
    logger.info("Using greedy decoding")
    prob: torch.Tensor = torch.softmax(logits, dim=-1)
    return torch.argmax(prob, dim=-1, keepdim=True)


class TextGenerator():
    def __init__(
            self,
            model: GPTModel,
            context_length: int,
            encoding: str,
            decode_strategy: AbstractDecodeStrategy
            ):

        # The model to generate text
        self.model: GPTModel = model

        # The tokenizer
        self.tokenizer: Encoding = tiktoken.get_encoding(encoding)

        # The context length
        self.context_length: int = context_length

        # The decoder strategy
        self.decode_strategy: AbstractDecodeStrategy = decode_strategy

        logger.info("Text generator initialized with:")
        logger.info(f"\tTokenizer encoding: {encoding}")
        logger.info(f"\tDecode strategy: {str(decode_strategy.__name__)}")

    def text_to_token_ids(self, text: str) -> torch.Tensor:
        '''
        This method tokenizes the input text, returning a list of integers,
            ie, the token ids.

        Args:
            text: str -> The text to be encoded

        Returns:
            encoded: torch.Tensor -> The list of integers with token ids
        '''

        encoded: List[int] = self.tokenizer.encode(
            text=text,
            allowed_special={'<|endoftext|>'}
            )

        encoded_tensor: torch.Tensor = torch.tensor(encoded).unsqueeze(0)

        logger.debug(f"Input text: {text}")
        logger.debug(f"Token ids: {encoded_tensor}")

        return encoded_tensor

    def token_ids_to_text(self, token_ids: torch.Tensor) -> str:
        '''
        This method converts token_ids to tokens.

        Args:
            token_ids: torch.Tensor -> The tensor with the list of token_ids

        Returns:
            decoded: str -> The string with tokens
        '''
        flat: torch.Tensor = token_ids.squeeze(0)

        decoded: str = self.tokenizer.decode(flat.tolist())

        logger.debug(f"Input token ids: {token_ids}")
        logger.debug(f"Decoded text: {decoded}")

        return decoded

    def to_text(
            self,
            input: torch.Tensor,
            max_new_tokens: int,
            ) -> torch.Tensor:
        '''
        This method generates max_new_tokens drawn from the model
        Args:
            input: torch.Tensor -> a tensor with the input token_ids
            max_new_tokens: int -> the number of tokens to generate

        Returns:
            input: torch.Tensor -> the same input tensor with the added
                new tokens
        '''

        for _ in range(max_new_tokens):
            input_trimmed: torch.Tensor = input[:, -self.context_length:]
            with torch.no_grad():
                logits: torch.Tensor = self.model(input_trimmed)

            logits = logits[:, -1, :]
            logits: torch.Tensor = torch.softmax(logits, dim=-1)

            next_token: torch.Tensor = self.decode_strategy.decode(logits)
            input = torch.cat((input, next_token), dim=1)

        logger.debug(f"Text generated: {input}")

        return input

    def generate_text(
            self,
            start_context: str
    ) -> str:

        # get the device it should run
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.eval()

        encoded: torch.Tensor = self.text_to_token_ids(start_context).to(
            device=device
        )

        with torch.no_grad():
            token_ids = self.to_text(
                encoded,
                max_new_tokens=50,
                # decode_strategy=self.decode_strategy
            )

        decoded_text = self.token_ids_to_text(token_ids=token_ids)

        self.model.train()

        logger.debug(f"New text: {decoded_text}")

        return decoded_text
