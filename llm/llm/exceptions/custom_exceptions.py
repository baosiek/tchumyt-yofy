from llm.llm import logger


class TchumytException(Exception):
    def __init__(self, error_message: str = "TchumytError was raised."):
        self.error_message = error_message
        super().__init__(self.error_message)
        print(self.error_message)


class EmptyFileError(Exception):
    def __init__(self, error_message: str):
        super().__init__(error_message)
        logger.error(error_message)


def run_tchumyt_exception(mode: str):
    message: str = f"{mode} is a message"

    if not message[0].isupper():
        raise TchumytException()
