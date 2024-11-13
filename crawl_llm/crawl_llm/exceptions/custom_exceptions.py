import logging

logger = logging.getLogger("Testing")


class TchumytException(Exception):
    def __init__(self, error_message: str = "TchumytError was raised"):
        self.error_message = error_message
        super().__init__(self.error_message)
        print(self.error_message)


def run_exception_tchumyt_exception(mode: str):
    message: str = f"{mode} is a message"

    if not message[0].isupper():
        raise TchumytException()
