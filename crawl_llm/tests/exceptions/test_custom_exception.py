import pytest

from crawl_llm.crawl_llm.exceptions.custom_exceptions  \
    import TchumytException


def runs_tchumyt_exception(mode: str):
    message: str = f"{mode} is a message"

    if not message[0].isupper():
        raise TchumytException()


def test_raise_tchumyt_exception():

    with pytest.raises(TchumytException):
        runs_tchumyt_exception("test")
