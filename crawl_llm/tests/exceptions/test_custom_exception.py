import pytest

from crawl_llm.crawl_llm.exceptions.custom_exceptions  \
    import TchumytException, run_tchumyt_exception


def test_tchumyt_exception():

    with pytest.raises(TchumytException):
        run_tchumyt_exception("test")
