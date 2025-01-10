import pytest

from typing import Any, Dict

from llm.llm.utils.commons import read_yaml


@pytest.fixture
def get_pathname() -> str:
    return "llm/configs/dataset_loader_config.yaml"


def test_yaml_loader(get_pathname: str):
    yaml_file: Dict[str, Any] = read_yaml(get_pathname)
    assert yaml_file['dataset'] == "nlp_data"
    assert yaml_file['collection'] == "crawl_dataset"
