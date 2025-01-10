import pytest

from typing import Any, Dict

from llm.utils import yaml_loader


@pytest.fixture
def get_pathname() -> str:
    return "llm/configs/dataset_config.yaml"


def test_yaml_loader(get_pathname: str):
    yaml_file: Dict[str, Any] = yaml_loader.get_yaml_file(get_pathname)
    assert yaml_file['database'] == "nlp_data"
    assert yaml_file['collection'] == "crawl_dataset"
