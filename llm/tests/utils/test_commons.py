import pytest

from llm.utils.commons import read_yaml
from llm.llm.exceptions.custom_exceptions import EmptyFileError


def test_read_yaml():
    config = read_yaml("llm/configs/dataset_loader_config.yaml")

    assert config['dataset'] == "nlp_data"


def test_read_yaml_raises_emptyFile_error():
    with pytest.raises(EmptyFileError):
        read_yaml("llm/configs/empty.yaml")


def test_read_yaml_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        read_yaml("llm/configs/inexistent_file.yaml")
