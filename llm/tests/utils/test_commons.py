import pytest

from llm.llm.utils.commons import read_yaml, read_json
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


def test_read_json():
    config = read_json("llm/configs/test_json_reader.json")

    assert config['name'] == "Fulano"


def test_read_json_raises_emptyFile_error():
    with pytest.raises(EmptyFileError):
        read_json("llm/configs/empty.json")


def test_read_json_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        read_json("llm/configs/inexistent_file.json")
