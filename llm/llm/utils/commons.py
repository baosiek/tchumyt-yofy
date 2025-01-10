import yaml
import os
import stat

from typing import Dict, Any
from llm import logger
from llm.llm.exceptions.custom_exceptions import EmptyFileError


def read_yaml(path_to_yaml: str) -> Dict[str, Any]:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        Dict[str, Any]: a python dictionary with yaml content 
    """

    try:
        with open(path_to_yaml) as yaml_file:

            # Checks if file is empty
            if os.stat(path_to_yaml)[stat.ST_SIZE] == 0:
                raise EmptyFileError(f"Yaml file {path_to_yaml} is empty")

            content: Dict[str, str] = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content

    except FileNotFoundError as error:
        logger.error(f"File {path_to_yaml} not found")
        raise error