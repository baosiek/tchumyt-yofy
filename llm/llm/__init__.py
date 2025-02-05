# import json
import logging
import logging.config
import os
import json
import yaml

from typing import Dict, Any, List

'''
The rule is:
Test configuration files must be prefixed with the token "test".
If the file does not contain this prefix, it is assumed to be
a production file.
'''


def prod_or_test_configuration(files: List[str]) -> str:

    # The configuration file to be loaded
    file_path: str = None

    if len(files) > 2:
        raise ValueError("There can only be one (production config), "
                         "or two files (production and test "
                         "configuration files). "
                         f"List contains [{len(files)}] files")

    # File cannot start with "test"
    elif len(files) == 1:
        if not files[0].startswith("test"):
            if os.path.exists(files[0]):
                return files[0]
            else:
                raise FileNotFoundError(f"File {files[0]} does not exist.")
        else:
            raise ValueError(
                "There is only one file. Cannot start with token [test]"
                f"File name provided is: [{files[0]}]."
                )

    # Files contains 2 files
    file_path: str = None
    for file in files:
        if os.path.split(file)[1].startswith("test"):
            if os.path.exists(file):
                return file
            else:
                raise FileNotFoundError(f"File {file} does not exist.")
        else:
            file_path = file

    if os.path.exists(file_path):
        return file_path
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")


'''
This module configures the log for the entire package
'''

with open('llm/configs/logging-config.json', 'r') as f:
    config = json.load(f)
    logging.config.dictConfig(config)
    logger = logging.getLogger("gpt_model")

# loads and model configuration
test_config: str = "llm/configs/gpt_config_124m_test.yaml"
prod_config: str = "llm/configs/gpt_config_124m.yaml"
file_path: str = ""

"""
Priority for test configuration.
In case it is found, loads it. Otherwise tries to
load the production one. If not found either, then
raises file not found exception.
"""
if os.path.exists(test_config):
    file_path = test_config
elif os.path.exists(prod_config):
    file_path = prod_config
else:
    raise FileExistsError(
        f'''
        Could not find neither {test_config} nor {prod_config}
        '''.strip()
    )

logger.info(f"Configuration file: {file_path}")

# Initialize the model configuration
cfg: Dict[str, Any] = None

# GPTModel configuration
try:
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)
except FileNotFoundError as error:
    logger.error(f"File [{file_path}] not found")
    raise error

# Trainer configuration files
trainer_config_files: List[str] = [
    "llm/configs/test_trainer_cfg.yaml",
    "llm/configs/trainer_cfg.yaml"
    ]

# Initialize the model configuration
trainer_cfg: Dict[str, Any] = None
trainer_config_file: str = prod_or_test_configuration(trainer_config_files)
with open(trainer_config_file, 'r') as file:
    trainer_cfg = yaml.safe_load(file)
    logger.info(f"Trainer configuration file: [{trainer_config_file}]")
