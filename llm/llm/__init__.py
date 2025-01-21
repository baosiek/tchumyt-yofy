# import json
import logging
import logging.config
import os
import json
import yaml

from typing import Dict, Any

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

cfg: Dict[str, Any] = None

# GPTModel configuration
try:
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)
except FileNotFoundError as error:
    logger.error(f"File [{file_path}] not found")
    raise error
