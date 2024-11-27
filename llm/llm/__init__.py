import json
import logging
import logging.config
import os
import yaml

'''
This module configures the log for the entire package
'''

# loads and configure logging
with open('llm/configs/logging-config.json', 'r') as f:
    config = json.load(f)
logging.config.dictConfig(config)

# gets this module logger
logger = logging.getLogger("crawl_model")

# loads and model configuration
test_config: str = "llm/configs/gpt_config_124m_test.yaml"
prod_config: str = "llm/configs/gpt_config_124m.yaml"
file_path: str = ""

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
with open(file_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
