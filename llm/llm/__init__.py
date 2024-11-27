import json
import logging
import logging.config
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
with open("llm/configs/gpt_config_124m.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
