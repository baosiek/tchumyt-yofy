import yaml
import logging
import logging.config

'''
This module configures the log for the entire package
'''

# loads and configure logging
with open('logging-config.json', 'r') as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)

# gets this module logger
logger = logging.getLogger("crawl_model")
