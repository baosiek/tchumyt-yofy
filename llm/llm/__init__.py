# import json
import logging
import logging.config
import os
import json
import yaml

from typing import Dict, Any


def initialize_logger(log_dir: str) -> logging.Logger:
    '''
    This function loads logging configuration file. These files
    can be one of either logging-config.json, for production, or
    logging-config-test.json, for test. If logging-config-test.json
    exists it is loaded first

    Args:
        log_dir: str -> the directory where to find log configuration file

    Return:
        logger: logging.Logger -> the logger for the application
    '''

    # production logger configuration
    prod_path: os.path = os.path.join(log_dir, "logging-config.json")

    # test logger configuration
    test_path: os.path = os.path.join(log_dir, "logging-config-test.json")

    # the chosen file to use
    log_path: os.path = prod_path

    # initialize logger
    if os.path.exists(test_path):
        log_path = test_path

    try:
        with open(log_path, 'r') as file:
            config = json.load(file)
            logging.config.dictConfig(config)
            logger = logging.getLogger("gpt_model")
            logger.info(f"logger configured with file: {log_path}")
    except FileNotFoundError as error:
        raise error(f"{log_path} was not found")

    return logger


def loads_model_configuration(config_dir: str) -> Dict[str, Any]:
    '''
    This function loads the model configuration file. These files
    can be one of either gpt_config.yaml, for production, or
    gpt_config_test.yaml, for test. If gpt_config_test.yaml
    exists it is loaded first

    Args:
        config_dir: str -> the directory where to find configuration file

    Return:
        trainer_cfg: Dict[str, Any] -> the dictionary with the configuration
    '''

    # production configuration
    prod_path: os.path = os.path.join(config_dir, "gpt-config.yaml")

    # test configuration
    test_path: os.path = os.path.join(config_dir, "gpt-config-test.yaml")

    # the chosen file to use
    config_path: os.path = prod_path

    # initialize logger
    if os.path.exists(test_path):
        config_path = test_path

    # trainer configuration
    try:
        with open(config_path, 'r') as file:
            model_cfg = yaml.safe_load(file)
    except FileNotFoundError as error:
        logger.error(f"File [{config_path}] not found")
        raise error

    return model_cfg


def loads_trainer_configuration(config_dir: str) -> Dict[str, Any]:
    '''
    This function loads the trainer configuration file. These files
    can be one of either trainer-cfg.yaml, for production, or
    trainer-cfg-test.yaml, for test. If trainer-cfg-test.yaml
    exists it is loaded first

    Args:
        config_dir: str -> the directory where to find configuration file

    Return:
        trainer_cfg: Dict[str, Any] -> the dictionary with the configuration
    '''

    # production configuration
    prod_path: os.path = os.path.join(config_dir, "trainer-cfg.yaml")

    # test configuration
    test_path: os.path = os.path.join(config_dir, "trainer-cfg-test.yaml")

    # the chosen file to use
    config_path: os.path = prod_path

    # initialize logger
    if os.path.exists(test_path):
        config_path = test_path

    # trainer configuration
    try:
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
    except FileNotFoundError as error:
        logger.error(f"File [{config_path}] not found")
        raise error

    return cfg


# Initializes the logger
logger: logging.Logger = initialize_logger("llm/configs/logger")

# Loads trainer configuration
trainer_cfg: Dict[str, Any] = loads_trainer_configuration(
    "llm/configs/trainer"
)

# Loads model configuration
model_cfg: Dict[str, Any] = loads_model_configuration(
    "llm/configs/model"
)
