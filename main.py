from llm.llm import logger, cfg

if __name__ == "__main__":
    logger.debug("Hello log!")
    logger.info(cfg['name'])
    logger.info(cfg.get('name'))
    logger.info(type(cfg))
