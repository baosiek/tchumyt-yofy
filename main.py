from llm.llm import logger, cfg
from llm.llm.pipelines.training.train import train

if __name__ == "__main__":

    logger.info(cfg.get('name'))
    train(cfg=cfg)
