from loguru import logger

from entitynet.config.main_config import Config
from entitynet.config.model_config import ModelFactoryC
from entitynet.models.lit_open_clip import LitOpenClip


def build_model_from_config(config: Config):
    model_config = config.model
    model_factory = model_config.model_factory
    if model_factory == ModelFactoryC.OPEN_CLIP:
        logger.info(f"Loading open_clip model {config.model.model_ident}")
        model = LitOpenClip(config)
    else:
        raise ValueError(f"Unknown model factory: {model_factory}")
    model.config = config
    return model
