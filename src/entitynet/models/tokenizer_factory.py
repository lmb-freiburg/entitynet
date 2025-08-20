from loguru import logger

from entitynet.config.model_config import ClipModelCfg
from entitynet.models.clip_misc_utils import HF_HUB_PREFIX, process_clip_model_name
from open_clip.factory import get_tokenizer
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer


def build_tokenizer_from_config(model_config: ClipModelCfg):
    model_name, _ = process_clip_model_name(model_config.model_ident)
    logger.info(
        f"{model_config.tokenizer_name=} {model_config.hf_text_encoder_name=} {model_name=}"
    )
    if model_config.tokenizer_name is not None:
        tokenizer_name = model_config.tokenizer_name
    elif model_config.hf_text_encoder_name is not None:
        tokenizer_name = f"{HF_HUB_PREFIX}{model_config.hf_text_encoder_name}"
    else:
        tokenizer_name = model_name
    logger.info(f"Load tokenizer: {tokenizer_name=}")
    tokenizer: SimpleTokenizer | HFTokenizer = get_tokenizer(
        tokenizer_name, context_length=model_config.context_length
    )
    return tokenizer
