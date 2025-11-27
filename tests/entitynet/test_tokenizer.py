from typedparser.typedattr import attrs_from_dict

from entitynet.config.model_config import ClipModelCfg
from entitynet.models.tokenizer_factory import build_tokenizer_from_config
from open_clip.tokenizer import SimpleTokenizer


def test_openclip_tokenizer():
    model_config_dict = {
        "model_factory": "open_clip",
        "model_ident": "ViT-L-14/open_clip_vit_l14",
        "model_loss_name": "clip",
        "tokenizer_name": None,
        "hf_text_encoder_name": None,
        "context_length": 32,
        "vis_preproc": {},
    }
    model_config = attrs_from_dict(ClipModelCfg, model_config_dict)
    tokenizer = build_tokenizer_from_config(model_config)
    assert isinstance(tokenizer, SimpleTokenizer)
    tokens = tokenizer("The tokens are nice.")
    assert tokens.shape == (1, 32)
    token_list = tokens[0].tolist()
    expected_tokens = [49406, 518, 23562, 631, 1805, 269, 49407]
    assert token_list[: len(expected_tokens)] == expected_tokens
    assert token_list[len(expected_tokens) :] == [0] * (32 - len(expected_tokens))
