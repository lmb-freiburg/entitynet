import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import open_clip
from entitynet.config.config_factory import load_config_from_file
from entitynet.config.model_config import ClipModelCfg
from entitynet.loss_ext import get_init_logits_for_loss_name
from entitynet.models.clip_misc_utils import HF_HUB_PREFIX, process_clip_model_name
from entitynet.models.tokenizer_factory import build_tokenizer_from_config
from entitynet.paths import get_entitynet_repo_root


@pytest.mark.cuda
@pytest.mark.parametrize(
    "config_name",
    [
        "eval_clip_zs/clip_vitb_laion.yaml",
        "eval_clip_zs/clip_vite_laion.yaml",
        "eval_clip_zs/clip_vitb_openai.yaml",
        "eval_clip_zs/bioclip.yaml",
    ],
)
def test_load_config_and_run_inference(config_name: str):
    """Test loading a config file, creating model, and running inference.

    Args:
        config_name: Name of the YAML config file in configs/projects/eval_clip_zs/
    """
    # Load model from config
    print(f"Loading model from config: {config_name}")
    model, preprocess, tokenizer = load_model_from_config(config_name)

    # Verify model components exist
    assert model is not None
    assert preprocess is not None
    assert tokenizer is not None

    # Run inference on test images
    sims = run_clip_inference(model, preprocess, tokenizer)
    if config_name in _GT_SIMS:
        assert np.allclose(sims, _GT_SIMS[config_name], atol=1e-4, rtol=1e-4)
    else:
        raise RuntimeError(f"\n\n{repr(sims)}\n\nNo ground truth for {config_name}")


def load_model_from_config(config_name: str):
    """Load a CLIP model from a config file.

    Args:
        config_name: Name of the YAML config file in configs/projects/

    Returns:
        tuple: (model, preprocess_fn, tokenizer)
    """
    config_path = Path("configs/projects") / config_name
    config = load_config_from_file(config_path)
    model_cfg: ClipModelCfg = config.model
    model_name, pretrained = process_clip_model_name(model_cfg.model_ident)
    init_logit_scale, init_logit_bias = get_init_logits_for_loss_name(model_cfg.model_loss_name)

    # Create the model using open_clip.create_model - matching lit_open_clip.py exactly
    update_text_cfg_dict = dict(context_length=model_cfg.context_length)
    hf_load_text_separately = False
    if model_cfg.hf_text_encoder_name is not None:
        # this can be used to frankenmerge two different towers together
        model_cfg.force_custom_text = True
        model_cfg.hf_text_encoder_name = model_cfg.hf_text_encoder_name.removeprefix(HF_HUB_PREFIX)
        update_text_cfg_dict["hf_model_name"] = model_cfg.hf_text_encoder_name
        hf_load_text_separately = True

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained,
        force_patch_dropout=model_cfg.force_patch_dropout,
        force_custom_text=model_cfg.force_custom_text,
        update_text_cfg_dict=update_text_cfg_dict,
        hf_load_text_separately=hf_load_text_separately,
        resize_text_pos_emb=model_cfg.resize_text_pos_emb,
        init_logit_scale=init_logit_scale,
        init_logit_bias=init_logit_bias,
        weights_only=model_cfg.weights_only,
        model_loss_name=model_cfg.model_loss_name,
    )
    tokenizer = build_tokenizer_from_config(model_cfg)
    return model, preprocess_val, tokenizer


def run_clip_inference(model, preprocess, tokenizer):
    """Run CLIP model on real test images.

    Args:
        model: CLIP model
        preprocess: Image preprocessing function
        tokenizer: Text tokenizer

    Returns:
        dict: Results of inference including similarities
    """
    image_dir = get_entitynet_repo_root() / "assets/images_cc0"
    image_filenames = ["rabbit.png", "running-horses.png", "running-rabbit.png"]
    images = []
    for filename in image_filenames:
        image_path = image_dir / filename
        image = Image.open(image_path).convert("RGB")
        images.append(image)
    image_inputs = torch.stack([preprocess(img) for img in images])
    test_texts = [
        "a rabbit sitting",
        "horses running",
        "a running rabbit",
        "animals in motion",
        "a photo of a dog",
        "a red square",
    ]
    text_inputs = tokenizer(test_texts)
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = image_features @ text_features.T
    return similarities.tolist()


_GT_SIMS = {
    "eval_clip_zs/clip_vitb_laion.yaml": [
        [
            0.3412649631500244,
            0.05294955521821976,
            0.31539392471313477,
            0.21113726496696472,
            0.2103252112865448,
            0.12124420702457428,
        ],
        [
            0.07053384184837341,
            0.2808022201061249,
            0.17362181842327118,
            0.27155405282974243,
            0.19467978179454803,
            0.12425138801336288,
        ],
        [
            0.299697607755661,
            0.06850817799568176,
            0.2927199900150299,
            0.1834738701581955,
            0.1961386501789093,
            0.1166120171546936,
        ],
    ],
    "eval_clip_zs/clip_vite_laion.yaml": [
        [
            0.5178165435791016,
            0.2791401147842407,
            0.48495712876319885,
            0.37400439381599426,
            0.3740038275718689,
            0.3293689489364624,
        ],
        [
            0.21608011424541473,
            0.48964163661003113,
            0.3129968047142029,
            0.42343950271606445,
            0.3175017237663269,
            0.34724849462509155,
        ],
        [
            0.41378334164619446,
            0.3622572124004364,
            0.5426278114318848,
            0.44202861189842224,
            0.3759613633155823,
            0.3279803395271301,
        ],
    ],
    "eval_clip_zs/clip_vitb_openai.yaml": [
        [
            0.27363741397857666,
            0.1582864373922348,
            0.2725609242916107,
            0.2413870394229889,
            0.19122067093849182,
            0.18303713202476501,
        ],
        [
            0.16386888921260834,
            0.2757876515388489,
            0.1915651261806488,
            0.2570391893386841,
            0.2113552689552307,
            0.20101404190063477,
        ],
        [
            0.28614163398742676,
            0.15145467221736908,
            0.2785494029521942,
            0.23576870560646057,
            0.19982708990573883,
            0.17729195952415466,
        ],
    ],
    "eval_clip_zs/bioclip.yaml": [
        [
            0.15536615252494812,
            0.14839503169059753,
            0.2704857885837555,
            0.025957435369491577,
            0.19077762961387634,
            0.17932459712028503,
        ],
        [
            0.025192175060510635,
            0.22395792603492737,
            0.06585542857646942,
            0.12266193330287933,
            0.23873421549797058,
            0.15957532823085785,
        ],
        [
            0.19818072021007538,
            0.2036179006099701,
            0.17007245123386383,
            0.07784426957368851,
            0.22788578271865845,
            0.20332179963588715,
        ],
    ],
}
