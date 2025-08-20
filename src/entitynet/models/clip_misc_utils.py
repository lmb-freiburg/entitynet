from entitynet.paths import get_entitynet_repo_root

HF_HUB_PREFIX = "hf-hub:"


def list_all_clip_architectures() -> list[str]:
    import open_clip

    model_names = []
    for model_name, pretrained in open_clip.list_pretrained():
        model_names.append(model_name)
    # we also need to add the model names from the config folder in case we added some ourselves
    model_config_dir = get_entitynet_repo_root() / "src/open_clip/model_configs"
    model_configs = list(model_config_dir.glob("*.json"))
    for model_config in model_configs:
        model_name = model_config.stem
        model_names.append(model_name)
    model_names = sorted(set(model_names))
    return model_names


def process_clip_model_name(
    model_name_input: str,
):  # hugging face example: "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    if model_name_input.startswith(HF_HUB_PREFIX):  # HF_HUB_PREFIX: "hf-hub:"
        model_name = model_name_input
        pretrained = None
    else:
        model_name, pretrained = model_name_input.split("/", 1)
        if pretrained == "none":
            pretrained = False
    return model_name, pretrained


def make_hf_model_name_safe(model_name_input: str):
    return model_name_input.replace("/", "__").replace(":", "__")
