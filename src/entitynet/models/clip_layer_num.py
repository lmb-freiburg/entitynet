"""
Map parameter_name to layer_number for CLIP models, in order to apply layerwise LR decay.
"""


def get_layer_num_and_tower_name_for_clip(
    param_name: str, num_vision_layers: int, num_text_layers: int, last_layer_same: bool = True
) -> tuple[int, str]:
    """
    Layer number will determine the LR factor in layer decay LR.
    In case the two towers have different number of layers,
    we can either fix the last layers of the two towers to be the same, or the first.
    """
    bigger_tower_size = max(num_vision_layers, num_text_layers)

    def _update_layer_num(tower_name_for_layer, layer_num_here):
        if num_vision_layers == num_text_layers:
            return layer_num_here, tower_name_for_layer
        if tower_name_for_layer == "text":
            this_tower_size, other_tower_size = num_text_layers, num_vision_layers
        elif tower_name_for_layer == "visual":
            this_tower_size, other_tower_size = num_vision_layers, num_text_layers
        else:
            raise ValueError(f"Unknown tower: {tower_name_for_layer}")
        if this_tower_size > other_tower_size:
            return layer_num_here, tower_name_for_layer
        if not last_layer_same:
            return layer_num_here, tower_name_for_layer
        # this is the only case where we have to update the layer number:
        # given a parameter of the smaller tower and we want the last layer number to be the same
        # so we add the tower size difference
        # example: 24 vision layers, 12 text layers, given text layer 3, we want to return 15
        return layer_num_here + other_tower_size - this_tower_size, tower_name_for_layer

    # text tower input and output layer
    if param_name.startswith("token_embedding.") or param_name == "positional_embedding":
        return _update_layer_num("text", 0)
    if param_name.startswith("ln_final.") or param_name == "text_projection":
        return _update_layer_num("text", num_text_layers + 1)

    # text tower blocks for default clip
    if param_name.startswith("transformer.resblocks."):
        layer_id = int(param_name.split(".")[2])
        return _update_layer_num("text", layer_id + 1)

    # text tower blocks for custom text clip
    if param_name.startswith("text."):
        param_name_notext = param_name[len("text.") :]
        layer_num = get_num_layer_for_custom_text_tower(param_name_notext, num_text_layers)
        return _update_layer_num("text", layer_num)

    # output layer after both towers are done
    if param_name == "logit_scale":
        return bigger_tower_size + 1, "other"

    # vision tower is implemented as a separate model in "visual" attribute
    if param_name.startswith("visual."):
        param_name_novisual = param_name[len("visual.") :]
        layer_num = get_num_layer_for_vision_tower(param_name_novisual, num_vision_layers)
        return _update_layer_num("visual", layer_num)

    raise ValueError(f"Unknown param name: {param_name}")


def get_num_layer_for_vision_tower(param_name, num_max_layer):
    if any(
        param_name.startswith(a) for a in ("ln_pre.", "conv1.", "trunk.patch_embed.")
    ) or param_name in set(
        ("class_embedding", "positional_embedding", "trunk.cls_token", "trunk.pos_embed")
    ):
        # input layer
        return 0
    if param_name == "proj" or any(
        param_name.startswith(a) for a in ("ln_post.", "trunk.norm.", "head.proj.", "trunk.head.")
    ):
        # output layer
        return num_max_layer + 1
    if param_name.startswith("transformer.resblocks.") or param_name.startswith("trunk.blocks."):
        # blocks (the actual layers)
        layer_id = int(param_name.split(".")[2])
        return layer_id + 1
    raise ValueError(f"Unknown param name: {param_name}")


def get_num_layer_for_custom_text_tower(param_name, num_max_layer):
    # for default transformer
    if param_name.startswith("token_embedding.") or param_name == "positional_embedding":
        return 0
    if param_name.startswith("ln_final.") or param_name == "text_projection":
        return num_max_layer + 1
    if param_name.startswith("transformer.resblocks."):
        layer_id = int(param_name.split(".")[2])
        return layer_id + 1

    # for bertmodel
    if param_name.startswith("transformer.embeddings."):
        # transformers/models/bert/modeling_bert.py BertEmbeddings
        # word_embeddings, position_embeddings, token_type_embeddings, LayerNorm
        return 0
    if param_name.startswith("proj."):  # proj.0.weight and proj.2.weight
        return num_max_layer + 1
    if param_name.startswith("transformer.encoder.layer."):
        layer_id = int(param_name.split(".")[3])
        return layer_id + 1
    raise ValueError(f"Unknown param name: {param_name}")
