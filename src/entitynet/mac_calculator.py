"""
Calculate MACs for various CLIP models
"""

from loguru import logger

from packg.iotools import load_json

from entitynet.paths import get_entitynet_repo_root

ARCH_SHORT = {
    "new-vits-16": "S-16",
    "new-vits-32": "S-32",
    "new-vitty-16": "Ti-16",
    "new-vitty-32": "Ti-32",
    "new-vitmu63-32": "Mu-32",
    "new-vitmu63-16": "Mu-16",
    "ViT-B-16": "B-16",
    "ViT-B-32": "B-32",
    "ViT-L-14": "L-14",
    "ViT-H-14": "H-14",
    "ViT-H-14-quickgelu": "H-14",
    "EVA02-E-14-plus": "E-14",
}


FUSE_MULTADD = True
IGNORE_DIV = True


class DotDict(dict):
    """Dictionary that supports dot notation access."""

    def __getattr__(self, key):
        if key in self:
            value = self[key]
            if isinstance(value, dict):  # Convert nested dicts to DotDict
                return DotDict(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def format_number(n):
    FULL_COL = 20
    COMPACT_COL = 10
    dot_format = f"{n:,}".replace(",", "_")
    dot_format = " " * (FULL_COL - len(dot_format)) + dot_format
    dot_format = "|" + dot_format

    # decimal format
    if n >= 1_000_000_000:
        compact_format = f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        compact_format = f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        compact_format = f"{n / 1_000:.1f}K"
    else:
        compact_format = str(n)
    compact_format = "  " * (COMPACT_COL - len(compact_format)) + compact_format
    compact_format = "|" + compact_format

    # binary format
    units = ["", "Ki", "Mi", "Gi", "Ti"]
    size = float(n)
    unit_index = 0
    while size >= 1024 and unit_index < len(units):
        size /= 1024
        unit_index += 1
    formatted_value = round(size, 1)
    bin_format = f"{formatted_value}{units[unit_index]}"

    return "  ".join([compact_format, dot_format])


def mac_layer_norm(seq_len, batch_size, dim):
    macs = 5 * seq_len * batch_size * dim + 2 * seq_len * batch_size
    # macs = seq_len*batch_size*dim*6
    return macs


def mac_linear(in_dim, out_dim, batch_size, bias):
    macs = batch_size * in_dim * out_dim * (1 if FUSE_MULTADD else 2)
    if bias:
        macs = macs + batch_size * out_dim
    return macs


def mac_in_proj(seq_len, batch_size, dim, bias):
    # maps from dim to 3x dim (Q, K, V)
    return mac_linear(in_dim=dim, out_dim=dim * 3, batch_size=batch_size, bias=bias) * seq_len


def mac_out_proj(seq_len, batch_size, dim, bias):
    # maps from dim to dim
    return mac_linear(in_dim=dim, out_dim=dim, batch_size=batch_size, bias=bias) * seq_len


def mac_softmax_vector(bs, dim):
    # exp
    macs = bs * dim  # exp for every item
    # sum
    macs = macs + bs * dim  # sum is dim-1 additions
    # div
    macs = macs + (bs * dim if not IGNORE_DIV else 0)
    return macs


def mac_softmax_matrix(bs, dim1, dim2):
    macs = mac_softmax_vector(bs, dim2) * dim1
    return macs


def mac_scaled_dot_product_attention(seq_len, batch_size, dim):
    # Q*K
    macs = seq_len * seq_len * batch_size * dim * (1 if FUSE_MULTADD else 2)
    # Q * 1/sqrt(d_q)
    macs = macs + seq_len * batch_size * dim
    # Softmax
    macs = macs + mac_softmax_matrix(batch_size, seq_len, seq_len)
    # Attention scores * V
    macs = macs + seq_len * batch_size * dim * (1 if FUSE_MULTADD else 2)
    return macs


def mac_mean_vector(bs, dim):
    # sum
    macs = bs * dim
    # div
    macs = macs + (bs if not IGNORE_DIV else 0)
    return macs


def mac_mean_matrix(bs, dim1, dim2):
    macs = mac_mean_vector(bs, dim2) * dim1
    return macs


def mac_gelu(seq_len, batch_size, dim):
    macs = seq_len * batch_size * dim * 12
    return macs


def mac_multi_head_attention(seq_len, batch_size, dim, bias):
    macs = mac_in_proj(seq_len, batch_size, dim, bias)  # in projection
    macs = macs + mac_scaled_dot_product_attention(seq_len, batch_size, dim)  # multi-head attention
    macs = macs + mac_out_proj(seq_len, batch_size, dim, bias)  # out projection
    macs = macs + mac_mean_matrix(batch_size, seq_len, dim)  # mean attention scores
    return macs


def mac_multi_head_attention_block(seq_len, batch_size, dim, mlp_dim, bias):
    # heads are ignored, as we would divide dim into chunks of size dim/heads
    # this is mult-add wise the same as just using dim
    macs = mac_layer_norm(seq_len, batch_size, dim)  # layer norm (only for Q, as Q=K=V)
    macs = macs + mac_multi_head_attention(seq_len, batch_size, dim, bias)  # the attention itself
    macs = macs + mac_layer_norm(seq_len, batch_size, dim)  # layer norm
    # feed forward
    macs = macs + seq_len * mac_linear(
        in_dim=dim, out_dim=mlp_dim, batch_size=batch_size, bias=bias
    )
    macs = macs + mac_gelu(seq_len, batch_size, mlp_dim)
    macs = macs + seq_len * mac_linear(
        in_dim=mlp_dim, out_dim=dim, batch_size=batch_size, bias=bias
    )
    return macs


def mac_transformer(seq_len, batch_size, dim, mlp_dim, num_layers, bias):
    macs = 0
    for _ in range(num_layers):
        tmp_macs = mac_multi_head_attention_block(seq_len, batch_size, dim, mlp_dim, bias)
        macs = macs + tmp_macs
    return macs


def mac_conv2d(batch_size, img_size, in_channels, out_channels, kernel_size, stride):
    # assuming square images, kernel and stride. and no padding needed
    patches = ((img_size - kernel_size) // stride + 1) ** 2
    macs = batch_size * patches * kernel_size * kernel_size * in_channels * out_channels
    return macs


def mac_siglip_loss_new(batch_size, dim):
    """
    changed logit bias from b*d to b*b, should not make a difference since it's still cheap.

    image_features: (b, d)
    text_features: (b, d)

    get logits: scale * img_features @ text_features.T (note: @ comes first!)
        1. img_features@text_features.T: result (b, b), computes b * b dot product of cost d each.
        2. scale: (b, b) scalar multiplications with the result
    """
    macs = batch_size * batch_size * dim + batch_size * batch_size
    macs = macs + batch_size * batch_size * 5
    return macs


def mac_siglip_loss_old(batch_size, dim):
    # get logits: scale * img_features@text_features.T
    macs = batch_size * batch_size * dim + batch_size * dim
    # -logsigmoid(labels*logits)
    macs = macs + batch_size * batch_size * 5
    return macs




def mac_CLIPLoss(batch_size, dim):
    # update logit_scale b*d to b*b
    # get logits: scale * img_features@text_features.T + scale * text_features@img_features.T
    macs = batch_size * batch_size * dim + batch_size * dim
    macs = macs + batch_size * batch_size * dim + batch_size * dim
    # cross entropy(image_logits, text_labels) + cross entropy(text_logits, image_labels)
    macs = macs + 2 * batch_size * batch_size * dim + batch_size
    macs = macs + 2 * batch_size * batch_size * dim + batch_size
    return macs


def mac_clip_final_ctx77(model_config, bs, loss_fn, print_results=False):
    # for fair comparison assume 77 context length always
    model_config["text_cfg"]["context_length"] = 77
    return mac_clip_final(model_config, bs, loss_fn, print_results)


def mac_clip_final(model_config, bs, loss_fn, print_results=False):
    # # print(" ==== config ====")
    # print(model_config)
    cfg = DotDict(model_config)
    kernel_size = stride = cfg.vision_cfg.patch_size
    img_token = ((cfg.vision_cfg.image_size - kernel_size) // stride + 1) ** 2

    IMG_CH = 3
    # ViT part
    # conv layer
    macs_conv = mac_conv2d(
        batch_size=bs,
        img_size=cfg.vision_cfg.image_size,
        in_channels=IMG_CH,
        out_channels=cfg.vision_cfg.width,
        kernel_size=kernel_size,
        stride=stride,
    )
    macs_vit = macs_conv

    # norm
    macs_lnpre = mac_layer_norm(seq_len=img_token, batch_size=bs, dim=cfg.vision_cfg.width)
    macs_vit = macs_vit + macs_lnpre

    # vision transformer
    n_cls = 1
    img_token += n_cls  # add class token
    if not hasattr(cfg.vision_cfg, "mlp_ratio"):
        # print(f"WARN: Config has no mlp_ratio, assuming 4: {model_config}")
        vision_mlp_ratio = 4
    else:
        vision_mlp_ratio = cfg.vision_cfg.mlp_ratio
    vision_mlp_dim = cfg.vision_cfg.width * vision_mlp_ratio
    macs_vt = mac_transformer(
        seq_len=img_token,
        batch_size=bs,
        dim=cfg.vision_cfg.width,
        mlp_dim=vision_mlp_dim,
        num_layers=cfg.vision_cfg.layers,
        bias=True,
    )
    macs_vit = macs_vit + macs_vt
    # norm
    macs_lnpost = mac_layer_norm(seq_len=img_token, batch_size=bs, dim=cfg.vision_cfg.width)
    macs_vit = macs_vit + macs_lnpost
    # output projection
    macs_out = (
        mac_linear(in_dim=cfg.vision_cfg.width, out_dim=cfg.embed_dim, batch_size=bs, bias=True)
        * n_cls
    )  # default is only one (the CLS) token

    # text transformer
    n_texts_per_image = 1
    if loss_fn == "msiglip":
        n_texts_per_image = cfg.text_cfg.n_texts_per_image
    bs_text = bs * n_texts_per_image
    text_mlp_ratio = cfg.text_cfg.mlp_ratio if hasattr(cfg.text_cfg, "mlp_ratio") else 4
    text_mlp_dim = cfg.text_cfg.width * text_mlp_ratio
    macs_tt = mac_transformer(
        seq_len=cfg.text_cfg.context_length,
        batch_size=bs_text,
        dim=cfg.text_cfg.width,
        mlp_dim=text_mlp_dim,
        num_layers=cfg.text_cfg.layers,
        bias=True,
    )
    macs_text = macs_tt
    macs_ln_text = mac_layer_norm(
        seq_len=cfg.text_cfg.layers, batch_size=bs_text, dim=cfg.text_cfg.width
    )
    macs_text = macs_text + macs_ln_text
    macs_text_outproj = (
        mac_linear(in_dim=cfg.text_cfg.width, out_dim=cfg.embed_dim, batch_size=bs_text, bias=True)
        * cfg.text_cfg.context_length
    )
    macs_text = macs_text + macs_text_outproj

    # loss
    if loss_fn.lower() == "siglip":
        macs_loss = mac_siglip_loss_new(batch_size=bs, dim=cfg.embed_dim)
    elif loss_fn.lower() == "clip":
        macs_loss = mac_CLIPLoss(batch_size=bs, dim=cfg.embed_dim)
    else:
        logger.error(f"Unknown loss function: {loss_fn}")
        return {
            "vision": 0,
            "text": 0,
            "loss": 0,
            "total": 0,
        }

    results = {
        "vision": macs_vit,
        "text": macs_text,
        "loss": macs_loss,
        "total": macs_vit + macs_text + macs_loss,
    }
    if not print_results:
        return results
    print(" === results ===")
    print("vision")
    print("  conv       ", format_number(macs_conv))
    print("  layer norm ", format_number(macs_lnpre))
    print("  vision T   ", format_number(macs_vt))
    print("  layer norm ", format_number(macs_lnpost))
    print("  out proj   ", format_number(macs_out))
    print("  total ViT  ", format_number(macs_vit))
    print("text")
    print("  text T     ", format_number(macs_tt))
    print("  layer norm ", format_number(macs_ln_text))
    print("  out proj   ", format_number(macs_text_outproj))
    print("  total text:", format_number(macs_text))
    print(f"loss: {loss_fn}:  ", format_number(macs_loss))
    print("total CLIP:  ", format_number(macs_vit + macs_text + macs_loss))
    print()
    return results


def main():
    for arch_conf_name, arch_name in ARCH_SHORT.items():
        print(arch_name)
        arch_file = get_entitynet_repo_root() / f"src/open_clip/model_configs/{arch_conf_name}.json"
        archconf = load_json(arch_file)

        mac_bef = None
        for context_length in 32, 77:
            archconf["text_cfg"]["context_length"] = context_length
            batch_size = 8192
            if arch_name == "E-14":
                # config is missing infos, fu it
                continue
            macs_dict = mac_clip_final(archconf, bs=batch_size, loss_fn="clip")
            macs_batch_forward = macs_dict["total"]
            macs_batch = macs_batch_forward * 3  # rule of thumb, training pass is ~3 forwards
            macs_per_image = macs_batch / batch_size
            if mac_bef is None:  # shakespeare
                mac_bef = macs_per_image
                mac_bef_str = ""
            else:
                mac_bef_rel = macs_per_image / mac_bef
                mac_bef_str = f" {mac_bef_rel-1:.0%}"
            print(f"  {context_length} {macs_per_image:.1e}{mac_bef_str}")


if __name__ == "__main__":
    main()

