from attr import asdict
from loguru import logger
from torchvision.transforms import (
    Compose,
    ElasticTransform,
    GaussianBlur,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    ToTensor,
)

from packg.iotools.jsonext import dump_json

import open_clip
from entitynet.config.model_config import ModelFactoryC, PreprocCfg, PreprocessorFactoryC
from entitynet.models.clip_misc_utils import process_clip_model_name
from entitynet.preprocessor.open_clip_prep import (
    get_preprocess_config_file,
    load_open_clip_preprocessor,
)
from open_clip.model import CLIP, get_model_preprocess_cfg
from open_clip.transform import color_jitter as color_jitter_fn
from open_clip.transform import gray_scale, image_transform


def get_simple_transform_to_tensor(
    image_size: int | tuple[int], mean=None, std=None, resize_mode="shortest"
):
    """
    Transform and crop to tensor. To disable normalization, set mean=0 and std=1.
    """
    return image_transform(image_size, is_train=False, mean=mean, std=std, resize_mode=resize_mode)


def build_vis_preprocessor_from_config(preproc_cfg: PreprocCfg):
    preproc_factory = preproc_cfg.preproc_factory

    if preproc_factory == PreprocessorFactoryC.OPEN_CLIP:
        # model_name, pretrained = preproc_cfg.preproc_ident.split("/")
        model_name, pretrained = process_clip_model_name(preproc_cfg.preproc_ident)
        overwrites = preproc_cfg.clip_pp_cfg
        overwrites = {} if overwrites is None else asdict(overwrites)
        vis_prep = load_open_clip_preprocessor(model_name, pretrained, is_train=False, **overwrites)
        aug_cfg = preproc_cfg.aug_cfg
        if aug_cfg is not None:
            # augmentations requested, find parameters from the loaded preprocessor
            size, mean, std = extract_information_from_compose(vis_prep)
            logger.info(f"Building train augmentations: {size=} {mean=} {std=} {aug_cfg=}")
            vis_prep = build_train_aug_cfg(size, mean, std, **aug_cfg)
        return vis_prep
    raise ValueError(f"Unknown preproc_factory: {preproc_factory}")


def extract_information_from_compose(compose):
    # transform is usually Resize, CenterCrop, _convert_to_rgb, ToTensor, normalize
    if not hasattr(compose, "transforms"):
        raise ValueError(
            f"Tried to find mean, std and image_size from the transform, but did not understand "
            f"the object that was passed: {type(compose)} - {compose}"
        )
    try:
        size = compose.transforms[0].size
    except AttributeError as e:
        raise ValueError(f"Could not find size in the transforms: {compose.transforms}") from e

    mean, std = None, None
    for tf in compose.transforms:
        if hasattr(tf, "mean") and hasattr(tf, "std"):
            mean, std = tf.mean, tf.std
    if mean is None or std is None:
        raise ValueError(f"Could not find mean and std in the transforms: {compose.transforms}")
    return size, mean, std


def _convert_to_rgb(image):
    return image.convert("RGB")


def build_train_aug_cfg(
    image_size,
    mean,
    std,
    scale: tuple[float, float] | None,
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    color_jitter: tuple[float, float, float, float] | None = None,
    color_jitter_prob: float | None = None,
    gray_scale_prob: float | None = None,
    elastic_alpha: float | None = None,
    elastic_sigma: float | None = None,
    rotation_degrees: float | tuple[float, float] | None = None,
    h_flip_probability: float | None = None,
    gaussian_blur_kernel_size: int | tuple[float, float] | None = None,
    gaussian_blur_sigma: float | tuple[float, float] | None = None,
):
    normalize = Normalize(mean=mean, std=std)
    train_transform = [
        RandomResizedCrop(
            image_size,
            scale=scale,
            ratio=ratio,
            interpolation=InterpolationMode.BICUBIC,
        ),
        _convert_to_rgb,
    ]
    if color_jitter_prob is not None:
        assert color_jitter is not None and len(color_jitter) == 4, f"Wrong values: {color_jitter}"
        train_transform.extend([color_jitter_fn(*color_jitter, p=color_jitter_prob)])
    if gray_scale_prob is not None and gray_scale_prob > 0.0:
        train_transform.extend([gray_scale(gray_scale_prob)])
    if elastic_alpha is not None and elastic_sigma is not None:
        train_transform.append(ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma))
    if rotation_degrees is not None:
        train_transform.append(RandomRotation(degrees=rotation_degrees))
    if h_flip_probability is not None:
        train_transform.append(RandomHorizontalFlip(p=h_flip_probability))
    if gaussian_blur_kernel_size is not None and gaussian_blur_sigma is not None:
        train_transform.append(
            GaussianBlur(kernel_size=gaussian_blur_kernel_size, sigma=gaussian_blur_sigma)
        )
    train_transform.extend(
        [
            ToTensor(),
            normalize,
        ]
    )
    train_transform = Compose(train_transform)
    return train_transform


def create_preprocessing_config_file_from_model(
    model_factory: str, model_name: str, pretrained: str
):
    filename = get_preprocess_config_file(model_name, pretrained)
    if filename.is_file():
        return

    logger.warning(f"Creating preprocessor config file: {filename}")
    if model_factory == ModelFactoryC.OPEN_CLIP:
        model: CLIP = open_clip.create_model(model_name, pretrained)
    else:
        raise ValueError(
            f"Unknown {model_factory=}, options: {ModelFactoryC.values_list()}."
            f"Failed to create preprocessor config file in {filename}, create it "
            f"manually or update this code to support this new model factory."
        )

    # assuming the model stores its preprocess config
    preprocess_cfg = get_model_preprocess_cfg(model)
    if len(preprocess_cfg) == 0:
        raise ValueError(f"Not able to extract preprocessing config for {model_name} {pretrained}.")
    dump_json(preprocess_cfg, filename, create_parent=True, indent=2, verbose=False)
    del model, preprocess_cfg
