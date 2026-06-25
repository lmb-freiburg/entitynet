"""
Utilities to log images, text, and other artifacts to different loggers in lightning.
"""

import os
from pathlib import Path

import numpy as np
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from loguru import logger
from PIL import Image

from packg.typext import PathType
from visiontext.denormalize import Denormalize, find_mean_and_std_for_denormalization
from visiontext.images import visualize_image_text_pairs_from_tensor


def log_image_tensor(
    metric_logger,
    key,  # training/image
    image_tensor,
    descriptions=None,
    transform=None,
    n_images_max=128,
    log_data_locally: bool = False,
    local_dir: PathType = None,
):
    # try to denormalize the image back to 0-1
    normed = False
    if transform is not None:
        mean, std = find_mean_and_std_for_denormalization(transform)
        if mean is not None and std is not None:
            denorm = Denormalize(mean, std)
            image_tensor = denorm(image_tensor)
            normed = True

    if not normed:
        # jointly normalize all channels to [0, 1] for each image separately
        min_ = image_tensor.amin(dim=(1, 2, 3), keepdim=True)
        max_ = image_tensor.amax(dim=(1, 2, 3), keepdim=True)
        image_tensor = (image_tensor - min_) / (max_ - min_)
        # add the normalization info to the descriptions
        for i in range(image_tensor.shape[0]):
            min_here = min_[i, 0, 0, 0].item()
            max_here = max_[i, 0, 0, 0].item()
            descriptions[i] = f"{descriptions[i]} (min={min_here:.3f}, max={max_here:.3f})"

    if isinstance(metric_logger, CSVLogger) or log_data_locally:
        assert (
            local_dir is not None
        ), "local_dir must be provided when using CSVLogger or log_data_locally=True"
        image_log_dir = Path(local_dir) / "image_log"
        os.makedirs(image_log_dir, exist_ok=True)
        pil_images = visualize_image_text_pairs_from_tensor(image_tensor, descriptions)
        for b in range(image_tensor.shape[0]):
            image_path = image_log_dir / f"{b:04d}.png"
            pil_images[b].save(image_path)
        return

    if isinstance(metric_logger, WandbLogger):
        import wandb  # type: ignore

        images = []
        for b in range(image_tensor.shape[0]):
            # image is shape (B, C, H, W)
            # "Note : torch.Tensor images are normalized. PIL Image is not." -> we already
            # denormalized so we can convert to PIL Image
            img_np = image_tensor[b].permute(1, 2, 0).cpu().numpy()
            # Convert to uint8 and create PIL Image
            img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            caption = descriptions[b] if descriptions is not None else f"Image {b}"
            images.append(wandb.Image(img, caption=caption))
        metric_logger.experiment.log({key: images})
        return
    if isinstance(metric_logger, DummyLogger):
        logger.warning(
            f"DummyLogger called to save image '{key}' shape {image_tensor.shape} description "
            f"{descriptions}."
        )
        return

    from lightning.pytorch.loggers import NeptuneLogger

    if isinstance(metric_logger, NeptuneLogger):
        from neptune.attributes import FileSeries  # type: ignore
        from neptune.handler import Handler  # type: ignore
        from neptune.types import File  # type: ignore

        # neptune has limited storage. make sure we only save a few images
        # somewhat hacky way to figure out how many images have been stored already
        handler: Handler = metric_logger.experiment[key]
        series: FileSeries = handler._container.get_attribute(handler._path)
        if series is None:
            item_count = 0
        else:
            item_count = series._backend.get_image_series_values(
                series._container_id, series._container_type, series._path, 0, 1
            ).totalItemCount

        if item_count > n_images_max > 0:
            logger.warning(f"Found already {item_count} logged images, not logging any more.")
        else:
            # image is shape (B, C, H, W)
            # neptune expects (H, W, C)
            for b in range(image_tensor.shape[0]):
                description = ""
                if descriptions is not None:
                    description = descriptions[b]
                metric_logger.experiment[key].append(
                    File.as_image(image_tensor[b].permute(1, 2, 0).cpu().numpy()),
                    description=description,
                )
        return
    raise NotImplementedError(f"Image saving not implemented for logger {metric_logger}")
