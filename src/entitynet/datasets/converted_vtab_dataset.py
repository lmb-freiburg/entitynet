from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from packg.typext import PathType

from clip_benchmark.datasets.en_zeroshot_classification_templates import EN_ZSCLS_TEMPLATES
from entitynet.datasets.cub import Cub


def build_converted_vtab_dataset(
    root: PathType,
    dataset_name: str,
    dataset_split: str,
    label_name: str,
    task: str,  # Task argument can be used to further customize if needed.
    transform: Optional[Callable] = None,
):
    """
    Constructs and returns a ConvertedVTABDataset.

    Args:
        root (PathType): The base directory under which the converted VTAB datasets reside.
        dataset_name (str): A short name for the dataset (should match the name used during conversion).
        dataset_split (str): The dataset split (e.g., "train", "val", "test").
        label_name (str): The key to use for the label in the output dict.
        task (str): The task identifier (can be used to differentiate behaviors if necessary).
        transform (Optional[Callable]): An optional transform to be applied on the PIL image.

    Returns:
        ConvertedVTABDataset: An instance of the dataset.
    """
    assert dataset_name.startswith("converted_vtab/")
    dataset_name = dataset_name[len("converted_vtab/") :]

    overwrite_classes = None
    dataset_dir = Path(root).parent / f"converted_vtab/{dataset_name}_{dataset_split}"
    return ConvertedVTABDataset(dataset_dir, label_name, dataset_name, dataset_split, transform)


class ConvertedVTABDataset(Dataset):
    def __init__(
        self,
        root: PathType,
        label_key: str,
        dataset_name: str,
        dataset_split: str,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: Directory that contains the converted VTAB dataset files:
                             "images", "labels.json", "info.json", and "classes.json".
            label_key The key to use in the returned dict for the label.
            transform: A callable that transforms the PIL image.
        """
        logger.debug(f"Create {type(self).__name__} in {root}")
        self.root = Path(root)
        self.images_dir = self.root / "images"

        # Load labels and meta data
        with open(self.root / "labels.json", "r") as f:
            self.labels = json.load(f)
        with open(self.root / "info.json", "r") as f:
            self.info = json.load(f)

        if dataset_name == "cub":
            dummy = Cub("test")
            self.classes = deepcopy(dummy.classes)
            del dummy
        else:
            with open(self.root / "classes.json", "r") as f:
                self.classes = json.load(f)
                self.classes = [c.strip().replace("_", " ") for c in self.classes]

        self.label_key = label_key
        self.transform = transform
        # The conversion function stored the input_mode in meta (e.g., "pil" or "tensor")
        self.input_mode = self.info.get("input_mode", "pil")
        self.num_examples = len(self.labels)
        self.templates = EN_ZSCLS_TEMPLATES[dataset_name]
        self.image_paths = [self.images_dir / f"{idx:08d}.png" for idx in range(self.num_examples)]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # Load image using a padded filename (e.g., 00000001.png)
        image_path = self.image_paths[idx]
        # self.images_dir / f"{idx:08d}.png"
        # torch needs an indexerror to figure out the dataset, it will not catch a filenotfound.
        image = Image.open(image_path).convert("RGB")

        # Apply a transform if provided.
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform is provided and the input_mode expects a tensor,
            # convert the image accordingly.
            if self.input_mode == "tensor":
                image = transforms.ToTensor()(image)
            # Otherwise, keep it as a PIL Image.

        # Get the corresponding label.
        label = self.labels[idx]
        # Return a dict with the image and the label keyed by label_key.
        return {"image": image, self.label_key: label, "idx": idx}
