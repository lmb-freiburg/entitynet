from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from packg.iotools import yield_lines_from_file
from packg.iotools.jsonext import dump_json, load_json
from packg.log import logger
from visiontext.distutils import barrier_safe, get_rank

from entitynet.paths import get_entitynet_data_dir

CUB_SUBDIR = "cub200"
CUB_ANN_SUBDIR = "CUB_200_2011"


def get_cub_dirs(root=None):
    if root is None:
        root = get_entitynet_data_dir() / CUB_SUBDIR
    root = Path(root)
    ann_root = root / CUB_ANN_SUBDIR
    return root, ann_root


class Cub(Dataset):
    """
    The Caltech-UCSD Birds-200-2011 Dataset

    Note that we start counting everything from 0 so it will not match the original IDs.

    splits: train 5994 test 5794 full 11788
    """

    def __init__(
        self,
        split: str,
        name: str = "default",
        transform=None,
        max_datapoints=None,
        root=None,
    ):
        root, ann_root = get_cub_dirs(root)
        cached_ann_file = root / f"cached_annotations.json"
        if not cached_ann_file.is_file() and get_rank() == 0:
            ann_data = load_cub_annotations_from_txt(ann_root)
            dump_json(ann_data, cached_ann_file, indent=2)
        barrier_safe()
        ann_data = load_json(cached_ann_file)
        # categories: name_cleaned, name
        # attributes: name (e.g. "has_back_pattern::spotted")
        # parts: name (e.g. "back")
        # images: file_name, class_idx, width, height, bbox, parts, att_vec

        if split == "train" or split == "test":
            split2imageids = {"train": [], "test": []}
            for i, line in enumerate(yield_lines_from_file(ann_root / "train_test_split.txt")):
                idx, is_train = line.split(" ")
                is_train = bool(int(is_train))
                assert int(idx) == i + 1, f"{idx} != {i + 1} in {line}"
                split_here = "train" if is_train else "test"
                split2imageids[split_here].append(i)
            imageids = split2imageids[split]
        elif split == "full":
            imageids = list(range(len(ann_data["images"])))
        else:
            raise ValueError(f"Unknown split {split}")

        if name == "default":
            self.image_root: Path = ann_root
        else:
            raise ValueError(f"Unknown name {name}")

        if max_datapoints is not None and max_datapoints > 0:
            imageids = imageids[:max_datapoints]
            logger.warning(f"Reducing dataset {self} to {max_datapoints} datapoints")

        classes = [cat["name_cleaned"] for cat in ann_data["categories"]]
        images = ann_data["images"]

        self.imageids: list[int] = imageids
        self.classes: list[str] = classes
        self.images: list[dict[str, Any]] = images
        self.transform = transform

    def get_keys(self):
        return self.imageids

    def get_obj_label_from_key(self, key):
        return self.images[key]["class_idx"]

    def get_image_file_from_key(self, key):
        image_info = self.images[key]
        img_path = self.image_root / image_info["file_name"]
        return img_path

    def __getitem__(self, index):
        image_idx = self.imageids[index]
        image_info = self.images[image_idx]
        img_path = self.image_root / image_info["file_name"]
        # path is like images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
        # root is /path/to/datasets/cub200/CUB_200_2011
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        obj_label = image_info["class_idx"]  # int in [0, 199]

        return {
            "image": img,
            "idx": image_idx,  # index,
            "label": obj_label,  # int64 (batch_size, ) with object label in [0, .., num_classes-1]
        }

    def get_image_file(self, index):
        image_idx = self.imageids[index]
        image_info = self.images[image_idx]
        img_path = self.image_root / image_info["file_name"]
        return img_path

    def __len__(self):
        return len(self.imageids)


def load_cub_annotations_from_txt(ann_root) -> dict:
    """
    ann_root: .../datasets/cub200/CUB_200_2011
    """
    # load 200 bird class names
    categories, categories_raw = [], []
    classes_file = ann_root / "classes.txt"
    for i, line in enumerate(yield_lines_from_file(classes_file)):
        # 1 001.Black_footed_Albatross
        idx, name_raw = line.split(" ", 1)
        idx = int(idx)
        idx_again = int(name_raw.split(".")[0])
        assert idx == idx_again, f"{idx} != {idx_again} in {line}"
        assert idx == i + 1, f"{idx} != {i + 1} in {line}"
        name_clean = process_raw_name(name_raw)
        categories.append(name_clean)
        categories_raw.append(name_raw)
    unique_names = sorted(set(categories))
    assert len(unique_names) == len(categories), "Duplicate category names"
    cat2i = {cat: i for i, cat in enumerate(categories)}

    # load images
    images = []
    images_file = ann_root / "images.txt"
    for i, line in enumerate(yield_lines_from_file(images_file)):
        # 1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
        idx, path = line.split(" ", 1)
        idx = int(idx)
        assert idx == i + 1, f"{idx} != {i + 1} in {line}"
        cat_name_raw = path.split("/")[0]
        cat_name = process_raw_name(cat_name_raw)
        class_idx = cat2i[cat_name]
        # path: 200.Common_Yellowthroat/Common_Yellowthroat_0080_190663.jpg
        image_file = (Path("images") / path).as_posix()
        full_image_dir = ann_root / image_file
        img = Image.open(full_image_dir).convert("RGB")
        width, height = img.size
        images.append(
            {
                "file_name": image_file,
                "class_idx": class_idx,
                "width": width,
                "height": height,
            }
        )

    # assert image class labels are same
    image_class_labels_file = ann_root / "image_class_labels.txt"
    for i, line in enumerate(yield_lines_from_file(image_class_labels_file)):
        # 1 1
        idx, class_idx = line.split(" ")
        idx = int(idx)
        class_idx = int(class_idx)
        assert idx == i + 1, f"{idx} != {i + 1} in {line}"
        image_dict = images[i]
        image_class_idx = image_dict["class_idx"]
        assert class_idx == image_class_idx + 1, f"{class_idx} != {images[i]['class_idx']}"

    logger.info(f"CUB: {len(categories)} cats, {len(images)} images")

    # convert into coco format
    final_data = {
        "info": {
            "date_created": "2024-09",
            "description": "Processed CUB-200-2011 (Caltech-UCSD Birds-200-2011) Dataset",
            "url": "https://data.caltech.edu/records/65de6-vp158",
            "version": "1.0",
            "contributor": "Dataset by Wah et al., 2022, modified by Ging et al., 2024",
            "license": {
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "id": 0,
                "name": "Commons Attribution 4.0 International",
            },
        },
        "categories": [
            {"name_cleaned": cat, "name": categories_raw[i]} for i, cat in enumerate(categories)
        ],
        "images": images,
    }
    return final_data


def process_raw_name(name_raw):
    # 001.Black_footed_Albatross
    name_raw_split = name_raw.split(".")
    name_clean = ".".join(name_raw_split[1:]).replace("_", " ")
    return name_clean
